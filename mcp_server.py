#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.0.0", "mcp>=1.0.0"]
# ///
"""
MCP server providing deep research via Perplexity Sonar Pro plus
local session-management tools backed by ``~/.claude/sessions/``.

Designed to complement Claude's built-in WebSearch tool:
- Built-in WebSearch: quick factual lookups, single-answer questions
- deep_research: multi-source synthesis, comparisons, ambiguous queries
- session_*: persist/restore conversation context across runs
"""

import asyncio
import json
import re
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent
from openai import OpenAI


# Patterns for secret-shape strings that may appear in scraped web content
# returned by upstream search providers. Applied at the response boundary so
# secrets do not get persisted in client transcripts.
#
# Order matters: the JWT pattern would otherwise eat substrings of nothing
# else here, but private-key blocks are matched first because they may
# contain other matchable substrings inside the body.
_REDACTION_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
            re.DOTALL,
        ),
        "[REDACTED_PRIVATE_KEY_BLOCK]",
    ),
    (re.compile(r"ya29\.[A-Za-z0-9_-]+"), "[REDACTED_GOOGLE_OAUTH_ACCESS]"),
    (re.compile(r"1//0[A-Za-z0-9_-]{30,}"), "[REDACTED_GOOGLE_OAUTH_REFRESH]"),
    (re.compile(r"AIza[A-Za-z0-9_-]{20,}"), "[REDACTED_GOOGLE_API_KEY]"),
    # JWT minimums relaxed from {30,30,20} to {10,10,10} per PR #1 review
    # (Gemini): minimal valid header `{"alg":"HS256"}` encodes to 20 chars,
    # which the original {30,} requirement missed.
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}"),
        "[REDACTED_JWT]",
    ),
    # Apple app-specific password format (xxxx-xxxx-xxxx-xxxx) was originally
    # included but removed per PR #1 review (Codex P2): the regex
    # \b[a-z]{4}-[a-z]{4}-[a-z]{4}-[a-z]{4}\b false-positives on ordinary
    # research-prose phrases like "real-time-data-flow" or
    # "zero-shot-text-only", silently mangling deep_research output. ASPs
    # leak via local-config / IMAP-debug paths this MCP doesn't touch, so
    # for prose-content redaction the safer trade is to drop the pattern.
)


def redact_secrets(value: Any) -> Any:
    """Recursively mask secret-shape substrings in arbitrary nested data.

    Walks strings, lists, tuples, and dicts; leaves other types untouched.
    Pure-stdlib (uses ``re``); no new dependencies.
    """
    if isinstance(value, str):
        for pattern, replacement in _REDACTION_PATTERNS:
            value = pattern.sub(replacement, value)
        return value
    if isinstance(value, dict):
        # Walk both keys and values so secrets-as-dict-key are also masked
        # (per PR #1 review, Gemini).
        return {redact_secrets(k): redact_secrets(v) for k, v in value.items()}
    if isinstance(value, list):
        return [redact_secrets(v) for v in value]
    if isinstance(value, tuple):
        return tuple(redact_secrets(v) for v in value)
    return value


def get_api_key_from_keychain(service: str, account: str) -> str:
    """Retrieve API key from macOS Keychain."""
    result = subprocess.run(
        ["security", "find-generic-password", "-s", service, "-a", account, "-w"],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise ValueError(
            f"Keychain item not found. Add with:\n"
            f"  security add-generic-password -s '{service}' -a '{account}' -w 'YOUR_API_KEY'"
        )
    return result.stdout.strip()


def run_check() -> None:
    """Validate configuration and exit. Used by install.sh to verify setup."""
    errors = 0
    try:
        get_api_key_from_keychain("api_tokens", "perplexity")
        print("ok: perplexity key found in keychain")
    except ValueError as e:
        print(f"fail: {e}")
        errors += 1
    sys.exit(errors)


# ─── Session management storage ───────────────────────────────────────
#
# Sessions are persisted as ``~/.claude/sessions/<uuid>.json`` with shape:
#     {
#       "session_id": str,
#       "name": str,
#       "created_at": ISO-8601 UTC,
#       "last_modified": ISO-8601 UTC,
#       "messages": [{"role": ..., "content": ...}, ...],
#       "metadata": {...}
#     }
SESSIONS_DIR = Path.home() / ".claude" / "sessions"
SESSIONS_DIR.mkdir(parents=True, exist_ok=True)


def get_session_file(session_id: str) -> Path:
    """Return the on-disk path for a session id."""
    return SESSIONS_DIR / f"{session_id}.json"


def list_sessions() -> list[dict[str, Any]]:
    """List all sessions, most-recently-modified first."""
    sessions: list[dict[str, Any]] = []
    if not SESSIONS_DIR.exists():
        return sessions

    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            sessions.append(
                {
                    "session_id": session_file.stem,
                    "name": data.get("name", "Untitled"),
                    "created_at": data.get("created_at"),
                    "last_modified": data.get("last_modified"),
                    "message_count": len(data.get("messages", [])),
                }
            )
        except (json.JSONDecodeError, OSError):
            continue

    sessions.sort(key=lambda x: x.get("last_modified") or "", reverse=True)
    return sessions


def save_session(
    name: str = "Untitled",
    messages: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a conversation session to ``SESSIONS_DIR``."""
    session_id = str(uuid.uuid4())
    now = datetime.utcnow().isoformat() + "Z"

    session_data = {
        "session_id": session_id,
        "name": name,
        "created_at": now,
        "last_modified": now,
        "messages": messages or [],
        "metadata": metadata or {},
    }

    session_file = get_session_file(session_id)
    with open(session_file, "w") as f:
        json.dump(session_data, f, indent=2)

    return {
        "success": True,
        "session_id": session_id,
        "name": name,
        "message_count": len(session_data["messages"]),
    }


def load_session(session_id: str) -> dict[str, Any]:
    """Load a previously saved session by id."""
    session_file = get_session_file(session_id)
    if not session_file.exists():
        raise ValueError(f"Session not found: {session_id}")

    with open(session_file, "r") as f:
        data = json.load(f)

    return {
        "session_id": data["session_id"],
        "name": data["name"],
        "created_at": data["created_at"],
        "last_modified": data["last_modified"],
        "messages": data["messages"],
        "metadata": data.get("metadata", {}),
    }


def update_session(session_id: str, name: str | None = None) -> dict[str, Any]:
    """Update mutable session metadata (name) and bump ``last_modified``."""
    session_file = get_session_file(session_id)
    if not session_file.exists():
        raise ValueError(f"Session not found: {session_id}")

    with open(session_file, "r") as f:
        data = json.load(f)

    if name:
        data["name"] = name
    data["last_modified"] = datetime.utcnow().isoformat() + "Z"

    with open(session_file, "w") as f:
        json.dump(data, f, indent=2)

    return {
        "success": True,
        "session_id": session_id,
        "name": data["name"],
        "last_modified": data["last_modified"],
    }


def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session file by id."""
    session_file = get_session_file(session_id)
    if not session_file.exists():
        raise ValueError(f"Session not found: {session_id}")

    session_file.unlink()
    return {"success": True, "session_id": session_id}


if "--check" in sys.argv:
    run_check()

# Initialize Perplexity client
perplexity_client = OpenAI(
    api_key=get_api_key_from_keychain("api_tokens", "perplexity"),
    base_url="https://api.perplexity.ai",
)

# Create MCP server
server = Server("ai-tools-mcp")


@server.list_tools()
async def list_tools() -> list[Tool]:
    """List available tools."""
    return [
        Tool(
            name="deep_research",
            description=(
                "Deep research using Perplexity Sonar Pro with multi-source "
                "synthesis and citations. Use instead of built-in WebSearch when: "
                "the answer spans multiple sources, requires cross-referencing, "
                "involves comparing tradeoffs/architectures/approaches, "
                "the query is ambiguous and benefits from AI-powered search reasoning, "
                "or you need comprehensive coverage with source citations. "
                "Do NOT use for simple factual lookups (use built-in WebSearch for those)."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic requiring deep investigation",
                    },
                    "max_tokens": {
                        "type": "integer",
                        "description": "Maximum tokens for response (default: 2048)",
                        "default": 2048,
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="list_sessions",
            description=(
                "List all saved conversation sessions, most recent first. "
                "Returns session id, name, created_at, last_modified, and "
                "message count for each."
            ),
            inputSchema={
                "type": "object",
                "properties": {},
                "required": [],
            },
        ),
        Tool(
            name="save_session",
            description=(
                "Persist the current conversation context to a new session "
                "file. Returns the new session id. Pass the full conversation "
                "history as the 'messages' array."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A descriptive name for the session",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Array of message objects from the conversation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {
                                    "type": "string",
                                    "enum": ["user", "assistant", "system"],
                                },
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional free-form metadata for the session",
                    },
                },
                "required": ["name", "messages"],
            },
        ),
        Tool(
            name="load_session",
            description=(
                "Load a previously saved session by its id. Returns the "
                "full conversation history and metadata."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to load",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="update_session",
            description=(
                "Update a saved session's name and bump its last_modified "
                "timestamp."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to update",
                    },
                    "name": {
                        "type": "string",
                        "description": "New name for the session",
                    },
                },
                "required": ["session_id"],
            },
        ),
        Tool(
            name="delete_session",
            description=(
                "Delete a saved session permanently. Use with caution."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "session_id": {
                        "type": "string",
                        "description": "The UUID of the session to delete",
                    },
                },
                "required": ["session_id"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    """Handle tool calls."""

    if name == "deep_research":
        query = arguments.get("query")
        max_tokens = arguments.get("max_tokens", 2048)

        response = perplexity_client.chat.completions.create(
            model="sonar-pro",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a thorough research assistant. Provide comprehensive, "
                        "well-sourced answers that synthesize information across multiple "
                        "sources. Include relevant details, comparisons, and caveats. "
                        "Always cite your sources."
                    ),
                },
                {"role": "user", "content": query},
            ],
            max_tokens=max_tokens,
        )

        message = response.choices[0].message
        # Redact secret-shape patterns from scraped web content before the
        # response leaves this server. Perplexity's synthesis can include raw
        # API keys / JWTs / private-key blocks lifted from indexed pages.
        content = redact_secrets(message.content or "")
        result = f"## Research Results\n\n{content}"

        return [TextContent(type="text", text=result)]

    if name == "list_sessions":
        sessions = list_sessions()
        if not sessions:
            return [TextContent(type="text", text="## Saved Sessions\n\nNo saved sessions found.\n")]
        lines = [
            "## Saved Sessions",
            "",
            "| Session ID | Name | Messages | Last Modified |",
            "|------------|------|----------|---------------|",
        ]
        for s in sessions:
            lines.append(
                f"| `{s['session_id']}` | {s['name']} | {s['message_count']} | {s['last_modified']} |"
            )
        return [TextContent(type="text", text="\n".join(lines) + "\n")]

    if name == "save_session":
        session_name = arguments.get("name", "Untitled")
        messages = arguments.get("messages", [])
        metadata = arguments.get("metadata", {})
        result = save_session(name=session_name, messages=messages, metadata=metadata)
        return [
            TextContent(
                type="text",
                text=f"Session saved: {result['session_id']} ({result['message_count']} messages)",
            )
        ]

    if name == "load_session":
        session_id = arguments.get("session_id")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            session = load_session(session_id)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        lines = [
            f"## Session: {session['name']}",
            "",
            f"**Created:** {session['created_at']}",
            f"**Last Modified:** {session['last_modified']}",
            "",
            "### Conversation History",
            "",
        ]
        for msg in session["messages"]:
            lines.append(f"**{msg.get('role', 'unknown').upper()}:** {msg.get('content', '')}")
            lines.append("")
        return [TextContent(type="text", text="\n".join(lines))]

    if name == "update_session":
        session_id = arguments.get("session_id")
        new_name = arguments.get("name")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            result = update_session(session_id, name=new_name)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        return [
            TextContent(
                type="text",
                text=f"Session updated: {result['session_id']} (name={result['name']})",
            )
        ]

    if name == "delete_session":
        session_id = arguments.get("session_id")
        if not session_id:
            return [TextContent(type="text", text="Error: session_id is required")]
        try:
            delete_session(session_id)
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {e}")]
        return [TextContent(type="text", text=f"Session deleted: {session_id}")]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
