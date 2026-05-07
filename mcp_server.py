#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = ["openai>=1.0.0", "mcp>=1.0.0"]
# ///
"""
MCP server providing deep research via Perplexity Sonar Pro.

Designed to complement Claude's built-in WebSearch tool:
- Built-in WebSearch: quick factual lookups, single-answer questions
- deep_research: multi-source synthesis, comparisons, ambiguous queries
"""

import asyncio
import json
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import Tool, TextContent, ListToolsResult
from openai import OpenAI


# ─── Configuration ────────────────────────────────────────────────────

SESSIONS_DIR = Path.home() / ".claude" / "sessions"
SESSIONS_DIR.mkdir(exist_ok=True)

perplexity_client = OpenAI(
    api_key=subprocess.check_output(
        ["security", "find-generic-password", "-s", "api_tokens", "-a", "perplexity", "-w"],
        text=True,
    ).strip(),
    base_url="https://api.perplexity.ai",
)

server = Server("ai-tools-mcp")


# ─── Session Management Helpers ───────────────────────────────────────


def get_session_file(session_id: str) -> Path:
    """Get the file path for a session."""
    return SESSIONS_DIR / f"{session_id}.json"


def list_sessions() -> list[dict[str, Any]]:
    """List all sessions with their metadata."""
    sessions = []
    if not SESSIONS_DIR.exists():
        return sessions

    for session_file in SESSIONS_DIR.glob("*.json"):
        try:
            with open(session_file, "r") as f:
                data = json.load(f)
            sessions.append({
                "session_id": session_file.stem,
                "name": data.get("name", "Untitled"),
                "created_at": data.get("created_at"),
                "last_modified": data.get("last_modified"),
                "message_count": len(data.get("messages", [])),
            })
        except (json.JSONDecodeError, IOError):
            continue

    # Sort by last_modified descending
    sessions.sort(key=lambda x: x.get("last_modified", ""), reverse=True)
    return sessions


def save_session(
    name: str = "Untitled",
    messages: list[dict[str, Any]] | None = None,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Save a conversation session."""
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
    """Load a saved session."""
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
    """Update session metadata."""
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

    return {"success": True, "session_id": session_id, "name": data["name"]}


def delete_session(session_id: str) -> dict[str, Any]:
    """Delete a session."""
    session_file = get_session_file(session_id)
    if not session_file.exists():
        raise ValueError(f"Session not found: {session_id}")

    session_file.unlink()
    return {"success": True, "session_id": session_id}


# ─── Perplexity Tool ───────────────────────────────────────────────────


@server.list_tools()
async def list_tools() -> ListToolsResult:
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
                "List all saved conversation sessions. Returns session metadata "
                "including session ID, name, creation date, last modified date, "
                "and message count. Use this to find sessions you can resume."
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
                "Save the current conversation context to a session file. "
                "Call this at the end of a session to preserve the conversation. "
                "The model should be instructed to call this before ending sessions. "
                "Pass the full conversation history as 'messages' array."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "A descriptive name for the session (e.g., 'Project Setup Discussion')",
                    },
                    "messages": {
                        "type": "array",
                        "description": "Array of message objects from the conversation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "metadata": {
                        "type": "object",
                        "description": "Optional metadata for the session",
                        "properties": {
                            "project": {"type": "string"},
                            "tags": {"type": "array", "items": {"type": "string"}},
                        },
                    },
                },
                "required": ["name", "messages"],
            },
        ),
        Tool(
            name="resume_session",
            description=(
                "Load a previous conversation session by its session ID. "
                "Returns the full conversation history including all messages. "
                "Use this to continue working on a previous topic."
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
            name="end_session",
            description=(
                "End the current session and save the conversation context. "
                "This is a convenience tool that combines saving with session "
                "cleanup. Call this when you're done with a conversation. "
                "It will automatically save all messages and mark the session as complete."
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
                        "description": "All messages from the conversation",
                        "items": {
                            "type": "object",
                            "properties": {
                                "role": {"type": "string", "enum": ["user", "assistant", "system"]},
                                "content": {"type": "string"},
                            },
                        },
                    },
                    "summary": {
                        "type": "string",
                        "description": "Optional summary of what was accomplished",
                    },
                },
                "required": ["name", "messages"],
            },
        ),
        Tool(
            name="delete_session",
            description=(
                "Delete a saved session permanently. Use with caution. "
                "Requires the session ID of the session to delete."
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
        result = f"## Research Results\n\n{message.content}"

        return [TextContent(type="text", text=result)]

    elif name == "list_sessions":
        sessions = list_sessions()
        result = "## Saved Sessions\n\n"
        if not sessions:
            result += "No saved sessions found.\n"
        else:
            result += "| Session ID | Name | Messages | Last Modified |\n"
            result += "|------------|------|----------|---------------|\n"
            for s in sessions:
                result += f"| `{s['session_id']}` | {s['name']} | {s['message_count']} | {s['last_modified']} |\n"
        return [TextContent(type="text", text=result)]

    elif name == "save_session":
        name_arg = arguments.get("name", "Untitled")
        messages = arguments.get("messages", [])
        metadata = arguments.get("metadata", {})

        result = save_session(name=name_arg, messages=messages, metadata=metadata)
        return [TextContent(type="text", text=f"Session saved: {result['session_id']}")]

    elif name == "resume_session":
        session_id = arguments.get("session_id")
        try:
            session = load_session(session_id)
            result = f"## Session: {session['name']}\n\n"
            result += f"**Created:** {session['created_at']}\n"
            result += f"**Last Modified:** {session['last_modified']}\n\n"
            result += "### Conversation History\n\n"
            for msg in session["messages"]:
                result += f"**{msg['role'].upper()}:** {msg['content']}\n\n"
            return [TextContent(type="text", text=result)]
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    elif name == "end_session":
        name_arg = arguments.get("name", "Untitled")
        messages = arguments.get("messages", [])
        summary = arguments.get("summary", "")

        result = save_session(name=name_arg, messages=messages, metadata={"summary": summary})
        return [TextContent(type="text", text=f"Session ended and saved: {result['session_id']}")]

    elif name == "delete_session":
        session_id = arguments.get("session_id")
        try:
            result = delete_session(session_id)
            return [TextContent(type="text", text=f"Session deleted: {session_id}")]
        except ValueError as e:
            return [TextContent(type="text", text=f"Error: {str(e)}")]

    else:
        raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
