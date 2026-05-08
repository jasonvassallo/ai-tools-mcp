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

import re
import subprocess
import sys
import asyncio
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
        # (per PR #1 review, Gemini). When two distinct secret-shape keys
        # would collide on the same redacted marker, append a numeric
        # suffix (#2, #3, ...) to preserve all entries — naive
        # dict-comprehension would silently lose data (per PR #2 review,
        # Codex P2 + Gemini medium).
        # Collision-handling is gated on string keys only (per PR #2 follow-up
        # review, Gemini medium): non-string keys (tuple, int, frozenset, ...)
        # pass through redact_secrets unchanged and therefore cannot collide
        # with redacted-string markers, so preserve their original type
        # without the f-string conversion path.
        out: dict[Any, Any] = {}
        for k, v in value.items():
            new_k = redact_secrets(k)
            new_v = redact_secrets(v)
            if isinstance(new_k, str) and new_k in out:
                i = 2
                while f"{new_k}#{i}" in out:
                    i += 1
                new_k = f"{new_k}#{i}"
            out[new_k] = new_v
        return out
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
        )
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
