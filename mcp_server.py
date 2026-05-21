#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "openai>=1.0.0",
#     "mcp>=1.0.0",
#     "httpx>=0.27",
#     "google-auth>=2.30",
#     "requests>=2.31",
# ]
# ///
"""
MCP server providing hosted deep-research tools.

Designed to complement Claude's built-in WebSearch tool:
- Built-in WebSearch: quick factual lookups, single-answer questions
- deep_research: Perplexity Sonar Pro — fast inline multi-source synthesis
- gemini_deep_research_start / _result: Gemini Deep Research — long-running
  (minutes), citation-dense reports via Google's hosted research agent
"""

import json
import re
import subprocess
import sys
import asyncio
from typing import Any
import httpx
import google.auth
import google.auth.transport.requests
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
    (
        re.compile(r"eyJ[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{30,}\.[A-Za-z0-9_-]{20,}"),
        "[REDACTED_JWT]",
    ),
    (
        re.compile(r"\b[a-z]{4}-[a-z]{4}-[a-z]{4}-[a-z]{4}\b"),
        "[REDACTED_APPLE_APP_PWD]",
    ),
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
        return {k: redact_secrets(v) for k, v in value.items()}
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


_ADC_SCOPES = ("https://www.googleapis.com/auth/cloud-platform",)


def _load_adc() -> tuple[Any, str]:
    """Load Google Cloud Application Default Credentials.

    Returns (credentials, billing_project). Raises a clear error if ADC is not
    configured. The credentials object is refreshable — tokens are minted
    lazily per request via `_get_bearer_token`.
    """
    try:
        creds, project = google.auth.default(scopes=list(_ADC_SCOPES))
    except google.auth.exceptions.DefaultCredentialsError as exc:
        raise ValueError(
            "Google Cloud Application Default Credentials not found. "
            "Run: gcloud auth application-default login"
        ) from exc
    if not project:
        raise ValueError(
            "Could not determine billing project from ADC. Run: "
            "gcloud auth application-default set-quota-project YOUR_PROJECT"
        )
    return creds, project


def run_check() -> None:
    """Validate configuration and exit. Used by install.sh to verify setup."""
    errors = 0
    try:
        get_api_key_from_keychain("api_tokens", "perplexity")
        print("ok: perplexity key found in keychain")
    except ValueError as e:
        print(f"fail: {e}")
        errors += 1

    try:
        creds, project = _load_adc()
        # Force a refresh so a stale/expired ADC fails the check here rather
        # than at first tool call.
        creds.refresh(google.auth.transport.requests.Request())
        print(f"ok: google ADC valid (billing project: {project})")
    except (ValueError, Exception) as e:  # noqa: BLE001 - report any auth issue
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

# Gemini Deep Research configuration. The /interactions endpoint is a separate
# surface from the standard Generative Language API and is not yet covered by
# the google-genai SDK at time of writing — call it directly via httpx.
#
# Authentication uses Google Cloud Application Default Credentials (ADC)
# rather than a static API key. Tokens are short-lived (~1 hour) and refreshed
# transparently by the google-auth library.
GEMINI_API_BASE = "https://generativelanguage.googleapis.com/v1beta"
GEMINI_MODELS = {
    "fast": "deep-research-preview-04-2026",
    "max": "deep-research-max-preview-04-2026",
}
# Strict allowlist: interaction IDs from tool parameters are concatenated into
# the request URL. Reject anything that could perform path traversal or escape
# the API host, since the ADC bearer token is attached to every request.
_INTERACTION_ID_RE = re.compile(r"^[A-Za-z0-9_-]{1,128}$")

_gemini_credentials, _gemini_billing_project = _load_adc()
_gemini_token_lock = asyncio.Lock()


async def _get_bearer_token() -> str:
    """Return a fresh ADC bearer token, refreshing on a worker thread if
    expired. The lock serializes concurrent refreshes from parallel tool calls.
    """
    async with _gemini_token_lock:
        if not _gemini_credentials.valid:
            await asyncio.to_thread(
                _gemini_credentials.refresh,
                google.auth.transport.requests.Request(),
            )
        token = _gemini_credentials.token
    if not token:
        raise RuntimeError("ADC refresh returned empty token")
    return token


async def _gemini_headers() -> dict[str, str]:
    return {
        "Authorization": f"Bearer {await _get_bearer_token()}",
        # Required when using OAuth (not API key) so the request is billed and
        # quota-attributed to the user's project rather than the credential's
        # home project.
        "x-goog-user-project": _gemini_billing_project,
        "Content-Type": "application/json",
    }


def _validate_interaction_id(interaction_id: str) -> str:
    """Reject interaction IDs that could redirect the authenticated request.

    The interaction_id is concatenated into the URL of an authenticated HTTP
    call; an attacker-controlled value containing ``/``, ``..``, or a scheme
    could cause the Gemini API key to be sent to an unintended host.
    """
    if not isinstance(interaction_id, str) or not _INTERACTION_ID_RE.fullmatch(
        interaction_id
    ):
        raise ValueError(
            "interaction_id must match ^[A-Za-z0-9_-]{1,128}$ — refusing to "
            "send authenticated request with untrusted path segment."
        )
    return interaction_id


async def _post_gemini_interaction(payload: dict[str, Any]) -> dict[str, Any]:
    """POST a Deep Research interaction. URL is fully static; no tool input.

    follow_redirects is disabled so the ADC bearer token cannot be forwarded
    to another host via a redirect response.
    """
    headers = await _gemini_headers()
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        response = await client.post(
            f"{GEMINI_API_BASE}/interactions",
            headers=headers,
            json=payload,
        )
        response.raise_for_status()
        return response.json()


async def _get_gemini_interaction(interaction_id: str) -> dict[str, Any]:
    """GET a Deep Research interaction by ID.

    The interaction_id MUST have already passed _validate_interaction_id; this
    helper re-validates as defense in depth so the URL cannot escape the API
    host even if a future caller forgets.
    """
    safe_id = _validate_interaction_id(interaction_id)
    headers = await _gemini_headers()
    async with httpx.AsyncClient(timeout=30.0, follow_redirects=False) as client:
        response = await client.get(
            f"{GEMINI_API_BASE}/interactions/{safe_id}",
            headers=headers,
        )
        response.raise_for_status()
        return response.json()


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
            name="gemini_deep_research_start",
            description=(
                "Start a Gemini Deep Research task (asynchronous). Returns an "
                "interaction_id you must poll with gemini_deep_research_result. "
                "Tasks run for several minutes and up to 60 minutes. Use when "
                "you need a citation-dense, multi-page report drawing on many "
                "sources. For quick inline research that completes in seconds, "
                "use `deep_research` (Perplexity) instead."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The research question or topic.",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["fast", "max"],
                        "default": "fast",
                        "description": (
                            "fast = deep-research-preview (speed/efficiency); "
                            "max = deep-research-max-preview (maximum comprehensiveness)."
                        ),
                    },
                    "collaborative_planning": {
                        "type": "boolean",
                        "default": False,
                        "description": "Enable collaborative planning mode.",
                    },
                    "thinking_summaries": {
                        "type": "string",
                        "enum": ["auto", "none"],
                        "default": "auto",
                        "description": "Whether the agent should emit thinking summaries.",
                    },
                },
                "required": ["query"],
            },
        ),
        Tool(
            name="gemini_deep_research_result",
            description=(
                "Retrieve the status or final result of a Gemini Deep Research "
                "task started with gemini_deep_research_start. Returns "
                "{status, output_text, steps_summary} when status='completed', "
                "{status: 'failed', error} on failure, or "
                "{status: 'in_progress', hint} while running. Poll roughly "
                "every 30 seconds."
            ),
            inputSchema={
                "type": "object",
                "properties": {
                    "interaction_id": {
                        "type": "string",
                        "description": "ID returned by gemini_deep_research_start.",
                    },
                },
                "required": ["interaction_id"],
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

    if name == "gemini_deep_research_start":
        query = arguments["query"]
        mode = arguments.get("mode", "fast")
        if mode not in GEMINI_MODELS:
            raise ValueError(
                f"mode must be one of {sorted(GEMINI_MODELS)}; got {mode!r}"
            )

        payload = {
            "agent": GEMINI_MODELS[mode],
            "input": query,
            "background": True,
            "agent_config": {
                "type": "deep-research",
                "thinking_summaries": arguments.get("thinking_summaries", "auto"),
                "collaborative_planning": bool(
                    arguments.get("collaborative_planning", False)
                ),
            },
        }

        data = await _post_gemini_interaction(payload)

        result = {
            "interaction_id": data.get("id"),
            "status": data.get("status", "in_progress"),
            "model": GEMINI_MODELS[mode],
            "hint": (
                "Poll gemini_deep_research_result with this interaction_id. "
                "Tasks take several minutes; up to 60 minutes max."
            ),
        }
        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    if name == "gemini_deep_research_result":
        safe_id = _validate_interaction_id(arguments["interaction_id"])
        data = await _get_gemini_interaction(safe_id)
        status = data.get("status", "unknown")
        result: dict[str, Any] = {"status": status}

        if status == "completed":
            # Route all model-emitted text through the redactor — Deep Research
            # can lift API keys, JWTs, and private-key blocks from the open web.
            result["output_text"] = redact_secrets(data.get("output_text", ""))
            steps = data.get("steps") or []
            result["steps_count"] = len(steps)
            result["steps_summary"] = [s.get("type") for s in steps]
        elif status == "failed":
            result["error"] = redact_secrets(data.get("error", "unknown error"))
        else:
            result["hint"] = "Still running. Poll again in ~30 seconds."

        return [TextContent(type="text", text=json.dumps(result, indent=2))]

    raise ValueError(f"Unknown tool: {name}")


async def main():
    """Run the MCP server."""
    async with stdio_server() as (read_stream, write_stream):
        await server.run(
            read_stream, write_stream, server.create_initialization_options()
        )


if __name__ == "__main__":
    asyncio.run(main())
