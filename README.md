# ai-tools-mcp

`ai-tools-mcp` is a small MCP server that exposes hosted AI providers behind a stable local MCP surface.

This repository is intentionally narrow in scope:

- It is for hosted API-backed MCP tooling.
- It is not a local-model repo.
- It currently exposes three tools:
  - `deep_research`
  - `gemini_deep_research_start`
  - `gemini_deep_research_result`

## Stable Public Surface

The following identifiers are meant to stay stable unless intentionally changed:

- MCP server key: `ai-tools-mcp`
- Tool names:
  - `deep_research`
  - `gemini_deep_research_start`
  - `gemini_deep_research_result`

## Provider Mapping

### `deep_research`

- Provider: Perplexity
- Model: `sonar-pro`
- Purpose: deep research with multi-source synthesis, cross-referencing, and citations
- Latency: seconds (synchronous)
- Use when: the answer should come back inline in the current session

### `gemini_deep_research_start` / `gemini_deep_research_result`

- Provider: Google Gemini Deep Research (`/v1beta/interactions`)
- Models: `deep-research-preview-04-2026` (fast) and `deep-research-max-preview-04-2026` (max)
- Purpose: long-running, citation-dense reports drawing on many sources
- Latency: minutes (up to 60); asynchronous, polled via the `_result` tool
- Use when: you need a standalone, multi-page report — not a quick answer

Together these complement Claude's built-in `WebSearch`: use `WebSearch` for
quick lookups, `deep_research` for thorough inline investigation, and the
`gemini_deep_research_*` pair when the deliverable IS the report.

## How It Works

The server is a single Python script (`mcp_server.py`) with PEP 723 inline dependency metadata. `uv run` resolves and caches dependencies automatically — no virtualenv, no `requirements.txt`, no build step.

It:

- starts an MCP server named `ai-tools-mcp`
- reads API credentials from the macOS Keychain
- calls the Perplexity API through the `openai` Python client
- returns plain text MCP responses

There are no local model weights, no background service, and no embedded secrets in the repo.

## Repository Layout

- `mcp_server.py`: Self-contained MCP server with inline dependency metadata
- `install.sh`: macOS installer (Keychain setup, Claude Code registration, preflight check)
- `scripts/launch_ai_tools_mcp.sh`: Legacy launcher (virtualenv-based, for Claude Desktop)
- `scripts/configure_claude_ai_tools_mcp.sh`: Claude Desktop registration helper
- `scripts/uv_sync_projects.sh`: Separate local helper for syncing Python projects with `uv`

## Installation

The recommended way to install is via the included installer:

```bash
./install.sh
```

This will:

1. Check prerequisites (`uv`, `jq`, macOS Keychain)
2. Install `mcp_server.py` to `~/.local/share/ai-tools-mcp/`
3. Prompt for API tokens and store them in the macOS Keychain
4. Register the MCP server in `~/.claude/.mcp.json`
5. Run a preflight check to verify dependencies and configuration

Safe to run multiple times — updates existing config without clobbering.

## Requirements

### System

- macOS
- `uv` installed and on `PATH`
- `jq` installed (for installer only)

### Keychain Entries

The server expects one API key in the macOS Keychain (for Perplexity):

- service `api_tokens`, account `perplexity`

The installer handles this automatically. For manual setup:

```bash
security add-generic-password -s 'api_tokens' -a 'perplexity' -w 'YOUR_PERPLEXITY_API_KEY'
```

### Google Cloud Application Default Credentials (ADC)

The Gemini Deep Research tools authenticate via **ADC**, not a static API
key. Set this up once with:

```bash
gcloud auth application-default login
gcloud auth application-default set-quota-project YOUR_GCP_PROJECT
```

The server reads ADC from the standard location
(`~/.config/gcloud/application_default_credentials.json`) and refreshes
short-lived bearer tokens transparently via the `google-auth` library. The
billing project is auto-detected from ADC.

The preflight check (`uv run mcp_server.py --check`) verifies both the
Perplexity key and ADC, refreshing a token to confirm credentials are live.

## Running

The recommended way to run is via `uv run`, which reads the PEP 723 inline metadata and manages dependencies automatically:

```bash
uv run mcp_server.py
```

No virtualenv creation or dependency installation needed — `uv` handles it from its global cache.

### Preflight Check

Validate that dependencies resolve and the Keychain entry exists without starting the server:

```bash
uv run mcp_server.py --check
```

## Claude Code Registration

The installer handles this. For manual setup, add to `~/.claude/.mcp.json`:

```json
{
  "mcpServers": {
    "ai-tools-mcp": {
      "command": "uv",
      "args": ["run", "/path/to/ai-tools-mcp/mcp_server.py"]
    }
  }
}
```

## Claude Desktop Registration

For Claude Desktop (uses the legacy virtualenv launcher):

```bash
./scripts/configure_claude_ai_tools_mcp.sh
```

## Tool Inputs

### `deep_research`

Input schema:

- `query`: required string
- `max_tokens`: optional integer, default `2048`

Output behavior:

- returns a formatted text block with research results
- relies on Perplexity response content to include citations
- response is routed through a redactor that masks secret-shape strings
  (Google API keys, OAuth tokens, JWTs, private-key blocks)

### `gemini_deep_research_start`

Input schema:

- `query`: required string
- `mode`: optional `"fast" | "max"` (default `"fast"`)
- `collaborative_planning`: optional boolean (default `false`)
- `thinking_summaries`: optional `"auto" | "none"` (default `"auto"`)

Output behavior:

- returns JSON `{interaction_id, status, model, hint}`
- the task runs in the background on Google's side; poll with the result tool

### `gemini_deep_research_result`

Input schema:

- `interaction_id`: required string (must match `^[A-Za-z0-9_-]{1,128}$`;
  this is enforced at the tool boundary to prevent the authenticated request
  from being redirected to an attacker-controlled host)

Output behavior:

- `{status: "completed", output_text, steps_count, steps_summary}` when done
- `{status: "failed", error}` on failure
- `{status: "in_progress", hint}` while still running — poll again in ~30s
- `output_text` and `error` are routed through the same secret-redactor as
  `deep_research`

## Development Notes

- This repo is intentionally small and single-purpose.
- Keep the public MCP surface stable.
- Avoid adding unrelated automation or local-model functionality here.
- If functionality drifts beyond hosted MCP tooling, it should likely live in a different repo.

## Secret Hygiene

This repo is intended to be safe to publish publicly. The current design keeps secrets out of source control by using the macOS Keychain.

Rules for working in this repo:

- never commit raw API keys, session tokens, or bearer tokens
- never commit `.env` files
- never commit certificate or private key files
- keep local debug dumps out of Git
- prefer Keychain lookups over environment-file storage for this project

## License

No license file has been added yet. If you plan to accept contributions or want reuse clarity, add one explicitly.
