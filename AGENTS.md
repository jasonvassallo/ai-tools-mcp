# ai-tools-mcp

This repository contains a small MCP server for hosted AI APIs. It is not a local-model repo.

## Purpose

- Expose `deep_research` for Perplexity Sonar Pro deep research with multi-source synthesis and citations.
- Expose `agent_research` for Perplexity Agent API Search-as-Code — a hosted sandbox agent that searches programmatically for bulk/enumerable research tasks.
- Expose `gemini_deep_research_start` / `_result` for Google Gemini Deep Research — long-running, citation-dense reports drawn from many sources.
- Complement Claude's built-in WebSearch (quick lookups) with thorough-research tiers (fast inline via Perplexity, programmatic bulk via the Agent API sandbox, multi-minute report via Gemini).

## Stable Public Surface

- MCP server key: `ai-tools-mcp`
- Tool names (research):
  - `quick_research`
  - `deep_research`
  - `agent_research`
  - `agent_research_result`
  - `gemini_deep_research_start`
  - `gemini_deep_research_result`
- Tool names (session management):
  - `list_sessions`
  - `save_session`
  - `load_session`
  - `update_session`
  - `delete_session`

Keep those tool names stable unless a change is explicitly requested.

## Packaging Formats

The same `mcp_server.py` is wrapped three ways. When making changes, update all three:

- Standalone: `install.sh` registers it directly in `~/.claude/.mcp.json`
- Claude Code plugin: `.claude-plugin/plugin.json` + `.mcp.json` + `commands/` + `skills/` + `hooks/`
- Claude Desktop extension: `mcpb/manifest.json` (built into `dist/ai-tools-mcp.mcpb` by `scripts/build_mcpb.sh`)

## Provider Mapping

- `quick_research`:
  - Provider: Perplexity
  - Model: `sonar`
- `deep_research`:
  - Provider: Perplexity
  - Model: `sonar-pro`
- `agent_research` / `agent_research_result`:
  - Provider: Perplexity Agent API (`/v1/responses`) with the `sandbox` tool
  - Models: `anthropic/claude-sonnet-4-6` (default) or `perplexity/sonar` — server-side allowlist (the Agent API does not offer `sonar-pro`)
  - Synchronous by default (runs take one to several minutes), or `background=true` returns a `response_id` to poll via `agent_research_result`; billed per-model tokens + per-container fee + per-search charges.
- `gemini_deep_research_start` / `gemini_deep_research_result`:
  - Provider: Google Gemini Deep Research (`/v1beta/interactions`)
  - Models: `deep-research-preview-04-2026` (fast), `deep-research-max-preview-04-2026` (max)
  - Asynchronous; start returns an `interaction_id`, result polls until `status="completed"`.

## Dependencies

`mcp_server.py` uses PEP 723 inline script metadata — no virtualenv or project config needed. `uv run mcp_server.py` handles everything.

Credentials:
- macOS Keychain: service `api_tokens`, account `perplexity` (required)
- Google Cloud ADC at `~/.config/gcloud/application_default_credentials.json` (required for Gemini Deep Research). Configure with `gcloud auth application-default login`. Billing project is auto-detected.

## Running

```bash
uv run mcp_server.py
```

## Claude Code Registration

Add to `~/.claude/.mcp.json`:

```json
{
  "ai-tools-mcp": {
    "command": "uv",
    "args": ["run", "/path/to/mcp_server.py"]
  }
}
```

## Guardrails

- Keep this repo focused on hosted API-backed MCP tooling.
- Do not add local-model features here.
- Do not add email-triage code here.
- Local-only AI projects belong in `local_llm_integration`, not this repo.
- Never publish secrets, API keys, tokens, `.env` files, certificates, or private keys to GitHub from this repo.
