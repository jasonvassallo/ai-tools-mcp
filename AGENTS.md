# ai-tools-mcp

This repository contains a small MCP server for hosted AI APIs. It is not a local-model repo.

## Purpose

- Expose `deep_research` for Perplexity Sonar Pro deep research with multi-source synthesis and citations.
- Expose `gemini_deep_research_start` / `_result` for Google Gemini Deep Research — long-running, citation-dense reports drawn from many sources.
- Complement Claude's built-in WebSearch (quick lookups) with two thorough-research tiers (fast inline via Perplexity, multi-minute report via Gemini).

## Stable Public Surface

- MCP server key: `ai-tools-mcp`
- Tool names:
  - `deep_research`
  - `gemini_deep_research_start`
  - `gemini_deep_research_result`

Keep those tool names stable unless a change is explicitly requested.

## Provider Mapping

- `deep_research`:
  - Provider: Perplexity
  - Model: `sonar-pro`
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
