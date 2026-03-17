# ai-tools-mcp

This repository contains a small MCP server for hosted AI APIs. It is not a local-model repo.

## Purpose

- Expose `deep_research` for Perplexity Sonar Pro deep research with multi-source synthesis and citations.
- Complement Claude's built-in WebSearch (quick lookups) with thorough research capability.

## Stable Public Surface

- MCP server key: `ai-tools-mcp`
- Tool names:
  - `deep_research`

Keep those tool names stable unless a change is explicitly requested.

## Provider Mapping

- `deep_research`:
  - Provider: Perplexity
  - Model: `sonar-pro`

## Dependencies

`mcp_server.py` uses PEP 723 inline script metadata — no virtualenv or project config needed. `uv run mcp_server.py` handles everything.

- macOS Keychain items:
  - service `api_tokens`, account `perplexity`

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
