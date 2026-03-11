# ai-tools-mcp

This repository contains a small MCP server for hosted AI APIs. It is not a local-model repo.

## Purpose

- Expose `kimi_think` for Moonshot Kimi K2 extended reasoning.
- Expose `web_search` for Perplexity Sonar Pro web search with citations.
- Provide helper scripts to register the server with Claude Desktop.

## Stable Public Surface

- MCP server key: `ai-tools-mcp`
- Tool names:
  - `kimi_think`
  - `web_search`

Keep those tool names stable unless a change is explicitly requested.

## Provider Mapping

- `kimi_think`:
  - Provider: Moonshot
  - Model: `kimi-k2-thinking`
- `web_search`:
  - Provider: Perplexity
  - Model: `sonar-pro`

## Dependencies

- Python virtual environment with:
  - `openai`
  - `mcp`
- macOS Keychain items:
  - service `moonshot-api`, account `kimi`
  - service `perplexity-api`, account `sonar`

## Local Run

From the repo root:

```bash
./scripts/launch_ai_tools_mcp.sh
```

Or directly:

```bash
.venv/bin/python mcp_server.py
```

## Claude Desktop Registration

Use:

```bash
./scripts/configure_claude_ai_tools_mcp.sh
```

This registers the MCP server in Claude Desktop under `ai-tools-mcp` and removes the old `ai-tools` entry if present.

## Guardrails

- Keep this repo focused on hosted API-backed MCP tooling.
- Do not add local-model features here.
- Do not add email-triage code here.
- Local-only AI projects belong in `local_llm_integration`, not this repo.
- Never publish secrets, API keys, tokens, `.env` files, certificates, or private keys to GitHub from this repo.
