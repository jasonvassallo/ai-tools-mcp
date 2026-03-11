# ai-tools-mcp

`ai-tools-mcp` is a small MCP server that exposes hosted AI providers behind a stable local MCP surface.

This repository is intentionally narrow in scope:

- It is for hosted API-backed MCP tooling.
- It is not a local-model repo.
- It currently exposes exactly two tools:
  - `kimi_think`
  - `web_search`

## Stable Public Surface

The following identifiers are meant to stay stable unless intentionally changed:

- MCP server key: `ai-tools-mcp`
- Tool name: `kimi_think`
- Tool name: `web_search`

## Provider Mapping

### `kimi_think`

- Provider: Moonshot
- Model: `kimi-k2-thinking`
- Purpose: deeper reasoning, multi-step analysis, planning, and problem solving

### `web_search`

- Provider: Perplexity
- Model: `sonar-pro`
- Purpose: web-backed search, current information, and citation-oriented responses

## How It Works

The server is a stdio MCP process implemented in Python. It:

- starts an MCP server named `ai-tools-mcp`
- reads API credentials from the macOS Keychain
- calls the provider APIs through the `openai` Python client
- returns plain text MCP responses

There are no local model weights, no background service, and no embedded secrets in the repo.

## Repository Layout

- `mcp_server.py`: MCP server implementation
- `scripts/launch_ai_tools_mcp.sh`: local launcher that ensures a usable virtualenv
- `scripts/configure_claude_ai_tools_mcp.sh`: Claude Desktop registration helper
- `scripts/uv_sync_projects.sh`: separate local helper for syncing Python projects with `uv`
- `test_moonshot.py`: direct Moonshot connectivity smoke test
- `AGENTS.md`: repo-specific working instructions

## Requirements

### System

- macOS
- Python available locally
- `uv` installed and on `PATH`
- `jq` installed for the Claude Desktop registration script

### Python Dependencies

- `openai`
- `mcp`

They are listed in `requirements.txt` and installed automatically by the launcher if needed.

### Keychain Entries

The server expects API keys in the macOS Keychain under these exact service/account pairs:

- service `moonshot-api`, account `kimi`
- service `perplexity-api`, account `sonar`

Example setup:

```bash
security add-generic-password -s 'moonshot-api' -a 'kimi' -w 'YOUR_API_KEY'
security add-generic-password -s 'perplexity-api' -a 'sonar' -w 'YOUR_API_KEY'
```

## Local Run

From the repo root:

```bash
./scripts/launch_ai_tools_mcp.sh
```

The launcher will:

- locate `uv`
- create `.venv` if it does not exist
- install dependencies if `mcp` or `openai` are missing
- exec the MCP server over stdio

You can also run the server directly if the environment is already prepared:

```bash
.venv/bin/python mcp_server.py
```

## Claude Desktop Registration

To register this MCP server with Claude Desktop:

```bash
./scripts/configure_claude_ai_tools_mcp.sh
```

The script updates:

- `~/Library/Application Support/Claude/claude_desktop_config.json`

It also:

- registers the server under the `ai-tools-mcp` key
- removes any stale `ai-tools` entry if present
- sets GUI environment variables so Claude can find Volta-managed Node tools when needed

## Example Claude Desktop Entry

Depending on how you register it, your Claude config may end up using either:

- the launcher script
- or a direct Python command and script path

Both point at the same repo and server implementation. The helper script prefers the launcher because it is more resilient to missing virtualenv state.

## Tool Inputs

### `kimi_think`

Input schema:

- `prompt`: required string
- `max_tokens`: optional integer, default `4096`

Output behavior:

- returns a formatted text block
- includes reasoning content if the provider response exposes it
- includes the final answer

### `web_search`

Input schema:

- `query`: required string
- `max_tokens`: optional integer, default `1024`

Output behavior:

- returns a formatted text block
- relies on Perplexity response content to include citations

## Smoke Testing

To test Moonshot directly outside MCP:

```bash
.venv/bin/python test_moonshot.py
```

That script checks:

- Keychain lookup works
- the Moonshot endpoint responds
- the `kimi-k2-thinking` model returns content

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

The `.gitignore` excludes common local and secret-bearing file patterns, but that is only a safeguard. It is not a substitute for checking what is staged before pushing.

## Publishing Checklist

Before making new changes public:

```bash
git status --short
git diff --cached
git log --stat --max-count=1
```

If you want a stronger manual scan:

```bash
rg -n "(ghp_|github_pat_|sk-|BEGIN [A-Z ]*PRIVATE KEY|Bearer )" .
```

## License

No license file has been added yet. If you plan to accept contributions or want reuse clarity, add one explicitly.
