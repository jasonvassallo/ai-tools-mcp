# ai-tools-mcp

Small MCP server for hosted AI APIs.

## Tools

- `kimi_think`: Moonshot `kimi-k2-thinking`
- `web_search`: Perplexity `sonar-pro`

## Requirements

- Python virtual environment with `openai` and `mcp`
- macOS Keychain items:
  - `moonshot-api` / `kimi`
  - `perplexity-api` / `sonar`

## Run

```bash
./scripts/launch_ai_tools_mcp.sh
```

Or:

```bash
.venv/bin/python mcp_server.py
```

## Claude Desktop

```bash
./scripts/configure_claude_ai_tools_mcp.sh
```
