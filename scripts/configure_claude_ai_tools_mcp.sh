#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
CLAUDE_CONFIG="$HOME/Library/Application Support/Claude/claude_desktop_config.json"
LAUNCHER_SCRIPT="$PROJECT_DIR/scripts/launch_ai_tools_mcp.sh"
SYSTEM_PATH="$HOME/.volta/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin"
NODE_BIN="$HOME/.volta/bin/node"
NPM_BIN="$HOME/.volta/bin/npm"
NPX_BIN="$HOME/.volta/bin/npx"

mkdir -p "$(dirname "$CLAUDE_CONFIG")"

if [ ! -f "$CLAUDE_CONFIG" ]; then
  printf '{\n  "mcpServers": {}\n}\n' > "$CLAUDE_CONFIG"
fi

TMP_FILE="$(mktemp)"
jq \
  --arg command "$LAUNCHER_SCRIPT" \
  --arg path "$SYSTEM_PATH" \
  --arg volta_home "$HOME/.volta" \
  --arg node_bin "$NODE_BIN" \
  --arg npm_bin "$NPM_BIN" \
  --arg npx_bin "$NPX_BIN" \
  '
  .mcpServers = (.mcpServers // {}) |
  del(.mcpServers["ai-tools"]) |
  .mcpServers["ai-tools-mcp"] = {
    command: $command,
    env: {
      PATH: $path,
      VOLTA_HOME: $volta_home,
      NODE: $node_bin,
      NPM: $npm_bin,
      NPX: $npx_bin
    }
  }
  ' \
  "$CLAUDE_CONFIG" > "$TMP_FILE"

mv "$TMP_FILE" "$CLAUDE_CONFIG"

# Ensure GUI-launched apps (including Claude) inherit a PATH with Volta shims.
launchctl setenv PATH "$SYSTEM_PATH" || true
launchctl setenv VOLTA_HOME "$HOME/.volta" || true
launchctl setenv NODE "$NODE_BIN" || true

echo "Updated: $CLAUDE_CONFIG"
