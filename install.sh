#!/usr/bin/env bash
# ╔══════════════════════════════════════════════════════════════════╗
# ║  MCP Server Installer for macOS                                 ║
# ║  Template — customize the CONFIG section for your own project   ║
# ╚══════════════════════════════════════════════════════════════════╝
#
# What this does:
#   1. Checks prerequisites (uv, jq)
#   2. Installs the MCP server script to ~/.local/share/<app-name>/
#   3. Stores required API tokens in macOS Keychain
#   4. Registers the MCP server in Claude Code (~/.claude/.mcp.json)
#   5. Verifies everything works
#
# Safe to run multiple times — updates existing config without clobbering.

set -euo pipefail

# ─── CONFIG (customize this section for each project) ────────────

APP_NAME="ai-tools-mcp"
APP_VERSION="0.2.0"
INSTALL_DIR="$HOME/.local/share/${APP_NAME}"
SCRIPT_NAME="mcp_server.py"

# MCP server key in .mcp.json
MCP_SERVER_KEY="ai-tools-mcp"

# Keychain service name (shared across all your MCP tools)
KEYCHAIN_SERVICE="api_tokens"

# API tokens required: "account_name|display_name|description"
REQUIRED_TOKENS=(
    "perplexity|Perplexity API Key|Get one at https://www.perplexity.ai/settings/api"
)

# ─── END CONFIG ──────────────────────────────────────────────────

CLAUDE_MCP_CONFIG="$HOME/.claude/.mcp.json"
BOLD='\033[1m'
DIM='\033[2m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
RED='\033[0;31m'
CYAN='\033[0;36m'
NC='\033[0m'

step=0
total_steps=5

print_step() {
    step=$((step + 1))
    echo ""
    echo -e "${CYAN}${BOLD}[$step/$total_steps]${NC} ${BOLD}$1${NC}"
}

print_ok() {
    echo -e "  ${GREEN}✓${NC} $1"
}

print_skip() {
    echo -e "  ${DIM}→ $1${NC}"
}

print_warn() {
    echo -e "  ${YELLOW}!${NC} $1"
}

print_fail() {
    echo -e "  ${RED}✗${NC} $1"
}

# ─── Header ──────────────────────────────────────────────────────

echo ""
echo -e "${BOLD}${APP_NAME} v${APP_VERSION} — Installer${NC}"
echo -e "${DIM}──────────────────────────────────────────${NC}"

# ─── Step 1: Prerequisites ───────────────────────────────────────

print_step "Checking prerequisites"

missing=0

if command -v uv >/dev/null 2>&1; then
    print_ok "uv $(uv --version 2>/dev/null | head -1)"
else
    print_fail "uv not found — install from https://docs.astral.sh/uv/"
    missing=1
fi

if command -v jq >/dev/null 2>&1; then
    print_ok "jq $(jq --version 2>/dev/null)"
else
    print_fail "jq not found — install with: brew install jq"
    missing=1
fi

if command -v security >/dev/null 2>&1; then
    print_ok "macOS Keychain (security)"
else
    print_fail "macOS Keychain not available — this installer requires macOS"
    missing=1
fi

if [[ $missing -ne 0 ]]; then
    echo ""
    echo -e "${RED}Missing prerequisites. Install them and re-run.${NC}"
    exit 1
fi

# ─── Step 2: Install script ─────────────────────────────────────

print_step "Installing ${SCRIPT_NAME}"

# Find the script — either in the same dir as this installer or current dir
INSTALLER_DIR="$(cd "$(dirname "$0")" && pwd)"
SOURCE_SCRIPT=""

if [[ -f "${INSTALLER_DIR}/${SCRIPT_NAME}" ]]; then
    SOURCE_SCRIPT="${INSTALLER_DIR}/${SCRIPT_NAME}"
elif [[ -f "./${SCRIPT_NAME}" ]]; then
    SOURCE_SCRIPT="$(pwd)/${SCRIPT_NAME}"
fi

if [[ -z "$SOURCE_SCRIPT" ]]; then
    print_fail "${SCRIPT_NAME} not found next to installer or in current directory"
    exit 1
fi

mkdir -p "$INSTALL_DIR"

if [[ -f "${INSTALL_DIR}/${SCRIPT_NAME}" ]]; then
    if diff -q "$SOURCE_SCRIPT" "${INSTALL_DIR}/${SCRIPT_NAME}" >/dev/null 2>&1; then
        print_skip "Already installed and up to date"
    else
        cp "$SOURCE_SCRIPT" "${INSTALL_DIR}/${SCRIPT_NAME}"
        print_ok "Updated ${INSTALL_DIR}/${SCRIPT_NAME}"
    fi
else
    cp "$SOURCE_SCRIPT" "${INSTALL_DIR}/${SCRIPT_NAME}"
    print_ok "Installed to ${INSTALL_DIR}/${SCRIPT_NAME}"
fi

# ─── Step 3: API tokens ─────────────────────────────────────────

print_step "Setting up API tokens (macOS Keychain)"

for token_spec in "${REQUIRED_TOKENS[@]}"; do
    IFS='|' read -r account display_name description <<< "$token_spec"

    # Check if already stored
    if security find-generic-password -s "$KEYCHAIN_SERVICE" -a "$account" -w >/dev/null 2>&1; then
        print_skip "${display_name} already stored"
        echo ""
        read -rp "  Overwrite existing token? [y/N] " overwrite
        if [[ "$overwrite" != "y" && "$overwrite" != "Y" ]]; then
            continue
        fi
    fi

    echo ""
    echo -e "  ${DIM}${description}${NC}"
    read -rsp "  Paste your ${display_name}: " token_value
    echo ""

    if [[ -z "$token_value" ]]; then
        print_warn "Skipped ${display_name} (empty input)"
        continue
    fi

    # Delete existing entry if present
    security delete-generic-password -s "$KEYCHAIN_SERVICE" -a "$account" 2>/dev/null || true

    # Store the new token
    security add-generic-password -s "$KEYCHAIN_SERVICE" -a "$account" -l "$display_name" -w "$token_value"
    print_ok "Stored ${display_name} in Keychain"
done

# ─── Step 4: Register with Claude Code ──────────────────────────

print_step "Registering MCP server with Claude Code"

# Find uv's absolute path for the config
UV_PATH="$(command -v uv)"
SCRIPT_PATH="${INSTALL_DIR}/${SCRIPT_NAME}"

# Build the server entry
SERVER_ENTRY=$(jq -n \
    --arg cmd "$UV_PATH" \
    --arg script "$SCRIPT_PATH" \
    '{command: $cmd, args: ["run", $script]}')

# Create or update .mcp.json
mkdir -p "$(dirname "$CLAUDE_MCP_CONFIG")"

if [[ -f "$CLAUDE_MCP_CONFIG" ]]; then
    # Merge into existing config
    UPDATED=$(jq --arg key "$MCP_SERVER_KEY" --argjson entry "$SERVER_ENTRY" \
        '.mcpServers[$key] = $entry' "$CLAUDE_MCP_CONFIG")
    echo "$UPDATED" > "$CLAUDE_MCP_CONFIG"
    print_ok "Updated ${CLAUDE_MCP_CONFIG}"
else
    # Create new config
    jq -n --arg key "$MCP_SERVER_KEY" --argjson entry "$SERVER_ENTRY" \
        '{mcpServers: {($key): $entry}}' > "$CLAUDE_MCP_CONFIG"
    print_ok "Created ${CLAUDE_MCP_CONFIG}"
fi

# Show the registered entry
echo -e "  ${DIM}$(jq --arg key "$MCP_SERVER_KEY" '.mcpServers[$key]' "$CLAUDE_MCP_CONFIG")${NC}"

# ─── Step 5: Verify ─────────────────────────────────────────────

print_step "Verifying installation"

errors=0

# Check script exists
if [[ -f "$SCRIPT_PATH" ]]; then
    print_ok "Script installed at ${SCRIPT_PATH}"
else
    print_fail "Script not found at ${SCRIPT_PATH}"
    errors=1
fi

# Check tokens
for token_spec in "${REQUIRED_TOKENS[@]}"; do
    IFS='|' read -r account display_name _ <<< "$token_spec"
    if security find-generic-password -s "$KEYCHAIN_SERVICE" -a "$account" -w >/dev/null 2>&1; then
        print_ok "${display_name} found in Keychain"
    else
        print_warn "${display_name} not in Keychain — server will fail to start"
        errors=1
    fi
done

# Check Claude config
if [[ -f "$CLAUDE_MCP_CONFIG" ]] && jq -e --arg key "$MCP_SERVER_KEY" '.mcpServers[$key]' "$CLAUDE_MCP_CONFIG" >/dev/null 2>&1; then
    print_ok "Registered in Claude Code config"
else
    print_fail "Not found in ${CLAUDE_MCP_CONFIG}"
    errors=1
fi

# Check uv can resolve deps
echo -e "  ${DIM}Testing uv run (dependency resolution)...${NC}"
if "$UV_PATH" run --quiet "$SCRIPT_PATH" --help >/dev/null 2>&1; then
    print_ok "Dependencies resolve correctly"
else
    # The script doesn't have --help, so it will try to start the server.
    # A quick timeout test: start it, wait briefly, kill it.
    "$UV_PATH" run "$SCRIPT_PATH" &>/dev/null &
    TEST_PID=$!
    sleep 2
    if kill -0 "$TEST_PID" 2>/dev/null; then
        kill "$TEST_PID" 2>/dev/null
        wait "$TEST_PID" 2>/dev/null || true
        print_ok "Server starts successfully"
    else
        wait "$TEST_PID" 2>/dev/null || true
        print_warn "Server exited quickly — check API token is set"
    fi
fi

# ─── Done ────────────────────────────────────────────────────────

echo ""
if [[ $errors -eq 0 ]]; then
    echo -e "${GREEN}${BOLD}Installation complete.${NC}"
else
    echo -e "${YELLOW}${BOLD}Installation complete with warnings.${NC}"
fi
echo ""
echo -e "  ${BOLD}Next:${NC} Restart Claude Code to load the new MCP server."
echo ""
