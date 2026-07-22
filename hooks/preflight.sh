#!/usr/bin/env bash
# ai-tools-mcp SessionStart preflight.
#
# Runs `uv run mcp_server.py --check` to verify:
#   - the Perplexity API key is present in the macOS Keychain
#   - Google Cloud Application Default Credentials are loaded and refreshable
#
# Output is emitted as a SessionStart additionalContext block (JSON on stdout)
# so Claude sees the result and can warn the user up-front if either credential
# is missing, rather than failing on the first tool call.
#
# Exit status is intentionally 0 in every code path: SessionStart hooks have
# their stdout JSON consumed as additionalContext only when exit is 0 (per the
# Claude Code hooks contract). Non-zero exits log the failure but discard the
# stdout payload, which would defeat the whole point of this hook — we WANT
# the error message to reach Claude even when the underlying check fails.

set -uo pipefail

SERVER="${CLAUDE_PLUGIN_ROOT}/mcp_server.py"

# Emit a SessionStart additionalContext envelope built via python's json module
# (concatenation rather than f-string — the captured tool output can contain
# any characters and shouldn't go through any interpolation surface).
emit() {
  local status="$1"
  local detail="$2"
  python3 -c "
import json, sys
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'SessionStart',
        'additionalContext': sys.argv[1] + '\n\n' + sys.argv[2]
    }
}))
" "${status}" "${detail}"
}

if [[ ! -f "${SERVER}" ]]; then
  emit "ai-tools-mcp preflight: ERROR" "mcp_server.py not found at ${SERVER}. Plugin is mis-installed."
  exit 0
fi

if ! command -v uv >/dev/null 2>&1; then
  emit "ai-tools-mcp preflight: ERROR" "uv binary not found on PATH. Install with 'brew install uv' or set the uv_path in the Desktop extension settings."
  exit 0
fi

if ! command -v python3 >/dev/null 2>&1; then
  # python3 ships with macOS so this branch is paranoid, but if we ever
  # land in an environment without it, the silent failure mode below
  # (emit() crashes) would leave the hook output empty.
  printf '{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"ai-tools-mcp preflight: ERROR — python3 not available."}}\n'
  exit 0
fi

# Capture both stdout and exit status. The --check helper prints one line
# per credential (ok:/fail:) and exits 0 if both pass, non-zero on failure.
# UV_PRERELEASE is pinned so the check verifies the same resolution the
# launch configs use — the hook runs from the user's project cwd, where
# the repo uv.toml is not discovered and ambient env would apply.
if check_output=$(UV_PRERELEASE="if-necessary-or-explicit" uv run "${SERVER}" --check 2>&1); then
  emit "ai-tools-mcp preflight: OK" "${check_output}"
else
  emit "ai-tools-mcp preflight: WARNING — some credentials are not healthy" "${check_output}"
fi
