#!/usr/bin/env bash
# ai-tools-mcp SessionStart preflight.
#
# Runs `uv run mcp_server.py --check` to verify:
#   - the Perplexity API key is present in the macOS Keychain
#   - Google Cloud Application Default Credentials are loaded and refreshable
#
# Output is emitted as a SessionStart additionalContext block so Claude sees
# the result and can warn the user up-front if either credential is missing,
# rather than failing on the first tool call.

set -euo pipefail

SERVER="${CLAUDE_PLUGIN_ROOT}/mcp_server.py"

if [[ ! -f "${SERVER}" ]]; then
  # Plugin file layout is broken — emit a clear error to Claude.
  cat <<EOF
{"hookSpecificOutput":{"hookEventName":"SessionStart","additionalContext":"ai-tools-mcp preflight: ERROR — mcp_server.py not found at ${SERVER}. Plugin is mis-installed."}}
EOF
  exit 0
fi

# Capture both stdout and exit status. The --check helper prints one line
# per credential (ok:/fail:) and exits 0 if both pass, non-zero on failure.
if check_output=$(uv run "${SERVER}" --check 2>&1); then
  status_line="ai-tools-mcp preflight: OK"
else
  status_line="ai-tools-mcp preflight: WARNING — some credentials are not healthy"
fi

# Format for SessionStart additionalContext. We escape the captured output
# for JSON via python so embedded newlines/quotes don't break the envelope.
python3 -c "
import json, sys
status = sys.argv[1]
detail = sys.argv[2]
print(json.dumps({
    'hookSpecificOutput': {
        'hookEventName': 'SessionStart',
        'additionalContext': f'{status}\n\n{detail}'
    }
}))
" "${status_line}" "${check_output}"
