#!/usr/bin/env bash
#
# Run Qodo PR-Agent on a GitHub PR using a LOCAL Ollama model, with the GitHub
# token taken from the environment ($GH_TOKEN) — never written to pr-agent's
# plaintext .secrets.toml. Part of the local pre-PR review gate (see CLAUDE.md).
#
#   scripts/pr_review_local.sh <pr_url> [review|describe|improve|ask "..."]
#
# qwen3.6 is a *thinking* model that runs away on pr-agent's single structured
# review call (timed out at 120s and 600s). pr-agent strips Ollama's `think:false`
# from its config, so this script stands up a tiny local proxy that injects
# `think:false` into every Ollama request (torn down on exit). Thinking-off makes
# the same model review in seconds with clean, non-`<think>`-polluted output.
#
# Overridable via env: PR_AGENT_OLLAMA_MODEL, PR_AGENT_MAX_TOKENS,
# PR_AGENT_AI_TIMEOUT, OLLAMA_API_BASE, PR_AGENT_PROXY_PORT.
# CONFIG__PUBLISH_OUTPUT=false dry-runs it (no posting to the PR).
#
set -euo pipefail

PR_URL="${1:?usage: pr_review_local.sh <pr_url> [command...]}"
shift || true
CMD=("${@:-review}")

: "${GH_TOKEN:?GH_TOKEN not set — load your GitHub PAT from Keychain into the env}"

UPSTREAM="${OLLAMA_API_BASE:-http://localhost:11434}"
PROXY_PORT="${PR_AGENT_PROXY_PORT:-11435}"
MODEL="${PR_AGENT_OLLAMA_MODEL:-ollama/qwen3.6:35b-a3b-coding-nvfp4}"

# --- stand up the think:false-injecting Ollama proxy (stdlib only) -----------
PROXY_PY="$(mktemp -t ollama_think_off.XXXXXX).py"
cat >"$PROXY_PY" <<'PY'
import json
import sys
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

UPSTREAM = sys.argv[1].rstrip("/")
PORT = int(sys.argv[2])


class Handler(BaseHTTPRequestHandler):
    def _forward(self, method):
        body = None
        n = int(self.headers.get("Content-Length", 0) or 0)
        if n:
            body = self.rfile.read(n)
            try:
                data = json.loads(body)
                if isinstance(data, dict):
                    data["think"] = False  # force qwen3 thinking OFF
                    body = json.dumps(data).encode()
            except (json.JSONDecodeError, ValueError):
                pass
        req = urllib.request.Request(
            UPSTREAM + self.path, data=body, method=method,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req) as resp:
            self.send_response(resp.status)
            self.send_header("Content-Type", resp.headers.get("Content-Type", "application/json"))
            self.end_headers()
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(chunk)

    def do_POST(self):
        self._forward("POST")

    def do_GET(self):
        self._forward("GET")

    def log_message(self, *args):
        pass


ThreadingHTTPServer(("127.0.0.1", PORT), Handler).serve_forever()
PY

python3 "$PROXY_PY" "$UPSTREAM" "$PROXY_PORT" &
PROXY_PID=$!
cleanup() { kill "$PROXY_PID" 2>/dev/null || true; rm -f "$PROXY_PY"; }
trap cleanup EXIT

# wait for the proxy to bind + verify it forwards to Ollama
for _ in 1 2 3 4 5; do
  curl -s --max-time 3 "http://127.0.0.1:${PROXY_PORT}/api/tags" >/dev/null 2>&1 && break
  sleep 1
done
curl -s --max-time 3 "http://127.0.0.1:${PROXY_PORT}/api/tags" >/dev/null \
  || { echo "think-off proxy failed to start on :${PROXY_PORT}" >&2; exit 1; }

# --- run pr-agent against the proxy (token from env, never on disk) ----------
GITHUB__USER_TOKEN="$GH_TOKEN" \
CONFIG__GIT_PROVIDER=github \
CONFIG__MODEL="$MODEL" \
CONFIG__FALLBACK_MODELS='[]' \
CONFIG__CUSTOM_MODEL_MAX_TOKENS="${PR_AGENT_MAX_TOKENS:-14000}" \
CONFIG__AI_TIMEOUT="${PR_AGENT_AI_TIMEOUT:-300}" \
OLLAMA__API_BASE="http://127.0.0.1:${PROXY_PORT}" \
pr-agent --pr_url "$PR_URL" "${CMD[@]}"
