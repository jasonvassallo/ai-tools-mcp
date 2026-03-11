#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
REQUIREMENTS="$PROJECT_DIR/requirements.txt"
SERVER_SCRIPT="$PROJECT_DIR/mcp_server.py"

resolve_uv() {
  if command -v uv >/dev/null 2>&1; then
    command -v uv
    return 0
  fi

  local candidates=(
    "/opt/homebrew/bin/uv"
    "/usr/local/bin/uv"
    "$HOME/.local/bin/uv"
  )
  local candidate
  for candidate in "${candidates[@]}"; do
    if [ -x "$candidate" ]; then
      echo "$candidate"
      return 0
    fi
  done

  return 1
}

UV_BIN="$(resolve_uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "ERROR: uv is not installed or not executable." >&2
  exit 1
fi

if [ ! -x "$VENV_PYTHON" ]; then
  "$UV_BIN" venv "$PROJECT_DIR/.venv"
fi

if ! "$VENV_PYTHON" -c "import mcp, openai" >/dev/null 2>&1; then
  "$UV_BIN" pip install --python "$VENV_PYTHON" -r "$REQUIREMENTS"
fi

exec "$VENV_PYTHON" "$SERVER_SCRIPT"
