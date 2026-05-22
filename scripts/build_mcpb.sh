#!/usr/bin/env bash
# Build the ai-tools-mcp Desktop Extension (.mcpb) archive.
#
# Output: dist/ai-tools-mcp.mcpb (a zip with manifest.json at the root +
# server/mcp_server.py). Drag the resulting file into Claude Desktop →
# Settings → Extensions to install.
#
# Uses Anthropic's official @anthropic-ai/mcpb CLI via npx. Requires
# Node/npm on PATH (Jason has both; Desktop's "Detected tools" panel
# confirms Node.js is available).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${REPO_ROOT}/mcpb"
DIST_DIR="${REPO_ROOT}/dist"
ARCHIVE="${DIST_DIR}/ai-tools-mcp.mcpb"

if [[ ! -f "${BUILD_DIR}/manifest.json" ]]; then
  echo "fatal: ${BUILD_DIR}/manifest.json not found" >&2
  exit 1
fi

# Refresh the server payload from the source mcp_server.py. We do this on
# every build so the bundled copy can never drift from the canonical
# source. The bundled file is intentionally a copy (not a symlink) because
# zip resolves symlinks differently across platforms and we want a single
# self-contained server file at the same relative path the manifest
# declares.
echo "→ refreshing server payload"
mkdir -p "${BUILD_DIR}/server"
cp "${REPO_ROOT}/mcp_server.py" "${BUILD_DIR}/server/mcp_server.py"

mkdir -p "${DIST_DIR}"
rm -f "${ARCHIVE}"

# `mcpb pack` validates the manifest against the official schema before
# building, then packages everything in BUILD_DIR (except patterns in
# .mcpbignore, if present) into the .mcpb archive at ARCHIVE.
echo "→ packing via @anthropic-ai/mcpb"
npx --yes @anthropic-ai/mcpb pack "${BUILD_DIR}" "${ARCHIVE}"

echo
echo "✓ built ${ARCHIVE}"
echo
echo "Install: drag ${ARCHIVE} into Claude Desktop → Settings → Extensions."
