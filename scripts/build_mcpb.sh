#!/usr/bin/env bash
# Build the ai-tools-mcp Desktop Extension (.mcpb) archive.
#
# Output: dist/ai-tools-mcp.mcpb (a zip with manifest.json at the root +
# server/mcp_server.py + the server/coding_agent/ package). Drag the file into Claude Desktop →
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

# The coding_agent package, under the same refresh-every-build rule and for
# the same reason: mcp_server.py imports it lazily inside call_tool, so a
# bundle without it advertises the coding_agent tool and then raises
# ModuleNotFoundError when Desktop calls it. Python puts the server script's
# own directory on sys.path, so server/ is where the import resolves from.
# Replaced wholesale rather than merged, so a module deleted upstream cannot
# survive in the bundle as a stale import target.
rm -rf "${BUILD_DIR}/server/coding_agent"
mkdir -p "${BUILD_DIR}/server/coding_agent"
cp "${REPO_ROOT}"/coding_agent/*.py "${BUILD_DIR}/server/coding_agent/"

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
