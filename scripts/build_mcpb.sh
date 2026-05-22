#!/usr/bin/env bash
# Build the ai-tools-mcp Desktop Extension (.mcpb) archive.
#
# Output: dist/ai-tools-mcp.mcpb (a zip with manifest.json at the root +
# server/mcp_server.py). Drag the resulting file into Claude Desktop →
# Settings → Extensions to install.

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

# Sanity-check the manifest parses as JSON before we package it.
echo "→ validating manifest.json"
python3 -c "import json; json.load(open('${BUILD_DIR}/manifest.json'))"

mkdir -p "${DIST_DIR}"
rm -f "${ARCHIVE}"

# Zip from inside the build dir so manifest.json lands at the archive root
# (Desktop reads manifest.json at the archive's top level, not nested).
echo "→ packaging ${ARCHIVE}"
(cd "${BUILD_DIR}" && zip -qr "${ARCHIVE}" manifest.json server)

echo
echo "✓ built $(du -h "${ARCHIVE}" | cut -f1) → ${ARCHIVE}"
echo
echo "Install: drag ${ARCHIVE} into Claude Desktop → Settings → Extensions."
