#!/bin/sh
# The §8.2 drift report: which expected tools the sandbox image is missing,
# and the installed version of the ones it has. This is presence/version
# VISIBILITY, not a correctness guarantee -- it does not know what version
# the host project actually needs, only what's on PATH in the image. Its job
# is to turn silent drift into something an operator reads before a
# sandboxed run fails confusingly, not to eliminate the drift itself.
#
# Usage: check.sh [image-tag]   (defaults to ai-tools-coding-agent:latest)
# Exit status: 0 if every expected tool is present, 1 if any are missing.
set -u
IMG="${1:-ai-tools-coding-agent:latest}"
missing=0

for t in git python3 uv pytest ruff mypy node npm shellcheck; do
    result=$(docker run --rm --network=none "$IMG" sh -c "
        if command -v '$t' >/dev/null 2>&1; then
            '$t' --version 2>&1 | head -n1
        else
            echo '__MISSING__'
        fi
    " 2>/dev/null)
    if [ "$result" = "__MISSING__" ] || [ -z "$result" ]; then
        printf '  MISSING %s\n' "$t"
        missing=$((missing + 1))
    else
        printf '  ok      %-10s %s\n' "$t" "$result"
    fi
done

echo
if [ "$missing" -gt 0 ]; then
    echo "$missing expected tool(s) missing from $IMG"
    exit 1
fi
echo "all expected tools present in $IMG"
