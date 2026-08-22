#!/bin/sh
# Build the coding_agent sandbox image (spec §8.2). Everything the untrusted
# local model might need at runtime must already be in the image -- the
# container it backs runs with --network=none, so nothing can be installed
# after this build step.
set -eu
cd "$(dirname "$0")"
docker build -t ai-tools-coding-agent:latest .
echo "built ai-tools-coding-agent:latest"
