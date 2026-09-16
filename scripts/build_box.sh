#!/usr/bin/env bash
set -euo pipefail
IMAGE="${1:-agent-forum-box:latest}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
docker build -t "$IMAGE" "$ROOT"
echo "built $IMAGE"
