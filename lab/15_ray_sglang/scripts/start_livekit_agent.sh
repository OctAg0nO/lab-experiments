#!/usr/bin/env bash
# Start the LiveKit agent worker with OctAg0nO brain.
#
# Prerequisites:
#   - LiveKit server running (docker run --rm -p 7880:7880 livekit/livekit-server)
#   - SGLang server running (bash scripts/launch_sglang.sh)
#   - LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET set in .env
#
# Usage:
#   bash scripts/start_livekit_agent.sh [workers]
#
# Defaults:
#   WORKERS=1

set -euo pipefail

WORKERS="${1:-1}"
SGLANG_ENDPOINT="${SGLANG_ENDPOINT:-http://localhost:30000/v1}"

echo "=== Starting LiveKit Agent Worker ==="
echo "  Workers:    $WORKERS"
echo "  SGLang:     $SGLANG_ENDPOINT"
echo ""

if [ "$WORKERS" -gt 1 ]; then
    echo "Starting $WORKERS workers via Ray..."
    uv run python -m lab.15_ray_sglang \
        --sglang-endpoint "$SGLANG_ENDPOINT" \
        --ray \
        livekit-worker --workers "$WORKERS"
else
    echo "Starting single worker..."
    uv run python -m lab.15_ray_sglang \
        --sglang-endpoint "$SGLANG_ENDPOINT" \
        livekit-worker
fi
