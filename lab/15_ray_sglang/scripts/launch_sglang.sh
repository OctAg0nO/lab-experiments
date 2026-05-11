#!/usr/bin/env bash
# Launch SGLang server with 4-bit AWQ/GPTQ quantization + FP8 KV cache.
# This is the production-ready configuration for Lab 15.
#
# Usage:
#   bash scripts/launch_sglang.sh [model-path] [port]
#
# Defaults:
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
#   PORT=30000

set -euo pipefail

MODEL="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
PORT="${2:-30000}"

echo "=== Launching SGLang Server ==="
echo "  Model:     $MODEL"
echo "  Port:      $PORT"
echo "  Quant:     4-bit AWQ"
echo "  KV Cache:  FP8"
echo "  Metrics:   enabled"
echo "  Cache:     enabled"
echo ""

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization awq \
  --kv-cache-dtype fp8_e4m3 \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-metrics \
  --enable-cache-report \
  --trust-remote-code
