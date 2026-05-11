#!/usr/bin/env bash
# Launch SGLang server with tensor parallelism for multi-GPU models.
#
# Usage:
#   bash scripts/launch_sglang_tp.sh [model-path] [tp-size] [port]
#
# Defaults:
#   MODEL=meta-llama/Meta-Llama-3.1-8B-Instruct
#   TP=2
#   PORT=30000

set -euo pipefail

MODEL="${1:-meta-llama/Meta-Llama-3.1-8B-Instruct}"
TP="${2:-2}"
PORT="${3:-30000}"

echo "=== Launching SGLang Server (Tensor Parallelism) ==="
echo "  Model:     $MODEL"
echo "  TP Size:   $TP"
echo "  Port:      $PORT"
echo "  Quant:     4-bit AWQ"
echo "  KV Cache:  FP8"
echo ""

python -m sglang.launch_server \
  --model-path "$MODEL" \
  --quantization awq \
  --kv-cache-dtype fp8_e4m3 \
  --tp "$TP" \
  --host 0.0.0.0 \
  --port "$PORT" \
  --enable-metrics \
  --enable-cache-report \
  --trust-remote-code
