#!/usr/bin/env bash
# Launch multiple SGLang servers for the OctAg0nO hybrid model architecture.
# Each model serves a different role on a different port.
#
# Usage:
#   bash scripts/launch_multi_model.sh [models...]
#
# Examples:
#   bash scripts/launch_multi_model.sh                    # All models
#   bash scripts/launch_multi_model.sh orchestrator       # Just Llama 4 Omni
#   bash scripts/launch_multi_model.sh researcher,verifier # DeepSeek + Phi

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
MODELS_PY="$SCRIPT_DIR/../config/models.py"

# Parse model selection
SELECTED="${1:-all}"

launch_model() {
    local MODEL_ID="$1"
    local PORT="$2"
    local QUANT="$3"
    local NAME="$4"
    local EXTRA="$5"

    echo "=== Launching $NAME on port $PORT ==="
    echo "  Model: $MODEL_ID"
    echo "  Quant: $QUANT"
    echo "  Extra: $EXTRA"
    echo ""

    python -m sglang.launch_server \
        --model-path "$MODEL_ID" \
        --host 0.0.0.0 \
        --port "$PORT" \
        --quantization "$QUANT" \
        --trust-remote-code \
        $EXTRA &
}

if [ "$SELECTED" = "all" ] || [[ "$SELECTED" == *"orchestrator"* ]]; then
    launch_model \
        "meta-llama/Llama-4-12B-Omni" \
        30001 \
        fp8 \
        "Llama 4-12B-Omni (Orchestrator)" \
        "--enable-metrics --enable-cache-report"
fi

if [ "$SELECTED" = "all" ] || [[ "$SELECTED" == *"researcher"* ]]; then
    launch_model \
        "deepseek-ai/DeepSeek-V4-Lite" \
        30002 \
        awq \
        "DeepSeek-V4-Lite-MoE (Researcher)" \
        "--max-running-requests 256 --max-total-tokens 32768"
fi

if [ "$SELECTED" = "all" ] || [[ "$SELECTED" == *"verifier"* ]]; then
    launch_model \
        "microsoft/Phi-4-pro-24B" \
        30003 \
        awq \
        "Phi-4-Pro-24B (Verifier)" \
        "--mem-fraction-static 0.9 --tp 2"
fi

if [ "$SELECTED" = "all" ] || [[ "$SELECTED" == *"tool_user"* ]]; then
    launch_model \
        "mistralai/Mistral-NeMo-v3-14B" \
        30004 \
        fp8 \
        "Mistral NeMo-v3-14B (Tool User)" \
        ""
fi

echo ""
echo "=== All requested models launched ==="
echo "Wait for each to show 'Server started' before connecting."
wait
