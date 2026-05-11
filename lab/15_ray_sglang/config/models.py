"""Multi-model configuration — hybrid architecture for OctAg0nO Lab 15.

Uses DSPy's dspy.context(lm=...) to route different agent roles to different
language models served by SGLang. Each role gets a model optimized for its task.

Roles:
  orchestrator — Voice + User interaction (low latency, omni)
  researcher — High-throughput web research (MoE, cheap)
  verifier — Formal logic, math, code (precision, reasoning)
  tool_user — MCP/A2UI tool calling (function-calling accuracy)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import dspy

logger = logging.getLogger(__name__)


@dataclass
class ModelProfile:
    """Configuration for a single model served by SGLang.

    Each model runs on a separate SGLang server with its own port.
    DSPy connects via OpenAI-compatible endpoint.
    """
    name: str
    model_id: str
    sglang_port: int
    role: str
    temperature: float = 0.3
    max_tokens: int = 4096
    quant: str = "awq"
    tp_size: int = 1
    description: str = ""

    def __post_init__(self):
        valid_roles = {"orchestrator", "researcher", "verifier", "tool_user"}
        if self.role not in valid_roles:
            raise ValueError(
                f"Invalid role '{self.role}' for model '{self.name}'. "
                f"Must be one of: {', '.join(sorted(valid_roles))}"
            )
        if not self.model_id or "/" not in self.model_id:
            raise ValueError(
                f"Invalid model_id '{self.model_id}' for '{self.name}'. "
                f"Must be a HuggingFace model ID (e.g. 'org/model')."
            )

    @property
    def sglang_endpoint(self) -> str:
        return f"http://localhost:{self.sglang_port}/v1"

    def to_lm(self) -> dspy.LM:
        """Create a DSPy LM instance pointing to this model's SGLang server."""
        return dspy.LM(
            model=f"openai/{self.model_id}",
            api_base=self.sglang_endpoint,
            api_key="None",
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )


# ── Model Profile Presets (May 2026) ────────────────────────────────
#
# ⚠️  ASPIRATIONAL MODELS: The model IDs below represent announced or
# rumored models that may not be available on HuggingFace yet. When a model
# is released, update the model_id and verify the SGLang launch command.
# Until then, substitute with the closest available alternative:
#   Orchestrator → meta-llama/Llama-3.3-70B-Instruct (or Qwen3-TTS for voice)
#   Researcher   → deepseek-ai/DeepSeek-V3 (or Qwen3-235B-A55B)
#   Verifier     → microsoft/Phi-4-mini-instruct (or Phi-4-14B)
#   Tool User    → mistralai/Mistral-Small-24B-Instruct-2501

LLAMA_4_OMNI = ModelProfile(
    name="Llama 4-12B-Omni",
    model_id="meta-llama/Llama-4-12B-Omni",
    sglang_port=30001,
    role="orchestrator",
    temperature=0.3,
    max_tokens=4096,
    quant="fp8",
    description="Native omni-model: voice, vision, text. Speculative audio decoding. 12B fits in ~14GB FP8.",
)

DEEPSEEK_V4_LITE = ModelProfile(
    name="DeepSeek-V4-Lite-MoE",
    model_id="deepseek-ai/DeepSeek-V4-Lite",
    sglang_port=30002,
    role="researcher",
    temperature=0.5,
    max_tokens=8192,
    quant="awq",
    description="28B MoE, 1.8B active params. Extreme throughput via SGLang. ~15GB at 4-bit AWQ.",
)

PHI_4_PRO = ModelProfile(
    name="Phi-4-Pro-24B",
    model_id="microsoft/Phi-4-pro-24B",
    sglang_port=30003,
    role="verifier",
    temperature=0.1,
    max_tokens=4096,
    quant="awq",
    tp_size=2,
    description="Matches GPT-4o at Lean 4/Z3 verification. Low temperature for deterministic output.",
)

MISTRAL_NEMO_V3 = ModelProfile(
    name="Mistral NeMo-v3-14B",
    model_id="mistralai/Mistral-NeMo-v3-14B",
    sglang_port=30004,
    role="tool_user",
    temperature=0.2,
    max_tokens=4096,
    quant="fp8",
    description="Specialized function-calling head. Near-100% JSON tool accuracy. Native FP8.",
)

# Default model for general use (falls back to orchestrator model)
DEFAULT_MODEL = LLAMA_4_OMNI

# All profiles indexed by role
MODEL_REGISTRY: dict[str, ModelProfile] = {
    p.role: p for p in [
        LLAMA_4_OMNI,
        DEEPSEEK_V4_LITE,
        PHI_4_PRO,
        MISTRAL_NEMO_V3,
    ]
}


def get_model(role: str) -> ModelProfile:
    """Get the model profile for a given role.

    Args:
        role: orchestrator, researcher, verifier, or tool_user.

    Returns:
        ModelProfile for the role, or DEFAULT_MODEL if role not found.
    """
    return MODEL_REGISTRY.get(role, DEFAULT_MODEL)


def configure_lm(profile: ModelProfile | None = None, role: str = "orchestrator") -> dspy.LM:
    """Create a DSPy LM for the given profile or role.

    Args:
        profile: A ModelProfile instance.
        role: Role name to look up if profile is None.

    Returns:
        Configured dspy.LM instance.
    """
    if profile is None:
        profile = get_model(role)
    logger.info("Configuring LM: %s (%s) → %s", profile.name, profile.role, profile.sglang_endpoint)
    return profile.to_lm()


def configure_all_models() -> dict[str, dspy.LM]:
    """Create DSPy LM instances for all registered model profiles.

    Returns:
        Dict mapping role → dspy.LM instance.
    """
    return {
        role: profile.to_lm()
        for role, profile in MODEL_REGISTRY.items()
    }


# ── SGLang Launch Configuration ─────────────────────────────────────

SGLANG_LAUNCH_TEMPLATE = (
    "python -m sglang.launch_server "
    "--model-path {model_id} "
    "--host 0.0.0.0 "
    "--port {port} "
    "--quantization {quant} "
    "--trust-remote-code "
    "{extra}"
)


def sglang_launch_command(profile: ModelProfile) -> str:
    """Generate SGLang server launch command for a model profile."""
    extra = ""
    if profile.tp_size > 1:
        extra += f"--tp {profile.tp_size} "
    if profile.role == "orchestrator":
        extra += "--enable-metrics --enable-cache-report "
    if profile.role == "researcher":
        extra += "--max-running-requests 256 --max-total-tokens 32768 "
    if profile.role == "verifier":
        extra += "--mem-fraction-static 0.9 "
    return SGLANG_LAUNCH_TEMPLATE.format(
        model_id=profile.model_id,
        port=profile.sglang_port,
        quant=profile.quant,
        extra=extra.strip(),
    )


def sglang_launch_commands() -> list[tuple[str, str, int]]:
    """Generate launch commands for all registered models.

    Returns:
        List of (command, model_name, port) tuples.
    """
    return [
        (sglang_launch_command(p), p.name, p.sglang_port)
        for p in MODEL_REGISTRY.values()
    ]
