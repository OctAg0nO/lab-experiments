"""Load environment config shared across lab sub-projects."""

import os
from pathlib import Path

# LiteLLM fetches a remote model cost map during `import dspy`.  Force the
# local fallback so the import doesn't hang on a flaky network request.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")


def get_env_or_raise(key: str) -> str:
    """Get an env var or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Missing {key}. Set it in the project root .env file "
            f"or export it in your shell."
        )
    return value


def get_env(key: str, default: str = "") -> str:
    """Get an env var with a fallback default."""
    return os.getenv(key, default)


def get_lm_model(default: str = "deepseek/deepseek-v4-flash") -> str:
    """Get the primary LLM model identifier from env."""
    return os.getenv("LLM_MODEL", default)


def get_student_lm_model(default: str = "ollama_chat/gemma4") -> str:
    """Get the student LLM model for distillation."""
    return os.getenv("STUDENT_LLM_MODEL", default)


def get_lm_temperature(default: float = 0.3) -> float:
    """Get LLM temperature from env."""
    try:
        return float(os.getenv("LLM_TEMPERATURE", str(default)))
    except (ValueError, TypeError):
        return default


def get_agent_port(agent_name: str = "orchestrator") -> int:
    return {"orchestrator": 8000, "explorer": 8001, "deepreader": 8002, "synthesizer": 8003, "critic": 8004}.get(agent_name, 8000)

def get_dapr_state_store() -> str:
    return os.getenv("DAPR_STATE_STORE", "research-state")

def get_dapr_pubsub() -> str:
    return os.getenv("DAPR_PUBSUB", "research-pubsub")

def get_dapr_llm_provider() -> str:
    return os.getenv("DAPR_LLM_PROVIDER", "llm-provider")

def project_root() -> Path:
    """Return the absolute path to the project root (parent of lab/)."""
    return Path(__file__).resolve().parent.parent.parent
