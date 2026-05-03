"""Load environment config shared across lab sub-projects."""

import os
from pathlib import Path


def get_env_or_raise(key: str) -> str:
    """Get an env var or raise a clear error."""
    value = os.getenv(key)
    if not value:
        raise ValueError(
            f"Missing {key}. Set it in the project root .env file "
            f"or export it in your shell."
        )
    return value


def project_root() -> Path:
    """Return the absolute path to the project root (parent of lab/)."""
    return Path(__file__).resolve().parent.parent.parent
