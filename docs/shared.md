# shared — Environment Config & LM Helpers

> **File:** `lab/shared/config.py`
> **Purpose:** Centralized environment variable loading and project path helpers shared across all labs.

## Setup

```python
from shared.config import (
    get_env_or_raise,
    get_env,
    get_lm_model,
    get_student_lm_model,
    get_lm_temperature,
    project_root,
)
```

## .env File Format

Place a `.env` file in the project root (parent of `lab/`). The config module reads from `os.environ` at runtime, so you can also export variables directly in your shell.

### Required

| Key | Description |
|-----|-------------|
| `DEEPSEEK_API_KEY` | API key for Deepseek LLM access. Without it, `get_env_or_raise("DEEPSEEK_API_KEY")` raises. |

### Optional

| Key | Default | Description |
|-----|---------|-------------|
| `LLM_MODEL` | `deepseek/deepseek-v4-flash` | Primary model identifier. Set to any DSPy-compatible model string. |
| `STUDENT_LLM_MODEL` | `ollama_chat/gemma4` | Student model for distillation experiments. |
| `LLM_TEMPERATURE` | `0.3` | Sampling temperature for LLM calls. |
| `CRAWL4AI_URL` | (none) | Crawl4AI service endpoint for web scraping (used in lab 08, 09, 10). |
| `DAPR_REDIS_HOST` | (none) | Redis host for Dapr state management (used in lab 10). |
| `DAPR_STATE_STORE` | (none) | Dapr state store component name (used in lab 10). |
| `DAPR_PUBSUB` | (none) | Dapr pubsub component name (used in lab 10). |

### Example

```bash
# Required
DEEPSEEK_API_KEY=sk-your-key-here

# Optional (shown with defaults)
LLM_MODEL=deepseek/deepseek-v4-flash
STUDENT_LLM_MODEL=ollama_chat/gemma4
LLM_TEMPERATURE=0.3

# Lab-specific
CRAWL4AI_URL=http://localhost:11235
DAPR_REDIS_HOST=localhost
DAPR_STATE_STORE=statestore
DAPR_PUBSUB=pubsub
```

## API Reference

### get_env_or_raise(key: str) -> str

Read a required environment variable. Raises `ValueError` with a clear message if the variable is missing or empty.

```python
api_key = get_env_or_raise("DEEPSEEK_API_KEY")
# Raises ValueError("Missing DEEPSEEK_API_KEY. Set it in the project root .env file or export it in your shell.")
```

Use this for credentials and other values that must be present at runtime.

### get_env(key: str, default: str = "") -> str

Read an optional environment variable with a fallback default.

```python
url = get_env("CRAWL4AI_URL", "http://localhost:11235")
# Returns the env value, or "http://localhost:11235" if not set.
```

Returns `str` regardless of the default type. Use `get_lm_temperature()` for numeric defaults.

### get_lm_model(default: str = "deepseek/deepseek-v4-flash") -> str

Read the primary LLM model identifier from the `LLM_MODEL` env var.

```python
model = get_lm_model()
# Returns os.getenv("LLM_MODEL", "deepseek/deepseek-v4-flash")
```

Pass a custom default to override the project default:

```python
model = get_lm_model(default="anthropic/claude-sonnet-4-20250514")
```

### get_student_lm_model(default: str = "ollama_chat/gemma4") -> str

Read the student LLM model identifier from the `STUDENT_LLM_MODEL` env var.

```python
student = get_student_lm_model()
# Returns os.getenv("STUDENT_LLM_MODEL", "ollama_chat/gemma4")
```

Used in distillation pipelines where a smaller/cheaper model learns from the primary model.

### get_lm_temperature(default: float = 0.3) -> float

Read the LLM temperature from the `LLM_TEMPERATURE` env var. Returns a `float`, falling back to the default on parse errors.

```python
temp = get_lm_temperature()
# Returns float(os.getenv("LLM_TEMPERATURE", "0.3"))

temp = get_lm_temperature(default=0.7)
# Override the default
```

Safe against invalid values: if the env var contains unparseable content, it returns the default instead of crashing.

### project_root() -> Path

Return the absolute path to the project root directory (the parent of `lab/`).

```python
root = project_root()
# Path("/home/user/projects/experiments")

env_path = project_root() / ".env"
```

Resolution is based on the file location of `shared/config.py`, so it works regardless of the current working directory.

## Summary

| Function | Reads Env Var | Default | Type | Raises |
|----------|---------------|---------|------|--------|
| `get_env_or_raise(key)` | any | none | `str` | `ValueError` if missing |
| `get_env(key, default)` | any | `""` | `str` | never |
| `get_lm_model(default)` | `LLM_MODEL` | `deepseek/deepseek-v4-flash` | `str` | never |
| `get_student_lm_model(default)` | `STUDENT_LLM_MODEL` | `ollama_chat/gemma4` | `str` | never |
| `get_lm_temperature(default)` | `LLM_TEMPERATURE` | `0.3` | `float` | never (falls back to default) |
| `project_root()` | none | n/a | `Path` | never |
