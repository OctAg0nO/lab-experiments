# shared — Environment Config & LM Helpers

> **Files:** `lab/shared/config.py`, `lab/shared/research.py`
> **Purpose:** Centralized environment variable loading, project path helpers, and research primitives shared across all labs.

## Setup

```python
from shared.config import (
    get_lm_model,
    get_student_lm_model,
    get_lm_temperature,
    get_agent_port,
    get_dapr_state_store,
    get_dapr_pubsub,
    project_root,
)
from shared.research import (
    ResearchDirection,
    ResearchFrontier,
    SATURATION_THRESHOLD,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_LABELED_DEMOS,
)
```

## .env File Format

Place a `.env` file in the project root (parent of `lab/`). The config module reads from `os.environ` at runtime, so you can also export variables directly in your shell.

### Required

| Key | Description |
|-----|-------------|
| `DEEPSEEK_API_KEY` | API key for Deepseek LLM access. Required for any LLM call. |

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

### get_agent_port(agent_name: str = "orchestrator") -> int

Returns the default gRPC port for a Dapr agent.

```python
port = get_agent_port("explorer")         # 8001
port = get_agent_port("orchestrator")     # 8000
port = get_agent_port("unknown")          # 8000 (fallback)
```

Used by `_make_dapr_cmd()` in the CLI to start each agent server on its assigned port.

### get_dapr_state_store() -> str

Returns the Dapr state store component name from `DAPR_STATE_STORE` env var, defaulting to `"research-state"`.

```python
store_name = get_dapr_state_store()
# os.getenv("DAPR_STATE_STORE", "research-state")
```

Used in `dapr_frontier.py`, `workflow.py`, and all 4 agent classes to avoid hardcoding.

### get_dapr_pubsub() -> str

Returns the Dapr pubsub component name from `DAPR_PUBSUB` env var, defaulting to `"research-pubsub"`.

### project_root() -> Path

Return the absolute path to the project root directory (the parent of `lab/`).

```python
root = project_root()
# Path("/home/user/projects/experiments")

env_path = project_root() / ".env"
```

Resolution is based on the file location of `shared/config.py`, so it works regardless of the current working directory.

---

## `research.py` — Shared Research Primitives

Module defining the core dataclass, ABC, and constants used by both `InMemoryFrontier` and `DaprFrontier`.

### ResearchDirection

A `@dataclass` representing a single research direction in the frontier.

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `topic` | `str` | required | Research topic/subtopic |
| `confidence` | `float` | `0.0` | Current confidence score 0–1 |
| `exploration_depth` | `int` | `0` | Number of times explored |
| `source_count` | `int` | `0` | Sources discovered |
| `last_updated` | `str` | `""` | ISO 8601 timestamp |
| `parent_topic` | `str\|None` | `None` | Parent direction (for hierarchical tracking) |
| `seed_query` | `str` | `""` | Query used to discover this direction |

**Methods:**

```python
d = ResearchDirection(topic="Transformers")
d.ucb_score(total_explorations=10)       # UCB1 score for exploration/exploitation
d.is_saturated(threshold=0.95)           # True if confidence >= threshold
d.to_dict()                              # Serialize to dict
ResearchDirection.from_dict(data)        # Deserialize from dict
```

### ResearchFrontier (ABC)

Abstract base class defining the frontier interface. All implementations must provide:

| Method | Returns | Description |
|--------|---------|-------------|
| `seed_from_query(query)` | `None` | Add initial research query as a direction |
| `seed_from_directions(topics, parent)` | `None` | Add new directions from discovered subtopics |
| `next_action()` | `ResearchDirection\|None` | Pick the best direction to explore next |
| `absorb_findings(topic, delta, sources, follow_ups)` | `None` | Update a direction with new findings |
| `saturated()` | `bool` | True if all directions are saturated |
| `summary()` | `str` | Human-readable frontier status |
| `directions` | `dict` (property) | All directions dict[str, ResearchDirection] |
| `total_explorations` | `int` | Running count of explorations |

**Concrete helpers:**

```python
frontier._active_count()                # Count of non-saturated directions
frontier._next_action_from_directins(candidates)  # UCB-based selection
```

### Constants

| Constant | Value | Used In |
|----------|-------|---------|
| `SATURATION_THRESHOLD` | `0.95` | Frontier saturation checks, workflow.py |
| `MAX_BOOTSTRAPPED_DEMOS` | `4` | All `BootstrapFewShot.compile()` calls |
| `MAX_LABELED_DEMOS` | `2` | All `BootstrapFewShot.compile()` calls |

## Summary

| Function | Reads Env Var | Default | Type | Raises |
|----------|---------------|---------|------|--------|
| `get_lm_model(default)` | `LLM_MODEL` | `deepseek/deepseek-v4-flash` | `str` | never |
| `get_student_lm_model(default)` | `STUDENT_LLM_MODEL` | `ollama_chat/gemma4` | `str` | never |
| `get_lm_temperature(default)` | `LLM_TEMPERATURE` | `0.3` | `float` | never (falls back to default) |
| `get_agent_port(name)` | none | `8000` | `int` | never |
| `get_dapr_state_store()` | `DAPR_STATE_STORE` | `research-state` | `str` | never |
| `get_dapr_pubsub()` | `DAPR_PUBSUB` | `research-pubsub` | `str` | never |
| `project_root()` | none | n/a | `Path` | never |
