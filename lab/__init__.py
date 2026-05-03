"""Lab experiments package."""
import os

# LiteLLM fetches a remote model cost map at import time (httpx.get).
# Ensure the local fallback is used so `import dspy` doesn't hang on a
# flaky network call. Must be set before any dspy import.
os.environ.setdefault("LITELLM_LOCAL_MODEL_COST_MAP", "True")
