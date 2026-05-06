"""Dapr durability layer for the meta-agent framework.

All DSPy modules remain the core reasoning engine. Dapr adds:
- Durable workflow checkpointing (survive crashes)
- State persistence (Redis-backed frontier, LSE, agent registry)
- Observability (OpenTelemetry spans via Zipkin/OTLP)
- Hot-reload configuration (swap LLM at runtime)
- Secrets management (API keys via Dapr secretstore)
- Retry policies (exponential backoff on tool calls)

Usage:
    from ..dapr.wrappers import GeneratedDurableAgent, wrap_module
    from ..dapr.frontier import DaprFrontier
    from ..dapr.lse import DaprLSEOptimizer
"""

from .wrappers import GeneratedDurableAgent, wrap_module

__all__ = ["GeneratedDurableAgent", "wrap_module"]
