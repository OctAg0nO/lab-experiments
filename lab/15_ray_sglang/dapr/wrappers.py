"""
GeneratedDurableAgent — wraps a dynamically generated DSPy module in a DurableAgent shell.

The DSPy module (RLM, ReAct, CodeAct, or ChainOfThought) stays as the core reasoning
engine — unchanged. The DurableAgent provides:
  - @workflow_entry for durable checkpointing
  - DaprChatClient as the LLM backend
  - AgentTool list from the MCP bridge
  - AgentStateConfig for state persistence
  - AgentExecutionConfig with retry policy
  - AgentObservabilityConfig for tracing
"""

from __future__ import annotations

from typing import Any

import dspy
from dapr_agents import DurableAgent, AgentTool
from dapr_agents.agents.configs import (
    AgentStateConfig,
    AgentExecutionConfig,
    ToolExecutionMode,
    WorkflowRetryPolicy,
    AgentObservabilityConfig,
    AgentTracingExporter,
)
from dapr_agents.llm import DaprChatClient
from dapr_agents.workflow import workflow_entry
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from ..meta.agent_stack import AgentEntry


def wrap_module(
    dspy_module: dspy.Module,
    entry: AgentEntry,
    *,
    agent_tools: list[AgentTool] | None = None,
    llm_component: str = "llm-provider",
    state_store: str = "agent-workflow",
    enable_tracing: bool = False,
    tracing_endpoint: str = "http://localhost:9411/api/v2/spans",
    max_iterations: int = 10,
    retry_attempts: int = 3,
) -> GeneratedDurableAgent:
    """Wrap a DSPy module in a GeneratedDurableAgent shell.

    Args:
        dspy_module: The DSPy module (RLM, ReAct, CodeAct, or CoT).
        entry: The AgentEntry describing the agent's role and tools.
        agent_tools: AgentTool list from MCPBridge.get_agent_tools().
        llm_component: Dapr Conversation component name.
        state_store: Dapr state store name for workflow state.
        enable_tracing: Enable OpenTelemetry spans.
        tracing_endpoint: Zipkin/OTLP endpoint.
        max_iterations: Max LLM-tool turns in the durable workflow.
        retry_attempts: Workflow retry count on failure.

    Returns:
        A GeneratedDurableAgent ready for AgentRunner.serve() or .run().
    """
    return GeneratedDurableAgent(
        dspy_module=dspy_module,
        name=entry.name,
        role=entry.role,
        goal=entry.goal,
        instructions=[],
        tools=agent_tools or [],
        llm_component=llm_component,
        state_store=state_store,
        enable_tracing=enable_tracing,
        tracing_endpoint=tracing_endpoint,
        max_iterations=max_iterations,
        retry_attempts=retry_attempts,
    )


class GeneratedDurableAgent(DurableAgent):
    """DurableAgent shell around a dynamically generated DSPy module.

    The inner DSPy module does the reasoning and tool-calling (via DSPy's
    built-in tool system using MCPBridge.get_dspy_tools()). The DurableAgent
    shell provides durability: workflow checkpointing, observability spans,
    state persistence, and retry policies.

    DSPy code is NOT replaced — it's the core engine. This is a wrapper.
    """

    def __init__(
        self,
        dspy_module: dspy.Module,
        name: str,
        role: str,
        goal: str,
        instructions: list[str] | None = None,
        tools: list[AgentTool] | None = None,
        llm_component: str = "llm-provider",
        state_store: str = "agent-workflow",
        enable_tracing: bool = False,
        tracing_endpoint: str = "http://localhost:9411/api/v2/spans",
        max_iterations: int = 10,
        retry_attempts: int = 3,
        **kwargs,
    ):
        self._module = dspy_module  # DSPy module stays as the core engine

        # Observability config (optional — disabled by default)
        observability = None
        if enable_tracing:
            observability = AgentObservabilityConfig(
                enabled=True,
                service_name=f"generated-{name}",
                tracing_enabled=True,
                tracing_exporter=AgentTracingExporter.ZIPKIN,
                endpoint=tracing_endpoint,
            )

        super().__init__(
            name=name,
            role=role,
            goal=goal,
            instructions=instructions or [],
            tools=tools or [],
            llm=DaprChatClient(component_name=llm_component),
            state=AgentStateConfig(
                store=StateStoreService(store_name=state_store),
            ),
            execution=AgentExecutionConfig(
                max_iterations=max_iterations,
                tool_execution_mode=ToolExecutionMode.PARALLEL,
                retry_policy=WorkflowRetryPolicy(
                    max_attempts=retry_attempts,
                    initial_backoff_seconds=5,
                    max_backoff_seconds=60,
                    backoff_multiplier=2.0,
                ),
            ),
            agent_observability=observability,
            **kwargs,
        )

    @workflow_entry
    def execute(self, ctx, input: dict) -> dict:
        """Durable workflow entry point.

        Delegates to the inner DSPy module. State is checkpointed before
        and after execution so a crash mid-iteration can be resumed.
        """
        yield ctx.set_state(f"{self.name}_input", input)

        # The inner DSPy module does all the reasoning — unchanged
        result = self._module(task=input.get("task", ""))

        output = {"result": str(result), "module_type": type(self._module).__name__}
        yield ctx.set_state(f"{self.name}_output", output)
        return output
