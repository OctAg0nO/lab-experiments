"""
DurableMetaAgent — Dapr-backed orchestrator that wraps the DSPy MetaAgent.

The DSPy MetaAgent remains the core orchestrator. DurableMetaAgent
wraps it in a DurableAgent workflow for crash-resistant execution.
Uses MetaAgent.run_stack_iter() internally — no loop duplication.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone

from dapr_agents import DurableAgent
from dapr_agents.llm import DaprChatClient
from dapr_agents.agents.configs import (
    AgentStateConfig,
    AgentExecutionConfig,
    WorkflowRetryPolicy,
    AgentObservabilityConfig,
    AgentTracingExporter,
    ToolExecutionMode,
    RuntimeSubscriptionConfig,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow import workflow_entry

from ..meta.meta_agent import MetaAgent, ResourceBudget
from ..meta.agent_generator import AgentGenerator
from ..memory.frontier import InMemoryFrontier
from ..dapr.frontier import DaprFrontier
from ..dapr.lse import DaprLSEOptimizer
from ..evolution.lse import LSEOptimizer


@dataclass
class DurableMetaConfig:
    """Configuration for DurableMetaAgent construction."""
    llm_component: str = "llm-provider"
    state_store: str = "meta-state"
    enable_tracing: bool = False
    tracing_endpoint: str = "http://localhost:9411/api/v2/spans"
    hot_reload_keys: list[str] | None = None
    use_dapr_frontier: bool = False
    use_dapr_lse: bool = False
    max_iterations_per_segment: int = 0
    """Max iterations before workflow restarts (Continue-as-New). 0 = no restart."""


class DurableMetaAgent(DurableAgent):
    """Durable orchestrator wrapping the DSPy MetaAgent.

    Accepts all dependencies directly in __init__ — no two-phase init needed.
    Uses MetaAgent.run_stack_iter() to avoid duplicating the iteration loop.
    """

    def __init__(
        self,
        generator: AgentGenerator,
        tool_defs: list[dict] | None = None,
        skills_dir: str = "/tmp/skills",
        budget: ResourceBudget | None = None,
        config: DurableMetaConfig | None = None,
        **kwargs,
    ):
        cfg = config or DurableMetaConfig()

        frontier = DaprFrontier() if cfg.use_dapr_frontier else InMemoryFrontier()
        lse = DaprLSEOptimizer() if cfg.use_dapr_lse else LSEOptimizer()

        self._meta = MetaAgent(
            llm=None,
            generator=generator,
            tool_defs=tool_defs or [],
            skills_dir=skills_dir,
            budget=budget or ResourceBudget(),
            frontier=frontier,
            lse=lse,
        )

        observability = None
        if cfg.enable_tracing:
            observability = AgentObservabilityConfig(
                enabled=True,
                service_name="durable-meta-agent",
                tracing_enabled=True,
                tracing_exporter=AgentTracingExporter.ZIPKIN,
                endpoint=cfg.tracing_endpoint,
            )

        hot_reload = None
        if cfg.hot_reload_keys:
            hot_reload = RuntimeSubscriptionConfig(
                store_name="runtime-config",
                keys=cfg.hot_reload_keys,
            )

        super().__init__(
            name="DurableMetaAgent",
            role="Research Orchestrator",
            goal="Conduct deep autonomous research using dynamically generated agents",
            instructions=[
                "Generate the right agents for each task",
                "Select the best agent via MultiChainComparison",
                "Track progress via frontier exploration",
                "Optimize agents via GFL pipeline",
                "Consolidate experience via Trace2Skill",
            ],
            llm=DaprChatClient(component_name=cfg.llm_component),
            state=AgentStateConfig(
                store=StateStoreService(store_name=cfg.state_store),
            ),
            execution=AgentExecutionConfig(
                max_iterations=30,
                tool_execution_mode=ToolExecutionMode.PARALLEL,
                retry_policy=WorkflowRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=5,
                    max_backoff_seconds=60,
                    backoff_multiplier=2.0,
                ),
            ),
            agent_observability=observability,
            configuration=hot_reload,
            **kwargs,
        )

    @workflow_entry
    def run_research(self, ctx, input: dict) -> dict:
        """Durable research workflow with Continue-as-New support.

        Uses MetaAgent.run_stack_iter() for the core DSPy loop.
        After max_iterations_per_segment iterations, restarts the workflow
        to purge execution history from the Dapr state store.
        """
        query = input.get("query", "")
        max_iterations = input.get("max_iterations", 5)
        segment_limit = input.get("segment_limit", 0)
        segment_start = input.get("segment_start", 0)

        yield ctx.set_state("query", query)
        yield ctx.set_state("started_at", datetime.now(timezone.utc).isoformat())

        meta = self._meta
        last_iteration = yield ctx.try_get_state("last_completed_iteration") or segment_start

        final_iteration = last_iteration
        for iteration, direction, entry, prediction, quality, state in meta.run_stack_iter(
            query, max_iterations
        ):
            if iteration <= last_iteration:
                continue
            final_iteration = iteration
            yield ctx.set_state("last_completed_iteration", iteration)

            if segment_limit and (iteration - segment_start) >= segment_limit:
                yield ctx.set_state("segment_restart_at", datetime.now(timezone.utc).isoformat())
                yield ctx.set_state("total_completed", iteration)
                yield ctx.call_workflow(
                    "run_research",
                    input={
                        "query": query,
                        "max_iterations": max_iterations,
                        "segment_limit": segment_limit,
                        "segment_start": iteration,
                    },
                )
                return {
                    "iterations": iteration,
                    "segment": True,
                    "frontier": meta.frontier.summary(),
                }

        yield ctx.set_state("completed_at", datetime.now(timezone.utc).isoformat())
        return {
            "iterations": final_iteration,
            "segment": False,
            "frontier": meta.frontier.summary(),
            "quality_trend": meta.lse.improvement_trend(),
        }
