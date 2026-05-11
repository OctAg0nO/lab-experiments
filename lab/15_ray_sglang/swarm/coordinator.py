from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Dict

from dapr_agents import DurableAgent
from dapr_agents.agents.configs import (
    AgentStateConfig,
    AgentExecutionConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    WorkflowRetryPolicy,
    AgentObservabilityConfig,
    AgentTracingExporter,
    ToolExecutionMode,
)
from dapr_agents.llm import DaprChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow import workflow_entry
from dapr_agents.workflow.utils.core import call_agent, trigger_agent

from ..dapr.frontier import DaprFrontier
from ..dapr.lse import DaprLSEOptimizer
from .messages import SwarmTask, SwarmDiscovery, SwarmHeartbeat, SwarmRegistration

logger = logging.getLogger(__name__)


class SwarmCoordinator(DurableAgent):
    """Orchestrator that dispatches research tasks to worker agents.

    Owns the single DaprFrontier. Workers receive tasks, execute them,
    and publish results back via pub/sub.

    Architecture:
        SwarmCoordinator (owns frontier, routes tasks)
            │
            ├── call_agent() ──► SwarmMetaAgent A (domain: research)
            ├── call_agent() ──► SwarmMetaAgent B (domain: verification)
            └── pub/sub ──────► swarm.heartbeat, swarm.discoveries
    """

    def __init__(
        self,
        frontier: DaprFrontier | None = None,
        lse: DaprLSEOptimizer | None = None,
        *,
        pubsub_name: str = "swarm-pubsub",
        state_store: str = "meta-state",
        heartbeat_timeout: int = 90,
        enable_tracing: bool = False,
        **kwargs,
    ):
        self.frontier = frontier or DaprFrontier()
        self.lse = lse or DaprLSEOptimizer()
        self._heartbeat_timeout = heartbeat_timeout
        self._workers: Dict[str, SwarmRegistration] = {}
        self._pending_tasks: Dict[str, str] = {}
        self._last_heartbeat: Dict[str, str] = {}

        tracing = None
        if enable_tracing:
            tracing = AgentObservabilityConfig(
                enabled=True,
                service_name="swarm-coordinator",
                tracing_enabled=True,
                tracing_exporter=AgentTracingExporter.ZIPKIN,
            )

        super().__init__(
            name="SwarmCoordinator",
            role="Swarm Orchestrator",
            goal="Coordinate multiple meta agents for distributed research",
            instructions=[
                "Dispatch research directions to the best-suited worker",
                "Track worker health via heartbeats",
                "Consolidate findings from all workers",
                "Detect and reassign tasks from failed workers",
            ],
            llm=DaprChatClient(),
            state=AgentStateConfig(
                store=StateStoreService(store_name=state_store),
            ),
            pubsub=AgentPubSubConfig(
                pubsub_name=pubsub_name,
                agent_topic="coordinator",
                broadcast_topic="swarm.broadcast",
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name=state_store),
                team_name="swarm-1",
            ),
            execution=AgentExecutionConfig(
                max_iterations=50,
                tool_execution_mode=ToolExecutionMode.PARALLEL,
                retry_policy=WorkflowRetryPolicy(
                    max_attempts=3,
                    initial_backoff_seconds=5,
                ),
            ),
            agent_observability=tracing,
            **kwargs,
        )

    def register_worker(self, registration: SwarmRegistration) -> None:
        self._workers[registration.agent_id] = registration
        logger.info("Worker registered: %s (domain=%s)", registration.agent_id, registration.domain)

    def _healthy_workers(self) -> list[SwarmRegistration]:
        now = datetime.now(timezone.utc)
        healthy = []
        for aid, reg in self._workers.items():
            last = self._last_heartbeat.get(aid, "")
            if last:
                last_time = datetime.fromisoformat(last)
                if (now - last_time).total_seconds() > self._heartbeat_timeout:
                    logger.warning("Worker %s heartbeat expired, marking offline", aid)
                    reg.status = "offline"
                    continue
            if reg.status == "available":
                healthy.append(reg)
        return healthy

    def _pick_worker(self, domain_hint: str = "") -> SwarmRegistration | None:
        healthy = self._healthy_workers()
        if not healthy:
            return None
        if domain_hint:
            domain_match = [w for w in healthy if w.domain == domain_hint]
            if domain_match:
                return domain_match[0]
        return healthy[0]

    @workflow_entry
    def run_swarm(self, ctx, input: dict) -> dict:
        """Main swarm orchestration loop.

        Owns the frontier, dispatches tasks to workers via call_agent(),
        collects results, and tracks LSE improvement.
        """
        query = input.get("query", "")
        max_iterations = input.get("max_iterations", 10)
        worker_app_ids = input.get("worker_app_ids", [])

        yield ctx.set_state("query", query)
        yield ctx.set_state("started_at", datetime.now(timezone.utc).isoformat())

        self.frontier.seed_from_query(query)
        iteration = 0
        all_discoveries: list[dict] = []

        while iteration < max_iterations:
            iteration += 1
            yield ctx.set_state("current_iteration", iteration)

            direction = self.frontier.next_action()
            if direction is None:
                logger.info("Frontier saturated at iteration %d", iteration)
                break

            worker_idx = (iteration - 1) % max(1, len(worker_app_ids))
            worker_app_id = worker_app_ids[worker_idx] if worker_app_ids else "swarm-worker"
            logger.info("Dispatching '%s' to %s", direction.topic, worker_app_id)

            try:
                result = yield call_agent(
                    ctx,
                    "execute_task",
                    input={
                        "direction": direction.topic,
                        "exploration_depth": direction.exploration_depth,
                        "max_iterations": input.get("task_iterations", 3),
                    },
                    app_id=worker_app_id,
                )
            except Exception as e:
                logger.error("Worker %s failed: %s", worker_app_id, e)
                yield ctx.set_state(f"task_{iteration}_error", str(e))
                continue

            if result:
                confidence_delta = result.get("quality_score", 0.0) * 0.3
                follow_ups = result.get("follow_up_directions", [])
                findings = result.get("findings", [])

                self.frontier.absorb_findings(
                    direction.topic, confidence_delta, len(findings), follow_ups
                )

                discovery = {
                    "topic": direction.topic,
                    "findings": findings,
                    "quality": result.get("quality_score", 0.0),
                    "worker": worker_app_id,
                }
                all_discoveries.append(discovery)

                directions_list = list(self.frontier.directions.values())
                non_sat = sum(1 for d in directions_list if not d.is_saturated(0.95))
                saturation = 1.0 - (non_sat / max(1, len(directions_list)))
                self.lse.record_run(
                    f"iter_{iteration}",
                    {
                        "num_directions": len(directions_list),
                        "num_findings": len(all_discoveries),
                        "frontier_saturation": saturation,
                    },
                    direction.topic,
                )

                yield ctx.set_state(f"iter_{iteration}_result", discovery)

        yield ctx.set_state("completed_at", datetime.now(timezone.utc).isoformat())
        yield ctx.set_state("total_iterations", iteration)

        return {
            "iterations": iteration,
            "frontier": self.frontier.summary(),
            "discoveries": len(all_discoveries),
            "quality_trend": self.lse.improvement_trend(),
        }
