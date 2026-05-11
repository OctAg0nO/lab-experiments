from __future__ import annotations

import logging
from datetime import datetime, timezone

from dapr_agents import DurableAgent, AgentRunner
from dapr_agents.agents.configs import (
    AgentStateConfig,
    AgentPubSubConfig,
    AgentRegistryConfig,
    AgentExecutionConfig,
    WorkflowRetryPolicy,
    AgentObservabilityConfig,
    AgentTracingExporter,
    ToolExecutionMode,
)
from dapr_agents.llm import DaprChatClient
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow import workflow_entry, message_router
from dapr_agents.workflow.utils.core import trigger_agent

from ..core.durable_meta_agent import DurableMetaAgent, DurableMetaConfig
from ..meta.meta_agent import MetaAgent, ResourceBudget
from ..meta.agent_generator import AgentGenerator
from ..meta.agent_stack import AgentEntry, AgentStack
from ..memory.frontier import InMemoryFrontier
from ..dapr.frontier import DaprFrontier
from ..evolution.lse import LSEOptimizer
from ..dapr.lse import DaprLSEOptimizer
from .messages import SwarmTask, SwarmDiscovery, SwarmHeartbeat, SwarmRegistration

logger = logging.getLogger(__name__)


class SwarmMetaAgent(DurableMetaAgent):
    """A DurableMetaAgent that receives tasks via pub/sub and publishes findings.

    Subscribes to swarm.tasks for incoming research directions.
    Runs the DSPy research loop, then publishes results to swarm.discoveries.
    Sends periodic heartbeats to swarm.heartbeat.

    All DSPy internals (AgentGenerator, GFL pipeline, LSE, Trace2Skill)
    are inherited from DurableMetaAgent unchanged.
    """

    def __init__(
        self,
        generator: AgentGenerator,
        tool_defs: list[dict] | None = None,
        skills_dir: str = "/tmp/skills",
        budget: ResourceBudget | None = None,
        config: DurableMetaConfig | None = None,
        *,
        agent_id: str = "swarm-worker",
        domain: str = "general",
        pubsub_name: str = "swarm-pubsub",
        heartbeat_interval: int = 30,
        **kwargs,
    ):
        self._domain = domain
        self._agent_id = agent_id
        self._heartbeat_interval = heartbeat_interval
        self._completed_tasks = 0
        self._failed_tasks = 0

        super().__init__(
            generator=generator,
            tool_defs=tool_defs,
            skills_dir=skills_dir,
            budget=budget,
            config=config,
            name=agent_id,
            pubsub=AgentPubSubConfig(
                pubsub_name=pubsub_name,
                agent_topic=agent_id,
                broadcast_topic="swarm.broadcast",
            ),
            registry=AgentRegistryConfig(
                store=StateStoreService(store_name=config.state_store if config else "meta-state"),
                team_name="swarm-1",
            ),
            **kwargs,
        )

    @message_router(message_model=dict, pubsub="swarm-pubsub", topic="swarm.tasks")
    async def on_task(self, message: dict) -> None:
        """Receive a research task from the coordinator and execute it."""
        task = SwarmTask(**message)
        logger.info("Received task: %s (depth=%d)", task.direction, task.exploration_depth)

        try:
            results = self._meta.run_stack(
                task.direction,
                max_iterations=task.max_iterations,
            )
            eval_result = self._meta.evaluate_self()
            quality = eval_result.get("avg_quality", 0.0)

            frontier_snapshot = list(self._meta.frontier.directions.keys())
            follow_ups = [t for t in frontier_snapshot if t != task.direction][-5:]

            discovery = SwarmDiscovery(
                topic=task.direction,
                findings=results,
                confidence_delta=quality * 0.3,
                quality_score=quality,
                follow_up_directions=follow_ups,
                worker_id=self._agent_id,
                correlation_id=task.correlation_id,
            )
            await self.publish("swarm.discoveries", discovery.model_dump())
            self._completed_tasks += 1
            logger.info("Completed task: %s (quality=%.3f)", task.direction, quality)

        except Exception as e:
            logger.error("Task failed: %s — %s", task.direction, e)
            self._failed_tasks += 1
            await self.publish("swarm.discoveries", SwarmDiscovery(
                topic=task.direction,
                findings=[],
                confidence_delta=0.0,
                quality_score=0.0,
                worker_id=self._agent_id,
                correlation_id=task.correlation_id,
            ).model_dump())

    @message_router(message_model=dict, pubsub="swarm-pubsub", topic="swarm.inquiry")
    async def on_inquiry(self, message: dict) -> None:
        """Respond to A2A inquiries from other agents."""
        from .messages import SwarmInquiry, SwarmResponse
        inquiry = SwarmInquiry(**message)
        logger.debug("Inquiry from %s: %s", inquiry.source_agent, inquiry.question[:60])

        response = SwarmResponse(
            answer=f"Agent {self._agent_id} (domain: {self._domain}) processed inquiry",
            source_agent=self._agent_id,
            target_agent=inquiry.source_agent,
            correlation_id=inquiry.correlation_id,
        )
        await self.publish("swarm.response", response.model_dump())

    async def send_heartbeat(self) -> None:
        """Publish liveness status to the swarm."""
        heartbeat = SwarmHeartbeat(
            agent_id=self._agent_id,
            status="alive",
            load=len(self._meta.stack) if self._meta else 0,
            completed_tasks=self._completed_tasks,
            failed_tasks=self._failed_tasks,
            domain=self._domain,
        )
        await self.publish("swarm.heartbeat", heartbeat.model_dump())

    def run_worker(self, port: int = 8001) -> None:
        """Serve this worker as a Dapr-enabled service."""
        runner = AgentRunner()
        runner.serve(self, port=port)
