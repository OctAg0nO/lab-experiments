"""
ResearchWorkflow — orchestrator DurableAgent with the LSE-driven research loop.
Dispatches sub-agents via call_agent() and persists state via Dapr state store.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone

from dapr_agents import DurableAgent
from dapr_agents.llm import DaprChatClient
from dapr_agents.agents.configs import (
    AgentStateConfig, AgentExecutionConfig, WorkflowRetryPolicy,
)
from dapr_agents.storage.daprstores.stateservice import StateStoreService
from dapr_agents.workflow import workflow_entry
from dapr_agents.workflow.utils.core import call_agent

from ..evolution.lse import LSEOptimizer
from ..memory.dapr_frontier import DaprFrontier
from ..mcp.bridge import MCPBridge


class ResearchWorkflow(DurableAgent):
    """Orchestrator agent that runs the LSE-driven research loop.

    Each research iteration:
    1. Selects next direction from DaprFrontier (persisted in Redis)
    2. Dispatches the appropriate agent via call_agent()
    3. Absorbs findings into frontier
    4. Tracks LSE improvement
    5. Checkpoints state after each iteration
    """

    def __init__(self, bridge: MCPBridge, frontier: DaprFrontier, **kwargs):
        self.bridge = bridge
        self.frontier = frontier
        self.lse = LSEOptimizer(quality_fn=lambda s: (
            min(1.0, s.get("num_directions", 0) / 10.0) * 0.4 +
            min(1.0, s.get("num_findings", 0) / max(1, s.get("num_directions", 1)) / 3.0) * 0.4 +
            s.get("frontier_saturation", 0.0) * 0.2
        ))
        self.all_findings: list[str] = []
        self.all_trajectories: list[dict] = []

        super().__init__(
            name="ResearchWorkflow",
            role="Research Orchestrator",
            goal="Conduct deep autonomous research using specialized agents",
            instructions=[
                "Select the right agent for each research phase",
                "Track progress and detect stagnation",
                "Consolidate findings into actionable knowledge",
            ],
            llm=DaprChatClient(component_name="llm-provider"),
            state=AgentStateConfig(store=StateStoreService(store_name="research-state")),
            execution=AgentExecutionConfig(
                max_iterations=30,
                retry_policy=WorkflowRetryPolicy(max_retry_attempts=2, retry_backoff_interval_seconds=3),
            ),
            **kwargs,
        )

    @workflow_entry
    def run_research(self, ctx, input: dict) -> dict:
        query = input.get("query", "")
        max_iter = input.get("max_iterations", 6)
        yield ctx.set_state("research_started_at", datetime.now(timezone.utc).isoformat())

        self.frontier.seed_from_query(query)
        yield ctx.set_state("frontier_summary", self.frontier.summary())

        iteration = 0
        while iteration < max_iter:
            iteration += 1
            yield ctx.set_state("current_iteration", iteration)

            direction = self.frontier.next_action()
            if direction is None:
                break

            topic = direction.topic

            # Dispatch agent based on frontier state
            if direction.exploration_depth == 0:
                result = yield call_agent(
                    ctx, "explore",
                    input={"topic": topic},
                    app_id="explorer-agent",
                )
                if result:
                    follow_ups = [d.get("topic", "") for d in result.get("directions", []) if d.get("topic")]
                    self.frontier.seed_from_directions(follow_ups, parent=topic)
                    self.frontier.absorb_findings(topic, 0.3, 1, follow_ups)
                    self.all_findings.append(json.dumps(result))

            elif direction.confidence < 0.6:
                result = yield call_agent(
                    ctx, "deep_read",
                    input={"topic": topic},
                    app_id="deepreader-agent",
                )
                if result:
                    n_findings = len(result.get("findings", []))
                    self.frontier.absorb_findings(topic, 0.2, n_findings, [])
                    self.all_findings.append(json.dumps(result))

            else:
                result = yield call_agent(
                    ctx, "synthesize",
                    input={"topic": topic, "findings": self.all_findings[-3:]},
                    app_id="synthesizer-agent",
                )
                if result:
                    gaps = result.get("gaps", [])
                    self.frontier.seed_from_directions(gaps, parent=topic)
                    self.frontier.absorb_findings(topic, 0.15, 0, gaps)

            # Heartbeat every 3 iterations
            if iteration % 3 == 0:
                yield ctx.set_state("heartbeat_frontier", self.frontier.summary())
                yield ctx.set_state("heartbeat_findings_count", len(self.all_findings))

            state = {
                "num_directions": len(self.frontier.directions),
                "num_findings": len(self.all_findings),
                "frontier_saturation": 1.0 - (
                    len([d for d in self.frontier.directions if d.confidence < 0.95])
                    / max(1, len(self.frontier.directions))
                ),
            }
            self.lse.record_run(f"iter_{iteration}", state, topic)
            yield ctx.set_state(f"lse_iter_{iteration}", {"quality": state})

        # Final checkpoint
        yield ctx.set_state("research_completed_at", datetime.now(timezone.utc).isoformat())
        yield ctx.set_state("final_iterations", iteration)
        yield ctx.set_state("final_findings_count", len(self.all_findings))

        return {
            "iterations": iteration,
            "frontier": self.frontier.summary(),
            "findings_count": len(self.all_findings),
            "improvement_trend": self.lse.improvement_trend(),
        }
