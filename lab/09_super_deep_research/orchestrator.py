"""
ResearchOrchestrator — the top-level LSE-driven research loop.

Dispatches specialized agents, maintains the ResearchFrontier, runs
heartbeat-based reflection, and consolidates trajectories into skills.
"""

from __future__ import annotations

import json

import dspy

from .frontier import ResearchFrontier
from .memory.store import MemoryStore
from .evolution.lse import LSEOptimizer
from .evolution.trace2skill import SkillConsolidator
from .agents import (
    create_explorer, create_deep_reader, create_synthesizer, create_critic,
)
from lab.shared.mcp import MCPClient


def _default_quality_fn(state: dict) -> float:
    """Measure research quality: coverage × depth × novelty proxy."""
    num_directions = state.get("num_directions", 0)
    num_findings = state.get("num_findings", 0)
    frontier_saturation = state.get("frontier_saturation", 0.0)
    if num_directions == 0:
        return 0.0
    coverage = min(1.0, num_directions / 10.0)
    depth = min(1.0, num_findings / max(1, num_directions) / 3.0)
    novelty = frontier_saturation
    return (coverage * 0.4 + depth * 0.4 + novelty * 0.2)


class ResearchOrchestrator:
    """Main research loop with LSE meta-optimization.

    Per iteration:
    1. Select next action from ResearchFrontier (UCB)
    2. Dispatch to the right agent
    3. Absorb findings into frontier + knowledge graph
    4. Heartbeat: reflect, consolidate, check stagnation
    5. Update LSE optimizer with quality delta
    """

    def __init__(
        self,
        mcp_client: MCPClient,
        tool_defs: list[dict],
        lm: dspy.LM,
        memory: MemoryStore,
        frontier: ResearchFrontier,
        max_iterations: int = 6,
    ):
        self.client = mcp_client
        self.tool_defs = tool_defs
        self.lm = lm
        self.memory = memory
        self.frontier = frontier
        self.max_iterations = max_iterations

        # Build tool subsets per agent
        all_fns = mcp_client.build_tool_fns(tool_defs)
        self.fetch_tools = [t for t in all_fns if t.__name__ in ("fetch", "md", "crawl")]
        self.search_tools = [t for t in all_fns if t.__name__ in ("search", "chat", "model_list")]
        self.all_tools = all_fns

        # Create agents
        self.explorer = create_explorer(self.search_tools or self.fetch_tools, lm)
        self.deep_reader = create_deep_reader(self.fetch_tools or all_fns, lm)
        self.synthesizer = create_synthesizer(self.all_tools, lm)
        self.critic = create_critic(lm)

        # Evolution components
        self.lse = LSEOptimizer(quality_fn=_default_quality_fn)
        self.skill_consolidator = SkillConsolidator(memory.base / "consolidated_skills")

        # State
        self.iteration = 0
        self.all_trajectories: list[dict] = []
        self.findings_text: list[str] = []

    # ------------------------------------------------------------------
    # Main loop
    # ------------------------------------------------------------------

    def run(self, user_query: str):
        self.frontier.seed_from_query(user_query)

        while self.iteration < self.max_iterations:
            self.iteration += 1
            direction = self.frontier.next_action()
            if direction is None:
                print(f"\n  [i{self.iteration}] Frontier saturated — stopping")
                break

            topic = direction.topic
            print(f"\n  [i{self.iteration}] Exploring: {topic[:80]}")

            # ---- Agent dispatch based on frontier state ----
            if direction.exploration_depth == 0:
                self._explore(topic)
            elif direction.confidence < 0.6:
                self._deep_read(topic)
            else:
                self._synthesize(topic)

            # ---- Critic review every 2 iterations ----
            if self.iteration % 2 == 0 and self.findings_text:
                self._critique()

            # ---- Heartbeat: reflect + consolidate ----
            if self.iteration % 3 == 0:
                self._heartbeat()

            state = {
                "num_directions": len(self.frontier.directions),
                "num_findings": len(self.findings_text),
                "frontier_saturation": 1.0 - (
                    len([d for d in self.frontier.directions if d.confidence < 0.95])
                    / max(1, len(self.frontier.directions))
                ),
            }
            self.lse.record_run(f"iter_{self.iteration}", state, topic)

        # Final consolidation
        if self.all_trajectories:
            skill = self.skill_consolidator.consolidate(self.all_trajectories)
            self.skill_consolidator.save_skill("final_consolidation", skill)
            self.memory.save_skill("orchestrator_skill", skill)

        return self._report()

    # ------------------------------------------------------------------
    # Agent dispatch methods
    # ------------------------------------------------------------------

    def _explore(self, topic: str):
        try:
            result = self.explorer(task=topic)
            if hasattr(result, "result") and result.result:
                directions = result.result.directions
                follow_ups = [d.topic for d in directions if hasattr(d, "topic")]
                self.frontier.seed_from_directions(follow_ups, parent=topic)
                self.frontier.absorb_findings(topic, 0.3, 1, follow_ups)
                self.memory.graph.add_finding(
                    f"explore_{self.iteration}", str([d.topic for d in directions]),
                    source="explorer", category="direction",
                )
                if hasattr(result, "trajectory"):
                    self.all_trajectories.append({"trajectory": result.trajectory, "agent": "explorer"})
        except Exception as e:
            print(f"    [!] Explorer failed: {e}")

    def _deep_read(self, topic: str):
        # Use fetch tool to get content about the topic
        fetch_fn = next((t for t in self.fetch_tools if t.__name__ == "fetch"), None)
        if fetch_fn:
            try:
                content = fetch_fn(url=topic)
                self.findings_text.append(f"Topic: {topic}\n{content[:1000]}")
                url = topic
            except Exception:
                content = f"Exploring: {topic}"
                url = ""
        else:
            content = f"Exploring: {topic}"
            url = ""

        try:
            result = self.deep_reader(topic=topic, url=url)
            if hasattr(result, "result") and result.result:
                r = result.result
                if hasattr(r, "findings"):
                    for idx, f in enumerate(r.findings):
                        self.memory.graph.add_finding(
                            f"deepread_{self.iteration}_{idx}",
                            f.claim[:500],
                            source=f.source,
                            category="finding",
                        )
                self.frontier.absorb_findings(topic, 0.2, len(getattr(r, "findings", [])), [])
                if hasattr(result, "trajectory"):
                    self.all_trajectories.append({"trajectory": result.trajectory, "agent": "deep_reader"})
        except Exception as e:
            print(f"    [!] DeepReader failed: {e}")

    def _synthesize(self, topic: str):
        findings_str = json.dumps([{"topic": topic}], indent=2)
        try:
            result = self.synthesizer(task=topic, findings=findings_str)
            if hasattr(result, "result") and result.result:
                r = result.result
                if hasattr(r, "gaps") and r.gaps:
                    self.frontier.seed_from_directions(r.gaps, parent=topic)
                self.frontier.absorb_findings(topic, 0.15, 0, getattr(r, "gaps", []))
                if hasattr(result, "trajectory"):
                    self.all_trajectories.append({"trajectory": result.trajectory, "agent": "synthesizer"})
        except Exception as e:
            print(f"    [!] Synthesizer failed: {e}")

    def _critique(self):
        summary = "\n".join(self.findings_text[-3:])
        try:
            result = self.critic(research_summary=summary)
            if hasattr(result, "result") and result.result:
                r = result.result
                if hasattr(r, "follow_ups") and r.follow_ups:
                    self.frontier.seed_from_directions(r.follow_ups, parent="critique")
                if hasattr(result, "trajectory"):
                    self.all_trajectories.append({"trajectory": result.trajectory, "agent": "critic"})
        except Exception as e:
            print(f"    [!] Critic failed: {e}")

    # ------------------------------------------------------------------
    # Heartbeat
    # ------------------------------------------------------------------

    def _heartbeat(self):
        print(f"\n  --- Heartbeat (iter {self.iteration}) ---")
        print(f"  {self.frontier.summary()}")
        print(f"  {self.memory.summary()}")

        trend = self.lse.improvement_trend()
        if trend and all(t < 0 for t in trend[-2:]):
            print("  [!] Stagnation detected — triggering Critic review")
            self._critique()

        if self.all_trajectories:
            skill = self.skill_consolidator.consolidate(self.all_trajectories[-3:])
            self.skill_consolidator.save_skill(f"heartbeat_{self.iteration}", skill)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _report(self) -> dict:
        return {
            "iterations": self.iteration,
            "frontier": self.frontier.summary(),
            "memory": self.memory.summary(),
            "findings_count": len(self.findings_text),
            "trajectories_count": len(self.all_trajectories),
            "improvement_trend": self.lse.improvement_trend(),
            "best_strategy": self.lse.best_strategy(),
        }
