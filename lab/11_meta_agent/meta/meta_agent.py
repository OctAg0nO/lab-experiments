"""MetaAgent — orchestrates dynamic agent generation, execution, LSE, and Trace2Skill."""

from __future__ import annotations

import json
from datetime import datetime
from typing import Any

import dspy

from ..evolution.lse import LSEOptimizer
from ..evolution.trace2skill import SkillConsolidator
from ..memory.frontier import InMemoryFrontier
from .agent_generator import AgentGenerator
from .agent_stack import AgentEntry, AgentStack


class SelectNextAgent(dspy.Signature):
    """Select the best agent from the stack for the current task."""
    task: str = dspy.InputField()
    agents_available: str = dspy.InputField(desc="JSON list of {name, role, run_count, avg_quality}")
    selected_agent: str = dspy.OutputField(desc="Name of the best agent")
    reasoning: str = dspy.OutputField(desc="Why this agent was chosen")


class MetaAgent:
    """Generates agents on the fly, runs them via stack, optimizes via LSE,
    and consolidates patterns via Trace2Skill."""

    def __init__(
        self,
        llm: dspy.LM,
        generator: AgentGenerator,
        tool_defs: list[dict] | None = None,
        skills_dir: str | None = None,
    ):
        self._llm = llm
        self._generator = generator
        self._tool_defs = tool_defs or []
        self.stack = AgentStack()
        self.frontier = InMemoryFrontier()
        self.lse = LSEOptimizer()
        self._selector = dspy.ChainOfThought(SelectNextAgent)
        self._skills_dir = skills_dir
        self._consolidator = SkillConsolidator(skills_dir or "/tmp/skills")

    # -- agent generation --

    def generate_agents(self, task: str) -> int:
        """Analyze task and generate needed agents onto the stack."""
        definitions = self._generator.analyze(task)
        count = 0
        for definition in definitions:
            name = definition.get("name", f"agent_{count}")
            if self.stack.get(name):
                continue
            entry = self._generator.generate(definition)
            self.stack.push(entry)
            count += 1
        return count

    def generate_additional(self, task: str, gap_description: str) -> AgentEntry:
        """Generate a new agent to fill a specific gap."""
        definition = {
            "name": f"gap_agent_{datetime.now().strftime('%H%M%S')}",
            "role": "Gap Filler",
            "goal": gap_description,
            "signature": "task: str -> result: str",
            "tools": [t.get("name", "") for t in self._tool_defs],
        }
        entry = self._generator.generate(definition)
        self.stack.push(entry)
        return entry

    # -- execution --

    def run_stack(
        self,
        task: str,
        max_iterations: int = 5,
        call_agent_fn: callable = None,
    ) -> list[dict]:
        """Run agents from the stack against a task using LSE loop."""
        results: list[dict] = []
        self.frontier.seed_from_query(task)

        for iteration in range(max_iterations):
            direction = self.frontier.next_action()
            if not direction:
                break

            # Select the best agent from stack
            agents_json = json.dumps([
                {"name": e.name, "role": e.role,
                 "run_count": e.run_count, "avg_quality": e.avg_quality}
                for e in self.stack
            ])
            selection = self._selector(
                task=direction.topic,
                agents_available=agents_json,
            )
            selected_name = getattr(selection, "selected_agent", "")
            entry = self.stack.get(selected_name)

            if entry is None and self.stack:
                entry = self.stack.peek()
            if entry is None:
                continue

            # Run the agent
            module = self._generator.generate_module(entry)
            try:
                prediction = module(task=direction.topic)
            except Exception as exc:
                prediction = dspy.Prediction(
                    result=f"Agent failed: {exc}", error=str(exc)
                )

            result_entry = {
                "iteration": iteration,
                "agent": entry.name,
                "topic": direction.topic,
                "prediction": prediction,
            }
            results.append(result_entry)

            # LSE evaluation
            quality = self._evaluate(prediction, direction.topic)
            self.stack.record_run(entry.name, quality)
            self.frontier.absorb_findings(direction.topic, quality * 0.3, 1, [])
            state = {
                "num_directions": len(self.frontier.directions),
                "num_findings": len(results),
                "frontier_saturation": 0.0,
            }
            self.lse.record_run(f"iter_{iteration}", state, direction.topic)

        return results

    # -- evaluation --

    def _evaluate(self, prediction: Any, topic: str) -> float:
        """Score prediction quality 0-1."""
        score = 0.5
        pred_dict = prediction if isinstance(prediction, dict) else {}
        if hasattr(prediction, "get"):
            pred_dict = prediction
        if not pred_dict:
            pred_dict = {}
        for v in pred_dict.values():
            if isinstance(v, str) and len(v) > 50:
                score += 0.1
            elif isinstance(v, list) and len(v) > 0:
                score += 0.1
            elif isinstance(v, dict) and len(v) > 0:
                score += 0.1
        return min(1.0, max(0.0, score))

    # -- consolidation --

    def consolidate(self, results: list[dict]) -> dict:
        """Run Trace2Skill on execution results."""
        trajectories: list[dict] = []
        for r in results:
            pred = r.get("prediction", {})
            pred_str = str(pred)[:200] if pred else ""
            trajectories.append({
                "strategy_id": r["agent"],
                "trajectory": [{
                    "reasoning": f"Executed agent {r['agent']} for task: {r['topic']}",
                    "code": f"module(task='{r['topic'][:50]}')",
                    "output": pred_str,
                }],
            })
        patterns = self._consolidator.consolidate(trajectories)
        return patterns

    def save_skill(self, patterns: dict) -> str:
        """Save consolidated patterns as a skill."""
        name = f"meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._consolidator.save_skill(name, patterns)
        return name

    def summary(self) -> str:
        return (
            f"Stack: {self.stack.summary()}\n"
            f"Frontier: {self.frontier.summary()}\n"
            f"LSE runs: {len(self.lse.runs)}"
        )
