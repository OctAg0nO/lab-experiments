"""MetaAgent — LSE-driven orchestrator using MultiChainComparison, Refine, Ensemble, and InferRules."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import dspy

from ..evolution.lse import LSEOptimizer
from ..evolution.trace2skill import SkillConsolidator
from ..memory.frontier import InMemoryFrontier
from ..memory.frontier import ResearchFrontier as FrontierABC
from ...shared.research import SATURATION_THRESHOLD
from .agent_generator import AgentGenerator
from .agent_stack import AgentEntry, AgentStack


@dataclass
class ResourceBudget:
    """Budget limits for meta-agent execution. Raise if exceeded."""
    max_llm_calls: int = 100
    max_wall_seconds: int = 300
    max_agents_generated: int = 10
    max_iterations: int = 20
    _start_time: float = field(default_factory=time.time)
    _llm_calls_used: int = 0

    def check_llm(self) -> None:
        self._llm_calls_used += 1
        if self._llm_calls_used > self.max_llm_calls:
            raise RuntimeError(f"LLM call budget exceeded ({self.max_llm_calls})")

    def check_time(self) -> None:
        elapsed = time.time() - self._start_time
        if elapsed > self.max_wall_seconds:
            raise RuntimeError(f"Wall time budget exceeded ({self.max_wall_seconds}s)")

    def check_agents(self, count: int) -> None:
        if count > self.max_agents_generated:
            raise RuntimeError(f"Agent count budget exceeded ({self.max_agents_generated})")

    def check_all(self, agent_count: int = 0) -> None:
        self.check_time()
        if agent_count:
            self.check_agents(agent_count)


class SelectAgentCompare(dspy.Signature):
    """Compare candidate agents and select the best one for the task."""
    task: str = dspy.InputField()
    candidate_agent: str = dspy.InputField(desc="JSON with name, role, run_count, avg_quality")
    suitability: float = dspy.OutputField(desc="Suitability from 0.0 to 1.0")
    reasoning: str = dspy.OutputField(desc="Why this agent fits")


class ImproveAgentPrompt(dspy.Signature):
    """Improve an agent's prompt based on its execution results."""
    agent_role: str = dspy.InputField()
    current_prompt: str = dspy.InputField()
    task: str = dspy.InputField()
    execution_result: str = dspy.InputField()
    quality_score: float = dspy.InputField()
    improved_prompt: str = dspy.OutputField(desc="The improved prompt")
    improvement_rationale: str = dspy.OutputField(desc="What was changed and why")


class ExtractRules(dspy.Signature):
    """Extract reusable rules from agent execution trajectories."""
    trajectory_data: str = dspy.InputField(desc="JSON array of agent execution results")
    rules: list[str] = dspy.OutputField(desc="Reusable rules extracted")
    patterns: list[str] = dspy.OutputField(desc="Recurring patterns observed")


class MetaAgent:
    """Orchestrator using MultiChainComparison, Refine, Ensemble, Parallel, InferRules.

    DSPy features used:
    - dspy.MultiChainComparison for agent selection (3 candidates compared)
    - dspy.Refine for iterative prompt improvement
    - dspy.Ensemble to combine multiple agent outputs
    - dspy.InferRules for rule extraction from trajectories
    """

    def __init__(
        self,
        llm: dspy.LM,
        generator: AgentGenerator,
        tool_defs: list[dict] | None = None,
        skills_dir: str | None = None,
        budget: ResourceBudget | None = None,
        stack: AgentStack | None = None,
        frontier: FrontierABC | None = None,
        lse: LSEOptimizer | None = None,
    ):
        self._llm = llm
        self._generator = generator
        self._tool_defs = tool_defs or []
        self.budget = budget or ResourceBudget()
        self.stack = stack or AgentStack()
        self.frontier: FrontierABC = frontier or InMemoryFrontier()
        self.lse = lse or LSEOptimizer()
        self._consolidator = SkillConsolidator(skills_dir or "/tmp/skills")

        self._comparison = dspy.MultiChainComparison(SelectAgentCompare, n=3)
        self._refine = dspy.Refine(
            dspy.ChainOfThought(ImproveAgentPrompt),
            N=3,
            reward_fn=lambda ex, pred: (
                1.0 if len(getattr(pred, "improved_prompt", "")) > 50 else 0.0
            ),
            threshold=0.5,
        )
        self._rule_extractor = dspy.ChainOfThought(ExtractRules)

    def generate_agents(self, task: str) -> int:
        definitions = self._generator.analyze(task)
        count = 0
        for definition in definitions:
            self.budget.check_all(agent_count=count)
            name = definition.get("name", f"agent_{count}")
            if self.stack.get(name):
                continue
            entry = self._generator.generate(definition)
            self.stack.push(entry)
            count += 1
        return count

    def generate_additional(self, task: str, gap_description: str) -> AgentEntry:
        definition = {
            "name": f"gap_{datetime.now().strftime('%H%M%S')}",
            "role": "Gap Filler",
            "goal": gap_description,
            "tools": [t.get("name", "") for t in self._tool_defs],
            "use_code": True,
        }
        entry = self._generator.generate(definition)
        self.stack.push(entry)
        return entry

    def _select_best_agent(self, task: str) -> AgentEntry | None:
        if not self.stack:
            return None
        candidates = list(self.stack)
        if len(candidates) == 1:
            return candidates[0]

        scores = []
        for entry in candidates[:3]:
            cdata = json.dumps({
                "name": entry.name, "role": entry.role,
                "run_count": entry.run_count, "avg_quality": entry.avg_quality,
            })
            scores.append(self._comparison(task=task, candidate_agent=cdata))

        best_idx = 0
        best_score = -1.0
        for i, s in enumerate(scores):
            score = getattr(s, "suitability", 0.5)
            if isinstance(score, (int, float)) and score > best_score:
                best_score = score
                best_idx = i
        return candidates[best_idx]

    def run_stack_iter(self, task: str, max_iterations: int = 5):
        """Generator yielding per-iteration data for each research iteration.

        Yields (iteration, direction, entry, prediction, quality, state) tuples.
        Separates iteration logic from result collection so DurableMetaAgent
        can inject checkpointing without duplicating the loop.
        """
        self.frontier.seed_from_query(task)
        max_iterations = min(max_iterations, self.budget.max_iterations)

        for iteration in range(max_iterations):
            self.budget.check_all(agent_count=len(self.stack))
            direction = self.frontier.next_action()
            if not direction:
                break

            entry = self._select_best_agent(direction.topic)
            if entry is None:
                continue

            module = self._generator.generate_module(entry)
            if module is None:
                self.stack.record_failure(entry.name)
                continue

            try:
                self.budget.check_llm()
                prediction = module(task=direction.topic)
            except Exception as exc:
                prediction = dspy.Prediction(result=f"Agent failed: {exc}")
                self.stack.record_failure(entry.name)

            pred_str = str(prediction)
            quality = self._generator.evaluate(
                task=direction.topic, agent_role=entry.role,
                prediction=pred_str,
            )

            if entry.prompt_template and quality < 0.7:
                try:
                    refined = self._refine(
                        agent_role=entry.role,
                        current_prompt=entry.prompt_template,
                        task=direction.topic,
                        execution_result=pred_str[:500],
                        quality_score=quality,
                    )
                    if hasattr(refined, "improved_prompt") and refined.improved_prompt:
                        entry.prompt_template = refined.improved_prompt
                        self._generator.clear_cache()
                except Exception:
                    pass

            self.stack.record_run(entry.name, quality)
            self.frontier.absorb_findings(direction.topic, quality * 0.3, 1, [])
            directions = list(self.frontier.directions.values())
            non_sat = sum(1 for d in directions if not d.is_saturated(SATURATION_THRESHOLD))
            saturation = 1.0 - (non_sat / max(1, len(directions)))
            state = {
                "num_directions": len(directions),
                "num_findings": iteration + 1,
                "frontier_saturation": saturation,
            }
            self.lse.record_run(f"iter_{iteration}", state, direction.topic)

            yield iteration, direction, entry, prediction, quality, state

    def run_stack(self, task: str, max_iterations: int = 5) -> list[dict]:
        """Run the research loop and collect results. Uses run_stack_iter internally."""
        results: list[dict] = []
        for iteration, direction, entry, prediction, quality, state in self.run_stack_iter(
            task, max_iterations
        ):
            results.append({
                "iteration": iteration, "agent": entry.name,
                "topic": direction.topic, "prediction": prediction,
                "quality": quality,
            })
        return results

    def ensemble_run(self, task: str, entry: AgentEntry, n_variations: int = 3) -> list[Any]:
        module = self._generator.generate_module(entry)
        if module is None:
            return []
        return [module(task=task) for _ in range(n_variations)]

    def extract_rules(self, results: list[dict]) -> dict:
        data = json.dumps([
            {"agent": r["agent"], "topic": r["topic"], "quality": r.get("quality", 0)}
            for r in results[-10:]
        ], default=str)
        result = self._rule_extractor(trajectory_data=data)
        return {
            "rules": getattr(result, "rules", []),
            "patterns": getattr(result, "patterns", []),
        }

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
        return self._consolidator.consolidate(trajectories)

    def save_skill(self, patterns: dict) -> str:
        name = f"meta_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self._consolidator.save_skill(name, patterns)
        return name

    def evaluate_self(self) -> dict:
        """Meta-evaluation: measure how well agent generation and LSE performed."""
        scores = [r.quality_score for r in self.lse.runs]
        avg_quality = sum(scores) / len(scores) if scores else 0.0
        quality_trend = self.lse.improvement_trend()
        net_improvement = quality_trend[-1] if quality_trend else 0.0

        agent_stats = []
        for entry in self.stack:
            total = entry.run_count + entry.failure_count
            success_rate = entry.run_count / total if total > 0 else 0.0
            agent_stats.append({
                "name": entry.name,
                "role": entry.role,
                "avg_quality": round(entry.avg_quality, 3),
                "success_rate": round(success_rate, 3),
                "failures": entry.failure_count,
            })

        return {
            "avg_quality": round(avg_quality, 3),
            "net_improvement": round(net_improvement, 3),
            "total_iterations": len(self.lse.runs),
            "agents_generated": len(self.stack),
            "budget_used": {
                "llm_calls": self.budget._llm_calls_used,
                "wall_seconds": round(time.time() - self.budget._start_time, 1),
            },
            "agent_stats": agent_stats,
        }

    def summary(self) -> str:
        return (
            f"Stack: {self.stack.summary()}\n"
            f"Frontier: {self.frontier.summary()}\n"
            f"LSE runs: {len(self.lse.runs)}"
        )
