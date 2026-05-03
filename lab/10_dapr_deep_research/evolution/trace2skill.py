"""
Trace2Skill — uses DSPy ChainOfThought + BootstrapFewShot to consolidate
execution trajectories into reusable skill demonstrations.
"""

from __future__ import annotations

import json
from pathlib import Path

import dspy


class ExtractPatterns(dspy.Signature):
    """Extract reusable reasoning patterns from an execution trajectory."""
    trajectory_context: str = dspy.InputField(desc="Execution steps with reasoning, code, and output")
    error_patterns: str = dspy.OutputField(desc="What went wrong and why")
    success_patterns: str = dspy.OutputField(desc="Effective reasoning patterns to reuse")
    improvement_suggestion: str = dspy.OutputField(desc="How to improve next attempt")


class SkillConsolidator:
    """Uses DSPy ChainOfThought + Parallel to extract patterns from trajectories.

    Trajectories are processed in parallel via dspy.Parallel, matching the
    Trace2Skill paper's parallel multi-agent patch proposal (Stage 2).
    Can be optimized with BootstrapFewShot by providing labeled examples.
    """

    def __init__(self, persist_dir: str | Path):
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._extractor = dspy.ChainOfThought(ExtractPatterns)
        self._parallel = dspy.Parallel(dspy.ChainOfThought(ExtractPatterns), n=4)

    def _build_trajectory_text(self, traj: dict) -> str:
        steps = traj if isinstance(traj, list) else traj.get("trajectory", [])
        parts = []
        for i, step in enumerate(steps[:8]):
            parts.append(f"Step {i+1}:\nReasoning: {step.get('reasoning', '')[:300]}\nCode: {step.get('code', '')[:150]}\nOutput: {str(step.get('output', ''))[:200]}")
        return "\n\n".join(parts)

    def consolidate(self, trajectories: list[dict]) -> dict:
        demos = []
        error_patterns = []
        success_patterns = []

        texts = [self._build_trajectory_text(t) for t in trajectories]
        texts = [t for t in texts if t.strip()]

        if len(texts) >= 4:
            predictions = self._parallel(trajectory_context=texts[:4])
            if not isinstance(predictions, list):
                predictions = [predictions]
        else:
            predictions = [self._extractor(trajectory_context=t) for t in texts]

        for pred in predictions:
            if pred is None:
                continue
            if hasattr(pred, "error_patterns") and pred.error_patterns and len(pred.error_patterns) > 10:
                error_patterns.append({"symptom": pred.error_patterns[:300], "extracted_by": "dspy.Parallel"})
            if hasattr(pred, "success_patterns") and pred.success_patterns and len(pred.success_patterns) > 10:
                success_patterns.append({"pattern": pred.success_patterns[:300], "extracted_by": "dspy.Parallel"})

        for traj in trajectories[:5]:
            steps = traj if isinstance(traj, list) else traj.get("trajectory", [])
            if steps and steps[-1].get("output"):
                demos.append({"reasoning": steps[-1].get("reasoning", "")[:500], "output": str(steps[-1].get("output", ""))[:200]})

        return {
            "error_patterns": error_patterns[:10],
            "success_patterns": success_patterns[:10],
            "demonstrations": demos[:5],
            "n_trajectories": len(trajectories),
        }

    def save_skill(self, name: str, skill: dict):
        (self.dir / f"{name}.json").write_text(json.dumps(skill, indent=2, default=str))

    def load_skills(self) -> list[dict]:
        return [json.loads(f.read_text()) for f in sorted(self.dir.glob("*.json"), reverse=True)]
