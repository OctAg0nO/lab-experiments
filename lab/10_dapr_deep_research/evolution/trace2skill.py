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
    """Uses DSPy ChainOfThought to extract patterns from trajectories.

    Can be optimized with BootstrapFewShot by providing labeled examples
    of good vs poor trajectory extractions.
    """

    def __init__(self, persist_dir: str | Path):
        self.dir = Path(persist_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self._extractor = dspy.ChainOfThought(ExtractPatterns)

    def consolidate(self, trajectories: list[dict]) -> dict:
        demos = []
        error_patterns = []
        success_patterns = []

        for traj in trajectories:
            steps = traj if isinstance(traj, list) else traj.get("trajectory", [])
            context_parts = []
            for i, step in enumerate(steps[:8]):
                reasoning = step.get("reasoning", "")[:300]
                output = str(step.get("output", ""))[:200]
                code = step.get("code", "")[:150]
                context_parts.append(f"Step {i+1}:\nReasoning: {reasoning}\nCode: {code}\nOutput: {output}")

            trajectory_text = "\n\n".join(context_parts)
            if not trajectory_text.strip():
                continue

            pred = self._extractor(trajectory_context=trajectory_text)

            if pred.error_patterns and len(pred.error_patterns) > 10:
                error_patterns.append({"symptom": pred.error_patterns[:300], "extracted_by": "dspy.CoT"})
            if pred.success_patterns and len(pred.success_patterns) > 10:
                success_patterns.append({"pattern": pred.success_patterns[:300], "extracted_by": "dspy.CoT"})

            # Last step output as demonstration
            if steps and steps[-1].get("output"):
                demos.append({
                    "reasoning": steps[-1].get("reasoning", "")[:500],
                    "improvement": pred.improvement_suggestion[:200],
                })

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
