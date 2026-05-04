"""
LSE — Learning to Self-Evolve using DSPy modules for quality evaluation
and improvement tracking.

Reward: r = quality(c1) - quality(c0) evaluated via dspy.ChainOfThought.
"""

from __future__ import annotations

from dataclasses import dataclass

import dspy

from lab.shared.research import MAX_BOOTSTRAPPED_DEMOS, MAX_LABELED_DEMOS


class QualityEvaluation(dspy.Signature):
    """Evaluate research iteration quality based on coverage, depth, and novelty."""
    num_directions: int = dspy.InputField(desc="Number of active research directions")
    num_findings: int = dspy.InputField(desc="Number of findings collected")
    frontier_saturation: float = dspy.InputField(desc="Fraction of directions at high confidence (0-1)")
    quality_score: float = dspy.OutputField(desc="Research quality from 0.0 to 1.0")
    explanation: str = dspy.OutputField(desc="Why this score was assigned")


@dataclass
class LSERun:
    strategy_id: str
    quality_score: float
    strategy_description: str
    num_directions: int
    num_findings: int


class LSEOptimizer:
    """Meta-optimizer that uses DSPy ChainOfThought to evaluate research quality.

    Tracks improvement across runs: r = quality(c1) - quality(c0).
    The quality evaluator is a compiled DSPy program that can itself be
    optimized via BootstrapFewShot.
    """

    def __init__(self):
        self.runs: list[LSERun] = []
        self._evaluator = dspy.ChainOfThought(QualityEvaluation)

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._evaluator
        student = dspy.ChainOfThought(QualityEvaluation) if student_lm else teacher
        if student_lm:
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(
            metric=lambda _ex, pred, _trace: hasattr(pred, "quality_score") and 0.0 <= pred.quality_score <= 1.0,
            max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS, max_labeled_demos=MAX_LABELED_DEMOS,
        )
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._evaluator = compiled

    def compute_improvement(self, current_quality: float) -> float:
        if not self.runs:
            return 0.0
        return current_quality - self.runs[-1].quality_score

    def record_run(self, strategy_id: str, state: dict, strategy_description: str):
        pred = self._evaluator(
            num_directions=state.get("num_directions", 0),
            num_findings=state.get("num_findings", 0),
            frontier_saturation=state.get("frontier_saturation", 0.0),
        )
        quality = max(0.0, min(1.0, pred.quality_score))
        self.runs.append(LSERun(
            strategy_id=strategy_id,
            quality_score=quality,
            strategy_description=strategy_description,
            num_directions=state.get("num_directions", 0),
            num_findings=state.get("num_findings", 0),
        ))

    def best_strategy(self) -> str | None:
        if not self.runs:
            return None
        return max(self.runs, key=lambda r: r.quality_score).strategy_id

    def improvement_trend(self) -> list[float]:
        scores = [r.quality_score for r in self.runs]
        if len(scores) < 2:
            return []
        return [scores[i] - scores[i - 1] for i in range(1, len(scores))]
