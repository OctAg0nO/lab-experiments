from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass
class LSERun:
    strategy_id: str
    quality_score: float
    strategy_description: str
    num_directions: int
    num_findings: int


class LSEOptimizer:
    def __init__(self, quality_fn: Callable[[Any], float]):
        self.quality_fn = quality_fn
        self.runs: list[LSERun] = []

    def compute_improvement(self, current_quality: float) -> float:
        if not self.runs:
            return 0.0
        return current_quality - self.runs[-1].quality_score

    def record_run(self, strategy_id: str, state: Any, strategy_description: str):
        quality = self.quality_fn(state)
        self.runs.append(LSERun(
            strategy_id=strategy_id, quality_score=quality,
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
        return [scores[i] - scores[i-1] for i in range(1, len(scores))]
