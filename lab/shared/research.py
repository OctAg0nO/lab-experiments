"""Shared research primitives — ResearchDirection dataclass, ResearchFrontier ABC."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional


@dataclass
class ResearchDirection:
    topic: str
    confidence: float = 0.0
    exploration_depth: int = 0
    source_count: int = 0
    last_updated: str = ""
    parent_topic: str | None = None
    seed_query: str = ""

    def ucb_score(self, total_explorations: int, exploration_constant: float = 1.4) -> float:
        if self.exploration_depth == 0:
            return float("inf")
        return self.confidence + exploration_constant * math.sqrt(
            math.log(total_explorations + 1) / (self.exploration_depth + 1)
        )

    def is_saturated(self, threshold: float = 0.95) -> bool:
        return self.confidence >= threshold

    def to_dict(self) -> dict:
        return {
            "topic": self.topic,
            "confidence": self.confidence,
            "exploration_depth": self.exploration_depth,
            "source_count": self.source_count,
            "last_updated": self.last_updated,
            "parent_topic": self.parent_topic,
            "seed_query": self.seed_query,
        }

    @classmethod
    def from_dict(cls, d: dict) -> ResearchDirection:
        return cls(**d)


SATURATION_THRESHOLD = 0.95
MAX_BOOTSTRAPPED_DEMOS = 4
MAX_LABELED_DEMOS = 2


class ResearchFrontier(ABC):
    """Abstract frontier — defines the interface for both in-memory and Dapr variants."""

    directions: Dict[str, ResearchDirection]
    total_explorations: int

    @abstractmethod
    def seed_from_query(self, query: str) -> None: ...

    @abstractmethod
    def seed_from_directions(self, topics: list[str], parent: str | None = None) -> None: ...

    @abstractmethod
    def next_action(self) -> Optional[ResearchDirection]: ...

    @abstractmethod
    def absorb_findings(
        self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str]
    ) -> None: ...

    @abstractmethod
    def saturated(self) -> bool: ...

    @abstractmethod
    def summary(self) -> str: ...

    # -- concrete helpers shared by all implementations --

    def _active_count(self) -> int:
        return sum(1 for d in self.directions.values() if not d.is_saturated(SATURATION_THRESHOLD))

    def _next_action_from_directions(
        self, candidates: list[ResearchDirection]
    ) -> Optional[ResearchDirection]:
        return max(candidates, key=lambda d: d.ucb_score(self.total_explorations)) if candidates else None
