"""In-memory frontier. Same UCB logic as DaprFrontier, no sidecar."""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone


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

    def to_dict(self) -> dict:
        return {"topic": self.topic, "confidence": self.confidence, "exploration_depth": self.exploration_depth, "source_count": self.source_count, "last_updated": self.last_updated, "parent_topic": self.parent_topic, "seed_query": self.seed_query}

    @classmethod
    def from_dict(cls, d: dict) -> ResearchDirection:
        return cls(**d)


class InMemoryFrontier:
    def __init__(self):
        self.directions: list[ResearchDirection] = []
        self._total_explorations = 0
        self._active_count = 0

    @property
    def total_explorations(self) -> int:
        return self._total_explorations

    def seed_from_query(self, query: str):
        self.directions.append(ResearchDirection(topic=query, confidence=0.0, exploration_depth=0, seed_query=query, last_updated=datetime.now(timezone.utc).isoformat()))
        self._active_count += 1

    def seed_from_directions(self, topics: list[str], parent: str | None = None):
        for t in topics:
            if not any(d.topic == t for d in self.directions):
                self.directions.append(ResearchDirection(topic=t, confidence=0.0, exploration_depth=0, parent_topic=parent, seed_query=t, last_updated=datetime.now(timezone.utc).isoformat()))
                self._active_count += 1

    def next_action(self) -> ResearchDirection | None:
        active = [d for d in self.directions if d.confidence < 0.95]
        return max(active, key=lambda d: d.ucb_score(self._total_explorations)) if active else None

    def absorb_findings(self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str]):
        for d in self.directions:
            if d.topic == topic:
                d.confidence = min(1.0, d.confidence + confidence_delta)
                d.exploration_depth += 1
                d.source_count += sources
                d.last_updated = datetime.now(timezone.utc).isoformat()
                self._total_explorations += 1
                break
        for fu in follow_ups:
            if not any(d.topic == fu for d in self.directions):
                self.directions.append(ResearchDirection(topic=fu, confidence=0.0, exploration_depth=0, parent_topic=topic, seed_query=fu, last_updated=datetime.now(timezone.utc).isoformat()))
                self._active_count += 1

    def saturated(self) -> bool:
        return all(d.confidence >= 0.95 for d in self.directions)

    def summary(self) -> str:
        return f"{self._active_count} active, {len(self.directions) - self._active_count} explored, {self._total_explorations} total explorations"
