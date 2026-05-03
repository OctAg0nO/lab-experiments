"""In-memory frontier. Same UCB logic as DaprFrontier, no sidecar."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Dict, Optional

from lab.shared.research import (
    ResearchDirection,
    ResearchFrontier,
    SATURATION_THRESHOLD,
)


class InMemoryFrontier(ResearchFrontier):
    def __init__(self):
        self.directions: Dict[str, ResearchDirection] = {}
        self.total_explorations = 0

    def seed_from_query(self, query: str):
        self.directions[query] = ResearchDirection(
            topic=query,
            confidence=0.0,
            exploration_depth=0,
            seed_query=query,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )

    def seed_from_directions(self, topics: list[str], parent: str | None = None):
        for t in topics:
            if t not in self.directions:
                self.directions[t] = ResearchDirection(
                    topic=t,
                    confidence=0.0,
                    exploration_depth=0,
                    parent_topic=parent,
                    seed_query=t,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                )

    def next_action(self) -> Optional[ResearchDirection]:
        active = [d for d in self.directions.values() if not d.is_saturated(SATURATION_THRESHOLD)]
        return self._next_action_from_directions(active)

    def absorb_findings(
        self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str]
    ):
        d = self.directions.get(topic)
        if d is not None:
            d.confidence = min(1.0, d.confidence + confidence_delta)
            d.exploration_depth += 1
            d.source_count += sources
            d.last_updated = datetime.now(timezone.utc).isoformat()
            self.total_explorations += 1
        for fu in follow_ups:
            if fu not in self.directions:
                self.directions[fu] = ResearchDirection(
                    topic=fu,
                    confidence=0.0,
                    exploration_depth=0,
                    parent_topic=topic,
                    seed_query=fu,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                )

    def saturated(self) -> bool:
        return all(d.is_saturated(SATURATION_THRESHOLD) for d in self.directions.values())

    def summary(self) -> str:
        total = len(self.directions)
        active = self._active_count()
        return f"{active} active, {total - active} explored, {self.total_explorations} total explorations"
