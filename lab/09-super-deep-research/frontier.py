"""
ResearchFrontier — autonomous topic discovery with UCB exploration/exploitation.

Replaces the fixed SCRAPE_URLS list with a dynamic priority queue that
selects what to explore next based on confidence, depth, and potential.
"""

from __future__ import annotations

import math
import json
from dataclasses import dataclass
from pathlib import Path
from datetime import datetime, timezone


@dataclass
class ResearchDirection:
    """A single direction for exploration."""
    topic: str
    confidence: float = 0.0          # 0.0 - 1.0 how well understood
    exploration_depth: int = 0       # how many times explored
    source_count: int = 0            # unique sources consulted
    last_updated: str = ""           # ISO timestamp
    parent_topic: str | None = None  # derived from which topic
    seed_query: str = ""             # initial search query used

    @property
    def ucb_score(self, total_explorations: int, exploration_constant: float = 1.4) -> float:
        if self.exploration_depth == 0:
            return float("inf")
        exploitation = self.confidence
        exploration = exploration_constant * math.sqrt(
            math.log(total_explorations + 1) / (self.exploration_depth + 1)
        )
        return exploitation + exploration

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


class ResearchFrontier:
    """Priority queue of research directions with UCB selection.

    Seeds from a user query, then autonomously expands as findings
    are absorbed. Persisted to file across runs.
    """

    def __init__(self, persist_path: str | Path | None = None):
        self.directions: list[ResearchDirection] = []
        self._total_explorations = 0
        self._persist_path = Path(persist_path) if persist_path else None
        self._load()

    # ------------------------------------------------------------------
    # Seeding
    # ------------------------------------------------------------------

    def seed_from_query(self, query: str):
        """Create the initial direction from a user query."""
        self.directions.append(ResearchDirection(
            topic=query,
            confidence=0.0,
            exploration_depth=0,
            seed_query=query,
            last_updated=datetime.now(timezone.utc).isoformat(),
        ))
        self._save()

    def seed_from_directions(self, topics: list[str], parent: str | None = None):
        """Seed multiple sub-directions from a broader topic."""
        for t in topics:
            if not any(d.topic == t for d in self.directions):
                self.directions.append(ResearchDirection(
                    topic=t,
                    confidence=0.0,
                    exploration_depth=0,
                    parent_topic=parent,
                    seed_query=t,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                ))
        self._save()

    # ------------------------------------------------------------------
    # Selection
    # ------------------------------------------------------------------

    def next_action(self) -> ResearchDirection | None:
        """Select highest-UCB direction for exploration."""
        active = [d for d in self.directions if d.confidence < 0.95]
        if not active:
            return None
        best = max(active, key=lambda d: d.ucb_score(self._total_explorations))
        return best

    # ------------------------------------------------------------------
    # Absorption
    # ------------------------------------------------------------------

    def absorb_findings(self, topic: str, confidence_delta: float, sources: int, follow_ups: list[str]):
        """Update a direction with new findings and spawn follow-ups."""
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
                self.directions.append(ResearchDirection(
                    topic=fu,
                    confidence=0.0,
                    exploration_depth=0,
                    parent_topic=topic,
                    seed_query=fu,
                    last_updated=datetime.now(timezone.utc).isoformat(),
                ))
        self._save()

    def saturated(self) -> bool:
        """All active directions have high confidence."""
        active = [d for d in self.directions if d.confidence < 0.95]
        return len(active) == 0

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self._persist_path and self._persist_path.exists():
            data = json.loads(self._persist_path.read_text())
            self.directions = [ResearchDirection.from_dict(d) for d in data.get("directions", [])]
            self._total_explorations = data.get("total_explorations", 0)

    def _save(self):
        if self._persist_path:
            self._persist_path.parent.mkdir(parents=True, exist_ok=True)
            self._persist_path.write_text(json.dumps({
                "directions": [d.to_dict() for d in self.directions],
                "total_explorations": self._total_explorations,
            }, indent=2))

    def summary(self) -> str:
        active = [d for d in self.directions if d.confidence < 0.95]
        explored = [d for d in self.directions if d.confidence >= 0.95]
        return f"{len(active)} active, {len(explored)} explored, {self._total_explorations} total explorations"
