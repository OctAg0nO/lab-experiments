"""DaprFrontier — ResearchFrontier backed by Dapr StateStoreService."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone

import dspy
from dapr_agents.storage.daprstores.stateservice import StateStoreService


class AssessBatchSaturation(dspy.Signature):
    """Assess saturation for multiple research directions at once."""
    directions_json: str = dspy.InputField(desc="JSON array: [{topic, confidence, exploration_depth, source_count}]")
    saturated_indices: list[int] = dspy.OutputField(desc="Indices of saturated directions")


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


class DaprFrontier:
    def __init__(self, store_name: str = "research-state", key: str = "frontier"):
        self._store = StateStoreService(store_name=store_name)
        self._key = key
        self.directions: list[ResearchDirection] = []
        self._total_explorations = 0
        self._active_count = 0
        self._saturation_batch = dspy.ChainOfThought(AssessBatchSaturation)
        self._load()

    def compile(self, trainset: list[dspy.Example], student_lm: dspy.LM | None = None):
        teacher = self._saturation_batch
        student = dspy.ChainOfThought(AssessBatchSaturation) if student_lm else teacher
        if student_lm:
            student.set_lm(student_lm)
        bs = dspy.BootstrapFewShot(metric=lambda _ex, pred, _trace: hasattr(pred, "saturated_indices"), max_bootstrapped_demos=4, max_labeled_demos=2)
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._saturation_batch = compiled

    def _load(self):
        raw = self._store.load(key=self._key)
        if raw:
            data = raw if isinstance(raw, dict) else {}
            self.directions = [ResearchDirection.from_dict(d) for d in data.get("directions", [])]
            self._total_explorations = data.get("total_explorations", 0)

    def _save(self):
        self._store.save(key=self._key, value={"directions": [d.to_dict() for d in self.directions], "total_explorations": self._total_explorations})

    @property
    def total_explorations(self) -> int:
        return self._total_explorations

    def seed_from_query(self, query: str):
        self.directions.append(ResearchDirection(topic=query, confidence=0.0, exploration_depth=0, seed_query=query, last_updated=datetime.now(timezone.utc).isoformat()))
        self._active_count += 1
        self._save()

    def seed_from_directions(self, topics: list[str], parent: str | None = None):
        for t in topics:
            if not any(d.topic == t for d in self.directions):
                self.directions.append(ResearchDirection(topic=t, confidence=0.0, exploration_depth=0, parent_topic=parent, seed_query=t, last_updated=datetime.now(timezone.utc).isoformat()))
                self._active_count += 1
        self._save()

    def _saturated_indices(self) -> set[int]:
        if not self.directions:
            return set()
        snapshot = [{"topic": d.topic, "confidence": d.confidence, "exploration_depth": d.exploration_depth, "source_count": d.source_count} for d in self.directions]
        pred = self._saturation_batch(directions_json=json.dumps(snapshot))
        return set(pred.saturated_indices) if hasattr(pred, "saturated_indices") else set()

    def next_action(self) -> ResearchDirection | None:
        saturated = self._saturated_indices()
        candidates = [d for i, d in enumerate(self.directions) if i not in saturated]
        return max(candidates, key=lambda d: d.ucb_score(self._total_explorations)) if candidates else None

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
        self._save()

    def saturated(self) -> bool:
        return len(self._saturated_indices()) == len(self.directions) if self.directions else False

    def summary(self) -> str:
        return f"{self._active_count} active, {len(self.directions) - self._active_count} explored, {self._total_explorations} total explorations"
