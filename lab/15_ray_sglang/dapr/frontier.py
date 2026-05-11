"""
DaprFrontier — ResearchFrontier backed by Dapr StateStoreService.

Ported from lab/10_dapr_deep_research/memory/dapr_frontier.py.
Same interface as InMemoryFrontier (same ResearchFrontier ABC) —
swap them without changing any DSPy code.

DSPy is used here for batch saturation assessment (AssessBatchSaturation
signature via ChainOfThought). This does NOT replace any DSPy module —
it adds a persistence layer underneath the existing frontier abstraction.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from typing import Dict, Optional, Set

import dspy
from dapr_agents.storage.daprstores.stateservice import StateStoreService

from ...shared.research import (
    ResearchDirection,
    ResearchFrontier,
    MAX_BOOTSTRAPPED_DEMOS,
    MAX_LABELED_DEMOS,
)


class AssessBatchSaturation(dspy.Signature):
    """Assess saturation for multiple research directions at once."""
    directions_json: str = dspy.InputField(
        desc="JSON array: [{topic, confidence, exploration_depth, source_count}]"
    )
    saturated_indices: list[int] = dspy.OutputField(
        desc="Indices of saturated directions"
    )


class DaprFrontier(ResearchFrontier):
    """ResearchFrontier backed by Dapr StateStoreService (Redis).

    Same UCB-based exploration logic as InMemoryFrontier, but state is
    persisted to Redis. Also adds batch saturation assessment via DSPy
    ChainOfThought with caching.

    Usage:
        frontier = DaprFrontier(store_name="meta-state", key="frontier")
        # Same API as InMemoryFrontier:
        frontier.seed_from_query("my topic")
        direction = frontier.next_action()
        frontier.absorb_findings(topic, 0.3, 1, [])
    """

    def __init__(
        self,
        store_name: str | None = None,
        key: str = "frontier",
    ):
        self._store = StateStoreService(
            store_name=store_name or "meta-state"
        )
        self._key = key
        self.directions: Dict[str, ResearchDirection] = {}
        self.total_explorations = 0
        self._saturation_batch = dspy.ChainOfThought(AssessBatchSaturation)
        self._saturation_cache: Optional[Set[int]] = None
        self._dirty = False
        self._load()

    def _invalidate_saturation_cache(self):
        self._saturation_cache = None

    # -- serialization --

    def _to_snapshot(self) -> list[dict]:
        return [d.to_dict() for d in self.directions.values()]

    @staticmethod
    def _from_snapshot(
        snapshot: list[dict],
    ) -> Dict[str, ResearchDirection]:
        return {d["topic"]: ResearchDirection.from_dict(d) for d in snapshot}

    # -- persistence (fail-fast if Dapr/Redis is down) --

    def _load(self):
        raw = self._store.load(key=self._key)
        if raw:
            data = raw if isinstance(raw, dict) else {}
            self.directions = self._from_snapshot(data.get("directions", []))
            self.total_explorations = data.get("total_explorations", 0)

    def _save(self):
        self._store.save(
            key=self._key,
            value={
                "directions": self._to_snapshot(),
                "total_explorations": self.total_explorations,
            },
        )

    def _flush(self):
        if self._dirty:
            self._save()
            self._dirty = False

    def seed_from_query(self, query: str):
        self.directions[query] = ResearchDirection(
            topic=query,
            confidence=0.0,
            exploration_depth=0,
            seed_query=query,
            last_updated=datetime.now(timezone.utc).isoformat(),
        )
        self._invalidate_saturation_cache()
        self._dirty = True

    def seed_from_directions(
        self, topics: list[str], parent: str | None = None
    ):
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
        self._invalidate_saturation_cache()
        self._dirty = True

    def next_action(self) -> Optional[ResearchDirection]:
        self._flush()
        saturated = self._get_saturated_indices()
        sorted_dirs = list(self.directions.values())
        candidates = [
            d for i, d in enumerate(sorted_dirs) if i not in saturated
        ]
        return self._next_action_from_directions(candidates)

    def absorb_findings(
        self,
        topic: str,
        confidence_delta: float,
        sources: int,
        follow_ups: list[str],
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
        self._invalidate_saturation_cache()
        self._dirty = True

    def saturated(self) -> bool:
        self._flush()
        saturated = self._get_saturated_indices()
        return (
            len(saturated) == len(self.directions)
            if self.directions
            else False
        )

    def summary(self) -> str:
        return (
            f"{self._active_count()} active, "
            f"{len(self.directions) - self._active_count()} explored, "
            f"{self.total_explorations} total explorations"
        )

    # -- saturation (Dapr-specific — DSPy ChainOfThought + cache) --

    def compile(
        self,
        trainset: list[dspy.Example],
        student_lm: dspy.LM | None = None,
    ):
        teacher = self._saturation_batch
        if student_lm:
            student = dspy.ChainOfThought(AssessBatchSaturation)
            student.set_lm(student_lm)
        else:
            student = teacher

        def _saturation_metric(ex, pred, trace=None):
            return hasattr(pred, "saturated_indices")

        bs = dspy.BootstrapFewShot(
            metric=_saturation_metric,
            max_bootstrapped_demos=MAX_BOOTSTRAPPED_DEMOS,
            max_labeled_demos=MAX_LABELED_DEMOS,
        )
        compiled = bs.compile(student, teacher=teacher, trainset=trainset)
        if student_lm:
            compiled.set_lm(student_lm)
        self._saturation_batch = compiled

    def _get_saturated_indices(self) -> Set[int]:
        if self._saturation_cache is not None:
            return self._saturation_cache
        if not self.directions:
            self._saturation_cache = set()
            return self._saturation_cache
        snapshot = [
            {
                "topic": d.topic,
                "confidence": d.confidence,
                "exploration_depth": d.exploration_depth,
                "source_count": d.source_count,
            }
            for d in self.directions.values()
        ]
        pred = self._saturation_batch(
            directions_json=json.dumps(snapshot)
        )
        self._saturation_cache = (
            set(pred.saturated_indices)
            if hasattr(pred, "saturated_indices")
            else set()
        )
        return self._saturation_cache
