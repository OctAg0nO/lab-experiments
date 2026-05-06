"""
DaprLSEOptimizer — LSE with Dapr-persisted run history.

Extends the DSPy-powered LSEOptimizer with StateStoreService persistence.
Saves on every record_run. Fail-fast if Dapr/Redis is down.
"""

from __future__ import annotations

from dapr_agents.storage.daprstores.stateservice import StateStoreService

from ..evolution.lse import LSEOptimizer, LSERun


class DaprLSEOptimizer(LSEOptimizer):
    """LSE optimizer that persists run history to Dapr state store.

    All DSPy functionality inherited from LSEOptimizer unchanged.
    """

    def __init__(
        self,
        store_name: str = "meta-state",
        key: str = "lse_runs",
    ):
        super().__init__()
        self._store = StateStoreService(store_name=store_name)
        self._key = key
        self._load()

    def _load(self):
        raw = self._store.load(key=self._key)
        if raw:
            data = raw if isinstance(raw, dict) else {}
            for item in data.get("runs", []):
                self.runs.append(LSERun(**item))

    def _save(self):
        self._store.save(
            key=self._key,
            value={
                "runs": [
                    {
                        "strategy_id": r.strategy_id,
                        "quality_score": r.quality_score,
                        "strategy_description": r.strategy_description,
                        "num_directions": r.num_directions,
                        "num_findings": r.num_findings,
                    }
                    for r in self.runs
                ]
            },
        )

    def record_run(self, strategy_id: str, state: dict, strategy_description: str):
        super().record_run(strategy_id, state, strategy_description)
        self._save()

    def update_quality(self, index: int, quality: float):
        if 0 <= index < len(self.runs):
            self.runs[index].quality_score = quality
            self._save()
