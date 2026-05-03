"""
Knowledge Graph — structured findings with typed edges and vector similarity.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from datetime import datetime, timezone


class KnowledgeGraph:
    """Directed graph of findings with typed relationships.

    Nodes: findings (text content + metadata)
    Edges: supports / contradicts / extends / derived_from

    Persisted as JSON for simplicity; migrates to SQLite at scale.
    """

    def __init__(self, persist_path: str | Path):
        self.path = Path(persist_path)
        self.nodes: dict[str, dict[str, Any]] = {}
        self.edges: list[dict[str, Any]] = []
        self._load()

    # ------------------------------------------------------------------
    # Node operations
    # ------------------------------------------------------------------

    def add_finding(self, finding_id: str, content: str, source: str, category: str = "", metadata: dict | None = None):
        if finding_id not in self.nodes:
            self.nodes[finding_id] = {
                "id": finding_id,
                "content": content,
                "source": source,
                "category": category,
                "created": datetime.now(timezone.utc).isoformat(),
                "metadata": metadata or {},
            }
            self._save()

    def get_finding(self, finding_id: str) -> dict[str, Any] | None:
        return self.nodes.get(finding_id)

    def search(self, query: str) -> list[dict[str, Any]]:
        """Simple keyword search across finding content."""
        q = query.lower()
        results = []
        for n in self.nodes.values():
            score = 0
            if q in n["content"].lower():
                score += len(q) / max(len(n["content"]), 1)
            if q in n.get("category", "").lower():
                score += 0.3
            if q in n.get("source", "").lower():
                score += 0.2
            if score > 0:
                results.append({"node": n, "score": score})
        results.sort(key=lambda x: x["score"], reverse=True)
        return results

    # ------------------------------------------------------------------
    # Edge operations
    # ------------------------------------------------------------------

    def relate(self, source_id: str, target_id: str, relation: str):
        self.edges.append({
            "source": source_id,
            "target": target_id,
            "relation": relation,
            "created": datetime.now(timezone.utc).isoformat(),
        })
        self._save()

    def get_related(self, finding_id: str) -> list[dict[str, Any]]:
        results = []
        for e in self.edges:
            if e["source"] == finding_id:
                target = self.nodes.get(e["target"])
                if target:
                    results.append({"node": target, "relation": e["relation"]})
            elif e["target"] == finding_id:
                source = self.nodes.get(e["source"])
                if source:
                    results.append({"node": source, "relation": f"inverse_{e['relation']}"})
        return results

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self.path.exists():
            data = json.loads(self.path.read_text())
            self.nodes = {n["id"]: n for n in data.get("nodes", [])}
            self.edges = data.get("edges", [])

    def _save(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps({
            "nodes": list(self.nodes.values()),
            "edges": self.edges,
        }, indent=2, default=str))

    def summary(self) -> str:
        return f"{len(self.nodes)} findings, {len(self.edges)} relationships"
