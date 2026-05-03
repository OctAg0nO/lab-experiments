"""
MemoryStore — unified persistence for the research platform.

Coordinates KnowledgeGraph, skill library, execution logs, and frontier state.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from datetime import datetime, timezone

from .knowledge_graph import KnowledgeGraph


class MemoryStore:
    """Unified memory: knowledge graph + skill library + execution logs."""

    def __init__(self, base_dir: str | Path):
        self.base = Path(base_dir)
        self.graph = KnowledgeGraph(self.base / "graph.json")
        self.skills_dir = self.base / "skills"
        self.logs_dir = self.base / "logs"
        self.skills_dir.mkdir(parents=True, exist_ok=True)
        self.logs_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Skills
    # ------------------------------------------------------------------

    def save_skill(self, name: str, data: dict):
        path = self.skills_dir / f"{name}.json"
        path.write_text(json.dumps({**data, "saved": datetime.now(timezone.utc).isoformat()}, indent=2))

    def load_skills(self) -> list[dict]:
        skills = []
        for f in sorted(self.skills_dir.glob("*.json"), reverse=True):
            skills.append(json.loads(f.read_text()))
        return skills

    # ------------------------------------------------------------------
    # Execution logs
    # ------------------------------------------------------------------

    def log_execution(self, agent: str, topic: str, trajectory: list, result: dict | None):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = re.sub(r"[^a-z0-9]+", "_", topic.lower())[:40].strip("_")
        entry = {
            "agent": agent,
            "topic": topic,
            "trajectory": trajectory,
            "result": result,
            "timestamp": timestamp,
        }
        (self.logs_dir / f"{timestamp}_{agent}_{slug}.json").write_text(
            json.dumps(entry, indent=2, default=str)
        )

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        skills = len(list(self.skills_dir.glob("*.json")))
        logs = len(list(self.logs_dir.glob("*.json")))
        return f"{self.graph.summary()}, {skills} skills, {logs} logs"
