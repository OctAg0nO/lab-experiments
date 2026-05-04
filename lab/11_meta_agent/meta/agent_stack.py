"""AgentStack — push/pop/query registry for dynamically generated agents."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional


@dataclass
class AgentEntry:
    name: str
    role: str
    goal: str
    signature: str
    tools: list[str]
    use_code: bool = False
    prompt_template: str = ""
    created_at: str = ""
    run_count: int = 0
    avg_quality: float = 0.0

    def __post_init__(self):
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()


class AgentStack:
    """Stack of dynamically generated agents.

    Supports push (new agent), pop (remove), query by role/goal,
    and tracking usage statistics for LSE optimization.
    """

    def __init__(self):
        self._entries: list[AgentEntry] = []
        self._by_name: Dict[str, AgentEntry] = {}

    # -- stack operations --

    def push(self, entry: AgentEntry) -> None:
        if entry.name in self._by_name:
            raise ValueError(f"Agent '{entry.name}' already exists on stack")
        self._entries.append(entry)
        self._by_name[entry.name] = entry

    def pop(self) -> Optional[AgentEntry]:
        if not self._entries:
            return None
        entry = self._entries.pop()
        self._by_name.pop(entry.name, None)
        return entry

    def peek(self) -> Optional[AgentEntry]:
        return self._entries[-1] if self._entries else None

    # -- query --

    def get(self, name: str) -> Optional[AgentEntry]:
        return self._by_name.get(name)

    def find(self, role: str | None = None, goal: str | None = None) -> list[AgentEntry]:
        results = list(self._entries)
        if role:
            results = [e for e in results if role.lower() in e.role.lower()]
        if goal:
            results = [e for e in results if goal.lower() in e.goal.lower()]
        return results

    def snapshot(self) -> list[AgentEntry]:
        return list(self._entries)

    def __len__(self) -> int:
        return len(self._entries)

    def __iter__(self):
        return iter(self._entries)

    # -- stats --

    def record_run(self, name: str, quality: float) -> None:
        entry = self._by_name.get(name)
        if entry is None:
            return
        total = entry.avg_quality * entry.run_count + quality
        entry.run_count += 1
        entry.avg_quality = total / entry.run_count

    def summary(self) -> str:
        if not self._entries:
            return "empty stack"
        parts = [f"{len(self._entries)} agent(s):"]
        for e in self._entries:
            parts.append(
                f"  {e.name}: {e.role} | runs={e.run_count} "
                f"| avg_q={e.avg_quality:.2f} | tools={len(e.tools)}"
            )
        return "\n".join(parts)
