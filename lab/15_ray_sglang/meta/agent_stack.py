"""AgentStack — push/pop/query registry for dynamically generated agents.

DaprAgentStack extends AgentStack with Dapr StateStoreService persistence.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Optional

from dapr_agents.storage.daprstores.stateservice import StateStoreService


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
    failure_count: int = 0

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

    def record_failure(self, name: str) -> None:
        entry = self._by_name.get(name)
        if entry is None:
            return
        entry.failure_count += 1

    def summary(self) -> str:
        if not self._entries:
            return "empty stack"
        parts = [f"{len(self._entries)} agent(s):"]
        for e in self._entries:
            parts.append(
                f"  {e.name}: {e.role} | runs={e.run_count} "
                f"| avg_q={e.avg_quality:.2f} | fails={e.failure_count} "
                f"| tools={len(e.tools)}"
            )
        return "\n".join(parts)


class DaprAgentStack(AgentStack):
    """AgentStack with per-entry state store keys (delta-updates).

    Each agent entry is stored under its own key: {base_key}:entries:{name}.
    Metadata (entry list order) is stored under {base_key}:meta.
    push() is O(1) — only writes the new entry. Full-state saves
    (pop, record_run, record_failure) use O(N) writes but happen rarely.
    """

    _entry_fields = [
        "name", "role", "goal", "signature", "tools", "use_code",
        "prompt_template", "created_at", "run_count", "avg_quality",
        "failure_count",
    ]

    def __init__(self, store_name: str = "meta-state", key: str = "agent_stack"):
        self._store = StateStoreService(store_name=store_name)
        self._key = key
        self._entries_key = f"{key}:entries"
        self._meta_key = f"{key}:meta"
        super().__init__()
        self._load()

    def _entry_to_dict(self, e: AgentEntry) -> dict:
        return {f: getattr(e, f) for f in self._entry_fields}

    @staticmethod
    def _dict_to_entry(d: dict) -> AgentEntry:
        return AgentEntry(
            name=d.get("name", ""),
            role=d.get("role", ""),
            goal=d.get("goal", ""),
            signature=d.get("signature", "task -> result"),
            tools=d.get("tools", []),
            use_code=d.get("use_code", False),
            prompt_template=d.get("prompt_template", ""),
            created_at=d.get("created_at", ""),
            run_count=d.get("run_count", 0),
            avg_quality=d.get("avg_quality", 0.0),
            failure_count=d.get("failure_count", 0),
        )

    def _load(self):
        meta_raw = self._store.load(key=self._meta_key)
        if meta_raw:
            names = meta_raw if isinstance(meta_raw, list) else meta_raw.get("names", [])
            for name in names:
                raw = self._store.load(key=f"{self._entries_key}:{name}")
                if raw:
                    entry = self._dict_to_entry(raw if isinstance(raw, dict) else {})
                    self._entries.append(entry)
                    self._by_name[entry.name] = entry

    def _save_meta(self):
        self._store.save(
            key=self._meta_key,
            value={"names": [e.name for e in self._entries]},
        )

    def push(self, entry: AgentEntry) -> None:
        super().push(entry)
        self._store.save(key=f"{self._entries_key}:{entry.name}", value=self._entry_to_dict(entry))
        self._save_meta()

    def pop(self) -> Optional[AgentEntry]:
        entry = super().pop()
        if entry:
            self._store.save(key=self._meta_key, value={"names": [e.name for e in self._entries]})
        return entry

    def record_run(self, name: str, quality: float) -> None:
        super().record_run(name, quality)
        entry = self._by_name.get(name)
        if entry:
            self._store.save(key=f"{self._entries_key}:{name}", value=self._entry_to_dict(entry))

    def record_failure(self, name: str) -> None:
        super().record_failure(name)
        entry = self._by_name.get(name)
        if entry:
            self._store.save(key=f"{self._entries_key}:{name}", value=self._entry_to_dict(entry))
