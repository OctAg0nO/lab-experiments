from __future__ import annotations

from datetime import datetime, timezone
from typing import Literal
from uuid import uuid4

from pydantic import BaseModel


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _new_id() -> str:
    return uuid4().hex[:12]


class SwarmEnvelope(BaseModel):
    """Universal message envelope for all swarm communication."""
    msg_id: str = ""
    source: str = ""
    target: str = ""
    timestamp: str = ""
    correlation_id: str = ""
    msg_type: str = ""


class SwarmTask(BaseModel):
    """A research direction dispatched to a worker agent."""
    direction: str
    exploration_depth: int = 0
    max_iterations: int = 5
    parent_topic: str = ""
    correlation_id: str = ""

    def to_envelope(self, source: str = "coordinator") -> SwarmEnvelope:
        return SwarmEnvelope(
            msg_id=_new_id(),
            source=source,
            target="",
            timestamp=_now(),
            correlation_id=self.correlation_id or _new_id(),
            msg_type="task",
        )


class SwarmDiscovery(BaseModel):
    """Findings published by a worker after executing a task."""
    topic: str
    findings: list[dict]
    confidence_delta: float = 0.0
    quality_score: float = 0.0
    follow_up_directions: list[str] = []
    worker_id: str = ""
    correlation_id: str = ""

    def to_envelope(self) -> SwarmEnvelope:
        return SwarmEnvelope(
            msg_id=_new_id(),
            source=self.worker_id,
            target="coordinator",
            timestamp=_now(),
            correlation_id=self.correlation_id,
            msg_type="discovery",
        )


class SwarmHeartbeat(BaseModel):
    """Liveness signal from a worker agent."""
    agent_id: str
    status: Literal["alive", "busy", "error"] = "alive"
    load: int = 0
    completed_tasks: int = 0
    failed_tasks: int = 0
    domain: str = ""

    def to_envelope(self) -> SwarmEnvelope:
        return SwarmEnvelope(
            msg_id=_new_id(),
            source=self.agent_id,
            target="coordinator",
            timestamp=_now(),
            msg_type="heartbeat",
        )


class SwarmRegistration(BaseModel):
    """Agent capability registration payload."""
    agent_id: str
    domain: str = ""
    port: int = 0
    status: Literal["available", "busy", "offline"] = "available"


class SwarmInquiry(BaseModel):
    """One agent asking another a question (A2A)."""
    question: str
    context: str = ""
    source_agent: str = ""
    target_agent: str = ""
    correlation_id: str = ""

    def to_envelope(self) -> SwarmEnvelope:
        return SwarmEnvelope(
            msg_id=_new_id(),
            source=self.source_agent,
            target=self.target_agent,
            timestamp=_now(),
            correlation_id=self.correlation_id,
            msg_type="inquiry",
        )


class SwarmResponse(BaseModel):
    """Response to an A2A inquiry."""
    answer: str
    source_agent: str = ""
    target_agent: str = ""
    correlation_id: str = ""

    def to_envelope(self) -> SwarmEnvelope:
        return SwarmEnvelope(
            msg_id=_new_id(),
            source=self.source_agent,
            target=self.target_agent,
            timestamp=_now(),
            correlation_id=self.correlation_id,
            msg_type="response",
        )
