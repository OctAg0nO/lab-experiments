"""AG-UI event handler — shared state, frontend tool routing, interrupt handling.

AG-UI is the Agent-User Interaction Protocol. It provides:
- Shared state sync (agent ↔ frontend)
- Frontend tool calls (browser APIs as agent tools)
- Interrupts (human-in-the-loop)
- Streaming chat events

This module implements the agent-side handlers.
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any

from ..mcp.a2ui_tool import set_active_channel

logger = logging.getLogger(__name__)


class AGUIEvent:
    """An AG-UI event received from the frontend."""
    def __init__(self, event_type: str, data: dict[str, Any]):
        self.type = event_type
        self.data = data


class AGUIHandler:
    """Handles AG-UI events from the frontend.

    Registered on a LiveKit room. Forwards state updates, tool calls,
    and interrupt responses to the appropriate handlers.
    """

    def __init__(self, room):
        self._room = room
        self._state: dict[str, Any] = {}
        self._state_listeners: list[callable] = []
        self._interrupt_futures: dict[str, asyncio.Future] = {}

        # Register data channel listener for AG-UI events
        @room.on("data_received")
        def on_data(packet):
            self._handle_packet(packet)

    def _handle_packet(self, packet):
        """Parse incoming data channel messages as AG-UI events."""
        try:
            data = json.loads(packet.data)
            event_type = data.get("type", "")
            payload = data.get("data", {})

            if event_type == "shared_state_update":
                self._handle_state_update(payload.get("path", ""), payload.get("value"))
            elif event_type == "frontend_tool_result":
                self._handle_tool_result(payload.get("tool_call_id", ""), payload.get("result"))
            elif event_type == "interrupt_response":
                self._handle_interrupt_response(
                    payload.get("interrupt_id", ""), payload.get("response")
                )
        except (json.JSONDecodeError, AttributeError):
            pass

    def _handle_state_update(self, path: str, value: Any):
        """Update shared state and notify listeners."""
        if path:
            parts = path.strip("/").split("/")
            target = self._state
            for p in parts[:-1]:
                target = target.setdefault(p, {})
            target[parts[-1]] = value
        else:
            self._state = value if isinstance(value, dict) else {}

        for listener in self._state_listeners:
            try:
                listener(path, value)
            except Exception:
                pass

    def _handle_tool_result(self, tool_call_id: str, result: Any):
        """Receive result of a frontend tool execution."""
        logger.info("Frontend tool result: %s = %s", tool_call_id, result)

    def _handle_interrupt_response(self, interrupt_id: str, response: Any):
        """Resolve a pending interrupt future."""
        future = self._interrupt_futures.pop(interrupt_id, None)
        if future and not future.done():
            future.set_result(response)

    def on_state_change(self, callback: callable):
        """Register a listener for shared state changes."""
        self._state_listeners.append(callback)

    @property
    def shared_state(self) -> dict:
        return dict(self._state)

    async def wait_for_interrupt(self, interrupt_id: str, timeout: float = 300) -> Any:
        """Wait for a human response to an interrupt.

        The agent calls this to pause and wait for user approval.
        The frontend shows a confirmation UI and sends the response
        back as an AG-UI event.

        Args:
            interrupt_id: Unique identifier for this interrupt.
            timeout: Max seconds to wait (default 300s / 5 min).

        Returns:
            The user's response (e.g. {"approved": True}).
        """
        future = asyncio.Future()
        self._interrupt_futures[interrupt_id] = future
        try:
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError:
            return {"timeout": True, "approved": False}
