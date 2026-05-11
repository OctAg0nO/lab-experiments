"""A2UI transport abstraction — LiveKit data channel or WebSocket.

Allows A2UI messages to be sent over different transports without
changing the A2UIChannel code. Default is LiveKit data channel.
WebSocket transport enables A2UI in non-voice contexts.
"""

from __future__ import annotations

import json
import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class A2UITransport(ABC):
    """Abstract transport for A2UI messages."""

    @abstractmethod
    async def send(self, payload: bytes) -> bool:
        ...

    @abstractmethod
    async def connect(self):
        ...

    @abstractmethod
    async def close(self):
        ...


class LiveKitDataTransport(A2UITransport):
    """A2UI transport over LiveKit data channel."""

    def __init__(self, room, topic: str = "a2ui"):
        self._room = room
        self._topic = topic

    async def connect(self):
        pass  # Room is already connected

    async def close(self):
        pass  # Room lifecycle managed externally

    async def send(self, payload: bytes) -> bool:
        if not self._room or not self._room.local_participant:
            return False
        try:
            self._room.local_participant.publish_data(
                data=payload,
                topic=self._topic,
            )
            return True
        except Exception as e:
            logger.error("LiveKit transport send failed: %s", e)
            return False


class WebSocketTransport(A2UITransport):
    """A2UI transport over WebSocket.

    Usage:
        transport = WebSocketTransport("ws://localhost:8765")
        await transport.connect()
        channel = A2UIChannel(transport=transport)
    """

    def __init__(self, url: str):
        self._url = url
        self._ws = None

    async def connect(self):
        import aiohttp
        self._session = aiohttp.ClientSession()
        self._ws = await self._session.ws_connect(self._url)
        logger.info("WebSocket transport connected: %s", self._url)

    async def close(self):
        if self._ws:
            await self._ws.close()
        if hasattr(self, '_session'):
            await self._session.close()

    async def send(self, payload: bytes) -> bool:
        if not self._ws:
            return False
        try:
            await self._ws.send_bytes(payload)
            return True
        except Exception as e:
            logger.error("WebSocket transport send failed: %s", e)
            return False
