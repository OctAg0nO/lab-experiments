"""Frontend tool call definitions — browser APIs as agent tools.

These tool definitions are registered in the AgentGenerator alongside
MCP tools. When the agent calls them, AG-UI forwards the request to
the frontend, which executes the browser API and returns the result.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def get_user_location() -> dict:
    """Get the user's current geographic location.

    Uses the browser's Geolocation API.
    Returns {"latitude": float, "longitude": float, "accuracy": float}.

    The agent calls this tool when it needs location-aware results
    (e.g., "find restaurants near me", "weather in my area").
    """
    # AG-UI forwards this to the frontend
    # The frontend calls navigator.geolocation.getCurrentPosition()
    # and returns the result via AG-UI event
    return {"status": "pending", "tool": "get_user_location"}


def pick_file(accept: str = "*/*") -> dict:
    """Let the user pick a file from their device.

    Args:
        accept: MIME type filter (e.g. "image/*", ".pdf", "*/*").

    Returns:
        {"name": str, "type": str, "size": int, "data": str (base64)}.

    The agent calls this when it needs the user to upload a file
    for analysis (e.g., "analyze this spreadsheet").
    """
    return {"status": "pending", "tool": "pick_file", "accept": accept}


def read_clipboard() -> dict:
    """Read text from the user's clipboard.

    Returns {"text": str}.

    Useful for "copy that result to clipboard" or "paste what you have".
    """
    return {"status": "pending", "tool": "read_clipboard"}


def write_clipboard(text: str) -> dict:
    """Write text to the user's clipboard.

    Args:
        text: The text to copy.

    Returns {"status": "copied"}.
    """
    return {"status": "pending", "tool": "write_clipboard", "text": text}
