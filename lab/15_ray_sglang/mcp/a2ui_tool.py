"""A2UI rendering MCP tools — rich component templates + progressive rendering.

Use from any agent tool chain:
    a2ui_surface_create(surface_id)    — Create a UI surface
    a2ui_show_results(title, headers, rows) — Render data table
    a2ui_show_card(title, content)      — Render info card
    a2ui_show_progress(message)         — Show loading state
    a2ui_update_data(path, value)      — Update data model
    a2ui_clear()                        — Remove all surfaces

Each call returns a status string the agent can reference in its response.
"""

from __future__ import annotations

import contextvars
import logging
from typing import Any

logger = logging.getLogger(__name__)

_active_channel: contextvars.ContextVar = contextvars.ContextVar("a2ui_channel", default=None)
_active_surface: contextvars.ContextVar = contextvars.ContextVar("a2ui_surface", default=None)


def set_active_channel(channel: Any):
    _active_channel.set(channel)


def clear_active_channel():
    _active_channel.set(None)
    _active_surface.set(None)


# ── Surface Management ──────────────────────────────────────────────


def a2ui_surface_create(surface_id: str = "main") -> str:
    """Create a new UI surface. Call this before rendering any components.
    Multiple surfaces allow organizing different content areas.
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    channel.create_surface(surface_id)
    _active_surface.set(surface_id)
    return f"Created UI surface '{surface_id}'."


def a2ui_surface_delete(surface_id: str = "main") -> str:
    """Remove a UI surface and all its components."""
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    channel.delete_surface(surface_id)
    return f"Removed UI surface '{surface_id}'."


def a2ui_clear() -> str:
    """Remove all UI surfaces and reset the display."""
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    channel.clear_all()
    return "Cleared all UI surfaces."


# ── Progress / Loading ─────────────────────────────────────────────


def a2ui_show_progress(message: str = "Researching...") -> str:
    """Show a loading/progress indicator on the user's screen.
    Use this at the start of any research operation so the user
    knows work is happening.
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.show_loading(surface, message)
    return f"Showing progress: {message}"


def a2ui_show_status(message: str, variant: str = "info") -> str:
    """Show a status banner. Variants: success, warning, error, info."""
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.show_status(surface, message, variant)
    return f"Status: {message}"


# ── Results Display ─────────────────────────────────────────────────


def a2ui_show_results(
    title: str,
    headers: list[str],
    rows: list[list],
    caption: str | None = None,
) -> str:
    """Render a formatted data table on the user's screen.

    Best for displaying structured research results, comparisons,
    analysis outputs, or any tabular data.

    Args:
        title: Table heading (e.g. "Research Findings").
        headers: Column names (e.g. ["Paper", "Relevance", "Year"]).
        rows: Row data matching headers (e.g. [["Paper A", 0.95, 2024]]).
        caption: Optional description below the table.

    Returns:
        Status message.
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.show_results(surface, title, headers, rows, caption)
    return f"Rendered table '{title}' with {len(rows)} rows."


def a2ui_show_card(
    title: str,
    content: str,
    badge: str | None = None,
    badge_variant: str = "info",
    card_id: str = "card",
) -> str:
    """Render a card component with optional status badge.

    Use for summaries, key findings, notifications, or highlighted content.

    Args:
        title: Card heading.
        content: Card body text (markdown supported).
        badge: Optional badge label (e.g. "Completed", "New").
        badge_variant: Badge color: info, success, warning, error.

    Returns:
        Status message.
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.show_card(surface, card_id, title, content, badge, badge_variant)
    return f"Rendered card '{title}'."


# ── Data Model ──────────────────────────────────────────────────────


def a2ui_update_data(path: str, value: Any) -> str:
    """Update a value in the UI's data model at a JSON Pointer path.

    Components bound to this path will reactively update.
    Useful for progressive updates as research completes.

    Example:
        a2ui_update_data("/research/status", "phase 2 complete")
        a2ui_update_data("/research/findings/count", 42)
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.update_data_model(surface, path, value)
    return f"Updated data at '{path}'."


def a2ui_update_progress(percent: int, message: str = "") -> str:
    """Update a progress indicator to show a specific percentage.

    Replaces the indeterminate progress bar with a determinate one.
    Call this as research iterations complete.

    Args:
        percent: Completion percentage (0-100).
        message: Current status message.
    """
    channel = _active_channel.get()
    if channel is None:
        return "Error: No active A2UI channel."
    surface = _active_surface.get() or "main"
    channel.update_components(surface, [{
        "id": f"{surface}-progress",
        "component": "ProgressBar",
        "props": {"label": message or f"{percent}% complete",
                   "value": percent, "max": 100},
    }])
    return f"Progress updated to {percent}%."
