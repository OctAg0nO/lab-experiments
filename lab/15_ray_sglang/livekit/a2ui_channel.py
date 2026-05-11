"""A2UIChannel — A2UI v0.10 protocol over LiveKit data channel.

A2UI (Agent-to-User Interface) protocol v0.10 lets agents send declarative
JSON describing UI components. The client renders them using its own trusted
component catalog.

Protocol messages (v0.10):
  createSurface   — Initialize a new UI surface
  updateComponents — Add/update components in a surface
  updateDataModel  — Patch surface data at a JSON Pointer path
  deleteSurface    — Remove a surface

Transport: LiveKit WebRTC data channel (ordered, low-latency).
"""

from __future__ import annotations

import json
import logging
from typing import Any

logger = logging.getLogger(__name__)


class A2UIChannel:
    """A2UI v0.10 protocol over LiveKit data channel.

    Manages multiple UI surfaces with state tracking and theming.
    Each surface is an independent UI region for different interaction phases.

    Usage:
        channel = A2UIChannel(room, theme={"primaryColor": "#00BFFF"})
        sid = channel.create_surface("results", "basic")
        channel.show_results(sid, "Analysis", ["Col"], [["val"]])
        channel.update_data_model(sid, "/status", "complete")
        channel.delete_surface(sid)
    """

    DEFAULT_THEME = {
        "primaryColor": "#2563EB",
        "secondaryColor": "#7C3AED",
        "backgroundColor": "#0F172A",
        "surfaceColor": "#1E293B",
        "textColor": "#F1F5F9",
        "textSecondary": "#94A3B8",
        "borderRadius": 12,
    }

    def __init__(
        self,
        room,
        label: str = "a2ui",
        theme: dict[str, Any] | None = None,
        catalog_id: str = "https://a2ui.org/specification/v0_10/basic_catalog.json",
    ):
        self._room = room
        self._label = label
        self._theme = {**self.DEFAULT_THEME, **(theme or {})}
        self._catalog_id = catalog_id
        self._surfaces: set[str] = set()

    def create_surface(
        self,
        surface_id: str,
        catalog_id: str | None = None,
        theme: dict[str, Any] | None = None,
        send_data_model: bool = True,
    ) -> str:
        """Create a new UI surface. Returns surface_id for chaining."""
        self._send({
            "createSurface": {
                "surfaceId": surface_id,
                "catalogId": catalog_id or self._catalog_id,
                "theme": theme or self._theme,
                "sendDataModel": send_data_model,
            },
        })
        self._surfaces.add(surface_id)
        return surface_id

    def update_components(self, surface_id: str, components: list[dict]) -> bool:
        """Add or update components in a surface. Flat adjacency list."""
        return self._send({
            "updateComponents": {"surfaceId": surface_id, "components": components},
        })

    def update_data_model(self, surface_id: str, path: str, value: Any) -> bool:
        """Patch the surface data model at a JSON Pointer path. Upsert semantics."""
        return self._send({
            "updateDataModel": {"surfaceId": surface_id, "path": path, "value": value},
        })

    def delete_surface(self, surface_id: str) -> bool:
        """Remove a surface and all its components/data."""
        self._surfaces.discard(surface_id)
        return self._send({"deleteSurface": {"surfaceId": surface_id}})

    def clear_all(self):
        """Remove all active surfaces."""
        for sid in list(self._surfaces):
            self.delete_surface(sid)

    def show_loading(self, surface_id: str, message: str = "Researching..."):
        """Show an indeterminate progress bar while working."""
        self.update_components(surface_id, [{
            "id": f"{surface_id}-progress",
            "component": "ProgressBar",
            "props": {"label": message, "indeterminate": True},
        }])

    def show_results(
        self, surface_id: str, title: str,
        headers: list[str], rows: list[list],
        caption: str | None = None,
    ):
        """Render a styled data table."""
        comps = [
            {"id": f"{surface_id}-title", "component": "Text",
             "props": {"content": f"## {title}"}},
            {"id": f"{surface_id}-table", "component": "DataTable",
             "props": {"columns": headers, "rows": rows, "striped": True}},
        ]
        if caption:
            comps.append({"id": f"{surface_id}-caption", "component": "Text",
                         "props": {"content": caption, "variant": "caption"}})
        self.update_components(surface_id, comps)

    def show_card(self, surface_id: str, card_id: str, title: str, content: str,
                  badge: str | None = None, badge_variant: str = "info"):
        """Render a card with optional status badge."""
        comps = [{"id": card_id, "component": "Card",
                  "props": {"title": title, "content": content, "variant": "elevated"}}]
        if badge:
            comps.append({"id": f"{card_id}-badge", "component": "Badge",
                         "props": {"label": badge, "variant": badge_variant}})
        self.update_components(surface_id, comps)

    def show_status(self, surface_id: str, message: str, variant: str = "info"):
        """Show a status banner with variant: success, warning, error, info."""
        self.update_components(surface_id, [{
            "id": f"{surface_id}-status", "component": "Banner",
            "props": {"content": message, "variant": variant, "dismissible": True},
        }])

    def show_error(self, surface_id: str, message: str):
        """Shortcut for error status."""
        self.show_status(surface_id, f"⚠ {message}", variant="error")

    def _send(self, message: dict) -> bool:
        if not self._room or not self._room.local_participant:
            return False
        payload = {"version": "v0.10", **message}
        try:
            self._room.local_participant.publish_data(
                data=json.dumps(payload).encode("utf-8"), topic=self._label,
            )
            return True
        except Exception as e:
            logger.error("A2UI send failed: %s", e)
            return False
