"""Human-in-the-loop interrupt system.

Allows the agent to pause and wait for user approval before executing
actions. The interrupt pauses the agent's workflow, renders a confirmation
card via A2UI, and waits for the user's response via AG-UI event.
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

logger = logging.getLogger(__name__)


class InterruptSystem:
    """Manages human-in-the-loop interrupts for the agent.

    Usage:
        interrupts = InterruptSystem(agui_handler, a2ui_channel)

        async def research_flow():
            # Agent wants to do something that needs approval
            approved = await interrupts.confirm(
                "Generate a security patch for this vulnerability?"
            )
            if approved:
                # proceed
            else:
                # skip
    """

    def __init__(self, agui_handler=None, a2ui_channel=None):
        self._agui = agui_handler
        self._a2ui = a2ui_channel
        self._surface_id = "interrupts"

    async def confirm(
        self,
        description: str,
        title: str = "Confirmation Required",
        timeout: float = 300,
    ) -> bool:
        """Ask the user to confirm an action.

        Renders a confirmation card and waits for approve/deny.

        Args:
            description: What the agent wants to do.
            title: Card title.
            timeout: Max seconds to wait (default 5 min).

        Returns:
            True if approved, False if denied or timed out.
        """
        interrupt_id = f"int_{uuid.uuid4().hex[:8]}"

        # Render confirmation UI via A2UI
        if self._a2ui:
            self._a2ui.show_card(
                self._surface_id, f"confirm-{interrupt_id}",
                title=title,
                content=description,
                badge="Awaiting Your Input",
                badge_variant="warning",
            )

        logger.info("Interrupt: %s — awaiting user approval", description[:80])

        # Wait for user response via AG-UI
        if self._agui:
            response = await self._agui.wait_for_interrupt(interrupt_id, timeout=timeout)
            approved = response.get("approved", False)
        else:
            # No AG-UI handler — auto-approve for development
            approved = True

        # Update UI with result
        if self._a2ui:
            self._a2ui.show_status(
                self._surface_id,
                f"Action {'approved' if approved else 'cancelled'}: {description[:60]}",
                variant="success" if approved else "warning",
            )

        return approved

    async def request_input(
        self,
        prompt: str,
        field_type: str = "text",
        timeout: float = 300,
    ) -> str | None:
        """Ask the user for input (text, choice, etc.).

        Args:
            prompt: What to ask the user.
            field_type: Type of input (text, choice, number).
            timeout: Max seconds to wait.

        Returns:
            User's input string, or None on timeout.
        """
        input_id = f"input_{uuid.uuid4().hex[:8]}"

        if self._a2ui:
            self._a2ui.show_card(
                self._surface_id, f"input-{input_id}",
                title="Input Required",
                content=prompt,
                badge="Awaiting Input",
                badge_variant="info",
            )

        if self._agui:
            response = await self._agui.wait_for_interrupt(input_id, timeout=timeout)
            return response.get("value")
        return None
