"""OctAg0nOAgent — LiveKit Agent that delegates reasoning to MetaAgent.

Uses the llm_node override pattern on LiveKit's Agent class.
Supports Dapr-style durability, streaming per-iteration progress,
thinking step visualization, and A2UI progress updates.
"""

from __future__ import annotations

import asyncio
import logging
from typing import AsyncIterable

from livekit.agents import llm as livekit_llm
from livekit.agents import Agent, ModelSettings

from ..meta.meta_agent import MetaAgent

logger = logging.getLogger(__name__)

_VOICE_MAX_ITERATIONS = 3

_pending_research: dict[str, dict] = {}


def get_pending_results(identity: str) -> dict | None:
    return _pending_research.pop(identity, None)


class OctAg0nOAgent(Agent):
    """LiveKit Agent that delegates reasoning to the OctAg0nO MetaAgent.

    Overrides llm_node to route user speech through the DSPy research
    loop. Streams per-iteration results for progressive TTS output,
    pushes thinking steps and progress to A2UI, and caches results
    for reconnection durability.
    """

    def __init__(
        self,
        meta_agent: MetaAgent,
        max_iterations: int = _VOICE_MAX_ITERATIONS,
        a2ui_channel=None,
        agui_handler=None,
    ):
        super().__init__(
            instructions=(
                "You are OctAg0nO, a durable meta-agent. "
                "You research topics, analyze data, and display results. "
                "Use tools to gather information and A2UI to show UI."
            ),
        )
        self._meta = meta_agent
        self._max_iterations = max_iterations
        self._a2ui = a2ui_channel
        self._agui = agui_handler

    async def llm_node(
        self,
        chat_ctx: livekit_llm.ChatContext,
        tools: list[livekit_llm.Tool],
        model_settings: ModelSettings,
    ) -> AsyncIterable[livekit_llm.ChatChunk | str]:
        """Override: route conversation turn through MetaAgent research loop.

        Streams per-iteration results for progressive TTS:
          Iteration 1 → yield "Analyzing..." → A2UI progress 33%
          Iteration 2 → yield "Found patterns..." → A2UI progress 66%
          Iteration 3 → yield "Synthesizing..." → A2UI progress 100%
        """
        last_text = chat_ctx.last_text_content()
        if not last_text:
            yield "I didn't catch that. Could you repeat?"
            return

        logger.info("OctAg0nOAgent processing: %.100s", last_text)

        # Thinking step — user sees this as a status message
        yield "Let me research that."
        self._push_thinking("Analyzing query...")

        try:
            loop = asyncio.get_event_loop()
            surface = "research"

            # Use the generator run_stack_iter() for per-iteration streaming
            async for chunk in self._stream_research(loop, last_text, surface):
                yield chunk

        except Exception as e:
            logger.error("MetaAgent research failed: %s", e)
            yield f"I encountered an error: {e}"

    async def _stream_research(self, loop, query: str, surface: str):
        """Stream research results per iteration with A2UI progress."""
        text_parts = []
        total = self._max_iterations

        # Create the generator, then pull items one-at-a-time from executor
        # This yields each iteration to the caller (TTS) as it completes,
        # rather than collecting all iterations then iterating.
        gen = self._meta.run_stack_iter(query, self._max_iterations)

        for i in range(total):
            try:
                result = await loop.run_in_executor(None, next, gen)
            except StopIteration:
                break

            iteration, direction, entry, prediction, quality, state = result
            pct = int((i + 1) / total * 100)
            pred_str = str(getattr(prediction, "result", str(prediction)))

            # Push thinking step
            thought = f"Agent {entry.name} exploring {direction.topic[:40]}..."
            self._push_thinking(thought)

            # Update A2UI progress
            if self._a2ui:
                self._a2ui.update_components(surface, [{
                    "id": f"{surface}-progress",
                    "component": "ProgressBar",
                    "props": {
                        "label": f"{thought} ({pct}%)",
                        "value": pct,
                        "max": 100,
                    },
                }])

            # Stream text for TTS
            if pred_str:
                text_parts.append(pred_str)
                yield pred_str + " "

            # Update data model with latest finding
            if self._a2ui:
                self._a2ui.update_data_model(
                    surface, f"/research/iteration_{i}",
                    {"agent": entry.name, "topic": direction.topic, "quality": quality},
                )

        # Final — show complete
        if self._a2ui:
            self._a2ui.update_components(surface, [{
                "id": f"{surface}-progress",
                "component": "ProgressBar",
                "props": {"label": "Research complete", "value": 100, "max": 100},
            }])

        # Cache for reconnection
        try:
            identity = getattr(chat_ctx, "metadata", {}).get("identity", "unknown")
        except Exception:
            identity = "unknown"
        _pending_research[identity] = {
            "query": query,
            "results": text_parts,
        }

    def _push_thinking(self, message: str):
        """Push a thinking/status message to A2UI data model."""
        if not self._a2ui:
            return
        try:
            self._a2ui.update_data_model("research", "/thinking", message)
        except Exception:
            pass
