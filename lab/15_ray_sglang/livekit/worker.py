"""LiveKit Agent Worker — voice agent with OctAg0nO brain.

Each worker connects to the LiveKit server and handles voice sessions.
Uses the Agent + AgentSession pattern with llm_node override.
"""

from __future__ import annotations

import logging

from livekit.agents import AgentServer, AgentSession, JobContext, cli, inference
from livekit.plugins import silero

from ..meta.meta_agent import MetaAgent
from ..mcp.a2ui_tool import set_active_channel, clear_active_channel
from .a2ui_channel import A2UIChannel
from .llm_adapter import OctAg0nOAgent, get_pending_results

logger = logging.getLogger(__name__)


def _setup_a2ui(ctx: JobContext) -> A2UIChannel:
    """Set up A2UI surface and check for reconnection results."""
    a2ui = A2UIChannel(ctx.room)
    set_active_channel(a2ui)
    identity = ctx.job.get("participant_identity", "unknown")
    pending = get_pending_results(identity)
    if pending:
        logger.info("Delivering pending results for %s", identity)
        sid = a2ui.create_surface("reconnect")
        a2ui.show_card(
            sid, "reconnect-card",
            title="Research Complete",
            content=f"While you were away, I completed research on: {pending['query']}",
            badge="Reconnected",
            badge_variant="success",
        )
    else:
        a2ui.create_surface("research")
        a2ui.show_loading("research", "OctAg0nO agent ready...")
    return a2ui


def _create_tts(tts_backend, tts_model, tts_voice, qwen3_mode, qwen3_speaker):
    """Create TTS provider: Qwen3-TTS (local) or LiveKit inference (cloud)."""
    if tts_backend == "qwen3":
        from .qwen3_tts import Qwen3TTS
        return Qwen3TTS(mode=qwen3_mode, speaker=qwen3_speaker)
    return inference.TTS(tts_model, voice=tts_voice)


def create_server(
    meta_agent: MetaAgent,
    stt_model: str = "deepgram/nova-3",
    tts_model: str = "cartesia/sonic-3",
    tts_voice: str = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    tts_backend: str = "livekit",
    qwen3_mode: str = "custom_voice",
    qwen3_speaker: str = "Vivian",
) -> AgentServer:
    """Create a LiveKit AgentServer wired to an OctAg0nO MetaAgent.

    Args:
        meta_agent: Initialized MetaAgent (the reasoning brain).
        stt_model: Speech-to-text model identifier.
        tts_model: Text-to-speech model identifier.
        tts_voice: TTS voice ID.

    Returns:
        AgentServer ready to run with cli.run_app().
    """
    server = AgentServer(permissions=AgentServer.config(
        can_publish_data=True,
    ))

    # Pre-warm the MetaAgent so the first voice turn doesn't hit cold-start latency.
    # run_stack with 1 iteration compiles DSPy modules and loads the frontier.
    server.setup_fnc = lambda: meta_agent.run_stack(
        "Pre-warm: initialize DSPy modules and frontier.", 1
    )

    @server.rtc_session()
    async def entrypoint(ctx: JobContext):
        logger.info("LiveKit session started: room=%s", ctx.room.name)
        a2ui = _setup_a2ui(ctx)
        octagono_agent = OctAg0nOAgent(meta_agent)
        tts_provider = _create_tts(tts_backend, tts_model, tts_voice, qwen3_mode, qwen3_speaker)

        session = AgentSession(
            vad=silero.VAD.load(),
            stt=inference.STT(stt_model),
            tts=tts_provider,
        )

        await session.start(agent=octagono_agent, room=ctx.room)
        await session.generate_reply(
            instructions=(
                "Greet the user and explain that OctAg0nO is connected. "
                "You can research topics and show results on their screen."
            )
        )

        await session.aclose()
        clear_active_channel()

    return server


def run_worker(
    meta_agent: MetaAgent,
    stt_model: str = "deepgram/nova-3",
    tts_model: str = "cartesia/sonic-3",
    tts_voice: str = "9626c31c-bec5-4cca-baa8-f8ba9e84c8bc",
    tts_backend: str = "livekit",
    qwen3_mode: str = "custom_voice",
    qwen3_speaker: str = "Vivian",
):
    """Run the LiveKit agent worker in the current process."""
    server = create_server(
        meta_agent, stt_model, tts_model, tts_voice,
        tts_backend=tts_backend, qwen3_mode=qwen3_mode, qwen3_speaker=qwen3_speaker,
    )
    cli.run_app(server)
