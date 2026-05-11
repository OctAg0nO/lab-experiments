"""Qwen3-TTS LiveKit plugin — voice cloning, voice design, custom voice presets.

Integrates Qwen3-TTS (by Qwen/Alibaba) as a LiveKit TTS provider.
Supports three modes:
  - CustomVoice: 9 preset speakers with natural-language instruction control
  - VoiceDesign: Generate voice from natural language description
  - VoiceClone: Clone a voice from a 3-second reference audio sample

Usage:
    tts = Qwen3TTS(model_size="1.7B", mode="custom_voice", speaker="Vivian")
    tts = Qwen3TTS(mode="voice_design")
    tts = Qwen3TTS(mode="voice_clone", ref_audio="speaker.wav", ref_text="transcript")
"""

from __future__ import annotations

import asyncio
import logging
import os
from typing import AsyncIterable

import numpy as np

from livekit.agents import tts

logger = logging.getLogger(__name__)


class Qwen3TTS(tts.TTS):
    """LiveKit TTS plugin using Qwen3-TTS.

    Args:
        model_size: "0.6B" or "1.7B" (default "1.7B" for best quality).
        mode: "custom_voice", "voice_design", or "voice_clone".
        speaker: Speaker name for custom_voice mode (e.g. "Vivian", "Ryan", "Serena").
        language: Language override (default "auto").
        instruct: Instruction for voice control (tone, emotion, speed).
        ref_audio: Reference audio file path/URL for voice clone mode.
        ref_text: Transcript of reference audio for voice clone mode.
        device: "cuda:0", "cpu", etc.
        dtype: "bfloat16" (default) or "float16".
    """

    def __init__(
        self,
        model_size: str = "1.7B",
        mode: str = "custom_voice",
        speaker: str = "Vivian",
        language: str = "Auto",
        instruct: str | None = None,
        ref_audio: str | None = None,
        ref_text: str | None = None,
        device: str = "cuda:0",
        dtype: str = "bfloat16",
    ):
        super().__init__(
            sample_rate=24000,
            num_channels=1,
        )
        self._model_size = model_size
        self._mode = mode
        self._speaker = speaker
        self._language = language
        self._instruct = instruct
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._device = device
        self._dtype = dtype
        self._model = None
        self._clone_prompt = None

    def _ensure_model(self):
        """Lazy-load the Qwen3TTS model."""
        if self._model is not None:
            return

        import torch
        from qwen_tts import Qwen3TTSModel

        torch_dtype = torch.bfloat16 if self._dtype == "bfloat16" else torch.float16

        if self._mode == "custom_voice":
            model_id = f"Qwen/Qwen3-TTS-12Hz-{self._model_size}-CustomVoice"
        elif self._mode == "voice_design":
            model_id = f"Qwen/Qwen3-TTS-12Hz-{self._model_size}-VoiceDesign"
        elif self._mode == "voice_clone":
            model_id = f"Qwen/Qwen3-TTS-12Hz-{self._model_size}-Base"
        else:
            raise ValueError(f"Unknown mode: {self._mode}")

        logger.info("Loading Qwen3-TTS model: %s", model_id)
        self._model = Qwen3TTSModel.from_pretrained(
            model_id,
            device_map=self._device,
            dtype=torch_dtype,
            attn_implementation="flash_attention_2",
        )

        # Build voice clone prompt if in clone mode with reference audio
        if self._mode == "voice_clone" and self._ref_audio and self._ref_text:
            self._clone_prompt = self._model.create_voice_clone_prompt(
                ref_audio=self._ref_audio,
                ref_text=self._ref_text,
            )

    def synthesize(
        self,
        text: str,
        *,
        language: str | None = None,
        voice: str | None = None,
        **kwargs,
    ) -> "Qwen3TTSStream":
        """Synthesize text to speech using Qwen3-TTS.

        Returns a Qwen3TTSStream that yields audio chunks.
        """
        self._ensure_model()
        return Qwen3TTSStream(
            self._model,
            text=text,
            mode=self._mode,
            speaker=voice or self._speaker,
            language=language or self._language,
            instruct=self._instruct,
            clone_prompt=self._clone_prompt,
            ref_audio=self._ref_audio,
            ref_text=self._ref_text,
            sample_rate=self.sample_rate,
        )


class Qwen3TTSStream(tts.ChunkedStream):
    """Stream that generates audio chunks from Qwen3-TTS."""

    def __init__(
        self,
        model,
        *,
        text: str,
        mode: str,
        speaker: str,
        language: str,
        instruct: str | None,
        clone_prompt,
        ref_audio: str | None,
        ref_text: str | None,
        sample_rate: int,
    ):
        super().__init__()
        self._model = model
        self._text = text
        self._mode = mode
        self._speaker = speaker
        self._language = language
        self._instruct = instruct
        self._clone_prompt = clone_prompt
        self._ref_audio = ref_audio
        self._ref_text = ref_text
        self._sample_rate = sample_rate

    async def _run(self) -> AsyncIterable[tts.AudioChunk]:
        """Run TTS inference in executor and yield audio chunks."""
        loop = asyncio.get_event_loop()

        try:
            wavs, sr = await loop.run_in_executor(
                None,
                self._synthesize_sync,
            )

            if not wavs:
                logger.warning("Qwen3-TTS returned no audio")
                return

            audio_data = wavs[0]

            # Convert to float32 if needed
            if audio_data.dtype != np.float32:
                audio_data = audio_data.astype(np.float32)

            # Yield as a single chunk (full audio)
            # For streaming, split into smaller frames
            frame_duration = 0.1  # 100ms frames
            frame_samples = int(self._sample_rate * frame_duration)
            total_samples = len(audio_data)

            for start in range(0, total_samples, frame_samples):
                end = min(start + frame_samples, total_samples)
                chunk = audio_data[start:end]

                yield tts.AudioChunk(
                    data=chunk.tobytes(),
                    sample_rate=self._sample_rate,
                    num_channels=1,
                    samples_per_channel=len(chunk),
                )

        except Exception as e:
            logger.error("Qwen3-TTS synthesis failed: %s", e)
            raise

    def _synthesize_sync(self):
        """Run Qwen3-TTS model inference (blocking, runs in executor)."""
        if self._mode == "custom_voice":
            return self._model.generate_custom_voice(
                text=self._text,
                language=self._language,
                speaker=self._speaker,
                instruct=self._instruct,
            )
        elif self._mode == "voice_design":
            return self._model.generate_voice_design(
                text=self._text,
                language=self._language,
                instruct=self._instruct or "Natural, clear speaking voice.",
            )
        elif self._mode == "voice_clone":
            kwargs = {"language": self._language}
            if self._clone_prompt is not None:
                kwargs["voice_clone_prompt"] = self._clone_prompt
            else:
                kwargs["ref_audio"] = self._ref_audio
                kwargs["ref_text"] = self._ref_text
            return self._model.generate_voice_clone(
                text=self._text,
                **kwargs,
            )
        raise ValueError(f"Unknown mode: {self._mode}")


# ── Preset Convenience Constructors ─────────────────────────────────


def qwen3_default_tts(**kwargs) -> Qwen3TTS:
    """Qwen3-TTS with Vivian (bright Chinese female voice)."""
    return Qwen3TTS(mode="custom_voice", speaker="Vivian", **kwargs)


def qwen3_english_tts(**kwargs) -> Qwen3TTS:
    """Qwen3-TTS with Ryan (dynamic English male voice)."""
    return Qwen3TTS(mode="custom_voice", speaker="Ryan", language="English", **kwargs)


def qwen3_voice_design_tts(**kwargs) -> Qwen3TTS:
    """Qwen3-TTS in voice design mode — describe the voice you want."""
    return Qwen3TTS(mode="voice_design", **kwargs)


def qwen3_voice_clone_tts(ref_audio: str, ref_text: str, **kwargs) -> Qwen3TTS:
    """Qwen3-TTS in voice clone mode — clone from reference audio."""
    return Qwen3TTS(mode="voice_clone", ref_audio=ref_audio, ref_text=ref_text, **kwargs)
