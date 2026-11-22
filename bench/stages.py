"""Candidate pipeline stages. Each runs locally; none downloads anything.

Ollama is reached on localhost only. Hugging Face loads are forced
offline so a missing model fails instead of downloading (plan §5.5).
"""

from __future__ import annotations

import asyncio
import base64
import io
import json
import os
import time
import wave
from typing import AsyncIterator

import httpx
import numpy as np

from bench.common import SR, pcm16_to_float

os.environ.setdefault("HF_HUB_OFFLINE", "1")

OLLAMA = "http://127.0.0.1:11434"
KEEP_ALIVE = "30m"
SEED = 42

# Context window per model. LFM3-Omni must match the 4,096 its speech
# endpoint loads with: any other size makes Ollama reload the model
# (about 6 s) every time a turn switches between chat and speech.
CONTEXT = {"qwen3:8b": 8192, "lfm3-omni:2b": 4096}

# Each model's shipped defaults (`ollama show <model> --parameters`,
# checked 2026-11-22), pinned with a fixed seed so replies repeat across
# runs without changing how the model normally behaves.
SAMPLING = {
    "qwen3:8b": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    "lfm3-omni:2b": {"temperature": 0.0},
}

SYSTEM_PROMPT = (
    "You are Autum, a friendly voice assistant running on the user's computer. "
    "Your replies are spoken aloud, so answer in one to three short sentences of plain "
    "conversational text, with no lists, headings, or markdown. You cannot see the "
    "screen or take actions yet; if asked to, say so briefly."
)


def wav_bytes(audio: np.ndarray, sr: int = SR) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes((np.clip(audio, -1, 1) * 32767).astype(np.int16).tobytes())
    return buffer.getvalue()


# ----- recognition ------------------------------------------------------


class WhisperAsr:
    def __init__(self, size: str = "base", beam_size: int = 5) -> None:
        from faster_whisper import WhisperModel

        self.name = f"whisper-{size}"
        self.beam_size = beam_size
        self.model = WhisperModel(size, device="cuda", compute_type="float16", local_files_only=True)

    async def transcribe(self, client: httpx.AsyncClient, audio: np.ndarray) -> str:
        return await asyncio.to_thread(self._run, audio)

    def _run(self, audio: np.ndarray) -> str:
        segments, _ = self.model.transcribe(
            audio, language="en", beam_size=self.beam_size,
            vad_filter=False, condition_on_previous_text=False,
        )
        return " ".join(s.text.strip() for s in segments).strip()


class OllamaAsr:
    def __init__(self, model: str) -> None:
        self.name = f"ollama-asr:{model}"
        self.model = model

    async def transcribe(self, client: httpx.AsyncClient, audio: np.ndarray) -> str:
        r = await client.post(
            f"{OLLAMA}/v1/audio/transcriptions",
            data={"model": self.model, "language": "en"},
            files={"file": ("speech.wav", wav_bytes(audio), "audio/wav")},
        )
        r.raise_for_status()
        return r.json()["text"].strip()


# ----- language models --------------------------------------------------


class OllamaChat:
    """Text in, streamed text out, through Ollama's native chat API."""

    takes_audio = False

    def __init__(self, model: str) -> None:
        self.name = f"ollama:{model}"
        self.model = model

    async def stream(self, client: httpx.AsyncClient, messages: list[dict], audio=None) -> AsyncIterator[str]:
        body = {
            "model": self.model, "messages": messages, "stream": True, "think": False,
            "keep_alive": KEEP_ALIVE,
            "options": {"num_ctx": CONTEXT[self.model], "seed": SEED, **SAMPLING[self.model]},
        }
        async with client.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line:
                    continue
                data = json.loads(line)
                if "error" in data:
                    raise RuntimeError(data["error"])
                text = data.get("message", {}).get("content", "")
                if text:
                    yield text


class OllamaAudioChat:
    """Audio in, streamed text out: the model hears the user directly."""

    takes_audio = True

    def __init__(self, model: str) -> None:
        self.name = f"ollama-audio:{model}"
        self.model = model

    async def stream(self, client: httpx.AsyncClient, messages: list[dict], audio=None) -> AsyncIterator[str]:
        encoded = base64.b64encode(wav_bytes(audio)).decode()
        user = {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": encoded, "format": "wav"}}]}
        sampling = SAMPLING[self.model]
        body = {
            "model": self.model, "messages": [*messages, user], "stream": True,
            "seed": SEED, "temperature": sampling["temperature"],
            **({"top_p": sampling["top_p"]} if "top_p" in sampling else {}),
        }
        async with client.stream("POST", f"{OLLAMA}/v1/chat/completions", json=body) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                delta = json.loads(line[6:])["choices"][0].get("delta", {})
                if delta.get("content"):
                    yield delta["content"]


# ----- synthesis --------------------------------------------------------


class OllamaTts:
    def __init__(self, model: str, voice: str = "female-american-english") -> None:
        self.name = f"ollama-tts:{model}"
        self.model = model
        self.voice = voice
        self.sr = 24_000

    async def synth(self, client: httpx.AsyncClient, text: str) -> tuple[float, np.ndarray]:
        """Return (seconds to first audio byte, audio)."""
        start = time.perf_counter()
        first = None
        chunks: list[bytes] = []
        body = {"model": self.model, "input": text, "voice": self.voice, "response_format": "pcm"}
        async with client.stream("POST", f"{OLLAMA}/v1/audio/speech", json=body) as r:
            r.raise_for_status()
            async for data in r.aiter_bytes():
                if data and first is None:
                    first = time.perf_counter() - start
                chunks.append(data)
        return (first or 0.0), pcm16_to_float(b"".join(chunks))


class ChatterboxTts:
    def __init__(self) -> None:
        import torch
        from chatterbox.tts_turbo import ChatterboxTurboTTS

        self.name = "chatterbox-turbo"
        self.model = ChatterboxTurboTTS.from_pretrained(device="cuda")
        self.sr = self.model.sr
        self._torch = torch

    async def synth(self, client: httpx.AsyncClient, text: str) -> tuple[float, np.ndarray]:
        start = time.perf_counter()
        audio = await asyncio.to_thread(self._run, text)
        return time.perf_counter() - start, audio  # not streaming: first audio = whole chunk

    def _run(self, text: str) -> np.ndarray:
        with self._torch.inference_mode():
            wav = self.model.generate(text)
        return wav.squeeze().float().cpu().numpy()
