"""LFM3-Omni through a local Ollama server: provisional pipeline D.

The model hears the user's audio directly, streams a text reply, and
speaks that reply with its own voice. Transcription is a separate call,
made after the reply, so captions and history get the user's words
without delaying the answer.

All three calls use the OpenAI-compatible endpoints, which load the model
with the same 4,096 context, so switching between them never reloads it
(methodology §11). Sampling matches the benchmark: temperature 0, seed 42.
"""

from __future__ import annotations

import base64
import json
from typing import Any, AsyncIterator, Sequence

import httpx

from autum.events import new_id
from autum.runtime import Message
from autum.tools import ToolCall

OLLAMA = "http://127.0.0.1:11434"
MODEL = "lfm3-omni:2b"
VOICE = "female-american-english"
SPEECH_RATE = 24_000  # 16-bit mono PCM from /v1/audio/speech
SEED = 42


def to_openai(message: Message) -> dict:
    """Runtime message -> OpenAI chat message. Audio goes as input_audio."""
    if message.get("audio") is not None:
        data = base64.b64encode(message["audio"]).decode()
        return {"role": "user", "content": [{"type": "input_audio", "input_audio": {"data": data, "format": "wav"}}]}
    if message["role"] == "tool":
        return {
            "role": "tool",
            "tool_call_id": message.get("tool_call_id", ""),
            "content": str(message.get("content") or ""),
        }
    content = message.get("content") or ""
    if message["role"] == "assistant":
        # Strip the runtime's internal interruption marker so small models do not
        # echo "[interrupted by the user]" aloud on the next turn.
        content = content.replace(" [interrupted by the user]", "").strip()
        if message.get("tool_calls"):
            return {"role": "assistant", "content": content, "tool_calls": list(message["tool_calls"])}
    if message["role"] == "user" and not content:
        content = "(unclear speech)"  # an earlier spoken turn whose transcript never arrived
    return {"role": message["role"], "content": content}


class OmniChat:
    """Inference: streams reply text chunks and optional ToolCall proposals."""

    def __init__(self, client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        operation_id: str,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai(m) for m in messages],
            "stream": True,
            "temperature": 0,
            "seed": SEED,
        }
        if tools:
            body["tools"] = list(tools)
        pending_calls: dict[int, dict[str, str]] = {}
        async with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama chat failed ({r.status_code}): {(await r.aread())[:200]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    yield delta["content"]
                for tc in delta.get("tool_calls") or ():
                    idx = int(tc.get("index", 0))
                    slot = pending_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                    if tc.get("id"):
                        slot["id"] = str(tc["id"])
                    fn = tc.get("function") or {}
                    if fn.get("name"):
                        slot["name"] += str(fn["name"])
                    if fn.get("arguments"):
                        slot["arguments"] += str(fn["arguments"])
        for idx in sorted(pending_calls):
            slot = pending_calls[idx]
            if not slot["name"]:
                continue
            try:
                args = json.loads(slot["arguments"]) if slot["arguments"] else {}
            except ValueError:
                args = {"_raw": slot["arguments"]}
            if not isinstance(args, dict):
                args = {"_raw": args}
            yield ToolCall(id=slot["id"] or new_id(), name=slot["name"], arguments=args)


class OmniSpeech:
    """Streams 24 kHz 16-bit mono PCM for one piece of text."""

    def __init__(
        self, client: httpx.AsyncClient, model: str = MODEL, voice: str = VOICE, base_url: str = OLLAMA
    ) -> None:
        self.client = client
        self.model = model
        self.voice = voice
        self.base_url = base_url.rstrip("/")

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        body = {"model": self.model, "input": text, "voice": self.voice, "response_format": "pcm"}
        async with self.client.stream("POST", f"{self.base_url}/v1/audio/speech", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama speech failed ({r.status_code}): {(await r.aread())[:200]!r}")
            async for chunk in r.aiter_bytes():
                if chunk:
                    yield chunk


class OmniTranscriber:
    def __init__(self, client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url.rstrip("/")

    async def transcribe(self, wav: bytes) -> str:
        r = await self.client.post(
            f"{self.base_url}/v1/audio/transcriptions",
            data={"model": self.model, "language": "en"},
            files={"file": ("speech.wav", wav, "audio/wav")},
        )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama transcription failed ({r.status_code}): {r.text[:200]}")
        return r.json()["text"].strip()


async def preload(client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> None:
    """Load the model now (default context, like the endpoints above) so the first turn is fast."""
    await client.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": "30m"}, timeout=60)


async def check_ready(client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> str | None:
    """None if Ollama runs and has the model; otherwise what to do about it."""
    try:
        r = await client.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
    except httpx.HTTPError:
        return "Ollama is not running. Start it (systemctl start ollama, or `ollama serve`) and try again."
    names = {m["name"] for m in r.json().get("models", [])}
    if model not in names:
        return f"Ollama does not have {model}. It is in the WP1 candidate list; pulling it needs your approval."
    return None
