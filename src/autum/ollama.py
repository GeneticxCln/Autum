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

import asyncio
import base64
import json
from typing import Any, AsyncIterator, Callable, Iterable, Sequence

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


class ModelSwapper:
    """Coordinates dynamic VRAM swapping between mutually exclusive heavy GPU models.

    Pinned lightweight models (such as the voice engine `lfm3-omni:2b`) stay
    resident in VRAM, while heavy chat (`gemma4:12b`) and vision (`qwen3-vl:8b`)
    models cleanly unload each other via `keep_alive: 0` before loading so both
    always run 100% on GPU within a 16 GB VRAM budget.
    """

    def __init__(
        self,
        client: httpx.AsyncClient,
        *,
        base_url: str = OLLAMA,
        pinned: Iterable[str] = (),
        on_swap: Callable[[str, str | None], None] | None = None,
    ) -> None:
        self.client = client
        self.base_url = base_url.rstrip("/")
        self.pinned = frozenset(pinned)
        self.on_swap = on_swap
        self.active: str | None = None
        self._lock = asyncio.Lock()

    async def ensure_active(self, model: str) -> None:
        """Ensure `model` is the active heavy model, unloading any prior heavy model first."""
        if model in self.pinned:
            return
        async with self._lock:
            if self.active == model:
                return
            prev = self.active
            if prev is not None and prev not in self.pinned:
                self.active = None
                await unload(self.client, model=prev, base_url=self.base_url)
            self.active = model
            if self.on_swap is not None and prev is not None:
                self.on_swap(model, prev)


class OmniChat:
    """Inference: streams reply text chunks and optional ToolCall proposals."""

    def __init__(
        self,
        client: httpx.AsyncClient,
        model: str = MODEL,
        base_url: str = OLLAMA,
        *,
        reasoning_effort: str | None = None,
        swapper: ModelSwapper | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.reasoning_effort = reasoning_effort
        self.swapper = swapper

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        operation_id: str,
        tools: Sequence[dict[str, Any]] | None = None,
    ) -> AsyncIterator[str | ToolCall]:
        if self.swapper is not None:
            await self.swapper.ensure_active(self.model)
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai(m) for m in messages],
            "stream": True,
            "temperature": 0,
            "seed": SEED,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = list(tools)
        pending_calls: dict[int, dict[str, str]] = {}
        in_think = False
        think_buf = ""
        async with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=body) as r:
            if r.status_code != 200:
                raise RuntimeError(f"Ollama chat failed ({r.status_code}): {(await r.aread())[:200]!r}")
            async for line in r.aiter_lines():
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                choice = json.loads(line[6:])["choices"][0]
                delta = choice.get("delta", {})
                if delta.get("content"):
                    think_buf += str(delta["content"])
                    while think_buf:
                        if in_think:
                            end_idx = think_buf.find("</think>")
                            if end_idx < 0:
                                think_buf = think_buf[-7:] if len(think_buf) > 7 else think_buf
                                break
                            think_buf = think_buf[end_idx + 8 :]
                            in_think = False
                        else:
                            start_idx = think_buf.find("<think>")
                            if start_idx < 0:
                                # Hold back a partial '<think' prefix at chunk boundary
                                partial = 0
                                for k in range(1, min(7, len(think_buf) + 1)):
                                    if "<think>".startswith(think_buf[-k:]):
                                        partial = k
                                emit = think_buf[:-partial] if partial else think_buf
                                think_buf = think_buf[-partial:] if partial else ""
                                if emit:
                                    yield emit
                                break
                            if start_idx > 0:
                                yield think_buf[:start_idx]
                            think_buf = think_buf[start_idx + 7 :]
                            in_think = True
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
            if think_buf and not in_think:
                yield think_buf
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
                err_bytes = await r.aread()
                if r.status_code == 400 and b"does not support speech" in err_bytes and self.model != MODEL:
                    self.model = MODEL
                else:
                    raise RuntimeError(f"Ollama speech failed ({r.status_code}): {err_bytes[:200]!r}")
            else:
                async for chunk in r.aiter_bytes():
                    if chunk:
                        yield chunk
                return
        # Automatic fallback to speech-capable voice engine (lfm3-omni:2b)
        fallback_body = {"model": self.model, "input": text, "voice": self.voice, "response_format": "pcm"}
        async with self.client.stream("POST", f"{self.base_url}/v1/audio/speech", json=fallback_body) as r2:
            if r2.status_code != 200:
                raise RuntimeError(f"Ollama speech failed ({r2.status_code}): {(await r2.aread())[:200]!r}")
            async for chunk in r2.aiter_bytes():
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
    await client.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": "30m"}, timeout=90)


async def unload(client: httpx.AsyncClient, model: str, base_url: str = OLLAMA) -> None:
    """Immediately release a model from Ollama GPU VRAM (`keep_alive: 0`)."""
    try:
        await client.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
    except httpx.HTTPError:
        pass


async def resolve_speech_model(client: httpx.AsyncClient, model: str, base_url: str = OLLAMA) -> str:
    """Return `model` if it supports Ollama TTS (`/v1/audio/speech`), else fall back to `lfm3-omni:2b`."""
    base = base_url.rstrip("/")
    try:
        r = await client.post(f"{base}/api/show", json={"name": model}, timeout=3)
        if r.status_code == 200:
            caps = r.json().get("capabilities")
            if isinstance(caps, list) and caps and "speech" not in caps:
                await unload(client, model=model, base_url=base)
                return MODEL
    except (httpx.HTTPError, ValueError):
        pass
    return model


async def check_ready(client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> str | None:
    """None if Ollama runs and has the model; otherwise what to do about it."""
    try:
        r = await client.get(f"{base_url.rstrip('/')}/api/tags", timeout=3)
    except httpx.HTTPError:
        return "Ollama is not running. Start it (systemctl start ollama, or `ollama serve`) and try again."
    names = {m["name"] for m in r.json().get("models", [])}
    if model not in names:
        return f"Ollama does not have {model}. Pull it with `ollama pull {model}` (with your approval) or update config.toml."
    return None
