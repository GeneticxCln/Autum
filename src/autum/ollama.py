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
from contextlib import asynccontextmanager
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

    async def _activate_locked(self, model: str) -> None:
        if self.active == model:
            return
        prev = self.active
        if prev is not None and prev not in self.pinned:
            self.active = None
            await unload(self.client, model=prev, base_url=self.base_url)
            await asyncio.sleep(0.25)  # allow CUDA VRAM release before loading next heavy model
        self.active = model
        if prev is not None:
            try:
                await preload(self.client, model=model, base_url=self.base_url)
            except (httpx.HTTPError, RuntimeError):
                pass
        if self.on_swap is not None and prev is not None:
            self.on_swap(model, prev)

    async def ensure_active(self, model: str) -> None:
        """Ensure `model` is the active heavy model, unloading any prior heavy model first."""
        if model in self.pinned:
            return
        async with self._lock:
            await self._activate_locked(model)

    @asynccontextmanager
    async def lease(self, model: str) -> AsyncIterator[None]:
        """Hold the swapper lock across a heavy model call so another model cannot unload it mid-stream."""
        if model in self.pinned:
            yield
            return
        async with self._lock:
            await self._activate_locked(model)
            yield


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
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [to_openai(m) for m in messages],
            "stream": True,
            "temperature": 0,
            "seed": SEED,
            "keep_alive": -1,
        }
        if self.reasoning_effort is not None:
            body["reasoning_effort"] = self.reasoning_effort
        if tools:
            body["tools"] = list(tools)
        pending_calls: dict[int, dict[str, str]] = {}
        next_auto_idx = 0
        hidden_close: str | None = None
        think_buf = ""
        hidden_tags = (
            ("<think>", "</think>"),
            ("<|think|>", "<|/think|>"),
            ("<|channel>thought", "<channel|>"),
            ("<|tool_call>", "<tool_call|>"),
        )
        max_open = max(len(o) for o, _ in hidden_tags)

        async def _run_stream() -> AsyncIterator[str | ToolCall]:
            nonlocal hidden_close, think_buf, next_auto_idx
            async with self.client.stream("POST", f"{self.base_url}/v1/chat/completions", json=body) as r:
                if r.status_code != 200:
                    raise RuntimeError(f"Ollama chat failed ({r.status_code}): {(await r.aread())[:200]!r}")
                async for line in r.aiter_lines():
                    if not line.startswith("data: ") or line == "data: [DONE]":
                        continue
                    try:
                        payload = json.loads(line[6:])
                    except ValueError:
                        continue
                    if not isinstance(payload, dict):
                        continue
                    choices = payload.get("choices")
                    if not isinstance(choices, list) or not choices:
                        continue
                    choice = choices[0]
                    if not isinstance(choice, dict):
                        continue
                    delta = choice.get("delta")
                    if not isinstance(delta, dict):
                        continue
                    if delta.get("content"):
                        think_buf += str(delta["content"])
                        while think_buf:
                            if hidden_close is not None:
                                end_idx = think_buf.find(hidden_close)
                                if end_idx < 0:
                                    keep = len(hidden_close) - 1
                                    think_buf = think_buf[-keep:] if len(think_buf) > keep else think_buf
                                    break
                                think_buf = think_buf[end_idx + len(hidden_close) :]
                                hidden_close = None
                            else:
                                earliest_idx = -1
                                earliest_open = ""
                                earliest_close = ""
                                for open_tag, close_tag in hidden_tags:
                                    idx = think_buf.find(open_tag)
                                    if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
                                        earliest_idx = idx
                                        earliest_open = open_tag
                                        earliest_close = close_tag
                                if earliest_idx < 0:
                                    # Hold back a partial '<think' or '<|tool_call' prefix at chunk boundary
                                    partial = 0
                                    for k in range(1, min(max_open, len(think_buf) + 1)):
                                        tail = think_buf[-k:]
                                        if any(open_tag.startswith(tail) for open_tag, _ in hidden_tags):
                                            partial = k
                                    emit = think_buf[:-partial] if partial else think_buf
                                    think_buf = think_buf[-partial:] if partial else ""
                                    if emit:
                                        yield emit
                                    break
                                if earliest_idx > 0:
                                    yield think_buf[:earliest_idx]
                                think_buf = think_buf[earliest_idx + len(earliest_open) :]
                                hidden_close = earliest_close
                    for tc in delta.get("tool_calls") or ():
                        if not isinstance(tc, dict):
                            continue
                        raw_idx = tc.get("index")
                        tc_id = str(tc["id"]) if tc.get("id") else ""
                        if isinstance(raw_idx, int) and not isinstance(raw_idx, bool):
                            idx = raw_idx
                            if (
                                tc_id
                                and idx in pending_calls
                                and pending_calls[idx]["id"]
                                and pending_calls[idx]["id"] != tc_id
                            ):
                                idx = max(pending_calls) + 1
                        elif tc_id:
                            existing = next((k for k, v in pending_calls.items() if v["id"] == tc_id), None)
                            idx = existing if existing is not None else (max(pending_calls) + 1 if pending_calls else 0)
                        else:
                            idx = next_auto_idx
                            next_auto_idx += 1
                        slot = pending_calls.setdefault(idx, {"id": "", "name": "", "arguments": ""})
                        if tc_id:
                            slot["id"] = tc_id
                        fn = tc.get("function")
                        if isinstance(fn, dict):
                            if fn.get("name"):
                                slot["name"] += str(fn["name"])
                            if fn.get("arguments"):
                                slot["arguments"] += str(fn["arguments"])
                if think_buf and hidden_close is None:
                    yield think_buf

        if self.swapper is not None:
            async with self.swapper.lease(self.model):
                async for item in _run_stream():
                    yield item
        else:
            async for item in _run_stream():
                yield item
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
        self,
        client: httpx.AsyncClient,
        model: str = MODEL,
        voice: str = VOICE,
        base_url: str = OLLAMA,
        *,
        on_fallback: Callable[[str], None] | None = None,
    ) -> None:
        self.client = client
        self.model = model
        self.voice = voice
        self.base_url = base_url.rstrip("/")
        self.on_fallback = on_fallback

    async def stream(self, text: str) -> AsyncIterator[bytes]:
        body = {"model": self.model, "input": text, "voice": self.voice, "response_format": "pcm"}
        async with self.client.stream("POST", f"{self.base_url}/v1/audio/speech", json=body) as r:
            if r.status_code != 200:
                err_bytes = await r.aread()
                if r.status_code == 400 and b"does not support speech" in err_bytes and self.model != MODEL:
                    self.model = MODEL
                    if self.on_fallback is not None:
                        self.on_fallback(MODEL)
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
        if r.status_code == 400 and "does not support" in r.text.lower() and self.model != MODEL:
            self.model = MODEL
            r = await self.client.post(
                f"{self.base_url}/v1/audio/transcriptions",
                data={"model": self.model, "language": "en"},
                files={"file": ("speech.wav", wav, "audio/wav")},
            )
        if r.status_code != 200:
            raise RuntimeError(f"Ollama transcription failed ({r.status_code}): {r.text[:200]}")
        return str((r.json() or {}).get("text") or "").strip()


async def preload(client: httpx.AsyncClient, model: str = MODEL, base_url: str = OLLAMA) -> None:
    """Pin the model in GPU VRAM (`keep_alive: -1`) so it never expires while Autum runs."""
    r = await client.post(
        f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": -1}, timeout=90
    )
    if r.status_code != 200:
        raise RuntimeError(f"Ollama preload failed for {model} ({r.status_code}): {r.text[:200]}")


async def unload(client: httpx.AsyncClient, model: str, base_url: str = OLLAMA) -> None:
    """Immediately release a model from Ollama GPU VRAM (`keep_alive: 0`)."""
    try:
        await client.post(f"{base_url.rstrip('/')}/api/generate", json={"model": model, "keep_alive": 0}, timeout=10)
    except httpx.HTTPError:
        pass


def model_candidates(model: str) -> set[str]:
    """Candidate Ollama model identifiers (matching untagged `<name>` with `<name>:latest`)."""
    clean = (model or "").strip()
    if not clean:
        return set()
    return {clean} if ":" in clean else {clean, f"{clean}:latest"}


async def release_models(client: httpx.AsyncClient, models: Iterable[str], base_url: str = OLLAMA) -> None:
    """On exit, unload Autum's pinned (`keep_alive: -1`) models that are still in VRAM.

    Only models Ollama reports as loaded are unloaded: asking Ollama to unload a
    model that is not loaded would load it first. Matches untagged model names
    against Ollama's normalized `<name>:latest` identifier in `/api/ps`.
    """
    base = base_url.rstrip("/")
    try:
        r = await client.get(f"{base}/api/ps", timeout=3)
        loaded = {str(m.get("name") or "") for m in (r.json() or {}).get("models") or () if isinstance(m, dict)}
    except (httpx.HTTPError, ValueError):
        return
    unloaded: set[str] = set()
    for model in dict.fromkeys(models):
        for loaded_name in sorted(model_candidates(model) & loaded):
            if loaded_name and loaded_name not in unloaded:
                unloaded.add(loaded_name)
                await unload(client, model=loaded_name, base_url=base)


_THINKING_MODEL_TAGS: tuple[str, ...] = (
    "gemma4",
    "gemma-4",
    "qwen3",
    "qwen-3",
    "lfm3",
    "lfm-3",
    "deepseek",
    "qwq",
    "r1",
)


async def resolve_reasoning_effort(
    client: httpx.AsyncClient,
    model: str,
    chat_think: bool,
    base_url: str = OLLAMA,
) -> str | None:
    """Return `'none'` when `chat_think` is False and `model` supports thinking."""
    if chat_think:
        return None
    low = model.lower()
    if any(tag in low for tag in _THINKING_MODEL_TAGS):
        return "none"
    base = base_url.rstrip("/")
    try:
        r = await client.post(f"{base}/api/show", json={"name": model}, timeout=2.5)
        if r.status_code == 200:
            info = r.json() or {}
            caps = info.get("capabilities")
            if isinstance(caps, list) and "thinking" in caps:
                return "none"
            tmpl = str(info.get("template") or "")
            if any(tok in tmpl for tok in ("enable_thinking", "<|think|>", "<|channel>thought", "<think>")):
                return "none"
    except (httpx.HTTPError, ValueError):
        pass
    return None


async def resolve_speech_model(client: httpx.AsyncClient, model: str, base_url: str = OLLAMA) -> str:
    """Return `model` if it supports Ollama TTS (`/v1/audio/speech`), else fall back to `lfm3-omni:2b`."""
    base = base_url.rstrip("/")
    try:
        r = await client.post(f"{base}/api/show", json={"name": model}, timeout=3)
        if r.status_code == 200:
            caps = (r.json() or {}).get("capabilities")
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
    if r.status_code != 200:
        return f"Ollama returned HTTP {r.status_code} at {base_url}; check that Ollama is healthy."
    try:
        payload = r.json()
    except ValueError:
        return f"Ollama at {base_url} returned non-JSON data; check ollama_url in config.toml."
    models = payload.get("models") if isinstance(payload, dict) else None
    names = {str(m["name"]) for m in (models or ()) if isinstance(m, dict) and m.get("name")}
    if not (model_candidates(model) & names):
        return f"Ollama does not have {model}. Pull it with `ollama pull {model}` (with your approval) or update config.toml."
    return None
