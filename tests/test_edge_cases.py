"""Regression tests for the 7 runtime edge cases fixed in v1.0.3."""

from __future__ import annotations

import asyncio
import json
import struct

import httpx

from autum.app import BubbleSupervisor, command_handler
from autum.audio import HandsFreeInput, SpeechPlayer, VoiceInput
from autum.bus import EventBus
from autum.ollama import ModelSwapper, OmniChat
from autum.runtime import Mode
from autum.transport import parse_command


def run(coro):
    return asyncio.run(coro)


def _pcm_chunk(amplitude: int, samples: int = 1600) -> bytes:
    return struct.pack("<" + "h" * samples, *([amplitude] * samples))


class _FakeStdout:
    def __init__(self, chunks: list[bytes]) -> None:
        self._chunks = list(chunks)

    async def read(self, _n: int) -> bytes:
        await asyncio.sleep(0.005)
        return self._chunks.pop(0) if self._chunks else b""


class _FakeProc:
    def __init__(self, chunks: list[bytes] | None = None, *, broken_pipe: bool = False) -> None:
        self.stdout = _FakeStdout(chunks or [])
        self.returncode: int | None = None
        self._done = asyncio.Event()
        self.stdin = _BrokenStdin() if broken_pipe else None

    def kill(self) -> None:
        self.returncode = -9
        self._done.set()

    def terminate(self) -> None:
        self.returncode = 0
        self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return int(self.returncode or 0)


class _BrokenStdin:
    def write(self, _data: bytes) -> None:
        raise BrokenPipeError("PipeWire sink closed")

    async def drain(self) -> None:
        raise BrokenPipeError("PipeWire sink closed")

    def close(self) -> None:
        pass


# 1. keep_alive: -1 persists in OmniChat.stream() and ModelSwapper swaps
def test_keep_alive_minus_one_persists_across_chat_and_swap():
    requests: list[tuple[str, dict]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        requests.append((request.url.path, body))
        if request.url.path == "/v1/chat/completions":
            chunk = {"choices": [{"delta": {"content": "Hi"}}]}
            return httpx.Response(200, text=f"data: {json.dumps(chunk)}\n\ndata: [DONE]\n\n")
        return httpx.Response(200, json={"done": True})

    async def scenario():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            swapper = ModelSwapper(client, pinned=("lfm3-omni:2b",))
            await swapper.ensure_active("gemma4:12b")
            await swapper.ensure_active("qwen3-vl:8b")
            chat = OmniChat(client, model="gemma4:12b", swapper=swapper)
            out = [t async for t in chat.stream([{"role": "user", "content": "hi"}], operation_id="op1")]
            return out

    assert run(scenario()) == ["Hi"]
    chat_reqs = [b for p, b in requests if p == "/v1/chat/completions"]
    assert chat_reqs and chat_reqs[0]["keep_alive"] == -1
    preload_reqs = [b for p, b in requests if p == "/api/generate" and b.get("keep_alive") == -1]
    assert [b["model"] for b in preload_reqs] == ["qwen3-vl:8b", "gemma4:12b"]


# 2. wake_word _verify_barge_in surfaces rate-limited transcription errors and recovers cleanly
def test_wake_word_verify_surfaces_and_recovers_transcription_errors():
    class _FlakyTranscriber:
        def __init__(self) -> None:
            self.fail = True

        async def transcribe(self, _wav: bytes) -> str:
            if self.fail:
                raise RuntimeError("Ollama offline")
            return "Autum what time is it?"

    class _DummyRuntime:
        mode = Mode.WAKE_WORD

        def __init__(self) -> None:
            self.barges: list[str] = []

        def barge_in_with_text(self, text: str, **_kwargs) -> None:
            self.barges.append(text)

        def submit_speech(self, *_args) -> None:
            pass

    class _DummyDetector:
        barge_in_frames = 3
        in_speech = False

        def reset(self) -> None:
            pass

    async def scenario():
        bus = EventBus("s")
        seen = []
        bus.subscribe(seen.append)
        tx = _FlakyTranscriber()
        rt = _DummyRuntime()
        hf = HandsFreeInput(bus, rt, tx, _DummyDetector())
        # Two rapid failures -> only one rate-limited error emitted
        await hf._verify_barge_in("t1", b"WAV")
        await hf._verify_barge_in("t2", b"WAV")
        # Recovery -> emits component_recovered and dispatches wake text
        tx.fail = False
        await hf._verify_barge_in("t3", b"WAV")
        return seen, rt.barges

    seen, barges = run(scenario())
    errs = [e for e in seen if e.type == "runtime.component_error"]
    recs = [e for e in seen if e.type == "runtime.component_recovered"]
    assert len(errs) == 1 and errs[0].payload["component"] == "transcription"
    assert len(recs) == 1 and recs[0].payload["component"] == "transcription"
    assert barges == ["Autum what time is it?"]


# 3. Tap-to-Talk auto-finishes on post-speech silence (800 ms) and command parses cleanly
def test_tap_to_talk_auto_finishes_on_post_speech_silence():
    assert parse_command(b'{"command": "talk_auto_finish"}') == {"command": "talk_auto_finish"}

    class _FakeRuntime:
        state = "speech_detected"
        mode = Mode.PUSH_TO_TALK
        direct_audio = False

        def __init__(self) -> None:
            self.submitted: list[tuple[str, bytes]] = []

        def submit_speech(self, turn_id: str, audio: bytes, _cause=None) -> None:
            self.submitted.append((turn_id, audio))

    class _FakeTx:
        async def transcribe(self, _wav: bytes) -> str:
            return "hello"

    # 3 loud speech chunks (300 ms) followed by 10 silent chunks (1000 ms)
    chunks = [_pcm_chunk(4000)] * 3 + [_pcm_chunk(0)] * 10 + [_pcm_chunk(0)] * 20

    async def spawn_(*_args, **_kwargs):
        return _FakeProc(chunks)

    async def scenario():
        bus = EventBus("s")
        rt = _FakeRuntime()
        voice = VoiceInput(bus, rt, _FakeTx(), spawn_=spawn_)
        handle = command_handler(rt, voice)
        await handle({"command": "talk_start"})
        await handle({"command": "talk_auto_finish"})
        # Wait for auto-finish after 3 speech + 8 quiet chunks (~60 ms simulated)
        for _ in range(100):
            if rt.submitted:
                break
            await asyncio.sleep(0.01)
        return rt.submitted, voice.capturing

    submitted, capturing = run(scenario())
    assert len(submitted) == 1
    assert not capturing


# 4. Concurrent BubbleSupervisor.reload() calls are serialized without spawning duplicates
def test_concurrent_bubble_reloads_are_serialized():
    async def scenario():
        bus = EventBus("s")
        errors = []
        bus.subscribe(errors.append, {"runtime.component_error"})
        procs: list[_FakeProc] = []

        async def starter():
            await asyncio.sleep(0.02)
            p = _FakeProc()
            procs.append(p)
            return p

        sup = BubbleSupervisor(bus, starter=starter, restart_delay_s=0.05)
        await sup.start()
        await asyncio.gather(sup.reload(), sup.reload())
        alive = [p for p in procs if p.returncode is None]
        await sup.stop()
        alive_after_stop = [p for p in procs if p.returncode is None]
        return len(procs), len(alive), len(alive_after_stop), errors

    started_count, alive_mid, alive_end, errors = run(scenario())
    assert started_count == 3
    assert alive_mid == 1
    assert alive_end == 0
    assert errors == []


# 5. SpeechPlayer handles BrokenPipeError cleanly without leaving a dead handle
def test_speech_player_handles_broken_pipe_cleanly():
    class _Synth:
        async def stream(self, _text: str):
            yield _pcm_chunk(1000)

    async def spawn_(*_args, **_kwargs):
        return _FakeProc(broken_pipe=True)

    async def scenario():
        bus = EventBus("s")
        bus.fence.open("op1")
        player = SpeechPlayer(bus, _Synth(), spawn_=spawn_)
        player.enqueue("Hello.", operation_id="op1", turn_id="t1")
        try:
            await player.drain("op1")
            return None, player._proc
        except RuntimeError as exc:
            return str(exc), player._proc

    err, proc = run(scenario())
    assert err is not None and "disconnected" in err
    assert proc is None


# 6. VoiceInput clears _awaiting_transcript WAV bytes immediately on runtime.component_error
def test_direct_audio_clears_awaiting_transcript_on_component_error():
    async def scenario():
        bus = EventBus("s")
        rt = type("R", (), {"submit_speech": lambda *a: None})()
        tx = type("T", (), {"transcribe": lambda self, w: asyncio.sleep(0, result="")})()
        voice = VoiceInput(bus, rt, tx)
        voice._awaiting_transcript["turn_1"] = b"RAW_WAV_BYTES"
        bus.emit("runtime.component_error", {"component": "response", "error": "boom"}, turn_id="turn_1")
        return dict(voice._awaiting_transcript)

    assert run(scenario()) == {}
