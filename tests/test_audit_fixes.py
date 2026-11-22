"""Regression tests for every bug found in the full-project audit."""

from __future__ import annotations

import asyncio
import json
import os
import time
from pathlib import Path

import pytest

from autum.audio import HandsFreeInput, VoiceInput
from autum.bus import EventBus
from autum.capabilities import build_default_registry
from autum.config import Config
from autum.events import Provenance, new_id
from autum.hotkey import PushToTalkKey
from autum.integrations import IntegrationContext, IntegrationManager
from autum.integrations.linux_desktop import LinuxDesktopIntegration
from autum.memory import MemoryStore, register_memory_capabilities
from autum.perception import register_perception_capabilities
from autum.runtime import Mode, Runtime, State
from autum.tools import ApprovalGate, ToolCall, validate_arguments
from autum.vad import TurnDetector, VAD_FRAME_BYTES
from mocks import MockInference, MockSpeech, Recorder


def run(coro):
    return asyncio.run(coro)


class _ScriptedInference:
    def __init__(self, turns):
        self.turns = list(turns)

    async def stream(self, messages, *, operation_id, tools=None):
        for item in self.turns.pop(0):
            yield item


# 1. Compact tool schemas while preserving strict argument validation
def test_compact_openai_tools_strips_validation_boilerplate_and_keeps_strict_validation(tmp_path):
    reg, _, _ = build_default_registry(tmp_path)
    register_memory_capabilities(reg, MemoryStore(tmp_path / "mem.json"))
    register_perception_capabilities(reg)
    mgr = IntegrationManager()
    mgr.add(LinuxDesktopIntegration())
    mgr.load_enabled(reg, IntegrationContext(), ("linux_desktop",))

    exported = reg.openai_tools()
    raw = json.dumps(exported)
    assert len(raw) < 8200
    for item in exported:
        params = item["function"]["parameters"]
        assert "additionalProperties" not in params
        for prop in params["properties"].values():
            assert not (set(prop) & {"minLength", "maxLength", "minimum", "maximum"})

    # Full validation constraints remain enforced at runtime
    set_timer = reg.get("set_timer")
    assert set_timer is not None
    assert validate_arguments(set_timer.input_schema, {"seconds": 0}) is not None
    assert validate_arguments(set_timer.input_schema, {"seconds": 10, "extra": 1}) is not None


# 2a. Hands-free speech does not clobber push-to-talk irreversible approval
def test_hands_free_speech_cannot_clobber_push_to_talk_irreversible_approval(tmp_path):
    async def scenario():
        bus = EventBus("s")
        registry, timers, notes = build_default_registry(tmp_path)
        notes.create("secret-plan", "keep")
        inf = _ScriptedInference(
            [
                [ToolCall("d1", "delete_note", {"title": "secret-plan"})],
                ["Deleted."],
            ]
        )
        rt = Runtime(
            bus,
            inf,
            MockSpeech(bus, synth_s=0.0, per_char_s=0.0),
            mode=Mode.HANDS_FREE,
            tools=registry,
            approval_gate=ApprovalGate(timeout_s=2.0),
        )
        rt.submit_text("Delete secret-plan.")
        deadline = time.monotonic() + 2.0
        while rt.state is not State.WAITING_FOR_APPROVAL and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert rt.state is State.WAITING_FOR_APPROVAL

        # User holds push-to-talk key, then HandsFreeInput also fires on the same voice
        ptt_turn = new_id()
        hf_turn = new_id()
        bus.emit("audio.speech_started", {"mode": "push_to_talk"}, provenance=Provenance.USER, turn_id=ptt_turn)
        bus.emit("audio.speech_started", {"mode": "hands_free"}, provenance=Provenance.USER, turn_id=hf_turn)
        # PTT approval resolves cleanly instead of being overwritten by hands_free
        bus.emit("audio.transcribed", {"text": "yes", "final": True}, provenance=Provenance.USER, turn_id=ptt_turn)
        await rt.wait_until_settled()
        timers.stop_all()
        return notes.exists("secret-plan")

    assert run(scenario()) is False


# 2b. Short "yes" (< 0.3s) during approval is kept, and approval WAV is not leaked
def test_short_approval_reply_is_not_dropped_and_does_not_leak_wav():
    class _FakeProc:
        def __init__(self, chunks):
            self._chunks = list(chunks)
            self.stdout = self
            self.returncode = None

        async def read(self, _n):
            if self._chunks:
                return self._chunks.pop(0)
            await asyncio.sleep(0.05)
            return b""

        def kill(self):
            self.returncode = -9

        async def wait(self):
            return -9

    class _WaitingRuntime:
        state = State.WAITING_FOR_APPROVAL

        def __init__(self):
            self.submitted = []

        def submit_speech(self, turn_id, wav, cause=None):
            self.submitted.append((turn_id, wav))

    class _Transcriber:
        async def transcribe(self, _wav):
            return "yes"

    async def scenario():
        bus = EventBus("s")
        rt = _WaitingRuntime()

        async def spawn_(*_args, **_kw):
            # 200 ms (< 300 ms MIN_UTTERANCE_S, >= 120 ms MIN_APPROVAL_UTTERANCE_S)
            chunk = b"\x00\x01" * 1600
            return _FakeProc([chunk, chunk])

        voice = VoiceInput(bus, rt, _Transcriber(), spawn_=spawn_)
        await voice.start()
        await asyncio.sleep(0.02)
        await voice.finish()
        return rt.submitted, voice._awaiting_transcript

    submitted, cached = run(scenario())
    assert len(submitted) == 1
    assert cached == {}  # approval WAV must not linger in _awaiting_transcript


# 3. Nested observation stack keeps screen-watch halo alive across transient reads
def test_nested_observation_keeps_screen_watch_active_until_explicitly_stopped():
    bus = EventBus("s")
    rec = Recorder(bus)
    rt = Runtime(bus, MockInference("ok"), MockSpeech(bus))

    rt.set_observation("screen")
    assert rt.observation == "screen"
    rt.set_observation("document")
    assert rt.observation == "document"
    rt.set_observation("none")
    assert rt.observation == "screen"  # screen watch still active!
    rt.end_observation("screen")
    assert rt.observation == "none"
    assert [e.payload["source"] for e in rec.of("perception.completed")] == ["document", "screen"]


# 4. Hands-free mid-session microphone stall triggers error & automatic recovery
def test_hands_free_recovers_from_mid_session_microphone_stall():
    class _StallingProc:
        def __init__(self, initial):
            self.stdout = asyncio.StreamReader()
            self.returncode = None
            self._done = asyncio.Event()
            for chunk in initial:
                self.stdout.feed_data(chunk)

        def kill(self):
            if self.returncode is None:
                self.returncode = -9
                self.stdout.feed_eof()
                self._done.set()

        async def wait(self):
            await self._done.wait()
            return self.returncode

    async def scenario():
        bus = EventBus("s")
        rec = Recorder(bus)
        rt = Runtime(bus, MockInference("ok"), MockSpeech(bus), mode=Mode.HANDS_FREE)
        procs = []

        async def spawn_(*_args, **_kw):
            p = _StallingProc([b"\x00" * VAD_FRAME_BYTES])
            procs.append(p)
            return p

        detector = TurnDetector(lambda _pcm: 0.0)
        hf = HandsFreeInput(bus, rt, None, detector, spawn_=spawn_, stall_s=0.06, reconnect_s=0.02)
        await hf.start()
        deadline = time.monotonic() + 2.0
        while not rec.of("runtime.component_recovered") and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await hf.stop()
        return rec, len(procs)

    rec, count = run(scenario())
    assert count >= 2
    assert rec.of("runtime.component_error")[0].payload["component"] == "microphone"
    assert rec.of("runtime.component_recovered")[0].payload == {"component": "microphone"}


# 5. Preamble text before a ToolCall transitions SPEAKING -> THINKING and keeps full text
def test_preamble_text_before_tool_call_transitions_cleanly_and_preserves_full_reply(tmp_path):
    async def scenario():
        bus = EventBus("s")
        rec = Recorder(bus)
        reg, timers, _ = build_default_registry(tmp_path)
        inf = _ScriptedInference(
            [
                ["Let me check the clock. ", ToolCall("t1", "get_time", {})],
                ["It is noon."],
            ]
        )
        rt = Runtime(bus, inf, MockSpeech(bus, synth_s=0.0, per_char_s=0.0), tools=reg)
        rt.submit_text("What time is it?")
        await rt.wait_until_settled()
        timers.stop_all()
        return rec, rt

    rec, rt = run(scenario())
    assert rt.state is State.IDLE
    finished = rec.of("conversation.response_finished")[0].payload["text"]
    assert "Let me check the clock." in finished and "It is noon." in finished
    assert "thinking" in rec.states() and rec.states()[-1] == "idle"


# 6 & 8. look_at_screen blocks active excluded windows & move_window verifies actual coordinates
def test_look_at_screen_blocks_excluded_active_window_and_move_verifies_position(tmp_path):
    class _FakeWatcher:
        active = True
        frame_age_s = 0.1

        def latest_frame(self):
            return b"\xff\xd8\xff\xd9"

    class _FakeWM:
        def __init__(self):
            self.excluded_hit: str | None = "keepassxc"
            self.actual_x = 0

        def is_available(self):
            return True

        async def active_excluded_window(self, _excluded):
            return self.excluded_hit

        async def list_windows(self, _excluded=()):
            return [{"id": "w1", "title": "Firefox", "app": "firefox", "x": self.actual_x, "y": 0, "width": 1280, "height": 1440, "minimized": False, "maximized": False, "desktop": 1}]

        async def move_window(self, _query, **_kw):
            return {"id": "w1", "title": "Firefox", "app": "firefox", "x": 0, "y": 0, "width": 1280, "height": 1440, "target_x": 1280, "target_y": 0, "desktop": 1}

    async def scenario():
        bus = EventBus("s")
        op = new_id()
        bus.fence.open(op)
        reg, _, _ = build_default_registry(tmp_path)
        wm = _FakeWM()
        integ = LinuxDesktopIntegration(screen_watcher=_FakeWatcher(), window_controller=wm)

        async def fake_vision(_b, _m, _q):
            return "Desktop visible."

        integ.register(reg, IntegrationContext(vision=fake_vision, excluded_windows=("keepassxc",)))

        blocked = await reg.execute(ToolCall("l1", "look_at_screen", {}), bus=bus, operation_id=op, turn_id="t1")
        wm.excluded_hit = None
        allowed = await reg.execute(ToolCall("l2", "look_at_screen", {}), bus=bus, operation_id=op, turn_id="t2")

        # Window failed to move to target_x=1280 (stayed at x=0) -> unverified!
        unverified = await reg.execute(
            ToolCall("m1", "move_window", {"window": "firefox", "position": "right"}),
            bus=bus, operation_id=op, turn_id="t3",
        )
        # Window actually moved to x=1280 -> verified!
        wm.actual_x = 1280
        verified = await reg.execute(
            ToolCall("m2", "move_window", {"window": "firefox", "position": "right"}),
            bus=bus, operation_id=op, turn_id="t4",
        )
        return blocked, allowed, unverified, verified

    blocked, allowed, unverified, verified = run(scenario())
    assert blocked.status == "failed" and "keepassxc" in (blocked.error or "")
    assert allowed.status == "verified"
    assert unverified.status == "unverified"
    assert verified.status == "verified"


# 7. PresenceModel latency buffers stay bounded over long sessions
def test_presence_model_latency_buffers_stay_bounded(tmp_path):
    pytest.importorskip("PySide6")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtGui import QGuiApplication
    from autum.presence.model import PresenceModel

    _ = QGuiApplication.instance() or QGuiApplication([])
    model = PresenceModel(tmp_path / "none.sock")
    bus = EventBus("s")
    op = new_id()
    bus.fence.open(op)
    for i in range(600):
        ev = bus.emit("speech.output_level", {"rms": 0.1 + (i % 5) * 0.05}, operation_id=op)
        model.apply(ev)
    assert len(model.applied_ms) == 256


# 9. Strict localhost validation rejects prefix tricks
def test_config_rejects_localhost_prefix_spoofing():
    with pytest.raises(ValueError, match="localhost"):
        Config(ollama_url="http://localhost.evil.com:11434")
    with pytest.raises(ValueError, match="localhost"):
        Config(ollama_url="https://127.0.0.1:11434")
    assert Config(ollama_url="http://127.0.0.1:11434").ollama_url == "http://127.0.0.1:11434"


# 10. PushToTalkKey.stop() blocks late dispatches cleanly
def test_hotkey_stop_prevents_dispatch():
    async def scenario():
        loop = asyncio.get_running_loop()
        calls = []
        hk = PushToTalkKey(loop, lambda: calls.append("press"), lambda: calls.append("release"), lambda m: calls.append(m))
        hk.stop()
        hk._dispatch(lambda: calls.append("late"))
        await asyncio.sleep(0.01)
        return calls

    assert run(scenario()) == []


# 11. Local data directory, audit log, memory store, and notes are strictly owner-only (0700 / 0600)
def test_local_data_files_are_owner_only_permissions(tmp_path):
    import stat

    data_dir = tmp_path / "autum-private"
    reg, timers, notes = build_default_registry(data_dir)
    reg.audit.record("tool_requested", tool="get_time")
    notes.create("private-note", "hello")
    mem = MemoryStore(data_dir / "memory.json")
    mem.remember("color", "blue")
    timers.stop_all()

    assert stat.S_IMODE(data_dir.stat().st_mode) == 0o700
    assert stat.S_IMODE((data_dir / "audit.jsonl").stat().st_mode) == 0o600
    assert stat.S_IMODE((data_dir / "memory.json").stat().st_mode) == 0o600
    assert stat.S_IMODE((data_dir / "notes").stat().st_mode) == 0o700
    assert stat.S_IMODE((data_dir / "notes" / "private-note.txt").stat().st_mode) == 0o600


# 12. Fuzzy KWin window resolution, position aliases, switch_desktop, and multi-model config
def test_kwin_fuzzy_matching_position_aliases_and_switch_desktop():
    from autum.integrations.kwin_windows import KWinController, normalize_position

    assert normalize_position("left_half") == "left"
    assert normalize_position("upper right") == "top_right"
    assert normalize_position("middle") == "center"
    with pytest.raises(ValueError):
        normalize_position("diagonal")

    wm = KWinController()

    async def fake_list(_excluded=()):
        return [
            {"id": "w1", "app": "google-chrome", "title": "DISTRUG VACANȚELE ROMÂNILOR - ZANZIBAR - YouTube", "desktop": 1},
            {"id": "w2", "app": "antigravity", "title": "Autum", "desktop": 2},
        ]

    wm.list_windows = fake_list  # type: ignore[method-assign]

    async def scenario():
        r1 = await wm._resolve_window("google chrome", ())
        r2 = await wm._resolve_window("browser", ())
        r3 = await wm._resolve_window("DISTRUG VACANÇÉ ROMÂNILOR - ZANZIBAR", ())
        return r1["id"], r2["id"], r3["id"]

    assert run(scenario()) == ("w1", "w1", "w1")

    cfg = Config()
    assert cfg.model == "lfm3-omni:2b"
    assert cfg.resolved_chat_model() == "gemma4:12b"
    assert cfg.resolved_vision_model() == "qwen3-vl:8b"
    assert cfg.dynamic_swap is True
    assert Config(chat_model="lfm3-omni:2b", vision_model="").resolved_vision_model() == "lfm3-omni:2b"


# 13. Dynamic VRAM ModelSwapper unloads outgoing heavy model and never evicts pinned voice model
def test_model_swapper_dynamically_swaps_heavy_models_and_pins_voice():
    import httpx
    from autum.ollama import ModelSwapper

    unloaded: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        if body.get("keep_alive") == 0:
            unloaded.append(body["model"])
        return httpx.Response(200, json={"done": True})

    async def scenario():
        swaps: list[tuple[str, str | None]] = []
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            swapper = ModelSwapper(
                client,
                pinned=("lfm3-omni:2b",),
                on_swap=lambda new, old: swaps.append((new, old)),
            )
            await swapper.ensure_active("lfm3-omni:2b")  # pinned: ignored
            await swapper.ensure_active("gemma4:12b")    # initial heavy load: no unload
            await swapper.ensure_active("gemma4:12b")    # same model: no-op
            await swapper.ensure_active("qwen3-vl:8b")   # unloads gemma4:12b
            await swapper.ensure_active("qwen3-vl:8b")   # stays hot: no unload
            await swapper.ensure_active("gemma4:12b")    # unloads qwen3-vl:8b
        return swaps

    swaps = run(scenario())
    assert unloaded == ["gemma4:12b", "qwen3-vl:8b"]
    assert "lfm3-omni:2b" not in unloaded
    assert swaps == [("qwen3-vl:8b", "gemma4:12b"), ("gemma4:12b", "qwen3-vl:8b")]


# 14. direct_audio=False transcribes via lfm3-omni first and sends clean text to gemma4
def test_direct_audio_false_transcribes_first_and_sends_text_to_chat():
    class _FakeTranscriber:
        async def transcribe(self, _wav: bytes) -> str:
            return "Move Chrome to the left."

    seen_messages: list[list[dict]] = []

    class _CaptureInference:
        async def stream(self, messages, *, operation_id, tools=None):
            seen_messages.append(list(messages))
            yield "Done."

    async def scenario():
        bus = EventBus("s")
        rt = Runtime(
            bus,
            _CaptureInference(),
            MockSpeech(bus, synth_s=0.0, per_char_s=0.0),
            transcriber=_FakeTranscriber(),
            direct_audio=False,
        )
        turn = new_id()
        bus.emit("audio.speech_started", {"mode": "push_to_talk"}, provenance=Provenance.USER, turn_id=turn)
        ended = bus.emit("audio.speech_ended", {"confidence": 1.0}, provenance=Provenance.USER, turn_id=turn)
        rt.submit_speech(turn, b"RIFFfakewav", ended)
        await rt.wait_until_settled()
        return rt

    rt = run(scenario())
    assert rt.state is State.IDLE
    assert len(seen_messages) == 1
    user_msg = seen_messages[0][-1]
    assert user_msg["content"] == "Move Chrome to the left."
    assert "audio" not in user_msg
