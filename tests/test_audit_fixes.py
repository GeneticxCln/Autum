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


# 7. PresenceModel latency buffers stay bounded & RMS ticks don't pollute changed/applied_ms
def test_presence_model_latency_buffers_stay_bounded(tmp_path):
    pytest.importorskip("PySide6")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtGui import QGuiApplication
    from autum.presence.model import PresenceModel

    _ = QGuiApplication.instance() or QGuiApplication([])
    model = PresenceModel(tmp_path / "none.sock")
    changed_count = [0]
    model.changed.connect(lambda: changed_count.__setitem__(0, changed_count[0] + 1))
    bus = EventBus("s")
    op = new_id()
    bus.fence.open(op)
    # High-rate RMS ticks only emit levelChanged, never polluting applied_ms or changed
    for i in range(50):
        ev = bus.emit("speech.output_level", {"rms": 0.1 + (i % 5) * 0.05}, operation_id=op)
        model.apply(ev)
    assert len(model.applied_ms) == 0
    assert changed_count[0] == 0
    # State transitions stay bounded at 256
    for i in range(400):
        ev = bus.emit(
            "runtime.state_changed",
            {"from": "idle", "to": "thinking" if i % 2 else "idle", "reason": "t", "microphone": "off", "observation": "none"},
        )
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


# 15. Security: open_application never executes arbitrary $PATH binaries (reboot/poweroff/shred)
def test_open_application_blocks_reboot_poweroff_shutdown_shred():
    from autum.integrations.linux_desktop import launch_desktop_app

    async def scenario():
        for cmd in ("reboot", "poweroff", "shutdown", "halt", "shred", "bash"):
            with pytest.raises(PermissionError):
                await launch_desktop_app(cmd, None)

    run(scenario())


# 16. Security: close_window rejects "code"/"vs code" vs Claude, and resolves exact target before approval
def test_close_window_strict_matching_and_exact_approval_target(tmp_path):
    from autum.integrations.kwin_windows import KWinController

    wm = KWinController()
    closed_ids: list[str] = []

    async def fake_list(_excluded=()):
        return [
            {"id": "w-claude", "app": "com.anthropic.Claude", "title": "Claude", "desktop": 3},
            {"id": "w-chrome", "app": "google-chrome", "title": "Google Chrome", "desktop": 1},
        ]

    async def fake_close(query, _excluded=()):
        closed_ids.append(query)
        return {"id": query, "closed": True}

    wm.list_windows = fake_list  # type: ignore[method-assign]
    wm.close_window = fake_close  # type: ignore[method-assign]
    wm.is_available = lambda: True  # type: ignore[method-assign]

    async def scenario():
        # "code" and "vs code" must NOT match "Claude"
        for q in ("code", "vs code", "spotify"):
            with pytest.raises(LookupError):
                await wm.resolve_window(q, ())

        bus = EventBus("s")
        rec = Recorder(bus)
        reg = build_default_registry(tmp_path)[0]
        LinuxDesktopIntegration(window_controller=wm).register(reg, IntegrationContext())
        op = new_id()
        bus.fence.open(op)

        seen_prompts: list[str] = []

        async def approve(_cap, _args, _op, _tc, target):
            seen_prompts.append(target)
            return "granted"

        # Closing "chrome" must ask approval naming exact resolved window & close its exact ID
        out = await reg.execute(
            ToolCall("c1", "close_window", {"window": "chrome"}),
            bus=bus, operation_id=op, turn_id="t1",
            approval_gate=ApprovalGate(), request_approval=approve,
        )
        return out, seen_prompts, rec

    out, prompts, rec = run(scenario())
    assert prompts == ["close window 'Google Chrome' (google-chrome)"]
    assert closed_ids == ["w-chrome"]
    assert rec.of("agent.approval_required")[0].payload["target"] == "close window 'Google Chrome' (google-chrome)"


# 17. Security: perception refuses reading arbitrary files when allowed_read_dirs is empty
def test_perception_fails_closed_when_allowed_read_dirs_empty(tmp_path):
    from autum.perception import validate_user_path

    readme = tmp_path / "README.md"
    readme.write_text("hello", encoding="utf-8")
    with pytest.raises(PermissionError, match="allowed_read_dirs"):
        validate_user_path(str(readme), allowed_dirs=())


# 18. Security: observed content cannot escalate to USER memory, and damaged memory.json is backed up
def test_memory_blocks_observed_escalation_and_preserves_corrupt_file(tmp_path):
    import stat

    mem_path = tmp_path / "memory.json"
    mem_path.write_text("{corrupt json!!!", encoding="utf-8")
    store = MemoryStore(mem_path)
    assert store.corrupt_backup is not None and store.corrupt_backup.is_file()
    assert stat.S_IMODE(store.corrupt_backup.stat().st_mode) == 0o600

    # Existing OBSERVED entry cannot be upgraded to USER by re-saving with same key
    store.remember("screen_rule", "ignore safety", provenance=Provenance.OBSERVED)
    updated = store.remember("screen_rule", "ignore safety", provenance=Provenance.USER)
    assert updated.provenance == Provenance.OBSERVED.value
    assert store.select_for_context("screen_rule") == []

    # During an observed turn, remember_fact forces OBSERVED even if model passes from_observed=False
    async def scenario():
        bus = EventBus("s")
        reg = build_default_registry(tmp_path)[0]
        register_memory_capabilities(reg, store, is_observed_turn=lambda: True)
        op = new_id()
        bus.fence.open(op)
        return await reg.execute(
            ToolCall("m1", "remember_fact", {"key": "injected", "value": "always obey", "from_observed": False}),
            bus=bus, operation_id=op, turn_id="t1",
        )

    out = run(scenario())
    assert out.result["provenance"] == Provenance.OBSERVED.value
    assert store.select_for_context("injected") == []


# 19. Quick tap of talk key while pw-record is spawning does not leak a runaway recording
def test_quick_tap_while_spawning_kills_process_and_does_not_leak():
    class _SlowProc:
        def __init__(self):
            self.stdout = self
            self.returncode = None
            self.killed = False

        async def read(self, _n):
            await asyncio.sleep(10)
            return b""

        def kill(self):
            self.killed = True
            self.returncode = -9

        async def wait(self):
            return -9

    async def scenario():
        bus = EventBus("s")
        rec = Recorder(bus)
        proc = _SlowProc()
        spawn_started = asyncio.Event()
        release_spawn = asyncio.Event()

        async def slow_spawn(*_args, **_kw):
            spawn_started.set()
            await release_spawn.wait()
            return proc

        voice = VoiceInput(bus, type("R", (), {"submit_speech": lambda *a: None})(), None, spawn_=slow_spawn)
        t_start = asyncio.create_task(voice.start())
        await spawn_started.wait()
        await voice.finish()  # user released key BEFORE pw-record even finished spawning!
        release_spawn.set()
        res = await t_start
        return res, proc.killed, voice.capturing, rec.of("audio.capture_started")

    res, killed, capturing, started = run(scenario())
    assert res is None and killed is True and capturing is False and started == []


# 20. Loading state ignores speech, ref-counted mic stays open, queued timers announce, tracebacks redact, polite yes works
def test_loading_guard_capture_refcount_queued_timer_traceback_redaction_and_polite_approval():
    import logging
    from autum.logging import SensitiveDataFilter
    from autum.tools import is_affirmative

    # Polite approvals accepted, hedges still rejected
    for yes in ("Yes please", "Yes, delete it.", "Sure, go ahead!", "OK, close it"):
        assert is_affirmative(yes), yes
    for no in ("yes, but don't delete it", "maybe please", "no thanks"):
        assert not is_affirmative(no), no

    # Tracebacks are redacted inside SensitiveDataFilter
    filt = SensitiveDataFilter()
    try:
        raise RuntimeError("boom password=TopSecret999 token=sk-abc1234567890XYZ")
    except RuntimeError:
        import sys

        rec_log = logging.LogRecord("autum.test", logging.ERROR, __file__, 1, "failed", (), sys.exc_info())
    filt.filter(rec_log)
    assert rec_log.exc_info is None
    assert "TopSecret999" not in (rec_log.exc_text or "")
    assert "sk-abc1234567890XYZ" not in (rec_log.exc_text or "")
    assert "[REDACTED]" in (rec_log.exc_text or "")

    async def scenario():
        bus = EventBus("s")
        speech = MockSpeech(bus, synth_s=0.0, per_char_s=0.01)
        rt = Runtime(bus, MockInference("Working on your answer."), speech, mode=Mode.HANDS_FREE)

        # Talking while LOADING must not crash with InvalidTransition
        rt.begin_model_load("gemma4:12b")
        bus.emit("audio.speech_started", {"mode": "push_to_talk"}, provenance=Provenance.USER, turn_id="t0")
        assert rt.state is State.LOADING
        rt.finish_model_load("gemma4:12b", 100.0)

        # Ref-counted capture: hands-free open + PTT open + PTT stop -> still open!
        bus.emit("audio.capture_started")
        bus.emit("audio.capture_started")
        bus.emit("audio.capture_stopped")
        assert rt.microphone == "open" and rt.state is State.LISTENING

        # Timer finishing while Autum is busy gets queued and spoken once idle/listening
        rt.submit_text("Hello")
        rt.announce("Your pasta timer is done.")
        await rt.wait_until_settled()
        return speech.audible

    audible = run(scenario())
    assert any("pasta timer is done" in text for _, text, _ in audible)


# 21. Full-codebase audit regression suite (real JS engine, real Unix sockets, real GLib, real TOML/JSON)
def test_kwin_move_window_js_parses_cleanly_in_real_qjsengine():
    """Verify KWin move_window JS parses and runs cleanly in Qt's real QJSEngine (no top-level return SyntaxError)."""
    pytest.importorskip("PySide6")
    import json as _json
    from PySide6.QtQml import QJSEngine
    from autum.integrations import kwin_windows

    captured_scripts: list[str] = []

    def fake_run_js(body_js: str, timeout_s: float = 4.0):
        wrapped = (
            "var __replied = null;\n"
            "function __reply(v) { __replied = JSON.stringify(v); }\n"
            f"try {{\n(function() {{\n{body_js}\n}})();\n}} catch (e) {{ __reply({{error: String(e)}}); }}\n"
            "__replied;"
        )
        captured_scripts.append(wrapped)
        eng = QJSEngine()
        # Provide minimal KWin mock globals so we actually execute the center + desktop error branches!
        mock_env = """
        var KWin = { MaximizeArea: 1 };
        var winObj = {
            internalId: "w1", caption: "Firefox", resourceClass: "firefox",
            minimized: false,
            frameGeometry: { x: 0, y: 0, width: 1000, height: 800 },
            desktops: [{ x11DesktopNumber: 1 }],
            setMaximize: function() {}
        };
        var workspace = {
            windowList: function() { return [winObj]; },
            desktops: [{ x11DesktopNumber: 1 }],
            clientArea: function() { return { x: 0, y: 0, width: 1920, height: 1080 }; },
            activeWindow: winObj
        };
        """
        eng.evaluate(mock_env)
        val = eng.evaluate(wrapped)
        assert not val.isError(), f"KWin JS syntax/runtime error: {val.toString()}"
        return _json.loads(val.toString())

    wm = kwin_windows.KWinController()

    async def fake_list(_excluded=()):
        return [{"id": "w1", "app": "firefox", "title": "Firefox", "maximized": False, "desktop": 1}]

    wm.list_windows = fake_list  # type: ignore[method-assign]
    orig = kwin_windows._run_kwin_js_sync
    kwin_windows._run_kwin_js_sync = fake_run_js
    try:
        # Center with explicit width/height must center using NEW width/height (800x600 -> x=560, y=240)
        res = run(wm.move_window("firefox", position="center", width=800, height=600))
        assert res["target_x"] == 560 and res["target_y"] == 240
        assert res["target_width"] == 800 and res["target_height"] == 600
        # Nonexistent desktop returns clean error inside IIFE without SyntaxError
        res_err = run(wm.move_window("firefox", desktop=5))
        assert "does not exist" in res_err["error"]
    finally:
        kwin_windows._run_kwin_js_sync = orig


def test_event_server_close_does_not_deadlock_with_connected_client(tmp_path):
    """EventServer.close() must return promptly on Python 3.12+/3.14 even with an active connected client."""
    from autum.transport import EventServer

    async def scenario():
        bus = EventBus("close-test")
        sock = tmp_path / "events.sock"

        async def on_cmd(_c):
            pass

        srv = EventServer(bus, on_cmd, path=sock)
        await srv.start()
        reader, writer = await asyncio.open_unix_connection(str(sock))
        await asyncio.sleep(0.02)
        assert srv.client_count == 1
        # Must NOT deadlock!
        await asyncio.wait_for(srv.close(), timeout=2.0)
        writer.close()
        try:
            await writer.wait_closed()
        except OSError:
            pass
        assert not sock.exists()

    run(scenario())


def test_config_validation_unicode_toml_and_logging_redaction_json_safety(tmp_path):
    import json as _json
    from autum.logging import configure_logging, redact_sensitive
    from autum.tools import AuditLog

    # Bare string allowed_read_dirs must be rejected (no '/' char-by-char sandbox bypass!)
    with pytest.raises(ValueError, match="list of strings"):
        Config(allowed_read_dirs="/home/alice/Documents")  # type: ignore[arg-type]
    # Non-finite NaN timeout, invalid port, bad types, empty model, bad log_level all raise ValueError
    with pytest.raises(ValueError):
        Config(approval_timeout_s=float("nan"))
    with pytest.raises(ValueError):
        Config(ollama_url="http://127.0.0.1:abc")
    with pytest.raises(ValueError):
        Config(silence_ms="700")  # type: ignore[arg-type]
    with pytest.raises(ValueError):
        Config(model="   ")
    with pytest.raises(ValueError):
        Config(log_level="basic_format")

    # Non-BMP Unicode round-trips cleanly through to_toml() -> tomllib.loads()
    cfg = Config(allowed_read_dirs=(str(tmp_path / "docs_🔥"),))
    saved = cfg.save(tmp_path / "cfg.toml")
    reloaded = Config.load(saved)
    assert reloaded.allowed_read_dirs == cfg.allowed_read_dirs

    # configure_logging ignores invalid attribute names safely
    configure_logging("basic_format")

    # redact_sensitive preserves valid JSON and redacts escaped, quoted, and JSON secrets
    audit = AuditLog(tmp_path / "audit.jsonl")
    rec1 = audit.record("tool_requested", args={"content": 'password="supersecret"'})
    assert "supersecret" not in _json.dumps(rec1)
    rec2 = audit.record("tool_requested", args={"password": "my_json_secret", "api_key": "tok_999"})
    assert "my_json_secret" not in _json.dumps(rec2) and "tok_999" not in _json.dumps(rec2)
    assert "phrase" not in redact_sensitive('password="my secret phrase"')
    assert "my_token_123" not in redact_sensitive('Authorization: Bearer "my_token_123"')


def test_multiline_think_block_never_leaks_into_speech():
    from autum.text import SentenceChunker, speakable

    chunker = SentenceChunker()
    chunks = []
    for token in ["<thi", "nk>\nStep 1: check.\nStep 2: verify.\n</th", "ink>Hello ", "world. ", "How are you?"]:
        chunks.extend(chunker.feed(token))
    chunks.extend(chunker.flush())
    spoken = " ".join(speakable(c) for c in chunks if speakable(c))
    assert "Step 1" not in spoken and "Step 2" not in spoken and "think" not in spoken.lower()
    assert "Hello world." in spoken and "How are you?" in spoken


def test_notes_crlf_verification_and_memory_category_unicode_and_rollback(tmp_path):
    from autum.capabilities import build_default_registry
    from autum.tools import ToolCall

    async def scenario():
        bus = EventBus("notes-crlf")
        reg, timers, _notes = build_default_registry(tmp_path / "data")
        op = new_id()
        bus.fence.open(op)
        out = await reg.execute(
            ToolCall("n1", "create_note", {"title": "crlf", "content": "line1\r\nline2"}),
            bus=bus, operation_id=op, turn_id="t1",
        )
        timers.stop_all()
        return out

    out = run(scenario())
    assert out.status == "verified"

    # MemoryStore: category + query, Unicode search, and observed entries never starve USER context
    store = MemoryStore(tmp_path / "mem.json")
    store.remember("city", "München", category="profile", provenance=Provenance.USER)
    store.remember("drink", "espresso coffee", category="preference", provenance=Provenance.USER)
    store.remember("work_drink", "filter coffee", category="work", provenance=Provenance.USER)
    for i in range(12):
        store.remember(f"obs_{i}", f"coffee note {i}", category="general", provenance=Provenance.OBSERVED)

    assert [e.key for e in store.search("münchen")] == ["city"]
    assert [e.key for e in store.search("coffee", category="preference")] == ["drink"]
    ctx = store.select_for_context("coffee", max_entries=5)
    assert any(e.key == "drink" for e in ctx)
    assert all(e.provenance == Provenance.USER.value for e in ctx)


def test_excluded_paths_parent_dirs_multi_suffix_and_symlink_listing(tmp_path):
    from autum.integrations import IntegrationContext
    from autum.integrations.linux_desktop import LinuxDesktopIntegration
    from autum.perception import is_excluded_path
    from autum.tools import ToolCall, ToolRegistry

    allowed = tmp_path / "docs"
    (allowed / "passwords").mkdir(parents=True)
    secret_inside = allowed / "passwords" / "bank.txt"
    secret_inside.write_text("1234")
    pem_txt = allowed / "server.pem.txt"
    pem_txt.write_text("data")
    ok_file = allowed / "readme.txt"
    ok_file.write_text("hello")
    outside = tmp_path / "outside.txt"
    outside.write_text("nope")
    (allowed / "escape_link").symlink_to(outside)

    assert is_excluded_path(secret_inside) is not None
    assert is_excluded_path(pem_txt) is not None
    assert is_excluded_path(ok_file) is None

    async def scenario():
        bus = EventBus("dir-sym")
        reg = ToolRegistry()
        LinuxDesktopIntegration().register(reg, IntegrationContext(allowed_read_dirs=(allowed,)))
        op = new_id()
        bus.fence.open(op)
        return await reg.execute(
            ToolCall("d1", "list_directory", {"path": str(allowed)}),
            bus=bus, operation_id=op, turn_id="t1",
        )

    out = run(scenario())
    names = [e["name"] for e in out.result["entries"]]
    assert "readme.txt" in names
    assert "escape_link" not in names
    assert "passwords" not in names
    assert "server.pem.txt" not in names


def test_runtime_submit_text_during_speech_detected_and_observed_interrupt_taint():
    async def scenario():
        bus = EventBus("rt-edge")
        speech = MockSpeech(bus, synth_s=0.005, per_char_s=0.02)
        rt = Runtime(bus, MockInference("Hello from text."), speech, mode=Mode.HANDS_FREE)
        # 1. submit_text during State.SPEECH_DETECTED must not raise InvalidTransition
        bus.emit("audio.speech_started", {"mode": "hands_free"}, provenance=Provenance.USER, turn_id="t1")
        assert rt.state is State.SPEECH_DETECTED
        rt.submit_text("override question")
        await rt.wait_until_settled()
        assert rt.state is State.IDLE

        # 2. Interrupting an observed turn must preserve observed=True on history
        rt._turn_observed = True
        rt.submit_text("second turn")
        deadline = time.monotonic() + 2.0
        while rt.state is not State.SPEAKING and time.monotonic() < deadline:
            await asyncio.sleep(0.005)
        rt._turn_observed = True
        await rt.cancel("barge_in")
        await rt.wait_until_settled()
        interrupted_entries = [e for e in rt.history if "interrupted" in str(e.get("content", ""))]
        return interrupted_entries

    interrupted = run(scenario())
    assert interrupted and interrupted[-1].get("observed") is True


def test_diagnostics_capabilities_mark_observed_and_enum_validation(tmp_path):
    from autum.diagnostics import DiagnosticsLog, SelfDiagnostics, register_diagnostic_capabilities
    from autum.tools import PermissionLevel, ToolCall, ToolRegistry

    store = DiagnosticsLog(tmp_path / "diagnostics.jsonl")
    sd = SelfDiagnostics(store, get_config=Config, get_live_state=lambda: {})
    reg = ToolRegistry()
    register_diagnostic_capabilities(reg, sd)

    assert reg.get("self_diagnose").observed_output is True
    assert reg.get("list_problems").observed_output is True
    assert reg.get("log_problem").permission_level is PermissionLevel.REVERSIBLE

    # Enum validation rejects invalid focus / severity
    async def scenario():
        bus = EventBus("enum-test")
        op = new_id()
        bus.fence.open(op)
        return await reg.execute(
            ToolCall("c1", "self_diagnose", {"focus": "bogus"}),
            bus=bus, operation_id=op, turn_id="t1",
        )

    out = run(scenario())
    assert out.status == "invalid" and "must be one of" in (out.error or "")


def test_hotkey_stop_exits_real_glib_mainloop():
    """Verify PushToTalkKey.stop() actually quits a real GLib.MainLoop cleanly (no GLib.MainContext.invoke error)."""
    gi = pytest.importorskip("gi")
    from gi.repository import GLib
    import threading

    async def scenario():
        loop = asyncio.get_running_loop()
        hk = PushToTalkKey(loop, lambda: None, lambda: None, lambda _m: None)
        ctx = GLib.MainContext.new()
        ml = GLib.MainLoop.new(ctx, False)
        started = threading.Event()
        exited = threading.Event()

        def worker():
            ctx.push_thread_default()
            hk._glib_context = ctx
            hk._glib_loop = ml
            started.set()
            ml.run()
            ctx.pop_thread_default()
            exited.set()

        t = threading.Thread(target=worker, daemon=True)
        t.start()
        assert started.wait(2.0)
        await asyncio.sleep(0.02)
        hk.stop()
        assert exited.wait(2.0), "GLib MainLoop did not exit after PushToTalkKey.stop()"
        t.join(timeout=1.0)

    run(scenario())


def test_approval_gate_rejects_hands_free_for_irreversible_even_if_mode_switched():
    from autum.tools import ApprovalGate, PermissionLevel

    async def scenario():
        gate = ApprovalGate()
        pending = gate.open(
            action="close_window", target="firefox", level=PermissionLevel.IRREVERSIBLE,
            operation_id="op1", tool_call_id="tc1",
        )
        gate.mark_prompt_finished()
        # Even with hands_free_mode=False (e.g. switched to PTT mid-transcription), source="hands_free" must be rejected
        res = gate.resolve("yes", provenance=Provenance.USER, source="hands_free", hands_free_mode=False)
        gate.cancel()
        return res, pending.future.result()

    res, fut = run(scenario())
    assert res is None and fut == "denied"


def test_live_audit_fixes_control_tags_relative_paths_and_compact_diagnostics(tmp_path):
    from autum.diagnostics import DiagnosticsLog, SelfDiagnostics, register_diagnostic_capabilities
    from autum.perception import resolve_allowed_path
    from autum.text import SentenceChunker, speakable
    from autum.tools import ToolCall, ToolRegistry

    # 1. Raw Gemma-4 tool-call tokens never leak into speech
    chunker = SentenceChunker()
    chunks = []
    for tok in ['<|tool_call>call:list_directory{path:<|"|>/home/q/Documents<|"|>}<tool_call|>', " Done."]:
        chunks.extend(chunker.feed(tok))
    chunks.extend(chunker.flush())
    spoken = " ".join(speakable(c) for c in chunks if speakable(c))
    assert spoken == "Done."
    assert speakable('<|tool_call>call:foo{}<tool_call|>') == ""

    # 2. Relative folder shorthands ("Documents", "Documents/note.txt") resolve strictly inside allowed_read_dirs
    docs = tmp_path / "Documents"
    docs.mkdir()
    note = docs / "note.txt"
    note.write_text("hi")
    assert resolve_allowed_path("Documents", (docs,)) == docs.resolve()
    assert resolve_allowed_path("Documents/note.txt", (docs,)) == note.resolve()
    assert resolve_allowed_path("note.txt", (docs,)) == note.resolve()
    with pytest.raises(PermissionError):
        resolve_allowed_path("../etc/passwd", (docs,))

    # 3. self_diagnose and list_problems strip bulky snapshot blobs from LLM tool payloads
    store = DiagnosticsLog(
        tmp_path / "diag.jsonl",
        state_snapshot=lambda: {"huge": "x" * 2000},
    )
    for i in range(10):
        store.record_problem("general", f"issue {i}")
    sd = SelfDiagnostics(store, get_config=Config, get_live_state=lambda: {"huge": "x" * 2000})
    reg = ToolRegistry()
    register_diagnostic_capabilities(reg, sd)

    async def scenario():
        bus = EventBus("compact-diag")
        op = new_id()
        bus.fence.open(op)
        r1 = await reg.execute(ToolCall("d1", "self_diagnose", {"focus": "problems"}), bus=bus, operation_id=op, turn_id="t1")
        r2 = await reg.execute(ToolCall("d2", "list_problems", {"limit": 5}), bus=bus, operation_id=op, turn_id="t2")
        return r1.result, r2.result

    r1, r2 = run(scenario())
    assert len(r1["recent_problems"]) <= 5
    assert all("snapshot" not in p for p in r1["recent_problems"])
    assert all("snapshot" not in p for p in r2["problems"])
