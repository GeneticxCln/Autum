"""M5 & R1 stability, bounded-context, audio-retention, and recovery tests (WP3 / WP11)."""

from __future__ import annotations

import asyncio

from autum.app import BubbleSupervisor
from autum.audio import MuteMonitor
from autum.bus import EventBus
from autum.events import Provenance, new_id
from autum.runtime import Mode, Runtime, State
from mocks import MockInference, MockSpeech, Recorder


def run(coro):
    return asyncio.run(coro)


def test_long_session_keeps_context_bounded_and_drops_raw_audio():
    async def scenario():
        bus = EventBus("s")
        speech = MockSpeech(bus, synth_s=0.0, per_char_s=0.0)
        inf = MockInference("A" * 120, token_delay_s=0.0)
        runtime = Runtime(bus, inf, speech, max_history=20, max_context_chars=600)

        for i in range(40):
            turn = new_id()
            bus.emit("audio.speech_started", {"mode": "push_to_talk"}, provenance=Provenance.USER, turn_id=turn)
            ended = bus.emit("audio.speech_ended", {"confidence": 1.0}, provenance=Provenance.USER, turn_id=turn)
            runtime.submit_speech(turn, b"RAW_WAV_BYTES" * 100, ended)
            await runtime.wait_until_settled()
            # Simulate half of transcriptions succeeding and half failing/never arriving
            if i % 2 == 0:
                bus.emit("audio.transcribed", {"text": f"turn {i} " + "x" * 60, "final": True}, provenance=Provenance.USER, turn_id=turn)

        return runtime

    runtime = run(scenario())
    assert len(runtime.history) <= 20
    total_chars = sum(len(str(e.get("content") or "")) for e in runtime.history)
    assert total_chars <= 600
    # Plan §5.3: zero raw audio bytes retained in history after turns complete
    assert all("audio" not in e for e in runtime.history)


def test_model_loading_lifecycle_and_response_recovery_events():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus, synth_s=0.0, per_char_s=0.0)

        class FlakyInference:
            def __init__(self) -> None:
                self.calls = 0

            async def stream(self, messages, *, operation_id: str):
                self.calls += 1
                if self.calls == 1:
                    raise RuntimeError("ollama restarted")
                yield "Back online."

        runtime = Runtime(bus, FlakyInference(), speech, mode=Mode.PUSH_TO_TALK)
        runtime.begin_model_load("lfm3-omni:2b")
        assert runtime.state is State.LOADING
        runtime.finish_model_load("lfm3-omni:2b", 125.4)
        assert runtime.state is State.IDLE

        runtime.submit_text("first try")
        await runtime.wait_until_settled()
        assert runtime.state is State.IDLE

        runtime.submit_text("second try")
        await runtime.wait_until_settled()
        assert runtime.state is State.IDLE
        return log

    log = run(scenario())
    assert log.of("runtime.model_loading")[0].payload == {"model": "lfm3-omni:2b"}
    assert log.of("runtime.model_ready")[0].payload == {"model": "lfm3-omni:2b", "load_ms": 125.4}
    assert log.of("runtime.component_error")[0].payload["component"] == "response"
    assert log.of("runtime.component_recovered")[0].payload == {"component": "response"}


def test_mute_monitor_reports_error_and_recovery():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        reads = iter([False, None, None, None, False])

        async def read():
            try:
                return next(reads)
            except StopIteration:
                return False

        monitor = MuteMonitor(bus, read, interval_s=0.005, error_after=3)
        task = asyncio.create_task(monitor.run())
        await asyncio.sleep(0.06)
        task.cancel()
        return log

    log = run(scenario())
    assert log.of("runtime.component_error")[0].payload["component"] == "mute_monitor"
    assert log.of("runtime.component_recovered")[0].payload == {"component": "mute_monitor"}


def test_bubble_supervisor_restarts_after_crash_and_stops_cleanly():
    class FakeBubble:
        def __init__(self) -> None:
            self.returncode = None
            self._done = asyncio.Event()

        def crash(self, code: int = 1) -> None:
            self.returncode = code
            self._done.set()

        def terminate(self) -> None:
            if self.returncode is None:
                self.returncode = 0
                self._done.set()

        def kill(self) -> None:
            self.terminate()

        async def wait(self) -> int:
            await self._done.wait()
            return self.returncode

    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        spawned: list[FakeBubble] = []

        async def starter():
            p = FakeBubble()
            spawned.append(p)
            return p

        sup = BubbleSupervisor(bus, starter=starter, restart_delay_s=0.01, max_restarts=3)
        await sup.start()
        assert len(spawned) == 1
        spawned[0].crash(139)
        await asyncio.sleep(0.05)
        assert len(spawned) == 2
        await sup.stop()
        return log, spawned

    log, spawned = run(scenario())
    assert log.of("runtime.component_error")[0].payload["component"] == "presence"
    assert log.of("runtime.component_recovered")[0].payload == {"component": "presence"}
    assert spawned[1].returncode == 0
