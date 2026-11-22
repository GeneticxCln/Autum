"""Voice engine audio with fake pw-record / pw-play processes."""

import asyncio
import io
import time
import wave

import pytest

from autum.audio import MIC_HELP, MuteMonitor, SpeechPlayer, VoiceInput, check_microphone
from autum.bus import EventBus
from autum.events import new_id
from mocks import Recorder

CHUNK_100MS_24K = b"\x10\x00" * 2400  # 0.1 s of 24 kHz s16 mono
CHUNK_100MS_16K = b"\x20\x00" * 1600  # 0.1 s of 16 kHz s16 mono


def run(coro):
    return asyncio.run(coro)


class FakeRecordProc:
    def __init__(self, chunks, interval=0.01):
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self._done = asyncio.Event()
        self._feeder = asyncio.create_task(self._feed(chunks, interval))

    async def _feed(self, chunks, interval):
        for chunk in chunks:
            self.stdout.feed_data(chunk)
            await asyncio.sleep(interval)

    def kill(self):
        if self.returncode is None:
            self.returncode = -9
            self._feeder.cancel()
            self.stdout.feed_eof()
            self._done.set()

    async def wait(self):
        await self._done.wait()
        return self.returncode


class FakeStdin:
    def __init__(self):
        self.data = bytearray()
        self.closed = False
        self.first_write = None

    def write(self, chunk):
        self.first_write = self.first_write or time.monotonic()
        self.data += chunk

    async def drain(self):
        pass

    def close(self):
        self.closed = True


class FakePlayProc:
    def __init__(self):
        self.stdin = FakeStdin()
        self.returncode = None

    def kill(self):
        self.returncode = -9

    async def wait(self):
        """Like pw-play: after input closes, it exits once the buffered audio has played."""
        while self.returncode is None and not self.stdin.closed:
            await asyncio.sleep(0.005)
        if self.returncode is None and self.stdin.first_write:
            ends = self.stdin.first_write + len(self.stdin.data) / 48000
            await asyncio.sleep(max(0.0, ends - time.monotonic()))
        self.returncode = self.returncode if self.returncode is not None else 0
        return self.returncode


class Spawner:
    def __init__(self, record_chunks=()):
        self.record_chunks = record_chunks
        self.procs = []

    async def __call__(self, *args, stdin=None, stdout=None):
        assert "--raw" in args  # without it pw-cat reads/writes a container header, not samples
        proc = FakeRecordProc(self.record_chunks) if args[0] == "pw-record" else FakePlayProc()
        self.procs.append((args[0], proc))
        return proc


class FakeRuntime:
    def __init__(self):
        self.submitted = []

    def submit_speech(self, turn_id, audio, cause=None):
        self.submitted.append((turn_id, audio))


class FakeTranscriber:
    async def transcribe(self, wav):
        return "what time is it"


class FakeSynth:
    def __init__(self, chunks_per_text=2, fail=False):
        self.chunks_per_text = chunks_per_text
        self.fail = fail

    async def stream(self, text):
        if self.fail:
            raise RuntimeError("synth broke")
        for _ in range(self.chunks_per_text):
            await asyncio.sleep(0.001)
            yield CHUNK_100MS_24K


# ----- microphone -------------------------------------------------------


def test_a_push_to_talk_turn_is_captured_submitted_and_transcribed_after_the_reply():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime = FakeRuntime()
        voice = VoiceInput(bus, runtime, FakeTranscriber(), spawn_=Spawner([CHUNK_100MS_16K] * 5))
        turn = await voice.start()
        await asyncio.sleep(0.1)
        await voice.finish()
        bus.emit("conversation.user_message", {"text": "", "source": "speech"}, turn_id=turn)
        op = new_id()
        bus.fence.open(op)
        bus.emit("conversation.response_finished", {"text": "It is noon."}, turn_id=turn, operation_id=op)
        await asyncio.sleep(0.05)
        return log, runtime, turn

    log, runtime, turn = run(scenario())
    assert [t for t in log.types() if t.startswith("audio.")] == [
        "audio.capture_started", "audio.speech_started", "audio.capture_stopped",
        "audio.speech_ended", "audio.transcribed",
    ]
    (submitted_turn, wav), = runtime.submitted
    assert submitted_turn == turn
    with wave.open(io.BytesIO(wav)) as w:
        assert (w.getframerate(), w.getnchannels(), w.getnframes()) == (16000, 1, 8000)
    transcript = log.of("audio.transcribed")[0]
    assert transcript.payload == {"text": "what time is it", "final": True} and transcript.turn_id == turn


def test_a_tap_too_short_to_be_speech_is_dropped():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime = FakeRuntime()
        voice = VoiceInput(bus, runtime, FakeTranscriber(), spawn_=Spawner([CHUNK_100MS_16K]))
        await voice.start()
        await asyncio.sleep(0.03)
        await voice.finish()
        return log, runtime

    log, runtime = run(scenario())
    assert runtime.submitted == []
    assert log.of("audio.transcribed")[0].payload == {"text": "", "final": True}


def test_a_silent_microphone_stops_the_turn_with_instructions():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        voice = VoiceInput(bus, FakeRuntime(), FakeTranscriber(), spawn_=Spawner([]), stall_s=0.1)
        start = time.monotonic()
        await voice.start()
        while not log.of("audio.transcribed") and time.monotonic() - start < 2:
            await asyncio.sleep(0.01)
        return log, voice, time.monotonic() - start

    log, voice, waited = run(scenario())
    assert not voice.capturing and waited < 1
    error = log.of("runtime.component_error")[0].payload
    assert error == {"component": "microphone", "error": MIC_HELP}
    assert log.of("audio.transcribed")[0].payload["text"] == ""


def test_muting_blocks_new_turns_and_drops_the_current_one():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime = FakeRuntime()
        voice = VoiceInput(bus, runtime, FakeTranscriber(), spawn_=Spawner([CHUNK_100MS_16K] * 50))
        await voice.start()
        await asyncio.sleep(0.05)
        bus.emit("audio.mute_changed", {"muted": True})
        await asyncio.sleep(0.05)
        refused = await voice.start()
        return log, runtime, voice, refused

    log, runtime, voice, refused = run(scenario())
    assert refused is None and not voice.capturing and runtime.submitted == []
    assert log.of("audio.transcribed")[0].payload["text"] == ""


def test_microphone_check():
    async def scenario():
        ok = await check_microphone(Spawner([CHUNK_100MS_16K]), timeout_s=0.5)
        start = time.monotonic()
        silent = await check_microphone(Spawner([]), timeout_s=0.1)
        return ok, silent, time.monotonic() - start

    ok, silent, waited = run(scenario())
    assert ok is True and silent is False and waited < 1


# ----- speech output ----------------------------------------------------


def player_setup(synth):
    bus = EventBus("s")
    log = Recorder(bus)
    spawner = Spawner()
    player = SpeechPlayer(bus, synth, spawn_=spawner)
    op = new_id()
    bus.fence.open(op)
    return bus, log, spawner, player, op


def test_speech_plays_in_real_time_and_reports_start_levels_and_finish():
    async def scenario():
        bus, log, spawner, player, op = player_setup(FakeSynth(chunks_per_text=2))
        start = time.monotonic()
        player.enqueue("First sentence.", operation_id=op, turn_id="t")
        player.enqueue("Second sentence.", operation_id=op, turn_id="t")
        await player.drain(op)
        return log, spawner, time.monotonic() - start

    log, spawner, took = run(scenario())
    (name, proc), = spawner.procs
    assert name == "pw-play" and len(proc.stdin.data) == 4 * len(CHUNK_100MS_24K) and proc.stdin.closed
    assert 0.35 <= took < 0.7  # drain returns when the 0.4 s of audio has played
    assert log.types()[0] == "speech.playback_started"
    assert log.of("speech.playback_finished")[0].payload == {"reason": "completed"}
    assert len(log.of("speech.output_level")) >= 3


def test_stop_silences_at_once_and_returns_only_what_was_heard():
    async def scenario():
        bus, log, spawner, player, op = player_setup(FakeSynth(chunks_per_text=3))  # 0.3 s each
        for text in ("One.", "Two.", "Three."):
            player.enqueue(text, operation_id=op, turn_id="t")
        await asyncio.sleep(0.4)
        bus.fence.begin_cancel(op)
        worker = player._worker
        heard = player.stop()
        written = len(spawner.procs[0][1].stdin.data)
        await asyncio.sleep(0.2)
        assert worker.cancelled()  # ended cleanly, no error raised while cancelling
        return log, spawner, heard, written

    log, spawner, heard, written = run(scenario())
    proc = spawner.procs[0][1]
    assert heard == "One. Two."
    assert proc.returncode == -9 and len(proc.stdin.data) == written  # nothing written after stop
    assert log.of("speech.playback_finished")[0].payload == {"reason": "cancelled"}


def test_a_synthesis_failure_surfaces_at_drain():
    async def scenario():
        bus, log, spawner, player, op = player_setup(FakeSynth(fail=True))
        player.enqueue("Hello.", operation_id=op, turn_id="t")
        with pytest.raises(RuntimeError, match="synth broke"):
            await player.drain(op)

    run(scenario())


# ----- mute -------------------------------------------------------------


def test_mute_monitor_reports_changes_only():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        states = iter([False, False, None, True, True, False])

        async def read():
            return next(states)

        monitor = MuteMonitor(bus, read, interval_s=0.001)
        task = asyncio.create_task(monitor.run())
        await asyncio.sleep(0.05)
        task.cancel()
        return log

    log = run(scenario())
    assert [e.payload["muted"] for e in log.of("audio.mute_changed")] == [False, True, False]
