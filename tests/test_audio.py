"""Voice engine audio with fake pw-record / pw-play processes."""

import asyncio
import io
import time
import wave

import pytest

from autum.audio import (
    MIC_HELP,
    HandsFreeInput,
    MuteMonitor,
    SpeechPlayer,
    VoiceInput,
    check_microphone,
    highpass_pcm,
    rms,
)
from autum.bus import EventBus
from autum.events import new_id
from autum.runtime import Mode, Runtime, State
from autum.vad import VAD_FRAME_BYTES, TurnDetector
from mocks import MockInference, MockSpeech, Recorder

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


# ----- hands-free mode (M3) ---------------------------------------------


def _hf_frame(tag: int) -> bytes:
    return bytes([tag & 0xFF]) * VAD_FRAME_BYTES


def _hf_detector(**kwargs) -> TurnDetector:
    defaults = dict(min_speech_ms=96, barge_in_ms=160, silence_ms=160, pre_roll_ms=96, tail_ms=32)
    defaults.update(kwargs)
    return TurnDetector(lambda pcm: 0.9 if pcm[0] == 1 else 0.05, **defaults)


class StreamingRecordProc:
    """Fake pw-record whose stream can be pushed frame-by-frame by a test."""

    def __init__(self, initial=()):
        self.stdout = asyncio.StreamReader()
        self.returncode = None
        self._done = asyncio.Event()
        for chunk in initial:
            self.stdout.feed_data(chunk)

    def push(self, *frames: bytes) -> None:
        for f in frames:
            self.stdout.feed_data(f)

    def eof(self) -> None:
        if self.returncode is None:
            self.returncode = 1
            self.stdout.feed_eof()
            self._done.set()

    def kill(self) -> None:
        if self.returncode is None:
            self.returncode = -9
            self.stdout.feed_eof()
            self._done.set()

    async def wait(self) -> int:
        await self._done.wait()
        return self.returncode


def test_hands_free_completes_a_turn_and_returns_to_listening():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus)
        runtime = Runtime(bus, MockInference("Hello back."), speech, mode=Mode.HANDS_FREE)
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(bus, runtime, FakeTranscriber(), _hf_detector(), spawn_=spawn_)
        await hf.start()
        await asyncio.sleep(0.02)
        assert runtime.state is State.LISTENING

        # 2 silence + 12 speech (~384 ms > MIN_UTTERANCE_S) + 5 silence -> full turn
        proc.push(*([_hf_frame(0)] * 2 + [_hf_frame(1)] * 12 + [_hf_frame(0)] * 5))
        deadline = time.monotonic() + 2.0
        while not log.of("audio.transcribed") and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()
        end_state = runtime.state
        end_mic = runtime.microphone
        await hf.stop()
        return log, runtime, end_state, end_mic

    log, runtime, end_state, end_mic = run(scenario())
    assert end_state is State.LISTENING and end_mic == "open"
    assert [e.payload["mode"] for e in log.of("audio.speech_started")] == ["hands_free"]
    assert log.of("audio.transcribed")[0].payload == {"text": "what time is it", "final": True}
    assert runtime.history[0]["content"] == "what time is it"
    assert runtime.history[1]["content"].strip() == "Hello back."


def test_hands_free_barge_in_cancels_active_speech_and_answers_new_turn():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus, synth_s=0.005, per_char_s=0.02)
        inf = MockInference("First sentence. Second sentence. Third sentence.", token_delay_s=0.002)
        runtime = Runtime(bus, inf, speech, mode=Mode.HANDS_FREE)
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(bus, runtime, FakeTranscriber(), _hf_detector(), spawn_=spawn_)
        await hf.start()
        await asyncio.sleep(0.02)

        # Trigger an initial reply so Autum is actively SPEAKING
        runtime.submit_text("tell me a long story")
        deadline = time.monotonic() + 2.0
        while runtime.state is not State.SPEAKING and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert runtime.state is State.SPEAKING

        # Short backchannel (3 frames < 5 barge_in frames): must NOT interrupt
        proc.push(*([_hf_frame(1)] * 3 + [_hf_frame(0)] * 3))
        await asyncio.sleep(0.03)
        assert runtime.state is State.SPEAKING
        assert log.of("operation.cancel_requested") == []

        # Sustained user speech (12 frames >= 5 barge_in frames): interrupts via barge_in!
        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 5))
        deadline = time.monotonic() + 2.0
        while len(log.of("audio.transcribed")) < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()
        await hf.stop()
        return log

    log = run(scenario())
    cancels = log.of("operation.cancel_requested")
    assert len(cancels) == 1 and cancels[0].payload["source"] == "barge_in"
    assert len(log.of("operation.cancelled")) == 1
    assert len(log.of("audio.speech_started")) == 1


def test_hands_free_speaker_safe_mode_never_cuts_off_own_voice_from_speakers():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus, synth_s=0.005, per_char_s=0.015)
        inf = MockInference("Hello! I am speaking a complete sentence out of your speakers.", token_delay_s=0.001)
        runtime = Runtime(bus, inf, speech, mode=Mode.HANDS_FREE)
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(
            bus,
            runtime,
            FakeTranscriber(),
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            post_speech_cooldown_s=0.06,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        runtime.submit_text("say hello")
        deadline = time.monotonic() + 2.0
        while runtime.state is not State.SPEAKING and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert runtime.state is State.SPEAKING

        # Simulate loud speaker bleed into the microphone while Autum is speaking
        proc.push(*([_hf_frame(1)] * 25))
        await runtime.wait_until_settled()
        await asyncio.sleep(0.08)
        await hf.stop()
        return log, runtime

    log, runtime = run(scenario())
    assert log.of("operation.cancel_requested") == []
    assert log.of("audio.speech_started") == []
    assert runtime.history[-1]["content"].strip() == "Hello! I am speaking a complete sentence out of your speakers."


def test_hands_free_smart_barge_in_ignores_speaker_echo_and_interrupts_on_wake_question():
    class ConfigurableTranscriber:
        def __init__(self, text: str):
            self.text = text

        async def transcribe(self, _wav: bytes) -> str:
            return self.text

    # 1. Speaker echo matching what Autum is currently saying -> ignored, speech completes!
    async def echo_scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus, synth_s=0.005, per_char_s=0.015)
        reply = "The capital of France is Paris and it is famous for the Eiffel Tower."
        runtime = Runtime(bus, MockInference(reply, token_delay_s=0.001), speech, mode=Mode.HANDS_FREE)
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        tx = ConfigurableTranscriber("capital of France is Paris and it is famous")
        hf = HandsFreeInput(
            bus,
            runtime,
            tx,
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            smart_barge_in=True,
            post_speech_cooldown_s=0.05,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        runtime.submit_text("tell me about france")
        deadline = time.monotonic() + 2.0
        while runtime.state is not State.SPEAKING and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        await runtime.wait_until_settled()
        await asyncio.sleep(0.08)
        await hf.stop()
        return log, runtime

    log_echo, rt_echo = run(echo_scenario())
    assert log_echo.of("operation.cancel_requested") == []
    assert rt_echo.history[-1]["content"].strip() == "The capital of France is Paris and it is famous for the Eiffel Tower."

    # 2. Wake-on-call ("Autum, what time is it?") while she speaks -> interrupts & answers new question!
    async def wake_question_scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        speech = MockSpeech(bus, synth_s=0.005, per_char_s=0.02)
        inf = MockInference("First long answer sentence. Second long answer sentence. Third sentence.", token_delay_s=0.002)
        runtime = Runtime(bus, inf, speech, mode=Mode.HANDS_FREE)
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        tx = ConfigurableTranscriber("Autum, what time is it right now?")
        hf = HandsFreeInput(
            bus,
            runtime,
            tx,
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            smart_barge_in=True,
            post_speech_cooldown_s=0.05,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        runtime.submit_text("tell me a long story")
        deadline = time.monotonic() + 2.0
        while runtime.state is not State.SPEAKING and time.monotonic() < deadline:
            await asyncio.sleep(0.01)

        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        deadline = time.monotonic() + 2.0
        while len(log.of("operation.cancel_requested")) < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()
        await hf.stop()
        return log, runtime

    log_wake, rt_wake = run(wake_question_scenario())
    cancels = log_wake.of("operation.cancel_requested")
    assert len(cancels) == 1 and cancels[0].payload["source"] == "barge_in"
    user_msgs = [m["content"] for m in rt_wake.history if m["role"] == "user"]
    assert "what time is it right now?" in user_msgs

    # 3. Follow-up questions & noun-initial questions are NEVER falsely dropped or mangled
    rt = Runtime(EventBus("echo"), MockInference("ok"), MockSpeech(EventBus("echo")), mode=Mode.HANDS_FREE)
    rt.current_spoken_text = "The capital of France is Paris and it is famous for the Eiffel Tower."
    assert rt.is_self_echo("capital of France is Paris and it is famous") is True
    assert rt.is_self_echo("What is the capital of France?") is False
    assert rt.is_self_echo("Paris?") is False

    assert Runtime.clean_wake_transcript("Atom bombs, how do they work?") == ("Atom bombs, how do they work?", False)
    assert Runtime.clean_wake_transcript("Autumn leaves are falling") == ("Autumn leaves are falling", False)
    assert Runtime.clean_wake_transcript("Otto von Bismarck") == ("Otto von Bismarck", False)
    assert Runtime.clean_wake_transcript("Autum, what time is it?") == ("what time is it?", False)
    assert Runtime.clean_wake_transcript("Hey Autumn, tell me a joke") == ("tell me a joke", False)

    # 4. Runtime.max_history setter actually resizes history deque maxlen live
    rt.max_history = 4
    assert rt.max_history == 4 and rt.history.maxlen == 4

    # 5. Loud first syllable must NOT cause normal ongoing speech to be treated as silence
    import array, math

    def _sine_frame(amp: float) -> bytes:
        return array.array("h", [int(amp * 32767 * math.sin(2 * math.pi * 400 * i / 16000)) for i in range(512)]).tobytes()

    det = TurnDetector(lambda _pcm: 0.9, min_speech_ms=96, silence_ms=160)
    det._use_energy_gate = True
    # 5 loud frames (RMS ~0.35), then 20 normal frames (RMS ~0.03) — must stay in speech without ending early!
    started = False
    ended_early = False
    for f in [_sine_frame(0.5)] * 5 + [_sine_frame(0.045)] * 20:
        ev = det.feed(f)
        if ev and ev.kind == "speech_started":
            started = True
        if ev and ev.kind == "speech_ended":
            ended_early = True
    assert started and not ended_early and det.in_speech


def test_hands_free_paused_mid_sentence_merges_first_and_second_halves():
    class SlowSequentialTranscriber:
        def __init__(self):
            self.replies = iter(["turn on the", "kitchen lights please"])

        async def transcribe(self, _wav: bytes) -> str:
            await asyncio.sleep(0.08)
            return next(self.replies)

    async def scenario():
        bus = EventBus("pause-merge")
        log = Recorder(bus)
        tx = SlowSequentialTranscriber()
        runtime = Runtime(
            bus,
            MockInference("Turning on the kitchen lights."),
            MockSpeech(bus),
            mode=Mode.HANDS_FREE,
            transcriber=tx,
            direct_audio=False,
        )
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(
            bus,
            runtime,
            tx,
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            smart_barge_in=True,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        # First half: "turn on the" -> enters State.PROCESSING while transcribing (80 ms)
        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        await asyncio.sleep(0.02)
        assert runtime.state is State.PROCESSING

        # User resumes speaking mid-transcription: "kitchen lights please"
        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        deadline = time.monotonic() + 2.0
        while len(log.of("audio.transcribed")) < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()
        await hf.stop()
        return runtime

    rt = run(scenario())
    user_msgs = [m["content"] for m in rt.history if m["role"] == "user"]
    assert user_msgs == ["turn on the kitchen lights please"]


def test_hands_free_short_sound_after_pause_neither_drops_nor_leaks_prefix():
    """Reproduces 'turn on the ... um' in both timings:
    (a) short sound ends BEFORE first transcription finishes -> first half is still answered,
    (b) short sound ends AFTER first transcription finishes -> first half is answered immediately
        and NEVER leaks onto a later question like 'what time is it'.
    """
    class ControlledTranscriber:
        def __init__(self, delay_s: float):
            self.delay_s = delay_s
            self.replies = iter(["turn on the", "what time is it"])

        async def transcribe(self, _wav: bytes) -> str:
            await asyncio.sleep(self.delay_s)
            return next(self.replies)

    async def scenario(tx_delay_s: float, short_sound_delay_s: float):
        bus = EventBus("short-um")
        log = Recorder(bus)
        tx = ControlledTranscriber(tx_delay_s)
        runtime = Runtime(
            bus,
            MockInference("Done."),
            MockSpeech(bus),
            mode=Mode.HANDS_FREE,
            transcriber=tx,
            direct_audio=False,
        )
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(
            bus,
            runtime,
            tx,
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            smart_barge_in=True,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        # 1. Speak "turn on the" (12 speech frames > MIN_UTTERANCE_S) + silence
        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        await asyncio.sleep(short_sound_delay_s)

        # 2. Short sound "um" (3 speech frames = 96 ms -> speech_started, then < MIN_UTTERANCE_S -> _nothing_heard)
        proc.push(*([_hf_frame(1)] * 3 + [_hf_frame(0)] * 6))
        deadline = time.monotonic() + 2.0
        while len(log.of("conversation.response_finished")) < 1 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()

        # 3. Later unrelated question: "what time is it" must arrive clean without "turn on the"
        proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
        deadline = time.monotonic() + 2.0
        while len(log.of("conversation.response_finished")) < 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await runtime.wait_until_settled()
        await hf.stop()
        return [m["content"] for m in runtime.history if m["role"] == "user"]

    # Case A: "um" ends BEFORE "turn on the" finishes transcribing
    assert run(scenario(tx_delay_s=0.10, short_sound_delay_s=0.02)) == ["turn on the", "what time is it"]
    # Case B: "turn on the" finishes transcribing BEFORE "um" ends
    assert run(scenario(tx_delay_s=0.01, short_sound_delay_s=0.05)) == ["turn on the", "what time is it"]


def test_wake_word_mode_ignores_chatter_wakes_on_name_and_interrupts_on_name():
    class ScriptedTranscriber:
        def __init__(self, items):
            self.items = iter(items)

        async def transcribe(self, _wav: bytes) -> str:
            return next(self.items)

    async def scenario():
        bus = EventBus("wake")
        log = Recorder(bus)
        tx = ScriptedTranscriber(
            [
                "so anyway I told my friend about the movie",   # ignored (no "Autum")
                "Autum, what time is it?",                      # wake + question in one breath
                "just chatting in the room again",              # ignored again after answering
                "Hey Autum",                                    # bare wake call -> "Yes?" + 12s window
                "open Chrome please",                           # answered inside wake window
            ]
        )
        runtime = Runtime(
            bus,
            MockInference("Sure thing."),
            MockSpeech(bus),
            mode=Mode.WAKE_WORD,
            transcriber=tx,
            direct_audio=False,
        )
        proc = StreamingRecordProc()

        async def spawn_(*args, **kw):
            return proc

        hf = HandsFreeInput(
            bus,
            runtime,
            tx,
            _hf_detector(),
            spawn_=spawn_,
            voice_barge_in=False,
            smart_barge_in=True,
        )
        await hf.start()
        await asyncio.sleep(0.02)

        async def say_turn():
            proc.push(*([_hf_frame(1)] * 12 + [_hf_frame(0)] * 6))
            await asyncio.sleep(0.06)
            await runtime.wait_until_settled()

        # 1. Background chatter -> ignored, no history added
        await say_turn()
        assert [m["content"] for m in runtime.history if m["role"] == "user"] == []

        # 2. "Autum, what time is it?" -> answered with stripped prefix
        await say_turn()
        assert [m["content"] for m in runtime.history if m["role"] == "user"] == ["what time is it?"]

        # 3. Chatter again -> ignored
        await say_turn()
        assert [m["content"] for m in runtime.history if m["role"] == "user"] == ["what time is it?"]

        # 4. Bare "Hey Autum" -> opens wake window & announces "Yes?"
        await say_turn()
        assert [m["content"] for m in runtime.history if m["role"] == "user"] == ["what time is it?"]

        # 5. Follow-up inside wake window -> answered without repeating "Autum"
        await say_turn()
        assert [m["content"] for m in runtime.history if m["role"] == "user"] == [
            "what time is it?",
            "open Chrome please",
        ]

        # 6. While speaking in WAKE_WORD mode: chatter without "Autum" does NOT interrupt,
        #    while "Autum, tell me a joke" DOES interrupt & answer!
        assert runtime.barge_in_with_text("background chatter in the room") is False
        assert runtime.barge_in_with_text("Autum, tell me a joke") is True
        await runtime.wait_until_settled()
        await hf.stop()
        return [m["content"] for m in runtime.history if m["role"] == "user"]

    user_msgs = run(scenario())
    assert user_msgs == ["what time is it?", "open Chrome please", "tell me a joke"]


def test_hands_free_respects_hard_mute_and_resumes_on_unmute():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime = Runtime(bus, MockInference("ok"), MockSpeech(bus), mode=Mode.HANDS_FREE)
        procs: list[StreamingRecordProc] = []

        async def spawn_(*args, **kw):
            p = StreamingRecordProc([_hf_frame(0)])
            procs.append(p)
            return p

        hf = HandsFreeInput(bus, runtime, FakeTranscriber(), _hf_detector(), spawn_=spawn_)
        await hf.start()
        await asyncio.sleep(0.02)
        assert runtime.state is State.LISTENING and len(procs) == 1

        # Start speaking, then hardware-mute mid-utterance
        procs[0].push(*([_hf_frame(1)] * 4))
        await asyncio.sleep(0.02)
        assert runtime.state is State.SPEECH_DETECTED
        bus.emit("audio.mute_changed", {"muted": True})
        await asyncio.sleep(0.04)
        muted_state = (runtime.state, runtime.microphone, hf.capturing)

        # Unmute: capture automatically resumes into LISTENING
        bus.emit("audio.mute_changed", {"muted": False})
        await asyncio.sleep(0.04)
        resumed_state = (runtime.state, runtime.microphone, hf.capturing, len(procs))
        await hf.stop()
        return log, muted_state, resumed_state

    log, muted_state, resumed_state = run(scenario())
    assert muted_state == (State.IDLE, "muted", False)
    assert resumed_state == (State.LISTENING, "open", True, 2)
    assert log.of("audio.transcribed")[0].payload["text"] == ""


def test_hands_free_recovers_after_microphone_disconnect():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime = Runtime(bus, MockInference("ok"), MockSpeech(bus), mode=Mode.HANDS_FREE)
        procs: list[StreamingRecordProc] = []

        async def spawn_(*args, **kw):
            p = StreamingRecordProc([_hf_frame(0)])
            procs.append(p)
            return p

        hf = HandsFreeInput(bus, runtime, FakeTranscriber(), _hf_detector(), spawn_=spawn_, reconnect_s=0.02)
        await hf.start()
        await asyncio.sleep(0.02)
        procs[0].eof()  # simulate USB microphone disconnect
        deadline = time.monotonic() + 2.0
        while not log.of("runtime.component_recovered") and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        await hf.stop()
        return log, len(procs)

    log, spawn_count = run(scenario())
    assert spawn_count >= 2
    assert log.of("runtime.component_error")[0].payload["component"] == "microphone"
    assert log.of("runtime.component_recovered")[0].payload == {"component": "microphone"}


def test_highpass_removes_dc_and_subsonic_rumble_while_keeping_speech():
    import array, math

    n = 1600  # 100 ms at 16 kHz
    rumble = array.array("h", [int(4000 * math.sin(2 * math.pi * 5 * i / 16000)) for i in range(n)]).tobytes()
    speech = array.array("h", [int(4000 * math.sin(2 * math.pi * 500 * i / 16000)) for i in range(n)]).tobytes()
    assert len(highpass_pcm(rumble)) == len(rumble)
    assert rms(highpass_pcm(rumble)) < rms(rumble) * 0.25
    assert rms(highpass_pcm(speech)) > rms(speech) * 0.85
