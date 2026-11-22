"""Voice engine audio through PipeWire's own tools (WP2).

- VoiceInput: push-to-talk capture with `pw-record`, then hands the
  utterance to the runtime and transcribes it once the reply is done.
- SpeechPlayer: streams synthesized speech into `pw-play`, paced to real
  time so it can stop at once and knows what the user actually heard.
- MuteMonitor: reads the operating system's microphone mute; Autum never
  changes it (plan §5.4).

Every wait on a device has a time limit: a microphone or speaker that
stops responding produces a clear error, never a hang.
"""

from __future__ import annotations

import array
import asyncio
import io
import math
import time
import wave
from typing import AsyncIterator, Awaitable, Callable, Protocol

from autum.bus import EventBus
from autum.events import Provenance, new_id
from autum.vad import TurnDetector, VAD_FRAME_BYTES

MIC_RATE = 16_000
SPEECH_RATE = 24_000
MIC_STALL_S = 2.0
MIC_RECONNECT_S = 1.0
DEVICE_TIMEOUT_S = 3.0
PLAY_AHEAD_S = 0.15
LEVEL_EVERY_S = 0.05
MIN_UTTERANCE_S = 0.3
MIN_APPROVAL_UTTERANCE_S = 0.12
AEC_SOURCE = "autum_aec_source"
AEC_SINK = "autum_aec_sink"
MIC_HELP = (
    "The microphone sent no audio for 2 seconds. Stop any program stuck on it, "
    "unplug the microphone and plug it back in, then try again."
)

PIPE = asyncio.subprocess.PIPE
Spawn = Callable[..., Awaitable[asyncio.subprocess.Process]]


async def spawn(*args: str, stdin=None, stdout=None) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        *args, stdin=stdin, stdout=stdout, stderr=asyncio.subprocess.DEVNULL
    )


def highpass_pcm(pcm: bytes, rate: int = MIC_RATE, cutoff_hz: float = 80.0) -> bytes:
    """Remove DC offset and sub-sonic rumble (< cutoff_hz) from 16-bit mono PCM.

    Preserves exact frame count. Keeps USB mic rumble (e.g. 0-20 Hz) from
    degrading speech recognition while leaving the speech band untouched.
    """
    samples = array.array("h", pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return b""
    alpha = 1.0 / (1.0 + 2.0 * math.pi * cutoff_hz / rate)
    out = array.array("h", [0]) * len(samples)
    prev_x = float(samples[0])
    prev_y = 0.0
    for i, s in enumerate(samples):
        x = float(s)
        y = alpha * (prev_y + x - prev_x)
        prev_x, prev_y = x, y
        out[i] = max(-32768, min(32767, int(round(y))))
    return out.tobytes()


def wav_bytes(pcm: bytes, rate: int) -> bytes:
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(pcm)
    return buffer.getvalue()


def rms(pcm: bytes) -> float:
    samples = array.array("h", pcm[: len(pcm) - len(pcm) % 2])
    if not samples:
        return 0.0
    return math.sqrt(sum(s * s for s in samples) / len(samples)) / 32768


def _kill(proc: asyncio.subprocess.Process | None) -> None:
    if proc is not None and proc.returncode is None:
        try:
            proc.kill()
        except ProcessLookupError:
            pass


async def check_microphone(spawn_: Spawn = spawn, timeout_s: float = MIC_STALL_S) -> bool:
    """True if the microphone delivers audio within timeout_s (nothing is kept)."""
    proc = await spawn_("pw-record", "--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16", "-", stdout=PIPE)
    try:
        await asyncio.wait_for(proc.stdout.readexactly(MIC_RATE // 50), timeout_s)  # 10 ms of audio
        return True
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        return False
    finally:
        _kill(proc)
        await proc.wait()


class Transcriber(Protocol):
    async def transcribe(self, wav: bytes) -> str: ...


class SpeechRuntime(Protocol):
    def submit_speech(self, turn_id: str, audio: bytes, cause=None) -> None: ...


class VoiceInput:
    """Push-to-talk: start() opens the microphone, finish() ends the turn."""

    def __init__(
        self,
        bus: EventBus,
        runtime: SpeechRuntime,
        transcriber: Transcriber,
        *,
        spawn_: Spawn = spawn,
        stall_s: float = MIC_STALL_S,
        target: str | None = None,
    ) -> None:
        self.bus = bus
        self.runtime = runtime
        self.transcriber = transcriber
        self.spawn = spawn_
        self.stall_s = stall_s
        self.target = target
        self.muted = False
        self._turn: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._reader: asyncio.Task | None = None
        self._watchdog: asyncio.Task | None = None
        self._pcm = bytearray()
        self._first_audio = asyncio.Event()
        self._awaiting_transcript: dict[str, bytes] = {}
        self._tasks: set[asyncio.Task] = set()
        bus.subscribe(
            self._on_event,
            {"audio.mute_changed", "conversation.response_finished", "operation.cancelled"},
        )

    @property
    def capturing(self) -> bool:
        return self._turn is not None

    async def start(self) -> str | None:
        """Begin a turn. Returns None if muted or already capturing."""
        if self.muted or self.capturing:
            return None
        turn = self._turn = new_id()
        self._pcm = bytearray()
        self._first_audio = asyncio.Event()
        target_args = ("--target", self.target) if self.target else ()
        self._proc = await self.spawn(
            "pw-record", *target_args, "--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16",
            "--latency", "20ms", "-", stdout=PIPE,
        )
        self._reader = asyncio.create_task(self._read(self._proc))
        self._watchdog = asyncio.create_task(self._watch(turn))
        self.bus.emit("audio.capture_started")
        self.bus.emit("audio.speech_started", {"mode": "push_to_talk"}, provenance=Provenance.USER, turn_id=turn)
        return turn

    async def finish(self) -> None:
        """End the turn and hand the utterance to the runtime."""
        if not self.capturing:
            return
        turn = self._turn
        pcm = await self._stop_capture()
        ended = self.bus.emit("audio.speech_ended", {"confidence": 1.0}, provenance=Provenance.USER, turn_id=turn)
        waiting_approval = str(getattr(self.runtime, "state", "")) == "waiting_for_approval"
        min_s = MIN_APPROVAL_UTTERANCE_S if waiting_approval else MIN_UTTERANCE_S
        if len(pcm) < min_s * MIC_RATE * 2:
            self._nothing_heard(turn)
            return
        wav = wav_bytes(highpass_pcm(pcm, MIC_RATE), MIC_RATE)
        if not waiting_approval and getattr(self.runtime, "direct_audio", True):
            while len(self._awaiting_transcript) >= 8:
                self._awaiting_transcript.pop(next(iter(self._awaiting_transcript)))
            self._awaiting_transcript[turn] = wav
        self.runtime.submit_speech(turn, wav, ended)

    async def abort(self, error: str | None = None) -> None:
        """Drop the current utterance (muted, stalled, or quitting)."""
        if not self.capturing:
            return
        turn = self._turn
        await self._stop_capture()
        if error:
            self.bus.emit("runtime.component_error", {"component": "microphone", "error": error})
        self._nothing_heard(turn)

    def _nothing_heard(self, turn: str) -> None:
        # An empty final transcript returns the runtime to rest.
        self.bus.emit("audio.transcribed", {"text": "", "final": True}, provenance=Provenance.USER, turn_id=turn)

    async def _read(self, proc: asyncio.subprocess.Process) -> None:
        while chunk := await proc.stdout.read(3200):
            self._pcm += chunk
            self._first_audio.set()

    async def _watch(self, turn: str) -> None:
        try:
            await asyncio.wait_for(self._first_audio.wait(), self.stall_s)
        except asyncio.TimeoutError:
            if self._turn == turn:
                await self.abort(MIC_HELP)

    async def _stop_capture(self) -> bytes:
        proc, reader, watchdog = self._proc, self._reader, self._watchdog
        self._turn = self._proc = self._reader = self._watchdog = None
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
        _kill(proc)
        if reader is not None:
            try:
                await asyncio.wait_for(reader, DEVICE_TIMEOUT_S)
            except asyncio.TimeoutError:
                reader.cancel()
        if proc is not None:
            await proc.wait()
        self.bus.emit("audio.capture_stopped")
        return bytes(self._pcm)

    def _on_event(self, event) -> None:
        if event.type == "audio.mute_changed":
            self.muted = bool(event.payload["muted"])
            if self.muted and self.capturing:
                self._track(self.abort())  # captured audio is dropped while muted
        elif event.turn_id in self._awaiting_transcript:
            self._track(self._transcribe(event.turn_id))

    async def _transcribe(self, turn: str) -> None:
        wav = self._awaiting_transcript.pop(turn, None)
        if wav is None:
            return
        try:
            text = await self.transcriber.transcribe(wav)
        except Exception as exc:
            self.bus.emit("runtime.component_error", {"component": "transcription", "error": str(exc)})
            return
        self.bus.emit("audio.transcribed", {"text": text, "final": True}, provenance=Provenance.USER, turn_id=turn)

    def _track(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class Synthesizer(Protocol):
    def stream(self, text: str) -> AsyncIterator[bytes]: ...


class SpeechPlayer:
    """SpeechOutput (runtime.py) on top of pw-play.

    Audio is written no more than PLAY_AHEAD_S ahead of real time, so
    stop() silences it almost at once and the timeline tells which
    sentences the user actually heard.
    """

    def __init__(
        self,
        bus: EventBus,
        synth: Synthesizer,
        *,
        spawn_: Spawn = spawn,
        target: str | None = None,
    ) -> None:
        self.bus = bus
        self.synth = synth
        self.spawn = spawn_
        self.target = target
        self._op: str | None = None
        self._turn: str | None = None
        self._queue: asyncio.Queue[str] = asyncio.Queue()
        self._worker: asyncio.Task | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._timeline: list[tuple[float, str]] = []
        self._written = 0
        self._t0: float | None = None
        self._next_level_s = 0.0
        self._timers: list[asyncio.TimerHandle] = []
        self._error: Exception | None = None

    # SpeechOutput protocol

    def enqueue(self, text: str, *, operation_id: str, turn_id: str | None) -> None:
        if operation_id != self._op:
            self._reset()
            self._op, self._turn = operation_id, turn_id
        self._queue.put_nowait(text)
        if self._worker is None:
            self._worker = asyncio.create_task(self._run(operation_id, self._queue))

    async def drain(self, operation_id: str) -> None:
        if operation_id != self._op:
            return
        await self._queue.join()
        if self._error is not None:
            error, self._error = self._error, None
            raise error
        proc = self._proc
        if proc is not None:
            proc.stdin.close()
            remaining = max(0.0, self._written_s() - self._elapsed())
            try:
                await asyncio.wait_for(proc.wait(), remaining + DEVICE_TIMEOUT_S)
            except asyncio.TimeoutError:
                _kill(proc)
                await proc.wait()
                raise RuntimeError("the speakers did not finish playing; check the audio output") from None
        started = self._t0 is not None
        self._reset()
        if started:
            self.bus.emit("speech.playback_finished", {"reason": "completed"}, turn_id=self._turn, operation_id=operation_id)

    def stop(self) -> str:
        if self._op is None:
            return ""
        heard = " ".join(text for start, text in self._timeline if start <= self._elapsed())
        op, turn, started = self._op, self._turn, self._t0 is not None
        self._reset()
        if started:
            self.bus.emit("speech.playback_finished", {"reason": "cancelled"}, turn_id=turn, operation_id=op)
        return heard

    # internals

    def _reset(self) -> None:
        if self._worker is not None and self._worker is not asyncio.current_task():
            self._worker.cancel()
        self._worker = None
        proc, self._proc = self._proc, None
        _kill(proc)
        if proc is not None:
            asyncio.create_task(proc.wait())  # reap it
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
        self._queue = asyncio.Queue()
        self._timeline.clear()
        self._written = 0
        self._t0 = None
        self._next_level_s = 0.0
        self._op = None

    def _written_s(self) -> float:
        return self._written / (SPEECH_RATE * 2)

    def _elapsed(self) -> float:
        return 0.0 if self._t0 is None else time.monotonic() - self._t0

    async def _run(self, op: str, queue: asyncio.Queue) -> None:
        # Uses its own queue: after stop() a new queue replaces it, and a
        # cancelled worker must not mark items done on the new one.
        while True:
            text = await queue.get()
            try:
                first = True
                async for pcm in self.synth.stream(text):
                    if first:
                        self._timeline.append((self._written_s(), text))
                        first = False
                    await self._write(op, pcm)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                self._error = exc
            finally:
                queue.task_done()

    async def _write(self, op: str, pcm: bytes) -> None:
        if self._proc is None:
            target_args = ("--target", self.target) if self.target else ()
            self._proc = await self.spawn(
                "pw-play", *target_args, "--raw", "--rate", str(SPEECH_RATE), "--channels", "1", "--format", "s16",
                "--latency", "20ms", "-", stdin=PIPE,
            )
        now = time.monotonic()
        if self._t0 is None:
            self._t0 = now
            self.bus.emit("speech.playback_started", turn_id=self._turn, operation_id=op)
        elif now - self._t0 > self._written_s():
            self._t0 = now - self._written_s()  # playback ran dry and resumes now
        ahead = self._written_s() - self._elapsed()
        if ahead > PLAY_AHEAD_S:
            await asyncio.sleep(ahead - PLAY_AHEAD_S)
        if self._written_s() >= self._next_level_s:
            delay = max(0.0, self._written_s() - self._elapsed())
            level = rms(pcm)
            self._timers.append(asyncio.get_running_loop().call_later(delay, self._level, op, level))
            self._next_level_s = self._written_s() + LEVEL_EVERY_S
        self._proc.stdin.write(pcm)
        try:
            await asyncio.wait_for(self._proc.stdin.drain(), DEVICE_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError("the speakers stopped accepting audio; check the audio output") from None
        self._written += len(pcm)

    def _level(self, op: str, level: float) -> None:
        self.bus.emit("speech.output_level", {"rms": round(level, 4)}, turn_id=self._turn, operation_id=op)


async def read_source_muted() -> bool | None:
    proc = await asyncio.create_subprocess_exec(
        "wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@", stdout=PIPE, stderr=asyncio.subprocess.DEVNULL
    )
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), 2.0)
    except asyncio.TimeoutError:
        _kill(proc)
        await proc.wait()
        return None
    return b"[MUTED]" in out


class MuteMonitor:
    """Publishes the OS microphone mute state whenever it changes. Never changes it."""

    def __init__(
        self,
        bus: EventBus,
        read: Callable[[], Awaitable[bool | None]] = read_source_muted,
        interval_s: float = 0.5,
        error_after: int = 4,
    ) -> None:
        self.bus = bus
        self.read = read
        self.interval_s = interval_s
        self.error_after = error_after
        self.muted: bool | None = None
        self._failures = 0
        self._errored = False

    async def run(self) -> None:
        while True:
            try:
                muted = await self.read()
            except Exception:
                muted = None
            if muted is None:
                self._failures += 1
                if self._failures >= self.error_after and not self._errored:
                    self._errored = True
                    self.bus.emit(
                        "runtime.component_error",
                        {"component": "mute_monitor", "error": "could not read microphone mute state from PipeWire"},
                    )
            else:
                self._failures = 0
                if self._errored:
                    self._errored = False
                    self.bus.emit("runtime.component_recovered", {"component": "mute_monitor"})
                if muted != self.muted:
                    self.muted = muted
                    self.bus.emit("audio.mute_changed", {"muted": muted})
            await asyncio.sleep(self.interval_s)


class HandsFreeInput:
    """Continuous capture with streaming VAD turn detection and voice barge-in (M3)."""

    def __init__(
        self,
        bus: EventBus,
        runtime: SpeechRuntime,
        transcriber: Transcriber,
        detector: TurnDetector,
        *,
        spawn_: Spawn = spawn,
        stall_s: float = MIC_STALL_S,
        reconnect_s: float = MIC_RECONNECT_S,
        target: str | None = None,
    ) -> None:
        self.bus = bus
        self.runtime = runtime
        self.transcriber = transcriber
        self.detector = detector
        self.spawn = spawn_
        self.stall_s = stall_s
        self.reconnect_s = reconnect_s
        self.target = target
        self.muted = False
        self._active = False
        self._autum_speaking = False
        self._errored = False
        self._turn: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._runner: asyncio.Task | None = None
        self._awaiting_transcript: dict[str, bytes] = {}
        self._tasks: set[asyncio.Task] = set()
        bus.subscribe(
            self._on_event,
            {
                "audio.mute_changed",
                "runtime.state_changed",
                "conversation.response_finished",
                "operation.cancelled",
            },
        )

    @property
    def active(self) -> bool:
        return self._active

    @property
    def capturing(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    async def start(self) -> None:
        """Enable hands-free listening. Starts capturing immediately unless muted."""
        if self._active:
            return
        self._active = True
        if not self.muted:
            self._start_runner()

    async def stop(self) -> None:
        """Disable hands-free listening and release the microphone."""
        self._active = False
        await self._stop_runner()

    def _start_runner(self) -> None:
        if self._runner is None or self._runner.done():
            self._runner = asyncio.create_task(self._run())

    async def _stop_runner(self) -> None:
        runner, self._runner = self._runner, None
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
        await self._close_capture()
        self._drop_turn()
        self.detector.reset()

    async def _run(self) -> None:
        try:
            while self._active and not self.muted:
                ok = await self._capture_session()
                if not self._active or self.muted:
                    break
                if not ok:
                    await asyncio.sleep(self.reconnect_s)
        except asyncio.CancelledError:
            raise
        finally:
            await self._close_capture()

    async def _capture_session(self) -> bool:
        target_args = ("--target", self.target) if self.target else ()
        try:
            proc = await self.spawn(
                "pw-record", *target_args, "--raw", "--rate", str(MIC_RATE), "--channels", "1",
                "--format", "s16", "--latency", "20ms", "-", stdout=PIPE,
            )
        except Exception as exc:
            self._report_error(str(exc))
            return False
        self._proc = proc
        self.detector.reset()
        self.bus.emit("audio.capture_started")
        got_audio = False
        try:
            while self._active and not self.muted:
                try:
                    frame = await asyncio.wait_for(proc.stdout.readexactly(VAD_FRAME_BYTES), self.stall_s)
                except asyncio.TimeoutError:
                    self._report_error(MIC_HELP)
                    return False
                except asyncio.IncompleteReadError:
                    if self._active and not self.muted:
                        self._report_error("microphone stream ended unexpectedly; retrying")
                    return False
                if not got_audio:
                    got_audio = True
                    if self._errored:
                        self._errored = False
                        self.bus.emit("runtime.component_recovered", {"component": "microphone"})
                self._handle_frame(frame)
            return True
        finally:
            await self._close_capture()
            self._drop_turn()

    def _handle_frame(self, frame: bytes) -> None:
        event = self.detector.feed(frame, speaking=self._autum_speaking)
        if event is None:
            return
        if event.kind == "speech_started":
            self._turn = new_id()
            self.bus.emit(
                "audio.speech_started",
                {"mode": "hands_free"},
                provenance=Provenance.USER,
                turn_id=self._turn,
            )
        elif event.kind == "speech_ended" and self._turn is not None:
            turn, self._turn = self._turn, None
            ended = self.bus.emit(
                "audio.speech_ended",
                {"confidence": round(event.confidence, 3)},
                provenance=Provenance.USER,
                turn_id=turn,
            )
            waiting_approval = str(getattr(self.runtime, "state", "")) == "waiting_for_approval"
            min_s = MIN_APPROVAL_UTTERANCE_S if waiting_approval else MIN_UTTERANCE_S
            if len(event.pcm) < min_s * MIC_RATE * 2:
                self._nothing_heard(turn)
                return
            wav = wav_bytes(highpass_pcm(event.pcm, MIC_RATE), MIC_RATE)
            if not waiting_approval and getattr(self.runtime, "direct_audio", True):
                while len(self._awaiting_transcript) >= 8:
                    self._awaiting_transcript.pop(next(iter(self._awaiting_transcript)))
                self._awaiting_transcript[turn] = wav
            self.runtime.submit_speech(turn, wav, ended)

    async def _close_capture(self) -> None:
        proc, self._proc = self._proc, None
        if proc is None:
            return
        _kill(proc)
        try:
            await asyncio.wait_for(proc.wait(), DEVICE_TIMEOUT_S)
        except asyncio.TimeoutError:
            pass
        self.bus.emit("audio.capture_stopped")

    def _drop_turn(self) -> None:
        if self._turn is not None:
            turn, self._turn = self._turn, None
            self._nothing_heard(turn)

    def _nothing_heard(self, turn: str) -> None:
        self.bus.emit("audio.transcribed", {"text": "", "final": True}, provenance=Provenance.USER, turn_id=turn)

    def _report_error(self, message: str) -> None:
        self._errored = True
        self.bus.emit("runtime.component_error", {"component": "microphone", "error": message})

    def _on_event(self, event) -> None:
        if event.type == "audio.mute_changed":
            self.muted = bool(event.payload["muted"])
            if not self._active:
                return
            if self.muted:
                self._track(self._stop_runner())
            else:
                self._start_runner()
        elif event.type == "runtime.state_changed":
            self._autum_speaking = event.payload["to"] in ("speaking", "thinking")
        elif event.turn_id in self._awaiting_transcript:
            self._track(self._transcribe(event.turn_id))

    async def _transcribe(self, turn: str) -> None:
        wav = self._awaiting_transcript.pop(turn, None)
        if wav is None:
            return
        try:
            text = await self.transcriber.transcribe(wav)
        except Exception as exc:
            self.bus.emit("runtime.component_error", {"component": "transcription", "error": str(exc)})
            return
        self.bus.emit("audio.transcribed", {"text": text, "final": True}, provenance=Provenance.USER, turn_id=turn)

    def _track(self, coro) -> None:
        task = asyncio.create_task(coro)
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)


class EchoCanceller:
    """PipeWire WebRTC echo-cancel virtual sink/source pair for hands-free speaker use."""

    def __init__(self, source: str = AEC_SOURCE, sink: str = AEC_SINK) -> None:
        self.source = source
        self.sink = sink
        self._module: str | None = None

    async def start(self) -> bool:
        """Load module-echo-cancel; returns True when both devices are ready."""
        await self._unload_leftovers()
        proc = await asyncio.create_subprocess_exec(
            "pactl", "load-module", "module-echo-cancel", "aec_method=webrtc",
            f"source_name={self.source}", f"sink_name={self.sink}",
            stdout=PIPE, stderr=PIPE,
        )
        out, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        if proc.returncode != 0:
            return False
        self._module = out.decode().strip()
        for _ in range(50):
            sinks = await self._pactl("list", "short", "sinks")
            sources = await self._pactl("list", "short", "sources")
            if self.sink in sinks and self.source in sources:
                return True
            await asyncio.sleep(0.1)
        await self.stop()
        return False

    async def stop(self) -> None:
        if self._module is None:
            return
        module, self._module = self._module, None
        await self._pactl("unload-module", module)

    async def _unload_leftovers(self) -> None:
        for line in (await self._pactl("list", "modules", "short")).splitlines():
            if "module-echo-cancel" in line and self.sink in line:
                await self._pactl("unload-module", line.split()[0])

    @staticmethod
    async def _pactl(*args: str) -> str:
        proc = await asyncio.create_subprocess_exec("pactl", *args, stdout=PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), 5.0)
        return out.decode(errors="replace")
