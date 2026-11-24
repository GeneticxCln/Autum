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
PLAY_AHEAD_S = 0.35
LEVEL_EVERY_S = 0.05
MIN_UTTERANCE_S = 0.18
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


async def check_microphone(
    spawn_: Spawn = spawn,
    timeout_s: float = MIC_STALL_S,
    target: str | None = None,
) -> bool:
    """True if the microphone delivers audio within timeout_s (nothing is kept)."""
    target_args = ("--target", target) if target else ()
    proc = await spawn_(
        "pw-record", *target_args, "--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16", "-",
        stdout=PIPE,
    )
    try:
        await asyncio.wait_for(proc.stdout.readexactly(MIC_RATE // 50), timeout_s)  # 10 ms of audio
        return True
    except (asyncio.TimeoutError, asyncio.IncompleteReadError):
        return False
    finally:
        _kill(proc)
        await proc.wait()


def list_pipewire_devices(timeout_s: float = 2.0) -> dict[str, list[dict[str, object]]]:
    """Return available PipeWire microphone inputs (non-monitor) and speaker outputs."""
    import json
    import subprocess

    def _run(*cmd: str) -> str:
        try:
            res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
            return res.stdout if res.returncode == 0 else ""
        except (OSError, subprocess.SubprocessError):
            return ""

    default_src = _run("pactl", "get-default-source").strip()
    default_sink = _run("pactl", "get-default-sink").strip()

    def _parse(kind: str, default_name: str, skip_monitors: bool) -> list[dict[str, object]]:
        raw = _run("pactl", "-f", "json", "list", kind)
        items: list[dict[str, object]] = []
        if raw.strip():
            try:
                parsed = json.loads(raw)
                if isinstance(parsed, list):
                    for entry in parsed:
                        if not isinstance(entry, dict):
                            continue
                        name = str(entry.get("name") or "").strip()
                        if not name:
                            continue
                        if skip_monitors and name.endswith(".monitor"):
                            continue
                        if name in (AEC_SOURCE, AEC_SINK):
                            continue
                        desc = str(entry.get("description") or name).strip()
                        state = str(entry.get("state") or "UNKNOWN").upper()
                        items.append(
                            {
                                "name": name,
                                "description": desc,
                                "state": state,
                                "is_default": name == default_name,
                            }
                        )
                    return items
            except ValueError:
                pass
        # Fallback to `pactl list short <kind>` when JSON output is unavailable
        short = _run("pactl", "list", "short", kind)
        for line in short.splitlines():
            parts = line.split("\t")
            if len(parts) < 2:
                continue
            name = parts[1].strip()
            if not name or (skip_monitors and name.endswith(".monitor")) or name in (AEC_SOURCE, AEC_SINK):
                continue
            state = parts[-1].strip().upper() if len(parts) >= 5 else "UNKNOWN"
            items.append(
                {
                    "name": name,
                    "description": name,
                    "state": state,
                    "is_default": name == default_name,
                }
            )
        return items

    return {
        "inputs": _parse("sources", default_src, skip_monitors=True),
        "outputs": _parse("sinks", default_sink, skip_monitors=False),
    }


def probe_microphone_sync(target: str = "", duration_s: float = 0.6, timeout_s: float = 2.0) -> dict[str, object]:
    """Capture a short PCM snippet from `target` (or default source) and return health & RMS level."""
    import select
    import subprocess

    clean_target = (target or "").strip()
    cmd = ["pw-record"]
    if clean_target:
        cmd.extend(["--target", clean_target])
    cmd.extend(["--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16", "--latency", "20ms", "-"])
    want_bytes = max(320, int(duration_s * MIC_RATE * 2))
    buf = bytearray()
    try:
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
    except OSError as exc:
        return {"ok": False, "rms": 0.0, "bytes_read": 0, "device": clean_target or "default", "error": str(exc)}
    deadline = time.monotonic() + timeout_s
    try:
        assert proc.stdout is not None
        fd = proc.stdout.fileno()
        while len(buf) < want_bytes:
            rem = deadline - time.monotonic()
            if rem <= 0:
                break
            ready, _, _ = select.select([fd], [], [], min(0.15, rem))
            if not ready:
                continue
            chunk = proc.stdout.read1(min(3200, want_bytes - len(buf)))  # type: ignore[attr-defined]
            if not chunk:
                break
            buf.extend(chunk)
    finally:
        if proc.poll() is None:
            proc.kill()
        try:
            proc.wait(timeout=1.0)
        except subprocess.TimeoutExpired:
            pass
    if len(buf) < MIC_RATE // 50:
        return {
            "ok": False,
            "rms": 0.0,
            "bytes_read": len(buf),
            "device": clean_target or "default",
            "error": "No audio received within 2 seconds (microphone may be stalled, muted, or disconnected).",
        }
    level = round(rms(bytes(buf)), 4)
    return {
        "ok": True,
        "rms": level,
        "bytes_read": len(buf),
        "device": clean_target or "default",
        "error": "",
    }


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
        self._auto_finish_turn: str | None = None
        self._auto_heard_speech: bool = False
        self._auto_quiet_chunks: int = 0
        self._auto_total_chunks: int = 0
        bus.subscribe(
            self._on_event,
            {
                "audio.mute_changed",
                "runtime.model_loading",
                "runtime.state_changed",
                "conversation.response_finished",
                "operation.cancelled",
                "runtime.component_error",
            },
        )

    @property
    def capturing(self) -> bool:
        return self._turn is not None

    async def start(self) -> str | None:
        """Begin a turn. Returns None if muted, loading models, or already capturing."""
        if self.muted or self.capturing or str(getattr(self.runtime, "state", "")) == "loading":
            return None
        turn = self._turn = new_id()
        pcm_buf = bytearray()
        first_audio = asyncio.Event()
        self._pcm = pcm_buf
        self._first_audio = first_audio
        target_args = ("--target", self.target) if self.target else ()
        try:
            proc = await self.spawn(
                "pw-record", *target_args, "--raw", "--rate", str(MIC_RATE), "--channels", "1", "--format", "s16",
                "--latency", "20ms", "-", stdout=PIPE,
            )
        except BaseException:
            if self._turn == turn:
                self._turn = None
            raise
        # Guard against quick tap / mute / model load while pw-record was still spawning
        if self._turn != turn or self.muted or str(getattr(self.runtime, "state", "")) == "loading":
            if self._turn == turn:
                self._turn = None
            _kill(proc)
            await proc.wait()
            return None
        self._proc = proc
        self._reader = asyncio.create_task(self._read(self._proc, pcm_buf, first_audio))
        self._watchdog = asyncio.create_task(self._watch(turn, first_audio))
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
        is_wake_mode = str(getattr(self.runtime, "mode", "")) == "wake_word"
        track_direct = not waiting_approval and not is_wake_mode and getattr(self.runtime, "direct_audio", True)
        if track_direct:
            while len(self._awaiting_transcript) >= 8:
                self._awaiting_transcript.pop(next(iter(self._awaiting_transcript)))
            self._awaiting_transcript[turn] = wav
        accepted = self.runtime.submit_speech(turn, wav, ended)
        if accepted is False and track_direct:
            self._awaiting_transcript.pop(turn, None)

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

    def enable_auto_finish(self) -> None:
        """Finish the active push-to-talk capture automatically once speech ends with silence."""
        if self._turn is not None:
            self._auto_finish_turn = self._turn
            self._auto_heard_speech = rms(bytes(self._pcm)) >= 0.015 if self._pcm else False
            self._auto_quiet_chunks = 0
            self._auto_total_chunks = 0

    async def _read(
        self,
        proc: asyncio.subprocess.Process,
        pcm_buf: bytearray,
        first_audio: asyncio.Event,
    ) -> None:
        while chunk := await proc.stdout.read(3200):
            pcm_buf += chunk
            first_audio.set()
            if self._auto_finish_turn is not None and self._auto_finish_turn == self._turn:
                self._auto_total_chunks += 1
                level = rms(chunk)
                if level >= 0.015:
                    self._auto_heard_speech = True
                    self._auto_quiet_chunks = 0
                elif self._auto_heard_speech:
                    if level < 0.012:
                        self._auto_quiet_chunks += 1
                    else:
                        self._auto_quiet_chunks = 0
                # Finish after 800 ms of post-speech silence, 4.5 s of initial silence, or 20 s max.
                should_finish = (
                    (self._auto_heard_speech and self._auto_quiet_chunks >= 8)
                    or (not self._auto_heard_speech and self._auto_total_chunks >= 45)
                    or self._auto_total_chunks >= 200
                )
                if should_finish:
                    self._auto_finish_turn = None
                    self._track(self.finish())
                    return

    async def _watch(self, turn: str, first_audio: asyncio.Event) -> None:
        try:
            await asyncio.wait_for(first_audio.wait(), self.stall_s)
        except asyncio.TimeoutError:
            if self._turn == turn:
                await self.abort(MIC_HELP)

    async def _stop_capture(self) -> bytes:
        proc, reader, watchdog, pcm_buf = self._proc, self._reader, self._watchdog, self._pcm
        was_started = proc is not None
        self._turn = self._proc = self._reader = self._watchdog = None
        self._auto_finish_turn = None
        if watchdog is not None and watchdog is not asyncio.current_task():
            watchdog.cancel()
        _kill(proc)
        if reader is not None and reader is not asyncio.current_task():
            try:
                await asyncio.wait_for(reader, DEVICE_TIMEOUT_S)
            except asyncio.TimeoutError:
                reader.cancel()
        if proc is not None:
            await proc.wait()
        if was_started:
            self.bus.emit("audio.capture_stopped")
        return bytes(pcm_buf)

    def _on_event(self, event) -> None:
        if event.type == "audio.mute_changed":
            self.muted = bool(event.payload["muted"])
            if self.muted and self.capturing:
                self._track(self.abort())  # captured audio is dropped while muted
        elif event.type == "runtime.model_loading":
            if self.capturing:
                self._track(self.abort())
        elif event.type == "runtime.state_changed":
            if event.payload.get("to") == "loading" and self.capturing:
                self._track(self.abort())
        elif event.type == "runtime.component_error":
            if event.turn_id in self._awaiting_transcript:
                self._awaiting_transcript.pop(event.turn_id, None)
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
        self._reapers: set[asyncio.Task] = set()
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
        if operation_id != self._op:
            return
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
        if operation_id != self._op:
            return
        started = self._t0 is not None
        turn = self._turn
        self._reset()
        if started:
            self.bus.emit("speech.playback_finished", {"reason": "completed"}, turn_id=turn, operation_id=operation_id)

    def stop(self) -> str:
        if self._op is None:
            return ""
        started = self._t0 is not None
        heard = " ".join(text for start, text in self._timeline if start <= self._elapsed()) if started else ""
        op, turn = self._op, self._turn
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
            reaper = asyncio.create_task(proc.wait())
            self._reapers.add(reaper)
            reaper.add_done_callback(self._reapers.discard)
        for timer in self._timers:
            timer.cancel()
        self._timers.clear()
        self._queue = asyncio.Queue()
        self._timeline.clear()
        self._written = 0
        self._t0 = None
        self._next_level_s = 0.0
        self._error = None
        self._op = None
        self._turn = None

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
            loop = asyncio.get_running_loop()
            if len(self._timers) >= 64:
                now_loop = loop.time()
                self._timers = [t for t in self._timers if not t.cancelled() and t.when() > now_loop]
            self._timers.append(loop.call_later(delay, self._level, op, level))
            self._next_level_s = self._written_s() + LEVEL_EVERY_S
        try:
            self._proc.stdin.write(pcm)
            await asyncio.wait_for(self._proc.stdin.drain(), DEVICE_TIMEOUT_S)
        except asyncio.TimeoutError:
            raise RuntimeError("the speakers stopped accepting audio; check the audio output") from None
        except (BrokenPipeError, ConnectionResetError, OSError):
            dead, self._proc = self._proc, None
            _kill(dead)
            if dead is not None:
                reaper = asyncio.create_task(dead.wait())
                self._reapers.add(reaper)
                reaper.add_done_callback(self._reapers.discard)
            raise RuntimeError("the audio output stream disconnected; check PipeWire/speakers") from None
        self._written += len(pcm)

    def _level(self, op: str, level: float) -> None:
        self.bus.emit("speech.output_level", {"rms": round(level, 4)}, turn_id=self._turn, operation_id=op)


async def read_source_muted(target: str | None = None) -> bool | None:
    if target:
        proc = await asyncio.create_subprocess_exec(
            "pactl", "get-source-mute", target, stdout=PIPE, stderr=asyncio.subprocess.DEVNULL
        )
        try:
            out, _ = await asyncio.wait_for(proc.communicate(), 2.0)
        except asyncio.TimeoutError:
            _kill(proc)
            await proc.wait()
            return None
        if proc.returncode == 0:
            return b"yes" in out.lower()
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
        voice_barge_in: bool = True,
        smart_barge_in: bool = False,
        post_speech_cooldown_s: float = 0.0,
    ) -> None:
        self.bus = bus
        self.runtime = runtime
        self.transcriber = transcriber
        self.detector = detector
        self.spawn = spawn_
        self.stall_s = stall_s
        self.reconnect_s = reconnect_s
        self.target = target
        self.voice_barge_in = voice_barge_in
        self.smart_barge_in = smart_barge_in
        self.post_speech_cooldown_s = post_speech_cooldown_s
        self.muted = False
        self._active = False
        self._autum_speaking = False
        self._runtime_state = "idle"
        self._cooldown_until = 0.0
        self._bleed_rms = 0.018
        self._candidate_barge_in = False
        self._candidate_check_echo = False
        self._candidate_started_ns = 0
        self._loud_run = 0
        self._errored = False
        self._turn: str | None = None
        self._proc: asyncio.subprocess.Process | None = None
        self._runner: asyncio.Task | None = None
        self._runner_gen = 0
        self._awaiting_transcript: dict[str, bytes] = {}
        self._tasks: set[asyncio.Task] = set()
        self._tx_errored = False
        self._last_tx_err_at = 0.0
        bus.subscribe(
            self._on_event,
            {
                "audio.mute_changed",
                "runtime.state_changed",
                "runtime.component_error",
                "speech.playback_started",
                "speech.output_level",
                "speech.playback_finished",
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
            self._runner_gen += 1
            gen = self._runner_gen
            self._runner = asyncio.create_task(self._run(gen))

    async def _stop_runner(self) -> None:
        self._runner_gen += 1
        gen = self._runner_gen
        runner, self._runner = self._runner, None
        if runner is not None and not runner.done():
            runner.cancel()
            try:
                await runner
            except asyncio.CancelledError:
                pass
        if self._runner_gen == gen:
            await self._close_capture()
            self._drop_turn()
            self.detector.reset()
            if self._active and not self.muted:
                self._start_runner()

    async def _run(self, gen: int) -> None:
        try:
            while self._active and not self.muted and self._runner_gen == gen:
                ok = await self._capture_session(gen)
                if not self._active or self.muted or self._runner_gen != gen:
                    break
                if not ok:
                    await asyncio.sleep(self.reconnect_s)
        except asyncio.CancelledError:
            raise
        finally:
            if self._runner_gen == gen:
                await self._close_capture()

    async def _capture_session(self, gen: int) -> bool:
        target_args = ("--target", self.target) if self.target else ()
        try:
            proc = await self.spawn(
                "pw-record", *target_args, "--raw", "--rate", str(MIC_RATE), "--channels", "1",
                "--format", "s16", "--latency", "20ms", "-", stdout=PIPE,
            )
        except Exception as exc:
            self._report_error(str(exc))
            return False
        if not self._active or self.muted or self._runner_gen != gen:
            _kill(proc)
            try:
                await asyncio.wait_for(proc.wait(), DEVICE_TIMEOUT_S)
            except asyncio.TimeoutError:
                pass
            return True
        self._proc = proc
        self.detector.reset()
        self.bus.emit("audio.capture_started")
        got_audio = False
        try:
            while self._active and not self.muted and self._runner_gen == gen:
                try:
                    frame = await asyncio.wait_for(proc.stdout.readexactly(VAD_FRAME_BYTES), self.stall_s)
                except asyncio.TimeoutError:
                    self._report_error(MIC_HELP)
                    return False
                except asyncio.IncompleteReadError:
                    if self._active and not self.muted and self._runner_gen == gen:
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
            if self._proc is proc:
                await self._close_capture()
            else:
                _kill(proc)
                try:
                    await asyncio.wait_for(proc.wait(), DEVICE_TIMEOUT_S)
                except asyncio.TimeoutError:
                    pass
                self.bus.emit("audio.capture_stopped")
            if self._runner_gen == gen:
                self._drop_turn()

    def _handle_frame(self, frame: bytes) -> None:
        now = time.monotonic()
        # Ignore microphone frames only while models are still loading at startup.
        # During State.PROCESSING (transcribing a mid-sentence pause), keep listening
        # so the second half of a paused sentence is never thrown away.
        if self._runtime_state == "loading":
            if self.detector.in_speech:
                self.detector.reset()
            self._candidate_barge_in = False
            self._candidate_check_echo = False
            self._loud_run = 0
            return
        if not self.voice_barge_in and not self.smart_barge_in:
            if self._runtime_state in ("thinking", "speaking") or self._autum_speaking:
                self._cooldown_until = now + self.post_speech_cooldown_s
                self._turn = None
                self.detector.reset()
                return
            if now < self._cooldown_until:
                self._turn = None
                self.detector.reset()
                return

        # In wake_word mode, ALWAYS buffer & verify for the wake word silently first
        # (never flicker the bubble to "Hearing you"/"Processing", never cancel an
        # active THINKING/SPEAKING reply, and never trigger loud-RMS cuts on TV/room
        # speech without checking for "Autum" or an open wake window!).
        is_wake_mode = str(getattr(self.runtime, "mode", "")) == "wake_word"
        in_wake_window = bool(getattr(self.runtime, "in_wake_window", False))
        # Only treat incoming audio as a speaker-barge-in window while Autum is
        # actually speaking out of the speakers (or during the brief post-speech echo tail).
        in_barge_window = self.smart_barge_in and (
            self._runtime_state == "speaking" or now < self._cooldown_until
        )
        silent_verify = is_wake_mode or in_barge_window
        frame_rms = TurnDetector._frame_rms(frame) if self.smart_barge_in else 1.0
        if in_barge_window and not self.detector.in_speech and frame_rms < 0.055:
            self._bleed_rms = 0.90 * self._bleed_rms + 0.10 * max(0.012, frame_rms)

        min_rms = max(0.018, self._bleed_rms * 1.25) if in_barge_window else 0.012
        event = self.detector.feed(frame, speaking=in_barge_window or self._autum_speaking, min_rms=min_rms)

        # In hands_free smart_barge_in mode (NOT wake_word mode!), loud close-mic voice
        # cuts playback immediately without waiting for end-of-sentence.
        if (
            not is_wake_mode
            and in_barge_window
            and self._candidate_barge_in
            and self.detector.in_speech
            and self._turn is not None
        ):
            if frame_rms >= max(0.085, self._bleed_rms * 2.4):
                self._loud_run += 1
                if self._loud_run >= self.detector.barge_in_frames:
                    self._candidate_barge_in = False
                    self._loud_run = 0
                    self.bus.emit(
                        "audio.speech_started",
                        {"mode": "hands_free"},
                        provenance=Provenance.USER,
                        turn_id=self._turn,
                    )
            else:
                self._loud_run = 0

        if self._candidate_barge_in and in_barge_window:
            self._candidate_check_echo = True

        if event is None:
            return
        if event.kind == "speech_started":
            self._turn = new_id()
            self._loud_run = 0
            # In WAITING_FOR_APPROVAL, playback has already finished and ready_at_ns guards
            # anti-spoofing, so accept approval replies directly even during the 0.45s cooldown.
            accept_direct = (
                self._runtime_state == "waiting_for_approval"
                or (
                    not in_barge_window
                    and is_wake_mode
                    and in_wake_window
                    and self._runtime_state in ("idle", "listening", "processing")
                )
            )
            if silent_verify and not accept_direct:
                # Buffer quietly first and verify via wake-word / non-echo transcript
                # without flickering the bubble or cancelling an in-progress reply.
                self._candidate_barge_in = True
                self._candidate_check_echo = in_barge_window
                self._candidate_started_ns = time.monotonic_ns()
                return
            self._candidate_barge_in = False
            self._candidate_check_echo = False
            self._candidate_started_ns = 0
            self.bus.emit(
                "audio.speech_started",
                {"mode": "hands_free"},
                provenance=Provenance.USER,
                turn_id=self._turn,
            )
        elif event.kind == "speech_ended" and self._turn is not None:
            turn, self._turn = self._turn, None
            was_candidate = self._candidate_barge_in
            was_check_echo = self._candidate_check_echo
            was_started_ns = self._candidate_started_ns
            self._candidate_barge_in = False
            self._candidate_check_echo = False
            self._candidate_started_ns = 0
            self._loud_run = 0
            waiting_approval = str(getattr(self.runtime, "state", "")) == "waiting_for_approval"
            min_s = MIN_APPROVAL_UTTERANCE_S if waiting_approval else MIN_UTTERANCE_S
            if len(event.pcm) < min_s * MIC_RATE * 2:
                if not was_candidate:
                    self.bus.emit(
                        "audio.speech_ended",
                        {"confidence": round(event.confidence, 3)},
                        provenance=Provenance.USER,
                        turn_id=turn,
                    )
                    self._nothing_heard(turn)
                return
            wav = wav_bytes(highpass_pcm(event.pcm, MIC_RATE), MIC_RATE)
            if was_candidate:
                self._track(
                    self._verify_barge_in(
                        turn, wav, check_echo=was_check_echo, captured_at_ns=was_started_ns
                    )
                )
                return
            ended = self.bus.emit(
                "audio.speech_ended",
                {"confidence": round(event.confidence, 3)},
                provenance=Provenance.USER,
                turn_id=turn,
            )
            track_direct = not waiting_approval and not is_wake_mode and getattr(self.runtime, "direct_audio", True)
            if track_direct:
                while len(self._awaiting_transcript) >= 8:
                    self._awaiting_transcript.pop(next(iter(self._awaiting_transcript)))
                self._awaiting_transcript[turn] = wav
            accepted = self.runtime.submit_speech(turn, wav, ended)
            if accepted is False and track_direct:
                self._awaiting_transcript.pop(turn, None)

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
        self._candidate_barge_in = False
        self._candidate_check_echo = False
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
            to_state = str(event.payload["to"])
            was_speaking = self._runtime_state == "speaking"
            self._runtime_state = to_state
            self._autum_speaking = to_state in ("speaking", "thinking")
            if not self.voice_barge_in and not self.smart_barge_in and (
                to_state in ("processing", "thinking", "speaking") or was_speaking
            ):
                if was_speaking and to_state != "speaking":
                    self._cooldown_until = time.monotonic() + self.post_speech_cooldown_s
                self._turn = None
                self.detector.reset()
            elif self.smart_barge_in and was_speaking and to_state != "speaking":
                self._cooldown_until = time.monotonic() + self.post_speech_cooldown_s
        elif event.type in ("speech.playback_started", "speech.output_level", "speech.playback_finished"):
            if not self.voice_barge_in and not self.smart_barge_in:
                self._cooldown_until = time.monotonic() + self.post_speech_cooldown_s
                self._turn = None
                self.detector.reset()
            elif self.smart_barge_in and event.type == "speech.playback_finished":
                self._cooldown_until = time.monotonic() + self.post_speech_cooldown_s
        elif event.type == "runtime.component_error":
            if event.turn_id in self._awaiting_transcript:
                self._awaiting_transcript.pop(event.turn_id, None)
        elif event.turn_id in self._awaiting_transcript:
            self._track(self._transcribe(event.turn_id))

    async def _verify_barge_in(
        self,
        turn: str,
        wav: bytes,
        *,
        check_echo: bool = True,
        captured_at_ns: int | None = None,
    ) -> None:
        """Transcribe a candidate utterance captured while Autum was speaking or in wake_word mode.

        If it is her own speaker echo (or background speech without 'Autum' in wake_word mode),
        discard it silently so she keeps listening/thinking/talking without flickering the UI.
        If it is a user wake call ('Autum, ...') or a valid follow-up, dispatch it cleanly.
        """
        try:
            text = (await self.transcriber.transcribe(wav)).strip()
        except Exception as exc:
            now = time.monotonic()
            if not self._tx_errored or now - self._last_tx_err_at >= 10.0:
                self._tx_errored = True
                self._last_tx_err_at = now
                self.bus.emit("runtime.component_error", {"component": "transcription", "error": str(exc)})
            return
        if self._tx_errored:
            self._tx_errored = False
            self.bus.emit("runtime.component_recovered", {"component": "transcription"})
        if not text:
            return
        barge_fn = getattr(self.runtime, "barge_in_with_text", None)
        if callable(barge_fn):
            try:
                barge_fn(text, turn_id=turn, check_echo=check_echo, captured_at_ns=captured_at_ns)
            except TypeError:
                try:
                    barge_fn(text, turn_id=turn, check_echo=check_echo)
                except TypeError:
                    barge_fn(text, turn_id=turn)

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
