"""Core runtime: owns state, conversation turns, and cancellation (plan §3.2, §4.2).

Components publish what they observe (speech started, playback started)
as events. The runtime reacts, owns the single primary state, and runs
each response as one asyncio task. Cancelling that task is how
cancellation reaches inference and synthesis; the operation fence stops
anything a misbehaving component emits afterwards.
"""

from __future__ import annotations

import asyncio
import enum
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Protocol, Sequence

from autum.bus import EventBus
from autum.events import Event, Provenance, new_id
from autum.text import SentenceChunker, speakable

log = logging.getLogger("autum.runtime")

# A chat message: {"role", "content"}. A user turn may also carry "audio"
# (WAV bytes) when the pipeline hears speech directly; its "content" is
# filled in from the transcript once that arrives. "turn_id" ties a history
# entry to its turn.
Message = Mapping[str, Any]


class Inference(Protocol):
    def stream(
        self, messages: Sequence[Message], *, operation_id: str
    ) -> AsyncIterator[str]:
        """Yield response text. Must stop when its task is cancelled."""
        ...


class SpeechOutput(Protocol):
    """Synthesizes and plays speech. Publishes the speech.* events."""

    def enqueue(self, text: str, *, operation_id: str, turn_id: str | None) -> None:
        """Queue text for synthesis and playback; returns immediately."""
        ...

    async def drain(self, operation_id: str) -> None:
        """Return when everything queued for this operation has played."""
        ...

    def stop(self) -> str:
        """Silence output now and drop the queue.

        Returns the text whose playback had started, so the conversation
        history records what the user actually heard.
        """
        ...


class Mode(enum.StrEnum):
    PUSH_TO_TALK = "push_to_talk"
    HANDS_FREE = "hands_free"


class State(enum.StrEnum):
    IDLE = "idle"
    LOADING = "loading"
    LISTENING = "listening"
    SPEECH_DETECTED = "speech_detected"
    PROCESSING = "processing"
    THINKING = "thinking"
    SPEAKING = "speaking"
    INTERRUPTED = "interrupted"
    WAITING_FOR_APPROVAL = "waiting_for_approval"
    ERROR = "error"


S = State
TRANSITIONS: Mapping[State, frozenset[State]] = {
    S.IDLE: frozenset({S.LISTENING, S.SPEECH_DETECTED, S.THINKING, S.LOADING, S.ERROR}),
    S.LOADING: frozenset({S.IDLE, S.LISTENING, S.ERROR}),
    S.LISTENING: frozenset({S.SPEECH_DETECTED, S.THINKING, S.IDLE, S.LOADING, S.ERROR}),
    S.SPEECH_DETECTED: frozenset({S.PROCESSING, S.LISTENING, S.IDLE, S.ERROR}),
    S.PROCESSING: frozenset({S.THINKING, S.SPEECH_DETECTED, S.LISTENING, S.IDLE, S.ERROR}),
    S.THINKING: frozenset({S.SPEAKING, S.WAITING_FOR_APPROVAL, S.INTERRUPTED, S.LISTENING, S.IDLE, S.ERROR}),
    S.SPEAKING: frozenset({S.INTERRUPTED, S.LISTENING, S.IDLE, S.ERROR}),
    S.INTERRUPTED: frozenset({S.SPEECH_DETECTED, S.THINKING, S.LISTENING, S.IDLE, S.ERROR}),
    S.WAITING_FOR_APPROVAL: frozenset({S.THINKING, S.INTERRUPTED, S.LISTENING, S.IDLE, S.ERROR}),
    S.ERROR: frozenset({S.IDLE, S.LISTENING, S.LOADING}),
}


class InvalidTransition(RuntimeError):
    pass


@dataclass
class _Operation:
    id: str
    turn_id: str
    task: asyncio.Task[None]


class Runtime:
    def __init__(
        self,
        bus: EventBus,
        inference: Inference,
        speech: SpeechOutput,
        *,
        mode: Mode = Mode.PUSH_TO_TALK,
        system_prompt: str | Callable[[], str] | None = None,
        max_history: int = 20,
        cancel_timeout_s: float = 1.0,
    ) -> None:
        self.bus = bus
        self.inference = inference
        self.speech = speech
        self.mode = mode
        self.system_prompt = system_prompt
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.cancel_timeout_s = cancel_timeout_s
        self.state = State.IDLE
        self.muted = False
        self.capturing = False
        self._op: _Operation | None = None
        self._input_turn: str | None = None
        self._tasks: set[asyncio.Task[None]] = set()
        bus.subscribe(
            self._on_event,
            {
                "audio.capture_started",
                "audio.capture_stopped",
                "audio.mute_changed",
                "audio.speech_started",
                "audio.speech_ended",
                "audio.transcribed",
                "speech.playback_started",
            },
        )

    # ----- state ---------------------------------------------------------

    @property
    def microphone(self) -> str:
        if self.muted:
            return "muted"
        return "open" if self.capturing else "off"

    @property
    def rest_state(self) -> State:
        if self.mode is Mode.HANDS_FREE and self.capturing and not self.muted:
            return State.LISTENING
        return State.IDLE

    def _set_state(self, new: State, reason: str, cause: Event | None = None) -> None:
        old = self.state
        if new is not old and new not in TRANSITIONS[old]:
            raise InvalidTransition(f"{old} -> {new} ({reason})")
        self.state = new
        self.bus.emit(
            "runtime.state_changed",
            {
                "from": old.value,
                "to": new.value,
                "reason": reason,
                "microphone": self.microphone,
                "observation": "none",
            },
            cause=cause,
        )

    # ----- input events --------------------------------------------------

    def _on_event(self, event: Event) -> None:
        match event.type:
            case "audio.mute_changed":
                self.muted = bool(event.payload["muted"])
                self._indicator_changed(event)
            case "audio.capture_started" | "audio.capture_stopped":
                self.capturing = event.type == "audio.capture_started"
                self._indicator_changed(event)
            case "audio.speech_started":
                self._on_speech_started(event)
            case "audio.speech_ended":
                if event.turn_id == self._input_turn and self.state is State.SPEECH_DETECTED:
                    self._set_state(State.PROCESSING, "speech_ended", event)
            case "audio.transcribed":
                if not event.payload["final"]:
                    return
                if event.turn_id == self._input_turn:
                    self._on_final_transcript(event)
                else:
                    self._fill_transcript(event)
            case "speech.playback_started":
                if self._op and event.operation_id == self._op.id and self.state is State.THINKING:
                    self._set_state(State.SPEAKING, "playback_started", event)

    def _indicator_changed(self, cause: Event) -> None:
        if self.state in (State.IDLE, State.LISTENING):
            self._set_state(self.rest_state, "microphone", cause)
        else:
            self._set_state(self.state, "microphone", cause)

    def _on_speech_started(self, event: Event) -> None:
        if self.muted:
            return  # the voice engine should not report speech while muted
        self._input_turn = event.turn_id
        if self._op is not None:
            source = "push_to_talk" if event.payload["mode"] == Mode.PUSH_TO_TALK else "barge_in"
            self._begin_cancel(source, event)
        self._set_state(State.SPEECH_DETECTED, "speech_started", event)

    def _fill_transcript(self, event: Event) -> None:
        """A transcript for a turn the model already heard as audio."""
        for entry in self.history:
            if entry.get("turn_id") == event.turn_id and "audio" in entry:
                entry.pop("audio")
                entry["content"] = str(event.payload["text"]).strip()

    def _on_final_transcript(self, event: Event) -> None:
        self._input_turn = None
        text = str(event.payload["text"]).strip()
        if not text:
            self._set_state(self.rest_state, "empty_transcript", event)
            return
        self._start_response(text, "speech", event.turn_id or new_id(), event)

    # ----- public API ----------------------------------------------------

    def submit_speech(self, turn_id: str, audio: bytes, cause: Event | None = None) -> None:
        """A finished utterance for a pipeline whose model hears audio directly.

        The transcript, when the voice engine publishes it, replaces the
        audio in the history.
        """
        if turn_id != self._input_turn:
            return  # superseded by a newer utterance
        self._input_turn = None
        self._start_response("", "speech", turn_id, cause, audio=audio)

    def submit_text(self, text: str) -> None:
        """Text input (plan §3.1): same path as a spoken turn."""
        text = text.strip()
        if not text:
            return
        if self._op is not None:
            self._begin_cancel("user_command", None)
        self._start_response(text, "text", new_id(), None)

    async def cancel(self, source: str = "user_command") -> bool:
        """Cancel the active response. Returns False if nothing was active."""
        finishing = self._begin_cancel(source, None)
        if finishing is None:
            return False
        await finishing
        return True

    async def wait_until_settled(self) -> None:
        """Wait for every response and cancellation task (used by tests and shutdown)."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    # ----- responses -----------------------------------------------------

    def _start_response(
        self, text: str, source: str, turn_id: str, cause: Event | None, audio: bytes | None = None
    ) -> None:
        op_id = new_id()
        self.bus.fence.open(op_id)
        message = self.bus.emit(
            "conversation.user_message",
            {"text": text, "source": source},
            provenance=Provenance.USER,
            turn_id=turn_id,
            operation_id=op_id,
            cause=cause,
        )
        entry: dict[str, Any] = {"role": "user", "content": text, "turn_id": turn_id}
        if audio is not None:
            entry["audio"] = audio
        self.history.append(entry)
        self._set_state(State.THINKING, "user_message", message)
        task = asyncio.get_running_loop().create_task(self._respond(op_id, turn_id, message))
        self._op = _Operation(op_id, turn_id, task)
        self._track(task)

    async def _respond(self, op_id: str, turn_id: str, cause: Event | None) -> None:
        messages: list[Message] = []
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(dict(entry) for entry in self.history)
        chunker = SentenceChunker()
        parts: list[str] = []
        try:
            started = self.bus.emit(
                "conversation.response_started", turn_id=turn_id, operation_id=op_id, cause=cause
            )
            async for delta in self.inference.stream(messages, operation_id=op_id):
                if not self.bus.fence.is_live(op_id):
                    break  # cancelled, but the component ignored it
                parts.append(delta)
                self.bus.emit(
                    "conversation.response_chunk",
                    {"text": delta},
                    provenance=Provenance.MODEL,
                    turn_id=turn_id,
                    operation_id=op_id,
                    cause=started,
                )
                self._speak(chunker.feed(delta), op_id, turn_id)
            if not self.bus.fence.is_live(op_id):
                return
            self._speak(chunker.flush(), op_id, turn_id)
            text = "".join(parts)
            finished = self.bus.emit(
                "conversation.response_finished",
                {"text": text},
                provenance=Provenance.MODEL,
                turn_id=turn_id,
                operation_id=op_id,
                cause=started,
            )
            await self.speech.drain(op_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # inference or synthesis failed
            log.exception("response %s failed", op_id)
            self._fail(op_id, exc)
            return
        if self._op is None or self._op.id != op_id:
            return  # superseded while draining
        self.history.append({"role": "assistant", "content": text})
        self._op = None
        self.bus.fence.close(op_id)
        self._set_state(self.rest_state, "response_finished", finished)

    def _speak(self, chunks: list[str], op_id: str, turn_id: str) -> None:
        for chunk in chunks:
            spoken = speakable(chunk)
            if spoken and self.bus.fence.is_live(op_id):
                self.speech.enqueue(spoken, operation_id=op_id, turn_id=turn_id)

    def _fail(self, op_id: str, exc: Exception) -> None:
        if self._op is None or self._op.id != op_id:
            return
        self.speech.stop()
        error = self.bus.emit(
            "runtime.component_error",
            {"component": "response", "error": f"{type(exc).__name__}: {exc}"},
            operation_id=op_id,
        )
        self._op = None
        self.bus.fence.close(op_id)
        self._set_state(State.ERROR, "response_failed", error)
        self._set_state(self.rest_state, "recovered", error)

    # ----- cancellation (plan §4.2) --------------------------------------

    def _begin_cancel(self, source: str, cause: Event | None) -> asyncio.Task[None] | None:
        """Synchronous part: silence audio and fence the operation at once.

        Returns the task that finishes the cancellation, or None if no
        response was active.
        """
        op = self._op
        if op is None:
            return None
        requested_ns = time.monotonic_ns()
        self._op = None
        spoken = self.speech.stop()
        self.bus.fence.begin_cancel(op.id)
        self.bus.emit(
            "operation.cancel_requested",
            {"source": source},
            turn_id=op.turn_id,
            operation_id=op.id,
            cause=cause,
        )
        if spoken:
            self.history.append(
                {"role": "assistant", "content": f"{spoken} [interrupted by the user]"}
            )
        op.task.cancel()
        if State.INTERRUPTED in TRANSITIONS[self.state]:
            self._set_state(State.INTERRUPTED, source, cause)
        finishing = asyncio.get_running_loop().create_task(
            self._finish_cancel(op, source, requested_ns)
        )
        self._track(finishing)
        return finishing

    async def _finish_cancel(self, op: _Operation, source: str, requested_ns: int) -> None:
        """Asynchronous part: wait for the task to stop, then close the operation."""
        done, _ = await asyncio.wait({op.task}, timeout=self.cancel_timeout_s)
        clean = bool(done)
        if not clean:
            self.bus.emit(
                "runtime.component_error",
                {
                    "component": "response",
                    "error": f"did not stop within {self.cancel_timeout_s * 1000:.0f} ms",
                },
                turn_id=op.turn_id,
                operation_id=op.id,
            )
        cancelled = self.bus.emit(
            "operation.cancelled",
            {
                "source": source,
                "elapsed_ms": (time.monotonic_ns() - requested_ns) / 1e6,
                "clean": clean,
            },
            turn_id=op.turn_id,
            operation_id=op.id,
        )
        self.bus.fence.close(op.id)
        if self.state is State.INTERRUPTED:
            self._set_state(self.rest_state, "cancelled", cancelled)

    def _track(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
