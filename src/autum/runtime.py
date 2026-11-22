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
import inspect
import json
import logging
import time
from collections import deque
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Mapping, Protocol, Sequence

from autum.bus import EventBus
from autum.events import Event, Provenance, new_id
from autum.text import SentenceChunker, speakable
from autum.tools import ApprovalGate, Capability, PermissionLevel, ToolCall, ToolRegistry

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
    S.SPEAKING: frozenset({S.THINKING, S.WAITING_FOR_APPROVAL, S.INTERRUPTED, S.LISTENING, S.IDLE, S.ERROR}),
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
        max_context_chars: int = 10_000,
        cancel_timeout_s: float = 1.0,
        tools: ToolRegistry | None = None,
        approval_gate: ApprovalGate | None = None,
        transcriber: Any | None = None,
        memory: Any | None = None,
        max_agent_steps: int = 5,
        direct_audio: bool = True,
    ) -> None:
        self.bus = bus
        self.inference = inference
        self.speech = speech
        self.mode = mode
        self.system_prompt = system_prompt
        self.history: deque[dict[str, Any]] = deque(maxlen=max_history)
        self.max_context_chars = max_context_chars
        self.cancel_timeout_s = cancel_timeout_s
        self.tools = tools
        self.approval_gate = approval_gate or ApprovalGate()
        self.transcriber = transcriber
        self.memory = memory
        self.max_agent_steps = max_agent_steps
        self.direct_audio = direct_audio
        self.state = State.IDLE
        self.muted = False
        self._capture_count = 0
        self.observation = "none"
        self._obs_stack: list[str] = []
        self._op: _Operation | None = None
        self._input_turn: str | None = None
        self._transcribe_task: asyncio.Task[None] | None = None
        self._approval_turn: tuple[str, str, int] | None = None  # (turn_id, source_mode, started_mono_ns)
        self._turn_observed = False
        self._pending_announcements: deque[str] = deque(maxlen=16)
        self._response_errored = False
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
    def capturing(self) -> bool:
        return self._capture_count > 0

    @capturing.setter
    def capturing(self, value: bool) -> None:
        self._capture_count = 1 if value else 0

    @property
    def turn_observed(self) -> bool:
        return self._turn_observed

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

    def set_mode(self, mode: Mode) -> None:
        """Switch between push-to-talk and hands-free mode."""
        if mode is self.mode:
            return
        self.mode = mode
        target = self.rest_state if self.state in (State.IDLE, State.LISTENING) else self.state
        self._set_state(target, "mode_changed")

    def begin_model_load(self, model: str) -> None:
        """Announce model loading and enter State.LOADING when at rest."""
        ev = self.bus.emit("runtime.model_loading", {"model": model})
        if State.LOADING in TRANSITIONS[self.state]:
            self._set_state(State.LOADING, "model_loading", ev)

    def finish_model_load(self, model: str, load_ms: float) -> None:
        """Announce model readiness and return to rest state."""
        ev = self.bus.emit("runtime.model_ready", {"model": model, "load_ms": round(load_ms, 1)})
        if self.state is State.LOADING:
            self._set_state(self.rest_state, "model_ready", ev)

    def set_observation(self, source: str) -> None:
        """Update the active observation indicator and emit perception events (§3.2, §5.4).

        Tracks nested observations (e.g. reading a document while live screen
        watching is active) so ending a transient observation never clears an
        ongoing screen watch indicator.
        """
        clean = (source or "none").strip() or "none"
        if clean != "none":
            if self._obs_stack and self._obs_stack[-1] == clean:
                return
            self._obs_stack.append(clean)
            self.observation = clean
            cause = self.bus.emit("perception.started", {"source": clean})
            self._set_state(self.state, "observation", cause)
            return
        if not self._obs_stack:
            return
        ended = self._obs_stack.pop()
        self.observation = self._obs_stack[-1] if self._obs_stack else "none"
        cause = self.bus.emit("perception.completed", {"source": ended})
        self._set_state(self.state, "observation", cause)

    def end_observation(self, source: str) -> None:
        """Remove a specific active observation source (e.g. stopping live screen watch)."""
        clean = (source or "").strip()
        if not clean or clean not in self._obs_stack:
            return
        idx = len(self._obs_stack) - 1 - self._obs_stack[::-1].index(clean)
        self._obs_stack.pop(idx)
        self.observation = self._obs_stack[-1] if self._obs_stack else "none"
        cause = self.bus.emit("perception.completed", {"source": clean})
        self._set_state(self.state, "observation", cause)

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
                "observation": self.observation,
                "mode": self.mode.value,
            },
            cause=cause,
        )

    # ----- input events --------------------------------------------------

    def _on_event(self, event: Event) -> None:
        match event.type:
            case "audio.mute_changed":
                self.muted = bool(event.payload["muted"])
                self._indicator_changed(event)
            case "audio.capture_started":
                self._capture_count += 1
                self._indicator_changed(event)
            case "audio.capture_stopped":
                self._capture_count = max(0, self._capture_count - 1)
                self._indicator_changed(event)
            case "audio.speech_started":
                self._on_speech_started(event)
            case "audio.speech_ended":
                if event.turn_id == self._input_turn and self.state is State.SPEECH_DETECTED:
                    self._set_state(State.PROCESSING, "speech_ended", event)
            case "audio.transcribed":
                if not event.payload["final"]:
                    return
                if self._approval_turn is not None and event.turn_id == self._approval_turn[0]:
                    _, src_mode, started_ns = self._approval_turn
                    self._approval_turn = None
                    self.resolve_approval(
                        str(event.payload["text"]),
                        provenance=event.provenance,
                        source=src_mode,
                        captured_at_ns=started_ns,
                    )
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
        if self.muted or self.state is State.LOADING:
            return  # ignore speech while muted or still loading models at startup
        # While waiting for approval after the prompt finished playing, capture
        # the user's yes/no reply without cancelling the pending approval.
        pending = self.approval_gate.pending
        if (
            self.state is State.WAITING_FOR_APPROVAL
            and pending is not None
            and pending.ready_at_ns is not None
            and event.mono_ns >= pending.ready_at_ns
        ):
            src_mode = str(event.payload["mode"])
            # §5.2: never let open-mic hands-free speech clobber a push-to-talk
            # approval or steal an irreversible approval in hands-free mode.
            if src_mode == Mode.HANDS_FREE and (
                (pending.level is PermissionLevel.IRREVERSIBLE and self.mode is Mode.HANDS_FREE)
                or (self._approval_turn is not None and self._approval_turn[1] == Mode.PUSH_TO_TALK)
            ):
                return
            self._approval_turn = (event.turn_id or new_id(), src_mode, event.mono_ns)
            return
        self._input_turn = event.turn_id
        if self._op is not None:
            source = "push_to_talk" if event.payload["mode"] == Mode.PUSH_TO_TALK else "barge_in"
            self._begin_cancel(source, event)
        self._set_state(State.SPEECH_DETECTED, "speech_started", event)

    def _fill_transcript(self, event: Event) -> None:
        """A transcript for a turn the model already heard as audio."""
        for entry in self.history:
            if entry.get("turn_id") == event.turn_id and ("audio" in entry or not entry.get("content")):
                entry.pop("audio", None)
                entry["content"] = str(event.payload["text"]).strip()
        self._trim_history()

    def _drop_turn_audio(self, turn_id: str) -> None:
        """Release raw WAV bytes once a turn has finished or been cancelled (plan §5.3)."""
        for entry in self.history:
            if entry.get("turn_id") == turn_id:
                entry.pop("audio", None)

    def _trim_history(self) -> None:
        """Keep conversation context within max_context_chars so long sessions stay bounded."""
        total = sum(len(str(entry.get("content") or "")) for entry in self.history)
        while len(self.history) > 1 and total > self.max_context_chars:
            dropped = self.history.popleft()
            total -= len(str(dropped.get("content") or ""))

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
        audio in the history. If waiting for approval, transcribes the reply
        to resolve the pending approval instead of starting a new turn.
        """
        if self._approval_turn is not None and turn_id == self._approval_turn[0]:
            _, src_mode, started_ns = self._approval_turn
            self._approval_turn = None
            if self.transcriber is not None:
                task = asyncio.get_running_loop().create_task(
                    self._resolve_approval_audio(audio, src_mode, started_ns)
                )
                self._track(task)
            return
        if turn_id != self._input_turn:
            return  # superseded by a newer utterance
        if not self.direct_audio and self.transcriber is not None:
            if self._transcribe_task is not None and not self._transcribe_task.done():
                self._transcribe_task.cancel()
            task = asyncio.get_running_loop().create_task(self._transcribe_and_respond(turn_id, audio))
            self._transcribe_task = task
            self._track(task)
            return
        self._input_turn = None
        self._start_response("", "speech", turn_id, cause, audio=audio)

    async def _transcribe_and_respond(self, turn_id: str, wav: bytes) -> None:
        try:
            text = await self.transcriber.transcribe(wav)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self.bus.emit("runtime.component_error", {"component": "transcription", "error": str(exc)})
            if turn_id == self._input_turn:
                self._input_turn = None
                self._set_state(self.rest_state, "transcription_failed")
            return
        finally:
            if self._transcribe_task is asyncio.current_task():
                self._transcribe_task = None
        if turn_id != self._input_turn:
            return  # superseded by cancel or a newer turn while transcribing
        self.bus.emit("audio.transcribed", {"text": text, "final": True}, provenance=Provenance.USER, turn_id=turn_id)

    async def _resolve_approval_audio(self, wav: bytes, source: str, started_ns: int) -> None:
        try:
            text = await self.transcriber.transcribe(wav)
        except Exception as exc:
            self.bus.emit("runtime.component_error", {"component": "transcription", "error": str(exc)})
            return
        self.resolve_approval(text, provenance=Provenance.USER, source=source, captured_at_ns=started_ns)

    def submit_text(self, text: str) -> None:
        """Text input (plan §3.1): resolves a pending approval or starts a spoken turn."""
        text = text.strip()
        if not text:
            return
        if self.state is State.WAITING_FOR_APPROVAL and self.approval_gate.pending is not None:
            self.resolve_approval(text, provenance=Provenance.USER, source="text")
            return
        if self.state is State.LOADING:
            return
        self._cancel_pending_transcription()
        if self._op is not None:
            self._begin_cancel("user_command", None)
        self._start_response(text, "text", new_id(), None)

    def _cancel_pending_transcription(self) -> bool:
        had_pending = self._input_turn is not None or (
            self._transcribe_task is not None and not self._transcribe_task.done()
        )
        self._input_turn = None
        if self._transcribe_task is not None and not self._transcribe_task.done():
            self._transcribe_task.cancel()
        self._transcribe_task = None
        return had_pending

    def resolve_approval(
        self,
        response: str,
        *,
        provenance: Provenance = Provenance.USER,
        source: str = "command",
        captured_at_ns: int | None = None,
    ) -> str | None:
        """Resolve a pending permission request through ApprovalGate (§5.2)."""
        return self.approval_gate.resolve(
            response,
            provenance=provenance,
            source=source,
            hands_free_mode=(self.mode is Mode.HANDS_FREE),
            captured_at_ns=captured_at_ns,
        )

    def announce(self, text: str) -> None:
        """Speak a short system announcement (such as a finished timer).

        If Autum is currently busy, queues the announcement and speaks it as
        soon as she returns to rest instead of dropping it.
        """
        spoken = speakable(text)
        if not spoken:
            return
        if self._op is not None or self.state not in (State.IDLE, State.LISTENING):
            self._pending_announcements.append(spoken)
            return
        self._start_announcement(spoken)

    def _start_announcement(self, spoken: str) -> None:
        op_id = new_id()
        self.bus.fence.open(op_id)
        self._set_state(State.THINKING, "announcement")
        task = asyncio.get_running_loop().create_task(self._speak_announcement(op_id, spoken))
        self._op = _Operation(op_id, new_id(), task)
        self._track(task)

    def _drain_pending_announcements(self) -> None:
        if (
            self._pending_announcements
            and self._op is None
            and self.state in (State.IDLE, State.LISTENING)
        ):
            self._start_announcement(self._pending_announcements.popleft())

    async def _speak_announcement(self, op_id: str, text: str) -> None:
        try:
            self.speech.enqueue(text, operation_id=op_id, turn_id=None)
            await self.speech.drain(op_id)
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            self._fail(op_id, exc)
            return
        if self._op is None or self._op.id != op_id:
            return
        self._op = None
        self.bus.fence.close(op_id)
        self._set_state(self.rest_state, "announcement_finished")
        self._drain_pending_announcements()

    async def cancel(self, source: str = "user_command") -> bool:
        """Cancel the active response or pending transcription. Returns False if nothing was active."""
        had_tx = self._cancel_pending_transcription()
        finishing = self._begin_cancel(source, None)
        if finishing is None:
            if had_tx and self.state in (State.SPEECH_DETECTED, State.PROCESSING):
                self._set_state(self.rest_state, source)
                return True
            return had_tx
        await finishing
        return True

    async def wait_until_settled(self) -> None:
        """Wait for every response and cancellation task (used by tests and shutdown)."""
        while self._tasks:
            await asyncio.wait(set(self._tasks))

    # ----- responses & agent loop (WP3 / WP5 / WP6) ----------------------

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
        self._trim_history()
        self._set_state(State.THINKING, "user_message", message)
        task = asyncio.get_running_loop().create_task(self._respond(op_id, turn_id, message))
        self._op = _Operation(op_id, turn_id, task)
        self._track(task)

    def _stream_inference(self, messages: Sequence[Message], op_id: str, tools: list[dict[str, Any]] | None):
        if tools:
            sig = inspect.signature(self.inference.stream)
            if "tools" in sig.parameters:
                return self.inference.stream(messages, operation_id=op_id, tools=tools)
        return self.inference.stream(messages, operation_id=op_id)

    async def _request_approval(
        self,
        cap: Capability,
        args: dict[str, Any],
        op_id: str,
        tc_id: str,
        target: str | None = None,
    ) -> str:
        resolved_target = target if target is not None else cap.target_summary(args)
        pending = self.approval_gate.open(
            action=cap.name,
            target=resolved_target,
            level=cap.permission_level,
            operation_id=op_id,
            tool_call_id=tc_id,
        )
        if cap.permission_level is PermissionLevel.IRREVERSIBLE and self.mode is Mode.HANDS_FREE:
            prompt = f"Should I {resolved_target}? Hold your talk key and say yes to confirm, or no to cancel."
        else:
            prompt = f"Should I {resolved_target}? Say yes to confirm, or no to cancel."
        self._speak([prompt], op_id, None)
        await self.speech.drain(op_id)
        if not self.bus.fence.is_live(op_id):
            self.approval_gate.cancel()
            return "denied"
        self.approval_gate.mark_prompt_finished()
        if State.WAITING_FOR_APPROVAL in TRANSITIONS[self.state]:
            self._set_state(State.WAITING_FOR_APPROVAL, "approval_required")
        outcome = await self.approval_gate.wait(pending)
        if self.state is State.WAITING_FOR_APPROVAL and self.bus.fence.is_live(op_id):
            self._set_state(State.THINKING, f"approval_{outcome}")
        return outcome

    async def _respond(self, op_id: str, turn_id: str, cause: Event | None) -> None:
        messages: list[dict[str, Any]] = []
        prompt = self.system_prompt() if callable(self.system_prompt) else self.system_prompt
        if self.memory is not None:
            latest_user = ""
            for entry in reversed(self.history):
                if entry.get("role") == "user" and entry.get("content"):
                    latest_user = str(entry["content"])
                    break
            selected = self.memory.select_for_context(latest_user)
            if selected:
                facts = "; ".join(f"{m.key}: {m.value}" for m in selected)
                mem_note = f" User-selected persistent memories: {facts}."
                prompt = f"{prompt}{mem_note}" if prompt else mem_note.strip()
        if prompt:
            messages.append({"role": "system", "content": prompt})
        messages.extend(dict(entry) for entry in self.history)
        tool_schemas = self.tools.openai_tools() if self.tools is not None else None
        max_steps = self.max_agent_steps if tool_schemas else 1
        emitted_parts: list[str] = []
        final_text = ""
        self._turn_observed = False
        try:
            started = self.bus.emit(
                "conversation.response_started", turn_id=turn_id, operation_id=op_id, cause=cause
            )
            for step in range(max_steps):
                if not self.bus.fence.is_live(op_id):
                    return
                if tool_schemas:
                    self.bus.emit("agent.thinking", turn_id=turn_id, operation_id=op_id, cause=started)
                chunker = SentenceChunker()
                parts: list[str] = []
                tool_calls: list[ToolCall] = []
                active_tools = tool_schemas if step < max_steps - 1 else None
                async for item in self._stream_inference(messages, op_id, active_tools):
                    if not self.bus.fence.is_live(op_id):
                        break  # cancelled, but the component ignored it
                    if isinstance(item, ToolCall):
                        tool_calls.append(item)
                        continue
                    parts.append(item)
                    if not tool_calls:
                        self.bus.emit(
                            "conversation.response_chunk",
                            {"text": item},
                            provenance=Provenance.MODEL,
                            turn_id=turn_id,
                            operation_id=op_id,
                            cause=started,
                        )
                        self._speak(chunker.feed(item), op_id, turn_id)
                if not self.bus.fence.is_live(op_id):
                    return
                if tool_calls and self.tools is not None:
                    preamble = "".join(parts)
                    if preamble:
                        self._speak(chunker.flush(), op_id, turn_id)
                        emitted_parts.append(preamble)
                        if self.state is State.SPEAKING:
                            await self.speech.drain(op_id)
                            if not self.bus.fence.is_live(op_id):
                                return
                            if self.state is State.SPEAKING:
                                self._set_state(State.THINKING, "tool_execution", started)
                    messages.append(
                        {
                            "role": "assistant",
                            "content": preamble,
                            "tool_calls": [
                                {
                                    "id": tc.id,
                                    "type": "function",
                                    "function": {
                                        "name": tc.name,
                                        "arguments": json.dumps(tc.arguments, separators=(",", ":")),
                                    },
                                }
                                for tc in tool_calls
                            ],
                        }
                    )
                    for tc in tool_calls:
                        if not self.bus.fence.is_live(op_id):
                            return
                        outcome = await self.tools.execute(
                            tc,
                            bus=self.bus,
                            operation_id=op_id,
                            turn_id=turn_id,
                            approval_gate=self.approval_gate,
                            request_approval=self._request_approval,
                        )
                        if outcome.observed and outcome.succeeded:
                            self._turn_observed = True
                        messages.append(
                            {
                                "role": "tool",
                                "tool_call_id": tc.id,
                                "content": json.dumps(outcome.to_model_payload(), separators=(",", ":")),
                            }
                        )
                    continue
                self._speak(chunker.flush(), op_id, turn_id)
                step_text = "".join(parts)
                final_text = (
                    " ".join(p.strip() for p in (*emitted_parts, step_text) if p.strip())
                    if emitted_parts
                    else step_text
                )
                break

            finished = self.bus.emit(
                "conversation.response_finished",
                {"text": final_text},
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
            self._drop_turn_audio(turn_id)
            return  # superseded while draining
        self._drop_turn_audio(turn_id)
        self.history.append({"role": "assistant", "content": final_text})
        self._trim_history()
        self._op = None
        self.bus.fence.close(op_id)
        if self._response_errored:
            self._response_errored = False
            self.bus.emit("runtime.component_recovered", {"component": "response"})
        self._turn_observed = False
        self._set_state(self.rest_state, "response_finished", finished)
        self._drain_pending_announcements()

    def _speak(self, chunks: list[str], op_id: str, turn_id: str | None) -> None:
        for chunk in chunks:
            spoken = speakable(chunk)
            if spoken and self.bus.fence.is_live(op_id):
                self.speech.enqueue(spoken, operation_id=op_id, turn_id=turn_id)

    def _fail(self, op_id: str, exc: Exception) -> None:
        if self._op is None or self._op.id != op_id:
            return
        turn_id = self._op.turn_id
        self.approval_gate.cancel()
        self.speech.stop()
        self._drop_turn_audio(turn_id)
        self._response_errored = True
        error = self.bus.emit(
            "runtime.component_error",
            {"component": "response", "error": f"{type(exc).__name__}: {exc}"},
            operation_id=op_id,
        )
        self._op = None
        self._turn_observed = False
        self.bus.fence.close(op_id)
        self._set_state(State.ERROR, "response_failed", error)
        self._set_state(self.rest_state, "recovered", error)
        self._drain_pending_announcements()

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
        self._approval_turn = None
        self._turn_observed = False
        self.approval_gate.cancel()
        self._drop_turn_audio(op.turn_id)
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
            self._drain_pending_announcements()

    def _track(self, task: asyncio.Task[None]) -> None:
        self._tasks.add(task)
        task.add_done_callback(self._tasks.discard)
