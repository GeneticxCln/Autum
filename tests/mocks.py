"""Mock components that follow the M0 interfaces, for runtime tests."""

from __future__ import annotations

import asyncio
import time
from collections import defaultdict
from typing import AsyncIterator, Sequence

from autum.bus import EventBus
from autum.events import Event, Provenance, new_id
from autum.runtime import Message


class Recorder:
    """Subscribes to everything and keeps the events in delivery order."""

    def __init__(self, bus: EventBus) -> None:
        self.events: list[Event] = []
        bus.subscribe(self.events.append)

    def types(self) -> list[str]:
        return [e.type for e in self.events]

    def of(self, type: str) -> list[Event]:
        return [e for e in self.events if e.type == type]

    def states(self) -> list[str]:
        return [e.payload["to"] for e in self.of("runtime.state_changed")]


class MockInference:
    def __init__(
        self,
        reply: str = "Hello there. This is a mock reply, spoken in a few sentences. Goodbye.",
        token_delay_s: float = 0.002,
        *,
        ignore_cancel: bool = False,
        fail_after: int | None = None,
    ) -> None:
        self.tokens = [w + " " for w in reply.split()]
        self.token_delay_s = token_delay_s
        self.ignore_cancel = ignore_cancel
        self.fail_after = fail_after
        self.stall_after_cancel_s = 0.3
        self.calls: list[list[Message]] = []
        self.ignored_cancels = 0

    async def stream(
        self, messages: Sequence[Message], *, operation_id: str
    ) -> AsyncIterator[str]:
        self.calls.append(list(messages))
        for i, token in enumerate(self.tokens):
            if self.fail_after is not None and i == self.fail_after:
                raise RuntimeError("mock inference failure")
            try:
                await asyncio.sleep(self.token_delay_s)
            except asyncio.CancelledError:
                if not self.ignore_cancel:
                    raise
                self.ignored_cancels += 1
                await asyncio.sleep(self.stall_after_cancel_s)  # a hung model call
            yield token


class MockSpeech:
    """Pretends to synthesize and play audio with realistic ordering."""

    def __init__(self, bus: EventBus, synth_s: float = 0.003, per_char_s: float = 0.0005) -> None:
        self.bus = bus
        self.synth_s = synth_s
        self.per_char_s = per_char_s
        self.audible: list[tuple[str, str, int]] = []  # (operation, text, mono_ns)
        self._queue: asyncio.Queue[tuple[str, str | None, str]] = asyncio.Queue()
        self._pending: defaultdict[str, int] = defaultdict(int)
        self._done: dict[str, asyncio.Event] = {}
        self._played: defaultdict[str, list[str]] = defaultdict(list)
        self._started: set[str] = set()
        self._current: str | None = None
        self._worker: asyncio.Task[None] | None = None

    def enqueue(self, text: str, *, operation_id: str, turn_id: str | None) -> None:
        if self._worker is None:
            self._worker = asyncio.get_running_loop().create_task(self._run())
        self._current = operation_id
        self._pending[operation_id] += 1
        self._done.setdefault(operation_id, asyncio.Event()).clear()
        self._queue.put_nowait((operation_id, turn_id, text))

    async def drain(self, operation_id: str) -> None:
        if self._pending[operation_id]:
            await self._done[operation_id].wait()
        if operation_id in self._started:
            self._finish(operation_id, "completed")

    def stop(self) -> str:
        if self._worker is not None:
            self._worker.cancel()
            self._worker = None
        while not self._queue.empty():
            self._queue.get_nowait()
        for op, count in list(self._pending.items()):
            if count:
                self._pending[op] = 0
                self._done[op].set()
            if op in self._started:
                self._finish(op, "cancelled")
        op, self._current = self._current, None
        return " ".join(self._played.pop(op, [])) if op else ""

    def _finish(self, op: str, reason: str) -> None:
        self._started.discard(op)
        self.bus.emit("speech.playback_finished", {"reason": reason}, operation_id=op)

    async def _run(self) -> None:
        while True:
            op, turn, text = await self._queue.get()
            await asyncio.sleep(self.synth_s)
            if op not in self._started:
                self._started.add(op)
                self.bus.emit("speech.playback_started", turn_id=turn, operation_id=op)
            self._played[op].append(text)
            self.audible.append((op, text, time.monotonic_ns()))
            self.bus.emit("speech.output_level", {"rms": 0.2}, turn_id=turn, operation_id=op)
            await asyncio.sleep(len(text) * self.per_char_s)
            self._pending[op] -= 1
            if not self._pending[op]:
                self._done[op].set()


class MockMic:
    """Stands in for the voice engine's input side."""

    def __init__(self, bus: EventBus, mode: str = "push_to_talk") -> None:
        self.bus = bus
        self.mode = mode

    def start_speaking(self) -> str:
        turn = new_id()
        self.bus.emit(
            "audio.speech_started", {"mode": self.mode}, provenance=Provenance.USER, turn_id=turn
        )
        return turn

    def finish(self, turn: str, text: str) -> None:
        self.bus.emit("audio.speech_ended", {"confidence": 0.95}, provenance=Provenance.USER, turn_id=turn)
        self.bus.emit(
            "audio.transcribed", {"text": text, "final": True}, provenance=Provenance.USER, turn_id=turn
        )

    async def say(self, text: str, speak_s: float = 0.005) -> str:
        turn = self.start_speaking()
        await asyncio.sleep(speak_s)
        self.finish(turn, text)
        return turn
