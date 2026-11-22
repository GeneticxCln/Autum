"""In-process event bus with a total order and an operation fence.

Ordering: every subscriber receives every event in the same order the
events were published. A subscriber that publishes while handling an
event does not jump the queue: its event is delivered after the current
one has reached all subscribers.

Callbacks run synchronously on the event loop and must return quickly.
Work that takes time is scheduled as a task by the subscriber.

Fence: once an operation is cancelled or finished, its output events are
dropped here, so no consumer has to remember which operations are stale
(plan §4.2).
"""

from __future__ import annotations

import enum
import logging
from collections import deque
from typing import Any, Callable, Iterable, Mapping

from autum.events import Event, Provenance

log = logging.getLogger("autum.bus")

Subscriber = Callable[[Event], None]

# Event types still accepted from an operation that is being cancelled:
# the cancellation itself, playback reporting that it stopped, and errors
# from components that failed to stop.
CANCEL_PHASE_TYPES: frozenset[str] = frozenset(
    {
        "operation.cancel_requested",
        "operation.cancelled",
        "speech.playback_finished",
        "runtime.component_error",
    }
)


class OpState(enum.Enum):
    ACTIVE = "active"
    CANCELLING = "cancelling"


class OperationFence:
    """Tracks which operations may still produce output.

    Only open operations are stored, so the fence stays small however long
    the session runs.
    """

    def __init__(self) -> None:
        self._ops: dict[str, OpState] = {}

    def open(self, operation_id: str) -> None:
        if operation_id in self._ops:
            raise ValueError(f"operation {operation_id} is already open")
        self._ops[operation_id] = OpState.ACTIVE

    def begin_cancel(self, operation_id: str) -> None:
        if operation_id in self._ops:
            self._ops[operation_id] = OpState.CANCELLING

    def close(self, operation_id: str) -> None:
        self._ops.pop(operation_id, None)

    def is_live(self, operation_id: str) -> bool:
        return self._ops.get(operation_id) is OpState.ACTIVE

    def allows(self, event: Event) -> bool:
        if event.operation_id is None:
            return True
        state = self._ops.get(event.operation_id)
        if state is OpState.ACTIVE:
            return True
        return state is OpState.CANCELLING and event.type in CANCEL_PHASE_TYPES


class EventBus:
    def __init__(self, session_id: str) -> None:
        self.session_id = session_id
        self.fence = OperationFence()
        self.dropped_stale = 0
        self.subscriber_errors = 0
        self._subscribers: list[tuple[Subscriber, frozenset[str] | None]] = []
        self._pending: deque[Event] = deque()
        self._dispatching = False

    def subscribe(
        self, callback: Subscriber, types: Iterable[str] | None = None
    ) -> Callable[[], None]:
        entry = (callback, frozenset(types) if types is not None else None)
        self._subscribers.append(entry)

        def unsubscribe() -> None:
            if entry in self._subscribers:
                self._subscribers.remove(entry)

        return unsubscribe

    def emit(
        self,
        type: str,
        payload: Mapping[str, Any] | None = None,
        *,
        provenance: Provenance = Provenance.SYSTEM,
        turn_id: str | None = None,
        operation_id: str | None = None,
        tool_call_id: str | None = None,
        cause: Event | None = None,
    ) -> Event | None:
        """Build and publish an event. Returns None if the fence dropped it."""
        event = Event(
            type=type,
            session_id=self.session_id,
            provenance=provenance,
            payload=payload or {},
            turn_id=turn_id,
            operation_id=operation_id,
            tool_call_id=tool_call_id,
            causation_id=cause.event_id if cause else None,
        )
        return event if self.publish(event) else None

    def publish(self, event: Event) -> bool:
        if event.session_id != self.session_id:
            raise ValueError("event belongs to another session")
        if not self.fence.allows(event):
            self.dropped_stale += 1
            return False
        self._pending.append(event)
        if self._dispatching:
            return True
        self._dispatching = True
        try:
            while self._pending:
                current = self._pending.popleft()
                # Re-check: the operation may have been cancelled while
                # this event waited behind a re-entrant publish.
                if not self.fence.allows(current):
                    self.dropped_stale += 1
                    continue
                for callback, types in list(self._subscribers):
                    if types is not None and current.type not in types:
                        continue
                    try:
                        callback(current)
                    except Exception:
                        self.subscriber_errors += 1
                        log.exception("subscriber failed on %s", current.type)
        finally:
            self._dispatching = False
        return True
