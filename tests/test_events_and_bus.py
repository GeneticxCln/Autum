import json
from pathlib import Path

import pytest

from autum.bus import EventBus
from autum.events import EVENT_TYPES, Event, Provenance, new_id

CONTRACT = Path(__file__).resolve().parents[1] / "docs" / "m0-runtime-contract.md"


def make(type="audio.capture_started", **kw):
    kw.setdefault("session_id", "s1")
    kw.setdefault("provenance", Provenance.SYSTEM)
    return Event(type=type, **kw)


# ----- envelope ---------------------------------------------------------


def test_event_round_trips_through_json_and_payload_is_read_only():
    event = make("audio.transcribed", payload={"text": "hi", "final": True}, turn_id="t1")
    decoded = Event.from_json(event.to_json())
    assert decoded == event
    with pytest.raises(TypeError):
        decoded.payload["text"] = "changed"  # type: ignore[index]


@pytest.mark.parametrize(
    "change, message",
    [
        ({"type": "audio.unknown"}, "unknown event type"),
        ({"type": "audio.transcribed", "payload": {"text": "x"}}, "missing"),
        ({"type": "speech.playback_started"}, "operation_id"),
        ({"provenance": "admin"}, "admin"),
        ({"session_id": ""}, "session_id"),
        ({"turn_id": 5}, "turn_id"),
        ({"mono_ns": "soon"}, "mono_ns"),
        ({"schema_version": 99}, "schema version"),
    ],
)
def test_invalid_events_are_rejected(change, message):
    data = json.loads(make().to_json()) | change
    with pytest.raises(ValueError, match=message):
        Event.from_json(json.dumps(data))


def test_unknown_fields_from_the_transport_are_rejected():
    data = json.loads(make().to_json()) | {"authorized": True}
    with pytest.raises(ValueError, match="unknown event fields"):
        Event.from_json(json.dumps(data))


def test_contract_document_lists_exactly_the_implemented_event_types():
    documented = {
        line.split()[0]
        for line in CONTRACT.read_text().splitlines()
        if line and line[0].islower() and "." in line.split()[0] and not line.startswith("#")
        and line.split()[0].split(".")[0] in {t.split(".")[0] for t in EVENT_TYPES}
    }
    assert documented == set(EVENT_TYPES)


# ----- ordering ---------------------------------------------------------


def test_every_subscriber_sees_the_same_order_even_with_reentrant_publish():
    bus = EventBus("s1")
    seen_a, seen_b = [], []

    def a(event):
        seen_a.append(event.type)
        if event.type == "audio.capture_started":
            bus.emit("audio.speech_started", {"mode": "push_to_talk"})

    bus.subscribe(a)
    bus.subscribe(lambda e: seen_b.append(e.type))
    bus.emit("audio.capture_started")
    bus.emit("audio.capture_stopped")

    expected = ["audio.capture_started", "audio.speech_started", "audio.capture_stopped"]
    assert seen_a == expected
    assert seen_b == expected


def test_a_failing_subscriber_does_not_stop_delivery_to_others():
    bus = EventBus("s1")
    seen = []

    def broken(event):
        raise RuntimeError("boom")

    bus.subscribe(broken)
    bus.subscribe(lambda e: seen.append(e.type))
    bus.emit("audio.capture_started")
    assert seen == ["audio.capture_started"]
    assert bus.subscriber_errors == 1


def test_type_filtered_subscription_and_unsubscribe():
    bus = EventBus("s1")
    seen = []
    unsubscribe = bus.subscribe(lambda e: seen.append(e.type), {"audio.capture_stopped"})
    bus.emit("audio.capture_started")
    bus.emit("audio.capture_stopped")
    unsubscribe()
    bus.emit("audio.capture_stopped")
    assert seen == ["audio.capture_stopped"]


def test_events_from_another_session_are_rejected():
    with pytest.raises(ValueError, match="another session"):
        EventBus("s1").publish(make(session_id="s2"))


# ----- fence ------------------------------------------------------------


def test_fence_drops_output_of_cancelled_and_closed_operations():
    bus = EventBus("s1")
    seen = []
    bus.subscribe(lambda e: seen.append(e.type))
    op = new_id()

    assert bus.emit("speech.playback_started", operation_id=op) is None  # never opened
    bus.fence.open(op)
    assert bus.emit("speech.playback_started", operation_id=op) is not None

    bus.fence.begin_cancel(op)
    assert bus.emit("conversation.response_chunk", {"text": "late"}, operation_id=op) is None
    assert bus.emit("speech.playback_finished", {"reason": "cancelled"}, operation_id=op)
    assert bus.emit("operation.cancelled", {"source": "x", "elapsed_ms": 1, "clean": True}, operation_id=op)

    bus.fence.close(op)
    assert bus.emit("operation.cancelled", {"source": "x", "elapsed_ms": 1, "clean": True}, operation_id=op) is None

    assert seen == ["speech.playback_started", "speech.playback_finished", "operation.cancelled"]
    assert bus.dropped_stale == 3


def test_an_event_queued_behind_a_reentrant_publish_is_rechecked_against_the_fence():
    bus = EventBus("s1")
    op = new_id()
    bus.fence.open(op)
    seen = []

    def cancel_on_first(event):
        if event.type == "speech.playback_started":
            bus.emit("conversation.response_chunk", {"text": "queued"}, operation_id=op)
            bus.fence.begin_cancel(op)

    bus.subscribe(cancel_on_first)
    bus.subscribe(lambda e: seen.append(e.type))
    bus.emit("speech.playback_started", operation_id=op)
    assert seen == ["speech.playback_started"]
    assert bus.dropped_stale == 1
