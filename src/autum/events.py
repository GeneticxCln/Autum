"""Runtime event contract (plan §4.1, docs/m0-runtime-contract.md).

Every event is an immutable envelope. Construction validates the event
type, its required payload fields, and its identifiers, so an invalid
event can never enter the bus — whether it was built in-process or
decoded from the transport.
"""

from __future__ import annotations

import enum
import json
import time
import types
import uuid
from dataclasses import dataclass, field, fields
from typing import Any, Mapping

SCHEMA_VERSION = 1


class Provenance(enum.StrEnum):
    """Where an event's content came from. Only USER may authorize actions."""

    USER = "user"
    SYSTEM = "system"
    MODEL = "model"
    OBSERVED = "observed"


# Event type -> payload fields that must be present.
EVENT_TYPES: Mapping[str, frozenset[str]] = types.MappingProxyType(
    {
        name: frozenset(required)
        for name, required in {
            # Runtime
            "runtime.state_changed": {"from", "to", "reason", "microphone", "observation"},
            "runtime.model_loading": {"model"},
            "runtime.model_ready": {"model", "load_ms"},
            "runtime.component_error": {"component", "error"},
            "runtime.component_recovered": {"component"},
            # Audio input
            "audio.capture_started": set(),
            "audio.capture_stopped": set(),
            "audio.mute_changed": {"muted"},
            "audio.speech_started": {"mode"},
            "audio.speech_ended": {"confidence"},
            "audio.transcribed": {"text", "final"},
            # Conversation
            "conversation.user_message": {"text", "source"},
            "conversation.response_started": set(),
            "conversation.response_chunk": {"text"},
            "conversation.response_finished": {"text"},
            # Speech output
            "speech.playback_started": set(),
            "speech.output_level": {"rms"},
            "speech.playback_finished": {"reason"},
            # Cancellation
            "operation.cancel_requested": {"source"},
            "operation.cancelled": {"source", "elapsed_ms", "clean"},
            # Agent and actions
            "agent.thinking": set(),
            "agent.tool_requested": {"tool", "arguments"},
            "agent.tool_started": {"tool"},
            "agent.tool_completed": {"tool", "result"},
            "agent.tool_verified": {"tool", "verified"},
            "agent.tool_failed": {"tool", "error"},
            "agent.approval_required": {"action", "target", "level"},
            "agent.approval_resolved": {"outcome"},
            # Perception
            "perception.started": {"source"},
            "perception.completed": {"source"},
        }.items()
    }
)

# Events that belong to one operation and must say which. The operation
# fence (bus.OperationFence) uses this ID to drop stale output.
OPERATION_SCOPED: frozenset[str] = frozenset(
    {
        "conversation.response_started",
        "conversation.response_chunk",
        "conversation.response_finished",
        "speech.playback_started",
        "speech.output_level",
        "speech.playback_finished",
        "operation.cancel_requested",
        "operation.cancelled",
        "agent.tool_requested",
        "agent.tool_started",
        "agent.tool_completed",
        "agent.tool_verified",
        "agent.tool_failed",
        "agent.approval_required",
        "agent.approval_resolved",
    }
)


def new_id() -> str:
    return uuid.uuid4().hex


@dataclass(frozen=True, slots=True)
class Event:
    type: str
    session_id: str
    provenance: Provenance
    payload: Mapping[str, Any] = field(default_factory=dict)
    turn_id: str | None = None
    operation_id: str | None = None
    tool_call_id: str | None = None
    causation_id: str | None = None
    event_id: str = field(default_factory=new_id)
    # CLOCK_MONOTONIC is system-wide on Linux, so mono_ns is comparable
    # across Autum's processes on one machine. Use it for latency.
    mono_ns: int = field(default_factory=time.monotonic_ns)
    wall_ms: int = field(default_factory=lambda: time.time_ns() // 1_000_000)
    schema_version: int = SCHEMA_VERSION

    def __post_init__(self) -> None:
        for name in ("type", "session_id", "event_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value:
                raise ValueError(f"{name} must be a non-empty string")
        for name in ("turn_id", "operation_id", "tool_call_id", "causation_id"):
            value = getattr(self, name)
            if value is not None and not isinstance(value, str):
                raise ValueError(f"{name} must be a string or null")
        for name in ("mono_ns", "wall_ms", "schema_version"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise ValueError(f"{name} must be an integer")
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"unsupported schema version {self.schema_version}")
        required = EVENT_TYPES.get(self.type)
        if required is None:
            raise ValueError(f"unknown event type {self.type!r}")
        if not isinstance(self.payload, Mapping):
            raise ValueError("payload must be a mapping")
        missing = required - self.payload.keys()
        if missing:
            raise ValueError(f"{self.type} payload missing {sorted(missing)}")
        if self.type in OPERATION_SCOPED and not self.operation_id:
            raise ValueError(f"{self.type} requires an operation_id")
        object.__setattr__(self, "provenance", Provenance(self.provenance))
        object.__setattr__(self, "payload", types.MappingProxyType(dict(self.payload)))

    def to_json(self) -> str:
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["payload"] = dict(self.payload)
        data["provenance"] = self.provenance.value
        return json.dumps(data, separators=(",", ":"))

    @classmethod
    def from_json(cls, line: str | bytes) -> Event:
        data = json.loads(line)
        if not isinstance(data, dict):
            raise ValueError("event must be a JSON object")
        known = {f.name for f in fields(cls)}
        unknown = data.keys() - known
        if unknown:
            raise ValueError(f"unknown event fields {sorted(unknown)}")
        return cls(**data)
