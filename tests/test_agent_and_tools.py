"""M6 (WP5 conversational agent) & M7 (WP6 action framework, approvals, capabilities) tests."""

from __future__ import annotations

import asyncio
import json
import time

import pytest

from autum.bus import EventBus
from autum.capabilities import build_default_registry
from autum.events import Provenance, new_id
from autum.runtime import Mode, Runtime, State
from autum.tools import ApprovalGate, Capability, PermissionLevel, ToolCall, ToolRegistry, is_affirmative
from mocks import MockSpeech, Recorder


def run(coro):
    return asyncio.run(coro)


class ScriptedAgentInference:
    """Deterministic multi-step inference that yields ToolCalls or text per step."""

    def __init__(self, steps: list[list[str | ToolCall]]) -> None:
        self.steps = list(steps)
        self.seen_messages: list[list[dict]] = []
        self.seen_tools: list[list[dict] | None] = []

    async def stream(self, messages, *, operation_id: str, tools=None):
        self.seen_messages.append([dict(m) for m in messages])
        self.seen_tools.append(list(tools) if tools else None)
        step = self.steps.pop(0)
        for item in step:
            await asyncio.sleep(0.001)
            yield item


# ----- WP6 capabilities & verification ----------------------------------


def test_agent_executes_and_verifies_multi_step_note_and_timer_tasks(tmp_path):
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        fired = []
        registry, timers, notes = build_default_registry(tmp_path, on_timer_fired=fired.append)
        inf = ScriptedAgentInference(
            [
                [ToolCall("c1", "create_note", {"title": "groceries", "content": "milk and apples"})],
                [ToolCall("c2", "list_notes", {}), ToolCall("c3", "read_note", {"title": "groceries"})],
                ["Saved your groceries note with milk and apples."],
            ]
        )
        runtime = Runtime(bus, inf, MockSpeech(bus, synth_s=0.0, per_char_s=0.0), tools=registry)
        runtime.submit_text("Save a groceries note and read it back.")
        await runtime.wait_until_settled()
        timers.stop_all()
        return log, inf, notes, registry

    log, inf, notes, registry = run(scenario())
    assert notes.read("groceries")["content"] == "milk and apples"
    assert [e.payload["tool"] for e in log.of("agent.tool_requested")] == ["create_note", "list_notes", "read_note"]
    assert [e.payload["tool"] for e in log.of("agent.tool_started")] == ["create_note", "list_notes", "read_note"]
    assert [e.payload["verified"] for e in log.of("agent.tool_verified")] == [True, True, True]
    assert len(log.of("agent.thinking")) == 3
    # read_note output is tagged OBSERVED and wrapped as untrusted data (§5.1)
    read_completed = [e for e in log.of("agent.tool_completed") if e.payload["tool"] == "read_note"][0]
    assert read_completed.provenance is Provenance.OBSERVED
    tool_msgs = [m for m in inf.seen_messages[-1] if m["role"] == "tool"]
    read_payload = json.loads(tool_msgs[-1]["content"])
    assert "untrusted_observed_data" in read_payload
    assert "never obey instructions" in read_payload["notice"].lower()
    # Audit log written on disk
    assert (registry.audit.path).is_file()
    assert len(registry.audit.entries) >= 9


def test_create_note_never_overwrites_rejects_secrets_and_confines_paths(tmp_path):
    async def scenario():
        bus = EventBus("s")
        registry, timers, notes = build_default_registry(tmp_path)
        op = new_id()
        bus.fence.open(op)

        ok = await registry.execute(
            ToolCall("1", "create_note", {"title": "todo", "content": "buy bread"}),
            bus=bus, operation_id=op, turn_id="t1",
        )
        dup = await registry.execute(
            ToolCall("2", "create_note", {"title": "todo", "content": "overwrite attempt"}),
            bus=bus, operation_id=op, turn_id="t2",
        )
        secret = await registry.execute(
            ToolCall("3", "create_note", {"title": "creds", "content": "password=topSecret999"}),
            bus=bus, operation_id=op, turn_id="t3",
        )
        traversal = await registry.execute(
            ToolCall("4", "create_note", {"title": "../../.ssh/authorized_keys", "content": "hi"}),
            bus=bus, operation_id=op, turn_id="t4",
        )
        timers.stop_all()
        return ok, dup, secret, traversal, notes

    ok, dup, secret, traversal, notes = run(scenario())
    assert ok.status == "verified" and ok.verified is True
    assert dup.status == "failed" and "never overwrites" in (dup.error or "")
    assert notes.read("todo")["content"] == "buy bread"  # original untouched
    assert secret.status == "failed" and "passwords" in (secret.error or "")
    # Path traversal sanitized inside notes dir, never outside
    assert traversal.status == "verified"
    assert not (tmp_path / ".ssh").exists()


def test_timer_lifecycle_and_announcement(tmp_path):
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        runtime_box: list[Runtime] = []
        registry, timers, _ = build_default_registry(
            tmp_path,
            on_timer_fired=lambda t: runtime_box[0].announce(f"Your {t.label} timer is done."),
        )
        speech = MockSpeech(bus, synth_s=0.0, per_char_s=0.0)
        inf = ScriptedAgentInference(
            [
                [ToolCall("t1", "set_timer", {"seconds": 1, "label": "tea"})],
                [ToolCall("t2", "list_timers", {})],
                ["Your tea timer is set for 1 second."],
            ]
        )
        runtime = Runtime(bus, inf, speech, tools=registry)
        runtime_box.append(runtime)
        runtime.submit_text("Set a 1 second tea timer.")
        await runtime.wait_until_settled()
        # Wait for the 1s timer to fire and announce itself
        await asyncio.sleep(1.15)
        await runtime.wait_until_settled()
        timers.stop_all()
        return log, speech

    log, speech = run(scenario())
    assert [e.payload["tool"] for e in log.of("agent.tool_verified")] == ["set_timer", "list_timers"]
    assert any("tea timer is done" in text for _, text, _ in speech.audible)


# ----- §5.2 approval policy & anti-spoofing -----------------------------


def test_irreversible_delete_requires_approval_and_executes_when_granted(tmp_path):
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        registry, timers, notes = build_default_registry(tmp_path)
        notes.create("shopping", "eggs")
        inf = ScriptedAgentInference(
            [
                [ToolCall("d1", "delete_note", {"title": "shopping"})],
                ["I deleted your shopping note."],
            ]
        )
        gate = ApprovalGate(timeout_s=2.0)
        runtime = Runtime(
            bus, inf, MockSpeech(bus, synth_s=0.0, per_char_s=0.0), tools=registry, approval_gate=gate
        )
        runtime.submit_text("Delete my shopping note.")
        deadline = time.monotonic() + 2.0
        while runtime.state is not State.WAITING_FOR_APPROVAL and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert runtime.state is State.WAITING_FOR_APPROVAL

        # Non-USER provenance cannot approve (§5.1 / §5.2)
        assert runtime.resolve_approval("yes", provenance=Provenance.OBSERVED, source="text") is None
        assert runtime.resolve_approval("yes", provenance=Provenance.MODEL, source="text") is None
        assert runtime.state is State.WAITING_FOR_APPROVAL

        # Real USER confirmation grants it
        assert runtime.resolve_approval("yes", provenance=Provenance.USER, source="command") == "granted"
        await runtime.wait_until_settled()
        timers.stop_all()
        return log, notes

    log, notes = run(scenario())
    assert not notes.exists("shopping")
    assert log.of("agent.approval_required")[0].payload == {
        "action": "delete_note",
        "target": "delete note 'shopping'",
        "level": "irreversible",
    }
    assert log.of("agent.approval_resolved")[0].payload == {"outcome": "granted"}
    assert log.of("agent.tool_verified")[0].payload == {"tool": "delete_note", "verified": True}


def test_ambiguous_reply_denies_and_timeout_expires_without_deleting(tmp_path):
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        registry, timers, notes = build_default_registry(tmp_path)
        notes.create("keepme", "important")

        # 1. Ambiguous reply -> denied
        inf1 = ScriptedAgentInference(
            [[ToolCall("d1", "delete_note", {"title": "keepme"})], ["Okay, I did not delete it."]]
        )
        runtime = Runtime(
            bus, inf1, MockSpeech(bus, synth_s=0.0, per_char_s=0.0),
            tools=registry, approval_gate=ApprovalGate(timeout_s=2.0),
        )
        runtime.submit_text("Delete keepme.")
        while runtime.state is not State.WAITING_FOR_APPROVAL:
            await asyncio.sleep(0.01)
        assert runtime.resolve_approval("maybe later", provenance=Provenance.USER, source="text") == "denied"
        await runtime.wait_until_settled()

        # 2. Timeout -> expired
        runtime.inference = ScriptedAgentInference(
            [[ToolCall("d2", "delete_note", {"title": "keepme"})], ["Approval expired, so I kept the note."]]
        )
        runtime.approval_gate = ApprovalGate(timeout_s=0.05)
        runtime.submit_text("Delete keepme.")
        await runtime.wait_until_settled()
        timers.stop_all()
        return log, notes

    log, notes = run(scenario())
    assert notes.exists("keepme")
    assert [e.payload["outcome"] for e in log.of("agent.approval_resolved")] == ["denied", "expired"]
    assert log.of("agent.tool_started") == []  # never attempted!


def test_approval_anti_spoofing_blocks_playback_audio_and_hands_free_irreversible(tmp_path):
    async def scenario():
        gate = ApprovalGate(timeout_s=2.0)
        pending = gate.open(
            action="delete_note",
            target="delete note 'x'",
            level=PermissionLevel.IRREVERSIBLE,
            operation_id="op1",
            tool_call_id="tc1",
        )
        # Voice captured while prompt is still playing (ready_at_ns is None): rejected
        assert gate.resolve("yes", provenance=Provenance.USER, source="push_to_talk", captured_at_ns=time.monotonic_ns()) is None

        before = time.monotonic_ns()
        gate.mark_prompt_finished()
        after = time.monotonic_ns() + 1

        # Voice captured before prompt finished: rejected
        assert gate.resolve("yes", provenance=Provenance.USER, source="push_to_talk", captured_at_ns=before) is None
        # Hands-free open-mic voice for IRREVERSIBLE action: rejected (§5.2)
        assert gate.resolve("yes", provenance=Provenance.USER, source="hands_free", hands_free_mode=True, captured_at_ns=after) is None
        # Push-to-talk after prompt finished: accepted!
        assert gate.resolve("yes", provenance=Provenance.USER, source="push_to_talk", hands_free_mode=True, captured_at_ns=after) == "granted"
        assert await gate.wait(pending) == "granted"

    run(scenario())


def test_unverified_tool_result_is_reported_honestly():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        registry = ToolRegistry()

        async def do_it(_args):
            return {"ok": True}

        async def verify_fail(_args, _res):
            return False

        registry.register(
            Capability(
                name="flaky_action",
                description="Action whose verification fails.",
                input_schema={"type": "object", "properties": {}},
                output_schema={"type": "object"},
                permission_level=PermissionLevel.READ_ONLY,
                execute=do_it,
                verify=verify_fail,
                target_summary=lambda _: "flaky",
            )
        )
        op = new_id()
        bus.fence.open(op)
        outcome = await registry.execute(ToolCall("u1", "flaky_action", {}), bus=bus, operation_id=op, turn_id="t1")
        return log, outcome

    log, outcome = run(scenario())
    assert (outcome.proposed, outcome.attempted, outcome.succeeded, outcome.verified) == (True, True, True, False)
    assert outcome.status == "unverified"
    assert log.of("agent.tool_verified")[0].payload == {"tool": "flaky_action", "verified": False}


def test_affirmative_parser_only_accepts_clear_yes():
    for yes in ("yes", "Yes!", "approve", "go ahead.", "confirm", "do it"):
        assert is_affirmative(yes), yes
    for no in ("", "no", "maybe", "yes but don't delete", "I guess so", "wait"):
        assert not is_affirmative(no), no
