"""M0 acceptance: event ordering and cancellation with mock components."""

import asyncio
import time

import pytest

from autum.bus import EventBus
from autum.runtime import InvalidTransition, Mode, Runtime, State
from mocks import MockInference, MockMic, MockSpeech, Recorder


class Harness:
    def __init__(self, mode=Mode.PUSH_TO_TALK, inference=None, **runtime_kw):
        self.bus = EventBus("session-1")
        self.log = Recorder(self.bus)
        self.inference = inference or MockInference()
        self.speech = MockSpeech(self.bus)
        self.mic = MockMic(self.bus, mode.value)
        self.runtime = Runtime(self.bus, self.inference, self.speech, mode=mode, **runtime_kw)

    async def until(self, predicate, timeout=2.0):
        deadline = time.monotonic() + timeout
        while not predicate():
            if time.monotonic() > deadline:
                raise AssertionError(f"timed out; state={self.runtime.state}, log={self.log.types()}")
            await asyncio.sleep(0.001)

    async def settle(self):
        await self.runtime.wait_until_settled()
        assert self.bus.subscriber_errors == 0, "a subscriber raised (see log)"

    def assert_causation_points_backwards(self):
        seen = set()
        for event in self.log.events:
            if event.causation_id is not None:
                assert event.causation_id in seen, f"{event.type} caused by a later or unknown event"
            seen.add(event.event_id)

    def assert_nothing_open(self):
        assert self.bus.fence._ops == {}
        assert self.runtime._op is None


def run(coro):
    return asyncio.run(coro)


# ----- ordering ---------------------------------------------------------


def test_push_to_talk_turn_produces_states_and_events_in_order():
    async def scenario():
        h = Harness()
        await h.mic.say("What time is it?")
        await h.until(lambda: h.runtime.state is State.IDLE and h.log.of("speech.playback_finished"))
        await h.settle()
        return h

    h = run(scenario())
    assert h.log.states() == ["speech_detected", "processing", "thinking", "speaking", "idle"]

    flow = [
        t for t in h.log.types()
        if t.startswith(("audio.", "conversation.", "speech.playback"))
        and t != "conversation.response_chunk"
    ]
    assert flow == [
        "audio.speech_started",
        "audio.speech_ended",
        "audio.transcribed",
        "conversation.user_message",
        "conversation.response_started",
        "speech.playback_started",
        "conversation.response_finished",
        "speech.playback_finished",
    ]
    user_message = h.log.of("conversation.user_message")[0]
    assert user_message.provenance == "user"
    assert user_message.turn_id == h.log.of("audio.speech_started")[0].turn_id
    assert all(e.provenance == "model" for e in h.log.of("conversation.response_chunk"))
    assert h.log.of("speech.playback_finished")[0].payload["reason"] == "completed"
    assert [m["role"] for m in h.runtime.history] == ["user", "assistant"]
    h.assert_causation_points_backwards()
    h.assert_nothing_open()


def test_hands_free_mode_rests_in_listening_and_shows_the_open_microphone():
    async def scenario():
        h = Harness(mode=Mode.HANDS_FREE)
        h.bus.emit("audio.capture_started")
        await h.mic.say("Hello")
        await h.until(lambda: h.log.of("speech.playback_finished") and h.runtime.state is State.LISTENING)
        await h.settle()
        return h

    h = run(scenario())
    assert h.log.states()[0] == "listening"
    assert h.log.states()[-1] == "listening"
    assert {e.payload["microphone"] for e in h.log.of("runtime.state_changed")} == {"open"}


def test_text_input_takes_the_same_path_as_speech():
    async def scenario():
        h = Harness()
        h.runtime.submit_text("typed question")
        await h.until(lambda: h.runtime.state is State.IDLE and h.log.of("speech.playback_finished"))
        await h.settle()
        return h

    h = run(scenario())
    assert h.log.of("conversation.user_message")[0].payload == {"text": "typed question", "source": "text"}
    assert h.log.states() == ["thinking", "speaking", "idle"]


# ----- cancellation -----------------------------------------------------


def test_barge_in_stops_audio_at_once_and_no_stale_output_follows():
    async def scenario():
        long_reply = " ".join(f"Sentence number {i} is here." for i in range(40))
        h = Harness(inference=MockInference(long_reply))
        await h.mic.say("Tell me a long story")
        await h.until(lambda: h.runtime.state is State.SPEAKING)
        old_op = h.runtime._op.id
        new_turn = h.mic.start_speaking()  # user talks over Autum
        await h.until(lambda: h.log.of("operation.cancelled"))
        h.mic.finish(new_turn, "Stop, something else")
        await h.until(lambda: len(h.log.of("conversation.response_finished")) == 1 and h.runtime.state is State.IDLE)
        await h.settle()
        return h, old_op

    h, old_op = run(scenario())
    started = h.log.of("audio.speech_started")[1]
    cancel = h.log.of("operation.cancel_requested")[0]
    stopped = [e for e in h.log.of("speech.playback_finished") if e.operation_id == old_op][0]

    assert cancel.payload["source"] == "push_to_talk"
    assert stopped.payload["reason"] == "cancelled"
    interrupt_ms = (stopped.mono_ns - started.mono_ns) / 1e6
    assert interrupt_ms < 50, interrupt_ms  # objective is 200 ms end to end; the runtime's share must be tiny

    # Nothing from the cancelled operation after the cancel request, except cancellation itself.
    after = [e for e in h.log.events if e.operation_id == old_op and e.mono_ns > cancel.mono_ns]
    assert {e.type for e in after} <= {"speech.playback_finished", "operation.cancelled"}
    assert not [a for a in h.speech.audible if a[0] == old_op and a[2] > cancel.mono_ns]
    assert h.log.of("operation.cancelled")[0].payload["clean"] is True

    states = h.log.states()
    i = states.index("interrupted")
    assert states[i - 1 : i + 2] == ["speaking", "interrupted", "speech_detected"]
    assert states[-1] == "idle"

    interrupted = [m for m in h.runtime.history if "[interrupted by the user]" in m["content"]]
    assert len(interrupted) == 1 and interrupted[0]["role"] == "assistant"
    h.assert_causation_points_backwards()
    h.assert_nothing_open()


def test_user_command_cancel_returns_to_rest_state():
    async def scenario():
        h = Harness(inference=MockInference("This keeps going. " * 100, token_delay_s=0.005))
        h.runtime.submit_text("go on")
        await h.until(lambda: h.runtime.state is State.SPEAKING)
        assert await h.runtime.cancel() is True
        assert await h.runtime.cancel() is False  # nothing left to cancel
        await h.settle()
        return h

    h = run(scenario())
    assert h.log.states()[-2:] == ["interrupted", "idle"]
    assert not h.log.of("conversation.response_finished")
    h.assert_nothing_open()


def test_a_component_that_ignores_cancellation_is_timed_out_and_fenced():
    async def scenario():
        stubborn = MockInference("This keeps going. " * 50, token_delay_s=0.01, ignore_cancel=True)
        h = Harness(inference=stubborn, cancel_timeout_s=0.05)
        h.runtime.submit_text("first")
        await h.until(lambda: h.runtime.state is State.SPEAKING)
        old_op = h.runtime._op.id
        await h.runtime.cancel()
        h.inference = None
        h.runtime.inference = MockInference("Second answer.")
        h.runtime.submit_text("second")
        await h.until(lambda: h.log.of("conversation.response_finished"))
        await h.settle()
        return h, old_op, stubborn

    h, old_op, stubborn = run(scenario())
    assert stubborn.ignored_cancels == 1
    cancelled = h.log.of("operation.cancelled")[0]
    assert cancelled.payload["clean"] is False
    assert any(e.operation_id == old_op for e in h.log.of("runtime.component_error"))
    finished = h.log.of("conversation.response_finished")
    assert len(finished) == 1 and finished[0].payload["text"].strip() == "Second answer."
    assert not [e for e in h.log.of("conversation.response_chunk") if e.operation_id == old_op and e.mono_ns > cancelled.mono_ns]
    h.assert_nothing_open()


def test_inference_failure_reports_an_error_and_recovers():
    async def scenario():
        h = Harness(inference=MockInference(fail_after=3))
        h.runtime.submit_text("break please")
        await h.until(lambda: h.log.of("runtime.component_error"))
        await h.settle()
        h.runtime.inference = MockInference("Working again.")
        h.runtime.submit_text("again")
        await h.until(lambda: h.log.of("conversation.response_finished"))
        await h.settle()
        return h

    h = run(scenario())
    states = h.log.states()
    assert states[states.index("error") + 1] == "idle"
    assert states[-1] == "idle"
    h.assert_nothing_open()


# ----- input handling ---------------------------------------------------


def test_hard_mute_shows_on_the_indicator_and_speech_while_muted_is_ignored():
    async def scenario():
        h = Harness(mode=Mode.HANDS_FREE)
        h.bus.emit("audio.capture_started")
        h.bus.emit("audio.mute_changed", {"muted": True})
        h.mic.start_speaking()
        await h.settle()
        return h

    h = run(scenario())
    assert h.log.of("runtime.state_changed")[-1].payload["microphone"] == "muted"
    assert h.runtime.state is State.IDLE  # muted hands-free does not claim to listen
    assert not h.log.of("conversation.user_message")


def test_a_late_transcript_from_an_abandoned_utterance_is_ignored():
    async def scenario():
        h = Harness()
        first = h.mic.start_speaking()
        second = h.mic.start_speaking()  # e.g. push-to-talk pressed again
        h.mic.finish(first, "stale words")
        await asyncio.sleep(0.01)
        assert not h.log.of("conversation.user_message")
        h.mic.finish(second, "fresh words")
        await h.until(lambda: h.log.of("conversation.response_finished"))
        await h.settle()
        return h

    h = run(scenario())
    assert [e.payload["text"] for e in h.log.of("conversation.user_message")] == ["fresh words"]


def test_invalid_state_transition_is_refused():
    async def scenario():
        h = Harness()
        with pytest.raises(InvalidTransition):
            h.runtime._set_state(State.SPEAKING, "bug")

    run(scenario())


def test_long_sessions_keep_bounded_state():
    async def scenario():
        h = Harness(inference=MockInference("Short.", token_delay_s=0), max_history=6)
        h.speech.per_char_s = 0
        for i in range(60):
            h.runtime.submit_text(f"turn {i}")
            await h.until(lambda: h.runtime.state is State.IDLE and h.runtime._op is None)
        await h.settle()
        return h

    h = run(scenario())
    assert len(h.runtime.history) == 6
    assert h.runtime._tasks == set()
    h.assert_nothing_open()
