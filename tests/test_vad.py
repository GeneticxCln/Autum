"""Streaming Silero VAD and turn-detection tests (WP2 / M3)."""

from __future__ import annotations

import pytest

from autum.vad import VAD_FRAME_BYTES, SileroVad, TurnDetector

SILENCE_FRAME = b"\x00" * VAD_FRAME_BYTES


def frame(tag: int) -> bytes:
    """Distinct 32 ms PCM frame tagged with a byte value so we can trace pre-roll and trimming."""
    return bytes([tag & 0xFF]) * VAD_FRAME_BYTES


def scorer_from_tags(speech_tags: set[int]):
    return lambda pcm: 0.9 if pcm[0] in speech_tags else 0.05


def test_silero_vad_runs_streaming_on_silence_and_resets_cleanly():
    pytest.importorskip("faster_whisper")
    vad = SileroVad()
    first = [vad(SILENCE_FRAME) for _ in range(5)]
    assert all(0.0 <= p < 0.1 for p in first)
    vad.reset()
    second = [vad(SILENCE_FRAME) for _ in range(5)]
    assert second == pytest.approx(first)


def test_turn_detector_starts_with_pre_roll_and_ends_with_trimmed_tail():
    det = TurnDetector(
        scorer_from_tags({1}),
        min_speech_ms=96,   # 3 frames
        barge_in_ms=96,     # 3 frames
        silence_ms=160,     # 5 frames
        pre_roll_ms=160,    # 5 frames (includes 2 silence pre-roll + 3 onset)
        tail_ms=64,         # 2 frames kept after speech
    )
    events = []
    stream = [frame(0)] * 4 + [frame(1)] * 6 + [frame(0)] * 5
    for f in stream:
        ev = det.feed(f)
        if ev:
            events.append(ev)

    assert [e.kind for e in events] == ["speech_started", "speech_ended"]
    # Pre-roll (5 frames: 2 silence + 3 onset) + 3 more speech + 2 tail silence = 10 frames
    pcm = events[1].pcm
    assert len(pcm) == 10 * VAD_FRAME_BYTES
    assert pcm[: 2 * VAD_FRAME_BYTES] == frame(0) * 2
    assert pcm[2 * VAD_FRAME_BYTES : 8 * VAD_FRAME_BYTES] == frame(1) * 6
    assert pcm[8 * VAD_FRAME_BYTES :] == frame(0) * 2
    assert events[1].confidence == pytest.approx(0.9)


def test_brief_noise_burst_and_mid_turn_pause_behave_correctly():
    det = TurnDetector(
        scorer_from_tags({1}),
        min_speech_ms=160,  # 5 frames
        silence_ms=256,     # 8 frames
        tail_ms=64,
    )
    # 3 frames of noise (< 5 required): ignored
    for f in [frame(1)] * 3 + [frame(0)] * 3:
        assert det.feed(f) is None
    assert not det.in_speech

    # Start speaking, pause 4 frames (< 8 silence frames), continue, then stop
    events = []
    for f in [frame(1)] * 5 + [frame(0)] * 4 + [frame(1)] * 4 + [frame(0)] * 8:
        ev = det.feed(f)
        if ev:
            events.append(ev)
    assert [e.kind for e in events] == ["speech_started", "speech_ended"]


def test_barge_in_ignores_short_backchannel_but_interrupts_on_sustained_speech():
    det = TurnDetector(
        scorer_from_tags({1}),
        min_speech_ms=128,  # 4 frames when idle
        barge_in_ms=256,    # 8 frames while Autum is speaking
        silence_ms=160,
    )
    # Short "mm-hm" (5 frames) while speaking: ignored
    for f in [frame(1)] * 5 + [frame(0)] * 4:
        assert det.feed(f, speaking=True) is None
    assert not det.in_speech

    # Sustained speech (8 frames) while speaking: triggers barge-in
    started = None
    for f in [frame(1)] * 8:
        ev = det.feed(f, speaking=True)
        if ev:
            started = ev
    assert started is not None and started.kind == "speech_started"


def test_max_utterance_caps_continuous_speech():
    det = TurnDetector(
        scorer_from_tags({1}),
        min_speech_ms=64,   # 2 frames
        barge_in_ms=64,
        silence_ms=640,
        pre_roll_ms=64,
        max_utterance_s=0.32,  # 10 frames
    )
    events = []
    for _ in range(10):
        ev = det.feed(frame(1))
        if ev:
            events.append(ev)
    assert [e.kind for e in events] == ["speech_started", "speech_ended"]
    assert len(events[1].pcm) == 10 * VAD_FRAME_BYTES
