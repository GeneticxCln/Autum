"""The benchmark's own measurements must be right before its numbers mean anything."""

import numpy as np
import pytest

from bench.common import FRAME, SR, active_span, normalize, percentile, simulate_endpoint, wer


def test_wer_counts_word_edits_after_normalization():
    assert wer("Set a timer for twenty minutes.", "set a timer for 20 minutes") == (0, 6)
    assert wer("What's the capital of Canada?", "whats the capital of canada") == (0, 5)
    assert wer("Remind me at four o'clock", "remind me at for oclock") == (1, 5)
    assert wer("one two three", "") == (3, 3)


def test_normalize_drops_fillers_and_spells_numbers():
    assert normalize("Open the, um, the 144 notes") == ["open", "the", "the", "one", "hundred", "forty", "four", "notes"]


def frames(pattern: str) -> np.ndarray:
    """'s' = speech frame, '.' = silence frame (32 ms each)."""
    return np.array([0.9 if c == "s" else 0.05 for c in pattern])


def test_endpoint_fires_after_the_configured_silence():
    frame_s = FRAME / SR
    ep = simulate_endpoint(frames("..." + "s" * 20 + "." * 30), silence_ms=320)
    assert ep.speech_start_s == pytest.approx(3 * frame_s)
    assert ep.endpoint_s == pytest.approx((23 + 10) * frame_s)


def test_a_short_pause_does_not_end_the_turn_but_a_long_one_does():
    pattern = "s" * 10 + "." * 12 + "s" * 10 + "." * 40  # 384 ms pause inside
    assert simulate_endpoint(frames(pattern), silence_ms=500).endpoint_s > 32 * FRAME / SR
    assert simulate_endpoint(frames(pattern), silence_ms=300).endpoint_s < 22 * FRAME / SR


def test_a_click_shorter_than_min_speech_is_not_a_turn():
    assert simulate_endpoint(frames("..s" + "." * 40), silence_ms=300).speech_start_s is None


def test_active_span_finds_the_speech_inside_silence():
    audio = np.zeros(SR * 2, np.float32)
    audio[SR // 2 : SR] = 0.3 * np.sin(np.arange(SR // 2) * 0.1)
    start, end = active_span(audio, SR)
    assert start == pytest.approx(0.5, abs=0.011)
    assert end == pytest.approx(1.0, abs=0.011)


def test_percentile_interpolates():
    assert percentile([1, 2, 3, 4, 5], 0.5) == 3
    assert percentile([0, 10], 0.95) == pytest.approx(9.5)
    assert percentile([], 0.5) is None
