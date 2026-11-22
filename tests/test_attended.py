"""Attended-test analysis, checked without playing or recording audio."""

import json
import subprocess

import numpy as np
import pytest

from bench import attended
from bench.attended import RATE, analyse, click_train, onset_after, roundtrip_ms, score_cues
from bench.common import FRAME, SR, speech_starts, write_wav

FRAME_S = FRAME / SR


def probs(pattern: str) -> np.ndarray:
    """'s' = speech frame, '.' = silence (32 ms each)."""
    return np.array([0.9 if c == "s" else 0.05 for c in pattern])


def test_speech_start_fires_once_per_utterance_and_rearms_after_silence():
    pattern = "." * 5 + "s" * 20 + "." * 4 + "s" * 10 + "." * 12 + "s" * 10
    starts = speech_starts(probs(pattern), min_speech_ms=96, rearm_ms=300)
    # A 128 ms gap does not re-arm; a 384 ms gap does.
    assert starts == pytest.approx([8 * FRAME_S, (5 + 20 + 4 + 10 + 12 + 3) * FRAME_S])


def test_a_longer_minimum_ignores_short_sounds():
    blip = "." * 10 + "sss" + "." * 20 + "s" * 15 + "." * 10  # 96 ms blip, then real speech
    assert len(speech_starts(probs(blip), min_speech_ms=96)) == 2
    assert len(speech_starts(probs(blip), min_speech_ms=200)) == 1


def test_roundtrip_finds_the_delay_and_refuses_to_guess_without_a_signal():
    clicks = click_train()
    delay = int(0.0473 * RATE)
    rng = np.random.default_rng(0)
    recorded = np.concatenate([np.zeros(delay), clicks * 0.05])[: len(clicks)]
    recorded = recorded + rng.standard_normal(len(clicks)) * 0.002
    assert roundtrip_ms(clicks, recorded) == pytest.approx(47.3, abs=0.1)
    assert roundtrip_ms(clicks, rng.standard_normal(len(clicks)) * 0.01) is None
    # Reverberant room: the click arrives smeared, with noise; still found.
    room = np.convolve(clicks, np.exp(-np.arange(2400) / 300) * rng.standard_normal(2400) * 0.05)[: len(clicks)]
    smeared = np.concatenate([np.zeros(delay), room])[: len(clicks)] + rng.standard_normal(len(clicks)) * 0.002
    assert roundtrip_ms(clicks, smeared) == pytest.approx(47.3, abs=3)


def test_onset_is_found_above_the_level_before_the_cue():
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(RATE * 4).astype(np.float32) * 0.01  # echo/noise floor
    audio[int(2.5 * RATE) : int(3.0 * RATE)] += 0.3 * np.sin(np.arange(int(0.5 * RATE)) * 0.05)
    assert onset_after(audio, 2.0) == pytest.approx(2.5, abs=0.02)
    assert onset_after(rng.standard_normal(RATE * 4) * 0.01, 2.0) is None


def test_cue_scoring_takes_the_first_detection_in_each_window():
    assert score_cues([1.0, 6.5, 7.0, 30.0], [6.0, 20.0]) == [6.5, None]


def make_run(speak_at: list[float], seconds: float = 61.0, offset: float = 0.8):
    """Fake recording: quiet noise, with 0.6 s of 'speech' after each time."""
    rng = np.random.default_rng(0)
    audio = rng.standard_normal(int(seconds * RATE)).astype(np.float32) * 0.005
    frames = int(seconds / FRAME_S)
    p = np.full(frames, 0.05)
    for t in speak_at:
        t += offset
        audio[int(t * RATE) : int((t + 0.6) * RATE)] += 0.3
        p[int(np.ceil(t / FRAME_S)) : int((t + 0.6) / FRAME_S)] = 0.9  # frames after the onset
    return audio, p, offset


def test_interruptions_are_scored_with_detection_and_delay():
    audio, p, offset = make_run([c + 0.7 for c in attended.CUE_TIMES_S[:3]])
    result = analyse("interruption", audio, p, offset, ["Stop."] * 4)
    assert result[96]["detected"] == "3/4"
    delays = [d for d in result[96]["delay_ms"] if d is not None]
    assert len(delays) == 3 and all(90 <= d <= 140 for d in delays)  # 96 ms + up to one frame
    assert result[96]["outside_cue_windows"] == 0


def test_echo_counts_every_detection_and_backchannels_count_false_interruptions():
    audio, p, offset = make_run([3.0, 15.0])
    assert analyse("echo", audio, p, offset, [])[96]["false_detections"] == 2
    audio, p, offset = make_run([attended.CUE_TIMES_S[1] + 0.5])
    result = analyse("backchannel", audio, p, offset, ["Mm-hm."] * 4)
    assert result[96]["false_interruptions"] == "1/4"


def test_playback_is_built_once_at_60_seconds_and_fixed_loudness(tmp_path, monkeypatch):
    replies = tmp_path / "outputs" / "20260101-000000" / "C"
    for item in attended.load_corpus()[:20]:
        write_wav(replies / f"{item.id}.wav", 0.2 * np.sin(np.arange(24_000 * 4) * 0.03), 24_000)
    monkeypatch.setattr(attended, "DATA", tmp_path)
    first = attended.build_playback(tmp_path / "attended" / "playback.wav")
    again = attended.build_playback(tmp_path / "attended" / "playback.wav")
    assert first["sha256"] == again["sha256"]
    audio, sr = attended.read_wav(tmp_path / "attended" / "playback.wav")
    assert sr == RATE and len(audio) == 60 * RATE
    assert 20 * np.log10(np.sqrt(np.mean(audio**2))) == pytest.approx(-20, abs=0.5)
    assert np.abs(audio).max() <= 10 ** (-1 / 20) + 1e-3  # never clipped
    assert first["rms_dbfs"] == pytest.approx(-20, abs=0.5)


def test_peaky_playback_is_limited_instead_of_clipped(tmp_path, monkeypatch):
    replies = tmp_path / "outputs" / "20260101-000000" / "C"
    spiky = np.zeros(24_000 * 4, np.float32)
    spiky[::2400] = 0.9  # crest factor far above 19 dB
    for item in attended.load_corpus()[:20]:
        write_wav(replies / f"{item.id}.wav", spiky, 24_000)
    monkeypatch.setattr(attended, "DATA", tmp_path)
    info = attended.build_playback(tmp_path / "attended" / "playback.wav")
    audio, _ = attended.read_wav(tmp_path / "attended" / "playback.wav")
    assert np.abs(audio).max() <= 10 ** (-1 / 20) + 1e-3
    assert info["rms_dbfs"] < -20


def test_echo_cancellation_is_unloaded_even_when_a_run_fails(monkeypatch):
    calls = []

    def fake_run(*args):
        calls.append(args)
        if args[:3] == ("pactl", "list", "short"):
            return f"1\t{attended.AEC_SINK}\n2\t{attended.AEC_SOURCE}\n"
        return ""

    monkeypatch.setattr(attended, "run", fake_run)
    monkeypatch.setattr(
        attended.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 0, stdout="536870913\n", stderr=""),
    )
    with pytest.raises(ValueError):
        with attended.echo_cancel():
            raise ValueError("run failed")
    assert ("pactl", "unload-module", "536870913") in calls


def test_a_failed_echo_cancellation_load_is_reported(monkeypatch):
    monkeypatch.setattr(attended, "run", lambda *a: "")
    monkeypatch.setattr(
        attended.subprocess, "run",
        lambda *a, **k: subprocess.CompletedProcess(a, 1, stdout="", stderr="Module initialization failed"),
    )
    with pytest.raises(RuntimeError, match="Module initialization failed"):
        with attended.echo_cancel():
            pass
