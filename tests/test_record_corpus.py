"""Recording-tool logic, tested without a microphone."""

from pathlib import Path

import numpy as np
import pytest

from bench.common import SR, Item, Vad, active_span, item_seed, noise_for, read_wav, resample
from bench.record_corpus import RATE, AutoStop, Labels, benchmark_copy, label_clip

SYNTHETIC = Path(__file__).resolve().parents[1] / "bench" / "data" / "synthetic-v1"


def feed_levels(stopper, levels_db, seconds_each):
    blocks = int(seconds_each / stopper.block_s)
    for level in levels_db:
        for _ in range(blocks):
            if stopper.feed(level):
                return True
    return False


def test_autostop_waits_for_speech_then_sustained_quiet():
    stopper = AutoStop(noise_floor_dbfs=-60)
    assert not feed_levels(stopper, [-60], 3.0)  # silence before speaking never stops it
    assert not feed_levels(stopper, [-20, -60, -20], 1.0)  # a one-second pause does not stop it
    assert feed_levels(stopper, [-60], 2.6)


def test_autostop_has_a_maximum_length():
    stopper = AutoStop(noise_floor_dbfs=-60)
    assert feed_levels(stopper, [-20], 30.0)
    assert stopper.elapsed_s == pytest.approx(25.0, abs=0.02)


def test_labels_are_flagged_when_energy_and_vad_disagree():
    agree = Labels(energy=(0.5, 2.0), vad=(0.52, 1.95), segments=[], peak=0.5)
    late_end = Labels(energy=(0.5, 2.0), vad=(0.5, 1.8), segments=[], peak=0.5)
    nothing = Labels(energy=(0.5, 2.0), vad=None, segments=[], peak=0.5)
    assert not agree.flagged
    assert late_end.flagged
    assert nothing.flagged


def test_problems_report_clipping_quietness_and_timing():
    lab = Labels(energy=(0.1, 4.0), vad=(0.1, 4.0), segments=[], peak=1.0)
    problems = " ".join(lab.problems(duration_s=4.5))
    assert "clipping" in problems and "too soon" in problems and "too little silence" in problems
    assert Labels((0.6, 2.0), (0.6, 2.0), [], 0.02).problems(4.5) == ["very quiet: move closer to the microphone"]


def test_noise_is_identical_for_the_same_prompt_in_every_set():
    assert np.array_equal(noise_for("noise-01", 1000), noise_for("noise-01", 1000))
    assert not np.array_equal(noise_for("noise-01", 1000), noise_for("noise-02", 1000))
    assert item_seed("x", 0) != item_seed("x", 1)


def test_benchmark_copy_is_16k_and_only_noise_prompts_get_noise():
    master = np.zeros(RATE * 2, np.float32)
    master[RATE // 2 : RATE] = 0.3
    quiet = Item("short-01", "short", "x", [], [], False)
    noisy = Item("noise-01", "noise", "x", [], [], False)
    plain = benchmark_copy(master, quiet, 0.5, 1.0)
    mixed = benchmark_copy(master, noisy, 0.5, 1.0)
    assert len(plain) == 2 * SR
    assert np.abs(plain[: SR // 4]).max() < 1e-3  # untouched silence
    assert np.abs(mixed[: SR // 4]).max() > 1e-3  # noise added everywhere


@pytest.mark.skipif(not SYNTHETIC.exists(), reason="needs the local synthetic-v1 clips")
def test_labels_of_real_speech_match_the_known_boundaries():
    import json

    known = json.loads((SYNTHETIC / "labels.json").read_text())
    vad = Vad()
    for item_id in ["short-01", "long-03", "pause-02", "correction-01"]:
        audio16k, sr = read_wav(SYNTHETIC / f"{item_id}.wav")
        lab = label_clip(resample(audio16k, sr, RATE), vad)
        truth = known[item_id]
        assert lab.energy[0] == pytest.approx(truth["speech_start_s"], abs=0.05)
        assert lab.energy[1] == pytest.approx(truth["speech_end_s"], abs=0.05)
        assert lab.vad is not None and not lab.flagged, (item_id, lab)
        if item_id.startswith("pause"):
            assert len(lab.segments) == 2
