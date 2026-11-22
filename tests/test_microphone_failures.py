"""A microphone that opens but sends nothing must stop the tools with
instructions, never leave them waiting forever (the 2026-11-22 failure)."""

import threading
import time

import numpy as np
import pytest

from bench import attended, record_corpus
from bench.common import MicrophoneSilent, capture_blocks, check_microphone


class FakeInputStream:
    """Stands in for sounddevice.InputStream. 'silent' never calls back."""

    def __init__(self, silent, blocks, **kwargs):
        self.callback = kwargs["callback"]
        self.block = kwargs["blocksize"]
        self.channels = kwargs["channels"]
        self.silent = silent
        self.blocks = blocks
        self.closed = False
        self._stop = threading.Event()

    def __enter__(self):
        if not self.silent:
            threading.Thread(target=self._feed, daemon=True).start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self.closed = True

    def _feed(self):
        for level in self.blocks:
            if self._stop.is_set():
                return
            self.callback(np.full((self.block, self.channels), level, np.float32), self.block, None, None)
            time.sleep(0.001)


class FakeSd:
    def __init__(self, silent=False, blocks=()):
        self.silent, self.blocks, self.streams = silent, blocks, []

    def InputStream(self, **kwargs):
        stream = FakeInputStream(self.silent, self.blocks, **kwargs)
        self.streams.append(stream)
        return stream


def test_a_silent_microphone_raises_within_the_timeout_with_instructions():
    start = time.monotonic()
    with pytest.raises(MicrophoneSilent, match="unplug the"):
        check_microphone(FakeSd(silent=True), timeout_s=0.2)
    assert time.monotonic() - start < 1.0


def test_a_working_microphone_passes_the_check_and_the_stream_is_closed():
    sd = FakeSd(blocks=[0.01] * 50)
    check_microphone(sd, timeout_s=1.0)
    assert sd.streams[0].closed


def test_capture_stops_with_an_error_if_the_microphone_stalls_midway():
    sd = FakeSd(blocks=[0.01] * 5)  # five blocks, then nothing
    got = []
    with pytest.raises(MicrophoneSilent):
        for block in capture_blocks(sd, 48_000, 2, 480, timeout_s=0.2):
            got.append(block)
    assert len(got) == 5 and sd.streams[0].closed


def test_recorder_returns_audio_from_a_working_microphone():
    sd = FakeSd(blocks=[0.0] * 300)  # 3 s of silence at 10 ms blocks
    audio = record_corpus.record(sd, noise_floor=None)  # the 2 s noise-floor capture
    assert len(audio) == 2 * record_corpus.RATE


def test_recorder_reports_a_silent_microphone(monkeypatch):
    monkeypatch.setattr("bench.common.capture_blocks.__defaults__", (0.2,))
    with pytest.raises(MicrophoneSilent):
        record_corpus.record(FakeSd(silent=True), noise_floor=-60.0)


def test_attended_run_with_missing_audio_is_rejected():
    attended.check_recording(np.zeros(int(59 * attended.RATE)), 60.0)  # fine
    with pytest.raises(MicrophoneSilent, match="Only 0.0 s"):
        attended.check_recording(np.zeros(24), 60.0)


def test_loopback_gives_up_when_the_duplex_stream_never_finishes(monkeypatch):
    class StuckStream:
        active = True

    class StuckSd:
        stopped = False

        def playrec(self, *a, **k):
            return np.zeros((10, 2), np.float32)

        def get_stream(self):
            return StuckStream()

        def stop(self):
            StuckSd.stopped = True

    monkeypatch.setattr(attended, "click_train", lambda: np.zeros(480, np.float32))
    monkeypatch.setitem(__import__("sys").modules, "sounddevice", StuckSd())
    start = time.monotonic()
    with pytest.raises(MicrophoneSilent):
        attended.loopback()
    assert StuckSd.stopped and time.monotonic() - start < 5
