"""Streaming voice activity and turn detection (WP2 / M3).

Uses the bundled Silero VAD v6 ONNX model (from faster-whisper; no download)
in streaming frame-by-frame mode: 512 samples (32 ms at 16 kHz) per step with
persistent LSTM state and 64-sample context.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Callable, Literal

import numpy as np

VAD_RATE = 16_000
VAD_FRAME = 512  # 32 ms at 16 kHz
VAD_FRAME_BYTES = VAD_FRAME * 2  # 16-bit mono PCM
VAD_CONTEXT = 64
FRAME_MS = VAD_FRAME * 1000 / VAD_RATE  # 32.0 ms


class SileroVad:
    """Streaming Silero VAD v6 scorer over 32 ms PCM-16 mono frames."""

    def __init__(self, session=None) -> None:
        if session is None:
            from faster_whisper.vad import get_vad_model

            session = get_vad_model().session
        self._session = session
        self.reset()

    def reset(self) -> None:
        self._h = np.zeros((1, 1, 128), dtype=np.float32)
        self._c = np.zeros((1, 1, 128), dtype=np.float32)
        self._context = np.zeros(VAD_CONTEXT, dtype=np.float32)

    def __call__(self, pcm_frame: bytes) -> float:
        samples = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0
        if samples.size < VAD_FRAME:
            samples = np.pad(samples, (0, VAD_FRAME - samples.size))
        elif samples.size > VAD_FRAME:
            samples = samples[:VAD_FRAME]
        inp = np.concatenate((self._context, samples))[None, :]
        out, self._h, self._c = self._session.run(
            None, {"input": inp, "h": self._h, "c": self._c}
        )
        self._context = samples[-VAD_CONTEXT:].copy()
        return float(out.reshape(-1)[0])


@dataclass(frozen=True)
class TurnEvent:
    kind: Literal["speech_started", "speech_ended"]
    pcm: bytes = b""
    confidence: float = 0.0


class TurnDetector:
    """Streaming speech-start, end-of-turn, and barge-in state machine.

    Parameters are in milliseconds so they map directly to M1 benchmark
    measurements and can be tuned without changing the logic.
    """

    def __init__(
        self,
        scorer: Callable[[bytes], float],
        *,
        threshold: float = 0.5,
        min_speech_ms: int = 160,
        barge_in_ms: int = 256,
        silence_ms: int = 700,
        pre_roll_ms: int = 320,
        tail_ms: int = 160,
        max_utterance_s: float = 120.0,
    ) -> None:
        self.scorer = scorer
        self.threshold = threshold
        self.neg_threshold = max(threshold - 0.15, 0.01)
        self.min_speech_frames = max(1, math.ceil(min_speech_ms / FRAME_MS))
        self.barge_in_frames = max(1, math.ceil(barge_in_ms / FRAME_MS))
        self.silence_frames = max(1, math.ceil(silence_ms / FRAME_MS))
        self.tail_frames = max(0, math.ceil(tail_ms / FRAME_MS))
        self.max_frames = max(1, math.ceil(max_utterance_s * 1000 / FRAME_MS))
        pre_frames = max(self.min_speech_frames, self.barge_in_frames, math.ceil(pre_roll_ms / FRAME_MS))
        self._pre_roll: deque[bytes] = deque(maxlen=pre_frames)
        self._utterance: list[bytes] = []
        self._speech_probs: list[float] = []
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        self._noise_rms = 0.005
        self._use_energy_gate = isinstance(scorer, SileroVad)

    @property
    def in_speech(self) -> bool:
        return self._in_speech

    def reset(self) -> None:
        self._pre_roll.clear()
        self._utterance.clear()
        self._speech_probs.clear()
        self._in_speech = False
        self._speech_run = 0
        self._silence_run = 0
        if hasattr(self.scorer, "reset"):
            self.scorer.reset()

    @staticmethod
    def _frame_rms(pcm_frame: bytes) -> float:
        if not pcm_frame:
            return 0.0
        samples = np.frombuffer(pcm_frame, dtype=np.int16).astype(np.float32) / 32768.0
        return float(np.sqrt(np.mean(samples * samples))) if samples.size else 0.0

    def feed(self, pcm_frame: bytes, *, speaking: bool = False, min_rms: float = 0.012) -> TurnEvent | None:
        prob = float(self.scorer(pcm_frame))
        frame_rms = self._frame_rms(pcm_frame) if self._use_energy_gate else 1.0
        if self._use_energy_gate and not self._in_speech and prob < 0.20:
            self._noise_rms = 0.92 * self._noise_rms + 0.08 * min(0.015, max(0.002, frame_rms))
        eff_min_rms = max(0.012, min_rms, self._noise_rms * 2.0)
        if not self._in_speech:
            self._pre_roll.append(pcm_frame)
            is_voice = prob >= self.threshold and (not self._use_energy_gate or frame_rms >= eff_min_rms)
            if is_voice:
                self._speech_run += 1
                self._speech_probs.append(prob)
            else:
                self._speech_run = 0
                self._speech_probs.clear()
            need = self.barge_in_frames if speaking else self.min_speech_frames
            if self._speech_run >= need:
                self._in_speech = True
                self._utterance = list(self._pre_roll)
                self._pre_roll.clear()
                self._silence_run = 0
                conf = sum(self._speech_probs) / len(self._speech_probs)
                return TurnEvent("speech_started", confidence=conf)
            return None

        self._utterance.append(pcm_frame)
        # Never scale the continuation floor off a loud peak syllable (which cuts off
        # normal speech after a loud start); anchor it gently above the quiet noise floor.
        min_continue_rms = max(0.009, min_rms * 0.80, self._noise_rms * 1.6) if self._use_energy_gate else 0.0
        if prob >= self.neg_threshold and frame_rms >= min_continue_rms:
            self._silence_run = 0
            if prob >= self.threshold:
                self._speech_probs.append(prob)
        else:
            self._silence_run += 1

        if self._silence_run >= self.silence_frames or len(self._utterance) >= self.max_frames:
            trim = max(0, self._silence_run - self.tail_frames)
            frames = self._utterance[:-trim] if trim else self._utterance
            pcm = b"".join(frames)
            conf = sum(self._speech_probs) / len(self._speech_probs) if self._speech_probs else 0.0
            self._utterance = []
            self._speech_probs = []
            self._in_speech = False
            self._speech_run = 0
            self._silence_run = 0
            return TurnEvent("speech_ended", pcm=pcm, confidence=conf)
        return None
