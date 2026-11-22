"""Shared pieces of the WP1 benchmark: audio, labels, endpointing, metrics."""

from __future__ import annotations

import hashlib
import json
import re
import statistics
import subprocess
import threading
import wave
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"  # audio stays local and out of git (plan §5.3)
RESULTS = ROOT / "results"
SR = 16_000  # recognition and VAD rate
FRAME = 512  # Silero VAD window at 16 kHz = 32 ms


# ----- audio files ------------------------------------------------------


def read_wav(path: Path) -> tuple[np.ndarray, int]:
    with wave.open(str(path), "rb") as w:
        if w.getsampwidth() != 2:
            raise ValueError(f"{path}: expected 16-bit PCM")
        frames = w.readframes(w.getnframes())
        audio = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
        if w.getnchannels() > 1:
            audio = audio.reshape(-1, w.getnchannels()).mean(axis=1)
        return audio, w.getframerate()


def write_wav(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(pcm.tobytes())


def pcm16_to_float(data: bytes) -> np.ndarray:
    usable = len(data) - len(data) % 2
    return np.frombuffer(data[:usable], dtype=np.int16).astype(np.float32) / 32768.0


def resample(audio: np.ndarray, sr_from: int, sr_to: int) -> np.ndarray:
    if sr_from == sr_to:
        return audio
    from math import gcd

    from scipy.signal import resample_poly

    g = gcd(sr_from, sr_to)
    return resample_poly(audio, sr_to // g, sr_from // g).astype(np.float32)


def active_span(audio: np.ndarray, sr: int, floor_db: float = -40.0) -> tuple[float, float]:
    """First and last time (s) the 10 ms envelope is within floor_db of the peak."""
    hop = sr // 100
    n = len(audio) // hop
    if n == 0:
        return 0.0, 0.0
    rms = np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1) + 1e-12)
    level = 20 * np.log10(rms / rms.max())
    active = np.flatnonzero(level > floor_db)
    if active.size == 0:
        return 0.0, 0.0
    return active[0] * hop / sr, (active[-1] + 1) * hop / sr


def pink_noise(n: int, rng: np.random.Generator) -> np.ndarray:
    spectrum = np.fft.rfft(rng.standard_normal(n))
    spectrum /= np.sqrt(np.maximum(np.arange(len(spectrum)), 1))
    noise = np.fft.irfft(spectrum, n)
    return (noise / np.sqrt(np.mean(noise**2))).astype(np.float32)


NOISE_SNR_DB = 10.0
BASE_SEED = 7


def item_seed(item_id: str, part: int = 0) -> int:
    """Stable per-prompt seed, independent of generation order."""
    digest = hashlib.sha256(f"{BASE_SEED}:{item_id}:{part}".encode()).digest()
    return int.from_bytes(digest[:4], "big")


def noise_for(item_id: str, n: int) -> np.ndarray:
    """The pink noise for one prompt: identical in every test set."""
    return pink_noise(n, np.random.default_rng(item_seed(item_id, -1)))


def mix_at_snr(speech: np.ndarray, noise: np.ndarray, snr_db: float, speech_span: tuple[int, int]) -> np.ndarray:
    a, b = speech_span
    p_speech = np.mean(speech[a:b] ** 2)
    scale = np.sqrt(p_speech / (10 ** (snr_db / 10)))
    return (speech + noise[: len(speech)] * scale).astype(np.float32)


# ----- corpus -----------------------------------------------------------


@dataclass
class Item:
    id: str
    category: str
    text: str
    history: list[dict]
    expect_any: list[str]
    recorded_only: bool

    @property
    def reference(self) -> str:
        return " ".join(part.strip() for part in self.text.split("|"))


def load_corpus() -> list[Item]:
    data = json.loads((ROOT / "corpus.json").read_text())
    return [
        Item(
            id=i["id"],
            category=i["category"],
            text=i["text"],
            history=i.get("history", []),
            expect_any=i.get("expect_any", []),
            recorded_only=i.get("recorded_only", False),
        )
        for i in data["items"]
    ]


def load_labels(corpus_dir: Path) -> dict[str, dict]:
    return json.loads((corpus_dir / "labels.json").read_text())


# ----- endpointing ------------------------------------------------------


class Vad:
    """Silero VAD v6 as bundled with faster-whisper (no download)."""

    def __init__(self) -> None:
        from faster_whisper.vad import get_vad_model

        self._model = get_vad_model()

    def probabilities(self, audio16k: np.ndarray) -> np.ndarray:
        pad = (-len(audio16k)) % FRAME
        padded = np.pad(audio16k, (0, pad)).astype(np.float32)
        return np.asarray(self._model(padded)).reshape(-1)


@dataclass
class Endpoint:
    speech_start_s: float | None
    endpoint_s: float | None  # time the detector decides the turn is over


def simulate_endpoint(
    probs: np.ndarray, silence_ms: int, threshold: float = 0.5, min_speech_ms: int = 96
) -> Endpoint:
    """Replay frame probabilities as a streaming detector would see them."""
    frame_s = FRAME / SR
    neg = max(threshold - 0.15, 0.01)
    need_silence = int(np.ceil(silence_ms / 1000 / frame_s))
    need_speech = int(np.ceil(min_speech_ms / 1000 / frame_s))
    start = None
    speech_run = silence_run = 0
    for i, p in enumerate(probs):
        if start is None:
            speech_run = speech_run + 1 if p >= threshold else 0
            if speech_run >= need_speech:
                start = (i - need_speech + 1) * frame_s
            continue
        silence_run = silence_run + 1 if p < neg else 0
        if silence_run >= need_silence:
            return Endpoint(start, (i + 1) * frame_s)
    return Endpoint(start, None)


def speech_starts(
    probs: np.ndarray, min_speech_ms: int = 96, threshold: float = 0.5, rearm_ms: int = 300
) -> list[float]:
    """Times (s) a barge-in detector fires: after min_speech_ms of speech.

    After firing it waits for rearm_ms of silence before it can fire again,
    so one utterance counts once.
    """
    frame_s = FRAME / SR
    neg = max(threshold - 0.15, 0.01)
    need_speech = int(np.ceil(min_speech_ms / 1000 / frame_s))
    need_quiet = int(np.ceil(rearm_ms / 1000 / frame_s))
    starts: list[float] = []
    armed, speech_run, quiet_run = True, 0, 0
    for i, p in enumerate(probs):
        if armed:
            speech_run = speech_run + 1 if p >= threshold else 0
            if speech_run >= need_speech:
                starts.append((i + 1) * frame_s)
                armed, quiet_run = False, 0
        else:
            quiet_run = quiet_run + 1 if p < neg else 0
            if quiet_run >= need_quiet:
                armed, speech_run = True, 0
    return starts


# ----- text metrics -----------------------------------------------------

_ONES = "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()
_FILLERS = {"um", "uh", "er", "erm", "hmm", "mm"}


def _number_words(n: int) -> str:
    if n < 20:
        return _ONES[n]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " and " + _number_words(rest))
    return str(n)


def normalize(text: str) -> list[str]:
    text = text.lower().replace("'", "").replace("’", "").replace("-", " ")
    text = re.sub(r"\d+", lambda m: _number_words(int(m.group())), text)
    words = re.sub(r"[^a-z ]", " ", text).split()
    return [w for w in words if w not in _FILLERS and w != "and"]


def wer(reference: str, hypothesis: str) -> tuple[int, int]:
    """Return (edit distance in words, reference length)."""
    ref, hyp = normalize(reference), normalize(hypothesis)
    row = list(range(len(hyp) + 1))
    for i, r in enumerate(ref, 1):
        prev, row[0] = row[0], i
        for j, h in enumerate(hyp, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (r != h))
    return row[-1], len(ref)


def percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    k = (len(ordered) - 1) * q
    lo, hi = int(np.floor(k)), int(np.ceil(k))
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (k - lo)


def summarize(values: list[float]) -> dict:
    if not values:
        return {"n": 0}
    return {
        "n": len(values),
        "median": statistics.median(values),
        "p95": percentile(values, 0.95),
        "max": max(values),
    }


# ----- microphone -------------------------------------------------------

MIC_HELP = (
    "The microphone is not delivering any audio (nothing arrived for {timeout:.0f} s).\n"
    "Stop any program still stuck on it (Ctrl+C in its terminal), unplug the\n"
    "microphone and plug it back in, then run this again."
)


class MicrophoneSilent(RuntimeError):
    pass


def capture_blocks(sd, rate: int, channels: int, block: int, timeout_s: float = 2.0):
    """Yield captured blocks (frames x channels) until the caller stops.

    Raises MicrophoneSilent if no block arrives within timeout_s, instead
    of waiting forever on a device that has stopped sending audio.
    """
    import queue

    blocks: queue.Queue = queue.Queue()

    def callback(indata, frames, time_info, status):
        blocks.put(indata.copy())

    with sd.InputStream(samplerate=rate, channels=channels, dtype="float32", blocksize=block, callback=callback):
        while True:
            try:
                yield blocks.get(timeout=timeout_s)
            except queue.Empty:
                raise MicrophoneSilent(MIC_HELP.format(timeout=timeout_s)) from None


def check_microphone(sd, rate: int = 48_000, timeout_s: float = 2.0) -> None:
    """Fail fast, with instructions, if the microphone sends nothing."""
    for _ in capture_blocks(sd, rate, 2, rate // 100, timeout_s):
        return


# ----- GPU memory -------------------------------------------------------


def gpu_used_mib() -> int:
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
        capture_output=True, text=True, check=True,
    )
    return int(out.stdout.split()[0])


class VramPeak:
    """Samples total GPU memory in the background and keeps the peak."""

    def __init__(self, interval_ms: int = 250) -> None:
        self.peak = 0
        self._stop = threading.Event()
        self._interval = interval_ms / 1000
        self._thread = threading.Thread(target=self._run, daemon=True)

    def __enter__(self) -> VramPeak:
        self._thread.start()
        return self

    def __exit__(self, *exc) -> None:
        self._stop.set()
        self._thread.join()

    def _run(self) -> None:
        while not self._stop.is_set():
            self.peak = max(self.peak, gpu_used_mib())
            self._stop.wait(self._interval)


def output_latency_ms() -> float:
    """Estimate of device output latency: two PipeWire quanta.

    Replace with a loopback measurement in the attended session.
    """
    quantum, rate = 1024, 48_000
    try:
        out = subprocess.run(["pw-metadata", "-n", "settings"], capture_output=True, text=True, timeout=2).stdout
        quantum = int(re.search(r"'clock.quantum' value:'(\d+)'", out).group(1))
        rate = int(re.search(r"'clock.rate' value:'(\d+)'", out).group(1))
    except Exception:
        pass
    return 2 * quantum / rate * 1000
