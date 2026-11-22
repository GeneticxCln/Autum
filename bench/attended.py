"""Attended tests: audio plays aloud, so run this yourself (methodology §6).

    python3 -m bench.attended              # full session, about 8 minutes
    python3 -m bench.attended --skip-aec   # without the echo-cancellation comparison
    python3 -m bench.attended --only loopback

The session:

1. Loopback: a few clicks play and are recorded, measuring real audio
   latency (replaces the 42.7 ms output estimate).
2. For each condition, echo cancellation off and then on (PipeWire's
   WebRTC echo canceller, loaded only after you confirm and unloaded
   afterwards), 60 seconds of Autum's voice plays three times:
   - echo: stay silent; every speech detection is Autum hearing itself;
   - interruption: say each prompt when the screen tells you to;
   - backchannel: say each "mm-hm"-style prompt when told to; any
     detection is a false interruption.

Playback continues while you speak; this measures detection, not the
runtime's reaction. Recordings stay local in bench/data/attended-v1/;
numbers go to bench/results/<time>-attended.json.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import re
import signal
import subprocess
import time
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from bench.common import (
    DATA, RESULTS, SR, Vad, load_corpus, read_wav, resample, speech_starts, write_wav,
)

RATE = 48_000
OUT = DATA / "attended-v1"
PLAYBACK_S = 60.0
CUE_TIMES_S = [6.0, 20.0, 34.0, 48.0]
CUE_WINDOW_S = 6.0
MIN_SPEECH_SWEEP_MS = [96, 200, 300]
AEC_SINK, AEC_SOURCE = "autum_aec_sink", "autum_aec_source"
MIN_MATCH = 0.1  # pure noise scores about 0.01; a real loopback far more


# ----- pure analysis (unit-tested) --------------------------------------


def roundtrip_ms(played: np.ndarray, recorded: np.ndarray, rate: int = RATE, max_ms: float = 500.0) -> float | None:
    """Delay of the played signal inside the recording, by cross-correlation.

    Returns None unless the recording at that delay clearly matches the
    played signal (normalised correlation of at least MIN_MATCH), which
    fails on headphones or with the speakers muted.
    """
    from scipy.signal import correlate

    corr = correlate(recorded, played, mode="full", method="fft")[len(played) - 1 :]
    corr = corr[: int(max_ms / 1000 * rate)]
    lag = int(np.argmax(np.abs(corr)))
    segment = recorded[lag : lag + len(played)]
    reference = played[: len(segment)]
    match = abs(float(np.dot(reference, segment))) / (np.linalg.norm(reference) * np.linalg.norm(segment) + 1e-12)
    if match < MIN_MATCH:
        return None
    return lag / rate * 1000


def click_train(rate: int = RATE, clicks: int = 6, gap_s: float = 0.8) -> np.ndarray:
    rng = np.random.default_rng(1)
    burst = rng.standard_normal(int(0.005 * rate)) * np.hanning(int(0.005 * rate)) * 0.6
    signal_ = np.zeros(int((0.5 + clicks * gap_s + 0.5) * rate), np.float32)
    for k in range(clicks):
        start = int((0.5 + k * gap_s) * rate)
        signal_[start : start + len(burst)] = burst
    return signal_


def onset_after(audio: np.ndarray, t: float, rate: int = RATE, rise_db: float = 10.0) -> float | None:
    """First moment after t where the level rises rise_db above the second before t.

    Used as the speaker's speech onset. With echo cancellation off,
    Autum's voice in the recording can hide the onset; then it is None.
    """
    hop = rate // 100
    n = len(audio) // hop
    level = 20 * np.log10(np.sqrt(np.mean(audio[: n * hop].reshape(n, hop) ** 2, axis=1)) + 1e-9)
    i0 = int(t * 100)
    before = level[max(0, i0 - 100) : i0]
    if before.size == 0:
        return None
    limit = np.percentile(before, 90) + rise_db
    run = 0
    for i in range(i0, min(n, i0 + int(CUE_WINDOW_S * 100))):
        run = run + 1 if level[i] > limit else 0
        if run == 3:
            return (i - 2) / 100
    return None


def score_cues(starts: list[float], cues: list[float], window_s: float = CUE_WINDOW_S) -> list[float | None]:
    """The first detection inside each cue's window, or None."""
    found = []
    for cue in cues:
        hits = [s for s in starts if cue <= s < cue + window_s]
        found.append(hits[0] if hits else None)
    return found


def analyse(kind: str, audio: np.ndarray, probs: np.ndarray, offset_s: float, prompts: list[str]) -> dict:
    """Numbers for one 60 s run. offset_s: playback start in the recording."""
    cues = [offset_s + c for c in CUE_TIMES_S]
    end = offset_s + PLAYBACK_S
    result: dict = {}
    for min_ms in MIN_SPEECH_SWEEP_MS:
        starts = [s for s in speech_starts(probs, min_ms) if offset_s <= s <= end]
        if kind == "echo":
            result[min_ms] = {"false_detections": len(starts)}
            continue
        hits = score_cues(starts, cues)
        in_windows = sum(1 for s in starts if any(c <= s < c + CUE_WINDOW_S for c in cues))
        entry = {"outside_cue_windows": len(starts) - in_windows}
        if kind == "interruption":
            delays = []
            for cue, hit in zip(cues, hits):
                onset = onset_after(audio, cue) if hit is not None else None
                delays.append(round((hit - onset) * 1000) if onset is not None and hit >= onset else None)
            entry |= {"detected": f"{sum(h is not None for h in hits)}/{len(cues)}", "delay_ms": delays}
        else:
            entry |= {"false_interruptions": f"{sum(h is not None for h in hits)}/{len(cues)}"}
        result[min_ms] = entry
    result["prompts"] = prompts
    return result


# ----- playback ---------------------------------------------------------


def build_playback(path: Path) -> dict:
    """60 s of Autum's voice from the latest saved omni-voice replies; built once."""
    if not path.exists():
        runs = sorted(DATA.glob("outputs/*/C"))
        if not runs:
            raise SystemExit("No saved replies. Run: python3 -m bench.run_pipelines --pipelines C --save-audio")
        pieces, total = [], 0.0
        for item in load_corpus():
            wav = runs[-1] / f"{item.id}.wav"
            if not wav.exists():
                continue
            audio, sr = read_wav(wav)
            audio = resample(audio, sr, RATE)
            pieces += [audio, np.zeros(int(0.4 * RATE), np.float32)]
            total += len(audio) / RATE + 0.4
            if total >= PLAYBACK_S:
                break
        clip = np.concatenate(pieces)[: int(PLAYBACK_S * RATE)]
        clip[-int(0.05 * RATE) :] *= np.linspace(1, 0, int(0.05 * RATE))
        # Aim for -20 dBFS RMS, but never let peaks pass -1 dBFS (no clipping).
        rms, peak = np.sqrt(np.mean(clip**2)) + 1e-9, np.abs(clip).max() + 1e-9
        clip *= min(10 ** (-20 / 20) / rms, 10 ** (-1 / 20) / peak)
        write_wav(path, clip, RATE)
        level = round(float(20 * np.log10(np.sqrt(np.mean(clip**2)))), 1)
        (path.parent / "playback.json").write_text(json.dumps({"source": str(runs[-1]), "rms_dbfs": level}))
    return {
        "file": str(path),
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        **json.loads((path.parent / "playback.json").read_text()),
    }


# ----- PipeWire ---------------------------------------------------------


def run(*args: str) -> str:
    return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout


def node_info(which: str) -> dict:
    info = run("wpctl", "inspect", f"@DEFAULT_AUDIO_{which}@")
    desc = re.search(r'node.description = "([^"]*)"', info)
    volume = re.search(r"Volume: ([0-9.]+)", run("wpctl", "get-volume", f"@DEFAULT_AUDIO_{which}@"))
    return {"device": desc.group(1) if desc else "unknown", "volume": float(volume.group(1)) if volume else None}


def unload_leftover_aec() -> None:
    """Remove an echo canceller this tool left behind if a run was killed."""
    for line in run("pactl", "list", "modules", "short").splitlines():
        if "module-echo-cancel" in line and AEC_SINK in line:
            run("pactl", "unload-module", line.split()[0])


@contextmanager
def echo_cancel():
    """Load PipeWire's WebRTC echo canceller; always unload it afterwards."""
    unload_leftover_aec()
    out = subprocess.run(
        ["pactl", "load-module", "module-echo-cancel", "aec_method=webrtc",
         f"source_name={AEC_SOURCE}", f"sink_name={AEC_SINK}"],
        capture_output=True, text=True, timeout=10,
    )
    if out.returncode != 0:
        raise RuntimeError(f"could not load echo cancellation: {out.stderr.strip()}")
    module = out.stdout.strip()
    try:
        for _ in range(50):
            if AEC_SINK in run("pactl", "list", "short", "sinks") and AEC_SOURCE in run("pactl", "list", "short", "sources"):
                break
            time.sleep(0.1)
        else:
            raise RuntimeError("echo-cancel devices did not appear")
        yield
    finally:
        run("pactl", "unload-module", module)


def play_and_record(playback: Path, out: Path, aec: bool, cues: list[tuple[float, str]]) -> tuple[np.ndarray, float]:
    """Play the file and record the microphone; returns (audio, playback start in recording)."""
    target_in = ["--target", AEC_SOURCE] if aec else []
    target_out = ["--target", AEC_SINK] if aec else []
    rec = subprocess.Popen(["pw-record", *target_in, "--rate", str(RATE), "--channels", "2", "--format", "s16", str(out)])
    rec_start = time.perf_counter()
    try:
        time.sleep(0.8)
        play_start = time.perf_counter()
        play = subprocess.Popen(["pw-play", *target_out, str(playback)])
        pending = list(cues)
        while play.poll() is None:
            if pending and time.perf_counter() - play_start >= pending[0][0]:
                print(f"\a  >>> Say now: {pending.pop(0)[1]}")
            time.sleep(0.01)
        time.sleep(1.0)
    finally:
        rec.send_signal(signal.SIGINT)
        rec.wait(timeout=5)
    audio, sr = read_wav(out)
    return resample(audio, sr, RATE), play_start - rec_start


# ----- session ----------------------------------------------------------


def loopback() -> dict:
    import sounddevice as sd

    clicks = click_train()
    trips = []
    for _ in range(3):
        recorded = sd.playrec(np.column_stack([clicks, clicks]), RATE, channels=2, dtype="float32", blocking=True)
        trips.append(roundtrip_ms(clicks, recorded.mean(axis=1)))
    with sd.Stream(samplerate=RATE, channels=(2, 2), dtype="float32") as stream:
        reported_in, reported_out = (x * 1000 for x in stream.latency)
    found = [t for t in trips if t is not None]
    median = float(np.median(found)) if found else None
    return {
        "roundtrip_ms": [None if t is None else round(t, 1) for t in trips],
        "roundtrip_median_ms": median,
        "reported_input_ms": round(reported_in, 1),
        "reported_output_ms": round(reported_out, 1),
        "output_latency_estimate_ms": None if median is None else round(median - reported_in, 1),
        "note": "round trip = output + air + input; output estimate subtracts the reported input latency",
    }


def ask(question: str) -> bool:
    return input(f"{question} [y/N] ").strip().lower() == "y"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-aec", action="store_true", help="do not load echo cancellation")
    parser.add_argument("--only", choices=["loopback", "echo", "interruption", "backchannel"])
    args = parser.parse_args()

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    session_dir = OUT / stamp
    session_dir.mkdir(parents=True, exist_ok=True)
    sink, source = node_info("SINK"), node_info("SOURCE")
    print(f"Speakers:   {sink['device']}   volume {sink['volume']}")
    print(f"Microphone: {source['device']}   level {source['volume']}")
    print("Audio will play aloud. Use speakers, not headphones, at your normal listening")
    print("volume, and keep that volume for the whole session.")
    if not ask("Continue?"):
        return
    report: dict = {"date": stamp, "speakers": sink, "microphone": source, "runs": {}}

    if args.only in (None, "loopback"):
        input("\nLoopback: a few clicks will play. Stay quiet. Press Enter.")
        report["loopback"] = loopback()
        print(json.dumps(report["loopback"], indent=1))

    if args.only != "loopback":
        report["playback"] = build_playback(OUT / "playback.wav")
        corpus = load_corpus()
        prompts = {
            "interruption": [i.text for i in corpus if i.category == "interruption"],
            "backchannel": [i.text for i in corpus if i.category == "backchannel"],
            "echo": [],
        }
        kinds = [args.only] if args.only else ["echo", "interruption", "backchannel"]
        conditions = [False] if args.skip_aec else [False, True]
        vad = Vad()
        for aec in conditions:
            label = "aec_on" if aec else "aec_off"
            if aec and not ask("\nLoad PipeWire's echo cancellation for the second half? It is unloaded afterwards."):
                break
            try:
                context = echo_cancel() if aec else _nothing()
                with context:
                    for kind in kinds:
                        if node_info("SINK")["volume"] != sink["volume"]:
                            print("WARNING: the speaker volume changed; set it back for consistent results.")
                        instructions = {
                            "echo": "stay completely silent while Autum talks for 60 seconds.",
                            "interruption": "when the screen says 'Say now', say that phrase over Autum's voice.",
                            "backchannel": "when the screen says 'Say now', say that sound as you would while listening.",
                        }[kind]
                        input(f"\n[{label}] {kind}: {instructions} Press Enter to start.")
                        cues = list(zip(CUE_TIMES_S, prompts[kind])) if prompts[kind] else []
                        wav = session_dir / f"{label}-{kind}.wav"
                        audio, offset = play_and_record(OUT / "playback.wav", wav, aec, cues)
                        probs = vad.probabilities(resample(audio, RATE, SR))
                        report["runs"].setdefault(label, {})[kind] = analyse(kind, audio, probs, offset, prompts[kind])
                        print(json.dumps(report["runs"][label][kind], indent=1))
            except RuntimeError as exc:
                print(f"{label} stopped: {exc}")
                report["runs"][label] = {"error": str(exc)}

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stamp}-attended.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {path}\nrecordings in {session_dir} (local only)")


@contextmanager
def _nothing():
    yield


if __name__ == "__main__":
    main()
