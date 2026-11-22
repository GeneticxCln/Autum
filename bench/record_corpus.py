"""Record and label the primary test set (methodology §2.2).

Run it yourself in a terminal; it records your voice from the default
microphone:

    python3 -m bench.record_corpus                 # record everything not yet accepted
    python3 -m bench.record_corpus --redo pause-03  # record one prompt again

Each prompt is recorded until you have been quiet for 2.5 seconds after
speaking. You then review the labels, listen if you like, and accept or
re-record. Accepted clips are saved at once, so you can stop at any time
and continue later; each sitting is logged as a session.

Files, all local and excluded from git:
  bench/data/<set>/masters/<id>.wav  48 kHz mono, as recorded (the master)
  bench/data/<set>/<id>.wav          16 kHz benchmark copy (noise prompts get the fixed noise)
  bench/data/<set>/labels.json       speech start/end, segments, how each label was made
  bench/data/<set>/session.json      microphone, input level, noise floor per sitting
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from bench.common import (
    DATA, FRAME, NOISE_SNR_DB, SR, Item, MicrophoneSilent, Vad, active_span, capture_blocks,
    check_microphone, load_corpus, mix_at_snr, noise_for, resample, write_wav,
)

RATE = 48_000
BLOCK = RATE // 100  # 10 ms
STOP_AFTER_SILENCE_S = 2.5
MAX_CLIP_S = 25.0
MIN_LEAD_S = 0.3
DISAGREE_S = 0.10
SEGMENT_GAP_S = 0.25


# ----- pure logic (unit-tested) -----------------------------------------


def dbfs(block: np.ndarray) -> float:
    return 20 * np.log10(np.sqrt(np.mean(block**2)) + 1e-9)


class AutoStop:
    """Decides when to stop recording: after speech, then sustained quiet.

    Only controls the recorder. Labels are computed separately.
    """

    def __init__(self, noise_floor_dbfs: float, block_s: float = BLOCK / RATE) -> None:
        self.threshold = max(noise_floor_dbfs + 12.0, -55.0)
        self.block_s = block_s
        self.voiced_s = 0.0
        self.quiet_s = 0.0
        self.elapsed_s = 0.0

    @property
    def heard_speech(self) -> bool:
        return self.voiced_s >= 0.1

    def feed(self, level_dbfs: float) -> bool:
        self.elapsed_s += self.block_s
        if level_dbfs > self.threshold:
            self.voiced_s += self.block_s
            self.quiet_s = 0.0
        elif self.heard_speech:
            self.quiet_s += self.block_s
        return (self.heard_speech and self.quiet_s >= STOP_AFTER_SILENCE_S) or self.elapsed_s >= MAX_CLIP_S


@dataclass
class Labels:
    energy: tuple[float, float]
    vad: tuple[float, float] | None
    segments: list[list[float]]
    peak: float

    @property
    def flagged(self) -> bool:
        if self.vad is None:
            return True
        return (
            abs(self.energy[0] - self.vad[0]) > DISAGREE_S
            or abs(self.energy[1] - self.vad[1]) > DISAGREE_S
        )

    def problems(self, duration_s: float) -> list[str]:
        found = []
        if self.vad is None:
            found.append("no speech detected by the VAD")
        if self.peak >= 0.99:
            found.append("clipping: lower the input level only if every clip so far is re-recorded")
        if self.peak < 0.05:
            found.append("very quiet: move closer to the microphone")
        if self.energy[0] < MIN_LEAD_S:
            found.append("started speaking too soon after recording began")
        if duration_s - self.energy[1] < 1.5:
            found.append("too little silence after speaking")
        return found


def label_clip(audio48k: np.ndarray, vad: Vad) -> Labels:
    """Energy labels plus a Silero VAD cross-check (methodology §2.2)."""
    energy = active_span(audio48k, RATE)
    audio16k = resample(audio48k, RATE, SR)
    probs = vad.probabilities(audio16k)
    frame_s = FRAME / SR
    speech = np.flatnonzero(probs >= 0.5)
    vad_span = None
    segments: list[list[float]] = []
    if speech.size:
        vad_span = (speech[0] * frame_s, (speech[-1] + 1) * frame_s)
        start = prev = speech[0]
        for frame in speech[1:]:
            if (frame - prev - 1) * frame_s >= SEGMENT_GAP_S:
                segments.append([round(start * frame_s, 3), round((prev + 1) * frame_s, 3)])
                start = frame
            prev = frame
        segments.append([round(start * frame_s, 3), round((prev + 1) * frame_s, 3)])
    return Labels(energy, vad_span, segments, float(np.abs(audio48k).max()))


def benchmark_copy(master48k: np.ndarray, item: Item, start_s: float, end_s: float) -> np.ndarray:
    """16 kHz copy used by the benchmark; noise prompts get their fixed noise."""
    clip = resample(master48k, RATE, SR)
    if item.category == "noise":
        span = (int(start_s * SR), int(end_s * SR))
        clip = mix_at_snr(clip, noise_for(item.id, len(clip)), NOISE_SNR_DB, span)
    return clip


def save_json(path: Path, data) -> None:
    """Write atomically so an interrupted save never corrupts the labels."""
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, indent=1))
    os.replace(tmp, path)


# ----- microphone and terminal ------------------------------------------


def pipewire_source() -> dict:
    def run(*args):
        return subprocess.run(args, capture_output=True, text=True, timeout=5).stdout

    info = run("wpctl", "inspect", "@DEFAULT_AUDIO_SOURCE@")
    desc = re.search(r'node.description = "([^"]*)"', info)
    name = re.search(r'node.name = "([^"]*)"', info)
    volume = re.search(r"Volume: ([0-9.]+)", run("wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"))
    node = name.group(1) if name else ""
    return {
        "device": desc.group(1) if desc else "unknown",
        "source_volume": float(volume.group(1)) if volume else None,
        "filtered": any(k in node.lower() for k in ("echo", "rnnoise", "noise", "filter")),
    }


def record(sd, noise_floor: float | None) -> np.ndarray:
    blocks = []
    stopper = AutoStop(noise_floor if noise_floor is not None else -60.0)
    for data in capture_blocks(sd, RATE, 2, BLOCK):
        mono = data.mean(axis=1)
        blocks.append(mono)
        if noise_floor is None:
            if len(blocks) * BLOCK >= 2 * RATE:
                break
        elif stopper.feed(dbfs(mono)):
            break
    return np.concatenate(blocks)


def show(item: Item) -> None:
    print("\n" + "=" * 70)
    print(f"{item.id}  ({item.category})")
    if item.history:
        print("  Context (do not read aloud):")
        for turn in item.history:
            print(f"    {turn['role']}: {turn['content']}")
    if item.category == "noise":
        print("  Read normally. The noise is added afterwards.")
    text = item.text.replace("|", " ... (pause about a second) ... ")
    print(f"\n    {text}\n")


def main() -> int:
    try:
        return session()
    except MicrophoneSilent as exc:
        print(f"\n{exc}\nAccepted clips so far are kept.")
        return 1


def session() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--set", default="recorded-v1")
    parser.add_argument("--redo", action="append", default=[], help="prompt ID to record again")
    args = parser.parse_args()

    import sounddevice as sd

    out = DATA / args.set
    (out / "masters").mkdir(parents=True, exist_ok=True)
    labels_path, session_path = out / "labels.json", out / "session.json"
    labels = json.loads(labels_path.read_text()) if labels_path.exists() else {}
    sessions = json.loads(session_path.read_text()) if session_path.exists() else []

    source = pipewire_source()
    print(f"Microphone: {source['device']}   input level: {source['source_volume']}")
    check_microphone(sd)
    print("Microphone check: audio is arriving.")
    if source["filtered"]:
        print("WARNING: the default source looks like a noise-suppression or echo-cancel filter.")
        print("The primary set must be recorded without processing (methodology §2.2).")
    if sessions:
        first = sessions[0]
        if (first["device"], first["source_volume"]) != (source["device"], source["source_volume"]):
            print(f"WARNING: earlier sittings used {first['device']} at level {first['source_volume']}.")
            print("Change it back, or this sitting is not consistent with the rest of the set.")
    if input("Continue? [y/N] ").strip().lower() != "y":
        return 0

    input("\nNoise floor check: stay quiet for 2 seconds. Press Enter to start.")
    floor = dbfs(record(sd, None))
    print(f"Room noise floor: {floor:.1f} dBFS")
    session = {
        "id": len(sessions) + 1,
        "started": dt.datetime.now().isoformat(timespec="seconds"),
        **source,
        "sample_rate": RATE,
        "channels": "mean of the device's 2 channels",
        "noise_floor_dbfs": round(floor, 1),
    }
    sessions.append(session)
    save_json(session_path, sessions)

    vad = Vad()
    todo = [i for i in load_corpus() if not i.recorded_only and (i.id not in labels or i.id in args.redo)]
    print(f"\n{len(todo)} prompts to record. Ctrl+C stops safely; accepted clips are kept.")
    try:
        for n, item in enumerate(todo, 1):
            print(f"\n[{n}/{len(todo)}]", end="")
            while True:
                show(item)
                input("Press Enter, wait half a second, then read. Recording stops after 2.5 s of quiet.")
                master = record(sd, floor)
                lab = label_clip(master, vad)
                duration = len(master) / RATE
                print(f"  energy {lab.energy[0]:.2f}-{lab.energy[1]:.2f} s   "
                      f"VAD {'none' if lab.vad is None else f'{lab.vad[0]:.2f}-{lab.vad[1]:.2f} s'}   "
                      f"peak {20 * np.log10(lab.peak + 1e-9):.1f} dBFS")
                for problem in lab.problems(duration):
                    print(f"  ! {problem}")
                if lab.flagged:
                    print("  ! energy and VAD disagree by more than 100 ms: listen, then choose e, v or m")
                choice = review(sd, master, lab)
                if choice == "r":
                    continue
                if choice == "q":
                    raise KeyboardInterrupt
                method, (start, end) = choice
                write_wav(out / "masters" / f"{item.id}.wav", master, RATE)
                write_wav(out / f"{item.id}.wav", benchmark_copy(master, item, start, end), SR)
                labels[item.id] = {
                    "speech_start_s": round(start, 3),
                    "speech_end_s": round(end, 3),
                    "segments": lab.segments,
                    "label_method": method,
                    "energy": [round(x, 3) for x in lab.energy],
                    "vad": [round(x, 3) for x in lab.vad] if lab.vad else None,
                    "flagged": lab.flagged,
                    "peak_dbfs": round(20 * np.log10(lab.peak + 1e-9), 1),
                    "session": session["id"],
                    "source": "recorded",
                }
                save_json(labels_path, labels)
                break
    except KeyboardInterrupt:
        print("\nStopped.")
    remaining = sum(1 for i in load_corpus() if not i.recorded_only and i.id not in labels)
    print(f"{len(labels)} clips accepted, {remaining} still to record. Saved in {out}")
    return 0


def play_with_timeout(sd, audio: np.ndarray, rate: int = RATE) -> None:
    """Play a clip; give up a few seconds after it should have ended."""
    sd.play(audio, rate)
    deadline = time.monotonic() + len(audio) / rate + 3.0
    while sd.get_stream().active and time.monotonic() < deadline:
        time.sleep(0.05)
    sd.stop()


def review(sd, master: np.ndarray, lab: Labels):
    """Return 'r', 'q', or (method, (start, end))."""
    while True:
        options = "[p] play  [r] re-record  [e] use energy  [v] use VAD  [m] type times  [q] quit"
        if not lab.flagged:
            options = "[Enter] accept  " + options
        answer = input(options + "\n> ").strip().lower()
        if answer == "" and not lab.flagged:
            return "energy", lab.energy
        if answer == "p":
            play_with_timeout(sd, master)
        elif answer in ("r", "q"):
            return answer
        elif answer == "e":
            return "energy", lab.energy
        elif answer == "v" and lab.vad:
            return "vad", lab.vad
        elif answer == "m":
            try:
                start, end = (float(x) for x in input("start,end in seconds: ").split(","))
                if 0 <= start < end <= len(master) / RATE:
                    return "manual", (start, end)
            except ValueError:
                pass
            print("  enter two times like 0.62,2.41")


if __name__ == "__main__":
    sys.exit(main())
