"""Render the corpus with Chatterbox to get a labelled synthetic test set.

Synthetic speech is cleaner and more regular than a real voice through a
real microphone, so results from it are a harness check and a first
comparison, not acceptance evidence. The recorded corpus
(record_corpus.py) is the real test set.

Every clip is seeded from its prompt ID, so a clip is the same whatever
order it is generated in. (synthetic-v1 was made before seeding.)

Usage: python3 -m bench.make_synthetic [--version synthetic-v2] [--only short-01]
"""

from __future__ import annotations

import argparse
import asyncio
import json

import numpy as np
import torch

from bench.common import (
    DATA, NOISE_SNR_DB, SR, active_span, item_seed, load_corpus, mix_at_snr, noise_for, resample, write_wav,
)
from bench.stages import ChatterboxTts

LEAD_S, PAUSE_S, TRAIL_S = 0.4, 0.9, 1.6


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--version", default="synthetic-v2")
    parser.add_argument("--only", help="generate one prompt (to check reproducibility)")
    args = parser.parse_args()
    out = DATA / args.version
    tts = ChatterboxTts()
    labels: dict[str, dict] = {}
    for item in load_corpus():
        if item.recorded_only or (args.only and item.id != args.only):
            continue
        pieces = [np.zeros(int(LEAD_S * SR), np.float32)]
        segments = []
        cursor = LEAD_S
        for i, text in enumerate(part.strip() for part in item.text.split("|")):
            seed = item_seed(item.id, i)
            torch.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
            _, audio = await tts.synth(None, text)
            audio = resample(audio, tts.sr, SR)
            a, b = active_span(audio, SR)
            speech = audio[int(a * SR) : int(b * SR)]
            if i:
                pieces.append(np.zeros(int(PAUSE_S * SR), np.float32))
                cursor += PAUSE_S
            pieces.append(speech)
            segments.append([round(cursor, 3), round(cursor + len(speech) / SR, 3)])
            cursor += len(speech) / SR
        pieces.append(np.zeros(int(TRAIL_S * SR), np.float32))
        clip = np.concatenate(pieces)
        clip *= 0.5 / max(np.abs(clip).max(), 1e-6)
        if item.category == "noise":
            span = (int(segments[0][0] * SR), int(segments[-1][1] * SR))
            clip = mix_at_snr(clip, noise_for(item.id, len(clip)), NOISE_SNR_DB, span)
        write_wav(out / f"{item.id}.wav", clip, SR)
        labels[item.id] = {
            "speech_start_s": segments[0][0],
            "speech_end_s": segments[-1][1],
            "segments": segments,
            "source": "synthetic:chatterbox-turbo",
            "seed": item_seed(item.id),
        }
        print(f"{item.id}: {segments[-1][1] - segments[0][0]:.2f} s speech")
    (out / "labels.json").write_text(json.dumps(labels, indent=1))
    print(f"wrote {len(labels)} clips to {out}")


if __name__ == "__main__":
    asyncio.run(main())
