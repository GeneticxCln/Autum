"""Run candidate voice pipelines end to end over a labelled corpus (WP1).

For each utterance the runner:

1. Replays the audio through Silero VAD as a streaming endpoint detector
   and measures how long after the true end of speech it fires.
2. Sends only the audio up to that point to the pipeline, as a live
   system would.
3. Streams the model's reply, cuts it into speakable chunks with Autum's
   own chunker, and synthesizes them in order while the model is still
   generating, so stages compete for the GPU as they will in use.
4. Records time to first audio, playback gaps, recognition errors, and
   the reply, and saves the spoken reply for listening.

Reported first-audio latency = endpoint delay + processing to the first
synthesized audio + estimated device output latency.

Usage:
  python3 -m bench.run_pipelines --corpus bench/data/synthetic-v2 [--pipelines A,C] [--limit 5]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import platform
import subprocess
import time
from pathlib import Path

import httpx
import numpy as np

from autum.text import SentenceChunker, speakable
from bench.common import (
    DATA, RESULTS, SR, Vad, VramPeak, gpu_used_mib, load_corpus, load_labels,
    output_latency_ms, read_wav, resample, simulate_endpoint, summarize, wer, write_wav,
)
from bench.stages import (
    SYSTEM_PROMPT, ChatterboxTts, OllamaAsr, OllamaAudioChat, OllamaChat, OllamaTts, WhisperAsr,
)

OMNI = "lfm3-omni:2b"
PIPELINES = {
    "A": ("cascade: whisper-base -> qwen3:8b -> omni speech", lambda: (WhisperAsr("base"), OllamaChat("qwen3:8b"), OllamaTts(OMNI))),
    "B": ("cascade: whisper-base -> qwen3:8b -> chatterbox", lambda: (WhisperAsr("base"), OllamaChat("qwen3:8b"), ChatterboxTts())),
    "C": ("omni only: omni transcribe -> omni chat -> omni speech", lambda: (OllamaAsr(OMNI), OllamaChat(OMNI), OllamaTts(OMNI))),
    "D": ("omni hears audio directly -> omni speech", lambda: (None, OllamaAudioChat(OMNI), OllamaTts(OMNI))),
    "E": ("cascade: whisper-base -> omni chat -> omni speech", lambda: (WhisperAsr("base"), OllamaChat(OMNI), OllamaTts(OMNI))),
}
SILENCE_SWEEP_MS = [300, 500, 700, 1000]
PIPELINE_SILENCE_MS = 500
PREMATURE_TOLERANCE_S = 0.05
WARMUP_CLIPS = 2  # not counted; the first also gives the cold-start time


def endpoint_report(corpus_dir: Path, items, labels, vad: Vad) -> tuple[dict, dict]:
    """Sweep the silence threshold; return per-threshold stats and per-item endpoints."""
    sweep: dict[int, dict] = {}
    chosen: dict[str, dict] = {}
    for silence_ms in SILENCE_SWEEP_MS:
        delays, premature, missed = [], 0, 0
        by_category: dict[str, list[int]] = {}
        for item in items:
            audio, sr = read_wav(corpus_dir / f"{item.id}.wav")
            probs = vad.probabilities(resample(audio, sr, SR))
            ep = simulate_endpoint(probs, silence_ms)
            eos = labels[item.id]["speech_end_s"]
            early = ep.endpoint_s is not None and ep.endpoint_s < eos - PREMATURE_TOLERANCE_S
            stats = by_category.setdefault(item.category, [0, 0])
            stats[1] += 1
            if ep.endpoint_s is None:
                missed += 1
            elif early:
                premature += 1
                stats[0] += 1
            else:
                delays.append((ep.endpoint_s - eos) * 1000)
            if silence_ms == PIPELINE_SILENCE_MS:
                chosen[item.id] = {"endpoint_s": ep.endpoint_s, "premature": early, "eos_s": eos}
        sweep[silence_ms] = {
            "endpoint_delay_ms": summarize(delays),
            "premature_rate": premature / len(items),
            "missed": missed,
            "premature_by_category": {c: f"{p}/{n}" for c, (p, n) in by_category.items()},
        }
    return sweep, chosen


async def run_item(client, asr, llm, tts, item, audio16k, endpoint, out_dir: Path | None) -> dict:
    cut = endpoint["endpoint_s"] or len(audio16k) / SR
    heard = audio16k[: int(cut * SR)]
    endpoint_delay_ms = max(0.0, (cut - endpoint["eos_s"]) * 1000)
    result: dict = {"id": item.id, "category": item.category, "premature": endpoint["premature"]}

    t0 = time.perf_counter()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *item.history]
    if llm.takes_audio:
        transcript = None
    else:
        transcript = await asr.transcribe(client, heard)
        result["asr_ms"] = (time.perf_counter() - t0) * 1000
        result["transcript"] = transcript
        result["word_errors"], result["ref_words"] = wer(item.reference, transcript)
        messages.append({"role": "user", "content": transcript or "(silence)"})

    chunks: asyncio.Queue[str | None] = asyncio.Queue()
    reply: list[str] = []
    marks: dict[str, float] = {}

    async def generate() -> None:
        chunker = SentenceChunker()
        async for delta in llm.stream(client, messages, audio=heard):
            marks.setdefault("first_token", time.perf_counter())
            reply.append(delta)
            for chunk in chunker.feed(delta):
                marks.setdefault("first_chunk", time.perf_counter())
                await chunks.put(chunk)
        for chunk in chunker.flush():
            marks.setdefault("first_chunk", time.perf_counter())
            await chunks.put(chunk)
        marks["llm_done"] = time.perf_counter()
        await chunks.put(None)

    async def speak() -> list[tuple[float, float, float]]:
        timeline = []  # (seconds to first audio from t0, audio seconds, synth seconds)
        while (chunk := await chunks.get()) is not None:
            text = speakable(chunk)
            if not text:
                continue
            start = time.perf_counter()
            ttfa, audio = await tts.synth(client, text)
            timeline.append((start + ttfa - t0, len(audio) / tts.sr, time.perf_counter() - start))
            spoken.append(audio)
        return timeline

    spoken: list[np.ndarray] = []
    _, timeline = await asyncio.gather(generate(), speak())

    text = "".join(reply)
    result["reply"] = text
    result["llm_first_token_ms"] = (marks.get("first_token", t0) - t0) * 1000
    result["llm_first_chunk_ms"] = (marks.get("first_chunk", t0) - t0) * 1000
    if not timeline:
        result["error"] = "no speech produced"
        return result
    first_audio_s = timeline[0][0]
    gaps, playhead = 0.0, first_audio_s
    for ready, duration, _ in timeline:
        gaps += max(0.0, ready - playhead)
        playhead = max(playhead, ready) + duration
    result["processing_to_first_audio_ms"] = first_audio_s * 1000
    result["endpoint_delay_ms"] = endpoint_delay_ms
    result["first_audio_latency_ms"] = endpoint_delay_ms + first_audio_s * 1000 + output_latency_ms()
    result["playback_gap_ms"] = gaps * 1000
    result["tts_rtf"] = sum(t[2] for t in timeline) / max(sum(t[1] for t in timeline), 1e-6)
    result["reply_audio_s"] = sum(t[1] for t in timeline)
    if item.expect_any:
        result["context_ok"] = any(word.lower() in text.lower() for word in item.expect_any)
    if out_dir:
        write_wav(out_dir / f"{item.id}.wav", np.concatenate(spoken), tts.sr)
    return result


async def ollama_loaded(client) -> list[dict]:
    r = await client.get("http://127.0.0.1:11434/api/ps")
    return [{"model": m["name"], "vram_mib": m.get("size_vram", 0) // 2**20} for m in r.json().get("models", [])]


async def unload_all(client) -> None:
    """Unload every Ollama model and wait until the GPU memory has settled."""
    for m in await ollama_loaded(client):
        await client.post("http://127.0.0.1:11434/api/generate", json={"model": m["model"], "keep_alive": 0})
    for _ in range(60):
        if not await ollama_loaded(client):
            break
        await asyncio.sleep(0.5)
    else:
        raise RuntimeError("Ollama did not unload its models within 30 s")
    previous = -1
    for _ in range(20):  # wait for two equal readings 0.5 s apart
        await asyncio.sleep(0.5)
        current = gpu_used_mib()
        if abs(current - previous) < 16:
            return
        previous = current


def summarize_pipeline(results: list[dict]) -> dict:
    ok = [r for r in results if "first_audio_latency_ms" in r]
    errors = sum(r.get("word_errors", 0) for r in results)
    words = sum(r.get("ref_words", 0) for r in results)
    checks = [r["context_ok"] for r in results if "context_ok" in r]
    return {
        "first_audio_latency_ms": summarize([r["first_audio_latency_ms"] for r in ok]),
        "processing_to_first_audio_ms": summarize([r["processing_to_first_audio_ms"] for r in ok]),
        "asr_ms": summarize([r["asr_ms"] for r in results if "asr_ms" in r]),
        "llm_first_chunk_ms": summarize([r["llm_first_chunk_ms"] for r in ok]),
        "wer": errors / words if words else None,
        "wer_by_category": _wer_by_category(results),
        "context_checks": f"{sum(checks)}/{len(checks)}" if checks else None,
        "utterances_with_gaps": sum(1 for r in ok if r["playback_gap_ms"] > 50),
        "tts_rtf_median": summarize([r["tts_rtf"] for r in ok]).get("median"),
        "failures": len(results) - len(ok),
    }


def _wer_by_category(results: list[dict]) -> dict:
    out: dict[str, list[int]] = {}
    for r in results:
        if "ref_words" in r:
            acc = out.setdefault(r["category"], [0, 0])
            acc[0] += r["word_errors"]
            acc[1] += r["ref_words"]
    return {c: round(e / n, 3) for c, (e, n) in out.items() if n}


def environment() -> dict:
    def cmd(*args):
        try:
            return subprocess.run(args, capture_output=True, text=True, timeout=10).stdout.strip()
        except Exception as exc:
            return f"unavailable: {exc}"

    import ctranslate2
    import faster_whisper
    import torch

    return {
        "date": dt.datetime.now().isoformat(timespec="seconds"),
        "gpu": cmd("nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"),
        "cpu": platform.processor() or platform.machine(),
        "python": platform.python_version(),
        "ollama": cmd("ollama", "--version"),
        "ollama_models": cmd("ollama", "list"),
        "faster_whisper": faster_whisper.__version__,
        "ctranslate2": ctranslate2.__version__,
        "torch": torch.__version__,
        "output_latency_estimate_ms": output_latency_ms(),
        "endpoint_silence_ms": PIPELINE_SILENCE_MS,
    }


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, default=DATA / "synthetic-v2")
    parser.add_argument("--pipelines", default=",".join(PIPELINES))
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--save-audio", action="store_true")
    args = parser.parse_args()

    labels = load_labels(args.corpus)
    items = [i for i in load_corpus() if i.id in labels]
    if args.limit:
        items = items[: args.limit]
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    report = {"corpus": str(args.corpus), "items": len(items), "environment": environment(), "pipelines": {}}

    vad = Vad()
    report["endpointing"], endpoints = endpoint_report(args.corpus, items, labels, vad)
    print(json.dumps(report["endpointing"], indent=1))

    async with httpx.AsyncClient(timeout=120) as client:
        for key in args.pipelines.split(","):
            title, build = PIPELINES[key]
            print(f"\n== {key}: {title}")
            await unload_all(client)
            baseline = gpu_used_mib()
            load_start = time.perf_counter()
            asr, llm, tts = build()
            for n, warm in enumerate(items[:WARMUP_CLIPS]):
                audio, sr = read_wav(args.corpus / f"{warm.id}.wav")
                await run_item(client, asr, llm, tts, warm, resample(audio, sr, SR), endpoints[warm.id], None)
                if n == 0:
                    load_s = time.perf_counter() - load_start
            out_dir = DATA / "outputs" / stamp / key if args.save_audio else None
            results = []
            with VramPeak() as vram:
                for item in items:
                    audio, sr = read_wav(args.corpus / f"{item.id}.wav")
                    try:
                        r = await run_item(client, asr, llm, tts, item, resample(audio, sr, SR), endpoints[item.id], out_dir)
                    except Exception as exc:
                        r = {"id": item.id, "category": item.category, "error": f"{type(exc).__name__}: {exc}"}
                    results.append(r)
                    print(f"{item.id:15} {r.get('first_audio_latency_ms', float('nan')):7.0f} ms  {r.get('transcript', '')[:50]!r} -> {r.get('reply', r.get('error', ''))[:70]!r}")
            summary = summarize_pipeline(results)
            summary["cold_start_s"] = load_s
            summary["vram_peak_mib"] = vram.peak
            summary["vram_baseline_mib"] = baseline
            summary["vram_over_baseline_mib"] = vram.peak - baseline
            summary["ollama_loaded"] = await ollama_loaded(client)
            report["pipelines"][key] = {"title": title, "summary": summary, "results": results}
            print(json.dumps(summary, indent=1))
            del asr, llm, tts
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception:
                pass
            await unload_all(client)

    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{stamp}-{args.corpus.name}.json"
    path.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
