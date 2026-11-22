"""Does closing a request actually stop the model? (methodology §7)

For each Ollama model and for Ollama speech output, compare GPU use in
the second after:

- cancel:  a long request is closed by the client after it has run a while;
- control: the same long request keeps running.

If cancellation works, GPU use after a cancel drops to the idle level,
and a short request sent right afterwards starts as fast as one sent to
an idle server.

Speech is probed twice: with a long passage, and with one sentence,
which is how Autum sends text to speech output.

Usage: python3 -m bench.cancel_probe [--only speech]
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import json
import statistics
import subprocess
import threading
import time

import httpx

from bench.common import RESULTS
from bench.stages import CONTEXT, KEEP_ALIVE, OLLAMA, SAMPLING, SEED

LONG_PROMPT = "Count slowly from one to five hundred in words, separated by commas."
SHORT_PROMPT = "Reply with the single word: ready."
LONG_SPEECH = " ".join(["This is a long passage that keeps the speech model busy for a while."] * 30)
SENTENCE = "Sure, the capital of Canada is Ottawa, which sits on the Ottawa River in the province of Ontario."
WATCH_S = 1.0
TRIALS = 3


class GpuUse:
    """Samples GPU utilisation every 100 ms while running."""

    def __init__(self) -> None:
        self.samples: list[tuple[float, int]] = []
        self._proc = subprocess.Popen(
            ["nvidia-smi", "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-lms", "100"],
            stdout=subprocess.PIPE, text=True,
        )
        self._thread = threading.Thread(target=self._read, daemon=True)
        self._thread.start()

    def _read(self) -> None:
        for line in self._proc.stdout:
            if line.strip().isdigit():
                self.samples.append((time.perf_counter(), int(line)))

    def mean(self, start: float, end: float) -> float | None:
        values = [u for t, u in self.samples if start <= t <= end]
        return statistics.mean(values) if values else None

    def close(self) -> None:
        self._proc.terminate()


async def chat_ttft(client: httpx.AsyncClient, model: str) -> float:
    body = {
        "model": model, "stream": True, "think": False, "keep_alive": KEEP_ALIVE,
        "messages": [{"role": "user", "content": SHORT_PROMPT}],
        "options": {"num_ctx": CONTEXT[model], "seed": SEED, "num_predict": 8, **SAMPLING[model]},
    }
    start = time.perf_counter()
    async with client.stream("POST", f"{OLLAMA}/api/chat", json=body) as r:
        async for line in r.aiter_lines():
            if line and json.loads(line).get("message", {}).get("content"):
                return (time.perf_counter() - start) * 1000
    return float("nan")


def long_chat(model: str) -> tuple[str, dict]:
    return f"{OLLAMA}/api/chat", {
        "model": model, "stream": True, "think": False, "keep_alive": KEEP_ALIVE,
        "messages": [{"role": "user", "content": LONG_PROMPT}],
        "options": {"num_ctx": CONTEXT[model], "seed": SEED, "num_predict": 3000, **SAMPLING[model]},
    }


def speech(model: str, text: str) -> tuple[str, dict]:
    return f"{OLLAMA}/v1/audio/speech", {
        "model": model, "input": text, "voice": "female-american-english", "response_format": "pcm",
    }


async def trial(client, gpu: GpuUse, request, model: str, cancel: bool, run_s: float) -> dict:
    url, body = request
    async def consume(r):
        async for _ in r.aiter_bytes():
            pass

    async with client.stream("POST", url, json=body) as r:
        reader = asyncio.create_task(consume(r))
        await asyncio.sleep(run_s)
        if cancel:
            reader.cancel()
            await r.aclose()  # the client hangs up, as Autum will on interruption
        mark = time.perf_counter()
        await asyncio.sleep(WATCH_S)
        util = gpu.mean(mark + 0.2, mark + WATCH_S)  # skip the first 200 ms of settling
        ttft = await chat_ttft(client, model)
        if not cancel:
            reader.cancel()
    return {"gpu_util_after_pct": util, "short_request_ttft_ms": ttft}


async def probe(client, gpu: GpuUse, model: str, kind: str, request, run_s: float) -> dict:
    idle_ttft = [await chat_ttft(client, model) for _ in range(TRIALS)]
    mark = time.perf_counter()
    await asyncio.sleep(WATCH_S)
    idle_util = gpu.mean(mark, mark + WATCH_S)
    cancelled = [await trial(client, gpu, request, model, True, run_s) for _ in range(TRIALS)]
    control = [await trial(client, gpu, request, model, False, run_s) for _ in range(TRIALS)]

    def med(rows, key):
        values = [r[key] for r in rows if r[key] is not None]
        return round(statistics.median(values), 1) if values else None

    result = {
        "model": model,
        "request": kind,
        "cancelled_after_s": run_s,
        "idle": {"gpu_util_pct": idle_util, "short_request_ttft_ms": round(statistics.median(idle_ttft), 1)},
        "after_cancel": {"gpu_util_pct": med(cancelled, "gpu_util_after_pct"), "short_request_ttft_ms": med(cancelled, "short_request_ttft_ms")},
        "still_running": {"gpu_util_pct": med(control, "gpu_util_after_pct"), "short_request_ttft_ms": med(control, "short_request_ttft_ms")},
        "trials": {"cancel": cancelled, "control": control},
    }
    # Stopped means both: the GPU goes quiet, and the model is free for the next request.
    after = result["after_cancel"]
    idle_ttft = result["idle"]["short_request_ttft_ms"]
    gpu_quiet = (after["gpu_util_pct"] or 0) < max(25.0, (result["still_running"]["gpu_util_pct"] or 0) / 3)
    model_free = (after["short_request_ttft_ms"] or 1e9) <= max(3 * idle_ttft, idle_ttft + 250)
    result["gpu_quiet_after_cancel"] = gpu_quiet
    result["model_free_after_cancel"] = model_free
    result["stops_on_cancel"] = gpu_quiet and model_free
    return result


CASES = [
    ("qwen3:8b", "chat", long_chat("qwen3:8b"), 1.0),
    ("lfm3-omni:2b", "chat", long_chat("lfm3-omni:2b"), 1.0),
    ("lfm3-omni:2b", "speech-passage", speech("lfm3-omni:2b", LONG_SPEECH), 1.0),
    ("lfm3-omni:2b", "speech-sentence", speech("lfm3-omni:2b", SENTENCE), 0.3),
]


async def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="run only cases whose request kind starts with this")
    args = parser.parse_args()
    gpu = GpuUse()
    results = []
    try:
        async with httpx.AsyncClient(timeout=120) as client:
            for model, kind, request, run_s in CASES:
                if args.only and not kind.startswith(args.only):
                    continue
                await chat_ttft(client, model)  # load
                results.append(await probe(client, gpu, model, kind, request, run_s))
                print(json.dumps({k: v for k, v in results[-1].items() if k != "trials"}, indent=1))
    finally:
        gpu.close()
    RESULTS.mkdir(parents=True, exist_ok=True)
    path = RESULTS / f"{dt.datetime.now():%Y%m%d-%H%M%S}-cancel-probe.json"
    path.write_text(json.dumps({"ollama": subprocess.run(["ollama", "--version"], capture_output=True, text=True).stdout.strip(), "results": results}, indent=1))
    print(f"wrote {path}")


if __name__ == "__main__":
    asyncio.run(main())
