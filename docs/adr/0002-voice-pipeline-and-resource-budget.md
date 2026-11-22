# ADR 0002 — Voice pipeline, resource budget, and acceptance thresholds

*Status: accepted by user decision on 2026-11-22 (with open real-voice WER and
hands-free speaker echo checks noted below). Reference machine: AMD Ryzen 7
5800XT, 32 GB RAM, NVIDIA GeForce RTX 4060 Ti 16 GB, PipeWire 1.6.11, Ollama
0.40.16.*

## Context

WP1 compared five local voice pipelines (A–E) on the reference machine
([docs/m1-results.md](../m1-results.md)) across latency, cancellation, memory,
and spoken continuity. Autum also needs local tool calling (WP5/WP6) and local
vision for user-supplied images and windows (WP7/WP10) within the 16 GB VRAM
budget.

## Measured evidence

From `bench/results/20261122-165225-synthetic-v2.json` (48 prompts, all 5
pipelines), `bench/results/20261122-170415-cancel-probe.json`, and live
measurements on the reference display:

| Pipeline | Processing after end-of-turn (median) | First audio incl. 500 ms VAD wait (median / p95) | GPU memory above desktop | Cold start | Stops on cancel | Native vision & tools |
|---|---|---|---|---|---|---|
| A: Whisper base → Qwen3 8B → omni voice | 438 ms | 1,023 / 1,565 ms | 10.2 GB | 5.2 s | Yes (`27 ms`) | Tools only (no vision) |
| B: Whisper base → Qwen3 8B → Chatterbox | 1,352 ms | 1,870 / 3,132 ms | 9.8 GB | 13.8 s | No (mid-sentence) | Tools only (no vision) |
| C: omni transcribe → omni chat → omni voice | 245 ms | 829 / 1,162 ms | 3.5 GB | 2.1 s | Yes (`20 ms`) | Yes |
| **D: omni hears audio → omni voice** | **172 ms** | **757 / 982 ms** | **3.4 GB** | **2.2 s** | **Yes (`20–36 ms`)** | **Yes** |
| E: Whisper base → omni chat → omni voice | 253 ms | 829 / 1,062 ms | 3.9 GB | 2.1 s | Yes (`20 ms`) | Yes |

Additional measured components on the reference machine:

- **Streaming Silero VAD v6** (`silero_vad_v6.onnx`, CPU): `0.08 ms` per 32 ms frame (`~0.25%` of one CPU core).
- **Visual presence** (native Wayland `wlr-layer-shell`, 300 px cloud): `6.0 ms` median / `71.2 ms` max event-to-screen lag (`n=20`), `14.0%` of one CPU core, `~302 MiB` RSS.
- **Microphone preprocessing:** the reference Blue Microphones USB source had `-32.8 dBFS` sub-sonic rumble below 20 Hz (`99.1%` of idle energy). Adding an 80 Hz high-pass filter (`highpass_pcm`) resolved quiet-clip misrecognition in live push-to-talk conversations.

## Decision

Confirm **Pipeline D** (`lfm3-omni:2b`, Ollama ID `4c0ee9400634`, Q4_K_M,
4,096 context across chat, speech, vision, and background transcription) paired
with:

- PipeWire (`pw-record` / `pw-play` / `wpctl`) with an 80 Hz high-pass input
  filter (`src/autum/audio.py`),
- Streaming Silero VAD v6 (`src/autum/vad.py`) and optional PipeWire WebRTC
  echo cancellation (`--aec`) for hands-free mode,
- `lfm3-omni:2b`'s native OpenAI-compatible `tool_calls` and `image_url`
  vision endpoints (both verified live end-to-end).

Because all subsystem interfaces (`Inference`, `SpeechOutput`, `Transcriber`,
`VisionAnalyzer`) remain decoupled behind `src/autum/runtime.py` and
`src/autum/tools.py`, any future larger multimodal model can replace
`lfm3-omni:2b` without changing the runtime, event contract, or visual presence.

## Resource budget (reference 16 GB RTX 4060 Ti)

| Subsystem | Measured memory | Notes |
|---|---|---|
| Desktop + compositor | ~2.0 GB VRAM | Reference KDE Plasma Wayland desktop |
| `lfm3-omni:2b` (audio + chat + speech + vision) | ~3.4 GB VRAM | Fixed 4,096 context across all endpoints (prevents 6.2 s reloads) |
| Visual presence (`autum.presence`) | ~302 MiB RAM, ~14% of 1 CPU core | Shared GPU surface; no separate model weights |
| Silero VAD v6 (hands-free only) | CPU-only (`0.08 ms`/frame) | Loaded lazily only when hands-free mode is active |
| **Headroom remaining** | **~10+ GB VRAM** | Leaves wide margin for future larger models or M11 |

## Proposed acceptance thresholds

| Metric | Threshold | Current measured baseline |
|---|---|---|
| Push-to-talk processing → first audible reply | ≤ 500 ms median, ≤ 1.0 s p95 | `172 ms` processing median (`synthetic-v2`) |
| Hands-free end-of-speech → first audible reply | ≤ 900 ms median, ≤ 1.5 s p95 | `757 ms` median / `982 ms` p95 at 500 ms silence (`synthetic-v2`) |
| Cancellation (interrupt → audio & generation stop) | ≤ 200 ms | `< 40 ms` Ollama stop + immediate `pw-play` kill |
| Visual state lag (event → screen frame) | ≤ 100 ms | `6.0 ms` median / `71.2 ms` max (Wayland `layer-shell`) |
| Memory growth over continuous session | Bounded | Bounded by `max_history`, `max_context_chars`, immediate WAV release |

## Trade-offs and open checks

- **Reasoning ceiling:** `lfm3-omni:2b` is a 1.7B model; complex multi-step
  planning is weaker than an 8B+ model, traded for sub-200 ms latency, 3.4 GB
  VRAM, and unified audio+vision+tools.
- **Deferred real-voice measurements:** the user confirmed live push-to-talk
  recognition is working well and chose not to record the 48-clip `recorded-v1`
  corpus or run the 8-minute `bench.attended` speaker test right now. Formal
  real-voice WER, hands-free speaker AEC vs. headphone requirement, and an
  8-hour continuous live soak remain open verification items if hands-free
  speaker use or a formal external release audit is pursued later.
