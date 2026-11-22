# ADR 0001 — Runtime language and event transport

*Status: accepted for M0–M2; revisit if M1 or M3 measurements show the
runtime adds meaningful latency. Date: 2026-11-22.*

## Context

The plan leaves the language open (plan §1.4). The runtime coordinates
audio, inference, and the visual presence, and owns cancellation. Its
own work is light: routing events and starting and cancelling tasks at
human timescales. The heavy work is model inference, which runs in
native code (CUDA, CTranslate2, llama.cpp via Ollama) whatever language
calls it.

Every candidate speech component on the reference machine is Python or
an HTTP service: faster-whisper, Chatterbox, Silero VAD, openWakeWord,
and Ollama.

## Options

| Option | For | Against |
|---|---|---|
| **Python 3 (asyncio) for runtime and components** | One language; direct use of every speech library; asyncio task cancellation maps onto plan §4.2 | GIL; must keep blocking model calls off the event loop |
| Rust runtime + Python model workers | Strong typing and predictable latency in the core | Two languages, an IPC layer between runtime and every model, cross-language schema sync; more to build before the first spoken turn |
| Node/TypeScript runtime + Python workers | Natural fit for a web-based presence UI | Same two-language cost; no speech libraries |

## Decision

Python 3.12+ with asyncio for the core runtime and the voice engine.
The core uses the standard library only. Blocking model calls run in
threads or behind HTTP (Ollama), never on the event loop.

Events cross process boundaries as JSON Lines over a Unix domain socket
under `$XDG_RUNTIME_DIR/autum/` (owner-only permissions). JSON keeps the
visual presence free to use any language.

## Consequences

- The runtime's latency share is measured in every M1–M3 run. The M0
  tests already bound the synchronous cancellation path (speech start →
  playback stopped) to under 50 ms with mock components.
- Model code must not block the event loop; long calls go to a thread
  or a subprocess.
- If measurements show the runtime is a bottleneck, the event contract
  and the socket transport allow replacing the core without changing
  components.
