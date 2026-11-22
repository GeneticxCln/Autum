# Autum

A local, voice-first assistant. See [the development plan](Autum-Development-Plan.md).

## Run Autum (M2, provisional)

Needs Ollama running with `lfm3-omni:2b`, and PipeWire. From this folder:

```bash
PYTHONPATH=src python3 -m autum
```

Autum runs in the background. Her only visible part is a particle cloud in the bottom-right corner of the screen. It drifts slowly when idle, gathers tight while listening, swirls while thinking, and bursts outward with her voice. A warm glow on its edge means the microphone is open; grey with a gap at the lower right means it is muted.

To talk, hold the talk key anywhere (KDE asks you to confirm it the first time; suggested Meta+Z) or hold the cloud. Speak, then let go. Holding it while Autum speaks interrupts her. Mute the microphone with your desktop's mute control; Autum never unmutes it. Stop Autum with Ctrl+C or by ending the process.

Everything runs locally and nothing is saved to disk.

## Tests

```bash
python3 -m pytest -q
```

## M1 benchmark

See [docs/m1-benchmark-methodology.md](docs/m1-benchmark-methodology.md). The steps that need you:

```bash
python3 -m bench.record_corpus
```

```bash
python3 -m bench.attended
```
