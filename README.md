# Autum

A local, voice-first assistant. See [the development plan](Autum-Development-Plan.md).

## Talk to Autum (M2, provisional)

Needs Ollama running with `lfm3-omni:2b`, and PipeWire. From this folder:

```bash
PYTHONPATH=src python3 -m autum
```

| Key | Does |
|---|---|
| Space | Start talking; Space again when you have finished. Pressing it while Autum speaks interrupts it. |
| s | Stop Autum |
| t | Type a message instead |
| q | Quit |

Everything runs locally and nothing is saved to disk. Mute the microphone with your desktop's mute control; Autum shows it and never unmutes it.

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
