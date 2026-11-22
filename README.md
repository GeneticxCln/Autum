# Autum

A local, voice-first AI assistant with a runtime-driven visual presence,
verified local capabilities, user-requested perception, and user-controlled
persistent memory. See [the development plan](Autum-Development-Plan.md),
[the runtime contract](docs/m0-runtime-contract.md),
[ADR 0001](docs/adr/0001-runtime-language-and-event-transport.md),
[ADR 0002](docs/adr/0002-voice-pipeline-and-resource-budget.md), and the
[acceptance matrix](docs/acceptance-matrix.md).

## Run Autum

Requires local Ollama with `lfm3-omni:2b` (`127.0.0.1:11434`), PipeWire
(`pw-record`, `pw-play`, `wpctl`), PySide6 (`qt6-declarative`), and on Wayland
`layer-shell-qt` (X11 fallback included).

```bash
# Push-to-talk (default)
PYTHONPATH=src python3 -m autum

# Hands-free conversation (streaming Silero VAD v6 + voice barge-in)
PYTHONPATH=src python3 -m autum --hands-free          # with headphones
PYTHONPATH=src python3 -m autum --hands-free --aec    # with speakers (PipeWire WebRTC AEC)
```

Autum runs in the background. Her visual presence is a 300 px 3D stippled face
suspended inside a frosted stardust orb in the bottom-right corner that follows
you across windows and virtual desktops via Wayland `wlr-layer-shell`:

- **Idle:** frosted silver-ice, slow stardust orbit and gentle 3D breathing sway
- **Listening / speech detected:** cool ice-cyan, gathered inward, attentive gaze, warm open-mic edge aura
- **Thinking / processing:** silver-violet swirling stardust bands with subtle crown wave
- **Speaking:** luminous silver-blue with audio-reactive jaw articulation and outward stardust burst
- **Waiting for approval:** warm gold tint
- **Observing (image / document / window / screen):** cool cyan-teal outer halo and attentive eye sheen
- **Muted:** grey with a lower-right gap (controlled only by your OS mic mute; Autum never unmutes it)

In push-to-talk mode, hold the global shortcut (KDE GlobalShortcuts portal,
suggested `Meta+Z`) or hold the cloud, speak, and release. Holding it while
Autum speaks interrupts her immediately.

## Capabilities & Privacy

- **Local tools:** clock (`get_time`), countdown timers (`set_timer`,
  `list_timers`, `cancel_timer`), local notes (`create_note` — never overwrites,
  `list_notes`, `read_note`, `delete_note`).
- **Perception (user-requested only):** `inspect_image` (local `lfm3-omni:2b`
  vision) and `read_document` (text/markdown/code/PDF). Secret stores
  (`~/.ssh`, `~/.gnupg`, `.env`, `*.kdbx`, `*.pem`, etc.) are blocked and
  outputs are tagged `observed` (untrusted data, never commands).
- **Persistent memory (user-selected only):** `remember_fact`,
  `recall_memories`, and `forget_memory` stored in
  `~/.local/share/autum/memory.json` with provenance tracking and secret
  exclusion. Conversations are not saved by default, and raw WAV audio is
  released after every turn.
- **Replaceable Linux desktop integration (`linux_desktop`):**
  - **Info & files:** `system_info`, scoped `list_directory` (`allowed_read_dirs`).
  - **Eyes (opt-in live screen vision):** `start_screen_watch` (one-time KDE ScreenCast share picker; keeps a single downscaled JPEG in RAM while the cloud's cyan-teal halo stays lit), `look_at_screen` (instant local vision on the latest live frame), `stop_screen_watch`, and one-shot `inspect_selected_window`.
  - **Hands (KDE KWin window control & apps):** `open_application`, `list_windows`, `focus_window`, `move_window` (snap/tile left/right/top/bottom/center/corners, coordinates, or move to another virtual desktop), `window_state` (minimize/maximize/restore), and `close_window` *(irreversible — requires approval)*. Password managers and wallets are excluded.
- **Approvals & audit:** irreversible actions (`delete_note`, `forget_memory`,
  `close_window`) always require explicit user approval (single-use, timed,
  anti-spoofed). Actions are recorded in a redacted audit log at
  `~/.local/share/autum/audit.jsonl`.

## Configuration

Optional TOML file at `~/.config/autum/config.toml` (or `--config PATH`):

```toml
mode = "push_to_talk"               # or "hands_free"
aec = false
silence_ms = 700
barge_in_ms = 256
max_history = 20
max_context_chars = 10000
allowed_read_dirs = ["~/Documents"] # required for list_directory; optional scope for perception
enabled_integrations = ["linux_desktop"]
require_reversible_approval = false
approval_timeout_s = 20.0
```

## Optional autostart (systemd user service)

A template unit is provided in [contrib/autum.service](contrib/autum.service):

```bash
mkdir -p ~/.config/systemd/user
cp contrib/autum.service ~/.config/systemd/user/autum.service
systemctl --user daemon-reload
systemctl --user enable --now autum.service
```

## Tests

```bash
python3 -m pytest -q
```

## M1 Benchmark Tools (Optional Re-measurement)

See [docs/m1-benchmark-methodology.md](docs/m1-benchmark-methodology.md):

```bash
python3 -m bench.record_corpus
python3 -m bench.attended
```
