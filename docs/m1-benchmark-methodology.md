# M1 — Voice-pipeline benchmark methodology

*Status: approved 2026-11-22. Changes made since approval are listed in
§11 for review.*

This document describes how M1 measures the candidate pipelines listed
in WP1 of the [development plan](../Autum-Development-Plan.md).

## 1. What M1 decides

- Which pipeline Autum builds on in M2 (recorded in ADR 0002).
- The measured resource budget on the reference machine.
- Proposed acceptance thresholds for the §2.1 objectives, based on
  measured results.

## 2. Test sets

### 2.1 Prompts

`bench/corpus.json` holds 58 prompts:

| Category | Count | Purpose |
|---|---|---|
| short | 8 | Quick requests |
| long | 8 | Multi-clause requests |
| pause | 8 | A pause inside the request (marked `\|`), for premature end-of-turn |
| correction | 8 | Self-corrections ("at three, no wait, at four") |
| followup | 8 | Questions that only make sense with the previous turn (given as history) |
| noise | 8 | Requests under background noise |
| interruption | 4 | Phrases spoken over Autum's voice (attended session only) |
| backchannel | 4 | "Mm-hm", "yeah", a cough over Autum's voice (attended session only) |
| echo | 2 | Silence while Autum's voice plays through the speakers (attended session only) |

Some prompts list `expect_any`, words a correct, context-aware reply
should contain. This is a rough automatic check; the reply text is
reviewed by a person.

### 2.2 Recorded set — primary

All recognition, end-of-turn, and latency numbers used for the M1
decision and the proposed thresholds come from this set.

Recording setup, kept identical for every clip:

- **Speaker:** the user. It is a single-speaker set, so the results
  describe Autum's actual user. A second speaker would be a separate set.
- **Microphone:** the reference microphone, the Blue Microphones USB
  microphone (PipeWire source "Blue Microphones Analog Stereo"), the
  default source on 2026-11-22.
- **Position and level:** the same position (about 20–30 cm, facing the
  microphone) and the same input level for the whole set. The source
  volume is noted at the start of the session (57% on 2026-11-22) and
  not changed.
- **Processing:** PipeWire noise suppression and echo cancellation are
  off for these clips.
- **Room:** the same quiet room, in one session if possible. If more
  than one session is needed, each clip's session is recorded.
- **Format:** captured at the device's native 48 kHz from both channels,
  averaged to mono, and kept as the master copy (`masters/<id>.wav`).
  The benchmark copy (`<id>.wav`) is resampled to 16 kHz with the same
  resampler used for the synthetic set (SciPy `resample_poly`).

Recording tool: `python3 -m bench.record_corpus`, run by the speaker in
a terminal. At the start of each sitting it shows the microphone and
input level, warns if the source is a noise-suppression or echo-cancel
filter or differs from earlier sittings, and measures the room's noise
floor from 2 seconds of silence. Accepted clips are saved immediately,
so the set can be recorded over several sittings.

Per prompt:

1. The tool shows the prompt (and, for follow-ups, the earlier turn,
   which is not read aloud) and starts recording when the speaker
   presses Enter.
2. The speaker waits at least half a second, reads the prompt naturally,
   and pauses about one second at each `|`, as they naturally would.
3. Recording stops after 2.5 seconds of quiet following speech (25
   seconds at most). This only controls the recorder; labels are
   computed separately.
4. The tool reports clipping, very quiet speech, speaking too soon, and
   too little trailing silence. The speaker can listen, accept, or
   re-record. A misread prompt is recorded again.

Labels (true start and end of speech, and the segments around pauses):

- Found automatically from the energy envelope: 10 ms frames, active
  when within 40 dB of the clip's peak.
- Cross-checked against Silero VAD (first and last 32 ms frame scoring at
  least 0.5). Where the two disagree by more than 100 ms, the clip is
  flagged and cannot be accepted until the speaker listens and chooses
  the energy label, the VAD label, or types the times by hand.
- Segments around pauses come from the VAD (gaps of 250 ms or more).
- Each label records which method produced it, both candidate labels,
  whether it was flagged, the peak level, and the sitting.

Noise prompts are recorded clean. The 16 kHz benchmark copy is then
mixed digitally with the same pink noise that `synthetic-v2` uses for
that prompt (seeded from the prompt ID) at 10 dB signal-to-noise ratio,
measured over the speech. This keeps the noise
condition identical across pipelines and reruns. Real acoustic
interference is covered by the attended tests (§6).

Storage: `bench/data/recorded-v1/`, local only. It is excluded from git and
never uploaded, and it can be deleted at any time. Committed result files
contain the transcripts of these clips, which are the corpus prompts.

Versioning: once recorded, the set is fixed as `recorded-v1`.
Re-recording any clip creates `recorded-v2`, and results from different
versions are not compared.

### 2.3 Synthetic set — supplementary

The synthetic set is used to check that the harness works, to compare
replies and speech output on identical input, and to catch regressions.
It is never used for recognition accuracy, end-of-turn tuning, or
acceptance thresholds, because generated speech is cleaner and more
regular than a real voice through a real microphone.

There are two versions, both generated by `bench/make_synthetic.py` on
2026-11-22 and kept in `bench/data/synthetic-v1/` and
`bench/data/synthetic-v2/` (local only, excluded from git).
`synthetic-v2` is the supplementary set from now on; `synthetic-v1` is
kept for comparison.

How they were generated:

| Step | Detail |
|---|---|
| Prompts | The 48 prompts not marked attended-only |
| Voice | Chatterbox-Turbo, revision `749d1c1a…`, its built-in default voice (`conds.pt`); one voice for every clip |
| Generation settings | Package defaults: temperature 0.8, top-p 0.95, top-k 1000, repetition penalty 1.2, loudness normalisation on |
| Seeds | v1: none. v2: each part seeded with the first 4 bytes of SHA-256(`7:<prompt id>:<part index>`), set in PyTorch before generating it, so a clip does not depend on generation order |
| Pauses | Each `\|`-separated part was generated separately, then joined with 0.9 s of silence |
| Trimming | Each part was trimmed to its active span (10 ms frames within 40 dB of its peak) |
| Padding | 0.4 s of silence before the speech, 1.6 s after |
| Level | Peak-normalised to 0.5 full scale |
| Resampling | 24 kHz output resampled to 16 kHz with SciPy `resample_poly` |
| Noise | The 8 noise clips were mixed with pink noise at 10 dB SNR over the speech span. v1: one NumPy generator seeded 7, shared in corpus order. v2: a generator per prompt, seeded with SHA-256(`7:<prompt id>:-1`), the same noise the recorded set uses. |
| Labels | Speech start, end, and segment boundaries taken from the construction itself (`labels.json`) |
| Format | 16 kHz, mono, 16-bit WAV |

Identity (set hash = SHA-256 of the sorted per-file `sha256  filename`
lines). Each version has 48 clips, 4.0 minutes in total, of which 2.4
minutes are speech:

| Version | Set SHA-256 | `labels.json` SHA-256 |
|---|---|---|
| synthetic-v1 | `22e5f102c24fd5536699a0110af5e04c29bc90918416723e2358eedd2037283b` | `ddc65da0a04c766831fb524d13e745b7e10ef4e057fb3761797e56e496933a00` |
| synthetic-v2 | `b0c4c34e98f24db1a97d49cd18efafa186d2ce4007c9e6b06851b49f3603deb9` | `52043cc8d9fb9638f4548a9af5e56d825648068e80c64763edbf6337f6a879c1` |

Known limitations:

- **Reproducibility.** `synthetic-v1` cannot be regenerated exactly.
  `synthetic-v2` can on this machine and software: the same clip
  generated in two separate processes was byte-identical. Other GPUs,
  drivers, or library versions may still give slightly different audio.
- **One voice**, clean, with regular timing.
- **Watermarked.** Every clip carries Chatterbox's inaudible Perth
  watermark.
- **Overlap with a candidate.** Chatterbox-Turbo is also the speech
  output candidate in pipeline B. That does not affect recognition
  results, which come only from the recorded set. Listening tests compare
  output voices on the same reply text, not on these clips.

## 3. Pipelines and candidates

See WP1 in the development plan for the candidate list (exact versions
and licences) and pipelines A–E.

Platform for every run: AMD Ryzen 7 5800XT, 32 GB RAM, RTX 4060 Ti
16 GB, NVIDIA driver 620.73.04, CachyOS (Linux 7.3.5), Python 3.14.10,
PipeWire 1.6.11 at 48 kHz with a 1024-sample quantum.

Harness-only libraries, which are not candidates: NumPy 2.5.3, SciPy
1.18.1, httpx 0.28.1, pytest 9.1.1. The harness reuses Autum's own
sentence chunker and text clean-up (`src/autum/text.py`), so the
benchmark speaks replies the way Autum will.

## 4. Procedure per utterance

`bench/run_pipelines.py` does the following for each clip and each
pipeline:

1. **End-of-turn.** Silero VAD scores each 32 ms frame. Speech starts
   after 96 ms of frames scoring at least 0.5. The turn ends when frames
   score below 0.35 for the silence threshold. The thresholds 300, 500,
   700, and 1000 ms are swept for every clip; pipeline runs use 500 ms.
2. **What the pipeline hears.** Only the audio up to the end-of-turn
   decision, as in live use. A premature end therefore truncates the
   request.
3. **Recognition** (A, B, C, E) runs on that audio after the
   end-of-turn decision. Streaming recognition during speech is a later
   optimisation and is not measured in M1.
4. **Reply.** The language model streams its reply with the shared
   system prompt and the prompt's history. Autum's chunker cuts it into
   speakable pieces.
5. **Speech output** synthesises the pieces in order while the model is
   still generating. The stages compete for the GPU as they will in use.
6. The first two clips of each pipeline are a warm-up and are not
   counted. The first one's time, including model loading, is reported
   as cold start.

Settings fixed for every run:

| Setting | Value |
|---|---|
| System prompt | `SYSTEM_PROMPT` in `bench/stages.py`: Autum persona, one to three short spoken sentences, no markdown |
| Ollama context | Qwen3 8B: 8,192. LFM3-Omni: 4,096, the size its speech endpoint loads with (see §11) |
| Ollama sampling | Each model's shipped defaults, pinned, plus seed 42. Qwen3 8B: temperature 0.6, top-p 0.95, top-k 20. LFM3-Omni: temperature 0. `think: false`; models kept loaded for 30 minutes |
| Whisper | Language `en`, beam size 5, float16, no VAD filter, no conditioning on previous text |
| LFM3-Omni voice | `female-american-english`, 24 kHz PCM, streamed |
| Chatterbox-Turbo | Default voice and settings, not streamed; each piece is fully generated before playback |
| Models between pipelines | All Ollama models unloaded before each pipeline starts |

## 5. Metrics

| Metric | Definition |
|---|---|
| End-of-turn delay | End-of-turn decision minus the labelled true end of speech |
| Premature end-of-turn | Decision more than 50 ms before the true end; reported per category |
| Missed end-of-turn | No decision before the clip ends |
| First-audio latency | End-of-turn delay + time from the decision to the first synthesised audio + device output latency |
| Device output latency | Estimated as two PipeWire quanta (42.7 ms) until measured by loopback (§6) |
| Word error rate | Word edits divided by reference words, after normalising both: lower case, apostrophes removed, hyphens split, digits spelled out, fillers (um, uh, er, hmm) and the word "and" removed |
| Language model timing | Time to first token and to the first speakable piece |
| Speech output | Time to first audio, real-time factor (generation time ÷ audio length), and playback gaps: moments when the next piece is not ready when the previous one ends (more than 50 ms counts as a gap) |
| Context check | Share of prompts with `expect_any` whose reply contains one of the words |
| GPU memory | Before each pipeline: all Ollama models unloaded (confirmed), then a baseline once readings settle. During it: peak sampled every 250 ms, and per-model figures from Ollama |
| Summaries | Median, 95th percentile, and maximum |

Spoken quality is judged by listening. For each speech output candidate
(LFM3-Omni voice, Chatterbox-Turbo), the same 10 replies and one
continuous 2-minute passage are rendered, with filenames shuffled so the
listener does not know which is which. The user rates intelligibility,
naturalness, and comfort for long listening from 1 to 5.

## 6. Attended tests

These need the user present, because audio plays aloud through the
speakers. Tool: `python3 -m bench.attended`, about 8 minutes.

| Test | Method |
|---|---|
| Output latency | Loopback: six clicks play through the default speakers and are recorded by the microphone in one duplex stream, three times. The delay is found by cross-correlation and accepted only if the recording clearly matches the clicks (normalised correlation ≥ 0.1). The round trip includes output, air, and input. Output latency is estimated as round trip minus the input latency the audio system reports, and replaces the 42.7 ms estimate. |
| Echo, self-interruption | 60 s of Autum's voice plays while the user stays silent. Every speech detection counts as Autum hearing itself. |
| Interruption | The same 60 s plays. At 6, 20, 34, and 48 s the screen tells the user to say one of the interruption prompts. Scored per prompt: detected or not within 6 s, and the delay from the user's speech onset to detection. |
| Backchannels | As for interruption, with the backchannel prompts. A detection within 6 s of a prompt is a false interruption. |

Common to the three voice tests:

- **Playback:** one fixed 60 s file, built once from the saved omni-voice
  replies of the synthetic-v2 run. It is scaled to −20 dBFS RMS but never
  let past −1 dBFS peak; the result is −21.2 dBFS RMS, SHA-256
  `9cdd3f6a1e20206f…`. The playback carries on while the user speaks.
  These tests measure detection, not the runtime's reaction.
- **Conditions:** each test runs with echo cancellation off, then on.
  "On" is PipeWire's WebRTC echo canceller with PipeWire's default
  settings, loaded with `pactl load-module module-echo-cancel` only after
  the user confirms. It is always unloaded afterwards, and a leftover
  copy from a killed run is removed at the next start.
- **Detector:** the barge-in detector that M3 would use. Silero VAD fires
  after a minimum stretch of speech (frames ≥ 0.5) and re-arms after
  300 ms of silence. The minimum is swept at 96, 200, and 300 ms: longer
  minimums ignore more short sounds but detect real interruptions later.
- **Onset:** the user's speech onset is the first moment the level rises
  10 dB above the second before the prompt, held for 30 ms. With echo
  cancellation off, Autum's voice can mask the onset; then no delay is
  reported for that prompt.
- **Consistency:** speaker volume, microphone level, and devices are
  recorded, and the tool warns if the volume changes during a session.
- **Storage:** recordings in `bench/data/attended-v1/<time>/` (local
  only); numbers in `bench/results/<time>-attended.json`.

The results decide between echo cancellation and a headphone requirement
for hands-free mode (§3.1), and the minimum speech length for barge-in.

## 7. Unattended tests

| Test | Method |
|---|---|
| Cancellation | For each Ollama model, start a long reply, close the request after 1 s, and immediately send a short request. Compare its first-token time with an idle baseline, and watch GPU use, to see whether generation actually stopped. Chatterbox-Turbo cannot stop mid-piece, so its longest piece time is the worst-case stop delay. |
| Long session | Replay the recorded set through the leading pipeline(s) for 2 hours. Track process memory, GPU memory, and latency drift. The 8-hour objective is verified at M5. |
| Resource budget | Per-pipeline peak GPU memory above the desktop baseline (about 2 GB on 2026-11-22), plus a reserve for the visual presence, proposed at 1.5 GB |

## 8. Open items

| Item | Status |
|---|---|
| Recording and labelling tool (§2.2) | Done: `bench/record_corpus.py`, unit-tested without a microphone. The recording itself needs the user. |
| Seeded synthetic set | Done: `synthetic-v2` (§2.3) |
| Repeatable model replies | Done: sampling pinned with seed 42 (§4) |
| Cancellation probe (§7) | Done: `bench/cancel_probe.py` |
| Attended-test tooling (§6) | Done: `bench/attended.py`. Its analysis is unit-tested without audio. Playing, recording, and loading echo cancellation are exercised only when the user runs it. |

## 9. Outputs

- `bench/results/<timestamp>-<set>.json` (committed): every number,
  transcript, and reply, plus the exact environment.
- Spoken replies for listening, in `bench/data/outputs/` (local only).
- ADR 0002: the pipeline decision, with alternatives and trade-offs.

Decision procedure:

1. Hard gates: runs offline; fits in GPU memory with the visual
   presence reserve; licence allows Autum's intended use.
2. Pipelines passing the gates are ranked on the recorded set by 95th
   percentile first-audio latency, word error rate, premature
   end-of-turn rate, the listening ratings, and the context checks.
3. Proposed thresholds come from the measured results of the chosen
   pipeline.

## 10. Current state

- The harness (`bench/`) works end to end on all five pipelines. Its
  metric and recording logic is unit-tested (`tests/test_bench.py`,
  `tests/test_record_corpus.py`).
- Supplementary results on `synthetic-v2` and the cancellation probe are
  in [m1-results.md](m1-results.md).
- The recorded set does not exist yet. Next step: the user runs
  `python3 -m bench.record_corpus`, 48 prompts, about 15–20 minutes.
- The attended session (§6) needs the user: `python3 -m bench.attended`.

## 11. Changes since approval

Found while building and checking the harness, for review:

1. **LFM3-Omni context 4,096 instead of 8,192.** Ollama's speech endpoint
   loads LFM3-Omni with a 4,096 context. A chat request with any other
   size makes Ollama reload the model, which took about 6.2 s each time,
   and the next speech request reloads it again. With 8,192, every turn
   in pipelines C, D, and E would have included two reloads. That would
   have measured my setting, not the candidates. Measured on 2026-11-22:
   chat right after speech took 6,154 ms to its first token at 8,192, and
   36 ms at 4,096. Qwen3 stays at 8,192. Consequence for M2: if LFM3-Omni
   is used for both chat and speech, its conversation context is limited
   to 4,096 tokens.
2. **Two warm-up clips instead of one.** After the first warm-up, Qwen3's
   first token still took 2.7 s once, against about 100 ms afterwards.
3. **GPU baseline per pipeline,** taken after Ollama confirms all models
   are unloaded and readings settle. A single baseline at the start had
   caught models still unloading (8,180 MiB instead of about 2,000 MiB).
4. **Recording stops after 2.5 s of quiet** (the approved text said
   about two seconds), and the mono master is the average of both
   channels.
5. **Two cancellation-probe files deleted.** They were made with the
   mismatched context size from item 1, so their speech results were
   wrong.
6. **Attended-test details added** in §6: the cue schedule, the 6 s
   window, the onset rule, the minimum-speech sweep (96, 200, 300 ms),
   the loopback acceptance rule, and the output-latency estimate (round
   trip minus reported input latency).
