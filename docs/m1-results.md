# M1 — Results so far

*2026-11-22. Supplementary evidence only. The M1 decision waits for the
recorded set (methodology §2.2).*

## 1. Harness check on synthetic-v2

Run `bench/results/20261122-165225-synthetic-v2.json`: 48 clips, all
five pipelines, end-of-turn at 500 ms of silence, two warm-up clips per
pipeline. These are Chatterbox-generated clips, so the recognition and
end-of-turn figures say little about real speech (methodology §2.3).
They show that the harness works and how the pipelines compare on
identical input.

| Pipeline | First audio, median | 95th pct. | Max | Processing after end-of-turn, median | GPU memory above desktop | Cold start |
|---|---|---|---|---|---|---|
| A: Whisper → Qwen3 → omni voice | 1,023 ms | 1,565 ms | 2,888 ms | 438 ms | 10.2 GB | 5.2 s |
| B: Whisper → Qwen3 → Chatterbox | 1,870 ms | 3,132 ms | 3,399 ms | 1,352 ms | 9.8 GB | 13.8 s |
| C: omni transcribe → omni chat → omni voice | 829 ms | 1,162 ms | 1,260 ms | 245 ms | 3.5 GB | 2.1 s |
| D: omni hears audio → omni voice | 757 ms | 982 ms | 1,033 ms | 172 ms | 3.4 GB | 2.2 s |
| E: Whisper → omni chat → omni voice | 829 ms | 1,062 ms | 1,192 ms | 253 ms | 3.9 GB | 2.1 s |

First audio includes waiting for end-of-turn (537 ms median at the
500 ms setting) and the 42.7 ms output estimate. The desktop itself used
about 2 GB of GPU memory.

What it shows:

1. **The harness works.** No failures in 240 utterance runs. Pinned
   sampling works: A and B share recognition and language model, and
   gave identical replies on 48 of 48 clips.
2. **Waiting for end-of-turn is most of the latency.** For C, D, and E,
   about 540 of roughly 760–830 ms is the silence wait. The threshold
   must be chosen on the recorded set. If real pauses force a long
   threshold, a semantic end-of-turn model may be needed. That would be
   a new candidate needing a download and your approval.
3. **Speech output.** The omni voice streams, and starts within about
   50 ms of receiving a sentence (real-time factor 0.15). Chatterbox-Turbo
   generates each sentence whole (real-time factor 0.34). That adds
   about 0.9 s to the median first audio and caused playback gaps in 2
   of 48 replies. A had 1 gap; C, D, and E had none. Spoken quality
   still needs your listening ratings: the replies are saved in
   `bench/data/outputs/20261122-165225/`.
4. **GPU memory.** Pipelines with Qwen3 need about 10 GB above the
   desktop. Omni-only pipelines need about 3.5 GB. All fit in 16 GB with
   the desktop's 2 GB and a 1.5 GB presence reserve.
5. **Reply honesty.** Asked "What time is it?", the omni model invented
   a time ("It's 2:30 PM.", "It's ten o'clock."). Qwen3 said it cannot
   check. This matters for WP5, where the agent must report only what it
   knows.
6. **Context checks** (rough keyword check, 20 prompts): C 16, D 15,
   A 14, B 14, E 13. Several misses are the pause prompts that
   end-of-turn cut short, so the model never heard the full question.
7. **Synthetic end-of-turn.** Every pause prompt was cut short at 300,
   500, and 700 ms, and none at 1,000 ms. That follows from the
   generated pauses all being 0.9 s long. Real pauses vary, so this
   says nothing about the right threshold.
8. **Synthetic recognition.** Word error rate was about 13%, almost all
   from pause prompts cut short (56% on those). Other categories were
   0–10%. This is not evidence about real speech.
9. **Unexplained transient.** In A, two consecutive clips (pause-06 and
   pause-07) waited about 2.5 s for Qwen3's first word, against about
   100 ms otherwise. Repeating one of them alone gave 80 ms with no
   hidden reasoning. The cause is not known; watch for it in the
   recorded-set runs.

## 2. Cancellation

Run `bench/results/20261122-170415-cancel-probe.json`, 3 trials per case,
median shown. "Next request" is a short chat request sent right after.

| Model | Request | Next request, idle server | After cancelling | Without cancelling | Stops on cancel |
|---|---|---|---|---|---|
| Qwen3 8B | Long chat reply | 41 ms | 27 ms | 53,976 ms | Yes |
| LFM3-Omni | Long chat reply | 21 ms | 20 ms | 16,214 ms | Yes |
| LFM3-Omni | Long speech passage | 20 ms | 36 ms | 16,308 ms | Yes |
| LFM3-Omni | One spoken sentence | 19 ms | 39 ms | 32 ms | Yes |

- Closing the request stops generation in every case. GPU use returns to
  idle, and the model is free again within 40 ms.
- Without cancellation, Ollama makes the next request to the same model
  wait until the old one finishes, up to 54 s here. Cancelling on
  interruption is required, not optional (plan §4.2).
- Chatterbox-Turbo cannot be stopped mid-sentence. Its worst-case stop
  delay is its longest sentence generation in the run, measured with the
  recorded set.
- LFM3-Omni must run chat at the same 4,096 context its speech endpoint
  uses. Otherwise each switch reloads the model, taking about 6.2 s
  (methodology §11).
