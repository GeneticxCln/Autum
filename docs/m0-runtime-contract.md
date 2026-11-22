# M0 — Runtime contract

*Status: draft for M2. Revise after the M1 pipeline decision.*

This document fixes the interfaces the voice engine, inference,
visual presence, and later subsystems build against. The code in
`src/autum/` is the reference implementation; `tests/` proves it.

## 1. Event envelope

Every event carries:

| Field | Meaning |
|---|---|
| `type` | One of the event types below |
| `session_id` | The running session |
| `turn_id` | The user utterance or text message the event belongs to |
| `operation_id` | The response operation; required for output events |
| `tool_call_id` | Set on tool events |
| `causation_id` | The event that caused this one; always an earlier event |
| `event_id` | Unique ID |
| `mono_ns` | `CLOCK_MONOTONIC` nanoseconds; comparable across Autum's processes, used for latency |
| `wall_ms` | Wall-clock milliseconds, for logs |
| `provenance` | `user`, `system`, `model`, or `observed` |
| `schema_version` | Currently `1` |
| `payload` | Type-specific fields, listed below |

Events are validated when constructed and when decoded from the
transport. Unknown types, unknown fields, missing payload fields, and
output events without an operation ID are rejected.

Only `user` provenance can authorize an action (plan §5.1). Perception
and integration output is always `observed`; model text is `model`.

## 2. Event types

Required payload fields follow each type. `*` marks events that must
carry an `operation_id`.

```text
runtime.state_changed           from, to, reason, microphone, observation
runtime.model_loading           model
runtime.model_ready             model, load_ms
runtime.component_error         component, error
runtime.component_recovered     component
audio.capture_started           -
audio.capture_stopped           -
audio.mute_changed              muted
audio.speech_started            mode (push_to_talk | hands_free)
audio.speech_ended              confidence
audio.transcribed               text, final
conversation.user_message       text, source (speech | text)
conversation.response_started   - *
conversation.response_chunk     text *
conversation.response_finished  text *
speech.playback_started         - *  (first audible sample)
speech.output_level             rms *
speech.playback_finished        reason (completed | cancelled) *
operation.cancel_requested      source (push_to_talk | barge_in | user_command) *
operation.cancelled             source, elapsed_ms, clean *
agent.thinking                  -
agent.tool_requested            tool, arguments *
agent.tool_started              tool *
agent.tool_completed            tool, result *
agent.tool_verified             tool, verified *
agent.tool_failed               tool, error *
agent.approval_required         action, target, level *
agent.approval_resolved         outcome (granted | denied | expired) *
perception.started              source
perception.completed            source
```

## 3. Ordering

The bus delivers every event to every subscriber in one total order.
An event published while another is being delivered waits until the
current event has reached all subscribers. Subscribers are synchronous
and must return quickly; slow work is scheduled as a task.

## 4. Runtime states

The runtime owns the single primary state (plan §3.2) and announces
every change with `runtime.state_changed`, which also carries the
indicators. Allowed transitions are listed in `TRANSITIONS` in
`src/autum/runtime.py`; any other transition is a bug and raises.

Indicators:

- `microphone`: `off`, `open`, or `muted`.
- `observation`: `none` until perception exists.

In hands-free mode the runtime rests in Listening while capture is open
and not muted; in push-to-talk mode it rests in Idle.

## 5. Component interfaces

**Voice input** (voice engine, WP2) publishes `audio.*` events. It
assigns a new `turn_id` at each `audio.speech_started` and uses it for
that utterance's `speech_ended` and `transcribed`. In hands-free mode it
only reports speech it has judged real (not a cough, a backchannel, or
Autum's own echo).

**Inference** (`Inference` in `runtime.py`): `stream(messages,
operation_id)` yields response text and stops when its task is
cancelled. A message is `{"role", "content"}`, plus `"turn_id"` for
history entries. A user message may also carry `"audio"` (WAV bytes) when
the pipeline's model hears speech directly.

**Spoken turns that the model hears directly** (pipeline D): after
`audio.speech_ended`, the voice engine calls
`runtime.submit_speech(turn_id, wav, cause)` instead of waiting for a
transcript. The runtime starts the response with the audio in the
history. The voice engine transcribes the audio once the reply has
finished or been cancelled, and publishes `audio.transcribed`. The
runtime then replaces the audio in the history with the text. A final
transcript for the turn the runtime is still waiting on starts a
response; a transcript for an earlier turn only fills in history.

**Speech output** (`SpeechOutput` in `runtime.py`):

- `enqueue(text, operation_id, turn_id)` queues text and returns at once.
- `drain(operation_id)` returns when everything queued has played.
- `stop()` silences output immediately, drops the queue, and returns
  the text whose playback had started.
- It publishes `speech.playback_started` at the first audible sample,
  `speech.output_level` while playing, and `speech.playback_finished`.

## 6. Cancellation

A response is one operation, run as one asyncio task. Cancelling it has
two parts.

Synchronous, inside the event that triggered it (no awaits, so it
cannot be delayed by other work):

1. Silence speech output (`stop()`).
2. Move the operation to *cancelling* in the fence. From now on the bus
   drops the operation's output; only cancellation events,
   `speech.playback_finished`, and component errors get through.
3. Publish `operation.cancel_requested`.
4. Record what the user actually heard in the history, marked as
   interrupted.
5. Cancel the task. `CancelledError` reaches the inference stream, which
   closes its request so the model stops generating.
6. Enter the Interrupted state.

Asynchronous:

7. Wait up to `cancel_timeout_s` for the task to end. If it does not,
   publish `runtime.component_error`; the fence already keeps its output
   out.
8. Publish `operation.cancelled` with `clean` and `elapsed_ms`, close
   the operation, and return to the rest state unless a new turn has
   already started.

Triggers: user speech or push-to-talk during a response (`barge_in`,
`push_to_talk`), new text input, or an explicit cancel
(`user_command`).

Tools (WP6) run inside the operation task, so they receive the same
cancellation. A tool that must not be interrupted mid-way will shield
its critical section and say so in its definition.

## 7. Hard mute

The hard mute is the operating system's mute on the microphone source
(PipeWire). The user toggles it with a global shortcut or the
microphone's own mute button. Autum:

- never unmutes the source; no Autum code calls the unmute API;
- reads the actual source mute state and publishes
  `audio.mute_changed`, which drives the indicator;
- drops captured audio while muted, as a second layer.

A hardware mute button on the microphone is a further layer Autum
cannot see; its own light is the indicator.

## 8. Transport

In-process components use the bus directly. Out-of-process consumers,
starting with the visual presence, connect to a Unix domain socket at
`$XDG_RUNTIME_DIR/autum/events.sock` (`src/autum/transport.py`). See
ADR 0001.

- **Access:** the directory is mode 0700 and the socket 0600, so only the
  user's own account can connect. A second Autum refuses to start while
  one is listening; a stale socket from a crash is replaced.
- **Events out:** every event as one JSON line. A new client first
  receives the latest `audio.mute_changed` and `runtime.state_changed`,
  in the order they happened, so it shows the right state at once.
- **Bounded memory:** each client has a 512-line queue. A client that
  falls that far behind is disconnected; it reconnects and gets a fresh
  snapshot.
- **Commands in:** one JSON object per line, validated before use; any
  other input is ignored. Lines over 8 KiB drop the client.

```text
{"command": "talk_start"}                        start a push-to-talk turn (or interrupt in hands-free mode)
{"command": "talk_finish"}                       end a push-to-talk turn
{"command": "stop"}                              drop the current utterance and cancel the response
{"command": "text", "text": "..."}               typed message, 1-2000 characters
{"command": "set_mode", "mode": "hands_free"}    switch between "push_to_talk" and "hands_free"
```

Commands are user actions in a window the user controls, so they carry
the same authority as keys in Autum's terminal. Approval commands will be
added with WP6 and follow §5.2 of the plan.
