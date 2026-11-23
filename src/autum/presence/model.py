"""What the bubble shows, taken only from runtime events (plan §3.2).

The bubble never guesses a state: it displays the last
runtime.state_changed it received, the microphone indicator, the voice
level from speech.output_level, and a red tint for a few seconds after a
component error. Holding the bubble sends talk commands to the runtime.
"""

from __future__ import annotations

from collections import deque
import json
import statistics
import time
from pathlib import Path

from PySide6.QtCore import Property, QObject, QTimer, Signal, Slot
from PySide6.QtGui import QGuiApplication
from PySide6.QtNetwork import QLocalSocket

from autum.events import Event

LABELS = {
    "idle": "Idle",
    "loading": "Loading",
    "listening": "Listening",
    "speech_detected": "Hearing you",
    "processing": "Processing",
    "thinking": "Thinking",
    "speaking": "Speaking",
    "interrupted": "Interrupted",
    "waiting_for_approval": "Needs your approval",
    "error": "Something went wrong",
}
ERROR_SHOWN_MS = 4000


class PresenceModel(QObject):
    changed = Signal()
    levelChanged = Signal()

    def __init__(self, socket_path: Path, measure: int = 0) -> None:
        super().__init__()
        self.socket_path = socket_path
        self.measure = measure
        self._state = "idle"
        self._microphone = "off"
        self._observation = "none"
        self._mode = "push_to_talk"
        self._level = 0.0
        self._error = ""
        self._connected = False
        self._buffer = b""
        cap = max(256, int(measure)) if measure else 256
        self._pending: deque[int] = deque(maxlen=cap)  # mono_ns of state changes not yet on screen
        self.applied_ms: deque[float] = deque(maxlen=cap)  # event -> model updated
        self.shown_ms: deque[float] = deque(maxlen=cap)  # event -> next frame on screen
        self._socket = QLocalSocket(self)
        self._socket.readyRead.connect(self._read)
        self._socket.connected.connect(self._on_connected)
        self._socket.disconnected.connect(self._on_disconnected)
        self._socket.errorOccurred.connect(lambda *_: self._on_disconnected())
        self._retry = QTimer(self, interval=1000, singleShot=True)
        self._retry.timeout.connect(self.start)
        self._error_timer = QTimer(self, interval=ERROR_SHOWN_MS, singleShot=True)
        self._error_timer.timeout.connect(self._clear_error)

    # ----- properties for QML ------------------------------------------

    state = Property(str, lambda self: self._state, notify=changed)
    stateLabel = Property(str, lambda self: LABELS.get(self._state, self._state), notify=changed)
    microphone = Property(str, lambda self: self._microphone, notify=changed)
    observation = Property(str, lambda self: self._observation, notify=changed)
    mode = Property(str, lambda self: self._mode, notify=changed)
    level = Property(float, lambda self: self._level, notify=levelChanged)
    error = Property(str, lambda self: self._error, notify=changed)
    connected = Property(bool, lambda self: self._connected, notify=changed)

    # ----- events -> display -------------------------------------------

    def apply(self, event: Event) -> None:
        p = event.payload
        match event.type:
            case "runtime.state_changed":
                if p["to"] != self._state:
                    self._pending.append(event.mono_ns)
                self._state, self._microphone, self._observation = p["to"], p["microphone"], p["observation"]
                if "mode" in p and p["mode"] in ("push_to_talk", "wake_word", "hands_free"):
                    self._mode = str(p["mode"])
                if self._state != "speaking":
                    self._set_level(0.0)
            case "speech.output_level":
                self._set_level(min(1.0, float(p["rms"]) * 4.0))
                return
            case "speech.playback_finished":
                self._set_level(0.0)
                return
            case "runtime.component_error":
                self._error = f"{p['component']}: {p['error']}"
                self._error_timer.start()
            case _:
                return
        self.applied_ms.append((time.monotonic_ns() - event.mono_ns) / 1e6)
        self.changed.emit()

    def _set_level(self, value: float) -> None:
        if value != self._level:
            self._level = value
            self.levelChanged.emit()

    def _clear_error(self) -> None:
        self._error = ""
        self.changed.emit()

    # ----- socket ------------------------------------------------------

    @Slot()
    def start(self) -> None:
        if self._socket.state() == QLocalSocket.LocalSocketState.UnconnectedState:
            self._socket.connectToServer(str(self.socket_path))

    def _on_connected(self) -> None:
        self._connected = True
        self.changed.emit()

    def _on_disconnected(self) -> None:
        if self._connected:
            self._connected = False
            self._state, self._microphone, self._observation = "idle", "off", "none"
            self._error = ""
            self._set_level(0.0)
            self.changed.emit()
        self._buffer = b""
        self._retry.start()

    def _read(self) -> None:
        self._buffer += bytes(self._socket.readAll())
        *lines, self._buffer = self._buffer.split(b"\n")
        for line in lines:
            try:
                self.apply(Event.from_json(line))
            except (ValueError, TypeError):
                continue  # malformed line: ignore it, keep the connection

    def _send(self, command: dict) -> None:
        if self._connected:
            self._socket.write(json.dumps(command).encode() + b"\n")
            self._socket.flush()

    @Slot()
    def talkStart(self) -> None:
        self._send({"command": "talk_start"})

    @Slot()
    def talkFinish(self) -> None:
        self._send({"command": "talk_finish"})

    @Slot()
    def talkAutoFinish(self) -> None:
        self._send({"command": "talk_auto_finish"})

    @Slot()
    def talkAbort(self) -> None:
        self._send({"command": "talk_abort"})

    @Slot(str)
    def setMode(self, mode: str) -> None:
        if mode in ("push_to_talk", "wake_word", "hands_free"):
            self._send({"command": "set_mode", "mode": mode})

    @Slot()
    def stopSpeaking(self) -> None:
        self._send({"command": "stop"})

    @Slot()
    def reloadConfig(self) -> None:
        self._send({"command": "reload_config"})

    @Slot()
    def reloadBubble(self) -> None:
        if self._connected:
            self._send({"command": "reload_bubble"})
        else:
            import os
            import sys

            os.execv(sys.executable, [sys.executable, "-m", "autum.presence", *sys.argv[1:]])

    @Slot()
    def closeBubble(self) -> None:
        if self._connected:
            self._send({"command": "close_bubble"})
        QTimer.singleShot(80, QGuiApplication.quit)

    # ----- measurement (WP4 acceptance: state lag) ----------------------

    @Slot()
    def frame_swapped(self) -> None:
        if not self._pending:
            return
        now = time.monotonic_ns()
        self.shown_ms.extend((now - ns) / 1e6 for ns in self._pending)
        self._pending.clear()
        if self.measure and len(self.shown_ms) >= self.measure:
            print(json.dumps(self.lag_report()), flush=True)
            QGuiApplication.quit()

    def lag_report(self) -> dict:
        def stats(values: list[float]) -> dict:
            ordered = sorted(values)
            return {
                "n": len(values),
                "median_ms": round(statistics.median(values), 1),
                "p95_ms": round(ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))], 1),
                "max_ms": round(max(values), 1),
            }

        return {"event_to_screen": stats(self.shown_ms), "event_to_model": stats(self.applied_ms)}
