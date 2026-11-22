"""The bubble, tested offscreen (no window appears)."""

import json
import os
import socket
import threading
import time
from pathlib import Path

import pytest

pytest.importorskip("PySide6")
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtCore import QEventLoop, QTimer, QUrl  # noqa: E402
from PySide6.QtGui import QGuiApplication  # noqa: E402
from PySide6.QtQml import QQmlApplicationEngine  # noqa: E402

from autum.bus import EventBus  # noqa: E402
from autum.events import new_id  # noqa: E402
from autum.presence import has_layer_shell  # noqa: E402
from autum.presence.cloud_item import FaceCloudItem, register_qml_types  # noqa: E402
from autum.presence.model import PresenceModel  # noqa: E402

APP = QGuiApplication.instance() or QGuiApplication([])
register_qml_types()
QML = Path(__file__).resolve().parents[1] / "src" / "autum" / "presence" / "Presence.qml"
WAYLAND_QML = QML.with_name("PresenceWayland.qml")


def spin(ms: int) -> None:
    loop = QEventLoop()
    QTimer.singleShot(ms, loop.quit)
    loop.exec()


def events():
    """A small conversation as the runtime would publish it."""
    bus = EventBus("s")
    out = []
    bus.subscribe(out.append)
    op, turn = new_id(), new_id()
    bus.fence.open(op)
    state = lambda a, b, mic="off": bus.emit("runtime.state_changed", {"from": a, "to": b, "reason": "t", "microphone": mic, "observation": "none"})  # noqa: E731
    state("idle", "speech_detected", "open")
    bus.emit("audio.speech_started", {"mode": "push_to_talk"}, turn_id=turn)
    state("speech_detected", "thinking")
    bus.emit("conversation.response_started", turn_id=turn, operation_id=op)
    bus.emit("conversation.response_chunk", {"text": "The capital "}, operation_id=op)
    bus.emit("conversation.response_chunk", {"text": "is Ottawa."}, operation_id=op)
    state("thinking", "speaking")
    bus.emit("speech.output_level", {"rms": 0.1}, operation_id=op)
    return out


def test_the_model_shows_only_what_the_events_say(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    assert model.state == "idle" and model.microphone == "off"
    for event in events():
        model.apply(event)
    assert (model.state, model.stateLabel, model.microphone) == ("speaking", "Speaking", "off")
    assert model.level == pytest.approx(0.4)
    assert model.error == ""
    model.apply(EventBus("s").emit("runtime.component_error", {"component": "microphone", "error": "no audio"}))
    assert model.error == "microphone: no audio"  # tints the bubble red for a few seconds


def test_leaving_speaking_resets_the_voice_level(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    for event in events():
        model.apply(event)
    model.apply(EventBus("s").emit("runtime.state_changed", {"from": "speaking", "to": "idle", "reason": "t", "microphone": "off", "observation": "none"}))
    assert model.level == 0.0


def test_the_qml_loads_without_warnings(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("presence", model)
    engine.load(QUrl.fromLocalFile(str(QML)))
    for event in events():
        model.apply(event)
    spin(100)
    assert engine.rootObjects() and warnings == []
    engine.deleteLater()


@pytest.mark.skipif(not has_layer_shell(), reason="layer-shell-qt not installed")
def test_the_wayland_layer_shell_qml_loads_without_warnings(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("presence", model)
    engine.load(QUrl.fromLocalFile(str(WAYLAND_QML)))
    spin(100)
    assert engine.rootObjects() and warnings == []
    engine.deleteLater()


def test_socket_round_trip_events_in_commands_out(tmp_path):
    path = tmp_path / "events.sock"
    server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    server.bind(str(path))
    server.listen(1)
    received = []

    def serve():
        conn, _ = server.accept()
        for event in events():
            conn.sendall(event.to_json().encode() + b"\n")
        conn.sendall(b"garbage line\n")  # ignored, connection kept
        conn.settimeout(2)
        buffer = b""
        deadline = time.monotonic() + 2
        while time.monotonic() < deadline and buffer.count(b"\n") < 2:
            try:
                buffer += conn.recv(4096)
            except socket.timeout:
                break
        received.extend(json.loads(line) for line in buffer.splitlines())
        conn.close()

    thread = threading.Thread(target=serve, daemon=True)
    thread.start()
    model = PresenceModel(path)
    model.start()
    spin(300)
    state_seen = (model.connected, model.state, model.level)
    model.talkStart()
    model.talkFinish()
    spin(300)
    thread.join(3)
    server.close()
    assert state_seen == (True, "speaking", pytest.approx(0.4))
    assert received == [{"command": "talk_start"}, {"command": "talk_finish"}]


def test_lag_is_measured_from_event_to_frame(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    for event in events():
        model.apply(event)
    model.frame_swapped()
    report = model.lag_report()
    assert report["event_to_screen"]["n"] == 3 and report["event_to_screen"]["max_ms"] >= 0


def test_face_cloud_renders_valid_premultiplied_frames_and_reacts_to_states():
    import numpy as np
    from PySide6.QtGui import QColor

    item = FaceCloudItem()
    item.tint = QColor("#dce6f2")
    idle = item.render_rgba(180, 180, 60.0)
    assert idle.shape == (180, 180, 4) and idle.dtype == np.uint8
    # Premultiplied alpha invariant: rgb <= alpha everywhere
    assert np.all(idle[:, :, :3] <= idle[:, :, 3:4])
    # Center orb is opaque & bright, corners are transparent
    assert idle[90, 90, 3] > 200 and idle[2, 2, 3] == 0
    assert int(idle[:, :, :3].max()) > 160

    # Speaking voice burst adds particles/brightness
    item.speaking = True
    item.voice = 0.85
    speaking = item.render_rgba(180, 180, 60.0)
    assert speaking[:, :, :3].mean() > idle[:, :, :3].mean()

    # Muted gap darkens the lower-right rim quadrant compared to idle
    item.speaking = False
    item.voice = 0.0
    item.gap = 1.0
    muted = item.render_rgba(180, 180, 60.0)
    assert muted[120:155, 120:155, :3].mean() < idle[120:155, 120:155, :3].mean()

    # Observation halo lights up the outer ring in cyan-teal
    item.gap = 0.0
    item.observing = True
    item.observeGlow = 0.8
    obs = item.render_rgba(180, 180, 60.0)
    assert obs[25, 90, 2] > idle[25, 90, 2] + 20
