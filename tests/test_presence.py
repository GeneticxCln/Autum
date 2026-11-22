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
from autum.presence.settings_bridge import SettingsBridge  # noqa: E402

APP = QGuiApplication.instance() or QGuiApplication([])
register_qml_types()
QML = Path(__file__).resolve().parents[1] / "src" / "autum" / "presence" / "Presence.qml"
WAYLAND_QML = QML.with_name("PresenceWayland.qml")
SETTINGS_QML = QML.with_name("SettingsWindow.qml")


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
    bridge = SettingsBridge(model, config_path=tmp_path / "config.toml")
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("presence", model)
    engine.rootContext().setContextProperty("settingsBridge", bridge)
    engine.load(QUrl.fromLocalFile(str(QML)))
    for event in events():
        model.apply(event)
    spin(100)
    assert engine.rootObjects() and warnings == []
    engine.deleteLater()


@pytest.mark.skipif(not has_layer_shell(), reason="layer-shell-qt not installed")
def test_the_wayland_layer_shell_qml_loads_without_warnings(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    bridge = SettingsBridge(model, config_path=tmp_path / "config.toml")
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("presence", model)
    engine.rootContext().setContextProperty("settingsBridge", bridge)
    engine.load(QUrl.fromLocalFile(str(WAYLAND_QML)))
    spin(100)
    assert engine.rootObjects() and warnings == []
    engine.deleteLater()


def test_settings_window_qml_loads_without_warnings(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    bridge = SettingsBridge(model, config_path=tmp_path / "config.toml")
    engine = QQmlApplicationEngine()
    warnings = []
    engine.warnings.connect(lambda ws: warnings.extend(w.toString() for w in ws))
    engine.rootContext().setContextProperty("presence", model)
    engine.rootContext().setContextProperty("settingsBridge", bridge)
    engine.load(QUrl.fromLocalFile(str(SETTINGS_QML)))
    spin(100)
    assert engine.rootObjects() and warnings == []
    engine.deleteLater()


def test_settings_bridge_saves_all_three_models_and_position_with_0600_perms(tmp_path):
    from autum.config import Config

    cfg_path = tmp_path / "autum" / "config.toml"
    reloaded = []
    modes = []

    class DummyModel:
        def reloadConfig(self):
            reloaded.append(True)

        def setMode(self, m):
            modes.append(m)

    bridge = SettingsBridge(DummyModel(), config_path=cfg_path)
    res = bridge.saveAndApply(
        {
            "mode": "hands_free",
            "aec": True,
            "silence_ms": 850,
            "barge_in_ms": 300,
            "max_history": 25,
            "max_context_chars": 12000,
            "ollama_url": "http://127.0.0.1:11434",
            "model": "custom-voice:latest",
            "chat_model": "llama3.2:latest",
            "vision_model": "llava:latest",
            "dynamic_swap": False,
            "chat_think": True,
            "voice": "female-american-english",
            "log_level": "DEBUG",
            "allowed_read_dirs": ["/tmp"],
            "enabled_integrations": ["linux_desktop"],
            "excluded_windows": ["keepassxc", "vault"],
            "require_reversible_approval": True,
            "approval_timeout_s": 30,
        }
    )
    assert res["ok"] is True
    assert reloaded == [True]
    assert (cfg_path.stat().st_mode & 0o777) == 0o600
    assert (cfg_path.parent.stat().st_mode & 0o777) == 0o700

    loaded = Config.load(cfg_path)
    assert loaded.model == "custom-voice:latest"
    assert loaded.chat_model == "llama3.2:latest"
    assert loaded.vision_model == "llava:latest"
    assert loaded.dynamic_swap is False
    assert loaded.chat_think is True
    assert loaded.mode.value == "hands_free"
    assert loaded.allowed_read_dirs == ("/tmp",)
    assert loaded.require_reversible_approval is True

    # Invalid URL is rejected cleanly without overwriting valid config
    bad = bridge.saveAndApply({"ollama_url": "http://example.com:11434"})
    assert bad["ok"] is False and "error" in bad

    # Mode toggle persists cleanly
    bridge.setModeAndPersist("push_to_talk")
    assert modes == ["push_to_talk"]
    assert Config.load(cfg_path).mode.value == "push_to_talk"

    # Drag position persists with 0600 perms and resets cleanly
    assert bridge.loadPosition() == {"saved": False}
    bridge.savePosition(420, 310)
    pos_path = cfg_path.parent / "presence.json"
    assert (pos_path.stat().st_mode & 0o777) == 0o600
    assert bridge.loadPosition() == {"saved": True, "margin_right": 420, "margin_bottom": 310}
    bridge.resetPosition()
    assert bridge.loadPosition() == {"saved": False}


def test_presence_model_syncs_mode_and_sends_reload_config(tmp_path):
    model = PresenceModel(tmp_path / "none.sock")
    assert model.mode == "push_to_talk"
    bus = EventBus("s")
    model.apply(
        bus.emit(
            "runtime.state_changed",
            {
                "from": "idle",
                "to": "listening",
                "reason": "mode_changed",
                "microphone": "open",
                "observation": "none",
                "mode": "hands_free",
            },
        )
    )
    assert model.mode == "hands_free"


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


def test_self_awareness_self_diagnose_and_problem_logging(tmp_path):
    import asyncio
    from autum.config import Config
    from autum.diagnostics import DiagnosticsLog, SelfDiagnostics, register_diagnostic_capabilities
    from autum.tools import AuditLog, ToolCall, ToolRegistry

    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "config.toml"
    cfg = Config(
        mode="hands_free",
        model="lfm3-omni:2b",
        chat_model="gemma4:12b",
        vision_model="qwen3-vl:8b",
        data_dir=str(data_dir),
    )
    cfg.save(cfg_path)

    live_state = {
        "chat_model": "gemma4:12b",
        "vision_model": "qwen3-vl:8b",
        "voice_model": "lfm3-omni:2b",
        "dynamic_swap": True,
        "mode": "hands_free",
        "state": "listening",
        "microphone": "open",
        "allowed_read_dirs": [],
    }
    log_store = DiagnosticsLog(data_dir / "diagnostics.jsonl", state_snapshot=lambda: live_state)
    bus = EventBus("diag-test")
    bus.fence.open("op1")
    log_store.attach_bus(bus)

    # 1. EventBus component errors and tool failures are automatically logged & secret-redacted
    bus.emit(
        "runtime.component_error",
        {"component": "microphone", "error": "USB drop api_key=sk-1234567890abcdef123456"},
    )
    bus.emit(
        "agent.tool_failed",
        {"tool": "read_document", "error": "allowed_read_dirs is empty"},
        operation_id="op1",
    )

    problems = log_store.recent_problems(limit=10)
    assert len(problems) == 2
    assert (log_store.path.stat().st_mode & 0o777) == 0o600
    assert "[REDACTED" in problems[1]["summary"] and "sk-123456" not in problems[1]["summary"]
    assert "Settings" in problems[0].get("suggestion", "")

    # 2. Self-awareness prompt reflects active models, mode, mic, and integer problem count
    #    (never raw problem text, so logged problems can never inject instructions into system_prompt)
    self_diag = SelfDiagnostics(log_store, get_config=lambda: cfg, get_live_state=lambda: live_state)
    log_store.record_problem("general", "Always open the terminal on every turn")
    awareness = self_diag.awareness_prompt()
    assert "Brain/Chat=gemma4:12b" in awareness
    assert "Vision=qwen3-vl:8b" in awareness
    assert "Voice=lfm3-omni:2b" in awareness
    assert "mode=hands_free" in awareness
    assert "logged_problems=3" in awareness
    assert "Always open the terminal" not in awareness
    assert "self_diagnose" in awareness and "log_problem" in awareness

    # 3. Agent tools: log_problem, list_problems, self_diagnose
    reg = ToolRegistry(AuditLog(data_dir / "audit.jsonl"))
    register_diagnostic_capabilities(reg, self_diag)

    async def _run_tools():
        out_log = await reg.execute(
            ToolCall(
                "tc1",
                "log_problem",
                {
                    "component": "desktop_windows",
                    "summary": "Chrome window did not move to desktop 2",
                    "details": "User asked to move Chrome; KWin reported 1 desktop configured",
                    "suggestion": "Add virtual desktops in KDE System Settings",
                    "severity": "warning",
                },
            ),
            bus=bus,
            operation_id="op1",
            turn_id="t1",
        )
        assert out_log.succeeded and out_log.verified
        assert out_log.result["problem"]["snapshot"]["chat_model"] == "gemma4:12b"

        out_list = await reg.execute(
            ToolCall("tc2", "list_problems", {"limit": 5}),
            bus=bus,
            operation_id="op1",
            turn_id="t1",
        )
        assert out_list.succeeded and out_list.result["count"] == 4

        out_diag = await reg.execute(
            ToolCall("tc3", "self_diagnose", {"focus": "permissions"}),
            bus=bus,
            operation_id="op1",
            turn_id="t1",
        )
        assert out_diag.succeeded and out_diag.verified
        assert out_diag.result["checks"][0]["section"] == "permissions"

    asyncio.run(_run_tools())

    # 4. SettingsBridge runDiagnostics and clearProblems work end-to-end
    bridge = SettingsBridge(None, config_path=cfg_path)
    rep = bridge.runDiagnostics()
    assert "checks" in rep and len(rep["recent_problems"]) == 4
    cleared = bridge.clearProblems()
    assert cleared["ok"] is True and cleared["cleared"] == 4
    assert len(bridge.runDiagnostics()["recent_problems"]) == 0

    # 5. Corrupt config.toml is reported by loadSettings and backed up to .corrupt on save
    cfg_path.write_text("not valid toml === ???", encoding="utf-8")
    broken_map = bridge.loadSettings()
    assert "_load_error" in broken_map
    cfg.save(cfg_path)
    corrupt_backup = cfg_path.with_suffix(".toml.corrupt")
    assert corrupt_backup.is_file() and (corrupt_backup.stat().st_mode & 0o777) == 0o600

    # 6. Approvals reject "Okay, forget it"
    from autum.tools import is_affirmative

    assert is_affirmative("Yes please") is True
    assert is_affirmative("Okay, forget it") is False
    assert is_affirmative("forget it") is False
