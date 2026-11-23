"""Tests for Autum v1.1 features (Phases 1–4)."""

from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path

import pytest

from autum.bus import EventBus
from autum.capabilities import NotesStore
from autum.config import Config
from autum.diagnostics import TranscriptLog
from autum.integrations.window_backends import (
    EwmhX11Controller,
    HyprlandController,
    SwayController,
)
from autum.memory import MemoryStore


def run(coro):
    return asyncio.run(coro)


# ----- Phase 1: Audio device config & PipeWire enumeration --------------

def test_config_round_trips_audio_input_and_output(tmp_path):
    cfg_path = tmp_path / "config.toml"
    cfg = Config(
        audio_input="alsa_input.usb-Logitech_StreamCam",
        audio_output="alsa_output.usb-SB_Omni",
    )
    cfg.save(cfg_path)
    loaded = Config.load(cfg_path)
    assert loaded.audio_input == "alsa_input.usb-Logitech_StreamCam"
    assert loaded.audio_output == "alsa_output.usb-SB_Omni"


# ----- Phase 2: TranscriptLog & SettingsBridge Memory/Notes/Transcript --

def test_transcript_log_redacts_secrets_enforces_0600_and_clears(tmp_path):
    path = tmp_path / "autum" / "transcript.jsonl"
    log = TranscriptLog(path, max_entries=5)
    bus = EventBus("s")
    bus.fence.open("op1")
    log.attach_bus(bus)
    bus.emit("conversation.user_message", {"text": "my key is sk-1234567890abcdef123456", "source": "text"})
    bus.emit("conversation.response_finished", {"text": "Got it, stored safely."}, operation_id="op1")
    entries = log.recent(10)
    assert len(entries) == 2
    assert (path.stat().st_mode & 0o777) == 0o600
    assert "sk-123456" not in entries[0]["text"] and "[REDACTED" in entries[0]["text"]
    assert entries[1]["role"] == "assistant"
    assert log.clear() == 2
    assert log.recent(10) == []


def test_settings_bridge_memory_notes_transcript_and_setup(tmp_path):
    pytest.importorskip("PySide6")
    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtGui import QGuiApplication
    from autum.presence.settings_bridge import SettingsBridge

    _app = QGuiApplication.instance() or QGuiApplication([])
    data_dir = tmp_path / "data"
    cfg_path = tmp_path / "config.toml"
    Config(data_dir=str(data_dir), audio_input="mic_a", audio_output="spk_b").save(cfg_path)

    mem = MemoryStore(data_dir / "memory.json")
    mem.remember("editor", "Neovim", category="preference")
    notes = NotesStore(data_dir)
    notes.create("shopping", "coffee and oats")
    tx = TranscriptLog(data_dir / "transcript.jsonl")
    tx.record("user", "Hello")
    tx.record("assistant", "Hi there!")

    bridge = SettingsBridge(None, config_path=cfg_path)
    snap = bridge.loadMemoryAndNotes()
    assert len(snap["memories"]) == 1 and snap["memories"][0]["key"] == "editor"
    assert len(snap["notes"]) == 1 and snap["notes"][0]["title"] == "shopping"
    assert len(snap["transcript"]) == 2

    assert bridge.deleteMemory("editor")["ok"] is True
    assert bridge.deleteNote("shopping")["ok"] is True
    assert bridge.clearTranscript()["cleared"] == 2
    after = bridge.loadMemoryAndNotes()
    assert after == {"memories": [], "notes": [], "transcript": []}

    # pullModelAsync rejects remote URLs (localhost-only invariant)
    events: list[dict] = []
    bridge.pullProgress.connect(events.append)
    bridge.pullModelAsync("gemma4:12b", "http://example.com:11434")
    assert len(events) == 1 and "localhost" in events[0]["error"]


# ----- Phase 3: Multi-compositor window backends ------------------------

def test_hyprland_controller_lists_excludes_moves_and_closes():
    dispatched: list[list[str]] = []

    def fake_run(cmd, _timeout):
        args = list(cmd)
        if args == ["hyprctl", "-j", "clients"]:
            return 0, json.dumps(
                [
                    {
                        "address": "0x111", "class": "google-chrome", "title": "GitHub",
                        "at": [0, 0], "size": [1920, 1080], "workspace": {"id": 1}, "mapped": True,
                    },
                    {
                        "address": "0x222", "class": "org.keepassxc.KeePassXC", "title": "Passwords",
                        "at": [0, 0], "size": [800, 600], "workspace": {"id": 1}, "mapped": True,
                    },
                ]
            )
        if args == ["hyprctl", "-j", "activewindow"]:
            return 0, json.dumps({"address": "0x111"})
        if args == ["hyprctl", "-j", "monitors"]:
            return 0, json.dumps([{"x": 0, "y": 0, "width": 1920, "height": 1080, "focused": True}])
        dispatched.append(args)
        return 0, "ok"

    ctrl = HyprlandController(runner=fake_run)
    wins = run(ctrl.list_windows(excluded=("keepassxc",)))
    assert len(wins) == 1 and wins[0]["app"] == "google-chrome" and wins[0]["active"] is True

    with pytest.raises(PermissionError):
        run(ctrl.focus_window("keepassxc", excluded=("keepassxc",)))

    moved = run(ctrl.move_window("chrome", position="left", desktop=2, excluded=("keepassxc",)))
    assert (moved["target_x"], moved["target_width"], moved["desktop"]) == (0, 960, 2)
    assert any("movetoworkspacesilent" in c for c in dispatched)
    assert run(ctrl.close_window("chrome", excluded=("keepassxc",)))["closed"] is True


def test_sway_controller_lists_excludes_and_focuses():
    dispatched: list[list[str]] = []

    def fake_run(cmd, _timeout):
        args = list(cmd)
        if args == ["swaymsg", "-t", "get_tree"]:
            return 0, json.dumps(
                {
                    "type": "root",
                    "nodes": [
                        {
                            "type": "workspace",
                            "num": 2,
                            "nodes": [
                                {"type": "con", "id": 42, "pid": 123, "app_id": "firefox", "name": "Docs", "rect": {"x": 0, "y": 0, "width": 1280, "height": 720}, "focused": True},
                                {"type": "con", "id": 99, "pid": 456, "app_id": "bitwarden", "name": "Vault", "rect": {"x": 0, "y": 0, "width": 800, "height": 600}, "focused": False},
                            ],
                        }
                    ],
                }
            )
        dispatched.append(args)
        return 0, '[{"success": true}]'

    ctrl = SwayController(runner=fake_run)
    wins = run(ctrl.list_windows(excluded=("bitwarden",)))
    assert len(wins) == 1 and wins[0]["id"] == "42" and wins[0]["desktop"] == 2
    res = run(ctrl.focus_window("firefox", excluded=("bitwarden",)))
    assert res["active"] is True and dispatched[-1] == ["swaymsg", "[con_id=42] focus"]


def test_ewmh_x11_controller_parses_wmctrl_and_moves():
    dispatched: list[list[str]] = []

    def fake_run(cmd, _timeout):
        args = list(cmd)
        if args == ["wmctrl", "-lGx"]:
            return 0, (
                "0x03200007  0 100  50 1200  800 navigator.Firefox host Mozilla Firefox\n"
                "0x04000001  0 200 100  600  400 keepassxc.KeePassXC host Secret Vault\n"
            )
        dispatched.append(args)
        return 0, ""

    ctrl = EwmhX11Controller(runner=fake_run)
    wins = run(ctrl.list_windows(excluded=("keepassxc",)))
    assert len(wins) == 1 and wins[0]["app"] == "Firefox"
    moved = run(ctrl.move_window("firefox", position="right", excluded=("keepassxc",)))
    assert (moved["target_x"], moved["target_width"]) == (960, 960)
    assert any("-e" in c and "0,960,0,960,1080" in c for c in dispatched)


# ----- v1.1 Edge-Case Audit Regressions ---------------------------------

def test_memory_store_syncs_external_deletions_without_resurrection(tmp_path):
    mem_path = tmp_path / "memory.json"
    daemon_store = MemoryStore(mem_path)
    daemon_store.remember("editor", "Neovim", category="preference")
    daemon_store.remember("theme", "dark", category="preference")

    # Simulate Settings UI process deleting a memory on disk
    ui_store = MemoryStore(mem_path)
    deleted = ui_store.delete("editor")
    assert deleted.key == "editor"

    # Ensure mtime advances even on coarse filesystems
    st = mem_path.stat()
    os.utime(mem_path, ns=(st.st_atime_ns, st.st_mtime_ns + 1_000_000))

    assert [m.key for m in daemon_store.list_all()] == ["theme"]
    assert daemon_store.find_by_key("editor") is None

    # Writing a new memory from the daemon must NOT resurrect the deleted key
    daemon_store.remember("shell", "fish", category="preference")
    fresh = MemoryStore(mem_path)
    assert sorted(m.key for m in fresh.list_all()) == ["shell", "theme"]


def test_sway_move_window_uses_real_output_geometry():
    dispatched: list[list[str]] = []

    def fake_run(cmd, _timeout):
        args = list(cmd)
        if args == ["swaymsg", "-t", "get_tree"]:
            return 0, json.dumps(
                {
                    "type": "root",
                    "nodes": [
                        {
                            "type": "workspace",
                            "num": 1,
                            "nodes": [
                                {"type": "con", "id": 7, "pid": 100, "app_id": "firefox", "name": "Web", "rect": {"x": 1920, "y": 0, "width": 1000, "height": 800}, "focused": True},
                            ],
                        }
                    ],
                }
            )
        if args == ["swaymsg", "-t", "get_outputs"]:
            return 0, json.dumps(
                [
                    {"name": "DP-1", "active": True, "focused": False, "rect": {"x": 0, "y": 0, "width": 1920, "height": 1080}},
                    {"name": "DP-2", "active": True, "focused": True, "rect": {"x": 1920, "y": 0, "width": 2560, "height": 1440}},
                ]
            )
        dispatched.append(args)
        return 0, '[{"success": true}]'

    ctrl = SwayController(runner=fake_run)
    moved = run(ctrl.move_window("firefox", position="left"))
    assert (moved["target_x"], moved["target_y"], moved["target_width"], moved["target_height"]) == (1920, 0, 1280, 1440)
    assert any("resize set 1280 px 1440 px, move position 1920 px 0 px" in " ".join(c) for c in dispatched)


def test_ewmh_x11_active_excluded_window_real_workarea_and_malformed_lines():
    dispatched: list[list[str]] = []

    def fake_run(cmd, _timeout):
        args = list(cmd)
        if args == ["wmctrl", "-lGx"]:
            return 0, (
                "0x03200007  0 100  50 1200  800 navigator.Firefox host Mozilla Firefox\n"
                "0x04000001  0 200 100  600  400 keepassxc.KeePassXC host Secret Vault\n"
                "0x05000002  0 bad  50  600  400 broken.App host Malformed Geometry\n"
            )
        if args == ["xprop", "-root", "_NET_ACTIVE_WINDOW"]:
            return 0, "_NET_ACTIVE_WINDOW(WINDOW): window id # 0x4000001\n"
        if args == ["wmctrl", "-d"]:
            return 0, "0  * DG: 3840x2160  VP: 0,0  WA: 0,40 3840x2120  Workspace 1\n"
        dispatched.append(args)
        return 0, ""

    ctrl = EwmhX11Controller(runner=fake_run)
    assert run(ctrl.active_excluded_window(excluded=("keepassxc",))) == "keepassxc"
    wins = run(ctrl.list_windows(excluded=("keepassxc",)))
    assert len(wins) == 1 and wins[0]["app"] == "Firefox" and wins[0]["active"] is False

    moved = run(ctrl.move_window("firefox", position="right", excluded=("keepassxc",)))
    assert (moved["target_x"], moved["target_y"], moved["target_width"], moved["target_height"]) == (1920, 40, 1920, 2120)
    assert any("-e" in c and "0,1920,40,1920,2120" in c for c in dispatched)


def test_pull_model_async_emits_done_exactly_once(tmp_path):
    pytest.importorskip("PySide6")
    import http.server
    import threading
    import time

    os.environ["QT_QPA_PLATFORM"] = "offscreen"
    from PySide6.QtCore import QCoreApplication
    from PySide6.QtGui import QGuiApplication
    from autum.presence.settings_bridge import SettingsBridge

    _app = QGuiApplication.instance() or QGuiApplication([])

    class _FakeOllamaPull(http.server.BaseHTTPRequestHandler):
        def do_POST(self):
            length = int(self.headers.get("Content-Length", "0"))
            if length:
                self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/x-ndjson")
            self.end_headers()
            for obj in (
                {"status": "downloading", "completed": 50, "total": 100},
                {"status": "success"},
            ):
                self.wfile.write((json.dumps(obj) + "\n").encode("utf-8"))
                self.wfile.flush()

        def log_message(self, format, *args):
            pass

    srv = http.server.HTTPServer(("127.0.0.1", 0), _FakeOllamaPull)
    port = srv.server_address[1]
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        cfg_path = tmp_path / "config.toml"
        Config().save(cfg_path)
        bridge = SettingsBridge(None, config_path=cfg_path)
        events: list[dict] = []
        bridge.pullProgress.connect(events.append)
        bridge.pullModelAsync("tiny:1b", f"http://127.0.0.1:{port}")
        deadline = time.monotonic() + 5.0
        while time.monotonic() < deadline and not any(e.get("done") for e in events):
            QCoreApplication.processEvents()
            time.sleep(0.02)
        # Allow any spurious duplicate emission time to arrive
        time.sleep(0.1)
        QCoreApplication.processEvents()
        done_events = [e for e in events if e.get("done")]
        assert len(done_events) == 1
        assert done_events[0]["percent"] == 100 and done_events[0]["error"] == ""
    finally:
        srv.shutdown()
        srv.server_close()


def test_self_diagnosis_audio_honors_configured_audio_input(monkeypatch):
    from autum import diagnostics

    monkeypatch.setattr(
        diagnostics,
        "query_ollama_models",
        lambda _url, timeout_s=2.0: [
            {"name": "lfm3-omni:2b", "size_gb": 1.4, "loaded": False, "status": ""},
            {"name": "gemma4:12b", "size_gb": 8.0, "loaded": False, "status": ""},
            {"name": "qwen3-vl:8b", "size_gb": 5.0, "loaded": False, "status": ""},
        ],
    )
    monkeypatch.setattr(diagnostics, "gpu_vram_gb", lambda: 16.0)
    monkeypatch.setattr(diagnostics.shutil, "which", lambda name: f"/usr/bin/{name}")

    cfg = Config(audio_input="alsa_input.usb-CustomMic")

    # Case A: configured mic is muted
    monkeypatch.setattr(
        diagnostics,
        "_run_cmd",
        lambda cmd, timeout_s=3.0: (0, "Mute: yes\n") if cmd == ["pactl", "get-source-mute", "alsa_input.usb-CustomMic"] else (0, ""),
    )
    rep_muted = diagnostics.run_system_health_checks(cfg)
    audio_muted = next(c for c in rep_muted["checks"] if c["section"] == "audio")
    assert audio_muted["status"] == "warning"
    assert "MUTED" in audio_muted["summary"]
    assert "alsa_input.usb-CustomMic" in audio_muted["details"]

    # Case B: configured mic is unplugged / missing in PipeWire
    monkeypatch.setattr(
        diagnostics,
        "_run_cmd",
        lambda cmd, timeout_s=3.0: (1, "") if cmd == ["pactl", "get-source-mute", "alsa_input.usb-CustomMic"] else (0, ""),
    )
    rep_disc = diagnostics.run_system_health_checks(cfg)
    audio_disc = next(c for c in rep_disc["checks"] if c["section"] == "audio")
    assert audio_disc["status"] == "warning"
    assert "disconnected" in audio_disc["summary"].lower()
    assert "alsa_input.usb-CustomMic" in audio_disc["summary"]
