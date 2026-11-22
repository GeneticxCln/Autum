"""M10 (WP10 replaceable environment integrations + WP7 window inspection) tests."""

from __future__ import annotations

import asyncio
import json
import struct
import zlib
from pathlib import Path

from autum.bus import EventBus
from autum.events import Provenance, new_id
from autum.integrations import IntegrationContext, IntegrationManager
from autum.integrations.linux_desktop import (
    LinuxDesktopIntegration,
    find_desktop_entry,
    launch_desktop_app,
)
from autum.runtime import Runtime
from autum.tools import Capability, PermissionLevel, ToolCall, ToolRegistry
from mocks import MockSpeech, Recorder


def run(coro):
    return asyncio.run(coro)


def make_png() -> bytes:
    def chunk(t: bytes, d: bytes) -> bytes:
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)

    raw = b"".join(b"\x00" + b"\x00\xff\x00" * 4 for _ in range(4))
    return (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", struct.pack(">IIBBBBB", 4, 4, 8, 2, 0, 0, 0))
        + chunk(b"IDAT", zlib.compress(raw))
        + chunk(b"IEND", b"")
    )


def test_integration_manager_adds_removes_replaces_and_handles_unavailable_gracefully():
    class DummyIntegration:
        def __init__(self, name: str, available: bool = True, fail: bool = False, cap_name: str = "dummy") -> None:
            self.name = name
            self._available = available
            self._fail = fail
            self._cap = cap_name

        def is_available(self) -> bool:
            return self._available

        def register(self, registry: ToolRegistry, context: IntegrationContext) -> list[str]:
            if self._fail:
                raise RuntimeError("boom")

            async def _ex(_):
                return {"from": self._cap}

            async def _vf(_, r):
                return True

            registry.register(
                Capability(
                    name=self._cap,
                    description="dummy",
                    input_schema={"type": "object", "properties": {}},
                    output_schema={"type": "object"},
                    permission_level=PermissionLevel.READ_ONLY,
                    observed_output=True,
                    execute=_ex,
                    verify=_vf,
                    target_summary=lambda _: "dummy",
                )
            )
            return [self._cap]

    mgr = IntegrationManager()
    mgr.add(DummyIntegration("desktop", cap_name="v1_tool"))
    mgr.add(DummyIntegration("unavailable", available=False, cap_name="never"))
    mgr.add(DummyIntegration("broken", fail=True, cap_name="broken_tool"))

    # Replace desktop integration without touching runtime or agent
    mgr.replace(DummyIntegration("desktop", cap_name="v2_tool"))

    reg = ToolRegistry()
    loaded = mgr.load_enabled(reg, IntegrationContext(), ["desktop", "unavailable", "broken", "missing"])
    assert loaded == ["desktop"]
    assert reg.names() == ["v2_tool"]

    # Removing integration leaves core registry clean
    mgr.remove("desktop")
    reg2 = ToolRegistry()
    assert mgr.load_enabled(reg2, IntegrationContext(), ["desktop"]) == []
    assert reg2.names() == []


def test_linux_desktop_system_info_and_scoped_directory_listing(tmp_path):
    allowed = tmp_path / "workspace"
    allowed.mkdir()
    (allowed / "readme.md").write_text("hi", encoding="utf-8")
    (allowed / ".env").write_text("SECRET=1", encoding="utf-8")
    (allowed / "my_passwords.txt").write_text("nope", encoding="utf-8")
    outside = tmp_path / "outside"
    outside.mkdir()

    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        reg = ToolRegistry()
        LinuxDesktopIntegration().register(reg, IntegrationContext(allowed_read_dirs=(allowed,)))
        op = new_id()
        bus.fence.open(op)

        sys_out = await reg.execute(ToolCall("s1", "system_info", {}), bus=bus, operation_id=op, turn_id="t1")
        dir_ok = await reg.execute(
            ToolCall("d1", "list_directory", {"path": str(allowed)}), bus=bus, operation_id=op, turn_id="t2"
        )
        dir_outside = await reg.execute(
            ToolCall("d2", "list_directory", {"path": str(outside)}), bus=bus, operation_id=op, turn_id="t3"
        )
        return log, sys_out, dir_ok, dir_outside

    log, sys_out, dir_ok, dir_outside = run(scenario())
    assert sys_out.status == "verified" and sys_out.observed is True
    assert sys_out.result["os"] and sys_out.result["home_disk_total_gb"] > 0
    assert dir_ok.status == "verified" and dir_ok.observed is True
    # Secret files (.env, my_passwords.txt) are excluded even inside allowed dirs (§5.2/§5.3)
    assert [e["name"] for e in dir_ok.result["entries"]] == ["readme.md"]
    assert dir_outside.status == "failed" and "allowed_read_dirs" in (dir_outside.error or "")
    # Every integration tool completion carries Provenance.OBSERVED (§5.1)
    assert all(e.provenance is Provenance.OBSERVED for e in log.of("agent.tool_completed"))


def test_window_inspection_respects_excluded_windows_and_toggles_observation_halo():
    async def scenario():
        bus = EventBus("s")
        log = Recorder(bus)
        reg = ToolRegistry()
        seen_obs = []
        rt_box: list[Runtime] = []

        async def fake_capture(timeout_s: float) -> bytes:
            seen_obs.append(rt_box[0].observation)
            return make_png()

        async def fake_vision(data: bytes, mime: str, question: str) -> str:
            assert mime == "image/png" and len(data) > 0
            return "A green editor window showing main.py."

        integ = LinuxDesktopIntegration(capture_window=fake_capture)
        integ.register(
            reg,
            IntegrationContext(
                vision=fake_vision,
                excluded_windows=("keepassxc", "kwallet", "password"),
                set_observation=lambda s: rt_box[0].set_observation(s),
            ),
        )

        class StepInf:
            def __init__(self, steps):
                self.steps = list(steps)
                self.msgs = []

            async def stream(self, messages, *, operation_id: str, tools=None):
                self.msgs.append(list(messages))
                for item in self.steps.pop(0):
                    yield item

        inf = StepInf(
            [
                [ToolCall("w1", "inspect_selected_window", {"question": "What file is open?", "window_hint": "editor"})],
                ["The window shows main.py in an editor."],
            ]
        )
        rt = Runtime(bus, inf, MockSpeech(bus, synth_s=0.0, per_char_s=0.0), tools=reg)
        rt_box.append(rt)

        # Excluded password manager window is blocked before capture (§5.3)
        op = new_id()
        bus.fence.open(op)
        blocked = await reg.execute(
            ToolCall("w0", "inspect_selected_window", {"window_hint": "KeePassXC vault"}),
            bus=bus, operation_id=op, turn_id="t0",
        )

        # Allowed user-selected window runs cleanly and wraps as untrusted observed data
        rt.submit_text("What file is open in my window?")
        await rt.wait_until_settled()
        return log, rt, inf, seen_obs, blocked

    log, rt, inf, seen_obs, blocked = run(scenario())
    assert blocked.status == "failed" and "excluded" in (blocked.error or "")
    assert seen_obs == ["window"]
    assert rt.observation == "none"
    assert [e.payload["source"] for e in log.of("perception.started")] == ["window"]
    assert [e.payload["source"] for e in log.of("perception.completed")] == ["window"]
    tool_msg = [m for m in inf.msgs[-1] if m["role"] == "tool"][0]
    assert "untrusted_observed_data" in json.loads(tool_msg["content"])


def test_open_application_resolves_desktop_entries_and_blocks_shells(tmp_path):
    app_dir = tmp_path / "applications"
    app_dir.mkdir()
    desktop = app_dir / "org.kde.kcalc.desktop"
    desktop.write_text("[Desktop Entry]\nName=KCalc\nExec=kcalc\n", encoding="utf-8")

    assert find_desktop_entry("kcalc", [app_dir]) == desktop
    assert find_desktop_entry("KCalc", [app_dir]) == desktop
    assert find_desktop_entry("../etc/passwd", [app_dir]) is None

    async def scenario():
        bus = EventBus("s")
        reg = ToolRegistry()
        launched = []

        async def fake_launcher(name: str, df: Path | None):
            launched.append((name, df))
            return {"app": name, "desktop_file": df.name if df else None, "launched": True}

        LinuxDesktopIntegration(launcher=fake_launcher, app_dirs=[app_dir]).register(reg, IntegrationContext())
        op = new_id()
        bus.fence.open(op)
        ok = await reg.execute(ToolCall("a1", "open_application", {"app": "KCalc"}), bus=bus, operation_id=op, turn_id="t1")

        # Real launcher blocks dangerous shell binaries
        try:
            await launch_desktop_app("bash", None)
            shell_blocked = False
        except PermissionError:
            shell_blocked = True
        return ok, launched, shell_blocked

    ok, launched, shell_blocked = run(scenario())
    assert ok.status == "verified" and launched == [("KCalc", desktop)]
    assert shell_blocked is True
