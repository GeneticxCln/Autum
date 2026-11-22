"""Eyes (live ScreenCast watch) and Hands (KWin window control) tests."""

from __future__ import annotations

import asyncio
import json
import time

from autum.bus import EventBus
from autum.events import Provenance, new_id
from autum.integrations import IntegrationContext, IntegrationManager
from autum.integrations.kwin_windows import KWinController, _is_excluded_window
from autum.integrations.linux_desktop import LinuxDesktopIntegration
from autum.runtime import Runtime, State
from autum.tools import ApprovalGate, ToolCall, ToolRegistry
from mocks import MockSpeech, Recorder


def run(coro):
    return asyncio.run(coro)


class FakeScreenWatcher:
    def __init__(self) -> None:
        self.active = False
        self.frame_age_s = 0.05
        self._frame = b"\xff\xd8\xffFAKE_JPEG_DATA"
        self.sources: list[str] = []

    async def start(self, source_type: str = "both") -> dict:
        self.sources.append(source_type)
        self.active = True
        return {"active": True, "node_id": 42, "frame_bytes": len(self._frame)}

    def latest_frame(self) -> bytes:
        if not self.active:
            raise RuntimeError("screen watching is not active; ask me to start watching your screen first")
        return self._frame

    async def stop(self) -> dict:
        self.active = False
        return {"active": False}


class FakeWindowController:
    def __init__(self) -> None:
        self.windows = [
            {"id": "{1}", "title": "Mozilla Firefox", "app": "firefox", "x": 100, "y": 100, "width": 1200, "height": 800, "minimized": False, "desktop": 1, "active": False},
            {"id": "{2}", "title": "Alacritty", "app": "Alacritty", "x": 200, "y": 200, "width": 800, "height": 600, "minimized": False, "desktop": 1, "active": True},
            {"id": "{3}", "title": "KeePassXC - Vault", "app": "keepassxc", "x": 0, "y": 0, "width": 600, "height": 400, "minimized": False, "desktop": 1, "active": False},
            {"id": "{4}", "title": "Autum", "app": "autum", "x": 2200, "y": 1100, "width": 300, "height": 300, "minimized": False, "desktop": 1, "active": False},
        ]

    def is_available(self) -> bool:
        return True

    async def list_windows(self, excluded=()):
        return [w for w in self.windows if not _is_excluded_window(w["title"], w["app"], excluded)]

    def _find(self, query: str, excluded=()):
        q = query.strip().lower()
        for pat in excluded:
            if pat and pat.lower() in q:
                raise PermissionError(f"excluded window matching {pat!r}")
        for w in self.windows:
            if _is_excluded_window(w["title"], w["app"], excluded):
                continue
            if q in w["app"].lower() or q in w["title"].lower() or q == w["id"].lower():
                return w
        raise LookupError(f"no window matching {query!r}")

    async def focus_window(self, query: str, excluded=()):
        w = self._find(query, excluded)
        w["minimized"] = False
        w["active"] = True
        return {"id": w["id"], "title": w["title"], "app": w["app"], "active": True, "minimized": False}

    async def move_window(self, query: str, *, position=None, x=None, y=None, width=None, height=None, desktop=None, excluded=()):
        w = self._find(query, excluded)
        if position == "left":
            w["x"], w["y"], w["width"], w["height"] = 0, 0, 1280, 1400
        if x is not None:
            w["x"] = x
        if y is not None:
            w["y"] = y
        if width is not None:
            w["width"] = width
        if height is not None:
            w["height"] = height
        if desktop is not None:
            w["desktop"] = desktop
        return {"id": w["id"], "title": w["title"], "app": w["app"], "x": w["x"], "y": w["y"], "width": w["width"], "height": w["height"], "desktop": w["desktop"]}

    async def set_window_state(self, query: str, action: str, excluded=()):
        w = self._find(query, excluded)
        w["minimized"] = action == "minimize"
        return {"id": w["id"], "title": w["title"], "app": w["app"], "action": action, "minimized": w["minimized"]}

    async def close_window(self, query: str, excluded=()):
        w = self._find(query, excluded)
        self.windows = [item for item in self.windows if item["id"] != w["id"]]
        return {"id": w["id"], "title": w["title"], "app": w["app"], "closed": True}


class ScriptedInf:
    def __init__(self, steps):
        self.steps = list(steps)
        self.msgs = []

    async def stream(self, messages, *, operation_id: str, tools=None):
        self.msgs.append(list(messages))
        for item in self.steps.pop(0):
            yield item


# ----- Eyes: real-time opt-in screen watch ------------------------------


def test_live_screen_watch_lifecycle_and_observation_halo():
    async def scenario():
        bus = EventBus("eyes")
        log = Recorder(bus)
        watcher = FakeScreenWatcher()
        reg = ToolRegistry()
        rt_box: list[Runtime] = []

        async def fake_vision(data: bytes, mime: str, question: str) -> str:
            assert mime == "image/jpeg" and b"FAKE_JPEG" in data
            return "A browser showing documentation."

        mgr = IntegrationManager()
        mgr.add(LinuxDesktopIntegration(screen_watcher=watcher, window_controller=FakeWindowController()))
        mgr.load_enabled(
            reg,
            IntegrationContext(vision=fake_vision, set_observation=lambda s: rt_box[0].set_observation(s)),
            ("linux_desktop",),
        )
        rt = Runtime(bus, ScriptedInf([]), MockSpeech(bus, synth_s=0.0, per_char_s=0.0), tools=reg)
        rt_box.append(rt)

        op = new_id()
        bus.fence.open(op)

        # Cannot look before opting in
        not_ready = await reg.execute(ToolCall("e0", "look_at_screen", {}), bus=bus, operation_id=op, turn_id="t0")
        assert not_ready.status == "failed" and "start watching" in (not_ready.error or "")

        # Opt in -> observation halo lights up ("screen")
        started = await reg.execute(ToolCall("e1", "start_screen_watch", {"source": "monitor"}), bus=bus, operation_id=op, turn_id="t1")
        obs_while_watching = rt.observation

        # Look instantly at live RAM frame -> observed output
        looked = await reg.execute(ToolCall("e2", "look_at_screen", {"question": "What app is open?"}), bus=bus, operation_id=op, turn_id="t2")

        # Stop watching -> observation halo turns off ("none")
        stopped = await reg.execute(ToolCall("e3", "stop_screen_watch", {}), bus=bus, operation_id=op, turn_id="t3")
        obs_after_stop = rt.observation

        # Verify shutdown cleans up an active watcher too
        await reg.execute(ToolCall("e4", "start_screen_watch", {}), bus=bus, operation_id=op, turn_id="t4")
        assert watcher.active is True
        await mgr.shutdown()
        return log, started, looked, stopped, obs_while_watching, obs_after_stop, watcher.active, rt.observation

    log, started, looked, stopped, obs_on, obs_off, active_after_shutdown, obs_final = run(scenario())
    assert started.status == "verified" and obs_on == "screen"
    assert looked.status == "verified" and looked.observed is True
    assert looked.result["observation"] == "A browser showing documentation."
    assert stopped.status == "verified" and obs_off == "none"
    assert active_after_shutdown is False and obs_final == "none"
    assert [e.payload["source"] for e in log.of("perception.started")] == ["screen", "screen"]
    assert [e.payload["source"] for e in log.of("perception.completed")] == ["screen", "screen"]


# ----- Hands: window control & irreversible close approval --------------


def test_window_control_moves_focuses_tiles_and_blocks_excluded_windows():
    async def scenario():
        bus = EventBus("hands")
        wm = FakeWindowController()
        reg = ToolRegistry()
        LinuxDesktopIntegration(screen_watcher=FakeScreenWatcher(), window_controller=wm).register(
            reg, IntegrationContext()
        )
        op = new_id()
        bus.fence.open(op)

        listed = await reg.execute(ToolCall("h1", "list_windows", {}), bus=bus, operation_id=op, turn_id="t1")
        focused = await reg.execute(ToolCall("h2", "focus_window", {"window": "firefox"}), bus=bus, operation_id=op, turn_id="t2")
        moved = await reg.execute(
            ToolCall("h3", "move_window", {"window": "firefox", "position": "left", "desktop": 2}),
            bus=bus, operation_id=op, turn_id="t3",
        )
        minimized = await reg.execute(
            ToolCall("h4", "window_state", {"window": "alacritty", "action": "minimize"}),
            bus=bus, operation_id=op, turn_id="t4",
        )
        blocked = await reg.execute(
            ToolCall("h5", "focus_window", {"window": "KeePassXC"}),
            bus=bus, operation_id=op, turn_id="t5",
        )
        return listed, focused, moved, minimized, blocked

    listed, focused, moved, minimized, blocked = run(scenario())
    assert listed.status == "verified"
    # KeePassXC and Autum are excluded from listing
    assert [w["app"] for w in listed.result["windows"]] == ["firefox", "Alacritty"]
    assert focused.status == "verified" and focused.result["active"] is True
    assert moved.status == "verified" and (moved.result["x"], moved.result["width"], moved.result["desktop"]) == (0, 1280, 2)
    assert minimized.status == "verified" and minimized.result["minimized"] is True
    assert blocked.status == "failed" and "excluded" in (blocked.error or "")


def test_close_window_is_irreversible_and_requires_explicit_user_approval():
    async def scenario():
        bus = EventBus("close_win")
        log = Recorder(bus)
        wm = FakeWindowController()
        reg = ToolRegistry()
        LinuxDesktopIntegration(screen_watcher=FakeScreenWatcher(), window_controller=wm).register(
            reg, IntegrationContext()
        )
        inf = ScriptedInf(
            [
                [ToolCall("c1", "close_window", {"window": "firefox"})],
                ["I closed Firefox."],
            ]
        )
        rt = Runtime(
            bus, inf, MockSpeech(bus, synth_s=0.0, per_char_s=0.0),
            tools=reg, approval_gate=ApprovalGate(timeout_s=2.0),
        )
        rt.submit_text("Close Firefox.")
        deadline = time.monotonic() + 2.0
        while rt.state is not State.WAITING_FOR_APPROVAL and time.monotonic() < deadline:
            await asyncio.sleep(0.01)
        assert rt.state is State.WAITING_FOR_APPROVAL
        assert any(w["app"] == "firefox" for w in wm.windows)  # still open until approved!
        assert rt.resolve_approval("yes", provenance=Provenance.USER, source="command") == "granted"
        await rt.wait_until_settled()
        return log, wm

    log, wm = run(scenario())
    assert all(w["app"] != "firefox" for w in wm.windows)
    assert log.of("agent.approval_required")[0].payload["action"] == "close_window"
    assert log.of("agent.approval_required")[0].payload["level"] == "irreversible"
    assert log.of("agent.tool_verified")[0].payload == {"tool": "close_window", "verified": True}


def test_live_kwin_list_windows_when_running_on_kde():
    wm = KWinController()
    if not wm.is_available():
        return
    wins = run(wm.list_windows(("keepassxc", "kwallet", "password")))
    assert isinstance(wins, list)
    assert all("autum" not in w["app"].lower() for w in wins)
