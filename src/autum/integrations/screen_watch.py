"""Opt-in real-time screen observation via XDG ScreenCast portal + PipeWire (WP7 / WP10).

Privacy & security guarantees (§5.1, §5.3, §5.4):
- Only starts when the user explicitly requests it and confirms the OS portal
  share dialog once.
- Keeps only the single latest downscaled JPEG frame in RAM (never writes
  screenshots to disk).
- Keeps the runtime `observation` indicator set to `"screen"` (cyan-teal halo
  on the cloud) for the entire duration of active screen watching.
- Stops immediately on `stop_screen_watch` or process shutdown, closing the
  portal session and PipeWire stream and wiping the in-memory frame buffer.
"""

from __future__ import annotations

import asyncio
import os
import secrets
import threading
import time
from typing import Any

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENCAST_IFACE = "org.freedesktop.portal.ScreenCast"
REQUEST_IFACE = "org.freedesktop.portal.Request"
SESSION_IFACE = "org.freedesktop.portal.Session"


def _portal_call_with_response(bus, method: str, args_variant, token: str, timeout_s: float = 45.0) -> tuple[int, dict]:
    """Invoke a ScreenCast portal method and wait for its Request::Response signal."""
    from gi.repository import Gio, GLib

    unique = bus.get_unique_name().lstrip(":").replace(".", "_")
    req_path = f"{PORTAL_PATH}/request/{unique}/{token}"
    ctx = GLib.MainContext.new()
    ctx.push_thread_default()
    loop = GLib.MainLoop.new(ctx, False)
    outcome: dict[str, Any] = {}

    def on_response(_conn, _sender, _path, _iface, _signal, params):
        code, results = params.unpack()
        outcome["code"] = int(code)
        outcome["results"] = results if isinstance(results, dict) else {}
        loop.quit()

    sub_id = bus.signal_subscribe(
        PORTAL_BUS, REQUEST_IFACE, "Response", req_path, None, Gio.DBusSignalFlags.NONE, on_response
    )
    timeout_src = GLib.timeout_source_new(int(timeout_s * 1000))
    timeout_src.set_callback(lambda *_: (outcome.__setitem__("timeout", True), loop.quit(), False)[2])
    timeout_src.attach(ctx)
    try:
        bus.call_sync(
            PORTAL_BUS, PORTAL_PATH, SCREENCAST_IFACE, method, args_variant,
            None, Gio.DBusCallFlags.NONE, 10_000, None,
        )
        loop.run()
    finally:
        timeout_src.destroy()
        bus.signal_unsubscribe(sub_id)
        ctx.pop_thread_default()

    if outcome.get("timeout"):
        raise TimeoutError(f"ScreenCast.{method} timed out waiting for user confirmation")
    return outcome.get("code", 1), outcome.get("results", {})


def _open_screencast_session(source_type: str = "both") -> tuple[str, int, int]:
    """Open an XDG ScreenCast session and return (session_handle, pipewire_fd, node_id)."""
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    sess_token = f"autum_sc_{secrets.token_hex(6)}"
    tok1 = f"autum_req_{secrets.token_hex(6)}"
    code, res = _portal_call_with_response(
        bus,
        "CreateSession",
        GLib.Variant("(a{sv})", ({
            "handle_token": GLib.Variant("s", tok1),
            "session_handle_token": GLib.Variant("s", sess_token),
        },)),
        tok1,
    )
    if code != 0 or not res.get("session_handle"):
        raise RuntimeError("could not create ScreenCast portal session")
    session_handle = str(res["session_handle"])

    # Portal source types bitmask: 1 = MONITOR, 2 = WINDOW
    types_mask = 1 if source_type == "monitor" else 2 if source_type == "window" else 3
    tok2 = f"autum_req_{secrets.token_hex(6)}"
    code, _ = _portal_call_with_response(
        bus,
        "SelectSources",
        GLib.Variant("(oa{sv})", (session_handle, {
            "handle_token": GLib.Variant("s", tok2),
            "types": GLib.Variant("u", types_mask),
            "multiple": GLib.Variant("b", False),
            "cursor_mode": GLib.Variant("u", 2),  # embedded cursor
        })),
        tok2,
    )
    if code != 0:
        _close_portal_session(session_handle)
        raise RuntimeError("screen sharing selection failed")

    tok3 = f"autum_req_{secrets.token_hex(6)}"
    code, res = _portal_call_with_response(
        bus,
        "Start",
        GLib.Variant("(osa{sv})", (session_handle, "", {
            "handle_token": GLib.Variant("s", tok3),
        })),
        tok3,
        timeout_s=60.0,
    )
    if code != 0:
        _close_portal_session(session_handle)
        raise RuntimeError("screen sharing was cancelled by the user")
    streams = res.get("streams") or []
    if not streams:
        _close_portal_session(session_handle)
        raise RuntimeError("screen sharing portal returned no video streams")
    node_id = int(streams[0][0])

    # OpenPipeWireRemote returns a file descriptor via UnixFDList
    fd_list = Gio.UnixFDList()
    reply, out_fds = bus.call_with_unix_fd_list_sync(
        PORTAL_BUS,
        PORTAL_PATH,
        SCREENCAST_IFACE,
        "OpenPipeWireRemote",
        GLib.Variant("(oa{sv})", (session_handle, {})),
        None,
        Gio.DBusCallFlags.NONE,
        10_000,
        fd_list,
        None,
    )
    fd_idx = int(reply.unpack()[0])
    pw_fd = os.dup(out_fds.get(fd_idx))
    return session_handle, pw_fd, node_id


def _close_portal_session(session_handle: str) -> None:
    try:
        from gi.repository import Gio

        bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
        bus.call_sync(
            PORTAL_BUS, session_handle, SESSION_IFACE, "Close", None,
            None, Gio.DBusCallFlags.NONE, 3000, None,
        )
    except Exception:
        pass


class ScreenWatcher:
    """Maintains a single live downscaled JPEG frame in RAM from a PipeWire ScreenCast."""

    def __init__(self, *, max_width: int = 1280, max_height: int = 720, fps: int = 2) -> None:
        self.max_width = max_width
        self.max_height = max_height
        self.fps = fps
        self._lock = threading.Lock()
        self._frame: bytes | None = None
        self._frame_ts: float = 0.0
        self._frame_ready = threading.Event()
        self._session_handle: str | None = None
        self._pw_fd: int | None = None
        self._pipeline = None
        self._active = False

    @property
    def active(self) -> bool:
        return self._active

    @property
    def frame_age_s(self) -> float | None:
        with self._lock:
            return None if self._frame is None else max(0.0, time.monotonic() - self._frame_ts)

    async def start(self, source_type: str = "both") -> dict[str, Any]:
        if self._active:
            return {"active": True, "already_watching": True}
        return await asyncio.to_thread(self._start_sync, source_type)

    def _start_sync(self, source_type: str) -> dict[str, Any]:
        import gi

        gi.require_version("Gst", "1.0")
        from gi.repository import Gst

        Gst.init(None)
        session_handle, pw_fd, node_id = _open_screencast_session(source_type)
        self._session_handle = session_handle
        self._pw_fd = pw_fd
        self._frame_ready.clear()

        desc = (
            f"pipewiresrc fd={pw_fd} path={node_id} do-timestamp=true ! "
            f"videorate ! video/x-raw,framerate={self.fps}/1 ! "
            f"videoconvert ! videoscale ! "
            f"video/x-raw,width=[1,{self.max_width}],height=[1,{self.max_height}] ! "
            f"jpegenc quality=82 ! "
            f"appsink name=sink emit-signals=true max-buffers=1 drop=true sync=false"
        )
        pipeline = Gst.parse_launch(desc)
        sink = pipeline.get_by_name("sink")
        sink.connect("new-sample", self._on_sample)
        ret = pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            pipeline.set_state(Gst.State.NULL)
            self._cleanup_sync()
            raise RuntimeError("failed to start PipeWire GStreamer capture pipeline")

        self._pipeline = pipeline
        self._active = True
        got_first = self._frame_ready.wait(timeout=5.0)
        if not got_first:
            self._cleanup_sync()
            raise RuntimeError("screen sharing started, but no video frames arrived within 5 seconds")
        with self._lock:
            size = len(self._frame or b"")
        return {"active": True, "node_id": node_id, "frame_bytes": size}

    def _on_sample(self, sink):
        from gi.repository import Gst

        sample = sink.emit("pull-sample")
        if sample is None:
            return Gst.FlowReturn.OK
        buf = sample.get_buffer()
        ok, map_info = buf.map(Gst.MapFlags.READ)
        if ok:
            try:
                data = bytes(map_info.data)
            finally:
                buf.unmap(map_info)
            with self._lock:
                self._frame = data
                self._frame_ts = time.monotonic()
            self._frame_ready.set()
        return Gst.FlowReturn.OK

    def latest_frame(self) -> bytes:
        with self._lock:
            if not self._active or not self._frame:
                raise RuntimeError("screen watching is not active; ask me to start watching your screen first")
            return self._frame

    async def stop(self) -> dict[str, Any]:
        await asyncio.to_thread(self._cleanup_sync)
        return {"active": False}

    def _cleanup_sync(self) -> None:
        self._active = False
        pipeline, self._pipeline = self._pipeline, None
        if pipeline is not None:
            try:
                from gi.repository import Gst

                pipeline.set_state(Gst.State.NULL)
            except Exception:
                pass
        pw_fd, self._pw_fd = self._pw_fd, None
        if pw_fd is not None:
            try:
                os.close(pw_fd)
            except OSError:
                pass
        session, self._session_handle = self._session_handle, None
        if session is not None:
            _close_portal_session(session)
        with self._lock:
            self._frame = None
            self._frame_ts = 0.0
        self._frame_ready.clear()
