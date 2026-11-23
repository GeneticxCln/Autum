"""Linux / Freedesktop desktop integration (WP10 / M10).

Provides optional, replaceable desktop capabilities:
- `system_info`: OS, session, uptime, load, memory, and disk usage.
- `list_directory`: read-only listing strictly inside user-configured `allowed_read_dirs`.
- `inspect_selected_window`: user-selected window capture via the XDG Desktop
  Screenshot portal (interactive picker), analyzed locally with vision, with
  password/vault window exclusions (§5.3) and immediate temp-file deletion.
- `open_application`: launches an installed desktop app by `.desktop` ID/name
  (never arbitrary shell commands; REVERSIBLE per §5.2).

All integration outputs are marked `observed_output=True` (untrusted data, §5.1).
"""

from __future__ import annotations

import asyncio
import os
import platform
import secrets
import shutil
import threading
import urllib.parse
from pathlib import Path
from typing import Any, Awaitable, Callable, Sequence

from autum.integrations import IntegrationContext
from autum.integrations.kwin_windows import SNAP_POSITIONS, KWinController, normalize_position
from autum.integrations.screen_watch import ScreenWatcher
from autum.logging import redact_sensitive
from autum.perception import is_excluded_path, resolve_allowed_path
from autum.tools import Capability, PermissionLevel, ToolRegistry

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENSHOT_IFACE = "org.freedesktop.portal.Screenshot"
REQUEST_IFACE = "org.freedesktop.portal.Request"

WindowCapturer = Callable[[float], Awaitable[bytes]]
AppLauncher = Callable[[str, Path | None], Awaitable[dict[str, Any]]]


# ----- system info ------------------------------------------------------


def read_linux_system_info() -> dict[str, Any]:
    os_name = platform.platform()
    os_release = Path("/etc/os-release")
    if os_release.is_file():
        for line in os_release.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("PRETTY_NAME="):
                os_name = line.split("=", 1)[1].strip().strip('"')
                break

    uptime_s: int | None = None
    uptime_path = Path("/proc/uptime")
    if uptime_path.is_file():
        try:
            uptime_s = int(float(uptime_path.read_text().split()[0]))
        except (OSError, ValueError, IndexError):
            pass

    mem_total_mb: int | None = None
    mem_avail_mb: int | None = None
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        vals: dict[str, int] = {}
        for line in meminfo.read_text(encoding="utf-8", errors="replace").splitlines():
            parts = line.split()
            if len(parts) >= 2 and parts[0].rstrip(":") in ("MemTotal", "MemAvailable"):
                try:
                    vals[parts[0].rstrip(":")] = int(parts[1]) // 1024
                except ValueError:
                    pass
        mem_total_mb = vals.get("MemTotal")
        mem_avail_mb = vals.get("MemAvailable")

    disk = os.statvfs(Path.home())
    disk_free_gb = round((disk.f_bavail * disk.f_frsize) / (1024**3), 1)
    disk_total_gb = round((disk.f_blocks * disk.f_frsize) / (1024**3), 1)
    load1, load5, load15 = os.getloadavg()

    return {
        "os": redact_sensitive(os_name),
        "desktop": os.environ.get("XDG_CURRENT_DESKTOP", "unknown"),
        "session_type": os.environ.get("XDG_SESSION_TYPE", "unknown"),
        "uptime_seconds": uptime_s,
        "load_average": [round(load1, 2), round(load5, 2), round(load15, 2)],
        "memory_total_mb": mem_total_mb,
        "memory_available_mb": mem_avail_mb,
        "home_disk_free_gb": disk_free_gb,
        "home_disk_total_gb": disk_total_gb,
    }


# ----- XDG Screenshot portal (interactive user-selected window) ---------


def _portal_screenshot_sync(timeout_s: float, cancel_event: threading.Event | None = None) -> bytes:
    """Ask the XDG Desktop Portal for a user-selected interactive screenshot.

    Returns PNG bytes and immediately deletes the temporary portal file so no
    raw screen capture is left on disk (§5.3).
    """
    from gi.repository import Gio, GLib

    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("window capture was cancelled")
    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    token = f"autum_{secrets.token_hex(8)}"
    unique = bus.get_unique_name().lstrip(":").replace(".", "_")
    req_path = f"{PORTAL_PATH}/request/{unique}/{token}"

    ctx = GLib.MainContext.new()
    ctx.push_thread_default()
    loop = GLib.MainLoop.new(ctx, False)
    outcome: dict[str, Any] = {}

    def _close_req() -> None:
        try:
            bus.call_sync(
                PORTAL_BUS, req_path, REQUEST_IFACE, "Close", None,
                None, Gio.DBusCallFlags.NONE, 2000, None,
            )
        except Exception:
            pass

    def on_response(_conn, _sender, _path, _iface, _signal, params):
        code, results = params.unpack()
        outcome["code"] = int(code)
        outcome["uri"] = results.get("uri", "") if isinstance(results, dict) else ""
        loop.quit()

    sub_id = bus.signal_subscribe(
        PORTAL_BUS, REQUEST_IFACE, "Response", req_path, None, Gio.DBusSignalFlags.NONE, on_response
    )

    def on_timeout(*_):
        outcome["timeout"] = True
        _close_req()
        loop.quit()
        return False

    timeout_src = GLib.timeout_source_new(int(timeout_s * 1000))
    timeout_src.set_callback(on_timeout)
    timeout_src.attach(ctx)

    poll_src = None
    if cancel_event is not None:
        def _check_cancel(*_):
            if cancel_event.is_set():
                outcome["cancelled"] = True
                _close_req()
                loop.quit()
                return False
            return True

        poll_src = GLib.timeout_source_new(100)
        poll_src.set_callback(_check_cancel)
        poll_src.attach(ctx)
    try:
        options = {
            "handle_token": GLib.Variant("s", token),
            "interactive": GLib.Variant("b", True),
            "modal": GLib.Variant("b", True),
        }
        bus.call_sync(
            PORTAL_BUS,
            PORTAL_PATH,
            SCREENSHOT_IFACE,
            "Screenshot",
            GLib.Variant("(sa{sv})", ("", options)),
            None,
            Gio.DBusCallFlags.NONE,
            10_000,
            None,
        )
        loop.run()
    finally:
        timeout_src.destroy()
        if poll_src is not None:
            poll_src.destroy()
        bus.signal_unsubscribe(sub_id)
        ctx.pop_thread_default()

    uri = outcome.get("uri", "")
    parsed = urllib.parse.urlparse(uri)
    shot_path = Path(urllib.parse.unquote(parsed.path)) if (parsed.scheme == "file" and parsed.path) else None
    if outcome.get("cancelled"):
        if shot_path is not None:
            try:
                shot_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError("window capture was cancelled")
    if outcome.get("timeout"):
        if shot_path is not None:
            try:
                shot_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise TimeoutError("window selection timed out")
    if outcome.get("code", 1) != 0:
        if shot_path is not None:
            try:
                shot_path.unlink(missing_ok=True)
            except OSError:
                pass
        raise RuntimeError("window capture was cancelled by the user")
    if shot_path is None:
        raise RuntimeError("screenshot portal did not return a local file")
    try:
        data = shot_path.read_bytes()
    finally:
        try:
            shot_path.unlink(missing_ok=True)
        except OSError:
            pass
    if not data:
        raise RuntimeError("captured window image was empty")
    return data


async def portal_screenshot_interactive(timeout_s: float = 30.0) -> bytes:
    cancel_event = threading.Event()
    try:
        return await asyncio.to_thread(_portal_screenshot_sync, timeout_s, cancel_event)
    except asyncio.CancelledError:
        cancel_event.set()
        raise


# ----- desktop application discovery & safe launch ----------------------


def desktop_app_dirs() -> list[Path]:
    raw: list[Path] = []
    data_home = os.environ.get("XDG_DATA_HOME")
    raw.append((Path(data_home) if data_home else Path.home() / ".local" / "share") / "applications")
    xdg_dirs = os.environ.get("XDG_DATA_DIRS", "/usr/local/share:/usr/share")
    for entry in xdg_dirs.split(":"):
        if entry.strip():
            raw.append(Path(entry.strip()) / "applications")
    for extra in (
        Path.home() / ".local" / "share" / "flatpak" / "exports" / "share" / "applications",
        Path("/var/lib/flatpak/exports/share/applications"),
        Path("/var/lib/snapd/desktop/applications"),
        Path("/usr/local/share/applications"),
        Path("/usr/share/applications"),
    ):
        raw.append(extra)
    seen: set[Path] = set()
    out: list[Path] = []
    for d in raw:
        try:
            resolved = d.resolve()
        except OSError:
            continue
        if resolved not in seen and resolved.is_dir():
            seen.add(resolved)
            out.append(resolved)
    return out


def _desktop_entry_meta(path: Path) -> tuple[bool, str]:
    """Return `(visible, lower_name)` for `[Desktop Entry]`; invisible if Hidden or NoDisplay."""
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return False, ""
    in_entry = False
    name = ""
    for line in text.splitlines():
        s = line.strip()
        if s.startswith("["):
            in_entry = s == "[Desktop Entry]"
        elif in_entry:
            low = s.lower()
            if low in ("hidden=true", "nodisplay=true"):
                return False, ""
            if s.startswith("Name=") and not name:
                name = s.split("=", 1)[1].strip().lower()
    return True, name


def find_desktop_entry(app_query: str, search_dirs: Sequence[Path] | None = None) -> Path | None:
    """Resolve an app name or desktop ID to a visible installed `.desktop` file (respecting XDG masking)."""
    q = app_query.strip().lower()
    if not q or "/" in q or "\\" in q or ".." in q:
        return None
    dirs = list(search_dirs) if search_dirs is not None else desktop_app_dirs()
    stem_target = q[:-8] if q.endswith(".desktop") else q

    # Collect effective entries by desktop ID in XDG priority order so a user-level
    # Hidden=true / NoDisplay=true override masks lower-priority system entries.
    effective: list[tuple[Path, bool, str]] = []
    seen_names: set[str] = set()
    for d in dirs:
        for p in sorted(d.glob("*.desktop")):
            key = p.name.lower()
            if key in seen_names:
                continue
            seen_names.add(key)
            visible, name = _desktop_entry_meta(p)
            effective.append((p, visible, name))

    # Pass 1: exact stem match (e.g. "firefox" -> firefox.desktop, "org.kde.dolphin")
    for p, visible, _ in effective:
        if visible and (p.stem.lower() == stem_target or p.stem.lower().endswith(f".{stem_target}")):
            return p

    # Pass 2: exact Name= match inside [Desktop Entry]
    for p, visible, name in effective:
        if visible and name == stem_target:
            return p
    return None


async def launch_desktop_app(app_name: str, desktop_file: Path | None) -> dict[str, Any]:
    """Launch an installed `.desktop` application cleanly without shell or arbitrary $PATH execution.

    Security (§5.2): never falls back to arbitrary executables on $PATH (such as
    reboot, poweroff, shutdown, halt, shred, or shells). Only verified `.desktop`
    entries installed in standard XDG application directories may be launched.
    """
    if desktop_file is None:
        raise PermissionError(
            f"no installed desktop application (.desktop entry) found matching {app_name!r}; "
            "arbitrary system commands are not permitted (plan §5.2)"
        )
    if not shutil.which("gtk-launch"):
        raise RuntimeError("launching desktop applications requires 'gtk-launch'")
    proc = await asyncio.create_subprocess_exec(
        "gtk-launch", "--", desktop_file.name,
        stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
        start_new_session=True,
    )
    # gtk-launch exits 0 quickly after handing off; wait briefly to catch immediate failure
    try:
        code = await asyncio.wait_for(proc.wait(), 1.0)
        if code != 0:
            raise RuntimeError(f"gtk-launch exited with status {code}")
    except asyncio.TimeoutError:
        asyncio.create_task(proc.wait())  # reap slow launcher cleanly without zombie process
    return {"app": app_name, "desktop_file": desktop_file.name, "launched": True}


# ----- integration implementation ---------------------------------------


class LinuxDesktopIntegration:
    """Replaceable Linux/Freedesktop integration implementing the WP10 protocol."""

    name = "linux_desktop"

    def __init__(
        self,
        *,
        capture_window: WindowCapturer = portal_screenshot_interactive,
        launcher: AppLauncher = launch_desktop_app,
        app_dirs: Sequence[Path] | None = None,
        screen_watcher: Any | None = None,
        window_controller: Any | None = None,
    ) -> None:
        self.capture_window = capture_window
        self.launcher = launcher
        self.app_dirs = app_dirs
        self.screen_watcher = screen_watcher if screen_watcher is not None else ScreenWatcher()
        self.window_controller = window_controller if window_controller is not None else KWinController()
        self._set_obs: Callable[[str], None] | None = None
        self._end_obs: Callable[[str], None] | None = None

    def is_available(self) -> bool:
        return platform.system() == "Linux"

    async def shutdown(self) -> None:
        was_active = bool(getattr(self.screen_watcher, "active", False))
        if hasattr(self.screen_watcher, "stop"):
            await self.screen_watcher.stop()
        if was_active:
            if self._end_obs is not None:
                self._end_obs("screen")
            elif self._set_obs is not None:
                self._set_obs("none")

    def register(self, registry: ToolRegistry, context: IntegrationContext) -> list[str]:
        registered: list[str] = []
        self._set_obs = context.set_observation
        self._end_obs = context.end_observation

        def _notify(source: str) -> None:
            if context.set_observation is not None:
                context.set_observation(source)

        def _end_screen_obs() -> None:
            if context.end_observation is not None:
                context.end_observation("screen")
            elif context.set_observation is not None:
                context.set_observation("none")

        # 1. System info (READ_ONLY, observed)
        async def _system_info(_args: dict[str, Any]) -> dict[str, Any]:
            return read_linux_system_info()

        async def _verify_sys(_args: dict[str, Any], result: dict[str, Any]) -> bool:
            return bool(result.get("os")) and result.get("home_disk_total_gb", 0) > 0

        registry.register(
            Capability(
                name="system_info",
                description="Read local system information (OS, desktop session, uptime, CPU load, RAM, and home disk space).",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                output_schema={"type": "object", "required": ["os", "desktop", "session_type"]},
                permission_level=PermissionLevel.READ_ONLY,
                observed_output=True,
                execute=_system_info,
                verify=_verify_sys,
                target_summary=lambda _: "local system information",
            )
        )
        registered.append("system_info")

        # 2. Scoped directory listing (READ_ONLY, observed, strict allowed_read_dirs)
        async def _list_dir(args: dict[str, Any]) -> dict[str, Any]:
            if not context.allowed_read_dirs:
                raise PermissionError(
                    "list_directory requires allowed_read_dirs to be configured in ~/.config/autum/config.toml (plan §5.2)"
                )
            allowed = [d.expanduser().resolve() for d in context.allowed_read_dirs]
            try:
                target = resolve_allowed_path(str(args["path"]), context.allowed_read_dirs)
            except PermissionError as exc:
                msg = str(exc).replace("path is outside the user-configured allowed read directories", "directory is outside user-configured allowed_read_dirs")
                raise PermissionError(msg) from None
            if not target.is_dir():
                raise NotADirectoryError(f"not a directory: {target}")
            entries: list[dict[str, str]] = []
            for child in sorted(target.iterdir(), key=lambda p: p.name.lower()):
                try:
                    resolved_child = child.resolve()
                except OSError:
                    continue
                if not any(resolved_child == d or d in resolved_child.parents for d in allowed):
                    continue  # Never enumerate or follow symlinks outside allowed_read_dirs (§5.2)
                if is_excluded_path(resolved_child) is not None:
                    continue  # Never enumerate secret stores or sensitive files (§5.2 / §5.3)
                entries.append({"name": child.name, "kind": "dir" if resolved_child.is_dir() else "file"})
                if len(entries) >= 100:
                    break
            entries.sort(key=lambda e: (e["kind"] != "dir", e["name"].lower()))
            return {"path": str(target), "entries": entries, "count": len(entries)}

        async def _verify_list_dir(_args: dict[str, Any], result: dict[str, Any]) -> bool:
            return Path(result["path"]).is_dir() and isinstance(result.get("entries"), list)

        registry.register(
            Capability(
                name="list_directory",
                description="List files and folders inside a user-configured allowed read directory.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "minLength": 1, "maxLength": 512, "description": "Directory path inside allowed_read_dirs"},
                    },
                    "required": ["path"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "required": ["path", "entries", "count"]},
                permission_level=PermissionLevel.READ_ONLY,
                observed_output=True,
                execute=_list_dir,
                verify=_verify_list_dir,
                target_summary=lambda a: f"directory '{a['path']}'",
            )
        )
        registered.append("list_directory")

        # 3. User-selected window inspection (READ_ONLY, observed, triggers window halo)
        async def _inspect_window(args: dict[str, Any]) -> dict[str, Any]:
            if context.vision is None:
                raise RuntimeError("local vision analyzer is not configured")
            hint = str(args.get("window_hint") or "").lower()
            for excluded in context.excluded_windows:
                if excluded and excluded.lower() in hint:
                    raise PermissionError(f"window matching excluded pattern {excluded!r} is blocked (plan §5.3)")
            question = str(args.get("question") or "Describe what is visibly shown in this window concisely.")
            _notify("window")
            try:
                png_bytes = await self.capture_window(30.0)
                observation = await context.vision(png_bytes, "image/png", question)
            finally:
                _notify("none")
            return {
                "source": "window",
                "size_bytes": len(png_bytes),
                "observation": observation,
            }

        async def _verify_window(_args: dict[str, Any], result: dict[str, Any]) -> bool:
            return bool(result.get("observation")) and int(result.get("size_bytes", 0)) > 0

        registry.register(
            Capability(
                name="inspect_selected_window",
                description="Ask the user to pick a window via the desktop screenshot portal and describe what is visibly shown.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1, "maxLength": 400, "description": "What to look for in the selected window"},
                        "window_hint": {"type": "string", "maxLength": 120, "description": "Optional app or window name mentioned by the user"},
                    },
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "required": ["source", "size_bytes", "observation"]},
                permission_level=PermissionLevel.READ_ONLY,
                timeout_s=60.0,
                observed_output=True,
                execute=_inspect_window,
                verify=_verify_window,
                target_summary=lambda _: "user-selected window",
            )
        )
        registered.append("inspect_selected_window")

        # 4. Open installed desktop application (REVERSIBLE per §5.2)
        async def _open_app(args: dict[str, Any]) -> dict[str, Any]:
            app_name = str(args["app"]).strip()
            desktop_file = find_desktop_entry(app_name, self.app_dirs)
            return await self.launcher(app_name, desktop_file)

        async def _verify_open_app(_args: dict[str, Any], result: dict[str, Any]) -> bool:
            return bool(result.get("launched"))

        registry.register(
            Capability(
                name="open_application",
                description="Launch a new desktop app that is not running yet (never use this to move, resize, or focus an already-open window).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "minLength": 1, "maxLength": 80, "description": "App name or .desktop ID"},
                    },
                    "required": ["app"],
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "required": ["app", "launched"]},
                permission_level=PermissionLevel.REVERSIBLE,
                cancellable=False,
                execute=_open_app,
                verify=_verify_open_app,
                target_summary=lambda a: f"open application '{a['app']}'",
            )
        )
        registered.append("open_application")

        # 5. Eyes: opt-in real-time screen watch (REVERSIBLE / READ_ONLY, observed)
        watcher = self.screen_watcher
        wm = self.window_controller

        async def _start_watch(args: dict[str, Any]) -> dict[str, Any]:
            source = str(args.get("source", "both"))
            if source not in ("monitor", "window", "both"):
                raise ValueError("source must be 'monitor', 'window', or 'both'")
            loop = asyncio.get_running_loop()
            started_flag = [False]

            def _mark_started() -> None:
                if not started_flag[0]:
                    started_flag[0] = True
                    _notify("screen")

            if hasattr(watcher, "set_on_started"):
                watcher.set_on_started(lambda: loop.call_soon_threadsafe(_mark_started) if not loop.is_closed() else None)
            if hasattr(watcher, "set_on_ended"):
                watcher.set_on_ended(lambda: loop.call_soon_threadsafe(_end_screen_obs) if not loop.is_closed() else None)
            res = await watcher.start(source)
            _mark_started()
            return res

        async def _verify_start_watch(_args: dict[str, Any], _res: dict[str, Any]) -> bool:
            return bool(watcher.active)

        registry.register(
            Capability(
                name="start_screen_watch",
                description="Start live opt-in screen sharing so Autum can see your desktop in real time until stopped.",
                input_schema={
                    "type": "object",
                    "properties": {
                        "source": {"type": "string", "enum": ["monitor", "window", "both"], "description": "Share target"},
                    },
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "required": ["active"]},
                permission_level=PermissionLevel.REVERSIBLE,
                timeout_s=65.0,
                observed_output=True,
                execute=_start_watch,
                verify=_verify_start_watch,
                target_summary=lambda _: "live screen sharing",
            )
        )
        registered.append("start_screen_watch")

        async def _look_at_screen(args: dict[str, Any]) -> dict[str, Any]:
            if context.vision is None:
                raise RuntimeError("local vision analyzer is not configured")
            # §5.3: refuse live screen vision while an excluded password/wallet window is active (fail-closed)
            if wm is not None and hasattr(wm, "active_excluded_window") and context.excluded_windows:
                try:
                    hit = await wm.active_excluded_window(context.excluded_windows)
                except Exception as exc:
                    raise PermissionError(
                        f"cannot verify active window exclusion before inspecting screen ({exc}); refusing fail-closed (plan §5.3)"
                    ) from None
                if hit is not None:
                    raise PermissionError(
                        f"cannot inspect live screen while excluded sensitive window matching {hit!r} is active (plan §5.3)"
                    )
            frame = watcher.latest_frame()
            question = str(args.get("question") or "Describe what is currently visible on the screen concisely.")
            observation = await context.vision(frame, "image/jpeg", question)
            return {
                "source": "screen",
                "frame_bytes": len(frame),
                "frame_age_s": round(watcher.frame_age_s or 0.0, 2),
                "observation": observation,
            }

        async def _verify_look(_args: dict[str, Any], res: dict[str, Any]) -> bool:
            return bool(res.get("observation")) and int(res.get("frame_bytes", 0)) > 0

        registry.register(
            Capability(
                name="look_at_screen",
                description="Inspect the latest live screen frame (use when screen watching is active to see what is on screen).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "minLength": 1, "maxLength": 400, "description": "What to look for"},
                    },
                    "additionalProperties": False,
                },
                output_schema={"type": "object", "required": ["source", "frame_bytes", "observation"]},
                permission_level=PermissionLevel.READ_ONLY,
                timeout_s=60.0,
                observed_output=True,
                execute=_look_at_screen,
                verify=_verify_look,
                target_summary=lambda _: "live screen frame",
            )
        )
        registered.append("look_at_screen")

        async def _stop_watch(_args: dict[str, Any]) -> dict[str, Any]:
            res = await watcher.stop()
            _end_screen_obs()
            return res

        async def _verify_stop_watch(_args: dict[str, Any], _res: dict[str, Any]) -> bool:
            return not bool(watcher.active)

        registry.register(
            Capability(
                name="stop_screen_watch",
                description="Stop live screen watching and close the share session.",
                input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                output_schema={"type": "object", "required": ["active"]},
                permission_level=PermissionLevel.REVERSIBLE,
                observed_output=True,
                execute=_stop_watch,
                verify=_verify_stop_watch,
                target_summary=lambda _: "stop live screen sharing",
            )
        )
        registered.append("stop_screen_watch")

        # 6. Hands: KDE KWin window control (list, focus, move/tile/desktop, state, close)
        if wm is not None and wm.is_available():
            async def _list_windows(_args: dict[str, Any]) -> dict[str, Any]:
                wins = await wm.list_windows(context.excluded_windows)
                return {"windows": wins, "count": len(wins)}

            async def _verify_list_windows(_args: dict[str, Any], res: dict[str, Any]) -> bool:
                return isinstance(res.get("windows"), list)

            registry.register(
                Capability(
                    name="list_windows",
                    description="List open desktop windows with app name, title, position, size, and virtual desktop.",
                    input_schema={"type": "object", "properties": {}, "additionalProperties": False},
                    output_schema={"type": "object", "required": ["windows", "count"]},
                    permission_level=PermissionLevel.READ_ONLY,
                    observed_output=True,
                    execute=_list_windows,
                    verify=_verify_list_windows,
                    target_summary=lambda _: "open desktop windows",
                )
            )
            registered.append("list_windows")

            async def _focus_window(args: dict[str, Any]) -> dict[str, Any]:
                return await wm.focus_window(str(args["window"]), context.excluded_windows)

            async def _verify_focus(_args: dict[str, Any], res: dict[str, Any]) -> bool:
                return bool(res.get("id")) and not bool(res.get("minimized"))

            registry.register(
                Capability(
                    name="focus_window",
                    description="Focus and bring an already-open window to the front.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "window": {"type": "string", "minLength": 1, "maxLength": 120, "description": "App name, title, or ID"},
                        },
                        "required": ["window"],
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["id", "title", "app"]},
                    permission_level=PermissionLevel.REVERSIBLE,
                    observed_output=True,
                    execute=_focus_window,
                    verify=_verify_focus,
                    target_summary=lambda a: f"focus window '{a['window']}'",
                )
            )
            registered.append("focus_window")

            async def _move_window(args: dict[str, Any]) -> dict[str, Any]:
                return await wm.move_window(
                    str(args["window"]),
                    position=args.get("position"),
                    x=args.get("x"),
                    y=args.get("y"),
                    width=args.get("width"),
                    height=args.get("height"),
                    desktop=args.get("desktop"),
                    excluded=context.excluded_windows,
                )

            async def _verify_move(args: dict[str, Any], res: dict[str, Any]) -> bool:
                wid = res.get("id")
                if not wid:
                    return False
                # Wayland clients apply KWin geometry/desktop configures asynchronously;
                # re-query briefly to verify the resulting window state.
                await asyncio.sleep(0.12)
                try:
                    wins = await wm.list_windows(context.excluded_windows)
                    cur = next((w for w in wins if w.get("id") == wid), None)
                except Exception:
                    cur = None
                check = cur if cur is not None else res
                if args.get("desktop") is not None and check.get("desktop") not in (0, int(args["desktop"])):
                    return False
                if args.get("x") is not None and abs(int(check.get("x", 0)) - int(args["x"])) > 80:
                    return False
                if args.get("y") is not None and abs(int(check.get("y", 0)) - int(args["y"])) > 80:
                    return False
                norm_pos = normalize_position(args.get("position")) if args.get("position") is not None else None
                if "target_x" in res and norm_pos in ("left", "right", "top", "bottom"):
                    if abs(int(check.get("x", 0)) - int(res["target_x"])) > 120 or abs(int(check.get("y", 0)) - int(res["target_y"])) > 120:
                        return False
                return True

            registry.register(
                Capability(
                    name="move_window",
                    description="Move, resize, snap/tile (left, right, top, bottom, center, corners), or send an open window to another virtual desktop.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "window": {"type": "string", "minLength": 1, "maxLength": 120, "description": "App name, title, or ID"},
                            "position": {"type": "string", "enum": sorted(SNAP_POSITIONS), "description": "Snap preset"},
                            "x": {"type": "integer", "minimum": -4096, "maximum": 16384},
                            "y": {"type": "integer", "minimum": -4096, "maximum": 16384},
                            "width": {"type": "integer", "minimum": 100, "maximum": 8192},
                            "height": {"type": "integer", "minimum": 100, "maximum": 8192},
                            "desktop": {"type": "integer", "minimum": 1, "maximum": 20, "description": "Virtual desktop (1-based)"},
                        },
                        "required": ["window"],
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["id", "x", "y", "width", "height"]},
                    permission_level=PermissionLevel.REVERSIBLE,
                    observed_output=True,
                    execute=_move_window,
                    verify=_verify_move,
                    target_summary=lambda a: f"move window '{a['window']}'",
                )
            )
            registered.append("move_window")

            async def _window_state(args: dict[str, Any]) -> dict[str, Any]:
                return await wm.set_window_state(str(args["window"]), str(args["action"]), context.excluded_windows)

            async def _verify_state(args: dict[str, Any], res: dict[str, Any]) -> bool:
                wid = res.get("id")
                if not wid:
                    return False
                await asyncio.sleep(0.10)
                try:
                    wins = await wm.list_windows(context.excluded_windows)
                    cur = next((w for w in wins if w.get("id") == wid), None)
                except Exception:
                    cur = None
                check = cur if cur is not None else res
                if args["action"] == "minimize":
                    return bool(check.get("minimized"))
                if args["action"] == "maximize" and "maximized" in check:
                    return not bool(check.get("minimized")) and bool(check.get("maximized"))
                return not bool(check.get("minimized"))

            registry.register(
                Capability(
                    name="window_state",
                    description="Minimize, maximize, or restore an open window.",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "window": {"type": "string", "minLength": 1, "maxLength": 120, "description": "App name, title, or ID"},
                            "action": {"type": "string", "enum": ["minimize", "maximize", "restore"]},
                        },
                        "required": ["window", "action"],
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["id", "action"]},
                    permission_level=PermissionLevel.REVERSIBLE,
                    observed_output=True,
                    execute=_window_state,
                    verify=_verify_state,
                    target_summary=lambda a: f"{a['action']} window '{a['window']}'",
                )
            )
            registered.append("window_state")

            async def _prepare_close(args: dict[str, Any]) -> tuple[dict[str, Any], str]:
                # Resolve the exact window BEFORE asking for irreversible approval (§5.2)
                # so the spoken confirmation names the real window and closes only that ID.
                if hasattr(wm, "resolve_window"):
                    target = await wm.resolve_window(str(args["window"]), context.excluded_windows)
                    title = str(target.get("title") or target.get("app") or args["window"])
                    app = str(target.get("app") or "")
                    label = f"close window '{title}' ({app})" if app and app.lower() != title.lower() else f"close window '{title}'"
                    return {"window": str(target["id"])}, label
                return args, f"close window '{args['window']}'"

            async def _close_window(args: dict[str, Any]) -> dict[str, Any]:
                return await wm.close_window(str(args["window"]), context.excluded_windows)

            async def _verify_close(_args: dict[str, Any], res: dict[str, Any]) -> bool:
                await asyncio.sleep(0.15)
                remaining = await wm.list_windows(context.excluded_windows)
                return all(w["id"] != res.get("id") for w in remaining)

            registry.register(
                Capability(
                    name="close_window",
                    description="Close an open window (irreversible; requires explicit approval).",
                    input_schema={
                        "type": "object",
                        "properties": {
                            "window": {"type": "string", "minLength": 1, "maxLength": 120, "description": "App name, title, or ID"},
                        },
                        "required": ["window"],
                        "additionalProperties": False,
                    },
                    output_schema={"type": "object", "required": ["id", "closed"]},
                    permission_level=PermissionLevel.IRREVERSIBLE,
                    cancellable=False,
                    observed_output=True,
                    prepare=_prepare_close,
                    execute=_close_window,
                    verify=_verify_close,
                    target_summary=lambda a: f"close window '{a['window']}'",
                )
            )
            registered.append("close_window")

            if hasattr(wm, "switch_desktop"):
                async def _switch_desktop(args: dict[str, Any]) -> dict[str, Any]:
                    return await wm.switch_desktop(int(args["desktop"]))

                async def _verify_switch(args: dict[str, Any], res: dict[str, Any]) -> bool:
                    return int(res.get("desktop", 0)) == int(args["desktop"])

                registry.register(
                    Capability(
                        name="switch_desktop",
                        description="Switch to a different virtual desktop by number (1-based).",
                        input_schema={
                            "type": "object",
                            "properties": {
                                "desktop": {"type": "integer", "minimum": 1, "maximum": 20},
                            },
                            "required": ["desktop"],
                            "additionalProperties": False,
                        },
                        output_schema={"type": "object", "required": ["desktop", "total"]},
                        permission_level=PermissionLevel.REVERSIBLE,
                        observed_output=True,
                        execute=_switch_desktop,
                        verify=_verify_switch,
                        target_summary=lambda a: f"switch to desktop {a['desktop']}",
                    )
                )
                registered.append("switch_desktop")

        return registered
