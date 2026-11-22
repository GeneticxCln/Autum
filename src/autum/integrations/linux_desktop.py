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
from autum.logging import redact_sensitive
from autum.perception import is_excluded_path
from autum.tools import Capability, PermissionLevel, ToolRegistry

PORTAL_BUS = "org.freedesktop.portal.Desktop"
PORTAL_PATH = "/org/freedesktop/portal/desktop"
SCREENSHOT_IFACE = "org.freedesktop.portal.Screenshot"
REQUEST_IFACE = "org.freedesktop.portal.Request"

BLOCKED_BINARIES: frozenset[str] = frozenset(
    {
        "sh", "bash", "zsh", "fish", "dash", "csh", "tcsh", "ksh",
        "rm", "rmdir", "dd", "mkfs", "fdisk", "parted", "wipefs",
        "sudo", "su", "doas", "pkexec", "chmod", "chown", "kill", "pkill", "killall",
        "python", "python3", "perl", "ruby", "node", "php", "lua",
        "curl", "wget", "nc", "ncat", "ssh", "scp", "rsync", "eval", "exec",
    }
)

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


def _portal_screenshot_sync(timeout_s: float) -> bytes:
    """Ask the XDG Desktop Portal for a user-selected interactive screenshot.

    Returns PNG bytes and immediately deletes the temporary portal file so no
    raw screen capture is left on disk (§5.3).
    """
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    token = f"autum_{secrets.token_hex(8)}"
    unique = bus.get_unique_name().lstrip(":").replace(".", "_")
    req_path = f"{PORTAL_PATH}/request/{unique}/{token}"

    loop = GLib.MainLoop()
    outcome: dict[str, Any] = {}

    def on_response(_conn, _sender, _path, _iface, _signal, params):
        code, results = params.unpack()
        outcome["code"] = int(code)
        outcome["uri"] = results.get("uri", "") if isinstance(results, dict) else ""
        loop.quit()

    sub_id = bus.signal_subscribe(
        PORTAL_BUS, REQUEST_IFACE, "Response", req_path, None, Gio.DBusSignalFlags.NONE, on_response
    )

    def on_timeout():
        outcome["timeout"] = True
        loop.quit()
        return False

    timer_id = GLib.timeout_add(int(timeout_s * 1000), on_timeout)
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
        GLib.source_remove(timer_id)
        bus.signal_unsubscribe(sub_id)

    if outcome.get("timeout"):
        raise TimeoutError("window selection timed out")
    if outcome.get("code", 1) != 0:
        raise RuntimeError("window capture was cancelled by the user")
    uri = outcome.get("uri", "")
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme != "file" or not parsed.path:
        raise RuntimeError("screenshot portal did not return a local file")
    shot_path = Path(urllib.parse.unquote(parsed.path))
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
    return await asyncio.to_thread(_portal_screenshot_sync, timeout_s)


# ----- desktop application discovery & safe launch ----------------------


def desktop_app_dirs() -> list[Path]:
    dirs = [
        Path.home() / ".local" / "share" / "applications",
        Path("/usr/local/share/applications"),
        Path("/usr/share/applications"),
    ]
    return [d for d in dirs if d.is_dir()]


def find_desktop_entry(app_query: str, search_dirs: Sequence[Path] | None = None) -> Path | None:
    """Resolve an app name or desktop ID to an installed `.desktop` file."""
    q = app_query.strip().lower()
    if not q or "/" in q or "\\" in q or ".." in q:
        return None
    dirs = list(search_dirs) if search_dirs is not None else desktop_app_dirs()
    stem_target = q[:-8] if q.endswith(".desktop") else q

    # Pass 1: exact stem match (e.g. "firefox" -> firefox.desktop, "org.kde.dolphin")
    for d in dirs:
        for p in sorted(d.glob("*.desktop")):
            if p.stem.lower() == stem_target or p.stem.lower().endswith(f".{stem_target}"):
                return p

    # Pass 2: exact Name= match inside [Desktop Entry]
    for d in dirs:
        for p in sorted(d.glob("*.desktop")):
            try:
                text = p.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            in_entry = False
            for line in text.splitlines():
                s = line.strip()
                if s.startswith("["):
                    in_entry = s == "[Desktop Entry]"
                elif in_entry and s.startswith("Name="):
                    if s.split("=", 1)[1].strip().lower() == stem_target:
                        return p
                    break
    return None


async def launch_desktop_app(app_name: str, desktop_file: Path | None) -> dict[str, Any]:
    """Launch an installed desktop app cleanly without shell execution."""
    if desktop_file is not None and shutil.which("gtk-launch"):
        proc = await asyncio.create_subprocess_exec(
            "gtk-launch", desktop_file.name,
            stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
            start_new_session=True,
        )
        # gtk-launch exits 0 quickly after handing off; wait briefly to catch immediate failure
        try:
            code = await asyncio.wait_for(proc.wait(), 1.0)
            if code != 0:
                raise RuntimeError(f"gtk-launch exited with status {code}")
        except asyncio.TimeoutError:
            pass
        return {"app": app_name, "desktop_file": desktop_file.name, "launched": True}

    clean = app_name.strip().lower()
    if clean in BLOCKED_BINARIES or not clean.replace("-", "").replace("_", "").isalnum():
        raise PermissionError(f"launching {app_name!r} is not permitted")
    exe = shutil.which(clean)
    if not exe:
        raise FileNotFoundError(f"no installed desktop application found matching {app_name!r}")
    proc = await asyncio.create_subprocess_exec(
        exe, stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL, start_new_session=True,
    )
    await asyncio.sleep(0.1)
    if proc.returncode is not None and proc.returncode != 0:
        raise RuntimeError(f"application {app_name!r} exited immediately with code {proc.returncode}")
    return {"app": app_name, "executable": exe, "pid": proc.pid, "launched": True}


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
    ) -> None:
        self.capture_window = capture_window
        self.launcher = launcher
        self.app_dirs = app_dirs

    def is_available(self) -> bool:
        return platform.system() == "Linux"

    def register(self, registry: ToolRegistry, context: IntegrationContext) -> list[str]:
        registered: list[str] = []

        def _notify(source: str) -> None:
            if context.set_observation is not None:
                context.set_observation(source)

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
            target = Path(str(args["path"])).expanduser().resolve()
            reason = is_excluded_path(target)
            if reason is not None:
                raise PermissionError(reason)
            allowed = [d.expanduser().resolve() for d in context.allowed_read_dirs]
            if not any(target == d or d in target.parents for d in allowed):
                raise PermissionError("directory is outside user-configured allowed_read_dirs (plan §5.2)")
            if not target.is_dir():
                raise NotADirectoryError(f"not a directory: {target}")
            entries: list[dict[str, str]] = []
            for child in sorted(target.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
                if is_excluded_path(child) is not None:
                    continue  # Never enumerate secret stores or sensitive files (§5.2 / §5.3)
                entries.append({"name": child.name, "kind": "dir" if child.is_dir() else "file"})
                if len(entries) >= 100:
                    break
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
                description="Open an installed desktop application by name (for example firefox, dolphin, calculator).",
                input_schema={
                    "type": "object",
                    "properties": {
                        "app": {"type": "string", "minLength": 1, "maxLength": 80, "description": "Installed desktop application name or .desktop ID"},
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

        return registered
