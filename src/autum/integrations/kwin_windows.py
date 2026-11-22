"""KDE KWin Wayland window controller (WP10 / hands).

Uses KWin's standard D-Bus Scripting interface (`org.kde.KWin /Scripting`)
with a private one-shot D-Bus reply callback so Autum can list, focus, move,
snap/tile, minimize/maximize/restore, move across virtual desktops, and close
normal user windows on KDE Plasma 6 Wayland — with zero extra packages.

Privacy & safety (§5.2 / §5.3):
- Strictly excludes Autum's own overlay and any window matching `excluded_windows`
  (password managers, wallets, pinentry, polkit).
- `close_window` is registered as IRREVERSIBLE so it always requires explicit
  user approval before closing an app window.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import re
import secrets
import tempfile
from pathlib import Path
from typing import Any, Sequence

KWIN_BUS = "org.kde.KWin"
KWIN_SCRIPTING_PATH = "/Scripting"
KWIN_SCRIPTING_IFACE = "org.kde.kwin.Scripting"
KWIN_SCRIPT_IFACE = "org.kde.kwin.Script"

SNAP_POSITIONS: frozenset[str] = frozenset(
    {"left", "right", "top", "bottom", "center", "top_left", "top_right", "bottom_left", "bottom_right"}
)

_POSITION_ALIASES: dict[str, str] = {
    "left half": "left",
    "left_half": "left",
    "left side": "left",
    "right half": "right",
    "right_half": "right",
    "right side": "right",
    "top half": "top",
    "top_half": "top",
    "upper half": "top",
    "bottom half": "bottom",
    "bottom_half": "bottom",
    "lower half": "bottom",
    "middle": "center",
    "centre": "center",
    "upper left": "top_left",
    "upper_left": "top_left",
    "top left": "top_left",
    "upper right": "top_right",
    "upper_right": "top_right",
    "top right": "top_right",
    "lower left": "bottom_left",
    "lower_left": "bottom_left",
    "bottom left": "bottom_left",
    "lower right": "bottom_right",
    "lower_right": "bottom_right",
    "bottom right": "bottom_right",
}


def normalize_position(position: str | None) -> str | None:
    if position is None:
        return None
    clean = position.strip().lower().replace("-", "_")
    mapped = _POSITION_ALIASES.get(clean, _POSITION_ALIASES.get(clean.replace("_", " "), clean))
    if mapped not in SNAP_POSITIONS:
        raise ValueError(f"position must be one of {sorted(SNAP_POSITIONS)}")
    return mapped

_NORM_RE = re.compile(r"[^a-z0-9]+")
_APP_ALIASES: dict[str, tuple[str, ...]] = {
    "browser": ("chrome", "chromium", "firefox", "brave", "vivaldi", "edge", "zen", "librewolf"),
    "web browser": ("chrome", "chromium", "firefox", "brave", "vivaldi", "edge", "zen", "librewolf"),
    "terminal": ("konsole", "kitty", "alacritty", "wezterm", "ghostty", "foot", "xterm"),
    "files": ("dolphin", "nautilus", "thunar", "nemo"),
    "file manager": ("dolphin", "nautilus", "thunar", "nemo"),
    "editor": ("antigravity", "code", "vscode", "zed", "kate"),
    "ide": ("antigravity", "code", "vscode", "zed"),
}


def _norm(text: str) -> str:
    return _NORM_RE.sub(" ", text.lower()).strip()


def _is_excluded_window(title: str, app_class: str, excluded: Sequence[str]) -> bool:
    hay = f"{title} {app_class}".lower()
    if "autum" in app_class.lower() or title.strip().lower() == "autum":
        return True
    return any(pat and pat.lower() in hay for pat in excluded)


def _run_kwin_js_sync(body_js: str, timeout_s: float = 4.0) -> Any:
    """Load, run, collect one JSON reply via D-Bus callback, and unload a KWin script."""
    from gi.repository import Gio, GLib

    bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
    unique = bus.get_unique_name()
    token = secrets.token_hex(6)
    obj_path = f"/autum/kwin_{token}"
    iface_name = "org.autum.KWinReply"
    xml = (
        f'<node><interface name="{iface_name}">'
        '<method name="Reply"><arg type="s" name="data" direction="in"/></method>'
        "</interface></node>"
    )
    node_info = Gio.DBusNodeInfo.new_for_xml(xml)
    ctx = GLib.MainContext.new()
    ctx.push_thread_default()
    loop = GLib.MainLoop.new(ctx, False)
    result: list[str] = []

    def on_call(_conn, _sender, _path, _iface, _method, params, invocation):
        result.append(str(params.unpack()[0]))
        invocation.return_value(None)
        loop.quit()

    reg_id = bus.register_object(obj_path, node_info.interfaces[0], on_call, None, None)
    script = (
        f"function __reply(v) {{ callDBus({json.dumps(unique)}, {json.dumps(obj_path)}, "
        f"{json.dumps(iface_name)}, 'Reply', JSON.stringify(v)); }}\n"
        f"try {{\n{body_js}\n}} catch (e) {{ __reply({{error: String(e)}}); }}\n"
    )
    plugin_name = f"autum_win_{token}"
    with tempfile.NamedTemporaryFile("w", suffix=".js", delete=False, encoding="utf-8") as fh:
        fh.write(script)
        script_path = Path(fh.name)
    timeout_src = GLib.timeout_source_new(int(timeout_s * 1000))
    timeout_src.set_callback(lambda *_: (loop.quit(), False)[1])
    timeout_src.attach(ctx)
    try:
        sid = bus.call_sync(
            KWIN_BUS, KWIN_SCRIPTING_PATH, KWIN_SCRIPTING_IFACE, "loadScript",
            GLib.Variant("(ss)", (str(script_path), plugin_name)),
            None, Gio.DBusCallFlags.NONE, 3000, None,
        ).unpack()[0]
        if int(sid) < 0:
            raise RuntimeError("KWin rejected the window control script")
        bus.call_sync(
            KWIN_BUS, f"/Scripting/Script{sid}", KWIN_SCRIPT_IFACE, "run",
            None, None, Gio.DBusCallFlags.NONE, 3000, None,
        )
        loop.run()
        try:
            bus.call_sync(
                KWIN_BUS, KWIN_SCRIPTING_PATH, KWIN_SCRIPTING_IFACE, "unloadScript",
                GLib.Variant("(s)", (plugin_name,)),
                None, Gio.DBusCallFlags.NONE, 3000, None,
            )
        except Exception:
            pass
    finally:
        timeout_src.destroy()
        bus.unregister_object(reg_id)
        ctx.pop_thread_default()
        script_path.unlink(missing_ok=True)

    if not result:
        raise TimeoutError("KWin did not reply within timeout")
    parsed = json.loads(result[0])
    if isinstance(parsed, dict) and parsed.get("error"):
        raise RuntimeError(str(parsed["error"]))
    return parsed


class KWinController:
    """Async KDE Plasma 6 KWin window manager controller."""

    def is_available(self) -> bool:
        try:
            from gi.repository import Gio

            bus = Gio.bus_get_sync(Gio.BusType.SESSION, None)
            bus.call_sync(
                KWIN_BUS, KWIN_SCRIPTING_PATH, "org.freedesktop.DBus.Peer", "Ping",
                None, None, Gio.DBusCallFlags.NONE, 1000, None,
            )
            return True
        except Exception:
            return False

    async def active_excluded_window(self, excluded: Sequence[str] = ()) -> str | None:
        """Return the matched exclusion pattern if the active window is excluded (§5.3), else None."""
        if not excluded:
            return None
        js = """
var w = workspace.activeWindow;
if (!w) { __reply({title: "", app: ""}); }
else { __reply({title: String(w.caption || ""), app: String(w.resourceClass || "")}); }
"""
        info = await asyncio.to_thread(_run_kwin_js_sync, js)
        hay = f"{info.get('title', '')} {info.get('app', '')}".lower()
        for pat in excluded:
            if pat and pat.lower() in hay:
                return pat
        return None

    async def list_windows(self, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
        js = """
var out = [];
var ws = workspace.windowList();
var activeId = workspace.activeWindow ? String(workspace.activeWindow.internalId) : "";
for (var i = 0; i < ws.length; i++) {
    var w = ws[i];
    if (!w.normalWindow || w.skipTaskbar) continue;
    var g = w.frameGeometry;
    var area = workspace.clientArea(KWin.MaximizeArea, w);
    var isMax = (Math.abs(g.width - area.width) <= 24 && Math.abs(g.height - area.height) <= 48);
    var dnum = (w.desktops && w.desktops.length > 0 && w.desktops[0].x11DesktopNumber) ? w.desktops[0].x11DesktopNumber : 0;
    out.push({
        id: String(w.internalId),
        app: String(w.resourceClass || ""),
        title: String(w.caption || "").slice(0, 60),
        x: Math.round(g.x), y: Math.round(g.y),
        width: Math.round(g.width), height: Math.round(g.height),
        minimized: Boolean(w.minimized),
        maximized: Boolean(isMax),
        desktop: dnum,
        active: String(w.internalId) === activeId
    });
}
__reply(out);
"""
        raw = await asyncio.to_thread(_run_kwin_js_sync, js)
        return [w for w in raw if not _is_excluded_window(w.get("title", ""), w.get("app", ""), excluded)]

    async def _resolve_window(self, query: str, excluded: Sequence[str]) -> dict[str, Any]:
        needle = query.strip().lower()
        needle_n = _norm(query)
        if not needle or not needle_n:
            raise ValueError("window query must be non-empty")
        for pat in excluded:
            if pat and pat.lower() in needle:
                raise PermissionError(f"control of excluded window matching {pat!r} is blocked (plan §5.3)")
        windows = await self.list_windows(excluded)
        if not windows:
            raise LookupError(f"no controllable window matching {query!r}")

        # 1. Exact ID or normalized app name ("google chrome" -> "google-chrome")
        for w in windows:
            if w["id"].lower() == needle or _norm(w["app"]) == needle_n:
                return w
        # 2. Substring match on normalized app or title
        for w in windows:
            if needle_n in _norm(w["app"]) or needle_n in _norm(w["title"]):
                return w
        # 3. Friendly category aliases (browser, terminal, files, editor)
        aliases = _APP_ALIASES.get(needle_n, ())
        if aliases:
            for w in windows:
                hay = f"{_norm(w['app'])} {_norm(w['title'])}"
                if any(alias in hay for alias in aliases):
                    return w
        # 4. Token overlap or fuzzy similarity (handles minor typos/diacritics)
        q_tokens = set(needle_n.split())
        best_score = 0.0
        best_win: dict[str, Any] | None = None
        for w in windows:
            app_n = _norm(w["app"])
            title_n = _norm(w["title"])
            hay_tokens = set(f"{app_n} {title_n}".split())
            overlap = len(q_tokens & hay_tokens) / max(len(q_tokens), 1)
            sim = max(
                difflib.SequenceMatcher(None, needle_n, app_n).ratio(),
                difflib.SequenceMatcher(None, needle_n, title_n).ratio(),
            )
            score = max(overlap, sim)
            if score > best_score:
                best_score = score
                best_win = w
        if best_win is not None and best_score >= 0.45:
            return best_win
        raise LookupError(f"no controllable window matching {query!r}")

    async def focus_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self._resolve_window(query, excluded)
        js = f"""
var ws = workspace.windowList();
var found = null;
for (var i = 0; i < ws.length; i++) {{
    if (String(ws[i].internalId) === {json.dumps(target["id"])}) {{ found = ws[i]; break; }}
}}
if (!found) {{ __reply({{error: "window disappeared"}}); }}
else {{
    found.minimized = false;
    workspace.activeWindow = found;
    __reply({{id: String(found.internalId), title: String(found.caption), app: String(found.resourceClass), active: workspace.activeWindow === found, minimized: Boolean(found.minimized)}});
}}
"""
        return await asyncio.to_thread(_run_kwin_js_sync, js)

    async def move_window(
        self,
        query: str,
        *,
        position: str | None = None,
        x: int | None = None,
        y: int | None = None,
        width: int | None = None,
        height: int | None = None,
        desktop: int | None = None,
        excluded: Sequence[str] = (),
    ) -> dict[str, Any]:
        position = normalize_position(position)
        target = await self._resolve_window(query, excluded)
        spec = {"id": target["id"], "position": position, "x": x, "y": y, "width": width, "height": height, "desktop": desktop}
        js = f"""
var spec = {json.dumps(spec)};
var ws = workspace.windowList();
var w = null;
for (var i = 0; i < ws.length; i++) {{
    if (String(ws[i].internalId) === spec.id) {{ w = ws[i]; break; }}
}}
if (!w) {{ __reply({{error: "window disappeared"}}); }}
else {{
    w.minimized = false;
    if (spec.desktop !== null && workspace.desktops && workspace.desktops.length > 0) {{
        var idx = Math.max(0, Math.min(workspace.desktops.length - 1, spec.desktop - 1));
        w.desktops = [workspace.desktops[idx]];
    }}
    var area = workspace.clientArea(KWin.MaximizeArea, w);
    var nx = w.frameGeometry.x, ny = w.frameGeometry.y, nw = w.frameGeometry.width, nh = w.frameGeometry.height;
    var wantsGeom = Boolean(spec.position || spec.x !== null || spec.y !== null || spec.width !== null || spec.height !== null);
    if (wantsGeom) {{
        w.setMaximize(false, false);
    }}
    if (spec.position) {{
        var halfW = Math.round(area.width / 2), halfH = Math.round(area.height / 2);
        if (spec.position === "left") {{ nx = area.x; ny = area.y; nw = halfW; nh = area.height; }}
        else if (spec.position === "right") {{ nx = area.x + halfW; ny = area.y; nw = area.width - halfW; nh = area.height; }}
        else if (spec.position === "top") {{ nx = area.x; ny = area.y; nw = area.width; nh = halfH; }}
        else if (spec.position === "bottom") {{ nx = area.x; ny = area.y + halfH; nw = area.width; nh = area.height - halfH; }}
        else if (spec.position === "top_left") {{ nx = area.x; ny = area.y; nw = halfW; nh = halfH; }}
        else if (spec.position === "top_right") {{ nx = area.x + halfW; ny = area.y; nw = area.width - halfW; nh = halfH; }}
        else if (spec.position === "bottom_left") {{ nx = area.x; ny = area.y + halfH; nw = halfW; nh = area.height - halfH; }}
        else if (spec.position === "bottom_right") {{ nx = area.x + halfW; ny = area.y + halfH; nw = area.width - halfW; nh = area.height - halfH; }}
        else if (spec.position === "center") {{
            if (nw >= area.width - 40 && spec.width === null) nw = Math.round(area.width * 0.72);
            if (nh >= area.height - 60 && spec.height === null) nh = Math.round(area.height * 0.78);
            nx = area.x + Math.round((area.width - nw) / 2);
            ny = area.y + Math.round((area.height - nh) / 2);
        }}
    }}
    if (spec.x !== null) nx = spec.x;
    if (spec.y !== null) ny = spec.y;
    if (spec.width !== null) nw = spec.width;
    if (spec.height !== null) nh = spec.height;
    if (wantsGeom) {{
        w.frameGeometry = {{x: nx, y: ny, width: nw, height: nh}};
    }}
    if (spec.desktop === null) {{
        workspace.activeWindow = w;
    }}
    var g = w.frameGeometry;
    var dnum = (w.desktops && w.desktops.length > 0 && w.desktops[0].x11DesktopNumber) ? w.desktops[0].x11DesktopNumber : 0;
    __reply({{id: String(w.internalId), title: String(w.caption), app: String(w.resourceClass), x: Math.round(g.x), y: Math.round(g.y), width: Math.round(g.width), height: Math.round(g.height), target_x: Math.round(nx), target_y: Math.round(ny), target_width: Math.round(nw), target_height: Math.round(nh), desktop: dnum}});
}}
"""
        return await asyncio.to_thread(_run_kwin_js_sync, js)

    async def switch_desktop(self, desktop: int) -> dict[str, Any]:
        """Switch KWin's active virtual desktop (1-based)."""
        js = f"""
var ds = workspace.desktops || [];
if (ds.length === 0) {{ __reply({{error: "no virtual desktops available"}}); }}
else {{
    var idx = Math.max(0, Math.min(ds.length - 1, {int(desktop)} - 1));
    workspace.currentDesktop = ds[idx];
    var cur = workspace.currentDesktop ? workspace.currentDesktop.x11DesktopNumber : 0;
    __reply({{desktop: cur, total: ds.length}});
}}
"""
        return await asyncio.to_thread(_run_kwin_js_sync, js)

    async def set_window_state(self, query: str, action: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        if action not in ("minimize", "maximize", "restore"):
            raise ValueError("action must be 'minimize', 'maximize', or 'restore'")
        target = await self._resolve_window(query, excluded)
        js = f"""
var ws = workspace.windowList();
var w = null;
for (var i = 0; i < ws.length; i++) {{
    if (String(ws[i].internalId) === {json.dumps(target["id"])}) {{ w = ws[i]; break; }}
}}
if (!w) {{ __reply({{error: "window disappeared"}}); }}
else {{
    if ({json.dumps(action)} === "minimize") {{ w.minimized = true; }}
    else if ({json.dumps(action)} === "maximize") {{ w.minimized = false; w.setMaximize(true, true); }}
    else {{ w.minimized = false; w.setMaximize(false, false); }}
    var g = w.frameGeometry;
    var area = workspace.clientArea(KWin.MaximizeArea, w);
    var isMax = (Math.abs(g.width - area.width) <= 24 && Math.abs(g.height - area.height) <= 48);
    __reply({{id: String(w.internalId), title: String(w.caption), app: String(w.resourceClass), action: {json.dumps(action)}, minimized: Boolean(w.minimized), maximized: Boolean(isMax)}});
}}
"""
        return await asyncio.to_thread(_run_kwin_js_sync, js)

    async def close_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self._resolve_window(query, excluded)
        js = f"""
var ws = workspace.windowList();
var w = null;
for (var i = 0; i < ws.length; i++) {{
    if (String(ws[i].internalId) === {json.dumps(target["id"])}) {{ w = ws[i]; break; }}
}}
if (!w) {{ __reply({{error: "window disappeared"}}); }}
else {{
    w.closeWindow();
    __reply({{id: String(w.internalId), title: String(w.caption), app: String(w.resourceClass), closed: true}});
}}
"""
        return await asyncio.to_thread(_run_kwin_js_sync, js)
