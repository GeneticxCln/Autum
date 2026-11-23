"""Multi-compositor Linux window controller backends (Phase 3 / v1.1).

Supports:
- KDE Plasma 6 Wayland (`KWinController` via KWin D-Bus scripting)
- Hyprland (`HyprlandController` via `hyprctl`)
- Sway / i3 (`SwayController` via `swaymsg`)
- Generic X11/EWMH (`EwmhX11Controller` via `wmctrl` / `xdotool`)

All backends share the exact same window-resolution rules, snap positions, and
`excluded_windows` privacy protections (plan §5.3) as `KWinController`.
"""

from __future__ import annotations

import asyncio
import difflib
import json
import os
import shutil
import subprocess
from typing import Any, Callable, Sequence

from autum.integrations.kwin_windows import (
    _APP_ALIASES,
    KWinController,
    _is_excluded_window,
    _norm,
    normalize_position,
)

CmdRunner = Callable[[Sequence[str], float], tuple[int, str]]


def _run_cmd(cmd: Sequence[str], timeout_s: float = 3.0) -> tuple[int, str]:
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
        return res.returncode, res.stdout
    except (OSError, subprocess.SubprocessError):
        return 1, ""


def _compute_snap_rect(
    position: str | None,
    area: tuple[int, int, int, int],
    cur: tuple[int, int, int, int],
    x: int | None,
    y: int | None,
    width: int | None,
    height: int | None,
) -> tuple[int, int, int, int]:
    ax, ay, aw, ah = area
    nx, ny, nw, nh = cur
    if position:
        half_w, half_h = round(aw / 2), round(ah / 2)
        if position == "left":
            nx, ny, nw, nh = ax, ay, half_w, ah
        elif position == "right":
            nx, ny, nw, nh = ax + half_w, ay, aw - half_w, ah
        elif position == "top":
            nx, ny, nw, nh = ax, ay, aw, half_h
        elif position == "bottom":
            nx, ny, nw, nh = ax, ay + half_h, aw, ah - half_h
        elif position == "top_left":
            nx, ny, nw, nh = ax, ay, half_w, half_h
        elif position == "top_right":
            nx, ny, nw, nh = ax + half_w, ay, aw - half_w, half_h
        elif position == "bottom_left":
            nx, ny, nw, nh = ax, ay + half_h, half_w, ah - half_h
        elif position == "bottom_right":
            nx, ny, nw, nh = ax + half_w, ay + half_h, aw - half_w, ah - half_h
        elif position == "center":
            if width is not None:
                nw = width
            elif nw >= aw - 40:
                nw = round(aw * 0.72)
            if height is not None:
                nh = height
            elif nh >= ah - 60:
                nh = round(ah * 0.78)
            nx = ax + round((aw - nw) / 2)
            ny = ay + round((ah - nh) / 2)
    if x is not None:
        nx = x
    if y is not None:
        ny = y
    if width is not None:
        nw = width
    if height is not None:
        nh = height
    return int(nx), int(ny), max(100, int(nw)), max(100, int(nh))


class _BaseWindowController:
    backend_name: str = "Unknown"

    def __init__(self, runner: CmdRunner = _run_cmd) -> None:
        self._run = runner

    async def list_windows(self, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
        raise NotImplementedError

    async def active_excluded_window(self, excluded: Sequence[str] = ()) -> str | None:
        if not excluded:
            return None
        for w in await self.list_windows(excluded=()):
            if w.get("active"):
                hay = f"{w.get('title', '')} {w.get('app', '')}".lower()
                for pat in excluded:
                    if pat and pat.lower() in hay:
                        return pat
        return None

    async def resolve_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
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
        for w in windows:
            if str(w["id"]).lower() == needle or _norm(str(w["app"])) == needle_n:
                return w
        for w in windows:
            if needle_n in _norm(str(w["app"])):
                return w
        is_vscode_query = needle_n in ("code", "vs code", "vscode", "visual studio code")
        needle_tokens = set(needle_n.split())
        for w in windows:
            title_n = _norm(str(w["title"]))
            if is_vscode_query and "claude" in title_n and "visual studio" not in title_n:
                continue
            title_tokens = set(title_n.split())
            if len(needle_n) <= 5:
                if needle_tokens and needle_tokens <= title_tokens:
                    return w
            elif needle_n in title_n:
                return w
        aliases = _APP_ALIASES.get(needle_n, ())
        if aliases:
            for w in windows:
                hay = f"{_norm(str(w['app']))} {_norm(str(w['title']))}"
                if any(alias in hay for alias in aliases):
                    return w
        if len(needle_n) >= 8:
            q_tokens = {t for t in needle_n.split() if len(t) >= 4}
            best_score = 0.0
            best_win: dict[str, Any] | None = None
            for w in windows:
                app_n = _norm(str(w["app"]))
                title_n = _norm(str(w["title"]))
                hay_tokens = {t for t in f"{app_n} {title_n}".split() if len(t) >= 4}
                overlap = (
                    len(q_tokens & hay_tokens) / max(len(q_tokens), 1)
                    if len(q_tokens) >= 2 and len(q_tokens & hay_tokens) >= 2
                    else 0.0
                )
                sim = max(
                    difflib.SequenceMatcher(None, needle_n, app_n).ratio(),
                    difflib.SequenceMatcher(None, needle_n, title_n).ratio(),
                )
                score = max(overlap, sim)
                if score > best_score:
                    best_score = score
                    best_win = w
            if best_win is not None and best_score >= 0.70:
                return best_win
        raise LookupError(f"no controllable window matching {query!r}")


class HyprlandController(_BaseWindowController):
    """Window controller for Hyprland Wayland via `hyprctl`."""

    backend_name = "Hyprland"

    def is_available(self) -> bool:
        if not os.environ.get("HYPRLAND_INSTANCE_SIGNATURE") or not shutil.which("hyprctl"):
            return False
        rc, _ = self._run(["hyprctl", "-j", "activeworkspace"], 1.5)
        return rc == 0

    async def list_windows(self, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
        rc, out = await asyncio.to_thread(self._run, ["hyprctl", "-j", "clients"], 2.5)
        if rc != 0 or not out.strip():
            return []
        try:
            clients = json.loads(out)
        except ValueError:
            return []
        rc_a, out_a = await asyncio.to_thread(self._run, ["hyprctl", "-j", "activewindow"], 1.5)
        active_addr = ""
        if rc_a == 0 and out_a.strip():
            try:
                active_addr = str((json.loads(out_a) or {}).get("address") or "")
            except ValueError:
                pass
        windows: list[dict[str, Any]] = []
        for c in clients if isinstance(clients, list) else ():
            if not isinstance(c, dict) or not c.get("mapped", True):
                continue
            title = str(c.get("title") or "")
            app = str(c.get("class") or c.get("initialClass") or "")
            if _is_excluded_window(title, app, excluded):
                continue
            at = c.get("at") or [0, 0]
            size = c.get("size") or [0, 0]
            ws = c.get("workspace") if isinstance(c.get("workspace"), dict) else {}
            addr = str(c.get("address") or "")
            windows.append(
                {
                    "id": addr,
                    "app": app,
                    "title": title[:60],
                    "x": int(at[0]),
                    "y": int(at[1]),
                    "width": int(size[0]),
                    "height": int(size[1]),
                    "minimized": bool(c.get("hidden", False)),
                    "maximized": bool(c.get("fullscreen", 0)),
                    "desktop": int(ws.get("id") or 1),
                    "active": bool(addr and addr == active_addr),
                }
            )
        return windows

    async def focus_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(
            self._run, ["hyprctl", "dispatch", "focuswindow", f"address:{target['id']}"], 2.0
        )
        if rc != 0:
            raise RuntimeError("hyprctl focuswindow failed")
        return {**target, "active": True, "minimized": False}

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
        target = await self.resolve_window(query, excluded)
        addr = f"address:{target['id']}"
        if desktop is not None:
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "movetoworkspacesilent", f"{int(desktop)},{addr}"], 2.0)
        wants_geom = any(v is not None for v in (position, x, y, width, height))
        nx, ny, nw, nh = target["x"], target["y"], target["width"], target["height"]
        if wants_geom:
            rc_m, out_m = await asyncio.to_thread(self._run, ["hyprctl", "-j", "monitors"], 2.0)
            area = (0, 0, 1920, 1080)
            if rc_m == 0 and out_m.strip():
                try:
                    mons = json.loads(out_m)
                    mon = next((m for m in mons if m.get("focused")), mons[0] if mons else None)
                    if isinstance(mon, dict):
                        area = (int(mon.get("x", 0)), int(mon.get("y", 0)), int(mon.get("width", 1920)), int(mon.get("height", 1080)))
                except ValueError:
                    pass
            nx, ny, nw, nh = _compute_snap_rect(position, area, (nx, ny, nw, nh), x, y, width, height)
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "setfloating", addr], 2.0)
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "resizewindowpixel", f"exact {nw} {nh},{addr}"], 2.0)
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "movewindowpixel", f"exact {nx} {ny},{addr}"], 2.0)
        return {
            **target,
            "x": nx, "y": ny, "width": nw, "height": nh,
            "target_x": nx, "target_y": ny, "target_width": nw, "target_height": nh,
            "desktop": int(desktop) if desktop is not None else target["desktop"],
        }

    async def switch_desktop(self, desktop: int) -> dict[str, Any]:
        rc, _ = await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "workspace", str(int(desktop))], 2.0)
        if rc != 0:
            raise RuntimeError(f"hyprctl workspace {desktop} failed")
        return {"desktop": int(desktop), "total": int(desktop)}

    async def set_window_state(self, query: str, action: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        if action not in ("minimize", "maximize", "restore"):
            raise ValueError("action must be 'minimize', 'maximize', or 'restore'")
        target = await self.resolve_window(query, excluded)
        addr = f"address:{target['id']}"
        await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "focuswindow", addr], 2.0)
        if action == "maximize":
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "fullscreen", "1"], 2.0)
        elif action == "restore":
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "fullscreen", "0"], 2.0)
        elif action == "minimize":
            await asyncio.to_thread(self._run, ["hyprctl", "dispatch", "movetoworkspacesilent", f"special:minimized,{addr}"], 2.0)
        return {
            "id": target["id"], "title": target["title"], "app": target["app"],
            "action": action, "minimized": action == "minimize", "maximized": action == "maximize",
        }

    async def close_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(
            self._run, ["hyprctl", "dispatch", "closewindow", f"address:{target['id']}"], 2.0
        )
        if rc != 0:
            raise RuntimeError("hyprctl closewindow failed")
        return {"id": target["id"], "title": target["title"], "app": target["app"], "closed": True}


class SwayController(_BaseWindowController):
    """Window controller for Sway Wayland via `swaymsg`."""

    backend_name = "Sway"

    def is_available(self) -> bool:
        if not os.environ.get("SWAYSOCK") or not shutil.which("swaymsg"):
            return False
        rc, _ = self._run(["swaymsg", "-t", "get_version"], 1.5)
        return rc == 0

    async def list_windows(self, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
        rc, out = await asyncio.to_thread(self._run, ["swaymsg", "-t", "get_tree"], 2.5)
        if rc != 0 or not out.strip():
            return []
        try:
            tree = json.loads(out)
        except ValueError:
            return []
        windows: list[dict[str, Any]] = []

        def _walk(node: dict[str, Any], ws_num: int = 1) -> None:
            if node.get("type") == "workspace" and isinstance(node.get("num"), int):
                ws_num = node["num"]
            if node.get("type") in ("con", "floating_con") and node.get("pid"):
                title = str(node.get("name") or "")
                app = str(node.get("app_id") or (node.get("window_properties") or {}).get("class") or "")
                if not _is_excluded_window(title, app, excluded):
                    rect = node.get("rect") or {}
                    windows.append(
                        {
                            "id": str(node.get("id")),
                            "app": app,
                            "title": title[:60],
                            "x": int(rect.get("x", 0)),
                            "y": int(rect.get("y", 0)),
                            "width": int(rect.get("width", 0)),
                            "height": int(rect.get("height", 0)),
                            "minimized": False,
                            "maximized": bool(node.get("fullscreen_mode", 0)),
                            "desktop": ws_num,
                            "active": bool(node.get("focused", False)),
                        }
                    )
            for child in (*node.get("nodes", ()), *node.get("floating_nodes", ())):
                if isinstance(child, dict):
                    _walk(child, ws_num)

        if isinstance(tree, dict):
            _walk(tree)
        return windows

    async def focus_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(self._run, ["swaymsg", f"[con_id={target['id']}] focus"], 2.0)
        if rc != 0:
            raise RuntimeError("swaymsg focus failed")
        return {**target, "active": True, "minimized": False}

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
        target = await self.resolve_window(query, excluded)
        cid = f"[con_id={target['id']}]"
        if desktop is not None:
            await asyncio.to_thread(self._run, ["swaymsg", f"{cid} move container to workspace number {int(desktop)}"], 2.0)
        nx, ny, nw, nh = target["x"], target["y"], target["width"], target["height"]
        if any(v is not None for v in (position, x, y, width, height)):
            area = (0, 0, 1920, 1080)
            rc_o, out_o = await asyncio.to_thread(self._run, ["swaymsg", "-t", "get_outputs"], 2.0)
            if rc_o == 0 and out_o.strip():
                try:
                    outputs = json.loads(out_o)
                    if isinstance(outputs, list):
                        out_obj = next((o for o in outputs if isinstance(o, dict) and o.get("focused")), None)
                        if out_obj is None:
                            out_obj = next((o for o in outputs if isinstance(o, dict) and o.get("active")), None)
                        rect = (out_obj or {}).get("rect")
                        if isinstance(rect, dict) and int(rect.get("width", 0)) > 0:
                            area = (
                                int(rect.get("x", 0)),
                                int(rect.get("y", 0)),
                                int(rect.get("width", 1920)),
                                int(rect.get("height", 1080)),
                            )
                except (ValueError, TypeError):
                    pass
            nx, ny, nw, nh = _compute_snap_rect(position, area, (nx, ny, nw, nh), x, y, width, height)
            await asyncio.to_thread(
                self._run,
                ["swaymsg", f"{cid} floating enable, resize set {nw} px {nh} px, move position {nx} px {ny} px"],
                2.0,
            )
        return {
            **target,
            "x": nx, "y": ny, "width": nw, "height": nh,
            "target_x": nx, "target_y": ny, "target_width": nw, "target_height": nh,
            "desktop": int(desktop) if desktop is not None else target["desktop"],
        }

    async def switch_desktop(self, desktop: int) -> dict[str, Any]:
        rc, _ = await asyncio.to_thread(self._run, ["swaymsg", f"workspace number {int(desktop)}"], 2.0)
        if rc != 0:
            raise RuntimeError(f"swaymsg workspace {desktop} failed")
        return {"desktop": int(desktop), "total": int(desktop)}

    async def set_window_state(self, query: str, action: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        if action not in ("minimize", "maximize", "restore"):
            raise ValueError("action must be 'minimize', 'maximize', or 'restore'")
        target = await self.resolve_window(query, excluded)
        cid = f"[con_id={target['id']}]"
        if action == "maximize":
            await asyncio.to_thread(self._run, ["swaymsg", f"{cid} fullscreen enable"], 2.0)
        elif action == "restore":
            await asyncio.to_thread(self._run, ["swaymsg", f"{cid} fullscreen disable, floating disable"], 2.0)
        elif action == "minimize":
            await asyncio.to_thread(self._run, ["swaymsg", f"{cid} move scratchpad"], 2.0)
        return {
            "id": target["id"], "title": target["title"], "app": target["app"],
            "action": action, "minimized": action == "minimize", "maximized": action == "maximize",
        }

    async def close_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(self._run, ["swaymsg", f"[con_id={target['id']}] kill"], 2.0)
        if rc != 0:
            raise RuntimeError("swaymsg kill failed")
        return {"id": target["id"], "title": target["title"], "app": target["app"], "closed": True}


class EwmhX11Controller(_BaseWindowController):
    """Window controller for X11 desktops via `wmctrl` (and optional `xdotool`)."""

    backend_name = "X11/EWMH"

    def is_available(self) -> bool:
        if os.environ.get("XDG_SESSION_TYPE", "").lower() != "x11" and not os.environ.get("DISPLAY"):
            return False
        if not shutil.which("wmctrl"):
            return False
        rc, _ = self._run(["wmctrl", "-m"], 1.5)
        return rc == 0

    async def list_windows(self, excluded: Sequence[str] = ()) -> list[dict[str, Any]]:
        rc, out = await asyncio.to_thread(self._run, ["wmctrl", "-lGx"], 2.5)
        if rc != 0:
            return []
        active_int: int | None = None
        rc_a, out_a = await asyncio.to_thread(self._run, ["xprop", "-root", "_NET_ACTIVE_WINDOW"], 1.5)
        if rc_a == 0 and "#" in out_a:
            tok = out_a.rsplit("#", 1)[-1].strip().split(",")[0].strip()
            try:
                active_int = int(tok, 16)
            except ValueError:
                active_int = None
        windows: list[dict[str, Any]] = []
        for line in out.splitlines():
            parts = line.split(None, 8)
            if len(parts) < 9:
                continue
            wid, desk, x, y, w, h, wm_class, _host, title = parts
            try:
                xi, yi, wi, hi = int(x), int(y), int(w), int(h)
            except ValueError:
                continue
            app = wm_class.split(".")[-1]
            if _is_excluded_window(title, app, excluded):
                continue
            try:
                wid_int = int(wid, 16)
            except ValueError:
                wid_int = None
            windows.append(
                {
                    "id": wid,
                    "app": app,
                    "title": title[:60],
                    "x": xi,
                    "y": yi,
                    "width": wi,
                    "height": hi,
                    "minimized": False,
                    "maximized": False,
                    "desktop": max(1, int(desk) + 1) if desk.lstrip("-").isdigit() else 1,
                    "active": bool(active_int is not None and wid_int == active_int),
                }
            )
        return windows

    async def focus_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(self._run, ["wmctrl", "-i", "-a", target["id"]], 2.0)
        if rc != 0:
            raise RuntimeError("wmctrl focus failed")
        return {**target, "active": True, "minimized": False}

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
        target = await self.resolve_window(query, excluded)
        wid = target["id"]
        if desktop is not None:
            await asyncio.to_thread(self._run, ["wmctrl", "-i", "-r", wid, "-t", str(max(0, int(desktop) - 1))], 2.0)
        nx, ny, nw, nh = target["x"], target["y"], target["width"], target["height"]
        if any(v is not None for v in (position, x, y, width, height)):
            area = (0, 0, 1920, 1080)
            rc_d, out_d = await asyncio.to_thread(self._run, ["wmctrl", "-d"], 2.0)
            if rc_d == 0:
                for line in out_d.splitlines():
                    if " * " in line and "WA:" in line:
                        try:
                            wa = line.split("WA:", 1)[1].strip().split()
                            ax, ay = (int(v) for v in wa[0].split(","))
                            aw, ah = (int(v) for v in wa[1].split("x"))
                            if aw > 0 and ah > 0:
                                area = (ax, ay, aw, ah)
                                break
                        except (ValueError, IndexError):
                            pass
            nx, ny, nw, nh = _compute_snap_rect(position, area, (nx, ny, nw, nh), x, y, width, height)
            await asyncio.to_thread(
                self._run, ["wmctrl", "-i", "-r", wid, "-b", "remove,maximized_vert,maximized_horz"], 2.0
            )
            await asyncio.to_thread(self._run, ["wmctrl", "-i", "-r", wid, "-e", f"0,{nx},{ny},{nw},{nh}"], 2.0)
        return {
            **target,
            "x": nx, "y": ny, "width": nw, "height": nh,
            "target_x": nx, "target_y": ny, "target_width": nw, "target_height": nh,
            "desktop": int(desktop) if desktop is not None else target["desktop"],
        }

    async def switch_desktop(self, desktop: int) -> dict[str, Any]:
        rc, _ = await asyncio.to_thread(self._run, ["wmctrl", "-s", str(max(0, int(desktop) - 1))], 2.0)
        if rc != 0:
            raise RuntimeError(f"wmctrl desktop {desktop} failed")
        return {"desktop": int(desktop), "total": int(desktop)}

    async def set_window_state(self, query: str, action: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        if action not in ("minimize", "maximize", "restore"):
            raise ValueError("action must be 'minimize', 'maximize', or 'restore'")
        target = await self.resolve_window(query, excluded)
        wid = target["id"]
        if action == "maximize":
            await asyncio.to_thread(self._run, ["wmctrl", "-i", "-r", wid, "-b", "add,maximized_vert,maximized_horz"], 2.0)
        elif action == "restore":
            await asyncio.to_thread(self._run, ["wmctrl", "-i", "-r", wid, "-b", "remove,maximized_vert,maximized_horz,hidden"], 2.0)
        elif action == "minimize":
            await asyncio.to_thread(self._run, ["wmctrl", "-i", "-r", wid, "-b", "add,hidden"], 2.0)
        return {
            "id": target["id"], "title": target["title"], "app": target["app"],
            "action": action, "minimized": action == "minimize", "maximized": action == "maximize",
        }

    async def close_window(self, query: str, excluded: Sequence[str] = ()) -> dict[str, Any]:
        target = await self.resolve_window(query, excluded)
        rc, _ = await asyncio.to_thread(self._run, ["wmctrl", "-i", "-c", target["id"]], 2.0)
        if rc != 0:
            raise RuntimeError("wmctrl close failed")
        return {"id": target["id"], "title": target["title"], "app": target["app"], "closed": True}


def detect_window_controller() -> Any:
    """Return the active compositor's window controller (KDE KWin, Hyprland, Sway, or X11/EWMH)."""
    kwin = KWinController()
    setattr(kwin, "backend_name", "KDE KWin")
    if kwin.is_available():
        return kwin
    for cls in (HyprlandController, SwayController, EwmhX11Controller):
        ctrl = cls()
        if ctrl.is_available():
            return ctrl
    return kwin
