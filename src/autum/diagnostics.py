"""Self-awareness, live self-diagnosis, and persistent problem logging for Autum.

Gives Autum:
1. Live self-awareness of her active models (Brain/Chat, Vision, Voice), GPU VRAM
   residency, listening mode, microphone state, permissions, and recent errors.
2. Automatic event-bus problem logging to `~/.local/share/autum/diagnostics.jsonl`
   (mode 0600, secret-redacted) whenever a component error or tool failure occurs.
3. Three agent capabilities (`self_diagnose`, `log_problem`, `list_problems`) so
   Autum can inspect her own health, explain what went wrong, and log problems
   for the user or developer to fix.
"""

from __future__ import annotations

import asyncio
from collections import deque
import datetime as dt
import http.client
import json
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
import urllib.error
import urllib.parse
import urllib.request
from typing import Any, Callable

from autum.bus import EventBus
from autum.config import Config, default_config_path
from autum.events import Event, new_id
from autum.logging import redact_sensitive
from autum.tools import Capability, PermissionLevel, ToolRegistry

MAX_LOG_ENTRIES = 500


def query_ollama_models(ollama_url: str = "http://127.0.0.1:11434", timeout_s: float = 2.0) -> list[dict[str, Any]]:
    """Query local Ollama for installed models and their live GPU residency (headless, no Qt dependency)."""
    try:
        parsed = urllib.parse.urlparse(ollama_url)
        port = parsed.port
    except ValueError:
        return []
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return []
    if port is not None and not (1 <= port <= 65535):
        return []
    base = ollama_url.rstrip("/")

    loaded_map: dict[str, str] = {}
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout_s) as resp:
            ps_data = json.loads(resp.read().decode("utf-8", errors="replace"))
        if isinstance(ps_data, dict):
            for item in ps_data.get("models") or []:
                if not isinstance(item, dict):
                    continue
                name = str(item.get("name") or "")
                if name:
                    vram = int(item.get("size_vram") or 0)
                    size = int(item.get("size") or 0)
                    if size > 0 and vram >= size:
                        loaded_map[name] = "100% GPU"
                    elif vram > 0:
                        loaded_map[name] = f"{round(vram / max(size, 1) * 100)}% GPU"
                    else:
                        loaded_map[name] = "loaded"
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        pass

    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout_s) as resp:
            tags_data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError, urllib.error.URLError, http.client.HTTPException):
        return []

    out: list[dict[str, Any]] = []
    if isinstance(tags_data, dict):
        for m in tags_data.get("models") or []:
            if not isinstance(m, dict):
                continue
            name = str(m.get("name") or "")
            if not name:
                continue
            size_bytes = int(m.get("size") or 0)
            details = m.get("details") if isinstance(m.get("details"), dict) else {}
            out.append(
                {
                    "name": name,
                    "size_gb": round(size_bytes / (1024**3), 1),
                    "params": str(details.get("parameter_size") or ""),
                    "loaded": name in loaded_map,
                    "status": loaded_map.get(name, ""),
                }
            )
    out.sort(key=lambda item: (not item["loaded"], item["name"].lower()))
    return out


def _now_iso() -> str:
    return dt.datetime.now().astimezone().isoformat(timespec="seconds")


def _suggest_fix(component: str, message: str) -> str:
    low = f"{component} {message}".lower()
    if "app id is required" in low or "globalshortcuts" in low:
        return "Copy contrib/autum.desktop to ~/.local/share/applications/autum.desktop so KDE GlobalShortcuts registers Autum."
    if "allowed_read_dirs" in low:
        return "Open Right-Click -> Settings -> Privacy, Folders & Desktop and add an Allowed Read Folder (e.g. ~/Documents)."
    if "no audio" in low or "microphone" in low or "pw-record" in low:
        return "Check `pactl get-default-source` / `wpctl status` and verify the microphone is connected and unmuted."
    if "ollama" in low or "connection refused" in low or "11434" in low:
        return "Verify `systemctl status ollama` is active and selected models are pulled (`ollama list`)."
    if "no window matched" in low or "kwin" in low:
        return "Call `list_windows` first or pass the short application name (e.g. 'chrome', 'code')."
    if "timed out" in low:
        return "Check GPU VRAM residency in Settings -> AI Models (`ollama ps`); enable Dynamic VRAM Swap if VRAM is full."
    return ""


class DiagnosticsLog:
    """Append-only, secret-redacted problem log at `~/.local/share/autum/diagnostics.jsonl` (0600)."""

    def __init__(
        self,
        path: Path,
        *,
        state_snapshot: Callable[[], dict[str, Any]] | None = None,
    ) -> None:
        self.path = path
        self._state_snapshot = state_snapshot
        self._lock = threading.RLock()
        self._recent: deque[dict[str, Any]] = deque(maxlen=MAX_LOG_ENTRIES)
        self._last_mtime_ns: int = -1
        self._disk_lines: int = 0
        self.total_errors = 0
        self.total_tool_failures = 0
        self.last_problem: dict[str, Any] | None = None
        with self._lock:
            self._ensure_file()
            self._load_existing()

    def _ensure_file(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed or self.path.parent.name == "autum":
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        if self.path.exists():
            try:
                os.chmod(self.path, 0o600)
            except OSError:
                pass

    def _load_existing(self) -> None:
        with self._lock:
            if not self.path.is_file():
                self._last_mtime_ns = -1
                self._disk_lines = 0
                return
            try:
                st = self.path.stat()
                self._last_mtime_ns = st.st_mtime_ns
                lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
                self._disk_lines = len(lines)
            except OSError:
                return
            self._recent.clear()
            for line in lines[-MAX_LOG_ENTRIES:]:
                try:
                    item = json.loads(line)
                    if isinstance(item, dict):
                        self._recent.append(item)
                        self.last_problem = item
                except ValueError:
                    continue
            if len(lines) > MAX_LOG_ENTRIES:
                self._compact_disk()

    def _compact_disk(self) -> None:
        with self._lock:
            self._ensure_file()
            fd, tmp_name = tempfile.mkstemp(prefix=".diagnostics.", suffix=".tmp", dir=str(self.path.parent))
            tmp = Path(tmp_name)
            fd_owned = True
            try:
                os.fchmod(fd, 0o600)
                fh = os.fdopen(fd, "w", encoding="utf-8")
                fd_owned = False
                with fh:
                    for entry in list(self._recent):
                        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
                tmp.replace(self.path)
                self._disk_lines = len(self._recent)
                self._last_mtime_ns = self.path.stat().st_mtime_ns
            except OSError:
                if fd_owned:
                    try:
                        os.close(fd)
                    except OSError:
                        pass
                tmp.unlink(missing_ok=True)

    def attach_bus(self, bus: EventBus) -> None:
        """Automatically record runtime component errors and failed tool calls."""
        bus.subscribe(
            self._on_bus_event,
            {"runtime.component_error", "agent.tool_failed", "agent.tool_verified"},
        )

    def _on_bus_event(self, event: Event) -> None:
        p = event.payload
        if event.type == "runtime.component_error":
            comp = str(p.get("component") or "runtime")
            err = str(p.get("error") or "unknown error")
            self.total_errors += 1
            self.record_problem(
                component=comp,
                summary=err,
                severity="error",
                source="runtime_event",
                suggestion=_suggest_fix(comp, err),
            )
        elif event.type == "agent.tool_failed":
            tool = str(p.get("tool") or "unknown_tool")
            err = str(p.get("error") or "tool execution failed")
            if tool in ("self_diagnose", "log_problem", "list_problems"):
                return
            self.total_tool_failures += 1
            self.record_problem(
                component=f"tool:{tool}",
                summary=err,
                severity="warning",
                source="tool_failure",
                suggestion=_suggest_fix(tool, err),
            )
        elif event.type == "agent.tool_verified" and not p.get("verified", True):
            tool = str(p.get("tool") or "unknown_tool")
            self.total_tool_failures += 1
            self.record_problem(
                component=f"tool:{tool}",
                summary=f"Post-condition verification returned false for {tool}",
                severity="warning",
                source="tool_verification",
                suggestion=_suggest_fix(tool, "verification failed"),
            )

    def record_problem(
        self,
        component: str,
        summary: str,
        *,
        severity: str = "warning",
        source: str = "agent",
        details: str = "",
        suggestion: str = "",
    ) -> dict[str, Any]:
        sev = severity.lower().strip()
        if sev not in ("info", "warning", "error"):
            sev = "warning"
        clean_comp = redact_sensitive(component.strip() or "general")[:80]
        clean_sum = redact_sensitive(summary.strip() or "unspecified problem")[:400]
        clean_det = redact_sensitive(details.strip())[:800] if details else ""
        clean_sug = redact_sensitive((suggestion or _suggest_fix(clean_comp, clean_sum)).strip())[:400]

        entry: dict[str, Any] = {
            "id": f"prob_{new_id()[:8]}",
            "timestamp": _now_iso(),
            "severity": sev,
            "source": source,
            "component": clean_comp,
            "summary": clean_sum,
        }
        if clean_det:
            entry["details"] = clean_det
        if clean_sug:
            entry["suggestion"] = clean_sug
        if self._state_snapshot is not None:
            try:
                entry["snapshot"] = self._state_snapshot()
            except Exception:
                pass

        with self._lock:
            self._recent.append(entry)
            self.last_problem = entry
            self._append_to_disk(entry)
        return entry

    def _append_to_disk(self, entry: dict[str, Any]) -> None:
        with self._lock:
            self._ensure_file()
            line = json.dumps(entry, ensure_ascii=False) + "\n"
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            fd_owned = True
            try:
                os.fchmod(fd, 0o600)
                fh = os.fdopen(fd, "a", encoding="utf-8")
                fd_owned = False
                with fh:
                    fh.write(line)
                self._disk_lines += 1
                if self._disk_lines > MAX_LOG_ENTRIES:
                    self._compact_disk()
                else:
                    self._last_mtime_ns = self.path.stat().st_mtime_ns
            except Exception:
                if fd_owned:
                    try:
                        os.close(fd)
                    except OSError:
                        pass

    def recent_problems(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            self._load_from_disk_if_updated()
            n = max(1, min(100, int(limit)))
            items = list(self._recent)[-n:]
            items.reverse()
            return items

    def _load_from_disk_if_updated(self) -> None:
        with self._lock:
            if not self.path.is_file():
                self._recent.clear()
                self.last_problem = None
                self._last_mtime_ns = -1
                self._disk_lines = 0
                return
            try:
                mtime = self.path.stat().st_mtime_ns
            except OSError:
                return
            if mtime == self._last_mtime_ns:
                return
            self._load_existing()

    def clear(self) -> int:
        with self._lock:
            count = len(self._recent)
            self._recent.clear()
            self.last_problem = None
            self._disk_lines = 0
            self._ensure_file()
            fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.fchmod(fd, 0o600)
            finally:
                os.close(fd)
            try:
                self._last_mtime_ns = self.path.stat().st_mtime_ns
            except OSError:
                self._last_mtime_ns = -1
            return count


def gpu_vram_gb() -> float | None:
    """Return total GPU VRAM in GiB via nvidia-smi if available, else None."""
    if not shutil.which("nvidia-smi"):
        return None
    rc, out = _run_cmd(["nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits"])
    if rc == 0 and out:
        try:
            mib = float(out.splitlines()[0].strip())
            return round(mib / 1024.0, 2)
        except ValueError:
            pass
    return None


def _compact_problem(entry: dict[str, Any]) -> dict[str, Any]:
    """Strip bulky snapshot objects before sending problem entries into LLM tool context."""
    out = {
        "id": entry.get("id", ""),
        "severity": entry.get("severity", "warning"),
        "component": entry.get("component", "general"),
        "summary": str(entry.get("summary", ""))[:200],
    }
    if entry.get("suggestion"):
        out["suggestion"] = str(entry["suggestion"])[:200]
    return out


def _run_cmd(args: list[str], timeout_s: float = 2.0) -> tuple[int, str]:
    try:
        proc = subprocess.run(
            args,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            check=False,
        )
        out = (proc.stdout or proc.stderr or "").strip()
        return proc.returncode, out
    except (OSError, subprocess.SubprocessError) as exc:
        return 1, str(exc)


def run_system_health_checks(
    cfg: Config | None = None,
    *,
    config_path: Path | None = None,
    problems_log: DiagnosticsLog | None = None,
    live_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Run non-destructive health checks across Ollama/VRAM, PipeWire/mic, KWin/Wayland, and permissions."""
    if cfg is None:
        try:
            cfg = Config.load(config_path)
        except Exception:
            cfg = Config()

    chat_m = cfg.resolved_chat_model()
    vis_m = cfg.resolved_vision_model()
    voice_m = cfg.model
    checks: list[dict[str, Any]] = []

    # 1. Ollama & AI Models
    models = query_ollama_models(cfg.ollama_url, timeout_s=2.0)
    installed_names = {m["name"] for m in models}
    loaded_models = [f"{m['name']} ({m['status']})" for m in models if m.get("loaded")]
    missing_models = [m for m in dict.fromkeys((chat_m, vis_m, voice_m)) if m not in installed_names]
    if not models:
        checks.append(
            {
                "section": "models",
                "title": "Ollama & AI Models",
                "status": "error",
                "summary": f"Cannot reach Ollama or no models found at {cfg.ollama_url}",
                "details": f"Configured Brain={chat_m}, Vision={vis_m}, Voice={voice_m}",
                "suggestion": "Verify `systemctl status ollama` is running.",
            }
        )
    elif missing_models:
        checks.append(
            {
                "section": "models",
                "title": "Ollama & AI Models",
                "status": "error",
                "summary": f"Missing configured model(s): {', '.join(missing_models)}",
                "details": f"Installed ({len(models)}): {', '.join(sorted(installed_names))}",
                "suggestion": f"Pull missing model(s) or pick an installed model in Settings -> AI Models.",
            }
        )
    else:
        vram_info = f"In VRAM: {', '.join(loaded_models)}" if loaded_models else "No models currently warm in VRAM"
        cpu_spilled = [
            f"{m['name']} ({m['status']})"
            for m in models
            if m.get("loaded") and "% GPU" in str(m.get("status", "")) and str(m.get("status", "")) != "100% GPU"
        ]
        total_vram = gpu_vram_gb()
        size_by_name = {m["name"]: float(m.get("size_gb") or 0.0) for m in models}
        chat_gb = size_by_name.get(chat_m, 0.0)
        voice_gb = 0.0 if voice_m == chat_m else size_by_name.get(voice_m, 0.0)
        vram_tight = total_vram is not None and (chat_gb + voice_gb + 2.5) > total_vram
        if cpu_spilled:
            checks.append(
                {
                    "section": "models",
                    "title": "Ollama & AI Models",
                    "status": "warning",
                    "summary": f"Model spilling onto CPU: {', '.join(cpu_spilled)} (causes lag)",
                    "details": f"Brain={chat_m} ({chat_gb:.1f} GB) • Voice={voice_m} ({voice_gb:.1f} GB) • {vram_info}",
                    "suggestion": "Pick a smaller quantization for Brain/Chat (e.g. Q4 ~8 GB) so both chat and voice stay 100% on GPU with desktop headroom.",
                }
            )
        elif vram_tight:
            checks.append(
                {
                    "section": "models",
                    "title": "Ollama & AI Models",
                    "status": "warning",
                    "summary": f"Tight VRAM ({chat_gb:.1f} + {voice_gb:.1f} GB + desktop > {total_vram:.1f} GB GPU)",
                    "details": f"Brain={chat_m} • Vision={vis_m} • Voice={voice_m} • {vram_info}",
                    "suggestion": "Choose a smaller chat model (e.g. gemma4:12b Q4 at 8 GB) to prevent desktop stalls and voice CPU spill.",
                }
            )
        else:
            checks.append(
                {
                    "section": "models",
                    "title": "Ollama & AI Models",
                    "status": "ok",
                    "summary": f"Brain={chat_m} • Vision={vis_m} • Voice={voice_m} (swap={'ON' if cfg.dynamic_swap else 'OFF'})",
                    "details": f"{len(models)} local models installed • {vram_info}",
                    "suggestion": "",
                }
            )

    # 2. Audio & Microphone (PipeWire)
    missing_pw = [t for t in ("pw-record", "pw-play", "wpctl") if not shutil.which(t)]
    configured_in = cfg.audio_input.strip()
    default_src = configured_in or "unknown"
    if not configured_in and shutil.which("pactl"):
        rc, out = _run_cmd(["pactl", "get-default-source"])
        if rc == 0 and out:
            default_src = out.splitlines()[0].strip()
    vol_out = ""
    configured_missing = False
    if configured_in and shutil.which("pactl"):
        rc, out = _run_cmd(["pactl", "get-source-mute", configured_in])
        if rc == 0 and out:
            vol_out = "[MUTED]" if "yes" in out.lower() else "Unmuted"
        else:
            configured_missing = True
    elif shutil.which("wpctl"):
        rc, out = _run_cmd(["wpctl", "get-volume", "@DEFAULT_AUDIO_SOURCE@"])
        if rc == 0 and out:
            vol_out = out.splitlines()[0].strip()

    if missing_pw:
        checks.append(
            {
                "section": "audio",
                "title": "Audio & Microphone",
                "status": "error",
                "summary": f"Missing PipeWire tools: {', '.join(missing_pw)}",
                "details": "",
                "suggestion": "Install pipewire-utils / wireplumber.",
            }
        )
    elif configured_missing:
        checks.append(
            {
                "section": "audio",
                "title": "Audio & Microphone",
                "status": "warning",
                "summary": f"Configured microphone is disconnected: {configured_in}",
                "details": f"Mode: {cfg.mode.value} • AEC: {'ON' if cfg.aec else 'OFF'} • Silence: {cfg.silence_ms} ms",
                "suggestion": "Reconnect the USB microphone or select System Default / another mic in Settings -> Voice & Audio.",
            }
        )
    elif "MUTED" in vol_out.upper():
        checks.append(
            {
                "section": "audio",
                "title": "Audio & Microphone",
                "status": "warning",
                "summary": f"Microphone is currently MUTED ({vol_out})",
                "details": f"Default input source: {default_src}",
                "suggestion": "Unmute your microphone in KDE audio settings or with your hardware mute key.",
            }
        )
    else:
        checks.append(
            {
                "section": "audio",
                "title": "Audio & Microphone",
                "status": "ok",
                "summary": f"Source: {default_src} ({vol_out or 'active'})",
                "details": f"Mode: {cfg.mode.value} • AEC: {'ON' if cfg.aec else 'OFF'} • Silence: {cfg.silence_ms} ms",
                "suggestion": "",
            }
        )

    # 3. Desktop Eyes & Hands (Wayland / X11)
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    x11 = os.environ.get("DISPLAY", "")
    desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "Linux")
    has_display = bool(wayland or x11)
    has_desktop_int = "linux_desktop" in cfg.enabled_integrations
    if not has_desktop_int:
        desk_status = "warning"
        desk_summary = f"{desktop_env} • Desktop integration disabled"
        desk_suggestion = "Enable Linux Desktop Integration in Settings to use window control and screen watch."
    elif not has_display:
        desk_status = "warning"
        desk_summary = f"{desktop_env} (headless) • No Wayland/X11 display detected"
        desk_suggestion = "Run Autum inside a graphical desktop session for window control and screen watch."
    else:
        from autum.integrations.window_backends import detect_window_controller

        wm_ctrl = detect_window_controller()
        wm_name = getattr(wm_ctrl, "backend_name", "KWin") if wm_ctrl.is_available() else "unavailable"
        desk_status = "ok"
        desk_summary = f"{desktop_env} ({'Wayland' if wayland else 'X11'}) • Window backend: {wm_name}"
        desk_suggestion = ""
    checks.append(
        {
            "section": "desktop",
            "title": "Desktop Integration",
            "status": desk_status,
            "summary": desk_summary,
            "details": f"Integrations: {', '.join(cfg.enabled_integrations) or 'none'}",
            "suggestion": desk_suggestion,
        }
    )

    # 4. Privacy & Folder Permissions
    allowed = cfg.resolved_allowed_read_dirs()
    if not allowed:
        checks.append(
            {
                "section": "permissions",
                "title": "File & Folder Access",
                "status": "warning",
                "summary": "No Allowed Read Folders configured (fail-closed protection active)",
                "details": "Autum will refuse read_document, inspect_image, and list_directory until a folder is added.",
                "suggestion": "Add a folder (such as ~/Documents) under Privacy, Folders & Desktop in Settings.",
            }
        )
    else:
        checks.append(
            {
                "section": "permissions",
                "title": "File & Folder Access",
                "status": "ok",
                "summary": f"{len(allowed)} folder(s) permitted: {', '.join(str(p) for p in allowed)}",
                "details": f"Reversible approval: {'ON' if cfg.require_reversible_approval else 'OFF'} • Excluded windows: {len(cfg.excluded_windows)}",
                "suggestion": "",
            }
        )

    recent = problems_log.recent_problems(limit=15) if problems_log is not None else []
    overall = "ok"
    if any(c["status"] == "error" for c in checks):
        overall = "error"
    elif any(c["status"] == "warning" for c in checks) or recent:
        overall = "warning"

    return {
        "timestamp": _now_iso(),
        "overall_status": overall,
        "live_state": live_state or {},
        "checks": checks,
        "recent_problems": recent,
    }


class SelfDiagnostics:
    """Bridges live runtime awareness, health checks, and problem logging into Autum."""

    def __init__(
        self,
        log_store: DiagnosticsLog,
        *,
        get_config: Callable[[], Config],
        get_live_state: Callable[[], dict[str, Any]],
    ) -> None:
        self.log_store = log_store
        self._get_config = get_config
        self._get_live_state = get_live_state

    def awareness_prompt(self) -> str:
        """Compact per-turn self-awareness summary injected into Autum's system prompt.

        Security note: never injects free-form text from diagnostics.jsonl into the
        system prompt (only integer problem counts), so logged problem strings can
        never become standing instructions.
        """
        cfg = self._get_config()
        st = self._get_live_state()
        chat_m = st.get("chat_model") or cfg.resolved_chat_model()
        vis_m = st.get("vision_model") or cfg.resolved_vision_model()
        voice_m = st.get("voice_model") or cfg.model
        mode = st.get("mode") or cfg.mode.value
        mic = st.get("microphone") or "off"
        swap = "on" if st.get("dynamic_swap", cfg.dynamic_swap) else "off"
        dirs = st.get("allowed_read_dirs") or [str(p) for p in cfg.resolved_allowed_read_dirs()]
        dirs_summary = ", ".join(dirs) if dirs else "none (fail-closed until user adds a folder in Settings)"
        prob_count = len(self.log_store.recent_problems(limit=50))
        prob_note = f" logged_problems={prob_count}." if prob_count else ""
        # Keep system prompt stable across turns (avoid volatile fields like mic=open/off
        # that would bust Ollama's 2k-token tool KV cache on every push-to-talk turn).
        return (
            f"Self-awareness: Your live configuration is Brain/Chat={chat_m}, Vision={vis_m}, "
            f"Voice={voice_m}, dynamic_vram_swap={swap}, mode={mode}, "
            f"allowed_read_dirs={dirs_summary}.{prob_note} "
            "You have self_diagnose, log_problem, and list_problems tools: call self_diagnose when asked "
            "about your models, health, audio, or why something failed, and call log_problem whenever you "
            "encounter or confirm a bug so it is saved to diagnostics.jsonl for fixing."
        )

    async def diagnose(self, focus: str = "all") -> dict[str, Any]:
        cfg = self._get_config()
        live = self._get_live_state()
        report = await asyncio.to_thread(
            run_system_health_checks,
            cfg,
            problems_log=self.log_store,
            live_state=live,
        )
        if focus and focus != "all":
            if focus == "problems":
                report["checks"] = []
            else:
                report["checks"] = [c for c in report["checks"] if c["section"] == focus]
        return report


def register_diagnostic_capabilities(
    registry: ToolRegistry,
    diagnostics: SelfDiagnostics,
) -> None:
    """Register `self_diagnose`, `log_problem`, and `list_problems` into the capability registry."""

    async def _exec_self_diagnose(args: dict[str, Any]) -> dict[str, Any]:
        focus = str(args.get("focus") or "all").strip().lower()
        rep = await diagnostics.diagnose(focus=focus)
        # Keep tool output compact so it never overflows 4,096-token local chat models
        return {
            "overall_status": rep.get("overall_status", "ok"),
            "checks": [
                {
                    "section": c.get("section", ""),
                    "status": c.get("status", "ok"),
                    "summary": str(c.get("summary", ""))[:200],
                    **({"suggestion": str(c["suggestion"])[:200]} if c.get("suggestion") else {}),
                }
                for c in (rep.get("checks") or [])
            ],
            "recent_problems": [_compact_problem(p) for p in (rep.get("recent_problems") or [])[:5]],
        }

    async def _exec_log_problem(args: dict[str, Any]) -> dict[str, Any]:
        entry = diagnostics.log_store.record_problem(
            component=str(args.get("component") or "general"),
            summary=str(args.get("summary") or ""),
            severity=str(args.get("severity") or "warning"),
            source="agent",
            details=str(args.get("details") or ""),
            suggestion=str(args.get("suggestion") or ""),
        )
        return {
            "logged": True,
            "path": str(diagnostics.log_store.path),
            "problem": entry,
        }

    async def _exec_list_problems(args: dict[str, Any]) -> dict[str, Any]:
        limit = int(args.get("limit") or 10)
        items = diagnostics.log_store.recent_problems(limit=limit)
        return {
            "count": len(items),
            "path": str(diagnostics.log_store.path),
            "problems": [_compact_problem(p) for p in items],
        }

    registry.register(
        Capability(
            name="self_diagnose",
            description=(
                "Run a live self-diagnosis of Autum's own systems: active AI models (Brain, Vision, Voice) "
                "and GPU VRAM residency, microphone/PipeWire audio, desktop Eyes & Hands (Wayland/KWin), "
                "allowed read folders, and recent logged problems."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "focus": {
                        "type": "string",
                        "enum": ["all", "models", "audio", "desktop", "permissions", "problems"],
                        "description": "Which subsystem to inspect (default: all).",
                    }
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            permission_level=PermissionLevel.READ_ONLY,
            observed_output=True,
            execute=_exec_self_diagnose,
            verify=lambda _a, res: asyncio.sleep(0, result=isinstance(res, dict) and "overall_status" in res),
            target_summary=lambda a: f"self-diagnosis ({a.get('focus', 'all')})",
            timeout_s=6.0,
        )
    )

    registry.register(
        Capability(
            name="log_problem",
            description=(
                "Log a bug, failure, misbehavior, or user-reported issue to Autum's persistent diagnostics "
                "log (~/.local/share/autum/diagnostics.jsonl) with live state context so it can be fixed."
            ),
            input_schema={
                "type": "object",
                "properties": {
                    "component": {
                        "type": "string",
                        "enum": [
                            "audio",
                            "models",
                            "vision",
                            "desktop_windows",
                            "screen_watch",
                            "tools",
                            "memory",
                            "settings",
                            "general",
                        ],
                        "description": "Subsystem where the problem occurred.",
                    },
                    "summary": {
                        "type": "string",
                        "minLength": 3,
                        "maxLength": 400,
                        "description": "Concise description of the problem or symptom.",
                    },
                    "details": {
                        "type": "string",
                        "maxLength": 800,
                        "description": "Optional error details or what was attempted.",
                    },
                    "suggestion": {
                        "type": "string",
                        "maxLength": 400,
                        "description": "Optional suggested fix or next step.",
                    },
                    "severity": {
                        "type": "string",
                        "enum": ["info", "warning", "error"],
                        "description": "Problem severity (default: warning).",
                    },
                },
                "required": ["component", "summary"],
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            permission_level=PermissionLevel.REVERSIBLE,
            execute=_exec_log_problem,
            verify=lambda _a, res: asyncio.sleep(0, result=bool(res.get("logged"))),
            target_summary=lambda a: f"log problem in {a.get('component', 'general')}: {a.get('summary', '')[:60]}",
            timeout_s=3.0,
        )
    )

    registry.register(
        Capability(
            name="list_problems",
            description="Read recent problems and errors from Autum's diagnostics log (~/.local/share/autum/diagnostics.jsonl).",
            input_schema={
                "type": "object",
                "properties": {
                    "limit": {
                        "type": "integer",
                        "minimum": 1,
                        "maximum": 50,
                        "description": "Maximum number of recent problems to return (default: 10).",
                    }
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object"},
            permission_level=PermissionLevel.READ_ONLY,
            observed_output=True,
            execute=_exec_list_problems,
            verify=lambda _a, res: asyncio.sleep(0, result=isinstance(res.get("problems"), list)),
            target_summary=lambda _a: "list recent diagnostic problems",
            timeout_s=3.0,
        )
    )


class TranscriptLog:
    """Bounded, secret-redacted recent conversation transcript (`transcript.jsonl`, 0600)."""

    def __init__(self, path: Path, *, max_entries: int = 40) -> None:
        self.path = path
        self.max_entries = max(1, int(max_entries))
        self._lock = threading.RLock()
        self._entries: deque[dict[str, Any]] = deque(maxlen=self.max_entries)
        self._load()

    def _load(self) -> None:
        with self._lock:
            self._entries.clear()
            if not self.path.is_file():
                return
            try:
                for line in self.path.read_text(encoding="utf-8", errors="replace").splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        obj = json.loads(line)
                    except ValueError:
                        continue
                    if isinstance(obj, dict) and obj.get("role") and obj.get("text"):
                        self._entries.append(obj)
            except OSError:
                pass

    def _write_locked(self) -> None:
        parent_existed = self.path.parent.exists()
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed or self.path.parent.name == "autum":
            try:
                os.chmod(self.path.parent, 0o700)
            except OSError:
                pass
        fd, tmp_name = tempfile.mkstemp(prefix=".transcript.", suffix=".tmp", dir=str(self.path.parent))
        tmp = Path(tmp_name)
        try:
            os.fchmod(fd, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            tmp.unlink(missing_ok=True)
            raise
        try:
            with fh:
                for item in self._entries:
                    fh.write(json.dumps(item, ensure_ascii=False) + "\n")
            tmp.replace(self.path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise

    def record(self, role: str, text: str) -> dict[str, Any] | None:
        clean = redact_sensitive((text or "").strip())
        if not clean:
            return None
        entry = {
            "timestamp": _now_iso(),
            "role": str(role),
            "text": clean[:1200],
        }
        with self._lock:
            self._entries.append(entry)
            try:
                self._write_locked()
            except OSError:
                pass
        return entry

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        with self._lock:
            self._load()
            cap = max(1, min(self.max_entries, int(limit)))
            return list(self._entries)[-cap:]

    def clear(self) -> int:
        with self._lock:
            self._load()
            count = len(self._entries)
            self._entries.clear()
            try:
                self.path.unlink(missing_ok=True)
            except OSError:
                pass
            return count

    def attach_bus(self, bus: EventBus) -> None:
        def _on_event(event: Event) -> None:
            if event.type == "conversation.user_message":
                txt = str(event.payload.get("text") or "").strip()
                if txt:
                    self.record("user", txt)
            elif event.type == "conversation.response_finished":
                txt = str(event.payload.get("text") or "").strip()
                if txt:
                    self.record("assistant", txt)

        bus.subscribe(_on_event, {"conversation.user_message", "conversation.response_finished"})
