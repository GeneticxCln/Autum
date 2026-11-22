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
import json
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any, Callable

from autum.bus import EventBus
from autum.config import Config, default_config_path
from autum.events import Event, new_id
from autum.logging import redact_sensitive
from autum.presence.settings_bridge import query_ollama_models
from autum.tools import Capability, PermissionLevel, ToolRegistry

MAX_LOG_ENTRIES = 500


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
        self._recent: deque[dict[str, Any]] = deque(maxlen=100)
        self.total_errors = 0
        self.total_tool_failures = 0
        self.last_problem: dict[str, Any] | None = None
        self._ensure_file()
        self._load_existing()

    def _ensure_file(self) -> None:
        self.path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
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
        if not self.path.is_file():
            return
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines[-100:]:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    self._recent.append(item)
                    self.last_problem = item
            except ValueError:
                continue

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

        self._recent.append(entry)
        self.last_problem = entry
        self._append_to_disk(entry)
        return entry

    def _append_to_disk(self, entry: dict[str, Any]) -> None:
        self._ensure_file()
        line = json.dumps(entry, ensure_ascii=False) + "\n"
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
        try:
            os.fchmod(fd, 0o600)
            with os.fdopen(fd, "a", encoding="utf-8") as fh:
                fh.write(line)
        except Exception:
            try:
                os.close(fd)
            except OSError:
                pass

    def recent_problems(self, limit: int = 20) -> list[dict[str, Any]]:
        self._load_from_disk_if_updated()
        n = max(1, min(100, int(limit)))
        items = list(self._recent)[-n:]
        items.reverse()
        return items

    def _load_from_disk_if_updated(self) -> None:
        if not self.path.is_file():
            self._recent.clear()
            self.last_problem = None
            return
        try:
            lines = self.path.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return
        parsed: list[dict[str, Any]] = []
        for line in lines[-100:]:
            try:
                item = json.loads(line)
                if isinstance(item, dict):
                    parsed.append(item)
            except ValueError:
                continue
        self._recent.clear()
        self._recent.extend(parsed)
        self.last_problem = parsed[-1] if parsed else None

    def clear(self) -> int:
        count = len(self._recent)
        self._recent.clear()
        self.last_problem = None
        self._ensure_file()
        fd = os.open(self.path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
        finally:
            os.close(fd)
        return count


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
    default_src = "unknown"
    if shutil.which("pactl"):
        rc, out = _run_cmd(["pactl", "get-default-source"])
        if rc == 0 and out:
            default_src = out.splitlines()[0].strip()
    vol_out = ""
    if shutil.which("wpctl"):
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

    # 3. Desktop Eyes & Hands (Wayland / KWin / Desktop file)
    wayland = os.environ.get("WAYLAND_DISPLAY", "")
    desktop_env = os.environ.get("XDG_CURRENT_DESKTOP", "Linux")
    desktop_file = Path.home() / ".local" / "share" / "applications" / "autum.desktop"
    has_desktop_file = desktop_file.is_file() or Path("/usr/share/applications/autum.desktop").is_file()
    if not has_desktop_file:
        checks.append(
            {
                "section": "desktop",
                "title": "Desktop & Global Shortcut",
                "status": "warning",
                "summary": f"{desktop_env} ({'Wayland' if wayland else 'X11'}) active; autum.desktop not installed for KDE shortcut portal",
                "details": "Hold-to-talk on the orb and Hands-Free mode work normally; KDE GlobalShortcuts needs autum.desktop.",
                "suggestion": "Run: cp contrib/autum.desktop ~/.local/share/applications/autum.desktop",
            }
        )
    else:
        checks.append(
            {
                "section": "desktop",
                "title": "Desktop & Global Shortcut",
                "status": "ok",
                "summary": f"{desktop_env} ({'Wayland' if wayland else 'X11'}) • Eyes & Hands enabled • autum.desktop present",
                "details": f"Integrations: {', '.join(cfg.enabled_integrations) or 'none'}",
                "suggestion": "",
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
        """Compact per-turn self-awareness summary injected into Autum's system prompt."""
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
        last_err = ""
        if self.log_store.last_problem is not None:
            lp = self.log_store.last_problem
            last_err = f" Last logged problem: [{lp.get('component')}] {lp.get('summary')}."
        return (
            f"Self-awareness: Your live configuration is Brain/Chat={chat_m}, Vision={vis_m}, "
            f"Voice={voice_m}, dynamic_vram_swap={swap}, mode={mode}, mic={mic}, "
            f"allowed_read_dirs={dirs_summary}.{last_err} "
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
        return await diagnostics.diagnose(focus=focus)

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
            "problems": items,
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
            permission_level=PermissionLevel.READ_ONLY,
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
            execute=_exec_list_problems,
            verify=lambda _a, res: asyncio.sleep(0, result=isinstance(res.get("problems"), list)),
            target_summary=lambda _a: "list recent diagnostic problems",
            timeout_s=3.0,
        )
    )
