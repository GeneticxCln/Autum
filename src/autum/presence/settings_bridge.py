"""Settings bridge between Autum's QML Settings app, local config.toml, and Ollama.

Allows the user to pick any locally installed Ollama model for Brain/Chat,
Vision, and Voice, toggle dynamic VRAM swap / hands-free / AEC / approvals /
allowed folders, save cleanly to `~/.config/autum/config.toml` (0600), and
hot-reload the running Autum runtime without restarting.
"""

from __future__ import annotations

import json
import os
import threading
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from autum.config import Config, default_config_path


def _presence_state_path(config_path: Path | None = None) -> Path:
    base = (config_path or default_config_path()).parent
    return base / "presence.json"


def query_ollama_models(ollama_url: str = "http://127.0.0.1:11434", timeout_s: float = 2.0) -> list[dict[str, Any]]:
    """Query local Ollama for installed models and their live GPU residency."""
    parsed = urllib.parse.urlparse(ollama_url)
    if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
        return []
    base = ollama_url.rstrip("/")

    loaded_map: dict[str, str] = {}
    try:
        with urllib.request.urlopen(f"{base}/api/ps", timeout=timeout_s) as resp:
            ps_data = json.loads(resp.read().decode("utf-8", errors="replace"))
        for item in ps_data.get("models", []):
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
    except (OSError, ValueError, urllib.error.URLError):
        pass

    try:
        with urllib.request.urlopen(f"{base}/api/tags", timeout=timeout_s) as resp:
            tags_data = json.loads(resp.read().decode("utf-8", errors="replace"))
    except (OSError, ValueError, urllib.error.URLError):
        return []

    out: list[dict[str, Any]] = []
    for m in tags_data.get("models", []):
        name = str(m.get("name") or "")
        if not name:
            continue
        size_bytes = int(m.get("size") or 0)
        details = m.get("details") or {}
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


class SettingsBridge(QObject):
    """QML-accessible bridge for Autum Settings & persisted orb position."""

    diagnosticsReady = Signal("QVariantMap")
    modelsReady = Signal("QVariantList")

    def __init__(
        self,
        model: Any | None = None,
        *,
        config_path: Path | None = None,
        parent: QObject | None = None,
    ) -> None:
        super().__init__(parent)
        self._model = model
        self._config_path = config_path

    @Slot(result="QVariantMap")
    def loadSettings(self) -> dict[str, Any]:
        try:
            return Config.load(self._config_path).to_dict()
        except Exception as exc:
            fallback = Config().to_dict()
            fallback["_load_error"] = f"Invalid config.toml ({exc}); saving will back up to config.toml.corrupt"
            return fallback

    @Slot(result="QVariantMap")
    def defaultSettings(self) -> dict[str, Any]:
        return Config().to_dict()

    @Slot("QVariantMap", result="QVariantMap")
    def saveAndApply(self, raw: dict[str, Any]) -> dict[str, Any]:
        prev: Config | None = None
        try:
            prev = Config.load(self._config_path)
        except Exception:
            prev = None
        try:
            allowed_dirs = [str(p).strip() for p in (raw.get("allowed_read_dirs") or []) if str(p).strip()]
            excluded = [str(w).strip() for w in (raw.get("excluded_windows") or []) if str(w).strip()]
            integrations = [str(i).strip() for i in (raw.get("enabled_integrations") or []) if str(i).strip()]
            cfg = Config(
                mode=str(raw.get("mode", "push_to_talk")),
                aec=bool(raw.get("aec", False)),
                silence_ms=int(raw.get("silence_ms", 700)),
                barge_in_ms=int(raw.get("barge_in_ms", 256)),
                max_history=int(raw.get("max_history", 20)),
                max_context_chars=int(raw.get("max_context_chars", 10_000)),
                ollama_url=str(raw.get("ollama_url", "http://127.0.0.1:11434")).strip(),
                model=str(raw.get("model", "lfm3-omni:2b")).strip(),
                chat_model=str(raw.get("chat_model", "gemma4:12b")).strip(),
                vision_model=str(raw.get("vision_model", "qwen3-vl:8b")).strip(),
                dynamic_swap=bool(raw.get("dynamic_swap", True)),
                chat_think=bool(raw.get("chat_think", False)),
                voice=str(raw.get("voice", "female-american-english")).strip(),
                log_level=str(raw.get("log_level", "INFO")).strip().upper(),
                data_dir=str(raw.get("data_dir", "")).strip(),
                allowed_read_dirs=tuple(allowed_dirs),
                enabled_integrations=tuple(integrations),
                excluded_windows=tuple(excluded),
                require_reversible_approval=bool(raw.get("require_reversible_approval", False)),
                approval_timeout_s=float(raw.get("approval_timeout_s", 20.0)),
            )
            saved = cfg.save(self._config_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc)}
        restart_fields: list[str] = []
        if prev is not None:
            if cfg.aec != prev.aec:
                restart_fields.append("AEC")
            if cfg.enabled_integrations != prev.enabled_integrations:
                restart_fields.append("integrations")
            if cfg.data_dir != prev.data_dir:
                restart_fields.append("data folder")
        if self._model is not None:
            self._model.reloadConfig()
        return {"ok": True, "path": str(saved), "restart_required": restart_fields}

    @Slot(str)
    def setModeAndPersist(self, mode: str) -> None:
        if mode not in ("push_to_talk", "hands_free"):
            return
        if self._model is not None:
            self._model.setMode(mode)
        try:
            current = Config.load(self._config_path)
            updated = Config(
                mode=mode,
                aec=current.aec,
                silence_ms=current.silence_ms,
                barge_in_ms=current.barge_in_ms,
                max_history=current.max_history,
                max_context_chars=current.max_context_chars,
                ollama_url=current.ollama_url,
                model=current.model,
                chat_model=current.chat_model,
                vision_model=current.vision_model,
                dynamic_swap=current.dynamic_swap,
                chat_think=current.chat_think,
                voice=current.voice,
                log_level=current.log_level,
                data_dir=current.data_dir,
                allowed_read_dirs=current.allowed_read_dirs,
                enabled_integrations=current.enabled_integrations,
                excluded_windows=current.excluded_windows,
                require_reversible_approval=current.require_reversible_approval,
                approval_timeout_s=current.approval_timeout_s,
            )
            updated.save(self._config_path)
        except Exception:
            pass

    @Slot(str, result="QVariantList")
    def listOllamaModels(self, ollama_url: str = "") -> list[dict[str, Any]]:
        url = (ollama_url or "").strip() or Config.load(self._config_path).ollama_url
        return query_ollama_models(url)

    @Slot(str)
    def listOllamaModelsAsync(self, ollama_url: str = "") -> None:
        def _worker() -> None:
            models = self.listOllamaModels(ollama_url)
            self.modelsReady.emit(models)

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(result="QVariantMap")
    def loadPosition(self) -> dict[str, Any]:
        path = _presence_state_path(self._config_path)
        if not path.is_file():
            return {"saved": False}
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, dict) and "margin_right" in data and "margin_bottom" in data:
                return {
                    "saved": True,
                    "margin_right": int(data["margin_right"]),
                    "margin_bottom": int(data["margin_bottom"]),
                }
        except (OSError, ValueError, TypeError):
            pass
        return {"saved": False}

    @Slot(int, int)
    def savePosition(self, margin_right: int, margin_bottom: int) -> None:
        path = _presence_state_path(self._config_path)
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(path.parent, 0o700)
        tmp = path.with_suffix(".json.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with fh:
            fh.write(json.dumps({"margin_right": int(margin_right), "margin_bottom": int(margin_bottom)}))
        tmp.replace(path)

    @Slot()
    def resetPosition(self) -> None:
        path = _presence_state_path(self._config_path)
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass

    def _diagnostics_path(self) -> Path:
        from autum.capabilities import default_data_dir

        try:
            cfg = Config.load(self._config_path)
            base = cfg.resolved_data_dir() or default_data_dir()
        except Exception:
            base = default_data_dir()
        return base / "diagnostics.jsonl"

    @Slot(result="QVariantMap")
    def runDiagnostics(self) -> dict[str, Any]:
        from autum.diagnostics import DiagnosticsLog, run_system_health_checks

        try:
            cfg = Config.load(self._config_path)
        except Exception:
            cfg = Config()
        store = DiagnosticsLog(self._diagnostics_path())
        return run_system_health_checks(cfg, problems_log=store)

    @Slot()
    def runDiagnosticsAsync(self) -> None:
        def _worker() -> None:
            rep = self.runDiagnostics()
            self.diagnosticsReady.emit(rep)

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(result="QVariantMap")
    def clearProblems(self) -> dict[str, Any]:
        from autum.diagnostics import DiagnosticsLog

        store = DiagnosticsLog(self._diagnostics_path())
        cleared = store.clear()
        return {"ok": True, "cleared": cleared}
