"""Settings bridge between Autum's QML Settings app, local config.toml, and Ollama.

Allows the user to pick any locally installed Ollama model for Brain/Chat,
Vision, and Voice, toggle dynamic VRAM swap / hands-free / AEC / approvals /
allowed folders, save cleanly to `~/.config/autum/config.toml` (0600), and
hot-reload the running Autum runtime without restarting.
"""

from __future__ import annotations

import json
import os
import tempfile
import threading
from pathlib import Path
from typing import Any

from PySide6.QtCore import QObject, Signal, Slot

from autum import __version__
from autum.config import Config, default_config_path
from autum.diagnostics import query_ollama_models


def _presence_state_path(config_path: Path | None = None) -> Path:
    base = (config_path or default_config_path()).parent
    return base / "presence.json"


class SettingsBridge(QObject):
    """QML-accessible bridge for Autum Settings & persisted orb position."""

    diagnosticsReady = Signal("QVariantMap")
    modelsReady = Signal("QVariantList")
    audioDevicesReady = Signal("QVariantMap")
    micTestReady = Signal("QVariantMap")
    setupStatusReady = Signal("QVariantMap")
    pullProgress = Signal("QVariantMap")

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
        self._seq_lock = threading.Lock()
        self._models_seq = 0
        self._diag_seq = 0
        self._audio_seq = 0
        self._mic_test_seq = 0
        self._setup_seq = 0

    def _safe_emit(self, signal: Any, payload: Any) -> None:
        """Emit a Qt signal from a worker thread without raising if the QObject was torn down."""
        try:
            signal.emit(payload)
        except RuntimeError:
            pass

    @Slot(result=str)
    def appVersion(self) -> str:
        return __version__

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
                audio_input=str(raw.get("audio_input", "")).strip(),
                audio_output=str(raw.get("audio_output", "")).strip(),
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
        applied_live = False
        if self._model is not None:
            applied_live = bool(getattr(self._model, "_connected", False))
            self._model.reloadConfig()
        return {
            "ok": True,
            "path": str(saved),
            "restart_required": restart_fields,
            "applied_live": applied_live,
        }

    @Slot(str)
    def setModeAndPersist(self, mode: str) -> None:
        if mode not in ("push_to_talk", "wake_word", "hands_free"):
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
                audio_input=current.audio_input,
                audio_output=current.audio_output,
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
        url = (ollama_url or "").strip()
        if not url:
            try:
                url = Config.load(self._config_path).ollama_url
            except Exception:
                url = Config().ollama_url
        try:
            return query_ollama_models(url)
        except Exception:
            return []

    @Slot(str)
    def listOllamaModelsAsync(self, ollama_url: str = "") -> None:
        with self._seq_lock:
            self._models_seq += 1
            seq = self._models_seq

        def _worker() -> None:
            try:
                models = self.listOllamaModels(ollama_url)
            except Exception:
                models = []
            with self._seq_lock:
                if seq != self._models_seq:
                    return
            self._safe_emit(self.modelsReady, models)

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
        parent_existed = path.parent.exists()
        path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed or path.parent.name == "autum":
            try:
                os.chmod(path.parent, 0o700)
            except OSError:
                pass
        fd, tmp_name = tempfile.mkstemp(prefix=".presence.", suffix=".tmp", dir=str(path.parent))
        tmp = Path(tmp_name)
        fd_owned = True
        try:
            os.fchmod(fd, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
            fd_owned = False
            with fh:
                fh.write(json.dumps({"margin_right": int(margin_right), "margin_bottom": int(margin_bottom)}))
            tmp.replace(path)
        except Exception:
            if fd_owned:
                try:
                    os.close(fd)
                except OSError:
                    pass
            tmp.unlink(missing_ok=True)
            raise

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
        with self._seq_lock:
            self._diag_seq += 1
            seq = self._diag_seq

        def _worker() -> None:
            try:
                rep = self.runDiagnostics()
            except Exception as exc:
                rep = {
                    "overall_status": "error",
                    "checks": [
                        {
                            "section": "diagnostics",
                            "title": "Self-Diagnosis",
                            "status": "error",
                            "summary": f"Diagnostics failed: {exc}",
                            "details": "",
                            "suggestion": "Check config.toml and Ollama URL.",
                        }
                    ],
                    "recent_problems": [],
                }
            with self._seq_lock:
                if seq != self._diag_seq:
                    return
            self._safe_emit(self.diagnosticsReady, rep)

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(result="QVariantMap")
    def clearProblems(self) -> dict[str, Any]:
        from autum.diagnostics import DiagnosticsLog

        with self._seq_lock:
            self._diag_seq += 1
        store = DiagnosticsLog(self._diagnostics_path())
        cleared = store.clear()
        return {"ok": True, "cleared": cleared}

    def _resolved_data_dir(self) -> Path:
        from autum.capabilities import default_data_dir

        try:
            cfg = Config.load(self._config_path)
            return cfg.resolved_data_dir() or default_data_dir()
        except Exception:
            return default_data_dir()

    # ----- Phase 1: PipeWire Audio Devices & Live Mic Test ---------------

    @Slot(result="QVariantMap")
    def listAudioDevices(self) -> dict[str, Any]:
        from autum.audio import list_pipewire_devices

        try:
            return list_pipewire_devices()
        except Exception:
            return {"inputs": [], "outputs": []}

    @Slot()
    def listAudioDevicesAsync(self) -> None:
        with self._seq_lock:
            self._audio_seq += 1
            seq = self._audio_seq

        def _worker() -> None:
            devices = self.listAudioDevices()
            with self._seq_lock:
                if seq != self._audio_seq:
                    return
            self._safe_emit(self.audioDevicesReady, devices)

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(str)
    def testMicrophoneAsync(self, target: str = "") -> None:
        from autum.audio import probe_microphone_sync

        with self._seq_lock:
            self._mic_test_seq += 1
            seq = self._mic_test_seq

        def _worker() -> None:
            try:
                res = probe_microphone_sync(target=target)
            except Exception as exc:
                res = {"ok": False, "rms": 0.0, "bytes_read": 0, "device": target or "default", "error": str(exc)}
            with self._seq_lock:
                if seq != self._mic_test_seq:
                    return
            self._safe_emit(self.micTestReady, res)

        threading.Thread(target=_worker, daemon=True).start()

    # ----- Phase 2: Memory, Notes & Transcript Viewer --------------------

    @Slot(result="QVariantMap")
    def loadMemoryAndNotes(self) -> dict[str, Any]:
        from autum.capabilities import NotesStore
        from autum.diagnostics import TranscriptLog
        from autum.memory import MemoryStore

        data_dir = self._resolved_data_dir()
        memories: list[dict[str, Any]] = []
        notes: list[dict[str, Any]] = []
        transcript: list[dict[str, Any]] = []
        try:
            mem_store = MemoryStore(data_dir / "memory.json")
            memories = [m.to_dict() for m in mem_store.list_all()]
        except Exception:
            pass
        try:
            notes_store = NotesStore(data_dir)
            for title in notes_store.list_notes():
                try:
                    item = notes_store.read(title)
                    notes.append({"title": title, "content": str(item.get("content") or "")[:300]})
                except Exception:
                    notes.append({"title": title, "content": ""})
        except Exception:
            pass
        try:
            transcript = TranscriptLog(data_dir / "transcript.jsonl").recent(20)
        except Exception:
            pass
        return {"memories": memories, "notes": notes, "transcript": transcript}

    @Slot(str, result="QVariantMap")
    def deleteMemory(self, identifier: str) -> dict[str, Any]:
        from autum.memory import MemoryStore

        try:
            store = MemoryStore(self._resolved_data_dir() / "memory.json")
            removed = store.delete(identifier)
            return {"ok": True, "key": removed.key}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @Slot(str, result="QVariantMap")
    def deleteNote(self, title: str) -> dict[str, Any]:
        from autum.capabilities import NotesStore

        try:
            store = NotesStore(self._resolved_data_dir())
            store.delete(title)
            return {"ok": True, "title": title}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    @Slot(result="QVariantMap")
    def clearTranscript(self) -> dict[str, Any]:
        from autum.diagnostics import TranscriptLog

        try:
            cleared = TranscriptLog(self._resolved_data_dir() / "transcript.jsonl").clear()
            return {"ok": True, "cleared": cleared}
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

    # ----- Phase 4: First-Run Setup & Model Pull -------------------------

    @Slot(result="QVariantMap")
    def setupStatus(self) -> dict[str, Any]:
        from autum.diagnostics import gpu_vram_gb

        cfg_path = self._config_path or default_config_path()
        try:
            cfg = Config.load(cfg_path)
        except Exception:
            cfg = Config()
        installed = query_ollama_models(cfg.ollama_url)
        installed_names = {str(m.get("name") or "") for m in installed}

        def _has(name: str) -> bool:
            cands = {name} if ":" in name else {name, f"{name}:latest"}
            return bool(cands & installed_names)

        required = list(dict.fromkeys((cfg.model, cfg.resolved_chat_model(), cfg.resolved_vision_model())))
        missing = [m for m in required if not _has(m)]
        vram = gpu_vram_gb()
        if vram is not None and vram < 12.0:
            recommended = {"voice": "lfm3-omni:2b", "chat": "gemma4:4b", "vision": "lfm3-omni:2b"}
        else:
            recommended = {"voice": "lfm3-omni:2b", "chat": "gemma4:12b", "vision": "qwen3-vl:8b"}
        return {
            "first_run": not cfg_path.is_file(),
            "ollama_reachable": len(installed) > 0 or not missing,
            "vram_gb": vram or 0.0,
            "missing_models": missing,
            "recommended": recommended,
        }

    @Slot()
    def setupStatusAsync(self) -> None:
        with self._seq_lock:
            self._setup_seq += 1
            seq = self._setup_seq

        def _worker() -> None:
            try:
                st = self.setupStatus()
            except Exception:
                st = {"first_run": False, "ollama_reachable": False, "vram_gb": 0.0, "missing_models": [], "recommended": {}}
            with self._seq_lock:
                if seq != self._setup_seq:
                    return
            self._safe_emit(self.setupStatusReady, st)

        threading.Thread(target=_worker, daemon=True).start()

    @Slot(str, str)
    def pullModelAsync(self, model_name: str, ollama_url: str = "") -> None:
        import urllib.parse
        import urllib.request

        name = (model_name or "").strip()
        if not name:
            self._safe_emit(self.pullProgress, {"model": "", "status": "error", "percent": 0, "done": True, "error": "empty model name"})
            return
        url = (ollama_url or "").strip()
        if not url:
            try:
                url = Config.load(self._config_path).ollama_url
            except Exception:
                url = Config().ollama_url
        parsed = urllib.parse.urlparse(url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            self._safe_emit(self.pullProgress, {"model": name, "status": "error", "percent": 0, "done": True, "error": "ollama_url must be localhost"})
            return

        def _worker() -> None:
            endpoint = f"{url.rstrip('/')}/api/pull"
            payload = json.dumps({"name": name, "stream": True}).encode("utf-8")
            req = urllib.request.Request(endpoint, data=payload, headers={"Content-Type": "application/json"}, method="POST")
            emitted_done = False
            try:
                self._safe_emit(self.pullProgress, {"model": name, "status": f"Starting pull for {name}...", "percent": 0, "done": False, "error": ""})
                with urllib.request.urlopen(req, timeout=600) as resp:
                    for raw_line in resp:
                        line = raw_line.decode("utf-8", errors="replace").strip()
                        if not line:
                            continue
                        try:
                            data = json.loads(line)
                        except ValueError:
                            continue
                        if data.get("error"):
                            emitted_done = True
                            self._safe_emit(self.pullProgress, {"model": name, "status": "error", "percent": 0, "done": True, "error": str(data["error"])})
                            return
                        total = int(data.get("total") or 0)
                        completed = int(data.get("completed") or 0)
                        pct = int(round((completed / total) * 100)) if total > 0 else 0
                        status = str(data.get("status") or "downloading")
                        done = status == "success"
                        if done:
                            emitted_done = True
                        self._safe_emit(self.pullProgress, {"model": name, "status": status, "percent": 100 if done else pct, "done": done, "error": ""})
                if not emitted_done:
                    self._safe_emit(self.pullProgress, {"model": name, "status": "success", "percent": 100, "done": True, "error": ""})
            except Exception as exc:
                self._safe_emit(self.pullProgress, {"model": name, "status": "error", "percent": 0, "done": True, "error": str(exc)})

        threading.Thread(target=_worker, daemon=True).start()
