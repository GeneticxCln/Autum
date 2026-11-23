"""Local configuration for Autum (WP3 / M5).

Reads an optional TOML file from `$XDG_CONFIG_HOME/autum/config.toml`
(default `~/.config/autum/config.toml`) using Python's standard `tomllib`.
Missing files simply use defaults; invalid values raise ValueError with a
clear message rather than silently misconfiguring the runtime.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
import time
import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from autum.runtime import Mode

_VALID_LOG_LEVELS = frozenset({"CRITICAL", "ERROR", "WARNING", "WARN", "INFO", "DEBUG", "NOTSET"})


def default_config_path() -> Path:
    base = os.environ.get("XDG_CONFIG_HOME")
    root = Path(base) if base else Path.home() / ".config"
    return root / "autum" / "config.toml"


@dataclass(frozen=True)
class Config:
    mode: Mode = Mode.PUSH_TO_TALK
    aec: bool = False
    silence_ms: int = 700
    barge_in_ms: int = 256
    max_history: int = 20
    max_context_chars: int = 10_000
    ollama_url: str = "http://127.0.0.1:11434"
    model: str = "lfm3-omni:2b"
    chat_model: str = "gemma4:12b"
    vision_model: str = "qwen3-vl:8b"
    dynamic_swap: bool = True
    chat_think: bool = False
    voice: str = "female-american-english"
    audio_input: str = ""
    audio_output: str = ""
    log_level: str = "INFO"
    data_dir: str = ""
    allowed_read_dirs: tuple[str, ...] = ()
    enabled_integrations: tuple[str, ...] = ("linux_desktop",)
    excluded_windows: tuple[str, ...] = (
        "keepassxc", "1password", "bitwarden", "kwallet", "polkit", "pinentry", "password",
    )
    require_reversible_approval: bool = False
    approval_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            if not isinstance(self.mode, str):
                raise ValueError(f"mode must be a string, got {type(self.mode).__name__}")
            object.__setattr__(self, "mode", Mode(self.mode))
        for bool_field in ("aec", "dynamic_swap", "chat_think", "require_reversible_approval"):
            if not isinstance(getattr(self, bool_field), bool):
                raise ValueError(f"{bool_field} must be a boolean")
        for str_field in (
            "ollama_url",
            "model",
            "chat_model",
            "vision_model",
            "voice",
            "audio_input",
            "audio_output",
            "log_level",
            "data_dir",
        ):
            if not isinstance(getattr(self, str_field), str):
                raise ValueError(f"{str_field} must be a string")
        if not self.model.strip():
            raise ValueError("model must be non-empty")
        if self.log_level.upper() not in _VALID_LOG_LEVELS:
            raise ValueError(f"log_level must be one of {sorted(_VALID_LOG_LEVELS)}")
        for tuple_field in ("allowed_read_dirs", "enabled_integrations", "excluded_windows"):
            val = getattr(self, tuple_field)
            if isinstance(val, (str, bytes)) or not isinstance(val, (list, tuple)):
                raise ValueError(f"{tuple_field} must be a list of strings")
            if not all(isinstance(item, str) for item in val):
                raise ValueError(f"{tuple_field} entries must all be strings")
            if isinstance(val, list):
                object.__setattr__(self, tuple_field, tuple(val))
        for int_field, lo, hi in (
            ("silence_ms", 100, 5000),
            ("barge_in_ms", 64, 2000),
            ("max_history", 2, 200),
            ("max_context_chars", 500, 200_000),
        ):
            val = getattr(self, int_field)
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{int_field} must be an integer")
            if val < lo or val > hi:
                raise ValueError(f"{int_field} must be between {lo} and {hi}")
        if (
            isinstance(self.approval_timeout_s, bool)
            or not isinstance(self.approval_timeout_s, (int, float))
            or not math.isfinite(self.approval_timeout_s)
        ):
            raise ValueError("approval_timeout_s must be a finite number")
        if self.approval_timeout_s < 2.0 or self.approval_timeout_s > 120.0:
            raise ValueError("approval_timeout_s must be between 2 and 120")
        try:
            parsed = urllib.parse.urlparse(self.ollama_url)
            port = parsed.port
        except ValueError as exc:
            raise ValueError(f"invalid ollama_url: {exc}") from None
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("ollama_url must point to localhost (plan §5.5)")
        if port is not None and not (1 <= port <= 65535):
            raise ValueError("ollama_url port must be between 1 and 65535")

    def resolved_data_dir(self) -> Path | None:
        return Path(self.data_dir).expanduser() if self.data_dir else None

    def resolved_chat_model(self) -> str:
        return (self.chat_model or self.model).strip()

    def resolved_vision_model(self) -> str:
        return (self.vision_model or self.chat_model or self.model).strip()

    def resolved_allowed_read_dirs(self) -> tuple[Path, ...]:
        return tuple(Path(p).expanduser() for p in self.allowed_read_dirs if p)

    def to_dict(self) -> dict[str, object]:
        return {
            "mode": self.mode.value,
            "aec": self.aec,
            "silence_ms": self.silence_ms,
            "barge_in_ms": self.barge_in_ms,
            "max_history": self.max_history,
            "max_context_chars": self.max_context_chars,
            "ollama_url": self.ollama_url,
            "model": self.model,
            "chat_model": self.chat_model,
            "vision_model": self.vision_model,
            "dynamic_swap": self.dynamic_swap,
            "chat_think": self.chat_think,
            "voice": self.voice,
            "audio_input": self.audio_input,
            "audio_output": self.audio_output,
            "log_level": self.log_level,
            "data_dir": self.data_dir,
            "allowed_read_dirs": list(self.allowed_read_dirs),
            "enabled_integrations": list(self.enabled_integrations),
            "excluded_windows": list(self.excluded_windows),
            "require_reversible_approval": self.require_reversible_approval,
            "approval_timeout_s": self.approval_timeout_s,
        }

    def to_toml(self) -> str:
        """Serialize this validated Config to clean TOML."""
        def _fmt(val: object) -> str:
            if isinstance(val, bool):
                return "true" if val else "false"
            if isinstance(val, (int, float)):
                return str(val)
            if isinstance(val, str):
                return json.dumps(val, ensure_ascii=False)
            if isinstance(val, (list, tuple)):
                return "[" + ", ".join(json.dumps(str(x), ensure_ascii=False) for x in val) + "]"
            raise TypeError(f"unsupported TOML value: {val!r}")

        lines = ["# Autum local configuration (managed by Autum Settings)"]
        for key, val in self.to_dict().items():
            lines.append(f"{key} = {_fmt(val)}")
        return "\n".join(lines) + "\n"

    def save(self, path: Path | None = None) -> Path:
        """Atomically write validated config to disk with owner-only permissions (0700 / 0600)."""
        cfg_path = path or default_config_path()
        parent_existed = cfg_path.parent.exists()
        cfg_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed or cfg_path.parent.name == "autum":
            try:
                os.chmod(cfg_path.parent, 0o700)
            except OSError:
                pass
        if cfg_path.is_file():
            try:
                self.load(cfg_path)
            except Exception:
                backup = cfg_path.with_name(f"{cfg_path.name}.corrupt.{time.time_ns() // 1_000_000}")
                try:
                    cfg_path.replace(backup)
                    os.chmod(backup, 0o600)
                except OSError:
                    pass
        fd, tmp_name = tempfile.mkstemp(prefix=f".{cfg_path.name}.", suffix=".tmp", dir=str(cfg_path.parent))
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
                fh.write(self.to_toml())
            tmp.replace(cfg_path)
        except Exception:
            tmp.unlink(missing_ok=True)
            raise
        return cfg_path

    @classmethod
    def load(cls, path: Path | None = None) -> Config:
        cfg_path = path or default_config_path()
        if not cfg_path.is_file():
            return cls()
        raw = tomllib.loads(cfg_path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{cfg_path}: expected a TOML table")
        allowed = {f.name for f in cls.__dataclass_fields__.values()}
        unknown = set(raw) - allowed
        if unknown:
            raise ValueError(f"{cfg_path}: unknown config keys {sorted(unknown)}")
        return cls(**raw)
