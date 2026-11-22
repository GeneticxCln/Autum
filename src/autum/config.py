"""Local configuration for Autum (WP3 / M5).

Reads an optional TOML file from `$XDG_CONFIG_HOME/autum/config.toml`
(default `~/.config/autum/config.toml`) using Python's standard `tomllib`.
Missing files simply use defaults; invalid values raise ValueError with a
clear message rather than silently misconfiguring the runtime.
"""

from __future__ import annotations

import json
import os
import tomllib
import urllib.parse
from dataclasses import dataclass
from pathlib import Path

from autum.runtime import Mode


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
            object.__setattr__(self, "mode", Mode(self.mode))
        for tuple_field in ("allowed_read_dirs", "enabled_integrations", "excluded_windows"):
            val = getattr(self, tuple_field)
            if isinstance(val, list):
                object.__setattr__(self, tuple_field, tuple(str(p) for p in val))
        if self.silence_ms < 100 or self.silence_ms > 5000:
            raise ValueError("silence_ms must be between 100 and 5000")
        if self.barge_in_ms < 64 or self.barge_in_ms > 2000:
            raise ValueError("barge_in_ms must be between 64 and 2000")
        if self.max_history < 2 or self.max_history > 200:
            raise ValueError("max_history must be between 2 and 200")
        if self.max_context_chars < 500 or self.max_context_chars > 200_000:
            raise ValueError("max_context_chars must be between 500 and 200000")
        if self.approval_timeout_s < 2.0 or self.approval_timeout_s > 120.0:
            raise ValueError("approval_timeout_s must be between 2 and 120")
        parsed = urllib.parse.urlparse(self.ollama_url)
        if parsed.scheme != "http" or parsed.hostname not in ("127.0.0.1", "localhost", "::1"):
            raise ValueError("ollama_url must point to localhost (plan §5.5)")

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
                return json.dumps(val)
            if isinstance(val, (list, tuple)):
                return "[" + ", ".join(json.dumps(str(x)) for x in val) + "]"
            raise TypeError(f"unsupported TOML value: {val!r}")

        lines = ["# Autum local configuration (managed by Autum Settings)"]
        for key, val in self.to_dict().items():
            lines.append(f"{key} = {_fmt(val)}")
        return "\n".join(lines) + "\n"

    def save(self, path: Path | None = None) -> Path:
        """Atomically write validated config to disk with owner-only permissions (0700 / 0600)."""
        cfg_path = path or default_config_path()
        cfg_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(cfg_path.parent, 0o700)
        if cfg_path.is_file():
            try:
                self.load(cfg_path)
            except Exception:
                backup = cfg_path.with_suffix(".toml.corrupt")
                try:
                    cfg_path.replace(backup)
                    os.chmod(backup, 0o600)
                except OSError:
                    pass
        tmp = cfg_path.with_suffix(".toml.tmp")
        fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
        try:
            os.fchmod(fd, 0o600)
            fh = os.fdopen(fd, "w", encoding="utf-8")
        except Exception:
            os.close(fd)
            raise
        with fh:
            fh.write(self.to_toml())
        tmp.replace(cfg_path)
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
