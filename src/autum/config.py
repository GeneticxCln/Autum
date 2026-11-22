"""Local configuration for Autum (WP3 / M5).

Reads an optional TOML file from `$XDG_CONFIG_HOME/autum/config.toml`
(default `~/.config/autum/config.toml`) using Python's standard `tomllib`.
Missing files simply use defaults; invalid values raise ValueError with a
clear message rather than silently misconfiguring the runtime.
"""

from __future__ import annotations

import os
import tomllib
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
    voice: str = "female-american-english"
    log_level: str = "INFO"
    data_dir: str = ""
    allowed_read_dirs: tuple[str, ...] = ()
    require_reversible_approval: bool = False
    approval_timeout_s: float = 20.0

    def __post_init__(self) -> None:
        if not isinstance(self.mode, Mode):
            object.__setattr__(self, "mode", Mode(self.mode))
        if isinstance(self.allowed_read_dirs, list):
            object.__setattr__(self, "allowed_read_dirs", tuple(str(p) for p in self.allowed_read_dirs))
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
        if not self.ollama_url.startswith(("http://127.0.0.1", "http://localhost", "http://[::1]")):
            raise ValueError("ollama_url must point to localhost (plan §5.5)")

    def resolved_data_dir(self) -> Path | None:
        return Path(self.data_dir).expanduser() if self.data_dir else None

    def resolved_allowed_read_dirs(self) -> tuple[Path, ...]:
        return tuple(Path(p).expanduser() for p in self.allowed_read_dirs if p)

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
