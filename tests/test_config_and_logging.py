"""Configuration and sensitive-data log filtering tests (WP3 / M5 / §5.3 / §5.5)."""

from __future__ import annotations

import logging

import pytest

from autum.config import Config
from autum.logging import SensitiveDataFilter, redact_sensitive
from autum.runtime import Mode


def test_config_defaults_and_toml_loading(tmp_path):
    assert Config.load(tmp_path / "missing.toml") == Config()

    cfg_file = tmp_path / "config.toml"
    cfg_file.write_text(
        'mode = "hands_free"\n'
        "aec = true\n"
        "silence_ms = 600\n"
        "barge_in_ms = 220\n"
        "max_history = 16\n"
        "max_context_chars = 8000\n",
        encoding="utf-8",
    )
    cfg = Config.load(cfg_file)
    assert cfg.mode is Mode.HANDS_FREE
    assert (cfg.aec, cfg.silence_ms, cfg.barge_in_ms) == (True, 600, 220)
    assert (cfg.max_history, cfg.max_context_chars) == (16, 8000)


def test_config_rejects_unknown_keys_and_non_local_ollama_url(tmp_path):
    bad_key = tmp_path / "bad_key.toml"
    bad_key.write_text("nonexistent = 1\n", encoding="utf-8")
    with pytest.raises(ValueError, match="unknown config keys"):
        Config.load(bad_key)

    remote = tmp_path / "remote.toml"
    remote.write_text('ollama_url = "https://api.example.com"\n', encoding="utf-8")
    with pytest.raises(ValueError, match="localhost"):
        Config.load(remote)


def test_sensitive_data_redaction_covers_secrets_and_audio_blobs():
    samples = [
        ("password=mySuperSecret123", "mySuperSecret123"),
        ('api_key: "sk-proj-abc1234567890XYZ"', "sk-proj-abc1234567890XYZ"),
        ("Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig", "eyJhbGciOiJIUzI1NiJ9"),
        ("token=ghp_1234567890abcdef1234567890", "ghp_1234567890abcdef1234567890"),
        ("-----BEGIN OPENSSH PRIVATE KEY-----\nSECRETKEYDATA\n-----END OPENSSH PRIVATE KEY-----", "SECRETKEYDATA"),
        ("wav=" + "A" * 200, "A" * 128),
    ]
    for raw, secret in samples:
        redacted = redact_sensitive(raw)
        assert secret not in redacted, raw


def test_logging_filter_redacts_formatted_log_records():
    records: list[logging.LogRecord] = []

    class Capture(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            records.append(record)

    logger = logging.getLogger("autum.test_redact")
    logger.setLevel(logging.INFO)
    logger.propagate = False
    handler = Capture()
    handler.addFilter(SensitiveDataFilter())
    logger.addHandler(handler)
    try:
        logger.info("connect failed with password=%s and token=%s", "hunter2", "sk-1234567890abcdef")
    finally:
        logger.removeHandler(handler)

    assert len(records) == 1
    msg = records[0].getMessage()
    assert "hunter2" not in msg and "sk-1234567890abcdef" not in msg
    assert "[REDACTED]" in msg
