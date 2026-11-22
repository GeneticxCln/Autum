"""Logging with sensitive-data filtering (WP3 / plan §5.3).

Ensures passwords, API keys, bearer tokens, private key blocks, and large
base64 payloads (such as encoded audio) are never written to logs.
"""

from __future__ import annotations

import logging
import re
import sys

_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    # Private key blocks
    (
        re.compile(r"-----BEGIN [A-Z0-9 ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z0-9 ]*PRIVATE KEY-----"),
        "[REDACTED_PRIVATE_KEY]",
    ),
    # Authorization / Bearer tokens
    (
        re.compile(r"(?i)(authorization\s*[:=]\s*(?:bearer|basic|token)\s+)[^\s,;\"']+"),
        r"\1[REDACTED]",
    ),
    (
        re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-~+/]+=*"),
        r"\1[REDACTED]",
    ),
    # Common key prefixes (OpenAI, Anthropic, GitHub, AWS, Google, Slack, etc.)
    (
        re.compile(r"\b(?:sk-[A-Za-z0-9_\-]{12,}|gh[pousr]_[A-Za-z0-9]{16,}|AKIA[0-9A-Z]{16}|AIza[0-9A-Za-z_\-]{20,}|xox[baprs]-[A-Za-z0-9\-]{10,})\b"),
        "[REDACTED_KEY]",
    ),
    # Key-value secret assignments: password=..., db_password=..., token: "...", service_api_key=...
    (
        re.compile(r"(?i)(\b[A-Za-z0-9_-]*(?:password|passwd|pwd|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|auth[_-]?token|private[_-]?key)\b\s*[:=]\s*['\"]?)([^'\"\s,;}{]+)"),
        r"\1[REDACTED]",
    ),
    # Long base64 blobs (e.g. raw input_audio WAV payloads)
    (
        re.compile(r"[A-Za-z0-9+/]{128,}={0,2}"),
        "[REDACTED_BLOB]",
    ),
)


def redact_sensitive(text: str) -> str:
    """Return text with known secret patterns and large base64 blobs masked."""
    out = text
    for pattern, replacement in _PATTERNS:
        out = pattern.sub(replacement, out)
    return out


class SensitiveDataFilter(logging.Filter):
    """Redacts sensitive values from every LogRecord message, args, and traceback."""

    _exc_formatter = logging.Formatter()

    def filter(self, record: logging.LogRecord) -> bool:
        try:
            rendered = record.getMessage()
        except Exception:
            rendered = str(record.msg)
        record.msg = redact_sensitive(rendered)
        record.args = ()
        if record.exc_info and not record.exc_text:
            record.exc_text = self._exc_formatter.formatException(record.exc_info)
            record.exc_info = None
        if record.exc_text:
            record.exc_text = redact_sensitive(record.exc_text)
        if record.stack_info:
            record.stack_info = redact_sensitive(record.stack_info)
        return True


def configure_logging(level: str = "INFO") -> logging.Logger:
    """Attach a filtered stderr handler to the `autum` logger hierarchy."""
    logger = logging.getLogger("autum")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False
    if not logger.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("autum: %(message)s"))
        handler.addFilter(SensitiveDataFilter())
        logger.addHandler(handler)
    else:
        for handler in logger.handlers:
            if not any(isinstance(f, SensitiveDataFilter) for f in handler.filters):
                handler.addFilter(SensitiveDataFilter())
    return logger
