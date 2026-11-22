"""Turn streamed model text into chunks that speech synthesis can speak."""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs.", "st."})
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s|\n+")
_CLAUSE_END = re.compile(r"[,;:—]\s")
_THINK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL)
_URL = re.compile(r"https?://\S+")
_MARKUP = re.compile(r"[*_#`>|~]+")
_SPACES = re.compile(r"\s+")


class SentenceChunker:
    """Split a token stream at sentence ends.

    The first chunk may also end at a clause break once it is at least
    ``first_clause_chars`` long, so the first audio starts sooner. Text
    without punctuation is cut at a space once it passes ``max_chars``,
    so speech never waits for the whole response.
    """

    def __init__(self, first_clause_chars: int = 40, max_chars: int = 200) -> None:
        self._buffer = ""
        self._first = True
        self._first_clause_chars = first_clause_chars
        self._max_chars = max_chars

    def feed(self, delta: str) -> list[str]:
        self._buffer += delta
        chunks: list[str] = []
        while chunk := self._take():
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        rest, self._buffer = self._buffer.strip(), ""
        return [rest] if rest else []

    def _take(self) -> str | None:
        start = 0
        while match := _SENTENCE_END.search(self._buffer, start):
            candidate = self._buffer[: match.end()]
            last_word = candidate.rstrip().rsplit(maxsplit=1)[-1].lower() if candidate.strip() else ""
            if last_word in _ABBREVIATIONS:
                start = match.end()
                continue
            return self._cut(match.end())
        if self._first:
            for match in _CLAUSE_END.finditer(self._buffer):
                if match.end() >= self._first_clause_chars:
                    return self._cut(match.end())
        if len(self._buffer) > self._max_chars:
            space = self._buffer.rfind(" ", 0, self._max_chars)
            return self._cut(space + 1 if space > 0 else self._max_chars)
        return None

    def _cut(self, end: int) -> str | None:
        chunk, self._buffer = self._buffer[:end].strip(), self._buffer[end:]
        if not chunk:
            return None if not self._buffer else self._take()
        self._first = False
        return chunk


def speakable(text: str) -> str:
    """Remove what should not be read aloud: reasoning, URLs, markup."""
    text = _THINK.sub("", text)
    text = _URL.sub("", text)
    text = _MARKUP.sub("", text)
    return _SPACES.sub(" ", text).strip()
