"""Turn streamed model text into chunks that speech synthesis can speak."""

from __future__ import annotations

import re

_ABBREVIATIONS = frozenset({"e.g.", "i.e.", "mr.", "mrs.", "ms.", "dr.", "vs.", "st."})
_SENTENCE_END = re.compile(r"[.!?…]+[\"')\]]*\s|\n+")
_CLAUSE_END = re.compile(r"[,;:—]\s")
_THINK = re.compile(
    r"<think>.*?(?:</think>|$)|</think>"
    r"|<\|think\|>.*?(?:<\|/think\|>|$)|<\|/think\|>"
    r"|<\|channel>thought.*?(?:<channel\|>|$)|<channel\|>",
    re.DOTALL,
)
_TOOL_CALL_TAG = re.compile(r"<\|tool_call>.*?(?:<tool_call\|>|$)|<tool_call\|>|<\|[^|>]{1,40}\|>", re.DOTALL)
_URL = re.compile(r"https?://\S+")
_MARKUP = re.compile(r"[*_#`>|~]+")
_SPACES = re.compile(r"\s+")

_HIDDEN_TAGS: tuple[tuple[str, str], ...] = (
    ("<think>", "</think>"),
    ("<|think|>", "<|/think|>"),
    ("<|channel>thought", "<channel|>"),
    ("<|tool_call>", "<tool_call|>"),
)


class SentenceChunker:
    """Split a token stream at sentence ends while stripping streamed <think> and raw tool-call blocks.

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
        self._hidden_close: str | None = None
        self._think_buf = ""

    def _filter_think(self, delta: str) -> str:
        """Strip <think>...</think> and raw <|tool_call>...<tool_call|> across arbitrary token boundaries."""
        self._think_buf += delta
        out: list[str] = []
        max_open = max(len(o) for o, _ in _HIDDEN_TAGS)
        while self._think_buf:
            if self._hidden_close is not None:
                close_tag = self._hidden_close
                end_idx = self._think_buf.find(close_tag)
                if end_idx < 0:
                    keep = len(close_tag) - 1
                    self._think_buf = self._think_buf[-keep:] if len(self._think_buf) > keep else self._think_buf
                    break
                self._think_buf = self._think_buf[end_idx + len(close_tag) :]
                self._hidden_close = None
            else:
                earliest_idx = -1
                earliest_open = ""
                earliest_close = ""
                for open_tag, close_tag in _HIDDEN_TAGS:
                    idx = self._think_buf.find(open_tag)
                    if idx >= 0 and (earliest_idx < 0 or idx < earliest_idx):
                        earliest_idx = idx
                        earliest_open = open_tag
                        earliest_close = close_tag
                if earliest_idx < 0:
                    partial = 0
                    for k in range(1, min(max_open, len(self._think_buf) + 1)):
                        tail = self._think_buf[-k:]
                        if any(open_tag.startswith(tail) for open_tag, _ in _HIDDEN_TAGS):
                            partial = k
                    emit = self._think_buf[:-partial] if partial else self._think_buf
                    self._think_buf = self._think_buf[-partial:] if partial else ""
                    if emit:
                        out.append(emit)
                    break
                if earliest_idx > 0:
                    out.append(self._think_buf[:earliest_idx])
                self._think_buf = self._think_buf[earliest_idx + len(earliest_open) :]
                self._hidden_close = earliest_close
        return "".join(out)

    def feed(self, delta: str) -> list[str]:
        visible = self._filter_think(delta)
        if not visible:
            return []
        self._buffer += visible
        chunks: list[str] = []
        while chunk := self._take():
            chunks.append(chunk)
        return chunks

    def flush(self) -> list[str]:
        if self._think_buf and self._hidden_close is None:
            self._buffer += self._think_buf
        self._think_buf = ""
        self._hidden_close = None
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
    """Remove what should not be read aloud: reasoning, raw tool-call tags, URLs, markup."""
    text = _THINK.sub("", text)
    text = _TOOL_CALL_TAG.sub("", text)
    text = _URL.sub("", text)
    text = _MARKUP.sub("", text)
    return _SPACES.sub(" ", text).strip()
