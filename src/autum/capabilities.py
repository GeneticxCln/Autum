"""Environment-neutral local capabilities: clock, timers, and notes (WP6 / M7).

All data stays inside Autum's own data folder (`$XDG_DATA_HOME/autum/` or
`~/.local/share/autum/`). No desktop integration or network access is used.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

from autum.events import new_id
from autum.logging import redact_sensitive
from autum.tools import AuditLog, Capability, PermissionLevel, ToolRegistry


def default_data_dir() -> Path:
    base = os.environ.get("XDG_DATA_HOME")
    root = Path(base) if base else Path.home() / ".local" / "share"
    return root / "autum"


# ----- timers -----------------------------------------------------------


@dataclass
class TimerEntry:
    id: str
    label: str
    seconds: int
    ends_at: float
    task: asyncio.Task[None] | None = None

    @property
    def remaining_s(self) -> int:
        return max(0, int(round(self.ends_at - time.monotonic())))


class TimerStore:
    """In-memory local countdown timers."""

    def __init__(self, on_fired: Callable[[TimerEntry], None] | None = None) -> None:
        self.on_fired = on_fired
        self._timers: dict[str, TimerEntry] = {}

    def start_timer(self, seconds: int, label: str = "timer") -> TimerEntry:
        tid = new_id()[:8]
        clean_label = label.strip() or "timer"
        entry = TimerEntry(id=tid, label=clean_label, seconds=seconds, ends_at=time.monotonic() + seconds)
        entry.task = asyncio.get_running_loop().create_task(self._wait(entry))
        self._timers[tid] = entry
        return entry

    async def _wait(self, entry: TimerEntry) -> None:
        try:
            await asyncio.sleep(entry.seconds)
        except asyncio.CancelledError:
            return
        self._timers.pop(entry.id, None)
        if self.on_fired is not None:
            self.on_fired(entry)

    def get(self, timer_id: str) -> TimerEntry | None:
        return self._timers.get(timer_id)

    def list_active(self) -> list[dict[str, Any]]:
        return [
            {"timer_id": t.id, "label": t.label, "remaining_seconds": t.remaining_s}
            for t in self._timers.values()
        ]

    def cancel(self, identifier: str) -> TimerEntry | None:
        needle = identifier.strip().lower()
        target = self._timers.get(identifier.strip())
        if target is None:
            for t in self._timers.values():
                if t.label.lower() == needle:
                    target = t
                    break
        if target is None:
            return None
        self._timers.pop(target.id, None)
        if target.task is not None and not target.task.done():
            target.task.cancel()
        return target

    def stop_all(self) -> None:
        for t in list(self._timers.values()):
            if t.task is not None and not t.task.done():
                t.task.cancel()
        self._timers.clear()


# ----- notes ------------------------------------------------------------

_SLUG_BAD = re.compile(r"[^a-z0-9_-]+")


def note_slug(title: str) -> str:
    slug = _SLUG_BAD.sub("-", title.strip().lower()).strip("-_")
    if not slug or len(slug) > 80:
        raise ValueError("note title must produce a valid name (1-80 letters or digits)")
    return slug


class NotesStore:
    """Plain-text notes confined strictly to `<data_dir>/notes/`.

    Never overwrites existing notes on creation (§5.2) and refuses to store
    secrets/passwords (§5.3).
    """

    def __init__(self, data_dir: Path) -> None:
        self.notes_dir = (data_dir / "notes").resolve()

    def _path(self, title: str) -> Path:
        slug = note_slug(title)
        path = (self.notes_dir / f"{slug}.txt").resolve()
        if path.parent != self.notes_dir:
            raise ValueError("invalid note path")
        return path

    def create(self, title: str, content: str) -> dict[str, Any]:
        if redact_sensitive(content) != content:
            raise ValueError("refusing to store passwords, keys, or tokens in notes (plan §5.3)")
        path = self._path(title)
        parent_existed = self.notes_dir.parent.exists()
        self.notes_dir.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        if not parent_existed or self.notes_dir.parent.name == "autum":
            try:
                os.chmod(self.notes_dir.parent, 0o700)
            except OSError:
                pass
        self.notes_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self.notes_dir, 0o700)
        # Exclusive creation (O_CREAT | O_EXCL, mode 0600): atomic no-overwrite + private (§5.2, §5.3).
        try:
            fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            raise FileExistsError(f"note {path.stem!r} already exists; create_note never overwrites") from None
        with os.fdopen(fd, "w", encoding="utf-8", newline="") as fh:
            fh.write(content)
        os.chmod(path, 0o600)
        return {"slug": path.stem, "bytes": len(content.encode("utf-8"))}

    def list_notes(self) -> list[str]:
        if not self.notes_dir.is_dir():
            return []
        return sorted(p.stem for p in self.notes_dir.glob("*.txt") if p.is_file())

    def read(self, title: str) -> dict[str, Any]:
        path = self._path(title)
        if not path.is_file():
            raise FileNotFoundError(f"note {path.stem!r} does not exist")
        with path.open("r", encoding="utf-8", newline="") as fh:
            content = fh.read()
        return {"slug": path.stem, "content": content}

    def delete(self, title: str) -> dict[str, Any]:
        path = self._path(title)
        if not path.is_file():
            raise FileNotFoundError(f"note {path.stem!r} does not exist")
        path.unlink()
        return {"slug": path.stem, "deleted": True}

    def exists(self, title: str) -> bool:
        try:
            return self._path(title).is_file()
        except ValueError:
            return False


# ----- registry factory -------------------------------------------------


def build_default_registry(
    data_dir: Path | None = None,
    *,
    on_timer_fired: Callable[[TimerEntry], None] | None = None,
) -> tuple[ToolRegistry, TimerStore, NotesStore]:
    """Build the M7 registry with clock, timers, and local notes capabilities."""
    root = (data_dir or default_data_dir()).resolve()
    audit = AuditLog(root / "audit.jsonl")
    registry = ToolRegistry(audit=audit)
    timers = TimerStore(on_fired=on_timer_fired)
    notes = NotesStore(root)

    # 1. Clock (READ_ONLY)
    async def _get_time(_args: dict[str, Any]) -> dict[str, Any]:
        now = dt.datetime.now().astimezone()
        return {
            "iso": now.isoformat(timespec="seconds"),
            "date": f"{now:%A, %d %B %Y}",
            "time": f"{now:%H:%M}",
            "timezone": now.tzname() or "local",
        }

    async def _verify_time(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return bool(result.get("iso") and result.get("time"))

    registry.register(
        Capability(
            name="get_time",
            description="Get the current local date, time, and timezone.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "required": ["iso", "date", "time", "timezone"]},
            permission_level=PermissionLevel.READ_ONLY,
            execute=_get_time,
            verify=_verify_time,
            target_summary=lambda _: "system clock",
        )
    )

    # 2. Set timer (REVERSIBLE)
    async def _set_timer(args: dict[str, Any]) -> dict[str, Any]:
        entry = timers.start_timer(int(args["seconds"]), str(args.get("label", "timer")))
        return {"timer_id": entry.id, "label": entry.label, "seconds": entry.seconds}

    async def _verify_set_timer(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return timers.get(result["timer_id"]) is not None

    registry.register(
        Capability(
            name="set_timer",
            description="Start a countdown timer in seconds.",
            input_schema={
                "type": "object",
                "properties": {
                    "seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                    "label": {"type": "string", "minLength": 1, "maxLength": 60},
                },
                "required": ["seconds"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["timer_id", "label", "seconds"]},
            permission_level=PermissionLevel.REVERSIBLE,
            execute=_set_timer,
            verify=_verify_set_timer,
            target_summary=lambda a: f"timer '{a.get('label', 'timer')}' for {a['seconds']} seconds",
        )
    )

    # 3. List timers (READ_ONLY)
    async def _list_timers(_args: dict[str, Any]) -> dict[str, Any]:
        return {"timers": timers.list_active()}

    async def _verify_list_timers(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return isinstance(result.get("timers"), list)

    registry.register(
        Capability(
            name="list_timers",
            description="List active timers and remaining seconds.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "required": ["timers"]},
            permission_level=PermissionLevel.READ_ONLY,
            execute=_list_timers,
            verify=_verify_list_timers,
            target_summary=lambda _: "active timers",
        )
    )

    # 4. Cancel timer (REVERSIBLE)
    async def _cancel_timer(args: dict[str, Any]) -> dict[str, Any]:
        cancelled = timers.cancel(str(args["identifier"]))
        if cancelled is None:
            raise LookupError(f"no active timer matching {args['identifier']!r}")
        return {"timer_id": cancelled.id, "label": cancelled.label, "cancelled": True}

    async def _verify_cancel_timer(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return timers.get(result["timer_id"]) is None

    registry.register(
        Capability(
            name="cancel_timer",
            description="Cancel an active timer by its ID or label.",
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "minLength": 1, "maxLength": 60, "description": "Timer ID or label"},
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["timer_id", "label", "cancelled"]},
            permission_level=PermissionLevel.REVERSIBLE,
            execute=_cancel_timer,
            verify=_verify_cancel_timer,
            target_summary=lambda a: f"timer '{a['identifier']}'",
        )
    )

    # 5. Create note (REVERSIBLE, never overwrites, verified from disk)
    async def _create_note(args: dict[str, Any]) -> dict[str, Any]:
        return notes.create(str(args["title"]), str(args["content"]))

    async def _verify_create_note(args: dict[str, Any], result: dict[str, Any]) -> bool:
        if not notes.exists(result["slug"]):
            return False
        return notes.read(result["slug"])["content"] == str(args["content"])

    registry.register(
        Capability(
            name="create_note",
            description="Create a new plain-text note (never overwrites existing notes).",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 80},
                    "content": {"type": "string", "minLength": 1, "maxLength": 8000},
                },
                "required": ["title", "content"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["slug", "bytes"]},
            permission_level=PermissionLevel.REVERSIBLE,
            cancellable=False,  # Shield atomic file write from partial cancellation
            execute=_create_note,
            verify=_verify_create_note,
            target_summary=lambda a: f"new note '{a['title']}'",
        )
    )

    # 6. List notes (READ_ONLY)
    async def _list_notes(_args: dict[str, Any]) -> dict[str, Any]:
        return {"notes": notes.list_notes()}

    async def _verify_list_notes(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return result.get("notes") == notes.list_notes()

    registry.register(
        Capability(
            name="list_notes",
            description="List saved note titles.",
            input_schema={"type": "object", "properties": {}, "additionalProperties": False},
            output_schema={"type": "object", "required": ["notes"]},
            permission_level=PermissionLevel.READ_ONLY,
            execute=_list_notes,
            verify=_verify_list_notes,
            target_summary=lambda _: "local notes folder",
        )
    )

    # 7. Read note (READ_ONLY, observed/untrusted output per §5.1)
    async def _read_note(args: dict[str, Any]) -> dict[str, Any]:
        return notes.read(str(args["title"]))

    async def _verify_read_note(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return notes.exists(result["slug"])

    registry.register(
        Capability(
            name="read_note",
            description="Read a saved local note.",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["slug", "content"]},
            permission_level=PermissionLevel.READ_ONLY,
            observed_output=True,  # Note contents are untrusted data, never commands (§5.1)
            execute=_read_note,
            verify=_verify_read_note,
            target_summary=lambda a: f"note '{a['title']}'",
        )
    )

    # 8. Delete note (IRREVERSIBLE — ALWAYS requires explicit approval per §5.2)
    async def _delete_note(args: dict[str, Any]) -> dict[str, Any]:
        return notes.delete(str(args["title"]))

    async def _verify_delete_note(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return not notes.exists(result["slug"])

    registry.register(
        Capability(
            name="delete_note",
            description="Permanently delete a note (irreversible; requires approval).",
            input_schema={
                "type": "object",
                "properties": {
                    "title": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["title"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["slug", "deleted"]},
            permission_level=PermissionLevel.IRREVERSIBLE,
            cancellable=False,
            precondition=lambda a: None if notes.exists(str(a["title"])) else f"note {a['title']!r} does not exist",
            execute=_delete_note,
            verify=_verify_delete_note,
            target_summary=lambda a: f"delete note '{a['title']}'",
        )
    )

    return registry, timers, notes
