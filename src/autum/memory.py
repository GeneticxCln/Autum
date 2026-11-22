"""User-controlled persistent memory (WP8 / M9).

Implements plan §5.1, §5.3, and §6 WP8:
- Only stores user-selected facts (never saves full conversation transcripts by default).
- Keeps persistent facts strictly separate from temporary session history.
- Records provenance (`user`, `observed`, `system`) on every entry; observed
  entries are always treated as untrusted data, never instructions.
- Excludes passwords, keys, and tokens (§5.3).
- Lets the user inspect, query, update, and delete entries (deletion is
  IRREVERSIBLE and always requires explicit approval per §5.2).
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

from autum.events import Provenance, new_id
from autum.logging import redact_sensitive
from autum.tools import Capability, PermissionLevel, ToolRegistry

MAX_MEMORIES = 500
_SECRET_KEY_WORDS = ("password", "passwd", "secret", "token", "api_key", "apikey", "private_key", "credential")
_WORD_RE = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class MemoryEntry:
    id: str
    key: str
    value: str
    category: str
    provenance: str  # "user" | "observed" | "system"
    created_ms: int
    updated_ms: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _check_not_secret(key: str, value: str) -> None:
    key_l = key.lower()
    if any(w in key_l for w in _SECRET_KEY_WORDS):
        raise ValueError("refusing to store secrets, passwords, or tokens in memory (plan §5.3)")
    if redact_sensitive(key) != key or redact_sensitive(value) != value:
        raise ValueError("refusing to store passwords, keys, or tokens in memory (plan §5.3)")


class MemoryStore:
    """Local JSON store for user-selected persistent memories."""

    def __init__(self, path: Path) -> None:
        self.path = path.resolve()
        self._entries: dict[str, MemoryEntry] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            raw = json.loads(self.path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return
        if not isinstance(raw, list):
            return
        for item in raw:
            if isinstance(item, dict) and {"id", "key", "value", "category", "provenance", "created_ms", "updated_ms"} <= item.keys():
                entry = MemoryEntry(
                    id=str(item["id"]),
                    key=str(item["key"]),
                    value=str(item["value"]),
                    category=str(item["category"]),
                    provenance=Provenance(item["provenance"]).value,
                    created_ms=int(item["created_ms"]),
                    updated_ms=int(item["updated_ms"]),
                )
                self._entries[entry.id] = entry

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = [e.to_dict() for e in self._entries.values()]
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(self.path)

    def remember(
        self,
        key: str,
        value: str,
        *,
        category: str = "general",
        provenance: Provenance = Provenance.USER,
    ) -> MemoryEntry:
        clean_key = key.strip()
        clean_val = value.strip()
        clean_cat = category.strip().lower() or "general"
        if not clean_key or not clean_val:
            raise ValueError("memory key and value must be non-empty")
        _check_not_secret(clean_key, clean_val)
        now = time.time_ns() // 1_000_000

        # Update existing entry with matching key (case-insensitive)
        existing = self.find_by_key(clean_key)
        if existing is not None:
            # Never silently upgrade an observed memory to USER provenance unless explicitly USER
            updated = MemoryEntry(
                id=existing.id,
                key=clean_key,
                value=clean_val,
                category=clean_cat,
                provenance=provenance.value,
                created_ms=existing.created_ms,
                updated_ms=now,
            )
            self._entries[existing.id] = updated
            self._save()
            return updated

        if len(self._entries) >= MAX_MEMORIES:
            raise RuntimeError(f"memory limit reached ({MAX_MEMORIES}); delete unused entries first")
        entry = MemoryEntry(
            id=new_id()[:10],
            key=clean_key,
            value=clean_val,
            category=clean_cat,
            provenance=provenance.value,
            created_ms=now,
            updated_ms=now,
        )
        self._entries[entry.id] = entry
        self._save()
        return entry

    def find_by_key(self, key: str) -> MemoryEntry | None:
        needle = key.strip().lower()
        for entry in self._entries.values():
            if entry.key.lower() == needle:
                return entry
        return None

    def get(self, identifier: str) -> MemoryEntry | None:
        ident = identifier.strip()
        return self._entries.get(ident) or self.find_by_key(ident)

    def list_all(self, category: str | None = None) -> list[MemoryEntry]:
        items = list(self._entries.values())
        if category:
            cat = category.strip().lower()
            items = [e for e in items if e.category == cat]
        return sorted(items, key=lambda e: (e.category, e.key.lower()))

    def search(self, query: str, *, limit: int = 10) -> list[MemoryEntry]:
        terms = set(_WORD_RE.findall(query.lower()))
        if not terms:
            return self.list_all()[:limit]
        scored: list[tuple[int, MemoryEntry]] = []
        for entry in self._entries.values():
            hay = set(_WORD_RE.findall(f"{entry.key} {entry.value} {entry.category}".lower()))
            overlap = len(terms & hay)
            if overlap > 0:
                scored.append((overlap, entry))
        scored.sort(key=lambda pair: (-pair[0], -pair[1].updated_ms))
        return [e for _, e in scored[:limit]]

    def select_for_context(self, query: str, *, max_entries: int = 5, max_chars: int = 600) -> list[MemoryEntry]:
        """Select relevant USER-provenance memories for cross-session context (§5.1 / WP8).

        Never injects observed-provenance entries into the system context.
        """
        candidates = [e for e in self.search(query, limit=max_entries * 2) if e.provenance == Provenance.USER.value]
        if not candidates:
            # Include general preferences when no specific keyword matches
            candidates = [
                e for e in self.list_all("preference") if e.provenance == Provenance.USER.value
            ][:max_entries]
        chosen: list[MemoryEntry] = []
        used = 0
        for entry in candidates:
            line_len = len(entry.key) + len(entry.value) + 4
            if len(chosen) >= max_entries or used + line_len > max_chars:
                break
            chosen.append(entry)
            used += line_len
        return chosen

    def delete(self, identifier: str) -> MemoryEntry:
        entry = self.get(identifier)
        if entry is None:
            raise LookupError(f"no memory matching {identifier!r}")
        self._entries.pop(entry.id, None)
        self._save()
        return entry


def register_memory_capabilities(registry: ToolRegistry, store: MemoryStore) -> None:
    """Register WP8 persistent memory capabilities in the ToolRegistry."""

    # 1. Remember / update user fact (REVERSIBLE)
    async def _remember(args: dict[str, Any]) -> dict[str, Any]:
        prov = Provenance.OBSERVED if args.get("from_observed") else Provenance.USER
        entry = store.remember(
            str(args["key"]),
            str(args["value"]),
            category=str(args.get("category", "general")),
            provenance=prov,
        )
        return entry.to_dict()

    async def _verify_remember(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        reloaded = MemoryStore(store.path).get(result["id"])
        return reloaded is not None and reloaded.value == result["value"]

    registry.register(
        Capability(
            name="remember_fact",
            description="Save or update a user-selected persistent fact across sessions (no secrets).",
            input_schema={
                "type": "object",
                "properties": {
                    "key": {"type": "string", "minLength": 1, "maxLength": 80},
                    "value": {"type": "string", "minLength": 1, "maxLength": 500},
                    "category": {"type": "string", "minLength": 1, "maxLength": 40},
                    "from_observed": {"type": "boolean"},
                },
                "required": ["key", "value"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["id", "key", "value", "provenance"]},
            permission_level=PermissionLevel.REVERSIBLE,
            cancellable=False,
            execute=_remember,
            verify=_verify_remember,
            target_summary=lambda a: f"remember '{a['key']}'",
        )
    )

    # 2. Recall memories (READ_ONLY, observed_output=True because stored entries may include observed data)
    async def _recall(args: dict[str, Any]) -> dict[str, Any]:
        query = str(args.get("query") or "")
        entries = store.search(query) if query else store.list_all(args.get("category"))
        return {"memories": [e.to_dict() for e in entries]}

    async def _verify_recall(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return isinstance(result.get("memories"), list)

    registry.register(
        Capability(
            name="recall_memories",
            description="Search or list saved persistent memories.",
            input_schema={
                "type": "object",
                "properties": {
                    "query": {"type": "string", "maxLength": 200},
                    "category": {"type": "string", "maxLength": 40},
                },
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["memories"]},
            permission_level=PermissionLevel.READ_ONLY,
            observed_output=True,
            execute=_recall,
            verify=_verify_recall,
            target_summary=lambda _: "persistent memory",
        )
    )

    # 3. Forget / delete a memory (IRREVERSIBLE — always requires explicit user approval)
    async def _forget(args: dict[str, Any]) -> dict[str, Any]:
        removed = store.delete(str(args["identifier"]))
        return {"id": removed.id, "key": removed.key, "deleted": True}

    async def _verify_forget(_args: dict[str, Any], result: dict[str, Any]) -> bool:
        return MemoryStore(store.path).get(result["id"]) is None

    registry.register(
        Capability(
            name="forget_memory",
            description="Permanently delete a saved memory by ID or key (irreversible; requires approval).",
            input_schema={
                "type": "object",
                "properties": {
                    "identifier": {"type": "string", "minLength": 1, "maxLength": 80},
                },
                "required": ["identifier"],
                "additionalProperties": False,
            },
            output_schema={"type": "object", "required": ["id", "key", "deleted"]},
            permission_level=PermissionLevel.IRREVERSIBLE,
            cancellable=False,
            precondition=lambda a: None if store.get(str(a["identifier"])) is not None else f"no memory matching {a['identifier']!r}",
            execute=_forget,
            verify=_verify_forget,
            target_summary=lambda a: f"delete memory '{a['identifier']}'",
        )
    )
