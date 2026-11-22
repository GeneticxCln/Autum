"""Replaceable environment integrations (WP10 / M10).

Core runtime, voice engine, and agent never import environment-specific code
directly. Integrations register capabilities through the documented ToolRegistry
interface, mark all outputs as observed content (§5.1), and can be added,
removed, or replaced without modifying core subsystems.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol, Sequence

from autum.perception import ObservationHook, VisionAnalyzer
from autum.tools import ToolRegistry

log = logging.getLogger("autum.integrations")


@dataclass(frozen=True)
class IntegrationContext:
    vision: VisionAnalyzer | None = None
    allowed_read_dirs: tuple[Path, ...] = ()
    excluded_windows: tuple[str, ...] = (
        "keepassxc", "1password", "bitwarden", "kwallet", "polkit", "pinentry", "password",
    )
    set_observation: ObservationHook | None = None


class Integration(Protocol):
    name: str

    def is_available(self) -> bool:
        """Return True if this integration can run in the current environment."""
        ...

    def register(self, registry: ToolRegistry, context: IntegrationContext) -> list[str]:
        """Register capabilities on registry and return their names."""
        ...


class IntegrationManager:
    """Manages optional, replaceable environment integrations (WP10)."""

    def __init__(self) -> None:
        self._integrations: dict[str, Integration] = {}
        self.loaded: dict[str, list[str]] = {}

    def add(self, integration: Integration) -> None:
        if integration.name in self._integrations:
            raise ValueError(f"integration {integration.name!r} already registered")
        self._integrations[integration.name] = integration

    def replace(self, integration: Integration) -> None:
        self._integrations[integration.name] = integration

    def remove(self, name: str) -> None:
        self._integrations.pop(name, None)

    def available_names(self) -> list[str]:
        return list(self._integrations)

    def load_enabled(
        self,
        registry: ToolRegistry,
        context: IntegrationContext,
        enabled: Sequence[str],
    ) -> list[str]:
        """Load each enabled, available integration without failing the core runtime."""
        active: list[str] = []
        for name in enabled:
            integration = self._integrations.get(name)
            if integration is None:
                log.warning("integration %r is enabled in config but not installed; skipping", name)
                continue
            try:
                if not integration.is_available():
                    log.info("integration %r is unavailable in this environment; skipping", name)
                    continue
                cap_names = integration.register(registry, context)
            except Exception as exc:
                log.warning("integration %r failed to load (%s); continuing without it", name, exc)
                continue
            self.loaded[name] = list(cap_names)
            active.append(name)
        return active

    async def shutdown(self) -> None:
        """Cleanly stop any loaded integrations that hold active resources."""
        for name in list(self.loaded):
            integration = self._integrations.get(name)
            stop_fn = getattr(integration, "shutdown", None)
            if callable(stop_fn):
                try:
                    await stop_fn()
                except Exception as exc:
                    log.warning("integration %r shutdown error: %s", name, exc)
