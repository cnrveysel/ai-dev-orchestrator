"""Capability registry: the providers AI Studio knows, and bounded discovery across them.

Discovery only REPORTS what is available; it never decides what evidence a phase needs (that is
the verification contract and Claude's evidence plan). It is bounded in time and size, is
read-only for the repository and never launches the project.
"""
from __future__ import annotations

import asyncio
import shutil
import sys
import time
from typing import Callable

from app.evidence.model import CONFIRMED, CONFIDENCES, INFERRED, Capability, CapabilitySnapshot
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext
from app.evidence.providers.browser import BrowserProvider
from app.evidence.providers.command import CommandProvider
from app.evidence.providers.files import RepositoryFileProvider
from app.evidence.providers.runtime import RuntimeProvider
from app.evidence.providers.unreal import UnrealProvider
from app.evidence.providers.visual import VisualProvider

_RANK = {c: i for i, c in enumerate(CONFIDENCES)}  # CONFIRMED < INFERRED < UNAVAILABLE


class CapabilityRegistry:
    def __init__(self, providers: list[CapabilityProvider]):
        self.providers = list(providers)
        self._by_name = {p.name: p for p in self.providers}

    def provider(self, name: str) -> CapabilityProvider | None:
        return self._by_name.get(name)

    async def discover(self, ctx: DiscoveryContext, *, timeout: float = 20.0, max_capabilities: int = 40,
                       known_confirmed: dict[str, str] | None = None) -> CapabilitySnapshot:
        """Ask every provider, within ONE shared time budget. `known_confirmed` (capability key -> why) carries
        capabilities already confirmed by execution earlier in this run."""
        known = known_confirmed or {}
        deadline = time.monotonic() + max(0.1, timeout)
        found: dict[str, Capability] = {}
        statuses: dict[str, str] = {}
        for p in self.providers:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                statuses[p.name] = "skipped: discovery time budget used"
                continue
            try:
                caps = await asyncio.wait_for(p.discover(ctx), timeout=remaining)
            except asyncio.TimeoutError:
                statuses[p.name] = f"timeout after {remaining:.1f}s"
                continue
            except asyncio.CancelledError:
                raise
            except Exception as exc:  # one broken provider never hides the others
                statuses[p.name] = f"error: {exc}"[:200]
                continue
            notes = list(getattr(p, "last_notes", []) or [])
            statuses[p.name] = f"ok ({len(caps)})" + ("; " + "; ".join(notes) if notes else "")
            now = time.time()
            for c in caps:
                c.provider = c.provider or p.name
                c.key = c.key or f"{p.name}:{c.name}"
                c.provenance = {**(c.provenance or {}), "discovered_by": p.name,
                                "provider_version": str(getattr(p, "version", "1")), "discovered_at": now}
                if c.confidence == INFERRED and c.key in known:
                    c.confidence, c.reason = CONFIRMED, known[c.key]
                    c.provenance["confirmed_by"] = known[c.key][:200]
                prev = found.get(c.key)
                if prev is None or _RANK[c.confidence] < _RANK[prev.confidence]:
                    found[c.key] = c
        caps = list(found.values())
        limit = max(1, int(max_capabilities))
        truncated = max(0, len(caps) - limit)
        caps = sorted(caps, key=lambda c: _RANK[c.confidence])[:limit] if truncated else caps
        for i, c in enumerate(caps, 1):
            c.id = f"cap-{i:03d}"
        return CapabilitySnapshot(caps, statuses, truncated, {"platform": sys.platform}, time.time())


def default_registry(which: Callable[[str], str | None] = shutil.which) -> CapabilityRegistry:
    """Every provider AI Studio ships. Each reports only what it can genuinely do here: the runtime provider
    only declared runtimes, the Unreal provider only an Unreal project, the browser provider nothing unless
    enabled. The visual provider can capture a window of a resource AI Studio started (Windows); image
    ANALYSIS stays unavailable - a captured image is never visual evidence by itself."""
    capture = None
    if sys.platform == "win32":
        from app.evidence.wincapture import WindowsWindowCapture
        capture = WindowsWindowCapture()
    return CapabilityRegistry([CommandProvider(which), RepositoryFileProvider(), RuntimeProvider(which),
                               UnrealProvider(), VisualProvider(capture), BrowserProvider()])
