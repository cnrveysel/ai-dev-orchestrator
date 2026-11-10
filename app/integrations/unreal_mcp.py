"""Unreal MCP reachability probe. Only relevant for Unreal projects; never required."""
from __future__ import annotations

import asyncio
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path


@dataclass
class UnrealStatus:
    is_unreal_project: bool
    uproject: str = ""
    mcp_reachable: bool | None = None  # None = not checked (non-Unreal project)
    detail: str = ""

    def describe(self) -> str:
        if not self.is_unreal_project:
            return "n/a (not an Unreal project)"
        if self.mcp_reachable:
            return f"reachable ({self.uproject})"
        return f"not reachable - degraded ({self.detail})" if self.detail else "not reachable - degraded"


def find_uproject(repo_path: str) -> str:
    try:
        found = sorted(Path(repo_path).glob("*.uproject"))
    except OSError:
        return ""
    return found[0].name if found else ""


def _reachable(url: str, timeout: float) -> tuple[bool, str]:
    try:
        urllib.request.urlopen(url, timeout=timeout)  # noqa: S310
        return True, ""
    except urllib.error.HTTPError as exc:  # server answered (e.g. 405/406 for GET on MCP)
        return True, f"HTTP {exc.code}"
    except Exception as exc:
        return False, str(getattr(exc, "reason", exc))


async def probe(repo_path: str | None, url: str, timeout: float = 2.0) -> UnrealStatus:
    uproject = find_uproject(repo_path) if repo_path else ""
    if not uproject:
        return UnrealStatus(False)
    ok, detail = await asyncio.to_thread(_reachable, url, timeout)
    return UnrealStatus(True, uproject, ok, "" if ok else detail)
