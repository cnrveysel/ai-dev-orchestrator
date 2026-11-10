"""Generic lifecycle of evidence resources: something that must run while evidence is observed.

    START -> READY_PROBE -> OBSERVE -> CAPTURE -> STOP

Providers implement start / ready / capture / stop for their lifecycle capabilities (readiness probes:
see readiness.py). The ResourceManager drives it, persists every started resource (pid + creation
time) BEFORE waiting on it, and always stops what it started. After a crash, reap_stale() stops
leftovers that are provably ours; a resource from an interrupted collection never counts as evidence.
Steps that observe a running resource receive its handle (ExecContext.resources) - that is how one
provider observes an application another provider started.
"""
from __future__ import annotations

import asyncio
import codecs
import subprocess
import time
from collections import deque
from typing import Any, Callable

from app.evidence import procutil
from app.evidence.model import EvidenceState
from app.workers.process_runner import IS_WINDOWS, build_env, terminate_process

LIVE_STATES = frozenset({"STARTING", "READY", "NOT_READY", "OBSERVING"})


class ServiceProcess:
    """A background process with bounded output capture: a declared shell command line, or an argv list
    built by a provider from validated paths (no shell involved)."""

    def __init__(self, command: str | list[str], cwd: str, capture_chars: int = 200_000):
        self.command, self.cwd, self.capture_chars = command, cwd, capture_chars
        self.started_monotonic = 0.0
        self.proc: asyncio.subprocess.Process | None = None
        self.pid: int | None = None
        self.start_time: float | None = None
        self._chunks: deque[str] = deque()
        self._size = 0
        self._dropped = False
        self._seen: list[str] = []  # markers found so far
        self._markers: list[str] = []
        self._pump: asyncio.Task | None = None

    async def start(self, markers: list[str] | None = None) -> None:
        self._markers = [m for m in markers or [] if m]
        kwargs: dict[str, Any] = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True
        common = dict(cwd=self.cwd, env=build_env(None), stdin=asyncio.subprocess.DEVNULL,
                      stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.STDOUT, **kwargs)
        if isinstance(self.command, list):
            self.proc = await asyncio.create_subprocess_exec(*self.command, **common)
        else:
            self.proc = await asyncio.create_subprocess_shell(self.command, **common)
        self.pid = self.proc.pid
        self.start_time = procutil.start_time(self.pid)
        self.started_monotonic = time.monotonic()
        self._pump = asyncio.ensure_future(self._read())

    async def _read(self) -> None:
        assert self.proc is not None and self.proc.stdout is not None
        decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        while True:
            chunk = await self.proc.stdout.read(65536)
            if not chunk:
                break
            text = decoder.decode(chunk).replace("\r\n", "\n")
            self._chunks.append(text)
            self._size += len(text)
            while self._size > self.capture_chars and len(self._chunks) > 1:  # keep the tail
                self._size -= len(self._chunks.popleft())
                self._dropped = True
            joined = "".join(list(self._chunks)[-2:])
            for m in self._markers:
                if m not in self._seen and m in joined:
                    self._seen.append(m)

    @property
    def running(self) -> bool:
        return self.proc is not None and self.proc.returncode is None

    @property
    def exit_code(self) -> int | None:
        return self.proc.returncode if self.proc else None

    def output(self) -> str:
        return ("...[earlier output dropped]...\n" if self._dropped else "") + "".join(self._chunks)

    def seen(self, text: str) -> bool:
        """A marker was printed (tracked incrementally for the markers given at start, else the kept tail)."""
        if not text:
            return False
        return text in self._seen or (text not in self._markers and text in "".join(self._chunks))

    async def wait_ready(self, marker: str, timeout: float) -> tuple[str, str]:
        """ready | exited | timeout. Without a marker the process only has to stay alive briefly."""
        start = time.monotonic()
        settle = min(1.0, max(0.1, timeout))
        while True:
            if marker and marker in self._seen:
                return "ready", f"ready marker {marker!r} seen"
            if not self.running:
                return "exited", f"exited with code {self.exit_code} before it was ready"
            elapsed = time.monotonic() - start
            if not marker and elapsed >= settle:
                return "ready", f"still running after {settle:.1f}s (no ready marker declared)"
            if elapsed >= timeout:
                return "timeout", f"not ready within {timeout:.0f}s"
            await asyncio.sleep(0.05)

    async def stop(self, grace: float = 2.0) -> None:
        if self.proc is None:
            return
        if self.proc.returncode is None:
            await asyncio.shield(terminate_process(self.proc, grace))
        if self._pump is not None:
            try:
                await asyncio.wait_for(asyncio.shield(self._pump), timeout=2)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                self._pump.cancel()
        transport = getattr(self.proc, "_transport", None)
        if transport is not None:
            try:
                transport.close()
            except Exception:
                pass


class ResourceManager:
    """Drives provider lifecycles for one evidence collection and records them in the state."""

    def __init__(self, state: EvidenceState, persist: Callable[[], None], log: Callable[[str], None]):
        self.state, self.persist, self.log = state, persist, log
        self.active: list[tuple[dict[str, Any], Any, Any, Any]] = []  # (record, provider, capability, handle)

    async def start(self, provider: Any, cap: Any, requirement_id: str, ctx: Any) -> tuple[Any, str, str]:
        """START + READY_PROBE. Returns (handle | None, ready | exited | timeout | unavailable, detail)."""
        rec = {"id": f"res-{len(self.state.resources) + 1}", "capability_id": cap.id, "requirement_id": requirement_id,
               "pid": None, "start_time": None, "state": "STARTING", "started_at": time.time(), "stopped_at": 0.0,
               "detail": ""}
        self.state.resources.append(rec)
        self.persist()
        try:
            handle = await provider.start(cap, ctx)
        except asyncio.CancelledError:
            rec["state"] = "FAILED"
            self.persist()
            raise
        except Exception as exc:
            rec["state"], rec["detail"] = "FAILED", f"could not start: {exc}"[:300]
            self.persist()
            return None, "unavailable", rec["detail"]
        rec["pid"], rec["start_time"] = getattr(handle, "pid", None), getattr(handle, "start_time", None)
        self.active.append((rec, provider, cap, handle))
        self.persist()  # the pid is on disk before we wait on it: a crash can never leak it silently
        status, detail = await provider.ready(handle, cap, ctx)
        rec["state"], rec["detail"] = ("READY" if status == "ready" else "NOT_READY"), detail[:300]
        self.persist()
        self.log(f"resource {rec['id']} ({cap.id}) {status}: {detail}")
        return handle, status, detail

    def capture(self, handle: Any) -> str:
        for rec, provider, _cap, h in self.active:
            if h is handle:
                return provider.capture(h)
        return ""

    def handles_for(self, requirement_id: str) -> dict[str, Any]:
        """capability key -> handle of the READY resources started for one requirement."""
        return {cap.key: h for rec, _p, cap, h in self.active
                if rec.get("requirement_id") == requirement_id and rec.get("state") == "READY"}

    async def stop_all(self) -> None:
        """STOP everything this collection started (also on errors and cancellation)."""
        while self.active:
            rec, provider, cap, handle = self.active.pop()
            try:
                await provider.stop(handle)
            except Exception as exc:  # never let cleanup hide the original outcome
                rec["detail"] = f"stop failed: {exc}"[:300]
            rec["state"], rec["stopped_at"] = "STOPPED", time.time()
            self.persist()

    def reap_stale(self) -> list[str]:
        """After a crash: stop leftovers of an interrupted collection that are provably ours."""
        notes = []
        for rec in self.state.resources:
            if rec.get("state") not in LIVE_STATES:
                continue
            outcome = procutil.reap(rec.get("pid"), rec.get("start_time"))
            rec["state"] = {"killed": "REAPED", "gone": "GONE"}.get(outcome, "ABANDONED")
            rec["stopped_at"] = time.time()
            notes.append(f"stale resource {rec['id']} ({rec.get('capability_id')}, pid {rec.get('pid')}) from an "
                         f"interrupted collection: {outcome}")
        if notes:
            self.persist()
        return notes
