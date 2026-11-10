"""Async subprocess runner with live stdout/stderr streaming, timeouts and
cancellation (graceful first, then a forced process-tree kill).

Commands are passed as argv lists (no shell), except run_shell() which exists
only for verification command lines. Batch files are refused
because CreateProcess would route them through cmd.exe with its own parsing.
"""
from __future__ import annotations

import asyncio
import codecs
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass, field
from typing import Callable, Mapping

LineCallback = Callable[[str], None]

IS_WINDOWS = sys.platform == "win32"


class ProcessLaunchError(RuntimeError):
    pass


@dataclass
class ProcessResult:
    argv: list[str]
    exit_code: int | None
    stdout: str
    stderr: str
    duration: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    pid: int | None = None
    pipes_held_open: bool = False  # exited, but descendants still held stdout/stderr
    terminated_reason: str | None = None  # set when stopped via terminate(reason)

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and not self.cancelled


@dataclass
class _Capture:
    limit: int
    parts: list[str] = field(default_factory=list)
    size: int = 0
    truncated: bool = False

    def add(self, line: str) -> None:
        if self.size >= self.limit:
            self.truncated = True
            return
        self.parts.append(line)
        self.size += len(line) + 1

    def text(self) -> str:
        return "\n".join(self.parts)


def build_env(overrides: Mapping[str, str | None] | None) -> dict[str, str]:
    """Copy of the current environment with child-only overrides applied.
    Never mutates os.environ."""
    env = dict(os.environ)
    for k, v in (overrides or {}).items():
        if v is None:
            env.pop(k, None)
        else:
            env[k] = v
    return env


class ProcessRunner:
    """Runs ONE process at a time. Create a new instance per concurrent use."""

    def __init__(self, grace_seconds: float = 5.0, drain_seconds: float = 5.0):
        self.grace_seconds = grace_seconds
        self.drain_seconds = drain_seconds
        self._proc: asyncio.subprocess.Process | None = None
        self._cancel_requested = False
        self._terminate_reason: str | None = None

    @property
    def running(self) -> bool:
        return self._proc is not None and self._proc.returncode is None

    @property
    def pid(self) -> int | None:
        return self._proc.pid if self._proc else None

    async def run(
        self,
        argv: list[str],
        *,
        cwd: str | None = None,
        env_overrides: Mapping[str, str | None] | None = None,
        stdin_data: str | None = None,
        timeout: float | None = None,
        on_stdout: LineCallback | None = None,
        on_stderr: LineCallback | None = None,
        capture_limit: int = 8 * 1024 * 1024,
        on_stdin_closed: Callable[[int], None] | None = None,
        _shell_command: str | None = None,
    ) -> ProcessResult:
        if self.running:
            raise RuntimeError("ProcessRunner is already running a process")
        if not argv and _shell_command is None:
            raise ProcessLaunchError("empty command")
        exe = argv[0] if argv else "(shell)"
        if exe.lower().endswith((".cmd", ".bat")):
            raise ProcessLaunchError(f"refusing to launch batch file directly: {exe}")

        self._cancel_requested = False
        self._terminate_reason = None
        kwargs: dict = {}
        if IS_WINDOWS:
            kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.CREATE_NEW_PROCESS_GROUP
        else:
            kwargs["start_new_session"] = True

        start = time.monotonic()
        common = dict(
            cwd=cwd,
            env=build_env(env_overrides),
            stdin=asyncio.subprocess.PIPE if stdin_data is not None else asyncio.subprocess.DEVNULL,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            **kwargs,
        )
        try:
            if _shell_command is not None:
                self._proc = await asyncio.create_subprocess_shell(_shell_command, **common)
            else:
                self._proc = await asyncio.create_subprocess_exec(*argv, **common)
        except (OSError, ValueError) as exc:
            raise ProcessLaunchError(f"failed to start {exe}: {exc}") from exc

        proc = self._proc
        out_cap, err_cap = _Capture(capture_limit), _Capture(capture_limit)

        async def pump(stream: asyncio.StreamReader, cap: _Capture, cb: LineCallback | None) -> None:
            decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
            buf = ""
            while True:
                chunk = await stream.read(65536)
                if not chunk:
                    break
                buf += decoder.decode(chunk)
                *lines, buf = buf.split("\n")
                for line in lines:
                    line = line.rstrip("\r")
                    cap.add(line)
                    if cb:
                        _safe_call(cb, line)
            buf += decoder.decode(b"", final=True)
            if buf:
                buf = buf.rstrip("\r")
                cap.add(buf)
                if cb:
                    _safe_call(cb, buf)

        # Output pumps start BEFORE stdin is written so a chatty child can never deadlock us.
        pumps = [asyncio.ensure_future(pump(proc.stdout, out_cap, on_stdout)),
                 asyncio.ensure_future(pump(proc.stderr, err_cap, on_stderr))]
        deadline = None if timeout is None else start + timeout

        timed_out = False
        cancelled = False
        pipes_held_open = False
        try:
            try:
                if stdin_data is not None:
                    # Write the complete input, then CLOSE stdin (EOF) before waiting for exit.
                    await asyncio.wait_for(self._send_stdin(proc, stdin_data, on_stdin_closed),
                                           timeout=_remaining(deadline))
                await self._wait_exit(proc, deadline)
            except asyncio.TimeoutError:
                timed_out = True
                await self._terminate(proc)
            except asyncio.CancelledError:
                cancelled = True
                await asyncio.shield(self._terminate(proc))
                raise
            # Drain remaining output after exit. Descendants that inherited the pipes can keep
            # them open indefinitely; don't wait on them beyond a short grace period.
            _done, pending = await asyncio.wait(pumps, timeout=self.drain_seconds)
            if pending:
                pipes_held_open = True
                for t in pending:
                    t.cancel()
        finally:
            if cancelled:
                for t in pumps:
                    t.cancel()
            _close_transport(proc)
            self._proc = None

        return ProcessResult(
            argv=list(argv) if argv else [_shell_command or ""],
            exit_code=proc.returncode,
            stdout=out_cap.text(),
            stderr=err_cap.text(),
            duration=time.monotonic() - start,
            timed_out=timed_out,
            cancelled=cancelled or self._cancel_requested,
            truncated=out_cap.truncated or err_cap.truncated,
            pid=proc.pid,
            pipes_held_open=pipes_held_open,
            terminated_reason=self._terminate_reason,
        )

    @staticmethod
    async def _send_stdin(proc: asyncio.subprocess.Process, data: str,
                          on_closed: Callable[[int], None] | None) -> None:
        stdin = proc.stdin
        assert stdin is not None
        payload = data.encode("utf-8")
        sent = 0
        try:
            try:
                stdin.write(payload)
                await stdin.drain()
                sent = len(payload)
            except (BrokenPipeError, ConnectionResetError):
                pass  # the child exited or closed its stdin early
            stdin.close()  # EOF for the child
            try:
                await stdin.wait_closed()  # buffered bytes flushed and the handle really closed
            except (BrokenPipeError, ConnectionResetError, OSError):
                pass
        except BaseException:
            # Timed out / cancelled while the child wasn't reading: drop unsent bytes and close
            # now instead of waiting for a flush that may never happen.
            stdin.transport.abort()
            raise
        if on_closed:
            _safe_call(on_closed, sent)

    @staticmethod
    async def _wait_exit(proc: asyncio.subprocess.Process, deadline: float | None) -> None:
        # Not proc.wait(): on Python 3.12 that only returns once every pipe is closed too,
        # so a descendant holding stdout/stderr would block it forever. returncode is set as
        # soon as the process itself exits.
        while proc.returncode is None:
            if deadline is not None and time.monotonic() >= deadline:
                raise asyncio.TimeoutError
            await asyncio.sleep(0.05)

    async def run_shell(self, command: str, **kwargs) -> ProcessResult:
        """Run a shell command line (cmd.exe /c on Windows). Only used for
        verification commands, which are inherently shell command lines;
        everything else uses argv lists via run()."""
        return await self.run([], _shell_command=command, **kwargs)

    async def cancel(self) -> None:
        """Request cancellation of the running process (graceful, then forced)."""
        self._cancel_requested = True
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await self._terminate(proc)

    async def terminate(self, reason: str) -> None:
        """Stop the running process for a caller-defined reason (e.g. a usage limit).
        Unlike cancel(), the result is not marked as cancelled; the reason is recorded."""
        if self._terminate_reason is None:
            self._terminate_reason = reason
        proc = self._proc
        if proc is not None and proc.returncode is None:
            await self._terminate(proc)

    async def _terminate(self, proc: asyncio.subprocess.Process) -> None:
        if proc.returncode is not None:
            return
        # 1) graceful
        try:
            if IS_WINDOWS:
                # Only reaches processes sharing our console; harmless otherwise.
                os.kill(proc.pid, signal.CTRL_BREAK_EVENT)
            else:
                os.killpg(proc.pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, AttributeError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=self.grace_seconds)
            return
        except asyncio.TimeoutError:
            pass
        # 2) forced: kill the whole tree (worker spawns shells, MCP servers, ...)
        if IS_WINDOWS:
            try:
                killer = await asyncio.create_subprocess_exec(
                    "taskkill", "/PID", str(proc.pid), "/T", "/F",
                    stdout=asyncio.subprocess.DEVNULL, stderr=asyncio.subprocess.DEVNULL,
                    creationflags=subprocess.CREATE_NO_WINDOW)
                await asyncio.wait_for(killer.wait(), timeout=15)
            except Exception:
                pass
        else:
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass
        try:
            proc.kill()
        except (OSError, ProcessLookupError):
            pass
        try:
            await asyncio.wait_for(proc.wait(), timeout=10)
        except asyncio.TimeoutError:
            pass


async def terminate_process(proc: asyncio.subprocess.Process, grace_seconds: float = 5.0) -> None:
    """Graceful stop, then a forced process-tree kill (the same policy ProcessRunner uses). For
    processes started outside ProcessRunner, e.g. V4 evidence resources."""
    await ProcessRunner(grace_seconds=grace_seconds)._terminate(proc)


def _remaining(deadline: float | None) -> float | None:
    return None if deadline is None else max(0.0, deadline - time.monotonic())


def _close_transport(proc: asyncio.subprocess.Process) -> None:
    """Close our ends of the pipes (never kills an exited process's descendants)."""
    transport = getattr(proc, "_transport", None)
    if transport is not None:
        try:
            transport.close()
        except Exception:
            pass


def _safe_call(cb, line) -> None:
    try:
        cb(line)
    except Exception:  # a UI callback must never kill the pump
        pass


def tail(text: str, max_chars: int = 4000) -> str:
    if len(text) <= max_chars:
        return text
    return "...[truncated]...\n" + text[-max_chars:]
