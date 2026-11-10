"""Runs a task's verification commands (build/test) after each worker attempt.

Commands come from Claude's plan and are shell command lines, so they are run
through the platform shell (cmd.exe /c) - the only place the app uses a shell.
Destructive commands are NOT executed; they are surfaced as needing approval.
"""
from __future__ import annotations

import time
from typing import Callable

from app.models.attempt import VerificationResult
from app.orchestration.safety import destructive_reason
from app.workers.process_runner import ProcessRunner, tail

LogFn = Callable[[str, str], None]


class Verifier:
    def __init__(self, log: LogFn, timeout: float = 900):
        self.log = log
        self.timeout = timeout
        self._runner: ProcessRunner | None = None

    async def run(self, commands: list[str], cwd: str, enabled: bool = True) -> list[VerificationResult]:
        results: list[VerificationResult] = []
        for cmd in commands:
            cmd = cmd.strip()
            if not cmd:
                continue
            if not enabled:
                results.append(VerificationResult(cmd, None, skipped_reason="verification commands disabled in Settings"))
                self.log("VERIFY", f"skipped (disabled): {cmd}")
                continue
            reason = destructive_reason(cmd)
            if reason:
                results.append(VerificationResult(cmd, None, skipped_reason=f"needs user approval: {reason}"))
                self.log("VERIFY", f"NOT RUN - destructive command needs approval ({reason}): {cmd}")
                continue
            self.log("VERIFY", f"$ {cmd}")
            start = time.time()
            lines: list[str] = []

            def on_line(line: str) -> None:
                lines.append(line)
                self.log("VERIFY", line)

            self._runner = ProcessRunner()
            try:
                res = await self._runner.run_shell(cmd, cwd=cwd, timeout=self.timeout,
                                             on_stdout=on_line, on_stderr=on_line)
            finally:
                self._runner = None
            code = res.exit_code if not res.timed_out else None
            out = "\n".join(lines)
            if res.timed_out:
                out += f"\n[timed out after {self.timeout}s]"
            results.append(VerificationResult(cmd, code, tail(out, 8000), time.time() - start))
            self.log("VERIFY", f"exit={code}{' (timeout)' if res.timed_out else ''} in {time.time() - start:.1f}s")
        return results

    async def cancel(self) -> None:
        if self._runner:
            await self._runner.cancel()
