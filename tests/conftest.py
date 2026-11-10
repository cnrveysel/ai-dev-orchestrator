import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Tests must never spend API/subscription usage: launching the real Claude CLI or
# Codex (NInfer or normal) is refused for the whole test session.
FORBIDDEN_EXECUTABLES = {"claude", "claude.exe", "codex", "codex.exe", "codex.cmd", "codex.ps1",
                         "codex-ninfer", "codex-ninfer.exe", "cline", "cline.exe", "cline.cmd", "cline.ps1"}


class RealModelLaunchBlocked(AssertionError):
    pass


def _first_token(cmd: str) -> str:
    cmd = cmd.strip()
    if cmd.startswith('"'):
        return cmd[1:].split('"', 1)[0]
    return cmd.split(" ", 1)[0] if cmd else ""


@pytest.fixture(autouse=True)
def _no_real_model_processes(monkeypatch):
    from app.workers import process_runner

    original = process_runner.ProcessRunner.run

    async def guarded(self, argv, *args, **kwargs):
        exe = argv[0] if argv else _first_token(kwargs.get("_shell_command") or "")
        if os.path.basename(exe).lower() in FORBIDDEN_EXECUTABLES:
            raise RealModelLaunchBlocked(f"test tried to launch a real model CLI: {exe}")
        return await original(self, argv, *args, **kwargs)

    monkeypatch.setattr(process_runner.ProcessRunner, "run", guarded)
