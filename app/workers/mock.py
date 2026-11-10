"""Mock worker: streams simulated output and edits NOTHING. Used to exercise the
sequential scheduler, pause/stop and the review loop without a model."""
from __future__ import annotations

import asyncio
import itertools
import time
from typing import Any, Callable

from app.workers.base import LogFn, PhaseTurnRequest, WorkerOutcome, WorkerProvider, WorkerTaskSpec


class MockWorker(WorkerProvider):
    name = "mock-worker"

    def __init__(self, log: LogFn | None = None, delay: float = 0.4, steps: int = 4):
        super().__init__(log)
        self.delay = delay
        self.steps = steps
        self.active = 0  # concurrency probe for tests
        self.max_active = 0
        self.invocations: list[str] = []
        # --- V2 (phase turns) ---
        self.resume_supported = True
        self.turns: list[dict[str, Any]] = []  # every run_turn() call: phase, purpose, session, resumed, prompt
        self.status_script: dict[str, list[str]] = {}  # phase id -> PHASE STATUS per turn (default COMPLETE)
        self.fail_resume = False  # resumed turns fail as if the session were gone
        self.infra_failure = False  # every turn fails immediately without doing anything
        # Test hook only (never set from settings): lets a test edit the repository during a turn.
        self.file_writer: Callable[[PhaseTurnRequest], None] | None = None
        self._ids = itertools.count(1)

    async def execute_task(self, spec: WorkerTaskSpec) -> WorkerOutcome:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        t = spec.task
        self.invocations.append(t.id)
        start = time.time()
        try:
            kind = f"fix (attempt {spec.retry.attempt_number})" if spec.retry else "execute"
            self.log("QWEN", f"[MOCK] {kind} {t.id}: {t.title} - no files are modified in mock mode")
            for i in range(self.steps):
                await asyncio.sleep(self.delay)
                self.log("QWEN", f"[MOCK] step {i + 1}/{self.steps}")
                if self.on_tool_call:
                    self.on_tool_call(i + 1)
            summary = f"SUMMARY: [MOCK] completed {t.id}\nFILES CHANGED: none (mock)"
            return WorkerOutcome(executable="(mock worker)", argv=["(mock)"], exit_code=0, summary=summary,
                                 stdout_tail="", stderr_tail="", duration=time.time() - start,
                                 tool_calls=self.steps, tool_call_log=[f"mock step {i + 1}" for i in range(self.steps)])
        finally:
            self.active -= 1

    async def session_resume_supported(self) -> bool:
        return self.resume_supported

    async def run_turn(self, request: PhaseTurnRequest) -> WorkerOutcome:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.invocations.append(request.phase_id)
        start = time.time()
        resumed = bool(request.session_id) and self.resume_supported
        record = {"phase": request.phase_id, "purpose": request.purpose, "resumed": resumed,
                  "session": request.session_id if resumed else "", "prompt": request.prompt}
        self.turns.append(record)
        try:
            def outcome(**kw: Any) -> WorkerOutcome:
                base = dict(executable="(mock worker)", argv=["(mock)", "resume" if resumed else "exec"], exit_code=0,
                            summary="", stdout_tail="", stderr_tail="", duration=time.time() - start)
                base.update(kw)
                return WorkerOutcome(**base)

            if self.infra_failure:
                await asyncio.sleep(self.delay)
                return outcome(exit_code=1, error="[MOCK] model server unreachable")
            if resumed and self.fail_resume:
                await asyncio.sleep(self.delay)
                return outcome(exit_code=1, resume_failed=True, error="[MOCK] session not found")
            sid = request.session_id if resumed else f"mock-session-{next(self._ids)}"
            record["session"] = sid
            if self.on_session:
                self.on_session(sid)
            mode = "resumed" if resumed else "new"
            self.log("QWEN", f"[MOCK] {request.purpose} {request.phase_id} ({mode} session {sid}) - "
                             "no files are modified in mock mode")
            todo: list[dict[str, Any]] = []
            for i in range(self.steps):
                await asyncio.sleep(self.delay)
                self.log("QWEN", f"[MOCK] step {i + 1}/{self.steps}")
                if self.on_tool_call:
                    self.on_tool_call(i + 1)
                todo = [{"text": f"step {j + 1}", "completed": j <= i} for j in range(self.steps)]
                if self.on_todo:
                    self.on_todo(todo)
            if self.file_writer:
                self.file_writer(request)
            script = self.status_script.get(request.phase_id)
            status = script.pop(0) if script else "COMPLETE"
            summary = (f"PHASE STATUS: {status}\nSUMMARY: [MOCK] {request.purpose} {request.phase_id}\n"
                       "FILES CHANGED: none (mock)")
            return outcome(summary=summary, tool_calls=self.steps, session_id=sid, resumed=resumed, todo=todo,
                           tool_call_log=[f"mock step {i + 1}" for i in range(self.steps)])
        finally:
            self.active -= 1

    async def cancel(self) -> None:
        return None
