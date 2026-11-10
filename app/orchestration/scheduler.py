"""Strictly sequential scheduler, implemented as an explicit state machine.

    STARTING_TASK -> WORKER_RUNNING -> VERIFYING -> REVIEWING
        PASS    -> ADVANCING -> STARTING_TASK (next task)
        RETRY   -> RETRY_PENDING -> WORKER_RUNNING (SAME task)
        BLOCKED -> BLOCKED (halt, ask user)

Each state has one handler that returns the next state. The loop checks for
pause/stop requests between stages only, so a running subprocess is never
interrupted by PAUSE. STOP cancels the current stage (killing its process).
There is exactly one worker invocation in flight at any time.
"""
from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from typing import Any, Awaitable, Callable

from app.config.settings import OrchestrationSettings
from app.models.attempt import Attempt
from app.models.review import ReviewDecision
from app.models.run import Run
from app.models.task import ACTIVE_STATUSES, Task, TaskResult, TaskStatus
from app.orchestration.providers.base import OrchestratorError, OrchestratorProvider, ReplanRequest
from app.orchestration.reviewer import ReviewContextTooLarge, ReviewPackage, summarize_previous_attempts
from app.orchestration.state_machine import EXECUTION_STATES, WAITING_STATES, RunState, StateMachine
from app.orchestration.task_graph import TaskGraph
from app.orchestration.verifier import Verifier
from app.persistence.database import Database
from app.workers.base import RetryContext, WorkerError, WorkerProvider, WorkerTaskSpec
from app.workers.profiles import RunWorkerProfiles

S = RunState
EmitFn = Callable[[str, dict[str, Any]], None]
LogFn = Callable[[str, str], None]


class _Halt(Exception):
    """Raised by a handler to halt the run in a specific state."""

    def __init__(self, state: RunState, message: str, resume_state: RunState | None):
        super().__init__(message)
        self.state = state
        self.message = message
        self.resume_state = resume_state


class _Park(Exception):
    """V4: raised by the handler of a WAITING state (AWAITING_HUMAN) that has nothing to do yet. The
    drive loop stops and the run stays in that state until the user acts."""

    def __init__(self, message: str):
        super().__init__(message)
        self.message = message


def pending_human(run: Run) -> dict[str, Any] | None:
    """V4: the unresolved human verification request of the current phase (for the UI), or None."""
    if run.pipeline_version != "v2" or not run.current_task_id or S.AWAITING_HUMAN.value not in (run.state,
                                                                                                  run.resume_state):
        return None
    rec = (run.pipeline_state.get("human") or {}).get(run.current_task_id)
    if not isinstance(rec, dict) or not rec.get("request") or rec.get("status") not in ("awaiting", "feedback_received"):
        return None
    return {"phase_id": run.current_task_id, "status": rec.get("status"), "request": rec.get("request"),
            "feedback": rec.get("feedback")}


def run_payload(run: Run) -> dict[str, Any]:
    """The "run" event payload (shared by the engine and both schedulers)."""
    snap = RunWorkerProfiles.from_dict(run.worker_snapshot).default  # V5: the run's locked worker profile
    return {"run_id": run.id, "state": run.state, "message": run.status_message,
            "current_task_id": run.current_task_id, "started_at": run.started_at,
            "resume_state": run.resume_state, "pipeline_version": run.pipeline_version,
            "human": pending_human(run),
            "worker_profile": {"id": snap.profile_id, "name": snap.display_name, "revision": snap.revision,
                               "runtime": snap.runtime}
            if snap else None}


@dataclass
class SchedulerDeps:
    db: Database
    orchestrator: OrchestratorProvider
    worker: WorkerProvider
    git: Any  # GitRepo | None (None when not a git repo / mock without git)
    verifier: Verifier
    settings: OrchestrationSettings
    emit: EmitFn
    log: LogFn
    codebase_memory_available: bool = False
    unreal_project: bool = False
    unreal_mcp_reachable: bool | None = None
    # V2 only: exact project lookup / indexing of this repository in codebase-memory.
    codebase_memory_executable: str = ""
    codebase_memory_project: str = ""
    auto_index_codebase_memory: bool = False
    codebase_memory_index_timeout: float = 900
    # V4 pass 2: capability providers for evidence collection, and where evidence artifacts are stored
    # (AI Studio's data folder - never the target repository). None / "" = defaults / no artifact files.
    evidence_registry: Any = None
    artifact_root: str = ""


class Scheduler:
    completion_message = "all tasks passed"
    # States the drive loop dispatches to a handler (V4 phase runs add the AWAITING_HUMAN waiting state).
    dispatch_states: frozenset[RunState] = EXECUTION_STATES

    def __init__(self, run: Run, sm: StateMachine, deps: SchedulerDeps,
                 attempts: dict[str, list[Attempt]] | None = None):
        assert run.plan is not None
        self.run = run
        self.sm = sm
        self.d = deps
        self.graph = TaskGraph(run.plan.tasks)
        self.attempts: dict[str, list[Attempt]] = attempts or {}
        self._pause_requested = False
        self._stop_requested = False
        self._step: asyncio.Task | None = None
        self._degraded_notes: list[str] = []
        self._handlers: dict[RunState, Callable[[], Awaitable[RunState]]] = {
            S.STARTING_TASK: self._on_starting_task,
            S.WORKER_RUNNING: self._on_worker_running,
            S.VERIFYING: self._on_verifying,
            S.REVIEWING: self._on_reviewing,
            S.RETRY_PENDING: self._on_retry_pending,
            S.ADVANCING: self._on_advancing,
        }

    # ------------------------------------------------------------ control

    def request_pause(self) -> None:
        self._pause_requested = True
        self.d.log("SYSTEM", "PAUSE requested: the current stage will finish, then the run pauses.")

    def request_stop(self) -> None:
        self._stop_requested = True
        self.d.log("SYSTEM", "STOP requested: cancelling the active stage.")
        if self._step is not None and not self._step.done():
            self._step.cancel()

    @property
    def current_task(self) -> Task | None:
        return self.graph.get(self.run.current_task_id) if self.run.current_task_id else None

    def max_attempts_for(self, task: Task) -> int:
        return self.d.settings.max_attempts + task.extra_attempts

    # ------------------------------------------------------------ main loop

    async def drive(self, entry: RunState) -> RunState:
        """Run from `entry` until the run halts or completes. Returns final state."""
        if self.run.started_at is None:
            self.run.started_at = time.time()
        if self.sm.state != entry:  # V4: the user's answer continues an AWAITING_HUMAN run in place
            self.sm.transition(entry, "resume" if self.sm.state in (S.PAUSED, S.STOPPED, S.BLOCKED, S.ERROR)
                               else "start")
        self._persist_run()
        while self.sm.state in self.dispatch_states:
            if self._stop_requested:
                self._halt(S.STOPPING, "stop requested", self.sm.state)
                break
            if self._pause_requested and self.sm.state not in WAITING_STATES:  # waiting is already a halt
                self._halt(S.PAUSED, "paused by user", self.sm.state)
                break
            current = self.sm.state
            handler = self._handlers[current]
            self._step = asyncio.ensure_future(handler())
            try:
                nxt = await self._step
            except asyncio.CancelledError:
                if not self._stop_requested:
                    raise  # the engine itself is being torn down
                self._halt(S.STOPPING, "stopped by user during " + current.value, current)
                break
            except _Halt as h:
                self._halt(h.state, h.message, h.resume_state)
                break
            except _Park as p:  # stays in the waiting state (persisted); nothing runs until the user acts
                self.run.resume_state = current.value
                self.run.status_message = p.message
                self._persist_run()
                break
            except ReviewContextTooLarge as exc:
                # Claude was not launched: the bounded review context could not fit its hard limit.
                self._halt(S.PAUSED, str(exc), current)
                break
            except OrchestratorError as exc:
                # Claude failed: never let the worker decide; pause and show the error.
                self._halt(S.PAUSED, f"Claude Code CLI failure: {exc}", current)
                break
            except WorkerError as exc:
                self._halt(S.ERROR, f"worker unavailable (no fallback is used): {exc}", current)
                break
            except Exception as exc:  # unexpected bug: surface it, keep state resumable
                self.d.log("SYSTEM", f"internal error in {current.value}: {exc!r}")
                self._halt(S.ERROR, f"internal error: {exc!r}", current)
                break
            finally:
                self._step = None
            self.sm.transition(nxt, f"{current.value} done")
            self.run.resume_state = None
            self._persist_run()
        if self.sm.state == S.STOPPING:
            self.run.interrupted = True
            self.sm.transition(S.STOPPED, "stopped")
            self._persist_run()
        if self.sm.state == S.COMPLETED:
            self.run.finished_at = time.time()
            self.run.status_message = self.completion_message
            self.d.log("SYSTEM", f"Run COMPLETED: {self.completion_message}.")
            self._persist_run()
        return self.sm.state

    def _halt(self, state: RunState, message: str, resume_state: RunState | None) -> None:
        task = self.current_task
        if task is not None and state in (S.PAUSED, S.STOPPING, S.ERROR) and task.status in ACTIVE_STATUSES | {TaskStatus.RETRY}:
            task.status = TaskStatus.PAUSED
            self._save_task(task)
        self.run.resume_state = resume_state.value if resume_state else None
        self.run.status_message = message
        self.sm.transition(state, message)
        self.d.log("SYSTEM", f"Run {state.value}: {message}")
        self._persist_run()

    # ------------------------------------------------------------ handlers

    async def _on_starting_task(self) -> RunState:
        active = self.graph.active_task()  # raises if the single-task invariant is broken
        task = active or self.graph.next_ready()
        if task is None:
            if self.graph.all_passed():
                self.run.current_task_id = None
                return S.COMPLETED
            raise _Halt(S.BLOCKED, "no runnable task: remaining tasks have unmet dependencies or failed",
                        S.STARTING_TASK)
        self.run.current_task_id = task.id
        if task.started_at is None:
            task.started_at = time.time()
        self.d.log("SYSTEM", f"Starting task {task.id}: {task.title}")
        self._save_task(task)
        return S.WORKER_RUNNING

    async def _on_worker_running(self) -> RunState:
        task = self._require_task()
        history = self.attempts.setdefault(task.id, [])
        reviewed = [a for a in history if a.review is not None]
        last_reviewed = reviewed[-1] if reviewed else None
        retry_ctx = None
        if last_reviewed and last_reviewed.review.decision != ReviewDecision.PASS:
            retry_ctx = RetryContext(
                attempt_number=task.attempts + 1,
                max_attempts=self.max_attempts_for(task),
                previous_summary=last_reviewed.worker_summary[-3000:],
                problems=last_reviewed.review.problems,
                corrective_instructions=last_reviewed.review.corrective_instructions,
            )
        attempt = Attempt(task_id=task.id, number=len(history) + 1,
                          kind="fix" if retry_ctx else "execute", started_at=time.time())

        # Snapshot BEFORE the worker touches anything, for exact attribution.
        pre = None
        if self.d.git is not None:
            pre = await self.d.git.snapshot(self.d.settings.precise_git_snapshots)
            self.d.log("GIT", f"pre-attempt snapshot: {len(pre.status)} pre-existing changed/untracked path(s)"
                              + (f", tree {pre.tree[:10]}" if pre.tree else ""))

        task.attempts += 1
        task.status = TaskStatus.RUNNING
        self._save_task(task)
        self.d.db.insert_attempt(self.run.id, attempt)
        history.append(attempt)
        self._emit_attempt(attempt)

        spec = WorkerTaskSpec.from_plan(
            self.run.plan, task, self.run.repo_path,
            dependency_results=self.graph.dependency_results(task), retry=retry_ctx,
            codebase_memory_available=self.d.codebase_memory_available,
            unreal_project=self.d.unreal_project, unreal_mcp_reachable=self.d.unreal_mcp_reachable)
        self.d.worker.on_tool_call = lambda n: self.d.emit("tool_calls", {"task_id": task.id, "count": n})
        self.d.log("SYSTEM", f"Worker {'fix' if retry_ctx else 'execute'} {task.id} attempt "
                             f"{task.attempts}/{self.max_attempts_for(task)}")
        try:
            if retry_ctx:
                outcome = await self.d.worker.fix_task(spec)
            else:
                outcome = await self.d.worker.execute_task(spec)
        except asyncio.CancelledError:
            attempt.interrupted = True
            attempt.finished_at = time.time()
            attempt.error = "interrupted by STOP"
            task.attempts -= 1  # interrupted attempts don't consume the retry budget
            self.d.db.update_attempt(attempt)
            self._save_task(task)
            raise
        except WorkerError as exc:
            attempt.finished_at = time.time()
            attempt.error = str(exc)
            task.attempts -= 1
            self.d.db.update_attempt(attempt)
            self._save_task(task)
            raise

        attempt.finished_at = time.time()
        attempt.exit_code = outcome.exit_code
        attempt.executable = outcome.executable
        attempt.command_line = outcome.argv
        attempt.worker_summary = outcome.summary
        attempt.stdout_tail = outcome.stdout_tail
        attempt.stderr_tail = outcome.stderr_tail
        attempt.tool_calls = outcome.tool_calls
        attempt.tool_call_log = outcome.tool_call_log[-200:]
        attempt.worker_usage = outcome.usage
        attempt.error = outcome.error
        self._degraded_notes = list(outcome.degraded_notes)
        self.d.log("SYSTEM", f"Worker exited with code {outcome.exit_code} after {outcome.duration:.1f}s, "
                             f"{outcome.tool_calls} tool call(s)")

        if self.d.git is not None and pre is not None:
            post = await self.d.git.snapshot(self.d.settings.precise_git_snapshots)
            changes = await self.d.git.attempt_changes(pre, post)
            attempt.changed_files = changes.changed_files
            attempt.preexisting_dirty = changes.preexisting_dirty
            attempt.generated_excluded = changes.generated_excluded
            attempt.generated_by_dir = changes.generated_by_dir
            attempt.diff_stat = changes.diff_stat
            attempt.diff = changes.diff
            self.d.log("GIT", f"attempt changed {len(changes.changed_files)} file(s) "
                              f"(pre-existing user changes excluded: {len(changes.preexisting_dirty)}; "
                              f"generated/dependency paths not attributed: {changes.generated_excluded})")
            for c in changes.changed_files[:40]:
                self.d.log("GIT", f"  {c.change} {c.path}" + (f"  [{c.note}]" if c.note else ""))
        self.d.db.update_attempt(attempt)
        self.d.db.add_worker_call(self.run.id, attempt.db_id, outcome, attempt.started_at)
        self._emit_attempt(attempt)
        return S.VERIFYING

    async def _on_verifying(self) -> RunState:
        task = self._require_task()
        attempt = self._last_attempt(task)
        task.status = TaskStatus.VERIFYING
        self._save_task(task)
        if task.verification_commands:
            attempt.verification = await self.d.verifier.run(
                task.verification_commands, self.run.repo_path, self.d.settings.run_verification_commands)
        else:
            self.d.log("VERIFY", f"{task.id}: no verification commands defined")
            attempt.verification = []
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        return S.REVIEWING

    async def _on_reviewing(self) -> RunState:
        task = self._require_task()
        attempt = self._last_attempt(task)
        task.status = TaskStatus.REVIEWING
        self._save_task(task)
        previous = [a for a in self.attempts.get(task.id, []) if a is not attempt]
        package = ReviewPackage(
            task=task, attempt=attempt, max_attempts=self.max_attempts_for(task),
            previous_attempts=summarize_previous_attempts(previous),
            dependency_results=self.graph.dependency_results(task),
            degraded_notes=self._degraded_notes, repo_path=self.run.repo_path)
        try:
            review = await self.d.orchestrator.review(package)
        finally:
            self._record_orchestrator_calls(task.id)
        attempt.review = review
        self.d.db.update_attempt(attempt)
        self.d.db.add_review(self.run.id, attempt)
        self._emit_attempt(attempt)
        self.d.log("CLAUDE", f"Review {task.id} attempt {task.attempts}: {review.decision.value} - {review.summary}")
        for p in review.problems:
            self.d.log("CLAUDE", f"  problem: {p}")
        if review.decision == ReviewDecision.PASS:
            return S.ADVANCING
        if review.decision == ReviewDecision.RETRY:
            task.status = TaskStatus.RETRY
            self._save_task(task)
            return S.RETRY_PENDING
        task.status = TaskStatus.BLOCKED
        self._save_task(task)
        raise _Halt(S.BLOCKED, f"Claude marked {task.id} BLOCKED: {review.summary}", S.RETRY_PENDING)

    async def _on_retry_pending(self) -> RunState:
        task = self._require_task()
        limit = self.max_attempts_for(task)
        if task.attempts >= limit:
            task.status = TaskStatus.FAILED
            self._save_task(task)
            raise _Halt(S.BLOCKED, f"{task.id} reached the retry limit ({limit} attempts) without PASS. "
                                   "Not skipping; resume to grant one more attempt.", S.RETRY_PENDING)
        task.status = TaskStatus.RETRY
        self._save_task(task)
        self.d.log("SYSTEM", f"RETRY {task.id}: attempt {task.attempts + 1}/{limit} with corrective feedback")
        return S.WORKER_RUNNING

    async def _on_advancing(self) -> RunState:
        task = self._require_task()
        attempt = self._last_attempt(task)
        review = attempt.review
        assert review is not None and review.decision == ReviewDecision.PASS
        tr = review.task_result or {}
        task.result = TaskResult(
            task_id=task.id,
            result=str(tr.get("result") or review.summary),
            files_changed=attempt.worker_files(),
            important_decisions=[str(x) for x in tr.get("important_decisions", []) or []],
            remaining_risks=[str(x) for x in tr.get("remaining_risks", []) or []],
            verification=[f"{v.command}: {'pass' if v.passed else (v.skipped_reason or f'exit {v.exit_code}')}"
                          for v in attempt.verification],
        )
        task.status = TaskStatus.PASSED
        task.finished_at = time.time()
        task.user_guidance = ""
        self._save_task(task)
        self.d.log("SYSTEM", f"PASSED {task.id} after {task.attempts} attempt(s)")
        self.run.current_task_id = None
        if review.plan_changes_required:
            await self._replan(f"review of {task.id} reported plan changes are required: {review.summary}")
        self.graph.refresh_ready_flags()
        self._save_all_tasks()
        return S.STARTING_TASK

    # ------------------------------------------------------------ helpers

    async def _replan(self, reason: str) -> None:
        self.d.log("SYSTEM", f"Requesting revised plan: {reason}")
        completed = [t.result for t in self.run.plan.tasks if t.result is not None]
        try:
            new_plan = await self.d.orchestrator.replan(
                ReplanRequest(self.run.plan, completed, reason, self.run.repo_path))
        finally:
            self._record_orchestrator_calls(None)
        merged = merge_replan(self.run.plan.tasks, new_plan.tasks)
        self.run.plan.tasks = merged
        if new_plan.architecture_notes:
            self.run.plan.architecture_notes = new_plan.architecture_notes
        self.graph = TaskGraph(merged)
        self.graph.topological_order()
        self._save_all_tasks()
        self.d.emit("plan", {"plan": self.run.plan.to_dict()})
        self.d.log("SYSTEM", f"Plan revised: {len(merged)} task(s)")

    def _require_task(self) -> Task:
        task = self.current_task
        if task is None:
            raise RuntimeError(f"state {self.sm.state.value} requires a current task")
        return task

    def _last_attempt(self, task: Task) -> Attempt:
        hist = self.attempts.get(task.id) or []
        if not hist or hist[-1].interrupted:
            raise RuntimeError(f"{task.id} has no completed attempt to continue from")
        return hist[-1]

    def _save_task(self, task: Task) -> None:
        self.d.db.save_task(self.run.id, task)
        self.d.emit("task", {"task": task.to_dict()})

    def _save_all_tasks(self) -> None:
        self.d.db.save_tasks(self.run.id, self.run.plan.tasks)
        self.d.emit("tasks", {"tasks": [t.to_dict() for t in self.run.plan.tasks]})

    def _persist_run(self) -> None:
        self.run.state = self.sm.state.value
        self.d.db.update_run(self.run)
        self.d.emit("run", run_payload(self.run))

    def _emit_attempt(self, a: Attempt) -> None:
        self.d.emit("attempt", {"attempt": a.to_dict()})

    def _record_orchestrator_calls(self, task_id: str | None) -> None:
        calls = self.d.orchestrator.calls
        while calls:
            c = calls.pop(0)
            self.d.db.add_orchestrator_call(self.run.id, task_id, c)
            self.d.emit("orchestrator_call", {"kind": c.kind, "cost_usd": c.cost_usd, "usage": c.usage,
                                              "duration": c.duration})


def merge_replan(old: list[Task], new: list[Task]) -> list[Task]:
    """PASSED tasks are immutable; everything else comes from the revised plan."""
    passed = {t.id: t for t in old if t.status == TaskStatus.PASSED}
    old_by_id = {t.id: t for t in old}
    merged: list[Task] = [t for t in old if t.status == TaskStatus.PASSED and t.id not in {n.id for n in new}]
    for n in new:
        if n.id in passed:
            merged.append(passed[n.id])
            continue
        prev = old_by_id.get(n.id)
        n.status = TaskStatus.PENDING
        if prev is not None:  # keep attempt accounting for revised-but-existing tasks
            n.attempts, n.extra_attempts, n.started_at = prev.attempts, prev.extra_attempts, prev.started_at
        merged.append(n)
    for i, t in enumerate(merged):
        t.position = i
    return merged
