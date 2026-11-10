"""Pipeline V2: phase-based execution on top of the V1 scheduler loop.

    STARTING_TASK   select the next phase (strictly in plan order, never skipping one)
    WORKER_RUNNING  ONE Qwen session works the whole phase: the first turn, then continuation
                    turns in the SAME session while the worker reports INCOMPLETE; then a local
                    verification self-check with up to N fix turns in the same session. V4 pass 3: an
                    implementation guard (wall time, tool calls, time without planned-subtask progress)
                    stops a runaway session -> Claude MID_PHASE_DIAGNOSE -> a FRESH session with that
                    guidance; after the maximum diagnoses another trip BLOCKS. When every planned
                    subtask is done and the phase verification is green, an INCOMPLETE / non-zero
                    worker exit is not a reason to keep the worker going: the phase moves on
    VERIFYING       authoritative verification (reuses the self-check when nothing changed since)
    COLLECTING_EVIDENCE  (V4 pass 2; only when the phase's verification contract lists evidence
                    requirements, or human-derived HF-* criteria exist) capability discovery (per
                    attempt) -> Claude EVIDENCE_PLAN over the discovered capability ids -> bounded
                    execution (fresh evidence only; unchanged inputs may be reused) -> at most N bounded
                    REPLANS when a strategy cannot be executed -> evidence bundle for the checkpoint
    REVIEWING       Claude CHECKPOINT of the whole phase (the last phase is the FINAL AUDIT); evidence
                    is re-checked for staleness first
        PASS contradicting its own evidence judgment -> CONSISTENCY_REVIEW (V4 pass 3, bounded, never
                    recursive): PASS (justified) -> ADVANCING, RETRY -> RETRY_PENDING, ASK_USER ->
                    AWAITING_HUMAN, BLOCKED -> BLOCKED
        PASS    -> ADVANCING -> STARTING_TASK        (-> REPLANNING if later phases must change)
        RETRY   -> RETRY_PENDING -> REPAIRING -> VERIFYING -> REVIEWING   (bounded repair, see below)
        BLOCKED -> BLOCKED                           (user decision; RESUME continues the phase's session)
        REPLAN  -> REPLANNING -> STARTING_TASK       (phase SUPERSEDED; FUTURE phases revised)
        ASK_USER -> AWAITING_HUMAN                   (V4, only when the phase's human gate is open)
    AWAITING_HUMAN  the run waits - across restarts, running nothing - for the user's structured answer
                    (LOOKS_GOOD / ISSUE_FOUND / CANNOT_VERIFY + note). Claude HUMAN_FOLLOWUP adjudicates it:
        PASS    -> ADVANCING
        RETRY   -> RETRY_PENDING -> REPAIRING ...        (the existing bounded repair, with the human feedback)
        BLOCKED -> BLOCKED
    REPAIRING       a FRESH Qwen session repairs the checkpoint findings (the work stays on disk). After
                    every turn: protected acceptance tests are enforced, the phase verification runs and is
                    fingerprinted/scored; a strictly better state is recorded as the best-known snapshot and a
                    regression beyond the grace round is restored to it. A guard (wall time, tool calls,
                    unchanged fingerprint, probe files, repeated regression) stops the session and escalates
                    to Claude DIAGNOSE_BLOCKER (REPAIR -> new session with the guidance; BLOCKED -> halt).

The drive loop, PAUSE/STOP handling, halting and persistence are inherited from the V1
Scheduler unchanged. Recovery rules:
  - the Qwen session id is persisted on the phase as soon as the worker reports it, so a
    killed/stopped/paused turn is continued in the same session after RESUME;
  - an interrupted attempt never consumes the retry budget and nothing is ever reset: the
    checkpoint reviews the phase's cumulative changes since the phase started;
  - PASSED and SUPERSEDED phases are history and are never re-run or rewritten by a replan.
"""
from __future__ import annotations

import asyncio
import os
import time
from typing import Any

from app.evidence import freshness as efresh
from app.evidence import model as evm
from app.evidence.artifacts import ArtifactStore
from app.evidence.executor import EvidenceExecutor, EvidenceLimits
from app.evidence.mutation import GitMutationGuard, ManifestMutationGuard, repo_fingerprint
from app.evidence.planning import (
    EvidencePlanPacket, capability_line, demote_failed, deterministic_plan, merge_replan, plan_counts,
    plan_fingerprint, strategy_failures, validate_plan,
)
from app.evidence.providers.base import DiscoveryContext
from app.evidence.registry import CapabilityRegistry, default_registry
from app.integrations import codebase_memory
from app.integrations.git import GitError, snapshot_from_dict, snapshot_to_dict
from app.models import human as hm
from app.models.attempt import Attempt, VerificationResult
from app.models.guard import ImplGuardState, guard_limits, impl_guard_trip
from app.models.phase import (
    TERMINAL_PHASE_STATUSES, Phase, PhaseReport, merge_phase_replan, parse_phase_report, phase_result_line,
)
from app.models.repair import RepairGuardState, RepairState
from app.models.review import FOLLOWUP_DECISIONS, Review, ReviewDecision
from app.models.task import TaskResult, TaskStatus
from app.models.verification import issue_line
from app.orchestration import repair as rp
from app.orchestration.checkpoint import CheckpointPackage
from app.orchestration.consistency import ConsistencyPacket, check_justified, find_conflict
from app.orchestration.human_gate import HumanFollowupPacket
from app.orchestration.impl_guard import MidPhasePacket
from app.orchestration.providers.base import OrchestratorError, PhaseReplanRequest
from app.orchestration.scheduler import S, RunState, Scheduler, _Halt, _Park
from app.orchestration.state_machine import EXECUTION_STATES
from app.orchestration.task_graph import TaskGraph
from app.workers.base import PhaseTurnRequest, RetryContext, WorkerError, WorkerOutcome
from app.workers.phase_prompts import (
    Handoff, PhaseWorkSpec, RepairSpec, build_phase_prompt, build_repair_prompt, build_repair_round_prompt,
    build_turn_prompt, turn_record,
)
from app.workers.process_runner import tail

TERMINAL = TERMINAL_PHASE_STATUSES


class _PauseBetweenTurns(Exception):
    pass


class _InfraFailure(Exception):
    """The worker failed repeatedly without doing any work (e.g. model server down)."""


def _no_work(o: WorkerOutcome) -> bool:
    failed = o.exit_code != 0 or o.timed_out or bool(o.error)
    return failed and o.tool_calls == 0 and not (o.summary or "").strip() and not o.cancelled


class PhaseScheduler(Scheduler):
    completion_message = "all phases passed (final audit passed)"
    dispatch_states = EXECUTION_STATES | {S.AWAITING_HUMAN}

    def __init__(self, run, sm, deps, attempts=None):
        super().__init__(run, sm, deps, attempts)
        self._handlers.update({
            S.RETRY_PENDING: self._on_phase_retry_pending,
            S.REPAIRING: self._on_repairing,
            S.STARTING_TASK: self._on_phase_start,
            S.WORKER_RUNNING: self._on_phase_work,
            S.VERIFYING: self._on_phase_verify,
            S.REVIEWING: self._on_checkpoint,
            S.ADVANCING: self._on_phase_advance,
            S.REPLANNING: self._on_replanning,
            S.AWAITING_HUMAN: self._on_awaiting_human,
            S.COLLECTING_EVIDENCE: self._on_collect_evidence,
            S.CONSISTENCY_REVIEW: self._on_consistency_review,
        })
        self._cm_checked = False
        self._last_post_tree: str | None = None
        self._tool_base = 0
        self._fresh_reason = ""
        self._attempt_todo: list[dict[str, Any]] = []
        self.run.pipeline_state.setdefault("replans", 0)

    # ------------------------------------------------------------ helpers

    @property
    def st(self) -> dict[str, Any]:
        return self.run.pipeline_state

    def _phases(self) -> list[Phase]:
        return sorted(self.run.plan.tasks, key=lambda p: p.position)

    def _next_phase(self) -> Phase | None:
        return next((p for p in self._phases() if p.status not in TERMINAL), None)

    def _index(self, phase: Phase) -> tuple[int, int]:
        ps = self._phases()
        return ps.index(phase) + 1, len(ps)

    def _is_final(self, phase: Phase) -> bool:
        return not any(p.status not in TERMINAL for p in self._phases() if p.position > phase.position)

    def _replan_allowance(self) -> int:
        return max(0, int(self.d.settings.v2_max_replans)) + int(self.st.get("extra_replans", 0))

    def _refresh_ready(self) -> None:
        """Display flag only: the next phase to run is READY, later ones PENDING."""
        nxt = None if self.graph.active_task() else self._next_phase()
        for p in self.run.plan.tasks:
            if p.status == TaskStatus.READY and p is not nxt:
                p.status = TaskStatus.PENDING
        if nxt is not None and nxt.status == TaskStatus.PENDING:
            nxt.status = TaskStatus.READY

    def _check_pause(self) -> None:
        if self._pause_requested:
            raise _PauseBetweenTurns()

    # ------------------------------------------------------------ STARTING_TASK

    async def _on_phase_start(self) -> RunState:
        phase = self.graph.active_task() or self._next_phase()
        if phase is None:
            self.run.current_task_id = None
            return S.COMPLETED
        self.run.current_task_id = phase.id
        if phase.status == TaskStatus.FAILED and phase.attempts >= self.max_attempts_for(phase):
            raise _Halt(S.BLOCKED, f"{phase.id} reached the retry limit ({self.max_attempts_for(phase)} attempts) "
                                   "without PASS. Not skipping; resume to grant one more attempt.", S.RETRY_PENDING)
        if phase.started_at is None:
            phase.started_at = time.time()
        await self._ensure_codebase_memory()
        i, n = self._index(phase)
        self.d.log("SYSTEM", f"Starting phase {phase.id} ({i}/{n}): {phase.title} - {len(phase.subtasks)} subtask(s)"
                             + (f", estimated ~{phase.estimated_minutes} min" if phase.estimated_minutes else ""))
        self._refresh_ready()
        self._save_all_tasks()
        return S.WORKER_RUNNING

    async def _ensure_codebase_memory(self) -> None:
        """Make sure the worker gets the EXACT codebase-memory project of this repository root."""
        if self._cm_checked:
            return
        self._cm_checked = True
        d = self.d
        if not d.codebase_memory_available or not d.codebase_memory_executable:
            return
        if d.codebase_memory_project:
            self.st["codebase_memory_project"] = d.codebase_memory_project
            d.log("MCP", f"codebase-memory: repository already indexed as {d.codebase_memory_project}")
            return
        if not d.auto_index_codebase_memory:
            d.log("MCP", "codebase-memory: this repository root is not indexed and auto-indexing is off; "
                         "the worker is told to index it by its absolute path")
            return
        d.log("MCP", f"codebase-memory: repository root not indexed - indexing {self.run.repo_path} "
                     "(exact absolute path) before the first phase")
        try:
            name, detail = await codebase_memory.ensure_indexed(d.codebase_memory_executable, self.run.repo_path,
                                                                d.codebase_memory_index_timeout)
        except asyncio.CancelledError:
            raise
        except Exception as exc:  # never fatal: the worker can still index it itself
            name, detail = "", str(exc)
        if name:
            d.codebase_memory_project = name
            self.st["codebase_memory_project"] = name
            if detail == codebase_memory.ALREADY_INDEXED:
                d.log("MCP", f"codebase-memory: repository already indexed as {name}")
            else:
                d.log("MCP", f"codebase-memory: repository indexed successfully as {name} ({detail})")
        else:
            d.log("MCP", f"codebase-memory: indexing did not succeed ({detail}); the worker will check "
                         "list_projects / index_repository itself")

    # ------------------------------------------------------------ WORKER_RUNNING

    async def _on_phase_work(self) -> RunState:
        phase = self._require_task()
        history = self.attempts.setdefault(phase.id, [])
        last_reviewed = next((a for a in reversed(history) if a.review is not None), None)
        retry_ctx = None
        if last_reviewed is not None and last_reviewed.review.decision != ReviewDecision.PASS:
            retry_ctx = RetryContext(
                attempt_number=phase.attempts + 1, max_attempts=self.max_attempts_for(phase),
                previous_summary=last_reviewed.worker_summary[-3000:], problems=last_reviewed.review.problems,
                corrective_instructions=last_reviewed.review.corrective_instructions)
        interrupted = bool(history) and history[-1].interrupted
        attempt = Attempt(task_id=phase.id, number=len(history) + 1, kind="fix" if retry_ctx else "execute",
                          started_at=time.time())

        pre = None
        if self.d.git is not None:
            pre = await self.d.git.snapshot(self.d.settings.precise_git_snapshots)
            bases = self.st.setdefault("phase_base", {})
            if phase.id not in bases:  # the phase's diff baseline survives retries, pauses and restarts
                bases[phase.id] = snapshot_to_dict(pre)
                self.d.log("GIT", f"phase {phase.id} baseline snapshot: {len(pre.status)} changed/untracked path(s) "
                                  "from earlier phases or the user are excluded from this phase's review")
            if "run_base" not in self.st:
                self.st["run_base"] = {"tree": pre.tree, "status": dict(pre.status), "hashes": {}}
            self._persist_run()

        phase.attempts += 1
        phase.status = TaskStatus.RUNNING
        phase.reported_status = ""  # the worker's status is per attempt, never carried over
        self._attempt_todo = []
        self._save_task(phase)
        self.d.db.insert_attempt(self.run.id, attempt)
        history.append(attempt)
        self._emit_attempt(attempt)
        self._tool_base = 0
        self.d.log("SYSTEM", f"Phase {phase.id} attempt {phase.attempts}/{self.max_attempts_for(phase)}"
                             + (" - fixing checkpoint feedback" if retry_ctx else "")
                             + (" - continuing after an interruption" if interrupted and not retry_ctx else ""))
        self._impl_prepare(phase, last_reviewed)
        try:
            await self._work_turns(phase, attempt, retry_ctx, interrupted)
            await self._self_check(phase, attempt, retry_ctx)
            attempt.finished_at = time.time()
            await self._attribute(phase, attempt, pre)
        except asyncio.CancelledError:
            self._abandon(phase, attempt, "interrupted by STOP")
            raise
        except _PauseBetweenTurns:
            self._abandon(phase, attempt, "paused between worker turns (the session continues on RESUME)")
            raise _Halt(S.PAUSED, f"paused by user between worker turns of {phase.id}", S.WORKER_RUNNING)
        except _InfraFailure as exc:
            self._abandon(phase, attempt, str(exc))
            raise _Halt(S.ERROR, f"{exc}. Fix the worker/model server, then RESUME.", S.WORKER_RUNNING)
        except Exception as exc:  # WorkerError, GitError (e.g. a stale index.lock), ...
            self._abandon(phase, attempt, f"{type(exc).__name__}: {exc}")
            raise
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        return S.VERIFYING

    def _abandon(self, phase: Phase, attempt: Attempt, reason: str) -> None:
        """An attempt that did not reach its checkpoint: kept for the record, never counted."""
        attempt.interrupted = True
        attempt.finished_at = time.time()
        attempt.error = (attempt.error + "\n" if attempt.error else "") + reason
        phase.attempts = max(0, phase.attempts - 1)
        self.d.db.update_attempt(attempt)
        self._save_task(phase)
        self._emit_attempt(attempt)

    async def _work_turns(self, phase: Phase, attempt: Attempt, retry_ctx: RetryContext | None,
                          interrupted: bool) -> None:
        max_turns = max(1, int(self.d.settings.v2_max_turns_per_attempt))
        purpose = "resume" if interrupted else ("retry" if retry_ctx else "start")
        turns = infra_failures = 0
        while True:
            await self._impl_check(phase, attempt)  # a pending trip / a used-up budget escalates before a new turn
            outcome, report = await self._turn(phase, attempt, purpose, retry_ctx=retry_ctx)
            turns += 1
            g = self._load_impl(phase)
            if g is not None and g.trip:  # the implementation guard stopped this turn
                if await self._done_and_green(phase, attempt):
                    return
                g.pending, g.trip = g.trip, ""
                self._save_impl(g)
                self._check_pause()
                purpose = "continue"  # a FRESH session follows, with the lead's diagnosis
                continue
            if _no_work(outcome):
                infra_failures += 1
                if infra_failures >= 2:
                    raise _InfraFailure(f"the worker failed twice without doing any work (exit {outcome.exit_code}: "
                                        f"{tail(outcome.error or outcome.stderr_tail, 300)})")
                self.d.log("SYSTEM", f"{phase.id}: worker turn failed without doing any work; retrying it once")
                self._check_pause()
                continue
            if report.status in ("COMPLETE", "BLOCKED"):
                return
            if not report.status and outcome.exit_code == 0 and not outcome.timed_out:
                return  # no explicit status: the checkpoint decides
            # V4 pass 3: every planned subtask is done and the orchestrator's verification is green - an
            # INCOMPLETE report or a non-zero worker exit is no reason to keep the worker going.
            if await self._done_and_green(phase, attempt):
                return
            if turns >= max_turns:
                self.d.log("SYSTEM", f"{phase.id}: {max_turns} worker turn(s) used in this attempt without a COMPLETE "
                                     "report; the checkpoint decides")
                return
            self._check_pause()
            purpose = "resume" if (outcome.exit_code != 0 or outcome.timed_out) else "continue"
            self.d.log("SYSTEM", f"{phase.id}: worker reported {report.status or 'no status'}"
                                 f"{' (process ended with exit ' + str(outcome.exit_code) + ')' if outcome.exit_code else ''}"
                                 f" - continuing in the same session (turn {turns + 1}/{max_turns})")

    async def _self_check(self, phase: Phase, attempt: Attempt, retry_ctx: RetryContext | None) -> None:
        """Local verification + fix turns in the same session, before any Claude call."""
        o = self.d.settings
        rounds = max(0, int(o.v2_local_fix_rounds))
        if rounds == 0 or not phase.verification_commands or not o.run_verification_commands:
            return
        if phase.reported_status == "BLOCKED":
            return
        if attempt.verification and attempt.verified_at_turn == len(attempt.turns) and \
                all(v.passed for v in attempt.verification if not v.skipped_reason):
            return  # verified green after the last turn already (see _done_and_green)
        for r in range(rounds + 1):
            self.d.log("VERIFY", f"{phase.id}: self-check before the checkpoint"
                                 + (f" (after local fix round {r})" if r else ""))
            attempt.verification = await self.d.verifier.run(phase.verification_commands, self.run.repo_path, True)
            attempt.verified_at_turn = len(attempt.turns)
            self.d.db.update_attempt(attempt)
            self._emit_attempt(attempt)
            failed = [v for v in attempt.verification if not v.passed and not v.skipped_reason]
            if not failed:
                return
            if r == rounds:
                self.d.log("VERIFY", f"{phase.id}: still failing after {rounds} local fix round(s); the checkpoint decides")
                return
            if self._pause_requested:
                self.d.log("SYSTEM", "PAUSE requested: skipping local fix rounds")
                return
            attempt.local_fix_rounds += 1
            self.d.log("SYSTEM", f"{phase.id}: {len(failed)} verification check(s) failed; local fix round "
                                 f"{r + 1}/{rounds} in the same Qwen session")
            outcome, _ = await self._turn(phase, attempt, "verify_fix", retry_ctx=retry_ctx, failures=failed,
                                          fix_round=r + 1)
            g = self._load_impl(phase)
            if g is not None and g.trip:  # the budget is used up: the checkpoint judges what exists
                self.d.log("SYSTEM", f"{phase.id}: the implementation guard stopped the local fix round ({g.trip}); "
                                     "the checkpoint decides")
                g.trip = ""
                self._save_impl(g)
                return
            if _no_work(outcome):
                self.d.log("SYSTEM", f"{phase.id}: local fix turn failed without doing any work")
                return

    # ------------------------------------------------------------ one Qwen turn

    async def _turn(self, phase: Phase, attempt: Attempt, purpose: str, *, retry_ctx: RetryContext | None = None,
                    failures: list[VerificationResult] | None = None,
                    fix_round: int = 0) -> tuple[WorkerOutcome, PhaseReport]:
        o = self.d.settings
        worker = self.d.worker
        use_session = bool(phase.session_id)
        why_fresh = self._fresh_reason
        self._fresh_reason = ""
        if use_session:
            if not await worker.session_resume_supported():
                why_fresh = "the worker cannot resume sessions"
            elif phase.fresh_session_requested:
                why_fresh = "a fresh session was requested by the user"
            elif phase.session_turns >= max(1, int(o.v2_max_session_turns)):
                why_fresh = f"the session already had {phase.session_turns} turns (bounding context growth)"
            use_session = not why_fresh
        while True:
            if use_session:
                session = phase.session_id
                prompt = build_turn_prompt(purpose, phase, retry=retry_ctx if purpose in ("retry", "resume") else None,
                                           failures=failures, fix_round=fix_round, fix_rounds=o.v2_local_fix_rounds,
                                           codebase_memory_available=self.d.codebase_memory_available,
                                           codebase_memory_project=self.d.codebase_memory_project
                                           if self.d.codebase_memory_available else "")
            else:
                session = ""
                handoff = self._handoff(phase, attempt, purpose, why_fresh)
                g = self._load_impl(phase)
                if g is not None and g.guidance:  # the lead's MID_PHASE_DIAGNOSE for this fresh session
                    handoff = handoff or Handoff(reason=why_fresh or "continuing the phase")
                    handoff.guidance = self._guidance_lines(g.guidance)
                    g.guidance = {}  # delivered with this prompt
                    self._save_impl(g)
                if handoff is not None:
                    self.d.log("SYSTEM", f"{phase.id}: starting a NEW Qwen session with a compact handoff ({handoff.reason})")
                prompt = build_phase_prompt(self._work_spec(phase, retry_ctx, handoff))
                if purpose == "verify_fix":
                    prompt += "\n\n" + build_turn_prompt("verify_fix", phase, failures=failures, fix_round=fix_round,
                                                         fix_rounds=o.v2_local_fix_rounds)
                phase.session_id = ""
                phase.session_turns = 0
                phase.fresh_session_requested = False
                phase.sessions_started += 1
                self._save_task(phase)
            mode = "resumed" if session else "new"
            n = len(attempt.turns) + 1
            attempt.turns.append({"n": n, "purpose": purpose, "mode": mode, "session_id": session, "running": True,
                                  "started_at": time.time()})
            self.d.db.update_attempt(attempt)
            self._emit_attempt(attempt)
            self._wire_callbacks(phase)
            self.d.log("SYSTEM", f"Qwen turn {n} for {phase.id}: {purpose} - "
                                 + (f"resuming session {session}" if session else "new session"))
            started = time.time()
            outcome = await self._impl_run(phase, PhaseTurnRequest(prompt, self.run.repo_path, session, phase.id,
                                                                   purpose))
            if session and outcome.resume_failed:
                self._record_turn(phase, attempt, purpose, mode, outcome, "", started, fix_round)
                self.d.log("SYSTEM", f"{phase.id}: could not resume Qwen session {session}; continuing in a NEW "
                                     "session with a compact handoff")
                phase.session_id = ""
                use_session = False
                why_fresh = "the previous session could not be resumed"
                continue
            break
        if outcome.session_id:
            phase.session_id = outcome.session_id
        phase.session_turns += 1
        if outcome.todo:
            self._attempt_todo = outcome.todo
        report = parse_phase_report(outcome.summary, [s.id for s in phase.subtasks], self._attempt_todo)
        if outcome.todo:
            phase.todo = outcome.todo
        if report.status:
            phase.reported_status = report.status
        self._sync_report_subtasks(phase, report)
        if outcome.context_exhausted:
            self.d.log("SYSTEM", f"{phase.id}: the worker ran out of context; the next turn uses a NEW session "
                                 "with a compact handoff")
            phase.session_id = ""
            self._fresh_reason = "the previous session ran out of context"
        self._record_turn(phase, attempt, purpose, mode, outcome, report.status, started, fix_round)
        self._save_task(phase)
        return outcome, report

    def _sync_report_subtasks(self, phase: Phase, report: PhaseReport) -> None:
        """A planned subtask is marked done ONLY by a structured completion event (the worker's todo
        list, see _on_todo) - never from the final message's prose or checklist. An explicit "[ ] S1"
        line may still reopen one. A COMPLETE report without the structured signals is logged as an
        inconsistency instead of marking everything done."""
        reopened = [s.id for s in phase.subtasks if s.id in report.open and s.done]
        for s in phase.subtasks:
            if s.id in reopened:
                s.done, s.completed_at, s.completed_via = False, 0.0, ""
        if reopened:
            self.d.log("SYSTEM", f"{phase.id}: subtasks {phase.subtasks_done}/{len(phase.subtasks)} after the worker's "
                                 f"report; reopened: {', '.join(reopened)}")
        missing = [s.id for s in phase.subtasks if not s.done]
        if report.status == "COMPLETE" and missing:
            claimed = [x for x in missing if x in report.done]
            self.d.log("SYSTEM", f"{phase.id}: inconsistent progress - the worker reported COMPLETE but sent no "
                                 f"structured completion (todo list) for {', '.join(missing)}; not marking "
                                 f"{'them' if len(missing) > 1 else 'it'} done (subtasks {phase.subtasks_done}/"
                                 f"{len(phase.subtasks)})" + ("; its final message claims them done" if claimed else ""))

    def _wire_callbacks(self, phase: Phase) -> None:
        w, base = self.d.worker, self._tool_base
        w.on_tool_call = lambda n: self.d.emit("tool_calls", {"task_id": phase.id, "count": base + n})
        w.on_session = lambda sid: self._on_session(phase, sid)
        w.on_todo = lambda items: self._on_todo(phase, items)

    def _on_session(self, phase: Phase, sid: str) -> None:
        if sid and sid != phase.session_id:
            phase.session_id = sid
            self._save_task(phase)  # persisted immediately: a killed turn can be resumed
            self.d.log("SYSTEM", f"{phase.id}: Qwen session {sid} recorded")

    def _on_todo(self, phase: Phase, items: list[dict[str, Any]]) -> list[str]:
        phase.todo = list(items)
        self._attempt_todo = list(items)
        newly = phase.apply_todo(items)
        if newly:
            self.d.log("SYSTEM", f"{phase.id}: subtask {', '.join(newly)} done (worker todo list) - "
                                 f"subtasks {phase.subtasks_done}/{len(phase.subtasks)}")
            if len(newly) > 1:
                self.d.log("SYSTEM", f"{phase.id}: note - {len(newly)} planned subtasks were reported complete in ONE "
                                     "todo update, not as each one finished")
        self._save_task(phase)  # persisted (and shown) immediately
        return newly

    # ------------------------------------------------------------ implementation guard (V4 pass 3)

    def _impl_enabled(self) -> bool:
        return bool(getattr(self.d.settings, "implementation_guard_enabled", True))

    def _load_impl(self, phase: Phase) -> ImplGuardState | None:
        if not self._impl_enabled():
            return None
        return ImplGuardState.from_dict((self.st.get("impl_guard") or {}).get(phase.id))

    def _save_impl(self, g: ImplGuardState) -> None:
        self.st.setdefault("impl_guard", {})[g.phase_id] = g.to_dict()
        self._persist_run()

    def _impl_prepare(self, phase: Phase, last_reviewed: Attempt | None) -> None:
        """The guard's budget belongs to a work cycle: it survives pauses and restarts (in-flight usage is
        folded in), and is renewed only by a new checkpoint cycle or by the user resuming a guard block."""
        if not self._impl_enabled():
            return
        g = self._load_impl(phase) or ImplGuardState(phase.id)
        g.fold_inflight()
        cycle = last_reviewed.number if last_reviewed is not None else 0
        if g.cycle_attempt != cycle:
            if g.wall_used or g.tool_calls or g.window > 1:
                g.new_window()
            g.cycle_attempt, g.diagnoses_used, g.pending, g.status, g.guidance = cycle, 0, "", "working", {}
        elif g.status == "blocked":
            g.new_window()
            g.diagnoses_used, g.pending, g.status = 0, "", "working"
            phase.session_id = ""
            self._fresh_reason = "the user resumed after the implementation guard stopped the phase"
            self.d.log("SYSTEM", f"{phase.id}: the user resumed after the implementation guard stopped the phase - "
                                 "fresh session and budget" + (" with the user's guidance" if phase.user_guidance else ""))
        self._save_impl(g)

    async def _impl_check(self, phase: Phase, attempt: Attempt) -> None:
        g = self._load_impl(phase)
        if g is None:
            return
        if not g.pending:
            trip = impl_guard_trip(g, self.d.settings, caps=self.d.worker.capabilities)
            if trip:  # e.g. the budget was used up before an interruption
                g.pending = trip
        if g.pending:
            await self._impl_escalate(phase, attempt, g)

    async def _impl_run(self, phase: Phase, request: PhaseTurnRequest) -> WorkerOutcome:
        """One worker turn under the implementation guard: stopped as soon as the wall-clock or tool budget is
        used up, or when no PLANNED subtask was reported done for too long (only a structured todo-list
        completion of a planned subtask counts as progress)."""
        worker, o = self.d.worker, self.d.settings
        g = self._load_impl(phase)
        if g is None:
            return await worker.run_turn(request)
        caps = worker.capabilities  # V5.1: limits whose signal this worker cannot report are not applied
        g.trip = ""
        started = time.monotonic()
        base = self._tool_base
        stopped: list[str] = []
        live = {"tools": 0}

        def stop(reason: str) -> None:
            if not stopped:
                stopped.append(reason)
                self.d.log("SYSTEM", f"{phase.id}: implementation guard: {reason} - stopping the worker turn")
                asyncio.ensure_future(worker.cancel())

        def on_tool(n: int) -> None:
            self.d.emit("tool_calls", {"task_id": phase.id, "count": base + n})
            live["tools"] = n
            g.inflight_tool_calls, g.inflight_wall = n, time.monotonic() - started
            self._save_impl(g)  # in-flight usage survives a crash
            trip = impl_guard_trip(g, o, extra_wall=g.inflight_wall, extra_tools=n, caps=caps)
            if trip:
                stop(trip)

        def on_todo(items: list[dict[str, Any]]) -> None:
            if self._on_todo(phase, items):  # structured progress of a planned subtask
                g.progress_wall = g.wall_used + (time.monotonic() - started)
                g.progress_subtasks = [s.id for s in phase.subtasks if s.done]
                self._save_impl(g)

        async def watchdog() -> None:
            while not stopped:
                elapsed = time.monotonic() - started
                trip = impl_guard_trip(g, o, extra_wall=elapsed, extra_tools=live["tools"], caps=caps)
                if trip:
                    stop(trip)
                    return
                await asyncio.sleep(0.1)

        worker.on_tool_call = on_tool
        worker.on_todo = on_todo
        dog = asyncio.ensure_future(watchdog())
        try:
            outcome = await worker.run_turn(request)
        except BaseException:  # STOP / a crash of the turn: its usage still counts (folded in on the next start)
            dog.cancel()
            g.inflight_wall = time.monotonic() - started
            self._save_impl(g)
            raise
        dog.cancel()
        g.inflight_tool_calls, g.inflight_wall = 0, 0.0
        g.wall_used += time.monotonic() - started
        g.tool_calls += outcome.tool_calls
        if stopped:
            g.trip = stopped[0]
        self._save_impl(g)
        return outcome

    async def _done_and_green(self, phase: Phase, attempt: Attempt) -> bool:
        """Every planned subtask is done (structured progress) AND the orchestrator's phase verification passes:
        the phase moves on to verification / evidence / the checkpoint instead of another worker turn."""
        o = self.d.settings
        if not phase.subtasks or not all(s.done for s in phase.subtasks):
            return False
        if not phase.verification_commands or not o.run_verification_commands:
            return False
        self.d.log("VERIFY", f"{phase.id}: every planned subtask is done - running the phase verification before "
                             "deciding whether the worker must continue")
        attempt.verification = await self.d.verifier.run(phase.verification_commands, self.run.repo_path, True)
        attempt.verified_at_turn = len(attempt.turns)
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        ran = [v for v in attempt.verification if not v.skipped_reason]
        if not ran or not all(v.passed for v in ran):
            return False
        g = self._load_impl(phase)
        if g is not None:
            g.pending = g.trip = ""
            self._save_impl(g)
        self.d.log("SYSTEM", f"{phase.id}: all planned subtasks are done and the phase verification is green - "
                             "proceeding to verification and the checkpoint (the worker's INCOMPLETE status / exit code "
                             "is not a reason to continue it)")
        return True

    @staticmethod
    def _guidance_lines(d: dict[str, Any]) -> list[str]:
        lines = [f"(the previous session was stopped by the orchestrator: {d.get('reason', '')})"]
        lines += [f"- blocker: {d['blocker']}"] if d.get("blocker") else []
        lines += [f"- diagnosis: {d['diagnosis']}"] if d.get("diagnosis") else []
        lines += [f"- do: {x}" for x in d.get("next_actions") or []]
        lines += [f"- avoid: {x}" for x in d.get("avoid") or []]
        if d.get("target_files"):
            lines.append("- target files: " + ", ".join(d["target_files"][:15]))
        if d.get("target_checks"):
            lines.append("- checks after each change: " + "; ".join(d["target_checks"][:6]))
        return lines

    async def _impl_escalate(self, phase: Phase, attempt: Attempt, g: ImplGuardState) -> None:
        """The guard stopped the session: a cheap, bounded Claude MID_PHASE_DIAGNOSE and a FRESH session with its
        guidance - or, when the diagnoses of this cycle are used up (or Claude says BLOCKED), a BLOCK."""
        reason = g.pending
        wall, tools, nop, max_diag = guard_limits(self.d.settings)
        g.add_trip({"at": time.time(), "reason": reason, "window": g.window, "wall_used": round(g.wall_used, 1),
                    "tool_calls": g.tool_calls, "subtasks_done": [s.id for s in phase.subtasks if s.done]})
        if g.diagnoses_used >= max(0, max_diag):
            g.status, g.pending = "blocked", ""
            self._save_impl(g)
            phase.status = TaskStatus.BLOCKED
            self._save_task(phase)
            raise _Halt(S.BLOCKED, f"the implementation guard stopped {phase.id} again ({reason}) after "
                                   f"{g.diagnoses_used} MID_PHASE_DIAGNOSE call(s). RESUME (optionally with guidance) "
                                   "continues in a fresh session with a fresh budget.", S.WORKER_RUNNING)
        self.d.log("SYSTEM", f"{phase.id}: implementation guard stopped the session ({reason}); escalating to Claude "
                             f"(diagnosis {g.diagnoses_used + 1}/{max_diag})")
        packet = await self._mid_phase_packet(phase, attempt, g, reason, (wall, tools, nop))
        self.d.log("CLAUDE", f"MID_PHASE_DIAGNOSE {phase.id}: {reason}")
        try:
            d = await self.d.orchestrator.mid_phase_diagnose(packet)
        finally:
            self._record_orchestrator_calls(phase.id)
        g.diagnoses_used += 1
        g.add_diagnosis({**d.to_dict(), "reason": reason, "at": time.time(), "window": g.window})
        g.pending = ""
        self.d.log("CLAUDE", f"MID_PHASE_DIAGNOSE {phase.id}: {d.status} - {d.blocker or d.diagnosis}")
        for x in d.next_actions:
            self.d.log("CLAUDE", f"  do: {x}")
        for x in d.avoid:
            self.d.log("CLAUDE", f"  avoid: {x}")
        if d.status == "BLOCKED":
            g.status = "blocked"
            self._save_impl(g)
            phase.status = TaskStatus.BLOCKED
            self._save_task(phase)
            raise _Halt(S.BLOCKED, f"Claude diagnosed the stopped implementation of {phase.id} as BLOCKED: "
                                   f"{d.blocked_reason or d.blocker}. RESUME (with guidance) continues in a fresh session.",
                        S.WORKER_RUNNING)
        g.guidance = {**d.to_dict(), "reason": reason}
        g.new_window()
        self._save_impl(g)
        phase.session_id = ""  # the next turn starts a FRESH session with the diagnosis
        self._fresh_reason = f"the implementation guard stopped the previous session ({reason})"
        self._save_task(phase)

    async def _mid_phase_packet(self, phase: Phase, attempt: Attempt, g: ImplGuardState, reason: str,
                                limits: tuple[float, int, float]) -> MidPhasePacket:
        o = self.d.settings
        verification: list[VerificationResult] = []
        if phase.verification_commands and o.run_verification_commands:
            self.d.log("VERIFY", f"{phase.id}: current verification state for the diagnosis")
            verification = await self.d.verifier.run(phase.verification_commands, self.run.repo_path, True)
            attempt.verification, attempt.verified_at_turn = verification, len(attempt.turns)
            self.d.db.update_attempt(attempt)
        changed: list[str] = []
        stat = ""
        git, base_d = self.d.git, self.st.get("phase_base", {}).get(phase.id)
        if git is not None and base_d:
            try:
                post = await git.snapshot(self.d.settings.precise_git_snapshots)
                ch = await git.attempt_changes(snapshot_from_dict(base_d, post.generated), post)
                changed, stat = [c.path for c in ch.changed_files], ch.diff_stat
            except GitError as exc:
                self.d.log("GIT", f"{phase.id}: diff unavailable for the diagnosis ({exc})")
        turns = [t for a in self.attempts.get(phase.id, []) for t in a.turns][-6:]
        return MidPhasePacket(
            phase=phase, reason=reason, repo_path=self.run.repo_path, goal=self.run.plan.goal or self.run.goal,
            usage={"wall_used": g.wall_used, "wall_limit": limits[0], "tool_calls": g.tool_calls,
                   "tool_limit": limits[1], "no_progress": g.wall_used - g.progress_wall,
                   "no_progress_limit": limits[2]},
            turns=turns, worker_summary=attempt.worker_summary, verification=verification, changed_files=changed,
            diff_stat=stat, previous=list(g.diagnoses), user_guidance=phase.user_guidance)

    def _record_turn(self, phase: Phase, attempt: Attempt, purpose: str, mode: str, o: WorkerOutcome,
                     status: str, started: float, fix_round: int) -> None:
        n = attempt.turns[-1]["n"] if attempt.turns and attempt.turns[-1].get("running") else len(attempt.turns) + 1
        rec = turn_record(n, purpose, mode, o, status)
        if attempt.turns and attempt.turns[-1].get("running"):
            attempt.turns[-1] = rec
        else:
            attempt.turns.append(rec)
        attempt.executable, attempt.command_line, attempt.exit_code = o.executable, o.argv, o.exit_code
        attempt.stdout_tail, attempt.stderr_tail = o.stdout_tail, o.stderr_tail
        attempt.tool_calls += o.tool_calls
        attempt.tool_call_log = (attempt.tool_call_log + o.tool_call_log)[-200:]
        attempt.worker_usage = o.usage
        attempt.session_id = phase.session_id
        if o.error:
            attempt.error = tail((attempt.error + "\n" if attempt.error else "") + f"turn {n}: {o.error}", 1500)
        if (o.summary or "").strip():
            if purpose == "verify_fix":
                attempt.worker_summary = (attempt.worker_summary + f"\n\nLOCAL FIX (round {fix_round}):\n"
                                          + tail(o.summary, 1500)).strip()
            else:
                attempt.worker_summary = o.summary
        self._tool_base += o.tool_calls
        for note in o.degraded_notes:
            if note not in self._degraded_notes:
                self._degraded_notes.append(note)
        self.d.db.update_attempt(attempt)
        self.d.db.add_worker_call(self.run.id, attempt.db_id, o, started)
        self._emit_attempt(attempt)
        self.d.log("SYSTEM", f"Qwen turn {n} ({purpose}, {mode} session) exited with code {o.exit_code} after "
                             f"{o.duration:.1f}s, {o.tool_calls} tool call(s)"
                             + (f", reported {status}" if status else ""))

    def _handoff(self, phase: Phase, attempt: Attempt, purpose: str, why: str) -> Handoff | None:
        history = self.attempts.get(phase.id, [])
        # Work may exist from a finished turn, a killed turn (still marked running) or an earlier attempt.
        prior_work = any(t.get("tool_calls") or t.get("running") for a in history for t in a.turns)             or any(a.changed_files for a in history)
        if not prior_work and purpose in ("start", "resume"):
            return None  # nothing to hand off: the new session starts the phase normally
        files: list[str] = []
        for a in reversed(history):
            if a.changed_files:
                files = a.worker_files()
                break
        summary = attempt.worker_summary or next((a.worker_summary for a in reversed(history) if a.worker_summary), "")
        return Handoff(reason=why or {"retry": "addressing checkpoint feedback",
                                      "resume": "continuing after an interruption"}.get(purpose, "continuing the phase"),
                       previous_summary=summary, files_changed=files,
                       subtasks_done=[s.id for s in phase.subtasks if s.done])

    def _work_spec(self, phase: Phase, retry_ctx: RetryContext | None, handoff: Handoff | None) -> PhaseWorkSpec:
        i, n = self._index(phase)
        ps = self._phases()
        plan = self.run.plan
        return PhaseWorkSpec(
            phase=phase, repo_path=self.run.repo_path, phase_index=i, phase_count=n, goal=plan.goal,
            plan_summary=plan.summary, architecture_notes=list(plan.architecture_notes),
            completed=[(p.id, p.title, p.result, p.status == TaskStatus.SUPERSEDED) for p in ps
                       if p.status in TERMINAL and p.position < phase.position],
            upcoming=[(p.id, p.title) for p in ps if p.position > phase.position and p.status not in TERMINAL],
            codebase_memory_available=self.d.codebase_memory_available,
            codebase_memory_project=self.d.codebase_memory_project,
            unreal_project=self.d.unreal_project, unreal_mcp_reachable=self.d.unreal_mcp_reachable,
            retry=retry_ctx, handoff=handoff)

    async def _attribute(self, phase: Phase, attempt: Attempt, pre) -> None:
        """Changes of the WHOLE phase so far (all attempts), so retries/interruptions lose nothing."""
        git = self.d.git
        if git is None or pre is None:
            return
        post = await git.snapshot(self.d.settings.precise_git_snapshots)
        self._last_post_tree = post.tree
        base_d = self.st.get("phase_base", {}).get(phase.id)
        changes = None
        if base_d:
            try:
                changes = await git.attempt_changes(snapshot_from_dict(base_d, pre.generated), post)
                attempt.diff_scope = "phase"
            except GitError as exc:  # e.g. the baseline tree object was garbage-collected
                self.d.log("GIT", f"phase-level diff unavailable ({exc}); reviewing this attempt's changes only")
        if changes is None:
            changes = await git.attempt_changes(pre, post)
            attempt.diff_scope = "attempt"
        user = set(self.run.initial_dirty)
        for c in changes.changed_files:
            if c.note:
                c.note = ("file also had pre-existing user changes" if c.path in user
                          else "file was also changed by an earlier phase")
        attempt.changed_files = changes.changed_files
        attempt.preexisting_dirty = changes.preexisting_dirty
        attempt.generated_excluded = changes.generated_excluded
        attempt.generated_by_dir = changes.generated_by_dir
        attempt.diff_stat = changes.diff_stat
        attempt.diff = changes.diff
        self.d.log("GIT", f"phase {phase.id} ({attempt.diff_scope} scope): {len(changes.changed_files)} authored file(s) "
                          f"changed; generated/temporary paths not attributed: {changes.generated_excluded}")
        for c in changes.changed_files[:40]:
            self.d.log("GIT", f"  {c.change} {c.path}" + (f"  [{c.note}]" if c.note else ""))

    # ------------------------------------------------------------ VERIFYING

    async def _on_phase_verify(self) -> RunState:
        phase = self._require_task()
        attempt = self._last_attempt(phase)
        phase.status = TaskStatus.VERIFYING
        self._save_task(phase)
        if attempt.verification and attempt.verified_at_turn == len(attempt.turns):
            self.d.log("VERIFY", f"{phase.id}: self-check results are current (no worker turn since); not re-running")
        elif phase.verification_commands:
            attempt.verification = await self.d.verifier.run(
                phase.verification_commands, self.run.repo_path, self.d.settings.run_verification_commands)
            attempt.verified_at_turn = len(attempt.turns)
        else:
            self.d.log("VERIFY", f"{phase.id}: no verification commands defined")
            attempt.verification = []
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        return S.COLLECTING_EVIDENCE if self._evidence_needed(phase) else S.REVIEWING

    # ------------------------------------------------------------ COLLECTING_EVIDENCE (V4 pass 2 / 3)

    def _evidence_needed(self, phase: Phase) -> bool:
        """Phases whose contract lists evidence requirements, or that gained human-derived HF-* criteria;
        legacy phases never collect evidence."""
        return bool(getattr(self.d.settings, "evidence_enabled", False)) and bool(phase.evidence_requirements())

    @property
    def registry(self) -> CapabilityRegistry:
        if self.d.evidence_registry is None:
            self.d.evidence_registry = default_registry()
        return self.d.evidence_registry

    def _load_evidence(self, phase: Phase) -> evm.EvidenceState | None:
        return evm.EvidenceState.from_dict((self.st.get("evidence") or {}).get(phase.id))

    def _save_evidence(self, es: evm.EvidenceState) -> None:
        self.st.setdefault("evidence", {})[es.phase_id] = es.to_dict()
        self._persist_run()

    def _elog(self, text: str) -> None:
        self.d.log("EVIDENCE", text)

    async def _repo_fp(self) -> tuple[str, str]:
        """The authored state of the repository (git snapshot tree, or a bounded manifest), or ("", "")."""
        return await repo_fingerprint(self.run.repo_path, self.d.git,
                                      max_files=int(getattr(self.d.settings, "evidence_guard_max_files", 5000)))

    def _executor(self, phase: Phase, attempt: Attempt, es: evm.EvidenceState, fp: tuple[str, str]) -> EvidenceExecutor:
        o = self.d.settings
        limits = EvidenceLimits.from_settings(o)
        store = ArtifactStore(self.d.artifact_root, self.run.id, phase.id, attempt.number,
                              max_artifacts=limits.max_artifacts, max_bytes=limits.max_artifact_bytes,
                              existing=len(es.artifacts),
                              max_run_bytes=int(getattr(o, "evidence_artifact_max_run_bytes", 0) or 0))
        log = lambda s: self._elog(f"{phase.id}: {s}")  # noqa: E731
        if self.d.git is not None:
            guard = GitMutationGuard(self.d.git, log)
        else:  # not a git repository: a bounded manifest (exact snapshot only within the byte budget)
            root = os.path.join(self.d.artifact_root, ".guard") if self.d.artifact_root else ""
            guard = ManifestMutationGuard(self.run.repo_path, root,
                                          max_files=int(getattr(o, "evidence_guard_max_files", 5000)),
                                          max_snapshot_bytes=int(getattr(o, "evidence_guard_max_snapshot_bytes",
                                                                         64_000_000)), log=log)
        return EvidenceExecutor(self.registry, limits, store, repo_path=self.run.repo_path,
                                persist=lambda: self._save_evidence(es), log=log, guard=guard,
                                repo_fp=fp[0], repo_fp_kind=fp[1])

    async def _on_collect_evidence(self) -> RunState:
        """Discovery (per attempt) -> plan (reused while its inputs are unchanged) -> bounded execution -> at most
        N bounded replans when an evidence strategy cannot be executed. Every stage is persisted, so a restart
        continues where it stopped, never trusts an interrupted step and never trusts a stale one."""
        phase = self._require_task()
        attempt = self._last_attempt(phase)
        if not self._evidence_needed(phase):  # e.g. evidence was disabled while this state waited for a resume
            self._elog(f"{phase.id}: evidence collection is not needed any more; continuing to the checkpoint")
            return S.REVIEWING
        phase.status = TaskStatus.VERIFYING
        self._save_task(phase)
        es = self._load_evidence(phase) or evm.EvidenceState(phase.id)
        previous = evm.CapabilitySnapshot.from_dict(es.discovery)
        new_attempt = es.attempt != attempt.number
        if new_attempt:
            es.begin_attempt(attempt.number)
            if previous is not None:  # capabilities may change with the work: discovered again for each attempt
                es.discovery = {}
            self._save_evidence(es)
        fp = await self._repo_fp()
        executor = self._executor(phase, attempt, es, fp)
        if not new_attempt and es.status in ("complete", "replanning") and es.bundle:
            b = es.bundle
            if b.get("repo_fp") and fp[0] and b["repo_fp"] != fp[0]:
                self._elog(f"{phase.id}: the evidence collected before the interruption is stale (the repository "
                           "changed since); collecting it again")
                es.status = "planned"
                self._save_evidence(es)
            else:  # collected just before an interruption (and still fresh)
                plan, snap = evm.EvidencePlan.from_dict(es.plan), evm.CapabilitySnapshot.from_dict(es.discovery)
                bundle = evm.EvidenceBundle.from_dict(b)
                if plan is not None and snap is not None and bundle is not None:
                    confirmed = {c.key for c in snap.capabilities if c.confidence == evm.CONFIRMED}
                    await self._replan_loop(phase, attempt, es, plan, snap, bundle, executor, confirmed)
                return self._finish_evidence(phase, attempt, es)
        snapshot = evm.CapabilitySnapshot.from_dict(es.discovery)
        if snapshot is None:
            snapshot = await self._discover(phase, attempt, previous)
            es.discovery = snapshot.to_dict()
            self._save_evidence(es)
            self._log_discovery(phase, snapshot, previous if new_attempt else None)
        plan = evm.EvidencePlan.from_dict(es.plan)
        pfp = plan_fingerprint(phase, snapshot)
        if plan is None or plan.fingerprint != pfp:
            plan = await self._plan_evidence(phase, attempt, snapshot, es)
            plan.fingerprint = pfp
            es.plan, es.status = plan.to_dict(), "planned"
            self._save_evidence(es)
            self._elog(f"{phase.id}: plan created for {len(plan.requirements)} requirement(s) ({plan_counts(plan)})"
                       + (" - no usable capability, no Claude call" if plan.source == "deterministic" else ""))
            self._log_plan_notes(phase, plan)
        elif es.status == "new":
            self._elog(f"{phase.id}: plan reused ({plan_counts(plan)}; contract, derived criteria and capabilities "
                       "unchanged)")
        else:
            self._elog(f"{phase.id}: continuing the interrupted evidence collection of attempt {attempt.number}")
        confirmed = {c.key for c in snapshot.capabilities if c.confidence == evm.CONFIRMED}
        bundle = await executor.run(es, plan, snapshot)
        await self._replan_loop(phase, attempt, es, plan, snapshot, bundle, executor, confirmed)
        return self._finish_evidence(phase, attempt, es)

    def _log_plan_notes(self, phase: Phase, plan: evm.EvidencePlan) -> None:
        for err in plan.validation_errors[:6]:
            self._elog(f"{phase.id}: plan validation: {hm.short(err, 200)}")
        for r in plan.requirements:
            for note in r.notes[:3]:
                self._elog(f"{phase.id} {r.requirement_id}: {hm.short(note, 200)}")

    def _log_discovery(self, phase: Phase, snapshot: evm.CapabilitySnapshot,
                       previous: evm.CapabilitySnapshot | None) -> None:
        if previous is not None:
            before = {(c.key, c.usable) for c in previous.capabilities}
            now = {(c.key, c.usable) for c in snapshot.capabilities}
            if before == now:
                self.d.log("DISCOVERY", f"{phase.id}: {snapshot.summary()} (unchanged since the previous attempt)")
                return
            added = sorted(k for k, u in now - before if u)
            gone = sorted(k for k, u in before - now if u)
            self.d.log("DISCOVERY", f"{phase.id}: {snapshot.summary()}; changed since the previous attempt"
                       + (f": now usable {', '.join(hm.short(x, 60) for x in added[:4])}" if added else "")
                       + (f"; no longer usable {', '.join(hm.short(x, 60) for x in gone[:4])}" if gone else ""))
        else:
            self.d.log("DISCOVERY", f"{phase.id}: {snapshot.summary()}")
        for name, status in snapshot.providers.items():
            if not status.startswith("ok"):
                self.d.log("DISCOVERY", f"{phase.id}: provider {name}: {status}")

    async def _replan_loop(self, phase: Phase, attempt: Attempt, es: evm.EvidenceState, plan: evm.EvidencePlan,
                           snapshot: evm.CapabilitySnapshot, bundle: evm.EvidenceBundle, executor: EvidenceExecutor,
                           confirmed_before: set[str]) -> evm.EvidenceBundle:
        """Bounded replanning when the evidence STRATEGY could not be executed (a capability turned out
        unavailable, a resource never started or never became ready, a provider error) - never for an
        ordinary FAILED check, which is evidence of a defect. The replan count is persisted with the
        attempt, so plan -> failure -> replan can never loop."""
        limit = max(0, int(getattr(self.d.settings, "evidence_max_replans", 1)))
        while True:
            failures = strategy_failures(bundle, plan, snapshot, confirmed_before)
            if not failures:
                return bundle
            rids = [f["requirement_id"] for f in failures]
            if es.replans >= limit:
                self._elog(f"{phase.id}: the evidence strategy for {', '.join(rids)} could not be executed or could "
                           f"not observe the property; replan limit ({limit}) reached - the results stay as they are")
                return bundle
            gen = es.replans + 1
            demoted = demote_failed(snapshot, failures)
            if not snapshot.usable():
                self._elog(f"{phase.id}: no usable capability is left to replan {', '.join(rids)}")
                return bundle
            reasons = [f"{f['requirement_id']} ({', '.join(f['capabilities']) or 'no capability'}): {f['reason']}"
                       for f in failures]
            label = "could not be executed" if all(f["kind"] != "unobserved" for f in failures) else \
                "could not be executed or could not observe the property"
            self._elog(f"{phase.id}: replan {gen}/{limit} for {', '.join(rids)} - the evidence strategy {label}: "
                       f"{hm.short(reasons[0], 200)}")
            es.status = "replanning"  # a crash during the Claude call repeats the replan (the count is not spent)
            es.discovery = snapshot.to_dict()
            self._save_evidence(es)
            new = await self._plan_evidence(phase, attempt, snapshot, es,
                                            replan={"generation": gen, "requirements": rids, "failures": reasons})
            plan = merge_replan(plan, new, rids, {f["requirement_id"] for f in failures if f["kind"] == "unobserved"})
            es.plan, es.replans, es.status = plan.to_dict(), gen, "planned"
            es.replan_log = (es.replan_log + [{"at": time.time(), "attempt": attempt.number, "generation": gen,
                                               "requirements": rids, "reasons": reasons[:6],
                                               "demoted": demoted}])[-10:]
            self._save_evidence(es)
            self._elog(f"{phase.id}: replanned {', '.join(rids)} ({plan_counts(new)})")
            self._log_plan_notes(phase, new)
            bundle = await executor.run(es, plan, snapshot)

    def _finish_evidence(self, phase: Phase, attempt: Attempt, es: evm.EvidenceState) -> RunState:
        bundle = evm.EvidenceBundle.from_dict(es.bundle)
        attempt.evidence = dict(es.bundle)
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        for r in bundle.results if bundle else []:
            partly = sorted((r.provenance or {}).get("reused_steps") or {}) if r.freshness == evm.FRESH else []
            self._elog(f"{phase.id} {r.requirement_id} {r.kind}: {r.status}"
                       + (f" ({r.freshness})" if r.freshness == evm.REUSED else "")
                       + (f" (ran now; reused unchanged {', '.join(partly[:4])})" if partly else "")
                       + (f" - {hm.short(r.summary or r.failure_reason, 160)}" if r.status != evm.SATISFIED else ""))
        for note in (bundle.notes if bundle else [])[:4]:
            self._elog(f"{phase.id}: {hm.short(note, 240)}")
        if bundle is not None and bundle.guard and bundle.guard.get("kind") == "none":
            self._elog(f"{phase.id}: {bundle.guard.get('detail')}")
        self._associate_criteria(phase, attempt, list((attempt.evidence or {}).get("results") or []))
        return S.REVIEWING

    def _associate_criteria(self, phase: Phase, attempt: Attempt, results: list[dict[str, Any]]) -> None:
        """Human-derived criteria are evidence requirements: record their latest evidence and status."""
        by = {c.id: c for c in phase.human_criteria}
        changed = []
        for r in results:
            c = by.get(str(r.get("requirement_id")))
            if c is not None and hm.record_criterion_evidence(c, r, attempt.number):
                changed.append(f"{c.id} {c.status}")
        if changed:
            self._save_task(phase)
            self._elog(f"{phase.id}: human-derived criteria: {', '.join(changed)}")

    async def _refresh_evidence(self, phase: Phase, attempt: Attempt, when: str) -> None:
        """Evidence must not stay trusted after its inputs changed (e.g. while the run waited or was stopped):
        results whose provenance no longer matches the repository are demoted to stale (never SATISFIED)."""
        results = (attempt.evidence or {}).get("results") or []
        if not any(r.get("provenance") for r in results):
            return
        fp, _kind = await self._repo_fp()
        if not fp:
            return  # nothing can prove a change (and nothing proves freshness either; it stays as collected)
        went = efresh.refresh(results, repo_fp=fp, repo_path=self.run.repo_path)
        if not went:
            return
        attempt.evidence["freshness_checked_at"] = time.time()
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        self._elog(f"{phase.id}: evidence went stale before {when}: "
                   + ", ".join(f"{rid} ({why})" for rid, why in went[:6]))
        self._associate_criteria(phase, attempt, results)

    async def _discover(self, phase: Phase, attempt: Attempt | None = None,
                        previous: evm.CapabilitySnapshot | None = None) -> evm.CapabilitySnapshot:
        ran: dict[str, VerificationResult] = {}
        for atts in self.attempts.values():  # everything verification actually ran in this run
            for a in atts:
                for v in a.verification:
                    if not v.skipped_reason:
                        ran[v.command.strip()] = v
        known: dict[str, str] = {}
        for pid, d in (self.st.get("evidence") or {}).items():
            snap = evm.CapabilitySnapshot.from_dict((d or {}).get("discovery"))
            for c in snap.capabilities if snap is not None else []:
                if c.confidence == evm.CONFIRMED and pid != phase.id:
                    known.setdefault(c.key, f"confirmed earlier in this run ({pid}: {c.reason})")
        for c in previous.capabilities if previous is not None else []:  # confirmed in an earlier attempt
            if c.confidence == evm.CONFIRMED:
                known.setdefault(c.key, f"confirmed in an earlier attempt of {phase.id} ({c.reason})")
        o = self.d.settings
        ctx = DiscoveryContext(self.run.repo_path, phase, self._phases(), ran, o, self._human_gate_open(phase),
                               changed_files=attempt.worker_files() if attempt is not None else [])
        return await self.registry.discover(ctx, timeout=float(getattr(o, "evidence_discovery_timeout_seconds", 20)),
                                            max_capabilities=int(getattr(o, "evidence_max_capabilities", 40)),
                                            known_confirmed=known)

    async def _plan_evidence(self, phase: Phase, attempt: Attempt, snapshot: evm.CapabilitySnapshot,
                             es: evm.EvidenceState, replan: dict[str, Any] | None = None) -> evm.EvidencePlan:
        o = self.d.settings
        if not snapshot.usable():
            return deterministic_plan(phase, "capability discovery found nothing usable in this environment")
        max_steps = int(getattr(o, "evidence_max_steps", 12))
        if replan:  # what is left of the attempt's step budget
            max_steps = max(0, max_steps - sum(1 for s in es.steps.values()
                                               if s.get("status") != evm.SKIPPED and not s.get("reused_from")))
        packet = EvidencePlanPacket(
            phase=phase, snapshot=snapshot, goal=self.run.plan.goal or self.run.goal, repo_path=self.run.repo_path,
            verification=list(attempt.verification), changed_files=attempt.worker_files(), diff_stat=attempt.diff_stat,
            prior=list(es.history), human_gate_open=self._human_gate_open(phase), max_steps=max_steps,
            step_timeout=int(getattr(o, "evidence_step_timeout_seconds", 180)),
            wall_seconds=int(getattr(o, "evidence_max_wall_seconds", 600)), replan=dict(replan or {}))
        what = f"replan {replan['generation']} of {', '.join(replan['requirements'])}" if replan else \
            f"{len(phase.evidence_requirements())} requirement(s)"
        self.d.log("SYSTEM", f"Claude EVIDENCE_PLAN for phase {phase.id}: {len(snapshot.usable())} usable "
                             f"capabilit(ies), {what}")
        try:
            raw = await self.d.orchestrator.evidence_plan(packet)
        finally:
            self._record_orchestrator_calls(phase.id)
        if replan:
            return validate_plan(raw, phase, snapshot, self.registry, max_steps, only=list(replan["requirements"]),
                                 generation=int(replan["generation"]))
        return validate_plan(raw, phase, snapshot, self.registry, max_steps)

    def _checkpoint_evidence(self, phase: Phase, attempt: Attempt) -> dict[str, Any]:
        if not attempt.evidence:
            return {}
        es = self._load_evidence(phase)
        snap = evm.CapabilitySnapshot.from_dict(es.discovery) if es is not None else None
        # Compact: the capabilities the plan used come with the bundle; list the others only briefly.
        used = {c.get("id") for c in attempt.evidence.get("capabilities") or []}
        lines = [capability_line(c, snap) for c in snap.capabilities if c.id in used] if snap else []
        lines += [f"{c.id} [{c.kind}] {c.confidence} - {c.name}" for c in snap.capabilities if c.id not in used][:12] \
            if snap else []
        return dict(attempt.evidence, discovered=lines[:20])

    # ------------------------------------------------------------ REVIEWING (checkpoint)

    async def _on_checkpoint(self) -> RunState:
        phase = self._require_task()
        attempt = self._last_attempt(phase)
        # A replan decided by an earlier, interrupted run of this checkpoint is void: this call decides.
        if isinstance(self.st.get("pending_replan"), dict) and self.st["pending_replan"].get("phase_id") == phase.id:
            self.st.pop("pending_replan")
            self._persist_run()
        hs = self._load_human(phase)  # likewise an unanswered human request of an interrupted checkpoint
        if hs is not None and hs.supersede_pending():
            self._save_human(hs)
        cs = self._load_consistency(phase)  # and a consistency review an interrupted checkpoint had requested
        if cs is not None and cs.get("status") == "pending":
            cs["status"] = "superseded"
            self._save_consistency(phase, cs)
        phase.status = TaskStatus.REVIEWING
        self._save_task(phase)
        await self._refresh_evidence(phase, attempt, "the checkpoint")
        i, n = self._index(phase)
        final = self._is_final(phase)
        previous = [{"number": a.number, "decision": a.review.decision.value, "summary": a.review.summary,
                     "problems": a.review.problems, "kind": a.review.kind}
                    for a in self.attempts.get(phase.id, []) if a is not attempt and a.review is not None]
        human_open = self._human_gate_open(phase)
        package = CheckpointPackage(
            phase=phase, attempt=attempt, max_attempts=self.max_attempts_for(phase), phase_index=i, phase_count=n,
            final=final, goal=self.run.plan.goal or self.run.goal,
            completed=[phase_result_line(p) for p in self._phases() if p.status in TERMINAL and p.position < phase.position],
            previous_checkpoints=previous, run_diff_stat=await self._run_diff_stat() if final else "",
            user_preexisting=list(self.run.initial_dirty), degraded_notes=list(self._degraded_notes),
            repo_path=self.run.repo_path, repair_notes=self._repair_notes(phase, attempt),
            human_verification_allowed=human_open, human_history=hs.rounds() if hs is not None else [],
            evidence=self._checkpoint_evidence(phase, attempt),
            max_questions=int(getattr(self.d.settings, "human_max_questions", 4) or 4))
        label = "FINAL AUDIT" if final else "CHECKPOINT"
        self.d.log("SYSTEM", f"Claude {label} for phase {phase.id} ({i}/{n}), attempt {attempt.number}"
                   + (" (human verification available)" if human_open else ""))
        try:
            review = await self.d.orchestrator.checkpoint(package)
        finally:
            self._record_orchestrator_calls(phase.id)
        if review.decision == ReviewDecision.ASK_USER:
            if not human_open:  # the schema never offers it then; a provider that answers it anyway is wrong
                raise OrchestratorError(f"Claude returned ASK_USER for {phase.id}, but human verification is not "
                                        "available for this phase (disabled in the settings, or its verification "
                                        "contract does not open the human gate)")
            if not review.human_request.get("questions"):
                raise OrchestratorError(f"Claude returned ASK_USER for {phase.id} without any question for the user")
        review.kind = "FINAL_AUDIT" if final else "CHECKPOINT"
        attempt.review = review
        self.d.db.update_attempt(attempt)
        self.d.db.add_review(self.run.id, attempt)
        self._emit_attempt(attempt)
        phase.last_checkpoint = review.decision.value
        conflict = find_conflict(review, attempt.evidence) \
            if getattr(self.d.settings, "consistency_review_enabled", True) else None
        if review.evidence_sufficient is not None:
            miss = review.missing_evidence
            self.d.log("CLAUDE", f"evidence sufficient: {str(review.evidence_sufficient).lower()}"
                       + (f" - {len(miss)} missing: {hm.short('; '.join(miss), 200)}" if miss else ""))
            if review.decision == ReviewDecision.PASS and review.evidence_sufficient is False and conflict is None:
                self.d.log("CLAUDE", f"  note: {phase.id} passed although the evidence was judged insufficient; the "
                                     "unverified requirements are recorded as risks")
        self.d.log("CLAUDE", f"{label} {phase.id} attempt {attempt.number}: {review.decision.value} - {review.summary}")
        for p in review.problems:
            self.d.log("CLAUDE", f"  problem: {p}")
        for p in review.regressions:
            self.d.log("CLAUDE", f"  regression: {p}")
        for x in review.issues[:8]:
            self.d.log("CLAUDE", f"  issue: {hm.short(issue_line(x), 200)}")
        if review.decision == ReviewDecision.ASK_USER:
            return self._ask_user(phase, attempt, review, final)
        if review.decision == ReviewDecision.PASS and conflict is not None:
            return self._begin_consistency(phase, attempt, review, conflict)
        if review.decision == ReviewDecision.PASS:
            self._save_task(phase)
            return S.ADVANCING
        if review.decision == ReviewDecision.RETRY:
            phase.status = TaskStatus.RETRY
            self._save_task(phase)
            return S.RETRY_PENDING
        if review.decision == ReviewDecision.REPLAN:
            self.st["pending_replan"] = {"phase_id": phase.id, "supersede": True,
                                         "reason": review.replan_reason or review.summary or "checkpoint requested REPLAN"}
            if int(self.st.get("replans", 0)) >= self._replan_allowance():
                phase.status = TaskStatus.BLOCKED
                self._save_task(phase)
                self._persist_run()
                raise _Halt(S.BLOCKED, f"Claude requested REPLAN at {phase.id}, but the replan limit "
                                       f"({self._replan_allowance()}) is reached. Resume to allow one more replan.",
                            S.REPLANNING)
            self._save_task(phase)
            self._persist_run()
            return S.REPLANNING
        phase.status = TaskStatus.BLOCKED
        self._save_task(phase)
        raise _Halt(S.BLOCKED, f"Claude marked phase {phase.id} BLOCKED: {review.summary}", S.RETRY_PENDING)

    async def _run_diff_stat(self) -> str:
        git, base = self.d.git, self.st.get("run_base") or {}
        if git is None or not base.get("tree"):
            return ""
        tree = self._last_post_tree
        if not tree:
            tree = (await git.snapshot(True)).tree
        if not tree:
            return ""
        try:
            return await git.tree_diff_stat(base["tree"], tree)
        except GitError as exc:
            self.d.log("GIT", f"run-level diff stat unavailable: {exc}")
            return ""

    # ------------------------------------------------------------ CONSISTENCY_REVIEW (V4 pass 3)

    def _load_consistency(self, phase: Phase) -> dict[str, Any] | None:
        d = (self.st.get("consistency") or {}).get(phase.id)
        return dict(d) if isinstance(d, dict) else None

    def _save_consistency(self, phase: Phase, rec: dict[str, Any]) -> None:
        self.st.setdefault("consistency", {})[phase.id] = rec
        self._persist_run()

    def _begin_consistency(self, phase: Phase, attempt: Attempt, review: Review, conflict: dict[str, Any]) -> RunState:
        """Persist the pending adjudication BEFORE leaving the checkpoint: a restart asks only for the consistency
        review again, never for a second checkpoint."""
        old = self._load_consistency(phase) or {}
        history = list(old.get("history") or [])
        if old.get("attempt") is not None:
            history = (history + [{k: old.get(k) for k in ("attempt", "status", "trigger", "result")}])[-10:]
        d = review.to_dict()
        self._save_consistency(phase, {
            "phase_id": phase.id, "attempt": attempt.number, "status": "pending", "trigger": conflict,
            "checkpoint": {k: d.get(k) for k in ("kind", "decision", "summary", "evidence_sufficient",
                                                 "missing_evidence", "remaining_risks")},
            "created_at": time.time(), "history": history})
        self._save_task(phase)
        self.d.log("CLAUDE", f"  inconsistent checkpoint: {conflict['reason']} - a bounded CONSISTENCY_REVIEW decides")
        return S.CONSISTENCY_REVIEW

    async def _on_consistency_review(self) -> RunState:
        phase = self._require_task()
        attempt = self._last_attempt(phase)
        cs = self._load_consistency(phase)
        if cs is None or cs.get("attempt") != attempt.number or cs.get("status") not in ("pending", "resolved"):
            raise _Halt(S.BLOCKED, f"{phase.id}: no consistency review is pending for attempt {attempt.number}. "
                                   "Resume to repeat the checkpoint.", S.REVIEWING)
        final = self._is_final(phase)
        done = attempt.review if attempt.review is not None and attempt.review.kind == "CONSISTENCY_REVIEW" else None
        if done is None:
            done = await self._consistency_review(phase, attempt, cs, final)
        elif cs.get("status") != "resolved":  # adjudicated just before an interruption: no second Claude call
            cs.update(status="resolved", result=self._consistency_result(done))
            self._save_consistency(phase, cs)
        if done.decision == ReviewDecision.PASS:
            return S.ADVANCING
        if done.decision == ReviewDecision.RETRY:
            phase.status = TaskStatus.RETRY
            self._save_task(phase)
            return S.RETRY_PENDING
        if done.decision == ReviewDecision.ASK_USER:
            return self._ask_user(phase, attempt, done, final)
        phase.status = TaskStatus.BLOCKED
        self._save_task(phase)
        raise _Halt(S.BLOCKED, f"Claude CONSISTENCY_REVIEW marked phase {phase.id} BLOCKED: {done.summary}",
                    S.RETRY_PENDING)

    @staticmethod
    def _consistency_result(r: Review) -> dict[str, Any]:
        c = r.consistency or {}
        return {"decision": r.decision.value, "summary": hm.short(r.summary, 600),
                "justification": str(c.get("justification") or "")[:2000],
                "dispositions": list(c.get("dispositions") or []), "at": time.time()}

    async def _consistency_review(self, phase: Phase, attempt: Attempt, cs: dict[str, Any], final: bool) -> Review:
        phase.status = TaskStatus.REVIEWING
        self._save_task(phase)
        await self._refresh_evidence(phase, attempt, "the consistency review")
        original = attempt.review
        human_open = self._human_gate_open(phase)
        trigger = dict(cs.get("trigger") or {})
        unresolved = [str(r.get("requirement_id"))
                      for r in evm.unresolved_required((attempt.evidence or {}).get("results") or [])]
        trigger["unresolved"] = unresolved or list(trigger.get("unresolved") or [])
        hs = self._load_human(phase)
        packet = ConsistencyPacket(
            phase=phase, checkpoint=original.to_dict() if original is not None else {},
            evidence=dict(attempt.evidence or {}), trigger=trigger, human_rounds=hs.rounds() if hs is not None else [],
            human_gate_open=human_open, final=final, goal=self.run.plan.goal or self.run.goal,
            repo_path=self.run.repo_path, user_guidance=phase.user_guidance)
        self.d.log("SYSTEM", f"Claude CONSISTENCY_REVIEW for phase {phase.id}: {hm.short(trigger.get('reason', ''), 200)}")
        try:
            review = await self.d.orchestrator.consistency_review(packet)
        finally:
            self._record_orchestrator_calls(phase.id)
        allowed = ["PASS", "RETRY", "BLOCKED"] + (["ASK_USER"] if human_open else [])
        if review.decision.value not in allowed:  # the answer is never another consistency round
            raise OrchestratorError(f"CONSISTENCY_REVIEW for {phase.id} returned {review.decision.value}; only "
                                    f"{', '.join(allowed)} are allowed")
        if review.decision == ReviewDecision.ASK_USER and not review.human_request.get("questions"):
            raise OrchestratorError(f"CONSISTENCY_REVIEW for {phase.id} returned ASK_USER without any question")
        why = check_justified(review, trigger["unresolved"])
        if why:  # defence in depth: an unjustified PASS never advances the phase
            raise OrchestratorError(f"CONSISTENCY_REVIEW for {phase.id} is invalid: {why}. The checkpoint stays "
                                    "unresolved; RESUME asks for the consistency review again")
        review.kind = "CONSISTENCY_REVIEW"
        review.consistency = {**(review.consistency or {}), "trigger": trigger, "checkpoint": cs.get("checkpoint", {}),
                              "at": time.time()}
        if original is not None:  # the checkpoint's evidence judgment stays on record with the adjudication
            if review.evidence_sufficient is None:
                review.evidence_sufficient = original.evidence_sufficient
            review.missing_evidence = review.missing_evidence or list(original.missing_evidence)
            review.evidence_assessment = review.evidence_assessment or list(original.evidence_assessment)
            review.issues = review.issues or list(original.issues)
            if review.decision == ReviewDecision.PASS:
                review.plan_changes_required = review.plan_changes_required or original.plan_changes_required
                review.replan_reason = review.replan_reason or original.replan_reason
                review.task_result = review.task_result or dict(original.task_result)
                review.remaining_risks = list(dict.fromkeys(list(original.remaining_risks) + review.remaining_risks))
        attempt.review = review
        self.d.db.update_attempt(attempt)
        self.d.db.add_review(self.run.id, attempt)
        self._emit_attempt(attempt)
        phase.last_checkpoint = review.decision.value
        self._save_task(phase)
        cs.update(status="resolved", trigger=trigger, result=self._consistency_result(review))
        self._save_consistency(phase, cs)
        self.d.log("CLAUDE", f"CONSISTENCY_REVIEW {phase.id}: {review.decision.value} - {review.summary}")
        c = review.consistency
        if c.get("justification"):
            self.d.log("CLAUDE", f"  justification: {hm.short(c['justification'], 300)}")
        for d in c.get("dispositions") or []:
            self.d.log("CLAUDE", f"  {d.get('requirement_id')}: {d.get('disposition')} - {hm.short(d.get('reason'), 200)}")
        for p in review.problems:
            self.d.log("CLAUDE", f"  problem: {p}")
        return review

    # ------------------------------------------------------------ V4 human gate (ASK_USER -> AWAITING_HUMAN)

    def _human_gate_open(self, phase: Phase) -> bool:
        """ASK_USER is offered only when the user allows it AND the phase's contract opens the human gate."""
        c = phase.verification_contract
        return bool(getattr(self.d.settings, "human_verification_enabled", False)) and c is not None \
            and c.human_gate.allowed

    def _load_human(self, phase: Phase) -> hm.HumanGateState | None:
        return hm.HumanGateState.from_dict((self.st.get("human") or {}).get(phase.id))

    def _save_human(self, hs: hm.HumanGateState) -> None:
        self.st.setdefault("human", {})[hs.phase_id] = hs.to_dict()
        self._persist_run()

    def _ask_user(self, phase: Phase, attempt: Attempt, review: Review, final: bool) -> RunState:
        """Persist the exact request (and the evidence it was based on), then wait for the user."""
        req = hm.build_request(phase, attempt.number, review, final, attempt.verification, attempt.evidence,
                               int(getattr(self.d.settings, "human_max_questions", hm.MAX_QUESTIONS) or hm.MAX_QUESTIONS))
        if not req.questions:
            raise OrchestratorError(f"Claude asked the user about {phase.id} without a usable question")
        hs = self._load_human(phase) or hm.HumanGateState(phase.id)
        hs.begin(req)
        phase.status = TaskStatus.AWAITING_HUMAN
        self._save_task(phase)
        self._save_human(hs)
        for i, q in enumerate(req.questions):
            t = req.targets[i] if i < len(req.targets) and req.targets[i] else ""
            self.d.log("CLAUDE", f"  question{f' ({t})' if t else ''}: {hm.short(q, 200)}")
        for x in req.dropped:
            self.d.log("CLAUDE", f"  not asked: {hm.short(x, 200)}")
        self.d.log("SYSTEM", f"{phase.id}: waiting for human verification ({len(req.questions)} question(s)); no worker "
                             "runs until the user answers")
        return S.AWAITING_HUMAN

    async def _on_awaiting_human(self) -> RunState:
        phase = self._require_task()
        hs = self._load_human(phase)
        if hs is None or hs.request is None:
            raise _Halt(S.BLOCKED, f"{phase.id} was waiting for human verification, but no request is recorded. "
                                   "Resume to repeat the checkpoint.", S.REVIEWING)
        if hs.status == hm.AWAITING:
            raise _Park(f"waiting for human verification of {phase.id}: {hs.request.reason}")
        if hs.feedback is None:
            raise _Park(f"waiting for human verification of {phase.id}: {hs.request.reason}")
        attempt = self._last_attempt(phase)
        done = attempt.review if attempt.review is not None and attempt.review.kind == "HUMAN_FOLLOWUP" else None
        if done is not None and done.human_feedback.get("created_at") == hs.feedback.created_at:
            # Adjudicated before an interruption: continue from the recorded decision (no second Claude call).
            if hs.status != hm.RESOLVED:
                hs.resolve(done.decision.value, done.summary, [str(c.get("id")) for c in done.derived_criteria])
                self._save_human(hs)
            return self._after_followup(phase, done)
        review = await self._human_followup(phase, attempt, hs)
        return self._after_followup(phase, review)

    async def _human_followup(self, phase: Phase, attempt: Attempt, hs: hm.HumanGateState) -> Review:
        """Claude adjudicates the user's answer. Its result replaces the ASK_USER review of the attempt, so
        advancing, the retry budget and the repair pipeline see an ordinary PASS / RETRY / BLOCKED."""
        req, fb = hs.request, hs.feedback
        await self._refresh_evidence(phase, attempt, "the human follow-up")
        asked = attempt.review.to_dict() if attempt.review is not None else {}
        rs = self._load_repair(phase)
        packet = HumanFollowupPacket(
            phase=phase, request=req, feedback=fb, goal=self.run.plan.goal or self.run.goal, repo_path=self.run.repo_path,
            checkpoint=asked, verification=list(attempt.verification), diff_stat=attempt.diff_stat,
            changed_files=attempt.worker_files(), diff=attempt.diff, human_history=list(hs.history),
            prior_diagnoses=list(rs.diagnoses) if rs is not None else [], final=req.final,
            evidence=dict(attempt.evidence))
        self.d.log("SYSTEM", f"Claude HUMAN_FOLLOWUP for phase {phase.id}: adjudicating the user's {fb.verdict}")
        try:
            review = await self.d.orchestrator.human_followup(packet)
        finally:
            self._record_orchestrator_calls(phase.id)
        if review.decision.value not in FOLLOWUP_DECISIONS:  # no ASK_USER recursion
            raise OrchestratorError(f"HUMAN_FOLLOWUP for {phase.id} returned {review.decision.value}; only "
                                    f"{', '.join(FOLLOWUP_DECISIONS)} are allowed")
        review.kind = "HUMAN_FOLLOWUP"
        review.human_request = req.to_dict()
        review.human_feedback = fb.to_dict()
        added, again = hm.merge_derived_criteria(phase.human_criteria, phase.id, review.derived_criteria, fb,
                                                 attempt.number, req.id)
        review.derived_criteria = [c.to_dict() for c in added]
        for c in again:
            self.d.log("CLAUDE", f"  the same issue was reported again: {c.id} (reported {c.reports} times; no "
                                 "duplicate criterion)")
        attempt.review = review
        self.d.db.update_attempt(attempt)
        self.d.db.add_review(self.run.id, attempt)
        self._emit_attempt(attempt)
        phase.last_checkpoint = review.decision.value
        self._save_task(phase)
        hs.resolve(review.decision.value, review.summary, [c.id for c in added])
        self._save_human(hs)
        self.d.log("CLAUDE", f"HUMAN_FOLLOWUP {phase.id}: {review.decision.value} - {review.summary}")
        for p in review.problems:
            self.d.log("CLAUDE", f"  problem: {p}")
        for c in added:
            self.d.log("CLAUDE", f"  derived criterion {c.id}: {hm.short(c.text, 200)}")
        return review

    def _after_followup(self, phase: Phase, review: Review) -> RunState:
        if review.decision == ReviewDecision.PASS:
            return S.ADVANCING
        if review.decision == ReviewDecision.RETRY:
            phase.status = TaskStatus.RETRY
            self._save_task(phase)
            return S.RETRY_PENDING
        phase.status = TaskStatus.BLOCKED
        self._save_task(phase)
        raise _Halt(S.BLOCKED, f"Claude HUMAN_FOLLOWUP marked phase {phase.id} BLOCKED: {review.summary}",
                    S.RETRY_PENDING)

    # ------------------------------------------------------------ ADVANCING / REPLANNING

    async def _on_phase_advance(self) -> RunState:
        # Idempotent: re-entered after a crash, a PASSED phase is not re-marked, but the run-level
        # bookkeeping (pending replan, current phase) is completed. The phase row is committed before
        # the run row, so a crash in between can never make a PASSED phase run again.
        phase = self.current_task
        if phase is not None:
            attempt = self._last_attempt(phase)
            review = attempt.review
            assert review is not None and review.decision == ReviewDecision.PASS
            if phase.status != TaskStatus.PASSED:
                await self._refresh_evidence(phase, attempt, "the phase was marked passed")
                self._mark_passed(phase, attempt, review)
            if review.plan_changes_required and self._is_final(phase):
                self.d.log("SYSTEM", "final audit passed with plan_changes_required, but no later phases exist; "
                                     "ignored (a follow-up needs REPLAN)")
            elif review.plan_changes_required and "pending_replan" not in self.st:
                if int(self.st.get("replans", 0)) < self._replan_allowance():
                    self.st["pending_replan"] = {
                        "phase_id": phase.id, "supersede": False,
                        "reason": review.replan_reason or f"checkpoint of {phase.id} reported that later phases must "
                                                          f"change: {review.summary}"}
                else:
                    self.d.log("SYSTEM", "checkpoint suggested plan changes but the replan limit is reached; "
                                         "continuing with the current plan")
            self.run.current_task_id = None
            self._persist_run()
        if self.st.get("pending_replan"):
            return S.REPLANNING
        self._refresh_ready()
        self._save_all_tasks()
        return S.STARTING_TASK

    def _mark_passed(self, phase: Phase, attempt: Attempt, review) -> None:
        tr = review.task_result or {}
        ev = [r for r in (attempt.evidence or {}).get("results") or []]
        # V4: a property is never recorded as verified without a SATISFIED (and fresh) evidence result.
        unverified = [f"UNVERIFIED {r.get('requirement_id')} [{r.get('kind')}]: evidence {r.get('status')}"
                      + (" (stale)" if r.get("freshness") == evm.STALE else "")
                      + f" - {hm.short(r.get('goal') or r.get('summary') or '', 160)}"
                      for r in ev if r.get("required") and not evm.is_satisfied(r)]
        decisions = [str(x) for x in tr.get("important_decisions", []) or []]
        cons = (review.consistency or {}) if review.kind == "CONSISTENCY_REVIEW" else {}
        if cons:  # accepted by a consistency review: the justification is part of the record
            decisions.append("accepted by a consistency review despite unresolved required evidence: "
                             + hm.short(str(cons.get("justification") or ""), 300)
                             + "".join(f"; {d.get('requirement_id')} {d.get('disposition')}"
                                       for d in cons.get("dispositions") or []))
        phase.result = TaskResult(
            task_id=phase.id,
            result=str(tr.get("result") or review.summary),
            files_changed=attempt.worker_files(),
            important_decisions=decisions,
            remaining_risks=[str(x) for x in (tr.get("remaining_risks") or review.remaining_risks or [])] + unverified,
            verification=[f"{v.command}: {'pass' if v.passed else (v.skipped_reason or f'exit {v.exit_code}')}"
                          for v in attempt.verification]
            + [f"evidence {r.get('requirement_id')} [{r.get('kind')}]: {r.get('status')}"
               + (" (stale)" if r.get("freshness") == evm.STALE else "") for r in ev],
        )
        unreported = [s.id for s in phase.subtasks if not s.done]
        now = time.time()
        for s in phase.subtasks:  # the lead accepted the whole phase
            if not s.done:
                s.done, s.completed_at, s.completed_via = True, now, "checkpoint"
        phase.status = TaskStatus.PASSED
        phase.finished_at = time.time()
        phase.user_guidance = ""
        phase.fresh_session_requested = False
        self._save_task(phase)
        self.st.get("phase_base", {}).pop(phase.id, None)
        self.st.get("repair", {}).pop(phase.id, None)
        self.st.get("impl_guard", {}).pop(phase.id, None)
        self.d.log("SYSTEM", f"PASSED phase {phase.id} after {phase.attempts} attempt(s)"
                   + (f"; subtask {', '.join(unreported)} not reported done by the worker, marked done with the "
                      "phase" if unreported else ""))

    async def _on_replanning(self) -> RunState:
        pending = self.st.get("pending_replan")
        if not pending:
            return S.STARTING_TASK
        phase = self.graph.get(str(pending.get("phase_id", "")))
        reviewed = [a for a in self.attempts.get(phase.id, []) if a.review is not None] if phase else []
        review = reviewed[-1].review if reviewed else None
        supersede = bool(pending.get("supersede")) and phase is not None
        if supersede and phase.status == TaskStatus.PASSED:
            supersede = False  # never rewrite a PASSED phase
        if supersede and phase.status != TaskStatus.SUPERSEDED:  # idempotent on re-entry
            phase.status = TaskStatus.SUPERSEDED
            phase.finished_at = time.time()
            phase.superseded_reason = str(pending.get("reason", ""))
            phase.result = TaskResult(
                task_id=phase.id,
                result=f"closed by REPLAN; its changes stay in the repository. {review.summary if review else ''}".strip(),
                files_changed=reviewed[-1].worker_files() if reviewed else [],
                remaining_risks=list(review.problems) if review else [])
            self._save_task(phase)
            self.st.get("phase_base", {}).pop(phase.id, None)
            self.st.get("repair", {}).pop(phase.id, None)
            if self.run.current_task_id == phase.id:
                self.run.current_task_id = None
            self._persist_run()
            self.d.log("SYSTEM", f"Phase {phase.id} SUPERSEDED by REPLAN (its work is kept): {phase.superseded_reason}")
        req = PhaseReplanRequest(
            goal=self.run.plan.goal or self.run.goal, plan=self.run.plan,
            reason=str(pending.get("reason") or "replan requested"), repo_path=self.run.repo_path,
            superseded_phase_id=phase.id if supersede else "",
            checkpoint_summary=review.summary if review else "",
            checkpoint_problems=list(review.problems) if review else [],
            user_guidance=str(pending.get("user_guidance", "")))
        self.d.log("SYSTEM", f"Requesting revised FUTURE phases: {req.reason}")
        try:
            revised = await self.d.orchestrator.replan_phases(req)
        finally:
            self._record_orchestrator_calls(None)
        merged = merge_phase_replan(self._phases(), revised)
        if not any(p.status not in TERMINAL for p in merged):
            raise _Halt(S.BLOCKED, "the revised plan contains no phase to run (every returned phase duplicated a "
                                   "completed one), so the run cannot be completed or final-audited. Resume to "
                                   "request the replan again.", S.REPLANNING)
        self.run.plan.tasks = merged
        self.graph = TaskGraph(merged)
        remaining = [p.id for p in merged if p.status not in TERMINAL]
        self.st["replans"] = int(self.st.get("replans", 0)) + 1
        self.st.setdefault("replan_history", []).append(
            {"at": time.time(), "phase_id": pending.get("phase_id"), "superseded": supersede,
             "reason": req.reason, "remaining": remaining})
        self.st.pop("pending_replan", None)
        self._refresh_ready()
        self._save_all_tasks()
        self._persist_run()
        self.d.emit("plan", {"plan": self.run.plan.to_dict()})
        self.d.log("SYSTEM", f"Phase plan revised (replan {self.st['replans']}): remaining phases "
                             f"{', '.join(remaining) or '(none)'}; completed phases unchanged")
        return S.STARTING_TASK

    # ------------------------------------------------------------ RETRY_PENDING -> REPAIRING

    async def _on_phase_retry_pending(self) -> RunState:
        """V1 retry-budget rules; a checkpoint RETRY then enters the bounded repair workflow. (A
        BLOCKED checkpoint resumed by the user keeps continuing the phase's session.)"""
        phase = self._require_task()
        nxt = await self._on_retry_pending()
        reviewed = self._last_reviewed(phase)
        if reviewed is not None and reviewed.review.decision == ReviewDecision.RETRY:
            return S.REPAIRING
        return nxt

    def _last_reviewed(self, phase: Phase) -> Attempt | None:
        return next((a for a in reversed(self.attempts.get(phase.id, [])) if a.review is not None), None)

    def _load_repair(self, phase: Phase) -> RepairState | None:
        d = (self.st.get("repair") or {}).get(phase.id)
        return RepairState.from_dict(d) if isinstance(d, dict) else None

    def _save_repair(self, rs: RepairState) -> None:
        self.st.setdefault("repair", {})[rs.phase_id] = rs.to_dict()
        self._persist_run()

    def _rlog(self, text: str) -> None:
        self.d.log("REPAIR", text)

    def _repo_rel(self, path: str) -> str:
        p = str(path or "")
        if os.path.isabs(p):
            try:
                p = os.path.relpath(p, self.run.repo_path)
            except ValueError:
                return ""
        p = rp.norm_path(p)
        return "" if p.startswith("../") else p

    # ------------------------------------------------------------ REPAIRING

    async def _on_repairing(self) -> RunState:
        phase = self._require_task()
        reviewed = self._last_reviewed(phase)
        if reviewed is None or reviewed.review is None:
            raise RuntimeError(f"{phase.id}: REPAIRING requires a reviewed phase attempt")
        history = self.attempts.setdefault(phase.id, [])
        interrupted = bool(history) and history[-1].interrupted and history[-1].kind == "repair"
        rs = self._load_repair(phase)
        if rs is None or rs.checkpoint_attempt != reviewed.number:
            rs = await self._begin_repair_attempt(phase, reviewed, rs)
            interrupted = False
        else:
            rs.guard.fold_inflight()  # a turn killed mid-flight still counts against the budget
            if rs.status == "blocked":  # the user resumed after DIAGNOSE_BLOCKER returned BLOCKED
                rs.status, rs.pending_escalation, rs.diagnoses_this_attempt = "working", "", 0
                rs.session_id, rs.guard = "", RepairGuardState()
                rs.fresh_reason = "the user resumed the blocked repair"
                self._rlog(f"{phase.id}: the user resumed the blocked repair - fresh session and budget"
                           + (" with the user's guidance" if phase.user_guidance else ""))
            elif interrupted:
                self._rlog(f"{phase.id}: continuing repair attempt {rs.repair_attempt} after an interruption (round "
                           f"{rs.round_no}; this session used {rs.guard.tool_calls} tool call(s), "
                           f"{rs.guard.wall_used:.0f}s)")
        if phase.fresh_session_requested and rs.session_id:
            rs.session_id, rs.fresh_reason = "", "a fresh session was requested by the user"
            phase.fresh_session_requested = False
        attempt = Attempt(task_id=phase.id, number=len(history) + 1, kind="repair", started_at=time.time())
        pre = await self.d.git.snapshot(self.d.settings.precise_git_snapshots) if self.d.git is not None else None
        phase.attempts += 1
        phase.status = TaskStatus.RUNNING
        phase.reported_status = ""
        self._attempt_todo = []
        self._save_task(phase)
        self.d.db.insert_attempt(self.run.id, attempt)
        history.append(attempt)
        self._emit_attempt(attempt)
        self._tool_base = 0
        self._save_repair(rs)
        self.d.log("SYSTEM", f"Phase {phase.id} attempt {phase.attempts}/{self.max_attempts_for(phase)} - REPAIRING "
                             f"the checkpoint findings (repair attempt {rs.repair_attempt})")
        try:
            await self._repair_loop(phase, attempt, rs, interrupted)
            attempt.finished_at = time.time()
            await self._attribute(phase, attempt, pre)
        except asyncio.CancelledError:
            self._abandon(phase, attempt, "interrupted by STOP")
            self._save_repair(rs)
            raise
        except _PauseBetweenTurns:
            self._abandon(phase, attempt, "paused between repair rounds (the repair continues on RESUME)")
            self._save_repair(rs)
            raise _Halt(S.PAUSED, f"paused by user between repair rounds of {phase.id}", S.REPAIRING)
        except _InfraFailure as exc:
            self._abandon(phase, attempt, str(exc))
            self._save_repair(rs)
            raise _Halt(S.ERROR, f"{exc}. Fix the worker/model server, then RESUME.", S.REPAIRING)
        except _Halt as h:
            self._abandon(phase, attempt, h.message)
            self._save_repair(rs)
            raise
        except Exception as exc:  # OrchestratorError (DIAGNOSE_BLOCKER), WorkerError, GitError, ...
            self._abandon(phase, attempt, f"{type(exc).__name__}: {exc}")
            self._save_repair(rs)
            raise
        self._save_repair(rs)
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        return S.VERIFYING

    async def _begin_repair_attempt(self, phase: Phase, reviewed: Attempt, old: RepairState | None) -> RepairState:
        """A checkpoint RETRY starts a repair attempt: record the checkpoint findings, the baseline
        evidence (= the first best-known state), the snapshot the repair starts from and the
        acceptance files that stay protected while repairing."""
        r = reviewed.review
        rs = RepairState(phase_id=phase.id, checkpoint_attempt=reviewed.number,
                         repair_attempt=(old.repair_attempt + 1) if old else 1)
        if old is not None:  # the phase's repair history (bounded) carries over
            rs.rounds, rs.diagnoses = old.rounds, old.diagnoses
        rs.checkpoint = {"summary": r.summary, "problems": list(r.problems)[:12],
                         "corrective_instructions": list(r.corrective_instructions)[:12]}
        rs.authorized = list(r.authorized_test_changes)
        rs.fresh_reason = "the checkpoint requested changes"
        human = r.kind == "HUMAN_FOLLOWUP"
        if r.kind == "CONSISTENCY_REVIEW":  # V4 pass 3: the adjudicated checkpoint asked for this repair
            rs.checkpoint["source"] = "consistency_review"
            rs.fresh_reason = "a consistency review of the checkpoint requested changes"
        if human:  # V4: the user's verification is part of what this repair must address
            fb = r.human_feedback or {}
            rs.checkpoint.update({
                "source": "human_followup",
                "human_feedback": {"verdict": str(fb.get("verdict") or ""), "note": str(fb.get("note") or "")[:2000],
                                   "questions": [str(q) for q in fb.get("questions") or []][:8]},
                "derived_criteria": [f"{c.get('id')}: {c.get('text')}" for c in r.derived_criteria][:8],
                "do_not_redo": list(r.do_not_redo)[:12]})
            rs.fresh_reason = "the user's verification found problems (human follow-up requested changes)"
        git = self.d.git
        base = None
        if git is not None:
            try:
                base = await git.snapshot(True)
            except GitError as exc:
                self._rlog(f"{phase.id}: repair baseline snapshot unavailable ({exc}); best-known restoration is off")
        if base is not None:
            rs.base_tree = base.tree or ""
            rs.base_paths = sorted(set(base.status) | set(base.generated))
        ev = rp.evaluate_verification(reviewed.verification) if reviewed.verification else None
        if ev is not None and ev.has_evidence:
            rs.current = ev.to_dict()
            exact = await self._exact_blobs(base) if base is not None and base.tree else {}
            rs.best = {**ev.to_dict(), "tree": rs.base_tree, "exact": exact, "round": 0}
        files = rp.acceptance_files(phase.verification_commands, self.run.repo_path, reviewed.worker_files(),
                                    phase.files_likely_involved)
        if files and git is not None:
            try:
                rs.protected = await git.hash_files(files, write=True)
            except GitError as exc:
                self._rlog(f"{phase.id}: acceptance test protection unavailable ({exc})")
        done = [s.id for s in phase.subtasks if s.done]
        source = "human follow-up" if human else ("consistency review" if r.kind == "CONSISTENCY_REVIEW" else "checkpoint")
        self._rlog(f"{phase.id}: entering repair mode after {source} RETRY "
                   f"(repair attempt {rs.repair_attempt}); completed subtasks stay done: "
                   f"{', '.join(done) or '(none reported)'}")
        if human:
            fb = rs.checkpoint["human_feedback"]
            self._rlog(f"{phase.id}: human feedback attached ({fb['verdict']}"
                       + (f': "{hm.short(fb["note"], 120)}"' if fb["note"] else "") + ")")
        self._rlog(f"{phase.id}: baseline verification: {ev.summary() if ev is not None else 'no verification evidence'}")
        if rs.protected:
            self._rlog(f"{phase.id}: {len(rs.protected)} protected acceptance file(s): "
                       + ", ".join(sorted(rs.protected)[:12]) + (" ..." if len(rs.protected) > 12 else ""))
        if rs.authorized:
            self._rlog(f"{phase.id}: the checkpoint authorized changes to acceptance test(s): {', '.join(rs.authorized)}")
        return rs

    async def _repair_loop(self, phase: Phase, attempt: Attempt, rs: RepairState, interrupted: bool) -> None:
        o = self.d.settings
        verify = bool(phase.verification_commands) and o.run_verification_commands
        trend_line = ""
        infra_failures = 0
        while True:
            if not rs.pending_escalation:
                trip = rp.guard_trip(rs.guard, o)  # e.g. the budget was used up before an interruption
                if trip:
                    rs.guard.trip = rs.pending_escalation = trip
            if rs.pending_escalation:
                if not await self._escalate(phase, attempt, rs):
                    break
                interrupted, trend_line = False, ""
            self._check_pause()
            kind = "fresh" if not rs.session_id else ("resume" if interrupted else "round")
            interrupted = False
            outcome, report = await self._repair_turn(phase, attempt, rs, kind, trend_line)
            if _no_work(outcome):
                infra_failures += 1
                if infra_failures >= 2:
                    raise _InfraFailure(f"the worker failed twice without doing any work (exit {outcome.exit_code}: "
                                        f"{tail(outcome.error or outcome.stderr_tail, 300)})")
                self._rlog(f"{phase.id}: repair turn failed without doing any work; retrying it once")
                self._save_repair(rs)
                continue
            infra_failures = 0
            rejected = await self._enforce_protected(phase, rs)
            await self._track_new_probes(rs)
            ev = None
            if verify:
                ev = await self._repair_verify(phase, attempt)
                trend_line = await self._observe(phase, attempt, rs, ev)
            self._add_round(rs, outcome, report, ev, trend_line, rejected)
            if report.status == "BLOCKED" and not rs.guard.trip:
                rs.guard.trip = "the worker reported PHASE STATUS: BLOCKED"
            trip = rs.guard.trip or rp.guard_trip(rs.guard, o)
            if ev is not None and ev.passed and (report.status not in ("INCOMPLETE", "BLOCKED") or trip):
                self._rlog(f"{phase.id}: full phase verification passed - handing the repair to the checkpoint")
                break
            if ev is None and (trip or report.status != "INCOMPLETE"):
                break  # no verification evidence to iterate on: the checkpoint judges the repair
            if ev is not None and ev.passed:
                trend_line = "verification passes, but you reported the repair INCOMPLETE: finish the checkpoint findings"
            if trip:
                rs.guard.trip = rs.pending_escalation = trip
                self._rlog(f"{phase.id}: repair guard stopped the session: {trip}")
            self._save_repair(rs)
        await self._ensure_best(phase, attempt, rs, "before the checkpoint")
        self._save_repair(rs)

    # ------------------------------------------------------------ one bounded repair turn

    def _repair_spec(self, phase: Phase, rs: RepairState) -> RepairSpec:
        o = self.d.settings
        files = next((a.worker_files() for a in reversed(self.attempts.get(phase.id, [])) if a.changed_files), [])
        diag = rs.last_diagnosis if rs.diagnoses and rs.diagnoses[-1].get("repair_attempt") == rs.repair_attempt \
            else None
        return RepairSpec(
            phase=phase, repo_path=self.run.repo_path, repair_attempt=rs.repair_attempt,
            attempt_number=phase.attempts, max_attempts=self.max_attempts_for(phase),
            reason=rs.fresh_reason or "the checkpoint requested changes", checkpoint=rs.checkpoint,
            current=rp.VerificationEvidence.from_dict(rs.current), best=rp.VerificationEvidence.from_dict(rs.best),
            files_changed=files, protected_tests=sorted(p for p in rs.protected if not rp.is_authorized(p, rs.authorized)),
            authorized_tests=list(rs.authorized), diagnosis=diag, notes=list(rs.notes),
            wall_seconds=rp._limit(o.repair_max_wall_seconds), tool_calls=rp._limit(o.repair_max_tool_calls),
            max_probe_files=rp._limit(o.repair_max_new_probe_files, 0),
            codebase_memory_available=self.d.codebase_memory_available,
            codebase_memory_project=self.d.codebase_memory_project if self.d.codebase_memory_available else "")

    async def _repair_turn(self, phase: Phase, attempt: Attempt, rs: RepairState, kind: str,
                           trend_line: str) -> tuple[WorkerOutcome, PhaseReport]:
        o = self.d.settings
        if kind != "fresh" and not await self.d.worker.session_resume_supported():
            rs.session_id, rs.fresh_reason, kind = "", "the worker cannot resume sessions", "fresh"
        while True:
            session = rs.session_id if kind != "fresh" else ""
            spec = self._repair_spec(phase, rs)
            if not session:
                prompt, purpose = build_repair_prompt(spec), "repair"
                rs.session_id = ""
                rs.session_no += 1
                phase.session_id, phase.session_turns = "", 0
                phase.sessions_started += 1
                self._rlog(f"{phase.id}: starting fresh repair session {rs.session_no} ({spec.reason})")
            else:
                g = rs.guard
                prompt = build_repair_round_prompt(
                    spec, rs.round_no + 1, trend_line, resumed_after_interrupt=kind == "resume",
                    wall_left=rp._limit(o.repair_max_wall_seconds) - g.wall_used,
                    tools_left=rp._limit(o.repair_max_tool_calls) - g.tool_calls)
                purpose = "repair_resume" if kind == "resume" else "repair_round"
            rs.notes = []  # delivered with this prompt
            self._save_task(phase)
            self._save_repair(rs)
            mode = "resumed" if session else "new"
            n = len(attempt.turns) + 1
            attempt.turns.append({"n": n, "purpose": purpose, "mode": mode, "session_id": session, "running": True,
                                  "started_at": time.time()})
            self.d.db.update_attempt(attempt)
            self._emit_attempt(attempt)
            self.d.log("SYSTEM", f"Qwen turn {n} for {phase.id}: {purpose} - "
                                 + (f"resuming repair session {session}" if session else "new repair session"))
            started = time.time()
            outcome = await self._guarded_run(phase, rs, PhaseTurnRequest(prompt, self.run.repo_path, session,
                                                                          phase.id, purpose))
            if session and outcome.resume_failed:
                self._record_turn(phase, attempt, purpose, mode, outcome, "", started, 0)
                self._rlog(f"{phase.id}: could not resume repair session {session}; starting a fresh one")
                rs.session_id, rs.fresh_reason, kind = "", "the previous repair session could not be resumed", "fresh"
                continue
            break
        rs.round_no += 1
        g = rs.guard
        g.inflight_tool_calls, g.inflight_wall = 0, 0.0
        g.tool_calls += outcome.tool_calls
        g.wall_used += outcome.duration
        if outcome.session_id:
            rs.session_id = phase.session_id = outcome.session_id
        phase.session_turns += 1
        if outcome.todo:
            self._attempt_todo = outcome.todo
            phase.todo = outcome.todo
        report = parse_phase_report(outcome.summary, [s.id for s in phase.subtasks], self._attempt_todo)
        if report.status:
            phase.reported_status = report.status
        self._sync_report_subtasks(phase, report)
        if outcome.context_exhausted:
            rs.session_id, rs.fresh_reason = "", "the repair session ran out of context"
        self._record_turn(phase, attempt, purpose, mode, outcome, report.status, started, 0)
        self._save_task(phase)
        return outcome, report

    async def _guarded_run(self, phase: Phase, rs: RepairState, request: PhaseTurnRequest) -> WorkerOutcome:
        """Run one worker turn under the repair guard: it is stopped as soon as the session's
        wall-clock or tool-call budget is used up or it keeps creating new probe files."""
        o, worker, g = self.d.settings, self.d.worker, rs.guard
        max_tools, max_wall = rp._limit(o.repair_max_tool_calls), rp._limit(o.repair_max_wall_seconds)
        max_probes = rp._limit(o.repair_max_new_probe_files, 0)
        tools_left = max(0, max_tools - g.tool_calls)
        wall_left = max(0.0, max_wall - g.wall_used)
        started = time.monotonic()
        base = self._tool_base
        stopped: list[str] = []

        def stop(reason: str) -> None:
            if not stopped:
                stopped.append(reason)
                self._rlog(f"{phase.id}: {reason} - stopping the repair turn")
                asyncio.ensure_future(worker.cancel())

        def on_tool(n: int) -> None:
            self.d.emit("tool_calls", {"task_id": phase.id, "count": base + n})
            g.inflight_tool_calls, g.inflight_wall = n, time.monotonic() - started
            self._save_repair(rs)  # in-flight usage survives a crash
            if n >= tools_left:
                stop(f"tool budget reached {g.tool_calls + n}/{max_tools}")

        def on_file(kind: str, path: str) -> None:
            rel = self._repo_rel(path)
            if (kind or "").lower() in ("add", "added", "create", "created") and rel and rp.is_probe_path(rel) \
                    and rel not in rs.base_paths and rel not in g.probe_files:
                g.probe_files.append(rel)
                if len(g.probe_files) > max_probes:
                    stop(f"{len(g.probe_files)} new probe/debug files without verification progress (max {max_probes})")

        worker.on_tool_call = on_tool
        worker.on_session = lambda sid: self._on_repair_session(phase, rs, sid)
        worker.on_todo = lambda items: self._on_todo(phase, items)
        worker.on_file_change = on_file
        timer = asyncio.get_running_loop().call_later(
            wall_left, lambda: stop(f"wall-clock budget reached ({g.wall_used + time.monotonic() - started:.0f}s/"
                                    f"{max_wall}s)"))
        try:
            outcome = await worker.run_turn(request)
        finally:
            timer.cancel()
            worker.on_file_change = None
        if stopped:
            g.trip = stopped[0]
        return outcome

    def _on_repair_session(self, phase: Phase, rs: RepairState, sid: str) -> None:
        if sid and sid != rs.session_id:
            rs.session_id = phase.session_id = sid
            self._save_task(phase)
            self._save_repair(rs)  # persisted immediately: an interrupted repair turn can be resumed
            self._rlog(f"{phase.id}: repair session {sid} recorded")

    # ------------------------------------------------------------ evidence after a repair turn

    async def _repair_verify(self, phase: Phase, attempt: Attempt) -> rp.VerificationEvidence:
        self.d.log("VERIFY", f"{phase.id}: repair verification (full phase verification)")
        attempt.verification = await self.d.verifier.run(phase.verification_commands, self.run.repo_path, True)
        attempt.verified_at_turn = len(attempt.turns)
        self.d.db.update_attempt(attempt)
        self._emit_attempt(attempt)
        return rp.evaluate_verification(attempt.verification)

    async def _observe(self, phase: Phase, attempt: Attempt, rs: RepairState, ev: rp.VerificationEvidence) -> str:
        """Compare the new evidence with the previous round and the best-known state; only
        measurable verification changes move the stuck counters. Returns the round's result line."""
        o, g = self.d.settings, rs.guard
        prev = rp.VerificationEvidence.from_dict(rs.current)
        best = rp.VerificationEvidence.from_dict(rs.best)
        t = rp.trend(prev, ev)
        fixed, new = rp.fixed_and_new(prev, ev)
        max_same = rp._limit(o.repair_max_same_failure_rounds)
        rs.current = ev.to_dict()
        if ev.passed:
            g.same_rounds = 0
            line = "verification PASSED"
        elif t == "same":
            g.same_rounds += 1
            self._rlog(f"{phase.id}: failure fingerprint unchanged {g.same_rounds}/{max_same} ({ev.summary()})")
            line = (f"UNCHANGED - the same failures as before ({g.same_rounds}/{max_same} rounds without any change in "
                    "the failing output)")
        else:
            g.same_rounds = 0
            if t == "improved":
                self._rlog(f"{phase.id}: verification improved: {prev.score.failing} -> {ev.score.failing} failing "
                           f"({ev.summary()})")
                line = f"IMPROVED: {prev.summary()} -> {ev.summary()}"
            elif t == "worse":
                line = f"WORSE than the previous round: {prev.summary()} -> {ev.summary()}"
            elif t == "changed":
                line = f"CHANGED failures, same count ({ev.summary()})"
            else:
                line = ev.summary()
        if fixed and prev is not None and prev.has_evidence:
            line += "; fixed: " + "; ".join(fixed[:4])
        if new and prev is not None and prev.has_evidence:
            line += "; new failures: " + "; ".join(new[:4])
        # Only an objective improvement - a score strictly better than the best-known state - is
        # progress. A different fingerprint with the same or a worse score is not, and neither is
        # beating the previous round while still short of the best (1 -> 3 -> 1 -> 3 must not reset).
        if best is not None:  # without earlier evidence the first result only sets the baseline
            if ev.score.better_than(best.score):
                g.no_improvement_rounds = 0
            else:
                g.no_improvement_rounds += 1
                max_no_imp = rp._limit(getattr(o, "repair_max_no_improvement_rounds", 4))
                what = {"same": "same failures", "changed": "different failures, score not better",
                        "worse": "worse", "improved": "better than the last round, not than the best-known state"
                        }.get(t, "score not better")
                self._rlog(f"{phase.id}: no improvement over the best-known state "
                           f"{g.no_improvement_rounds}/{max_no_imp} ({what})")
                line += (f"; no improvement over the best-known state for {g.no_improvement_rounds}/{max_no_imp} "
                         "rounds")
        if best is None or ev.score.better_than(best.score):
            tree, exact = await self._best_snapshot()
            rs.best = {**ev.to_dict(), "tree": tree, "exact": exact, "round": rs.round_no}
            g.worse_rounds = 0
            g.probe_files = []  # measurable progress
            if best is not None:
                self._rlog(f"{phase.id}: new best-known state recorded (round {rs.round_no}): {ev.summary()}"
                           + ("" if tree else " - no snapshot available, restoration impossible"))
        elif best.score.better_than(ev.score):
            g.worse_rounds += 1
            grace = max(0, int(o.repair_regression_grace_rounds))
            self._rlog(f"{phase.id}: regression detected: {best.score.failing} -> {ev.score.failing} failing (best-known "
                       f"round {rs.best.get('round')}: {best.summary()}; now {ev.summary()}; grace "
                       f"{min(g.worse_rounds, grace)}/{grace})")
            if g.worse_rounds > grace:
                if await self._restore_best(phase, attempt, rs, ev):
                    line += "; the orchestrator RESTORED the best-known state (see notes)"
            else:
                rs.notes.append(f"REGRESSION: verification is now worse than the best-known state of round "
                                f"{rs.best.get('round')} ({best.summary()} -> {ev.summary()}). Correct it in the next "
                                "round, otherwise the orchestrator restores the best-known state.")
        else:
            g.worse_rounds = 0
        max_probes = rp._limit(o.repair_max_new_probe_files, 0)
        if g.probe_files and len(g.probe_files) >= max_probes:
            rs.notes.append(f"You created {len(g.probe_files)} new probe/debug file(s) without verification progress "
                            f"(max {max_probes}); one more stops this session.")
        return line

    async def _best_snapshot(self) -> tuple[str, dict[str, str]]:
        """(tree, exact blobs) of the current working tree for a best-known record."""
        if self.d.git is None:
            return "", {}
        try:
            snap = await self.d.git.snapshot(True)
        except GitError as exc:
            self._rlog(f"snapshot unavailable: {exc}")
            return "", {}
        if not snap.tree:
            return "", {}
        return snap.tree, await self._exact_blobs(snap)

    async def _exact_blobs(self, snap) -> dict[str, str]:
        """Exact bytes (--no-filters blobs) of every changed/untracked authored file of a snapshot -
        worker-created files, earlier phases' and user changes. The snapshot TREE stores
        EOL-normalized content (core.autocrlf / .gitattributes), so restoring these paths from the
        tree alone would not reproduce the verified bytes; the blobs make that exact."""
        paths = [p for p, xy in sorted(snap.status.items()) if "D" not in xy][:rp.MAX_EXACT_BLOBS]
        try:
            return await self.d.git.hash_files(paths, write=True)
        except GitError as exc:
            self._rlog(f"exact-byte snapshot unavailable ({exc}); restoration falls back to the snapshot tree")
            return {}

    async def _restore_best(self, phase: Phase, attempt: Attempt, rs: RepairState,
                            worse: rp.VerificationEvidence) -> bool:
        """Put the best-known repair snapshot back. Only authored paths that differ between the
        two snapshots are touched (pre-existing user changes are part of both snapshots and stay);
        the regressed state is kept as a snapshot tree and logged."""
        git, best = self.d.git, rs.best
        best_ev = rp.VerificationEvidence.from_dict(best)
        if git is None or not best.get("tree") or best_ev is None:
            self._rlog(f"{phase.id}: no best-known snapshot available; cannot restore")
            return False
        self._rlog(f"{phase.id}: restoring best-known state (round {best.get('round')}: {best_ev.summary()})")
        try:
            cur = await git.snapshot(True)
            if not cur.tree:
                raise GitError("precise snapshot unavailable")
            restored = await git.restore_tree(best["tree"], cur.tree, exact=best.get("exact") or {})
        except (GitError, OSError) as exc:
            self._rlog(f"{phase.id}: restoring the best-known state failed ({exc}); continuing from the current state")
            return False
        paths = [p for _, p in restored]
        rs.guard.restores += 1
        rs.guard.worse_rounds = 0
        rs.restore_log = (rs.restore_log + [{"at": time.time(), "attempt": rs.repair_attempt, "round": rs.round_no,
                                             "regressed_tree": cur.tree, "best_tree": best["tree"],
                                             "paths": paths[:50]}])[-10:]
        self._rlog(f"{phase.id}: restored {len(paths)} file(s): {', '.join(paths[:10]) or '(none)'}; the regressed "
                   f"state is kept as snapshot tree {cur.tree}")
        await self._enforce_protected(phase, rs)
        ev = await self._repair_verify(phase, attempt)
        rs.current = ev.to_dict()
        rs.notes.append(f"REGRESSION RESTORED: your later changes made verification worse ({worse.summary()} vs the "
                        f"best-known {best_ev.summary()}). The orchestrator restored the best-known state of round "
                        f"{best.get('round')} ({', '.join(paths[:10]) or 'no files'}); verification now: {ev.summary()}. "
                        "Start from the restored files and do not repeat the approach that regressed.")
        return True

    async def _ensure_best(self, phase: Phase, attempt: Attempt, rs: RepairState, when: str) -> None:
        cur = rp.VerificationEvidence.from_dict(rs.current)
        best = rp.VerificationEvidence.from_dict(rs.best)
        if cur is not None and best is not None and best.score.better_than(cur.score):
            self._rlog(f"{phase.id}: the current state ({cur.summary()}) is worse than the best-known state {when}")
            await self._restore_best(phase, attempt, rs, cur)

    async def _enforce_protected(self, phase: Phase, rs: RepairState) -> list[str]:
        """Existing acceptance files stay exactly as they were when the repair began, unless the
        lead authorized a change. Returns the files whose edits were rejected (and restored)."""
        git = self.d.git
        if git is None or not rs.protected:
            return []
        try:
            now = await git.hash_files(list(rs.protected))
        except GitError as exc:
            self._rlog(f"{phase.id}: protected acceptance test check unavailable: {exc}")
            return []
        changed = [p for p, blob in rs.protected.items() if now.get(p) != blob]
        allowed = [p for p in changed if rp.is_authorized(p, rs.authorized)]
        restored: list[str] = []
        for p in changed:
            if p in allowed:
                continue
            try:
                await git.write_blob(p, rs.protected[p])
                restored.append(p)
            except (GitError, OSError) as exc:
                self._rlog(f"{phase.id}: could not restore protected acceptance file {p}: {exc}")
        if allowed:
            self._rlog(f"{phase.id}: acceptance test change authorized by the lead: {', '.join(allowed)}")
        if restored:
            what = ", ".join(restored)
            self._rlog(f"{phase.id}: protected acceptance test modification rejected: {what} (restored to the original; "
                       "the repair continues against the original acceptance criteria)")
            rs.rejected_tests = sorted(set(rs.rejected_tests) | set(restored))
            rs.notes.append(f"Your changes to the protected acceptance test(s) {what} were REJECTED and reverted. Fix "
                            "the implementation, not the tests (new diagnostic tests in new files are fine).")
        return restored

    async def _track_new_probes(self, rs: RepairState) -> None:
        if self.d.git is None:
            return
        try:
            entries = await self.d.git.status()
        except GitError:
            return
        base = set(rs.base_paths)
        for e in entries:
            if "?" in e.xy and e.path not in base and rp.is_probe_path(e.path) and e.path not in rs.guard.probe_files:
                rs.guard.probe_files.append(e.path)

    def _add_round(self, rs: RepairState, o: WorkerOutcome, report: PhaseReport, ev: rp.VerificationEvidence | None,
                   trend_line: str, rejected: list[str]) -> None:
        rs.add_round({
            "n": rs.round_no, "attempt": rs.repair_attempt, "session": rs.session_no,
            "trend": trend_line[:300] if ev is not None else "no verification", "summary":
            ev.summary() if ev is not None else "no verification evidence", "fingerprint": ev.fingerprint if ev else "",
            "tool_calls": o.tool_calls, "duration": round(o.duration, 1), "status": report.status,
            "stopped": rs.guard.trip, "worker": tail(o.summary or "", 400),
            "note": ("rejected edits of protected tests: " + ", ".join(rejected)) if rejected else ""})

    # ------------------------------------------------------------ escalation (DIAGNOSE_BLOCKER)

    async def _escalate(self, phase: Phase, attempt: Attempt, rs: RepairState) -> bool:
        """The guard stopped the session: ask Claude for a cheap, structured diagnosis instead of
        giving the worker another long continuation. False = no diagnosis budget left (the
        best-known state goes to the checkpoint)."""
        reason = rs.pending_escalation
        limit = max(0, int(self.d.settings.repair_max_diagnoses))
        if rs.diagnoses_this_attempt >= limit:
            self._rlog(f"{phase.id}: {reason}; escalation budget used ({rs.diagnoses_this_attempt}/{limit} diagnoses) - "
                       "handing the best-known state to the checkpoint")
            rs.pending_escalation = ""
            self._save_repair(rs)
            return False
        await self._ensure_best(phase, attempt, rs, "before escalating")
        self._rlog(f"{phase.id}: escalating blocker to Claude ({reason})")
        packet = await self._diagnosis_packet(phase, rs, reason)
        self.d.log("CLAUDE", f"DIAGNOSE_BLOCKER {phase.id}: {reason}")
        try:
            d = await self.d.orchestrator.diagnose_blocker(packet)
        finally:
            self._record_orchestrator_calls(phase.id)
        rs.diagnoses_this_attempt += 1
        rs.add_diagnosis({**d.to_dict(), "reason": reason, "at": time.time(), "repair_attempt": rs.repair_attempt})
        rs.pending_escalation = ""
        self.d.log("CLAUDE", f"DIAGNOSE_BLOCKER {phase.id}: {d.status} - {d.diagnosis}")
        for x in d.recommended_actions:
            self.d.log("CLAUDE", f"  do: {x}")
        for x in d.avoid:
            self.d.log("CLAUDE", f"  avoid: {x}")
        if d.authorized_test_changes:
            rs.authorized = sorted(set(rs.authorized) | set(d.authorized_test_changes))
            self._rlog(f"{phase.id}: the lead authorized changes to acceptance test(s): "
                       f"{', '.join(d.authorized_test_changes)}")
        if d.status == "BLOCKED":
            rs.status = "blocked"
            phase.status = TaskStatus.BLOCKED
            self._save_task(phase)
            self._save_repair(rs)
            raise _Halt(S.BLOCKED, f"Claude diagnosed the stuck repair of {phase.id} as BLOCKED: "
                                   f"{d.blocked_reason or d.diagnosis}. RESUME (with guidance) continues the repair in "
                                   "a fresh session.", S.REPAIRING)
        rs.session_id, rs.guard = "", RepairGuardState()
        rs.fresh_reason = "a new session with the lead's diagnosis of the blocker"
        self._save_repair(rs)
        return True

    async def _diagnosis_packet(self, phase: Phase, rs: RepairState, reason: str) -> rp.DiagnosisPacket:
        changed: list[str] = []
        diff = ""
        git = self.d.git
        if git is not None and rs.base_tree:
            try:
                post = await git.snapshot(True)
                if post.tree:
                    ch = await git.attempt_changes(snapshot_from_dict({"tree": rs.base_tree}), post)
                    changed, diff = [c.path for c in ch.changed_files], ch.diff
            except GitError as exc:
                self._rlog(f"{phase.id}: repair diff unavailable for the diagnosis ({exc})")
        rounds = [r for r in rs.rounds if r.get("attempt") == rs.repair_attempt] or rs.rounds[-6:]
        return rp.DiagnosisPacket(
            phase=phase, reason=reason, repo_path=self.run.repo_path, goal=self.run.plan.goal or self.run.goal,
            checkpoint=rs.checkpoint, current=rp.VerificationEvidence.from_dict(rs.current),
            best=rp.VerificationEvidence.from_dict(rs.best), changed_files=changed, diff=diff, rounds=rounds,
            previous_diagnoses=list(rs.diagnoses), protected_tests=sorted(rs.protected),
            user_guidance=phase.user_guidance)

    def _repair_notes(self, phase: Phase, attempt: Attempt) -> list[str]:
        """Compact repair summary for the checkpoint that reviews a repair attempt."""
        if attempt.kind != "repair":
            return []
        rs = self._load_repair(phase)
        if rs is None:
            return []
        rounds = [r for r in rs.rounds if r.get("attempt") == rs.repair_attempt]
        cur = rp.VerificationEvidence.from_dict(rs.current)
        best = rp.VerificationEvidence.from_dict(rs.best)
        notes = [f"repair attempt {rs.repair_attempt}: {len(rounds)} round(s) in {rs.session_no} fresh session(s)"]
        fb = (rs.checkpoint or {}).get("human_feedback")
        if fb:
            notes.append(f"this repair addressed the user's verification ({fb.get('verdict')})"
                         + (f": {hm.short(fb.get('note', ''), 300)}" if fb.get("note") else ""))
        if cur is not None:
            notes.append(f"final verification: {cur.summary()}")
        if best is not None and best.has_evidence and (cur is None or best.fingerprint != cur.fingerprint):
            notes.append(f"best-known state: {best.summary()} (round {rs.best.get('round')})")
        restores = [x for x in rs.restore_log if x.get("attempt") == rs.repair_attempt]
        if restores:
            notes.append(f"{len(restores)} regression(s) restored to the best-known state")
        if rs.rejected_tests:
            notes.append("rejected (reverted) edits of protected acceptance tests: " + ", ".join(rs.rejected_tests))
        for d in (x for x in rs.diagnoses if x.get("repair_attempt") == rs.repair_attempt):
            notes.append(f"DIAGNOSE_BLOCKER ({d.get('reason')}): {d.get('status')} - {d.get('diagnosis')}")
        if rs.protected:
            notes.append("protected acceptance files: " + ", ".join(sorted(rs.protected)[:12]))
        return notes
