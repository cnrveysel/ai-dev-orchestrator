"""EvidenceExecutor: runs ONLY validated evidence-plan steps, within hard bounds, and turns what
actually happened into per-requirement EvidenceResults.

Bounds (settings): total wall time, per-step timeout, number of steps, captured output bytes,
number/size of artifacts, retries of a step that timed out or errored.

It is verification, not implementation: it never edits, repairs or commits anything. Every step
is recorded (RUNNING first, then its outcome) so a crash leaves an honest trace: an interrupted
step is never trusted and runs again, and resources started by the interrupted collection are
stopped. If a step changed the project anyway, a mutation guard restores it where it can (git
snapshot, or a bounded snapshot for other directories) and the affected evidence becomes
INCONCLUSIVE - protection that was not available is reported as such.

Freshness (V4 pass 3): every executed step is stamped with provenance (repository fingerprint,
capability definition, file inputs, signature). A completed step is skipped only while it is
provably fresh (or nothing shows otherwise within the same attempt); a step recorded before a change
of its inputs runs again. A completed step of an EARLIER attempt is reused only when its signature
matches and its inputs are provably unchanged.

Requirement status (derived only from executed steps):
  strategy human / unavailable, a step UNAVAILABLE, a resource that cannot start -> UNAVAILABLE
  a step FAILED (exit code, failure marker, missing marker), a service that exits
  before it is ready                                                              -> FAILED
    (exit 0 with the planned marker missing: cause "unobserved" - FAILED, but a bounded replan may try
    a capability that observes the property)
  a step TIMEOUT / ERROR, a service that never gets ready, part of the steps ran  -> INCONCLUSIVE
  every step PASSED with DIRECT coverage                                          -> SATISFIED
  every step PASSED with INDIRECT coverage (supporting evidence only)             -> INCONCLUSIVE
  nothing ran because a budget was used up                                        -> SKIPPED
The cause says why: product (a check FAILED), strategy (the strategy could not be executed - the
only cause a bounded replan may address), plan, budget, timeout, mutation.
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any, Callable

from app.evidence import freshness as fr
from app.evidence.artifacts import ArtifactStore
from app.evidence.lifecycle import ResourceManager
from app.evidence.model import (
    CAUSE_BUDGET, CAUSE_MUTATION, CAUSE_PLAN, CAUSE_PRODUCT, CAUSE_STRATEGY, CAUSE_TIMEOUT, CAUSE_UNOBSERVED, CONFIRMED,
    DIRECT, ERROR,
    EXECUTE, FAILED, FRESH, HUMAN, INCONCLUSIVE, INTERRUPTED, PASSED, REUSED, RUNNING, SATISFIED, SKIPPED, STEP_FINAL,
    TIMEOUT, UNAVAILABLE, Artifact, CapabilitySnapshot, EvidenceBundle, EvidencePlan, EvidenceResult, EvidenceState,
    PlanStep, RequirementPlan, StepResult,
)
from app.evidence.mutation import GitMutationGuard, MutationGuard
from app.evidence.providers.base import ExecContext, StepOutcome

STATE_TAIL_CHARS = 1500  # step output kept in the run state; the artifact file holds the rest
MAX_DETAILS_CHARS = 4000
MAX_DIAGNOSTIC_LINES = 10
_RUN, _DONE, _REUSE = "run", "done", "reuse"


@dataclass(frozen=True)
class EvidenceLimits:
    wall_seconds: float = 600
    step_timeout: float = 180
    max_steps: int = 12
    max_output_bytes: int = 16000  # captured per step for the artifact
    max_artifacts: int = 24
    max_artifact_bytes: int = 1_000_000
    max_retries: int = 0
    ready_timeout: float = 60

    @classmethod
    def from_settings(cls, o: Any) -> "EvidenceLimits":
        def g(name: str, default: Any) -> Any:
            return getattr(o, name, default)
        return cls(wall_seconds=max(1.0, float(g("evidence_max_wall_seconds", 600))),
                   step_timeout=max(1.0, float(g("evidence_step_timeout_seconds", 180))),
                   max_steps=max(0, int(g("evidence_max_steps", 12))),
                   max_output_bytes=max(256, int(g("evidence_max_output_bytes", 16000))),
                   max_artifacts=max(0, int(g("evidence_max_artifacts", 24))),
                   max_artifact_bytes=max(1024, int(g("evidence_max_artifact_bytes", 1_000_000))),
                   max_retries=max(0, min(3, int(g("evidence_max_retries", 0)))),
                   ready_timeout=max(1.0, float(g("evidence_service_ready_timeout_seconds", 60))))


class EvidenceExecutor:
    def __init__(self, registry: Any, limits: EvidenceLimits, artifacts: ArtifactStore, *, repo_path: str,
                 persist: Callable[[], None], log: Callable[[str], None], git: Any = None,
                 guard: MutationGuard | None = None, repo_fp: str = "", repo_fp_kind: str = ""):
        self.registry, self.limits, self.artifacts = registry, limits, artifacts
        self.repo_path, self.persist, self.log, self.git = repo_path, persist, log, git
        self.guard = guard if guard is not None else (GitMutationGuard(git, log) if git is not None else None)
        self.repo_fp, self.repo_fp_kind = repo_fp, repo_fp_kind
        self._notes: list[str] = []

    # ------------------------------------------------------------ entry point

    async def run(self, state: EvidenceState, plan: EvidencePlan, snapshot: CapabilitySnapshot) -> EvidenceBundle:
        notes: list[str] = []
        self._notes = notes
        resources = ResourceManager(state, self.persist, self.log)
        notes += resources.reap_stale()  # leftovers of an interrupted collection: stopped, never evidence
        if state.inflight:
            rec = state.steps.get(state.inflight)
            if rec is not None and rec.get("status") == RUNNING:
                rec.update(status=INTERRUPTED, reason="interrupted (the application stopped); the step runs again",
                           finished_at=time.time())
                notes.append(f"step {state.inflight} was interrupted; its partial result is ignored and it runs again")
            state.inflight = ""
        state.status = "executing"
        if not state.started_at:
            state.started_at = time.time()
        self.persist()
        decisions = {s.key: self._decide(state, rp, s, snapshot) for rp in plan.requirements if rp.strategy == EXECUTE
                     for s in rp.steps}
        token = await self.guard.before() if self.guard is not None and _RUN in decisions.values() else None
        results: list[EvidenceResult] = []
        try:
            for rp in plan.requirements:
                results.append(await self._requirement(state, rp, snapshot, resources, decisions, plan.generation))
        finally:
            await asyncio.shield(resources.stop_all())  # STOP everything, also on errors and STOP
        guard_info: dict[str, Any] = {}
        if self.guard is not None and _RUN in decisions.values():
            report = await self.guard.after(token)
            guard_info = self.guard.describe(token)
            if report.detected:
                if report.unrestored:
                    notes.append("evidence steps changed repository files; restoration cannot be guaranteed "
                                 f"({report.detail}): " + ", ".join(report.unrestored[:10]))
                if report.restored:
                    notes.append("evidence steps changed repository files; restored to the pre-evidence state: "
                                 + ", ".join(report.restored[:10]))
                for r in results:
                    if r.status == SATISFIED:
                        r.status, r.confidence, r.cause = INCONCLUSIVE, "none", CAUSE_MUTATION
                        r.failure_reason = ("a step modified repository files (restored); the result is not trusted"
                                            if not report.unrestored else
                                            "a step modified repository files (NOT restored); the result is not trusted")
        elif self.guard is None and _RUN in decisions.values():
            guard_info = {"kind": "none", "exact": False, "detail": "no repository-mutation protection was available"}
        if self.artifacts.dropped:
            why = "; ".join(dict.fromkeys(self.artifacts.drop_reasons)) or f"limit {self.limits.max_artifacts}"
            notes.append(f"{self.artifacts.dropped} artifact(s) not stored ({why})")
        state.status = "complete"
        state.discovery = snapshot.to_dict()  # capabilities confirmed by execution stay confirmed
        bundle = EvidenceBundle(
            phase_id=state.phase_id, attempt=state.attempt, results=results,
            artifacts=[Artifact.from_dict(a) for a in state.artifacts], plan_summary=plan.summary,
            plan_source=plan.source, limitations=list(plan.limitations),
            capabilities=_used_capabilities(plan, snapshot), capability_summary=snapshot.summary(), notes=notes,
            wall_used=round(state.wall_used, 1),
            steps_run=sum(1 for s in state.steps.values()
                          if s.get("status") in STEP_FINAL - {SKIPPED} and not s.get("reused_from")),
            created_at=time.time(), repo_fp=self.repo_fp, repo_fp_kind=self.repo_fp_kind,
            plan_fingerprint=plan.fingerprint, plan_generation=plan.generation, guard=guard_info,
            replans=[dict(x) for x in state.replan_log if x.get("attempt") == state.attempt])
        state.bundle = bundle.to_dict()
        self.persist()
        return bundle

    # ------------------------------------------------------------ freshness decisions

    def _identity(self, rp: RequirementPlan, step: PlanStep, snapshot: CapabilitySnapshot) -> tuple[Any, Any, str, str,
                                                                                                    dict[str, str] | None]:
        cap = snapshot.get(step.capability_id)
        prov = self.registry.provider(cap.provider) if cap is not None else None
        if cap is None or prov is None:
            return cap, prov, "", "", None
        cap_fp = fr.capability_fp(cap)
        res_fps = [fr.capability_fp(c) for c in (snapshot.get(x) for x in rp.resources) if c is not None]
        sig = fr.step_signature(cap_fp, step, res_fps)
        inputs = fr.file_inputs(self.repo_path, prov.input_files(cap, step, self.repo_path)) \
            if prov.input_scope(cap, step, self.repo_path) == fr.SCOPE_FILES else None
        return cap, prov, cap_fp, sig, inputs

    def _decide(self, state: EvidenceState, rp: RequirementPlan, step: PlanStep, snapshot: CapabilitySnapshot) -> str:
        cap, prov, cap_fp, sig, inputs = self._identity(rp, step, snapshot)
        rec = state.steps.get(step.key)
        if rec is not None and rec.get("status") in STEP_FINAL:
            if cap is None:
                return _DONE
            fresh, why = fr.check(rec.get("provenance"), cap_fp=cap_fp, repo_fp=self.repo_fp, inputs=inputs)
            if fresh is False:  # recorded before its inputs changed: never trusted
                self._notes.append(f"{step.key} ran before a change ({why}); it runs again")
                return _RUN
            return _DONE  # fresh, or nothing shows a change within this attempt
        if cap is None or not sig:
            return _RUN
        pooled = state.reusable.get(sig)
        if pooled is not None and pooled.get("status") in (PASSED, FAILED):
            fresh, _why = fr.check(pooled.get("provenance"), cap_fp=cap_fp, repo_fp=self.repo_fp, inputs=inputs)
            if fresh is True:  # across attempts only provable freshness counts
                return _REUSE
        return _RUN

    def _reuse(self, state: EvidenceState, rp: RequirementPlan, step: PlanStep, snapshot: CapabilitySnapshot) -> None:
        _cap, _prov, _fp, sig, _inputs = self._identity(rp, step, snapshot)
        pooled = dict(state.reusable[sig])
        attempt = int(pooled.pop("attempt", 0) or 0)
        metas = pooled.pop("artifact_meta", []) or []
        known = {a.get("id") for a in state.artifacts}
        state.artifacts += [dict(m) for m in metas if m.get("id") not in known]
        rec = StepResult.from_dict(pooled)
        rec.key, rec.requirement_id, rec.capability_id, rec.reused_from = step.key, rp.requirement_id, \
            step.capability_id, attempt
        self._record(state, rec)
        self.log(f"{step.key} {step.capability_id}: {rec.status} (reused from attempt {attempt}; its inputs are "
                 "unchanged)")

    # ------------------------------------------------------------ one requirement

    @staticmethod
    def _needs_run(state: EvidenceState, key: str) -> bool:
        rec = state.steps.get(key)
        return rec is None or rec.get("status") not in STEP_FINAL

    def _ctx(self, state: EvidenceState, timeout: float, rp: RequirementPlan | None = None,
             resources: ResourceManager | None = None) -> ExecContext:
        ctx = ExecContext(self.repo_path, timeout, self.limits.max_output_bytes, self.log,
                          min(self.limits.ready_timeout, max(0.5, self.limits.wall_seconds - state.wall_used)),
                          phase_id=state.phase_id, max_artifact_bytes=self.limits.max_artifact_bytes,
                          collection_started=state.started_at)
        if rp is not None:
            ctx.requirement_id = rp.requirement_id
            arts = {a.get("id"): a for a in state.artifacts}
            ctx.prior_artifacts = {s.key: [arts[x] for x in (state.steps.get(s.key) or {}).get("artifacts") or []
                                           if x in arts] for s in rp.steps if s.key in state.steps}
            if resources is not None:
                ctx.resources = resources.handles_for(rp.requirement_id)
        return ctx

    async def _requirement(self, state: EvidenceState, rp: RequirementPlan, snapshot: CapabilitySnapshot,
                           resources: ResourceManager, decisions: dict[str, str], generation: int) -> EvidenceResult:
        base = dict(requirement_id=rp.requirement_id, kind=rp.kind, required=rp.required, goal=rp.goal,
                    strategy=rp.strategy, human_derived=rp.requirement_id.startswith("HF-"))
        if rp.strategy != EXECUTE:
            why = rp.reason or ("needs a person to observe the running result" if rp.strategy == HUMAN
                                else "no available capability can provide it")
            prefix = "human observation needed: " if rp.strategy == HUMAN else "no automated evidence: "
            return EvidenceResult(**base, status=UNAVAILABLE, source="plan", failure_reason=why, summary=prefix + why,
                                  cause=CAUSE_PLAN)
        started = time.time()
        extra_artifacts: list[str] = []
        problem = ("", "", "")  # (status of the failed resource, detail, its capability id)
        to_run = [s for s in rp.steps if decisions.get(s.key) == _RUN]
        if rp.resources and to_run:
            problem = await self._start_resources(state, rp, snapshot, resources, extra_artifacts)
        for step in rp.steps:
            action = decisions.get(step.key, _RUN)
            if action == _DONE:
                continue  # completed (and still fresh) before an interruption: not repeated
            if action == _REUSE:
                self._reuse(state, rp, step, snapshot)
                continue
            if problem[0]:
                self._record(state, StepResult(step.key, rp.requirement_id, step.capability_id, SKIPPED,
                                               reason=problem[1], finished_at=time.time()))
                continue
            await self._step(state, rp, step, snapshot, resources)
        for rec, prov, cap, handle in list(resources.active) if not problem[0] else []:  # CAPTURE, then STOP
            if rec.get("requirement_id") == rp.requirement_id:
                art = self._save_artifact(state, "log", f"{cap.id}-runtime", prov.capture(handle),
                                          f"{cap.name} output while {rp.requirement_id} was observed", cap.id)
                if art is not None:
                    extra_artifacts.append(art.id)
        await resources.stop_all()
        steps = [StepResult.from_dict(state.steps[s.key]) for s in rp.steps if s.key in state.steps]
        result = self._result(rp, steps, base, started, problem, extra_artifacts)
        if result.source != "plan":
            ran = [s for s in steps if s.status != SKIPPED]
            # "reused" only when nothing that produced the result ran now: a result decided by a step that ran in
            # this collection is fresh, even if unchanged supporting steps were reused
            result.freshness = REUSED if ran and all(s.reused_from for s in ran) else FRESH
            result.provenance = fr.result_provenance(steps, repo_fp=self.repo_fp, repo_fp_kind=self.repo_fp_kind,
                                                     generation=generation)
            lines: list[str] = []
            for s in steps:
                for x in (s.details or {}).get("diagnostics") or []:
                    if x not in lines:
                        lines.append(str(x)[:300])
            result.diagnostics = lines[:MAX_DIAGNOSTIC_LINES]
        return result

    async def _start_resources(self, state: EvidenceState, rp: RequirementPlan, snapshot: CapabilitySnapshot,
                               resources: ResourceManager, arts: list[str]) -> tuple[str, str, str]:
        """START + READY_PROBE of the requirement's resources. Returns ("", "", "") or (status, detail, cid)."""
        for cid in rp.resources:
            cap = snapshot.get(cid)
            prov = self.registry.provider(cap.provider) if cap is not None else None
            if cap is None or prov is None or not cap.usable:
                return "unavailable", f"resource {cid} is not available", cid
            t0 = time.monotonic()
            handle, status, detail = await resources.start(prov, cap, rp.requirement_id, self._ctx(state, 0))
            state.wall_used += time.monotonic() - t0
            self.persist()
            if handle is not None and status == "ready":
                self._touch(cap, f"{rp.requirement_id} resource", confirmed=True)
                continue
            if handle is not None:
                art = self._save_artifact(state, "log", f"{cid}-startup", prov.capture(handle),
                                          f"{cap.name} output while starting for {rp.requirement_id}", cid)
                if art is not None:
                    arts.append(art.id)
            return status, f"resource {cid} ({cap.name}) {status}: {detail}", cid
        return "", "", ""

    @staticmethod
    def _result(rp: RequirementPlan, steps: list[StepResult], base: dict[str, Any], started: float,
                problem: tuple[str, str, str], extra_artifacts: list[str]) -> EvidenceResult:
        statuses = [s.status for s in steps]
        common = dict(base, capability_ids=list(dict.fromkeys([*rp.resources, *(s.capability_id for s in steps)])),
                      steps=[s.key for s in steps], artifacts=[a for s in steps for a in s.artifacts] + extra_artifacts,
                      duration=round(sum(s.duration for s in steps), 1), started_at=started, finished_at=time.time())
        proxy = "observed" if rp.coverage == DIRECT else "proxy"
        if problem[0] and all(s == SKIPPED for s in statuses):
            status = {"unavailable": UNAVAILABLE, "exited": FAILED}.get(problem[0], INCONCLUSIVE)
            cause = CAUSE_PRODUCT if status == FAILED else CAUSE_STRATEGY  # never ready / cannot start: strategy
            return EvidenceResult(**common, status=status, summary=problem[1], failure_reason=problem[1],
                                  confidence=proxy if status == FAILED else "none", cause=cause,
                                  failed_capability=problem[2] if cause == CAUSE_STRATEGY else "")
        bad = next((s for s in steps if s.status == UNAVAILABLE), None)
        if bad is not None:
            return EvidenceResult(**common, status=UNAVAILABLE, summary=bad.summary or bad.reason,
                                  failure_reason=bad.reason, cause=CAUSE_STRATEGY, failed_capability=bad.capability_id)
        bad = next((s for s in steps if s.status == FAILED), None)
        if bad is not None:
            unobserved = (bad.details or {}).get("cause") == CAUSE_UNOBSERVED
            return EvidenceResult(**common, status=FAILED, summary=bad.summary or bad.reason, failure_reason=bad.reason,
                                  confidence=proxy, cause=CAUSE_UNOBSERVED if unobserved else CAUSE_PRODUCT,
                                  failed_capability=bad.capability_id if unobserved else "")
        bad = next((s for s in steps if s.status in (TIMEOUT, ERROR, INTERRUPTED, RUNNING)), None)
        if bad is not None:
            cause = {TIMEOUT: CAUSE_TIMEOUT, ERROR: CAUSE_STRATEGY}.get(bad.status, "")
            return EvidenceResult(**common, status=INCONCLUSIVE, summary=bad.summary or bad.reason,
                                  failure_reason=f"{bad.key}: {bad.status.lower()} - {bad.reason}", cause=cause,
                                  failed_capability=bad.capability_id if cause == CAUSE_STRATEGY else "")
        if not steps or all(s == SKIPPED for s in statuses):
            why = next((s.reason for s in steps if s.reason), "no step ran")
            return EvidenceResult(**common, status=SKIPPED, summary=why, failure_reason=why, cause=CAUSE_BUDGET)
        if SKIPPED in statuses:
            why = next(s.reason for s in steps if s.status == SKIPPED)
            return EvidenceResult(**common, status=INCONCLUSIVE, summary=f"only part of the planned steps ran: {why}",
                                  failure_reason=why, cause=CAUSE_BUDGET)
        summary = "; ".join(s.summary for s in steps if s.summary)[:600]
        if rp.coverage != DIRECT:
            return EvidenceResult(**common, status=INCONCLUSIVE, confidence="proxy",
                                  summary=f"supporting evidence only (indirect): {summary}",
                                  failure_reason="the steps passed but do not observe the property itself",
                                  cause=CAUSE_PLAN)
        return EvidenceResult(**common, status=SATISFIED, summary=summary, confidence="observed")

    # ------------------------------------------------------------ one step

    async def _step(self, state: EvidenceState, rp: RequirementPlan, step: PlanStep, snapshot: CapabilitySnapshot,
                    resources: ResourceManager) -> None:
        cap, prov, cap_fp, sig, inputs = self._identity(rp, step, snapshot)
        prior = state.steps.get(step.key)
        tries = int(prior.get("tries") or 0) if prior and prior.get("status") in (RUNNING, INTERRUPTED) else 0
        if cap is None or prov is None or not cap.usable:
            self._record(state, StepResult(step.key, rp.requirement_id, step.capability_id, UNAVAILABLE,
                                           reason="capability not available", finished_at=time.time()))
            return
        ran = sum(1 for k, s in state.steps.items()
                  if k != step.key and s.get("status") != SKIPPED and not s.get("reused_from"))
        if ran >= self.limits.max_steps:
            self._record(state, StepResult(step.key, rp.requirement_id, cap.id, SKIPPED, finished_at=time.time(),
                                           reason=f"evidence step budget ({self.limits.max_steps}) used up"))
            return
        scope = prov.input_scope(cap, step, self.repo_path)
        while True:
            remaining = self.limits.wall_seconds - state.wall_used
            if remaining < 0.5:
                self._record(state, StepResult(step.key, rp.requirement_id, cap.id, SKIPPED, finished_at=time.time(),
                                               reason=f"evidence wall-time budget ({self.limits.wall_seconds:.0f}s) "
                                                      "used up"))
                return
            declared = float(cap.constraints.get("timeout_seconds") or self.limits.step_timeout)
            timeout = max(0.5, min(self.limits.step_timeout, declared, remaining))
            tries += 1
            started = time.time()
            self._record(state, StepResult(step.key, rp.requirement_id, cap.id, RUNNING, started_at=started,
                                           tries=tries), inflight=True)
            t0 = time.monotonic()
            try:
                outcome = await prov.execute(cap, step, self._ctx(state, timeout, rp, resources))
            except asyncio.CancelledError:
                state.wall_used += time.monotonic() - t0
                self.persist()  # stays RUNNING and in flight: never trusted after a restart
                raise
            except Exception as exc:  # a provider bug is an ERROR, never a pass
                outcome = StepOutcome(ERROR, reason=f"{type(exc).__name__}: {exc}"[:300],
                                      summary=f"{cap.id} raised {type(exc).__name__}")
            elapsed = time.monotonic() - t0
            state.wall_used += elapsed
            if outcome.status == TIMEOUT and timeout < min(declared, self.limits.step_timeout):
                outcome.reason = f"evidence wall-time budget ({self.limits.wall_seconds:.0f}s) reached"
            arts = []
            art = self._save_artifact(state, outcome.artifact_kind, step.key, outcome.output,
                                      f"output of {step.key} ({cap.name})", cap.id)
            if art is not None:
                arts.append(art.id)
            for f in outcome.files or []:
                a = self._save_file(state, step, rp, cap, f)
                if a is not None:
                    arts.append(a.id)
            tail = outcome.output[-STATE_TAIL_CHARS:]
            details = dict(outcome.details or {})
            if outcome.cause:  # the provider's hint (strategy / unobserved) - the requirement's cause derives from it
                details["cause"] = outcome.cause
            if outcome.diagnostics:
                details["diagnostics"] = [str(x)[:300] for x in outcome.diagnostics[:MAX_DIAGNOSTIC_LINES]]
            if len(json.dumps(details, default=str)) > MAX_DETAILS_CHARS:
                details = {"diagnostics": details.get("diagnostics", [])[:MAX_DIAGNOSTIC_LINES], "truncated": True,
                           **({"cause": details["cause"]} if details.get("cause") else {})}
            provenance = fr.make_provenance(cap=cap, cap_fp=cap_fp, signature=sig, scope=scope, repo_fp=self.repo_fp,
                                            repo_fp_kind=self.repo_fp_kind, inputs=inputs, attempt=state.attempt)
            self._record(state, StepResult(
                step.key, rp.requirement_id, cap.id, outcome.status, outcome.exit_code, outcome.summary[:400],
                tail, outcome.truncated or len(outcome.output) > STATE_TAIL_CHARS, round(outcome.duration or elapsed, 2),
                started, time.time(), arts, tries, outcome.reason[:300], provenance=provenance, details=details))
            executed = outcome.status in (PASSED, FAILED)
            if executed and cap.confidence != CONFIRMED:
                cap.confidence, cap.reason = CONFIRMED, f"executed during evidence collection ({outcome.status})"
            self._touch(cap, step.key, confirmed=executed)
            self.log(f"{step.key} {cap.id}: {outcome.status}" + (f" - {outcome.summary[:160]}" if outcome.summary else ""))
            if outcome.status in (TIMEOUT, ERROR) and tries <= self.limits.max_retries:
                self.log(f"{step.key}: retry {tries}/{self.limits.max_retries}")
                continue
            return

    @staticmethod
    def _touch(cap: Any, used_by: str, confirmed: bool) -> None:
        """Provenance: when the capability was last validated by use, and by which step."""
        p = cap.provenance
        p["validated_at"] = time.time()
        p["used_by"] = (list(p.get("used_by") or []) + [used_by])[-10:]
        if confirmed:
            p["confirmed_by"] = f"executed as {used_by}"

    def _record(self, state: EvidenceState, rec: StepResult, inflight: bool = False) -> None:
        state.steps[rec.key] = rec.to_dict()
        state.inflight = rec.key if inflight else ""
        self.persist()

    def _save_artifact(self, state: EvidenceState, kind: str, label: str, text: str, description: str,
                       capability_id: str) -> Artifact | None:
        if not text:
            return None
        art = self.artifacts.save_text(kind, label, text, description, capability_id)
        if art is not None:
            state.artifacts.append(art.to_dict())  # metadata only; persisted with the next record
        return art

    def _save_file(self, state: EvidenceState, step: PlanStep, rp: RequirementPlan, cap: Any,
                   f: dict[str, Any]) -> Artifact | None:
        data = f.get("data")
        if not isinstance(data, (bytes, bytearray)) or not data:
            return None
        meta = dict(f.get("metadata") or {})
        meta.update(phase_id=state.phase_id, requirement_id=rp.requirement_id, step=step.key,
                    source_capability=cap.id, captured_at=meta.get("captured_at") or time.time())
        art = self.artifacts.save_bytes(str(f.get("kind") or "file"), str(f.get("label") or step.key), bytes(data),
                                        str(f.get("media_type") or "application/octet-stream"),
                                        str(f.get("description") or f"{cap.name} output of {step.key}"), cap.id, meta,
                                        ext=str(f.get("ext") or ""))
        if art is not None:
            state.artifacts.append(art.to_dict())
        return art


def _used_capabilities(plan: EvidencePlan, snapshot: CapabilitySnapshot) -> list[dict[str, Any]]:
    ids = list(dict.fromkeys(cid for r in plan.requirements for cid in [*r.resources, *(s.capability_id
                                                                                       for s in r.steps)]))
    return [{"id": c.id, "kind": c.kind, "name": c.name, "confidence": c.confidence, "source": c.source,
             "provider": c.provider, "provenance": {k: v for k, v in (c.provenance or {}).items()
                                                     if k in ("discovered_by", "declared_in", "changed_in_phase",
                                                              "executable", "confirmed_by", "provider_version")}}
            for c in (snapshot.get(cid) for cid in ids) if c is not None]
