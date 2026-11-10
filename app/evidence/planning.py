"""EVIDENCE_PLAN: Claude decides how convincing evidence for a phase can be obtained with the
capabilities that ACTUALLY exist, and the orchestrator validates that plan before anything runs.

Validation (no model output is ever executed as-is):
  - one entry per contract requirement; unknown requirement ids are ignored; a requirement the
    planner skipped is kept as `unavailable` (never silently dropped)
  - `required` always comes from the contract; the planner cannot downgrade it
  - every step references a discovered, usable capability id; unknown / unavailable ids,
    free-form commands and arguments the capability does not declare are rejected
  - services are only started through `resources`; a step that observes a running resource
    (`requires_resource`) is only valid when the plan starts that resource for the requirement
  - a step that consumes an earlier step's artifact (e.g. analysing a captured image) may only
    reference an earlier step of the same requirement of the declared kind
  - the total number of steps is bounded
  - a non-observational capability (build, static analysis, file inspection, visual capture) can
    only give INDIRECT coverage of a different kind of requirement: a green build is not a running
    result, and a captured screenshot is not a judgment of how it looks

V4 pass 3: the requirements are the contract's plus the human-derived HF-* criteria; a bounded
REPLAN may re-plan only the requirements whose strategy could not be executed (never an ordinary
product failure), with new step keys so earlier results stay traceable. A requirement whose check ran
cleanly but never showed what the plan expected ("unobserved": exit 0, marker missing - e.g. a command
that does not print the result) may be re-planned too, with that capability excluded; unless the replan
finds an executable strategy the FAILED result stands.
"""
from __future__ import annotations

import hashlib
import json
import time
from dataclasses import dataclass, field
from typing import Any

from app.evidence.model import (
    CAUSE_STRATEGY, CAUSE_UNOBSERVED, CONFIRMED, DIRECT, EXECUTE, FAILED, HUMAN, INCONCLUSIVE, INDIRECT,
    NON_OBSERVATIONAL_KINDS, NOT_AUTOMATABLE, STRATEGIES, UNAVAILABLE, CapabilitySnapshot, EvidencePlan, PlanStep,
    RequirementPlan, result_line,
)
from app.evidence import readiness as rd
from app.models.attempt import VerificationResult
from app.models.human import criteria_lines
from app.models.phase import Phase
from app.models.verification import contract_lines
from app.orchestration.reviewer import _cut, _filter_stat, utf8_len

_STR_LIST = {"type": "array", "items": {"type": "string"}}
STEP_KEYS = frozenset({"capability_id", "args", "expect_output", "fail_output", "purpose"})
MAX_LITERAL = 200

EVIDENCE_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "requirements": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "requirement_id": {"type": "string", "description": "A requirement id of the contract (E1, ...)."},
                    "strategy": {"type": "string", "enum": list(STRATEGIES)},
                    "goal": {"type": "string", "description": "What the evidence must demonstrate."},
                    "coverage": {"type": "string", "enum": [DIRECT, INDIRECT],
                                 "description": "direct: the steps observe the required property itself; indirect: "
                                                "they only support it."},
                    "resources": {**_STR_LIST, "description": "Service capability ids to start before the steps "
                                                              "(stopped afterwards)."},
                    "steps": {
                        "type": "array",
                        "items": {
                            "type": "object", "additionalProperties": False,
                            "properties": {
                                "capability_id": {"type": "string"},
                                "args": {**_STR_LIST, "description": "Only for capabilities that accept arguments."},
                                "expect_output": {"type": "string", "description": "Literal text that must appear."},
                                "fail_output": {"type": "string", "description": "Literal text that means failure."},
                                "purpose": {"type": "string"},
                            },
                            "required": ["capability_id"],
                        },
                    },
                    "success_signal": {"type": "string"},
                    "failure_signal": {"type": "string"},
                    "reason": {"type": "string", "description": "Why this strategy (required for human / unavailable)."},
                },
                "required": ["requirement_id", "strategy", "goal", "reason"],
            },
        },
        "limitations": _STR_LIST,
    },
    "required": ["summary", "requirements"],
}


@dataclass
class EvidencePlanPacket:
    phase: Phase
    snapshot: CapabilitySnapshot
    goal: str = ""
    repo_path: str = ""
    verification: list[VerificationResult] = field(default_factory=list)  # the phase verification just run
    changed_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    prior: list[dict[str, Any]] = field(default_factory=list)  # earlier evidence of this phase (compact)
    human_gate_open: bool = False
    max_steps: int = 12
    step_timeout: int = 180
    wall_seconds: int = 600
    # V4 pass 3: a bounded REPLAN of the requirements whose strategy could not be executed:
    # {"generation": n, "requirements": [ids], "failures": [lines]}; {} = a normal plan
    replan: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class EvidencePlanLimits:
    max_prompt_bytes: int = 30000

    @classmethod
    def from_settings(cls, s: Any) -> "EvidencePlanLimits":
        return cls(max(6000, int(getattr(s, "evidence_plan_max_prompt_bytes", 30000))))


def capability_line(c: Any, snapshot: CapabilitySnapshot | None = None, step_timeout: int = 0) -> str:
    k = c.constraints or {}
    if k.get("lifecycle") == "service":
        ready = k.get("readiness") or {}
        how = "service: start it via `resources`" + (f"; ready when {rd.describe(ready)}" if ready else (
            f"; ready when its output shows {k['ready_output']!r}" if k.get("ready_output") else ""))
    elif k.get("accepts_args"):
        how = f"arguments: up to {k.get('max_args', 4)}" + (f" matching /{c.details.get('arg_pattern')}/"
                                                            if c.details.get("arg_pattern") else "")
        if k.get("args_help"):
            how += f" ({k['args_help']})"
    else:
        how = "no arguments"
    need = k.get("requires_resource")
    if need:
        res = snapshot.find_key(need) if snapshot is not None and need != "*" else None
        how += ("; observes a running resource: list any service capability under `resources`" if need == "*" else
                f"; observes a running resource: list {res.id if res else need} under `resources`")
    if k.get("input_step"):
        how += f"; its argument is the number of an earlier step of the same requirement of kind {k['input_step']}"
    if k.get("lifecycle") == "service":
        t = f"; must be ready within {k['ready_timeout_seconds']}s" if k.get("ready_timeout_seconds") else ""
    else:
        t = f"; timeout {k['timeout_seconds']}s" if k.get("timeout_seconds") else ""
        if step_timeout and int(k.get("timeout_seconds") or 0) > step_timeout:  # what it really gets here
            t = f"; runs at most {step_timeout}s here (the evidence step limit; it declares up to {k['timeout_seconds']}s)"
    trust = "; DECLARED IN A FILE THE WORKER CHANGED IN THIS PHASE" if (c.provenance or {}).get("changed_in_phase") \
        else ""
    return f"{c.id} [{c.kind}] {c.confidence} - {c.name}: {c.description} ({c.reason}; {how}{t}{trust})"


_RULES = """PLAN (structured answer)
- One entry per contract requirement (`requirement_id`), strategy:
  execute     steps that use the capability ids above ONLY
  human       only a person observing the running result can provide it
  unavailable no listed capability can provide it (say what is missing in `reason`)
- A step is {capability_id, args (only where the capability accepts arguments), expect_output / fail_output (optional
  literal text that marks success / failure in the output), purpose}. There is no other way to run anything: no other
  commands, programs, browsers, devices, servers or tools exist for this plan. Never invent a command or a path.
- `resources`: service capability ids to start before the steps; they are stopped afterwards. A capability that
  "observes a running resource" only works when that resource is listed in the same requirement's `resources`.
- `coverage`: direct only if the steps observe the required property itself; indirect if they merely make it
  plausible. A build, a static check or a file's content never directly demonstrates runtime, visual or interaction
  behaviour; a captured screenshot is not a judgment of how something looks (that needs an image analysis step or a
  person); a passing test demonstrates only what that test exercises.
- Keep every requirement, including the human-derived HF-* ones: never call required evidence unnecessary to make the
  plan easier. Prefer INFERRED capabilities only when no CONFIRMED one fits; say which uncertainty remains in
  `limitations`. A capability declared in a file the worker changed in this phase deserves extra scrutiny.
- A CONFIRMED command only ran; it does not show WHAT its output contains. Prefer a capability of the requirement's
  kind that reports the property itself (e.g. one that parses per-test results) over expecting a marker in the output
  of a generic command. A generated file (log, report, build output) proves something only if an evidence step of
  this plan writes it; a leftover one is not trusted.
- Evidence collection is verification only: nothing may edit code, fix failures or change tests."""


def requirement_lines(p: Phase) -> list[str]:
    """The requirements to plan: the contract's, then the human-derived ones (with their provenance)."""
    lines = [x for x in contract_lines(p.verification_contract) if not x.startswith(("human gate", "  human checklist"))
             and not x.startswith("success dimensions")]
    for c in p.human_criteria:
        ev = c.evidence or {}
        lines.append(f"{c.id} [{c.kind or 'unspecified'}, required, human-derived] {c.text} (from the user's "
                     f"{c.verdict or 'answer'} in attempt {c.attempt}"
                     + (f"; last evidence {ev.get('status')} in attempt {ev.get('attempt')}" if ev else "") + ")")
    return lines


def build_evidence_plan_prompt(pk: EvidencePlanPacket, limits: EvidencePlanLimits) -> tuple[str, dict[str, int]]:
    p, snap = pk.phase, pk.snapshot
    head = [f"EVIDENCE_PLAN - how can AI Studio obtain convincing evidence that phase {p.id} ({p.title}) succeeded?",
            "You plan VERIFICATION with the capabilities listed below (discovered in this repository/environment); "
            "this session has no tools.", "", "OBJECTIVE", _cut(p.description.strip() or "(none)", 1500),
            "", "ACCEPTANCE CRITERIA", *[f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)]]
    if p.human_criteria:
        head += ["", "HUMAN-DERIVED ACCEPTANCE CRITERIA (from the user's verification; plan evidence for them too)",
                 *[f"- {x}" for x in criteria_lines(p.human_criteria, with_status=True)]]
    head += ["", "VERIFICATION CONTRACT", *contract_lines(p.verification_contract)]
    wanted = set(pk.replan.get("requirements") or [])
    reqs = [x for x in requirement_lines(p) if not wanted or x.split(" ", 1)[0] in wanted]
    head += ["", "REQUIREMENTS TO PLAN (one entry each)", *reqs]
    if pk.replan:
        head += ["", f"REPLAN {pk.replan.get('generation', 1)} - the previous strategy for these requirements could not "
                     "be EXECUTED, or it ran but could not OBSERVE the property (exit 0 without the expected output: the "
                     "capability may simply not report it). Plan ONLY them again, with other capabilities - preferably "
                     "one that reports the property itself - or mark them human / unavailable; the capabilities that "
                     "failed are now listed as unavailable:",
                 *[f"- {_cut(str(x), 300)}" for x in pk.replan.get("failures") or []]]
    done = [f"{s.id} {s.title}" for s in p.subtasks if s.done]
    head += ["", "COMPLETED SUBTASKS: " + (", ".join(done) or "(none reported)")]
    usable = snap.usable()
    head += ["", "AVAILABLE CAPABILITIES (the ONLY things a step can use)"]
    head += [_cut(capability_line(c, snap, pk.step_timeout), 500) for c in usable] or ["(none)"]
    gone = [c for c in snap.capabilities if not c.usable]
    if gone:
        head += ["", "UNAVAILABLE (explicitly checked; cannot be planned)"]
        head += [_cut(f"{c.id} [{c.kind}] {c.name}: {c.reason}", 300) for c in gone[:20]]
    head += ["", "LIMITS", f"at most {pk.max_steps} step(s) in total, {pk.step_timeout}s per step, {pk.wall_seconds}s "
                           "overall; steps run sequentially; outputs are captured (bounded)",
             "a person can be asked to observe the result at the checkpoint" if pk.human_gate_open
             else "no person can be asked to verify this phase"]
    mandatory = "\n".join(head)

    def compose(budget: int) -> str:
        parts = [mandatory]
        if pk.verification:
            lines = ["PHASE VERIFICATION ALREADY RUN (by the orchestrator)"]
            for v in pk.verification[:10]:
                st = f"SKIPPED ({v.skipped_reason})" if v.skipped_reason else (
                    "PASS" if v.passed else ("TIMEOUT" if v.exit_code is None else f"FAIL (exit {v.exit_code})"))
                lines.append(f"- {v.command}: {st}")
            parts.append(_cut("\n".join(lines), budget))
        if pk.diff_stat or pk.changed_files:
            stat = _filter_stat(pk.diff_stat)
            parts.append(_cut("REPOSITORY CHANGES OF THIS PHASE\n" + ("\n".join(stat) if stat
                                                                     else ", ".join(pk.changed_files[:60])), budget))
        if pk.prior:
            lines = ["EARLIER EVIDENCE OF THIS PHASE (previous attempts)"]
            for b in pk.prior[-2:]:
                lines.append(f"attempt {b.get('attempt')}:")
                lines += [f"  {result_line(r, 160)}" for r in b.get("results") or []]
            parts.append(_cut("\n".join(lines), budget))
        parts.append(_RULES)
        return "\n\n".join(parts) + "\n"

    prompt, level = "", 0
    for level, budget in enumerate((4000, 1500, 0)):
        prompt = compose(budget)
        if utf8_len(prompt) <= limits.max_prompt_bytes:
            break
    else:
        prompt = _cut(mandatory, limits.max_prompt_bytes - utf8_len(_RULES) - 4) + "\n\n" + _RULES + "\n"
        level = 3
    # deterministic compaction levels: 0 = complete, 1-2 = optional sections shortened, 3 = mandatory part cut
    return prompt, {"prompt_bytes": utf8_len(prompt), "limit_bytes": limits.max_prompt_bytes, "compacted": level}


# ------------------------------------------------------------ validation

def plan_fingerprint(phase: Phase, snapshot: CapabilitySnapshot) -> str:
    """The inputs a plan depends on. Unchanged inputs (e.g. a repair of the same phase) reuse the plan."""
    c = phase.verification_contract
    data: dict[str, Any] = {"contract": c.to_dict() if c else None, "human": [h.text for h in phase.human_criteria],
                            "caps": sorted((x.id, x.key, x.kind, x.usable) for x in snapshot.capabilities)}
    kinds = [h.kind for h in phase.human_criteria]
    if any(kinds):  # only present when used, so fingerprints of earlier runs stay valid
        data["human_kinds"] = kinds
    return hashlib.sha1(json.dumps(data, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def _s(v: Any, n: int = 600) -> str:
    return str(v).strip()[:n] if v is not None else ""


def validate_plan(raw: Any, phase: Phase, snapshot: CapabilitySnapshot, registry: Any, max_steps: int, *,
                  only: list[str] | None = None, generation: int = 0) -> EvidencePlan:
    """Validate Claude's raw plan. `only` (a replan) restricts it to those requirement ids; `generation` > 0 gives
    the replanned steps new keys (<id>.r<n>.s<k>) so the earlier steps stay on record."""
    reqs = [r for r in phase.evidence_requirements() if only is None or r.id in only]
    errors: list[str] = []
    entries = raw.get("requirements") if isinstance(raw, dict) else None
    if not isinstance(entries, list):
        errors.append("the plan contained no requirement list")
        entries = []
    by_id: dict[str, dict[str, Any]] = {}
    known = {r.id for r in reqs}
    for e in entries:
        if not isinstance(e, dict):
            continue
        rid = _s(e.get("requirement_id"), 24)
        if rid not in known:
            scope = (" (only the contract and the human-derived criteria define requirements)" if only is None
                     else f" (this replan covers {', '.join(only)})")
            errors.append(f"unknown requirement {rid!r} ignored{scope}")
        elif rid in by_id:
            errors.append(f"duplicate entry for {rid} ignored")
        else:
            by_id[rid] = e
    budget = max(0, int(max_steps))
    used = 0
    out: list[RequirementPlan] = []
    for creq in reqs:
        e = by_id.get(creq.id)
        if e is None:
            errors.append(f"{creq.id} was not planned; kept as unavailable")
            out.append(RequirementPlan(creq.id, creq.kind, creq.required, NOT_AUTOMATABLE, creq.goal,
                                       reason="not planned by the evidence planner"))
            continue
        strategy = _s(e.get("strategy"), 20).lower()
        rp = RequirementPlan(creq.id, creq.kind, creq.required, strategy if strategy in STRATEGIES else NOT_AUTOMATABLE,
                             _s(e.get("goal")) or creq.goal, reason=_s(e.get("reason")),
                             success_signal=_s(e.get("success_signal"), 300),
                             failure_signal=_s(e.get("failure_signal"), 300))
        if strategy not in STRATEGIES:
            rp.notes.append(f"invalid strategy {strategy!r}; treated as unavailable")
        if rp.strategy == EXECUTE:
            used = _validate_execute(rp, e, creq, snapshot, registry, budget, used, generation)
        out.append(rp)
    summary = _s(raw.get("summary"), 1000) if isinstance(raw, dict) else ""
    lims = raw.get("limitations") if isinstance(raw, dict) else None
    limitations = [_s(x, 300) for x in (lims if isinstance(lims, list) else []) if _s(x)][:10]
    return EvidencePlan(out, summary, limitations, errors, created_at=time.time(), generation=generation)


def _step_key(rid: str, n: int, generation: int) -> str:
    return f"{rid}.s{n}" if not generation else f"{rid}.r{generation}.s{n}"


def _validate_execute(rp: RequirementPlan, e: dict[str, Any], creq: Any, snapshot: CapabilitySnapshot, registry: Any,
                      budget: int, used: int, generation: int = 0) -> int:
    rejected: list[str] = []
    for rid in [_s(x, 16) for x in e.get("resources") or [] if _s(x)][:4]:
        cap = snapshot.get(rid)
        prov = registry.provider(cap.provider) if cap is not None else None
        if cap is None:
            rejected.append(f"resource {rid}: unknown capability")
        elif not cap.usable:
            rejected.append(f"resource {rid}: unavailable ({cap.reason})")
        elif prov is None or not prov.supports_lifecycle(cap):
            rejected.append(f"resource {rid}: not a service capability")
        elif rid not in rp.resources:
            rp.resources.append(rid)
    budget_hit = False
    for i, s in enumerate(e.get("steps") or []):
        if not isinstance(s, dict):
            rejected.append(f"step {i + 1}: not an object")
            continue
        extra = sorted(set(s) - STEP_KEYS)
        cid = _s(s.get("capability_id"), 16)
        if extra:  # e.g. "command": a free-form command line is never executed
            rejected.append(f"step {i + 1}: unsupported field(s) {', '.join(extra)} - steps can only reference "
                            "capability ids")
            continue
        cap = snapshot.get(cid)
        if cap is None:
            rejected.append(f"step {i + 1}: unknown capability {cid!r}")
            continue
        if not cap.usable:
            rejected.append(f"step {i + 1}: {cid} is unavailable ({cap.reason})")
            continue
        prov = registry.provider(cap.provider)
        args = [str(a) for a in s.get("args") or []] if isinstance(s.get("args"), list) else []
        need = str(cap.constraints.get("requires_resource") or "")
        if need:  # the step observes a running resource: the plan must start it for this requirement
            started = [snapshot.get(x) for x in rp.resources]
            if need == "*" and not rp.resources:
                rejected.append(f"step {i + 1} ({cid}): observes a running resource; list the service under "
                                "`resources`")
                continue
            if need != "*" and not any(c is not None and c.key == need for c in started):
                res = snapshot.find_key(need)
                rejected.append(f"step {i + 1} ({cid}): observes {res.id if res else need}, which is not listed in "
                                "`resources`")
                continue
        consumes = str(cap.constraints.get("input_step") or "")
        if consumes:  # e.g. analysing the image an earlier step captured: args = [<number of that step>]
            ok = len(args) == 1 and args[0].isdigit() and 0 < int(args[0]) <= len(rp.steps)
            ref = rp.steps[int(args[0]) - 1] if ok else None
            ref_cap = snapshot.get(ref.capability_id) if ref is not None else None
            if ref_cap is None or ref_cap.kind != consumes:
                rejected.append(f"step {i + 1} ({cid}): its argument must be the number of an earlier step of this "
                                f"requirement that uses a {consumes} capability")
                continue
            args = [ref.key]
        step = PlanStep(_step_key(rp.requirement_id, len(rp.steps) + 1, generation), cid, args,
                        _s(s.get("expect_output"), MAX_LITERAL), _s(s.get("fail_output"), MAX_LITERAL),
                        _s(s.get("purpose"), 200))
        why = "no provider" if prov is None else prov.validate(cap, step)
        if why:
            rejected.append(f"step {i + 1} ({cid}): {why}")
            continue
        if used >= budget:
            budget_hit = True
            rp.notes.append(f"step {i + 1} ({cid}) dropped: the evidence step budget ({budget}) is used up")
            continue
        rp.steps.append(step)
        used += 1
    rp.notes += rejected
    if not rp.steps:
        rp.strategy = NOT_AUTOMATABLE
        rp.reason = ("the evidence step budget was used up by earlier requirements" if budget_hit else
                     "no valid step: " + "; ".join(rejected[:3]) if rejected else "the plan contained no step") \
            + (f" (planner: {rp.reason})" if rp.reason else "")
        return used
    coverage = _s(e.get("coverage"), 10).lower()
    rp.coverage = coverage if coverage in (DIRECT, INDIRECT) else INDIRECT
    kinds = {snapshot.get(s.capability_id).kind for s in rp.steps}
    weak = sorted(k for k in kinds if k in NON_OBSERVATIONAL_KINDS and k != creq.kind)
    if rp.coverage == DIRECT and weak and len(weak) == len(kinds):
        rp.coverage = INDIRECT
        rp.notes.append(f"coverage set to indirect: {', '.join(weak)} does not observe {creq.kind}")
    return used


def deterministic_plan(phase: Phase, reason: str) -> EvidencePlan:
    """No usable capability at all: every requirement is unavailable, without a Claude call."""
    reqs = [RequirementPlan(r.id, r.kind, r.required, NOT_AUTOMATABLE, r.goal, reason=reason)
            for r in phase.evidence_requirements()]
    return EvidencePlan(reqs, f"no automated evidence possible: {reason}", [reason], source="deterministic",
                        created_at=time.time())


def plan_counts(plan: EvidencePlan) -> str:
    n = {s: sum(1 for r in plan.requirements if r.strategy == s) for s in STRATEGIES}
    parts = [f"{n[EXECUTE]} execute"] + ([f"{n[HUMAN]} human"] if n[HUMAN] else []) + \
            ([f"{n[NOT_AUTOMATABLE]} unavailable"] if n[NOT_AUTOMATABLE] else [])
    return ", ".join(parts)


# ------------------------------------------------------------ bounded replanning (V4 pass 3)

def strategy_failures(bundle: Any, plan: EvidencePlan, snapshot: CapabilitySnapshot,
                      confirmed_before: set[str]) -> list[dict[str, Any]]:
    """Requirements whose evidence STRATEGY could not be executed - a capability that turned out unavailable,
    a resource that could not start or never became ready, a provider error - plus required requirements the
    plan left unavailable although a capability of their kind got CONFIRMED by execution in this collection.
    An ordinary FAILED check (a product defect) is never a strategy failure; a FAILED check that ran cleanly but
    never showed the expected output (cause unobserved) is ambiguous and may be re-planned once."""
    out: list[dict[str, Any]] = []
    newly = {c.kind for c in snapshot.capabilities if c.confidence == CONFIRMED and c.key not in confirmed_before}
    for r in bundle.results:
        rp = plan.get(r.requirement_id)
        if rp is None:
            continue
        if r.cause == CAUSE_STRATEGY and r.status in (UNAVAILABLE, INCONCLUSIVE):
            out.append({"requirement_id": r.requirement_id,
                        "capabilities": [r.failed_capability] if r.failed_capability else [],
                        "reason": (r.failure_reason or r.summary)[:300], "kind": "strategy"})
        elif r.cause == CAUSE_UNOBSERVED and r.status == FAILED and r.failed_capability:
            out.append({"requirement_id": r.requirement_id, "capabilities": [r.failed_capability],
                        "reason": ("ran (exit 0) but did not show the expected output: "
                                   + (r.summary or r.failure_reason))[:300], "kind": "unobserved"})
        elif rp.strategy == NOT_AUTOMATABLE and r.kind in newly and r.required:
            out.append({"requirement_id": r.requirement_id, "capabilities": [],
                        "reason": f"a {r.kind} capability was confirmed by execution after the plan was made",
                        "kind": "opportunity"})
    return out


def demote_failed(snapshot: CapabilitySnapshot, failures: list[dict[str, Any]]) -> list[str]:
    """Capabilities whose execution failed for strategy reasons become UNAVAILABLE (for this collection), so a
    replan cannot pick them again. Returns the demoted capability ids."""
    demoted = []
    for f in failures:
        for cid in f.get("capabilities") or []:
            c = snapshot.get(cid)
            if c is not None and c.usable:
                c.confidence = UNAVAILABLE
                c.reason = f"failed during evidence collection: {f.get('reason', '')}"[:300]
                demoted.append(cid)
    return demoted


def merge_replan(plan: EvidencePlan, new: EvidencePlan, rids: list[str],
                 keep_unless_executable: set[str] | frozenset[str] = frozenset()) -> EvidencePlan:
    """The plan with the replanned requirements replaced (same order); everything else unchanged. A requirement in
    `keep_unless_executable` (its check FAILED as unobserved) keeps its old strategy - and so its FAILED result -
    unless the replan found steps to execute: a replan never turns a failed check into "no evidence"."""
    by = {r.requirement_id: r for r in new.requirements}
    merged = []
    for r in plan.requirements:
        n = by.get(r.requirement_id) if r.requirement_id in rids else None
        if n is not None and r.requirement_id in keep_unless_executable and not (n.strategy == EXECUTE and n.steps):
            n = None
        merged.append(n or r)
    return EvidencePlan(merged, plan.summary + (f" | replan {new.generation}: {new.summary}" if new.summary else ""),
                        list(dict.fromkeys(plan.limitations + new.limitations))[:10],
                        plan.validation_errors + [f"replan {new.generation}: {x}" for x in new.validation_errors],
                        plan.fingerprint, plan.source, plan.created_at, new.generation)


