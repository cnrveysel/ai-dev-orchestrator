"""Evidence models: capabilities, evidence plans and evidence results.

  capability  something AI Studio can genuinely do in this repository/environment. Its kind is
              semantic, open vocabulary. Its confidence is honest:
                CONFIRMED    actually executed / probed in this run
                INFERRED     declared or suggested, not proven
                UNAVAILABLE  explicitly checked and not usable (with the reason)
  plan        Claude's EVIDENCE_PLAN after validation: per contract requirement a strategy
              (execute | human | unavailable) whose steps reference capability ids only
  result      what execution actually showed, per requirement: SATISFIED | FAILED |
              UNAVAILABLE | INCONCLUSIVE | SKIPPED. A property counts as verified ONLY when an
              EvidenceResult says SATISFIED.

Everything here is compact JSON; full outputs are artifact files referenced by id.

V4 pass 3 adds, all additive (older records simply lack them):
  provenance  who discovered a capability, from what, when it was last validated and which steps
              used it; for every step: what it ran against (repository fingerprint, capability
              definition, file inputs)
  freshness   fresh | reused | stale | unknown per result. Evidence whose inputs changed after it
              was collected is STALE and never counts as SATISFIED (its original status is kept)
  cause       why a requirement is not satisfied: product (a FAILED check), strategy (the evidence
              strategy could not be executed - the only cause that may trigger a bounded replan),
              plan, budget, stale, mutation, timeout, unobserved (ran, exit 0, but the planned marker never
              appeared: FAILED, and a bounded replan may pick a capability that observes the property)
  reuse       a result is "reused" only when EVERY step that produced it was reused from an earlier attempt;
              a result with a step that ran now is "fresh" (its provenance lists the reused steps)
"""
from __future__ import annotations

import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

# --- capability confidence
CONFIRMED, INFERRED, UNAVAILABLE = "CONFIRMED", "INFERRED", "UNAVAILABLE"
CONFIDENCES = (CONFIRMED, INFERRED, UNAVAILABLE)

# --- evidence freshness (V4 pass 3)
FRESH, REUSED, STALE, UNKNOWN = "fresh", "reused", "stale", "unknown"
# --- why a requirement is not satisfied (V4 pass 3)
CAUSE_PRODUCT, CAUSE_STRATEGY, CAUSE_PLAN, CAUSE_BUDGET = "product", "strategy", "plan", "budget"
CAUSE_STALE, CAUSE_MUTATION, CAUSE_TIMEOUT, CAUSE_OPPORTUNITY = "stale", "mutation", "timeout", "opportunity"
# the check ran without error but never showed what the plan expected (exit 0, marker missing): the product may lack
# the property OR the capability may not report it - still FAILED; a bounded replan may try a capability that observes it
CAUSE_UNOBSERVED = "unobserved"

# --- plan strategies
EXECUTE, HUMAN, NOT_AUTOMATABLE = "execute", "human", "unavailable"
STRATEGIES = (EXECUTE, HUMAN, NOT_AUTOMATABLE)
DIRECT, INDIRECT = "direct", "indirect"

# --- step outcomes (one capability invocation)
PASSED, FAILED, TIMEOUT, ERROR, INTERRUPTED, SKIPPED, RUNNING = (
    "PASSED", "FAILED", "TIMEOUT", "ERROR", "INTERRUPTED", "SKIPPED", "RUNNING")
STEP_FINAL = frozenset({PASSED, FAILED, TIMEOUT, ERROR, UNAVAILABLE, SKIPPED})  # INTERRUPTED/RUNNING re-run

# --- requirement outcomes
SATISFIED, INCONCLUSIVE = "SATISFIED", "INCONCLUSIVE"
REQUIREMENT_STATUSES = (SATISFIED, FAILED, UNAVAILABLE, INCONCLUSIVE, SKIPPED)

# Capability kinds that never OBSERVE a runtime, visual or interaction property by themselves: their
# success can support another kind of requirement only indirectly (a green build is not a running app,
# and a captured screenshot is not a judgment of how it looks - that needs image_analysis or a person).
NON_OBSERVATIONAL_KINDS = frozenset({"build", "static_analysis", "file_inspection", "visual_capture"})

MAX_HISTORY = 5
MAX_REUSABLE = 40  # completed step records of earlier attempts kept for freshness-checked reuse


def _kw(cls, d: Any) -> dict[str, Any]:
    names = {f.name for f in fields(cls)}
    return {k: v for k, v in (d or {}).items() if k in names} if isinstance(d, dict) else {}


@dataclass
class Capability:
    id: str
    kind: str  # semantic, open vocabulary (build, automated_test, runtime_behavior, file_inspection, ...)
    name: str
    description: str = ""
    source: str = ""  # phase_verification | run_history | settings | repository | environment
    confidence: str = INFERRED
    provider: str = ""  # registry name of the provider that validates and executes it
    key: str = ""  # provider-defined identity, stable across phases of a run
    constraints: dict[str, Any] = field(default_factory=dict)  # timeout_seconds, lifecycle, accepts_args, ...
    details: dict[str, Any] = field(default_factory=dict)  # provider data (e.g. the declared command)
    reason: str = ""  # why it is CONFIRMED / INFERRED / UNAVAILABLE
    # V4 pass 3: discovered_by, provider_version, declared_in (+ sha1, changed_in_phase), executable,
    # discovered_at, validated_at, confirmed_by, used_by (step keys). Never secrets or environment dumps.
    provenance: dict[str, Any] = field(default_factory=dict)

    @property
    def usable(self) -> bool:
        return self.confidence in (CONFIRMED, INFERRED)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "Capability":
        kw = _kw(cls, d)
        kw["constraints"] = dict(kw.get("constraints") or {})
        kw["details"] = dict(kw.get("details") or {})
        kw["provenance"] = dict(kw.get("provenance") or {})
        if kw.get("confidence") not in CONFIDENCES:
            kw["confidence"] = INFERRED
        return cls(**{"id": "", "kind": "", "name": "", **kw})


@dataclass
class CapabilitySnapshot:
    capabilities: list[Capability] = field(default_factory=list)
    providers: dict[str, str] = field(default_factory=dict)  # provider -> ok (n) | timeout | error: ...
    truncated: int = 0  # capabilities dropped by the discovery limit
    environment: dict[str, str] = field(default_factory=dict)
    discovered_at: float = 0.0

    def get(self, cid: str) -> Capability | None:
        return next((c for c in self.capabilities if c.id == cid), None)

    def find_key(self, key: str) -> Capability | None:
        return next((c for c in self.capabilities if c.key == key), None)

    def counts(self) -> dict[str, int]:
        return {k: sum(1 for c in self.capabilities if c.confidence == k) for k in CONFIDENCES}

    def summary(self) -> str:
        n = self.counts()
        extra = f", {n[UNAVAILABLE]} unavailable" if n[UNAVAILABLE] else ""
        return (f"{len(self.capabilities)} capabilities found ({n[CONFIRMED]} confirmed, {n[INFERRED]} inferred{extra})"
                + (f"; {self.truncated} more not listed (limit)" if self.truncated else ""))

    def usable(self) -> list[Capability]:
        return [c for c in self.capabilities if c.usable]

    def to_dict(self) -> dict[str, Any]:
        return {"capabilities": [c.to_dict() for c in self.capabilities], "providers": dict(self.providers),
                "truncated": self.truncated, "environment": dict(self.environment),
                "discovered_at": self.discovered_at}

    @classmethod
    def from_dict(cls, d: Any) -> "CapabilitySnapshot | None":
        if not isinstance(d, dict) or "capabilities" not in d:
            return None
        return cls([Capability.from_dict(c) for c in d.get("capabilities") or [] if isinstance(c, dict)],
                   {str(k): str(v) for k, v in (d.get("providers") or {}).items()}, int(d.get("truncated") or 0),
                   {str(k): str(v) for k, v in (d.get("environment") or {}).items()},
                   float(d.get("discovered_at") or 0.0))


# ------------------------------------------------------------ plan

@dataclass
class PlanStep:
    key: str  # "<requirement id>.s<n>": stable identity for crash-safe resumption
    capability_id: str
    args: list[str] = field(default_factory=list)
    expect_output: str = ""  # literal text that must appear in the output for the step to pass
    fail_output: str = ""  # literal text whose presence fails the step
    purpose: str = ""

    @classmethod
    def from_dict(cls, d: Any) -> "PlanStep":
        kw = _kw(cls, d)
        kw["args"] = [str(a) for a in kw.get("args") or []]
        return cls(**{"key": "", "capability_id": "", **kw})


@dataclass
class RequirementPlan:
    requirement_id: str
    kind: str  # from the contract
    required: bool  # from the contract - never from the planner
    strategy: str  # execute | human | unavailable
    goal: str = ""
    coverage: str = DIRECT  # direct: the steps observe the property itself; indirect: supporting evidence only
    resources: list[str] = field(default_factory=list)  # lifecycle capability ids started before the steps
    steps: list[PlanStep] = field(default_factory=list)
    success_signal: str = ""
    failure_signal: str = ""
    reason: str = ""
    notes: list[str] = field(default_factory=list)  # validation notes (rejected steps, coverage downgrade)

    @classmethod
    def from_dict(cls, d: Any) -> "RequirementPlan":
        kw = _kw(cls, d)
        kw["steps"] = [PlanStep.from_dict(s) for s in kw.get("steps") or [] if isinstance(s, dict)]
        kw["resources"] = [str(x) for x in kw.get("resources") or []]
        kw["notes"] = [str(x) for x in kw.get("notes") or []]
        return cls(**{"requirement_id": "", "kind": "", "required": True, "strategy": NOT_AUTOMATABLE, **kw})


@dataclass
class EvidencePlan:
    requirements: list[RequirementPlan] = field(default_factory=list)
    summary: str = ""
    limitations: list[str] = field(default_factory=list)
    validation_errors: list[str] = field(default_factory=list)
    fingerprint: str = ""  # inputs the plan was made for; an unchanged fingerprint reuses the plan
    source: str = "claude"  # claude | deterministic
    created_at: float = 0.0
    generation: int = 0  # V4 pass 3: bounded replans merged into this plan (0 = the original plan)

    def get(self, rid: str) -> RequirementPlan | None:
        return next((r for r in self.requirements if r.requirement_id == rid), None)

    def step_count(self) -> int:
        return sum(len(r.steps) for r in self.requirements)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "EvidencePlan | None":
        if not isinstance(d, dict) or "requirements" not in d:
            return None
        kw = _kw(cls, d)
        kw["requirements"] = [RequirementPlan.from_dict(r) for r in kw.get("requirements") or [] if isinstance(r, dict)]
        kw["limitations"] = [str(x) for x in kw.get("limitations") or []]
        kw["validation_errors"] = [str(x) for x in kw.get("validation_errors") or []]
        return cls(**kw)


# ------------------------------------------------------------ results

@dataclass
class Artifact:
    id: str
    kind: str  # log | report | image | trace | benchmark | recording | ... (never assumed to be an image)
    path_or_ref: str
    description: str = ""
    created_by_capability: str = ""
    bytes: int = 0
    truncated: bool = False
    created_at: float = 0.0
    # V4 pass 3: "" (= text/plain, older records) | image/png | ...; images carry width, height, captured_at,
    # source capability, phase and requirement in metadata. An image artifact is what was captured - never a
    # judgment of what it shows.
    media_type: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "Artifact":
        kw = _kw(cls, d)
        kw["metadata"] = dict(kw.get("metadata") or {})
        return cls(**{"id": "", "kind": "", "path_or_ref": "", **kw})


@dataclass
class StepResult:
    key: str
    requirement_id: str
    capability_id: str
    status: str  # PASSED | FAILED | TIMEOUT | ERROR | UNAVAILABLE | SKIPPED | INTERRUPTED | RUNNING
    exit_code: int | None = None
    summary: str = ""
    output_tail: str = ""  # bounded; the full output is an artifact
    truncated: bool = False
    duration: float = 0.0
    started_at: float = 0.0
    finished_at: float = 0.0
    artifacts: list[str] = field(default_factory=list)
    tries: int = 0
    reason: str = ""
    # V4 pass 3: what the step ran against (signature, capability fingerprint, scope, repository fingerprint,
    # file inputs, created_at), whether it was reused from an earlier attempt, and structured provider details
    # (e.g. compact build diagnostics). Records without provenance are never reused across attempts.
    provenance: dict[str, Any] = field(default_factory=dict)
    reused_from: int = 0
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "StepResult":
        kw = _kw(cls, d)
        kw["artifacts"] = [str(x) for x in kw.get("artifacts") or []]
        kw["provenance"] = dict(kw.get("provenance") or {})
        kw["details"] = dict(kw.get("details") or {})
        return cls(**{"key": "", "requirement_id": "", "capability_id": "", "status": INTERRUPTED, **kw})


@dataclass
class EvidenceResult:
    requirement_id: str
    kind: str
    required: bool
    status: str  # SATISFIED | FAILED | UNAVAILABLE | INCONCLUSIVE | SKIPPED
    summary: str = ""
    goal: str = ""
    strategy: str = ""
    capability_ids: list[str] = field(default_factory=list)
    steps: list[str] = field(default_factory=list)
    artifacts: list[str] = field(default_factory=list)
    duration: float = 0.0
    failure_reason: str = ""
    confidence: str = "none"  # observed (direct evidence) | proxy (indirect only) | none
    source: str = "execution"  # execution | plan (nothing was executed)
    started_at: float = 0.0
    finished_at: float = 0.0
    # --- V4 pass 3 (additive) ---
    freshness: str = ""  # fresh | reused | stale | unknown ("" = recorded before freshness existed)
    stale_reason: str = ""
    stale_status: str = ""  # the status the result had before it went stale (it no longer counts)
    cause: str = ""  # product | strategy | plan | budget | stale | mutation | timeout | opportunity
    provenance: dict[str, Any] = field(default_factory=dict)  # repo_fp, scope, inputs, collected_at, reused_from
    diagnostics: list[str] = field(default_factory=list)  # compact provider diagnostics (e.g. first build errors)
    human_derived: bool = False  # an HF-* requirement derived from the user's verification
    failed_capability: str = ""  # cause strategy: the capability that could not be executed (never the others)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "EvidenceResult":
        kw = _kw(cls, d)
        for k in ("capability_ids", "steps", "artifacts", "diagnostics"):
            kw[k] = [str(x) for x in kw.get(k) or []]
        kw["provenance"] = dict(kw.get("provenance") or {})
        return cls(**{"requirement_id": "", "kind": "", "required": True, "status": INCONCLUSIVE, **kw})


@dataclass
class EvidenceBundle:
    phase_id: str
    attempt: int
    results: list[EvidenceResult] = field(default_factory=list)
    artifacts: list[Artifact] = field(default_factory=list)
    plan_summary: str = ""
    plan_source: str = ""
    limitations: list[str] = field(default_factory=list)
    capabilities: list[dict[str, Any]] = field(default_factory=list)  # compact: the ones the plan used
    capability_summary: str = ""
    notes: list[str] = field(default_factory=list)  # budget / cleanup / repository-mutation notes
    wall_used: float = 0.0
    steps_run: int = 0
    created_at: float = 0.0
    # --- V4 pass 3 (additive) ---
    repo_fp: str = ""  # repository fingerprint the evidence was collected against ("" = unknown)
    repo_fp_kind: str = ""  # git_tree | manifest | ""
    plan_fingerprint: str = ""
    plan_generation: int = 0
    guard: dict[str, Any] = field(default_factory=dict)  # repository-mutation protection that was actually active
    replans: list[dict[str, Any]] = field(default_factory=list)  # bounded replans of this collection

    def missing_required(self) -> list[EvidenceResult]:
        return [r for r in self.results if r.required and r.status != SATISFIED]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "EvidenceBundle | None":
        if not isinstance(d, dict) or "results" not in d:
            return None
        kw = _kw(cls, d)
        kw["results"] = [EvidenceResult.from_dict(r) for r in kw.get("results") or [] if isinstance(r, dict)]
        kw["artifacts"] = [Artifact.from_dict(a) for a in kw.get("artifacts") or [] if isinstance(a, dict)]
        for k in ("limitations", "notes"):
            kw[k] = [str(x) for x in kw.get(k) or []]
        kw["capabilities"] = [dict(x) for x in kw.get("capabilities") or [] if isinstance(x, dict)]
        kw["guard"] = dict(kw.get("guard") or {})
        kw["replans"] = [dict(x) for x in kw.get("replans") or [] if isinstance(x, dict)]
        return cls(**{"phase_id": "", "attempt": 0, **kw})


STATUS_MARK = {SATISFIED: "✓", FAILED: "✗", UNAVAILABLE: "?", INCONCLUSIVE: "~", SKIPPED: "-"}


def result_line(r: EvidenceResult | dict[str, Any], n: int = 240) -> str:
    """One compact line per requirement (logs, prompts, the human request)."""
    d = r.to_dict() if isinstance(r, EvidenceResult) else r
    text = " ".join(str(d.get("summary") or d.get("failure_reason") or "").split())
    if d.get("freshness") == STALE and d.get("stale_reason"):
        text = f"STALE ({d.get('stale_reason')}); was {d.get('stale_status') or '?'}: {text}".rstrip(": ")
    text = text if len(text) <= n else text[:n - 3] + "..."
    req = "" if d.get("required", True) else ", optional"
    tag = ""
    if d.get("freshness") == STALE:
        tag = " (stale)"
    elif d.get("freshness") == REUSED:
        rf = (d.get("provenance") or {}).get("reused_from")
        tag = f" (reused from attempt {rf})" if rf else " (reused)"
    elif d.get("freshness") == FRESH and (d.get("provenance") or {}).get("reused_steps"):
        steps = (d.get("provenance") or {}).get("reused_steps") or {}
        tag = f" (ran now; unchanged step(s) reused: {', '.join(sorted(steps)[:4])})"
    return f"{STATUS_MARK.get(d.get('status'), '')} {d.get('requirement_id')} [{d.get('kind')}{req}] {d.get('status')}" \
           + tag + (f" - {text}" if text else "")


def is_satisfied(r: EvidenceResult | dict[str, Any]) -> bool:
    """SATISFIED and not stale: the ONLY way a property counts as verified."""
    d = r.to_dict() if isinstance(r, EvidenceResult) else r
    return d.get("status") == SATISFIED and d.get("freshness") != STALE


def unresolved_required(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Required results that do not verify their requirement (not SATISFIED, or stale)."""
    return [r for r in results or [] if r.get("required", True) and not is_satisfied(r)]


def mark_stale(r: dict[str, Any], reason: str) -> bool:
    """Demote a result whose inputs changed after it was collected. Returns True if it changed. The
    external status vocabulary is kept: a stale result becomes INCONCLUSIVE (its old status is kept in
    stale_status), so no consumer can ever read it as SATISFIED."""
    if r.get("freshness") == STALE:
        return False
    if r.get("source") == "plan":  # nothing was executed: there is nothing to go stale
        return False
    r["stale_status"] = r.get("status")
    r["status"] = INCONCLUSIVE
    r["freshness"] = STALE
    r["stale_reason"] = reason[:300]
    r["cause"] = CAUSE_STALE
    r["confidence"] = "none"
    r["failure_reason"] = f"stale evidence: {reason}"[:300]
    return True


# ------------------------------------------------------------ persisted state (pipeline_state.evidence[phase])

@dataclass
class EvidenceState:
    """Evidence collection of ONE phase. Discovery is kept for the phase; plan, steps, resources and
    bundle belong to the phase attempt being verified (a new attempt archives the bundle)."""
    phase_id: str
    attempt: int = 0
    status: str = "new"  # new | planned | executing | complete
    discovery: dict[str, Any] = field(default_factory=dict)
    plan: dict[str, Any] = field(default_factory=dict)
    steps: dict[str, dict[str, Any]] = field(default_factory=dict)  # step key -> StepResult
    inflight: str = ""  # step key running right now (a crash leaves it here: never trusted)
    resources: list[dict[str, Any]] = field(default_factory=list)  # lifecycle resources (pid, start time, state)
    artifacts: list[dict[str, Any]] = field(default_factory=list)  # Artifact metadata only
    bundle: dict[str, Any] = field(default_factory=dict)
    history: list[dict[str, Any]] = field(default_factory=list)  # compact bundles of earlier attempts
    wall_used: float = 0.0
    started_at: float = 0.0
    # --- V4 pass 3 ---
    # Completed step records of earlier attempts (signature -> record + "attempt" + "artifact_meta"): a step
    # whose inputs are provably unchanged is reused instead of re-run; anything else runs again.
    reusable: dict[str, dict[str, Any]] = field(default_factory=dict)
    replans: int = 0  # bounded replans of THIS attempt's collection (persisted: never an endless loop)
    replan_log: list[dict[str, Any]] = field(default_factory=list)  # every replan of the phase (bounded)

    def begin_attempt(self, attempt: int) -> None:
        if self.bundle:
            b = dict(self.bundle)
            b["results"] = [{k: r.get(k) for k in ("requirement_id", "kind", "required", "status", "summary",
                                                    "freshness")}
                            for r in b.get("results") or []]
            self.history = (self.history + [{"attempt": self.attempt, "results": b["results"],
                                              "created_at": b.get("created_at")}])[-MAX_HISTORY:]
        arts = {a.get("id"): a for a in self.artifacts}
        for rec in self.steps.values():  # completed, provenance-stamped steps may be reused if still fresh
            sig = (rec.get("provenance") or {}).get("signature")
            if sig and rec.get("status") in (PASSED, FAILED):
                self.reusable[sig] = dict(rec, attempt=rec.get("reused_from") or self.attempt,
                                          artifact_meta=[arts[a] for a in rec.get("artifacts") or [] if a in arts])
        if len(self.reusable) > MAX_REUSABLE:
            newest = sorted(self.reusable.items(), key=lambda kv: float(kv[1].get("finished_at") or 0))
            self.reusable = dict(newest[-MAX_REUSABLE:])
        self.attempt, self.status = attempt, "new"
        self.steps, self.inflight, self.resources, self.artifacts, self.bundle = {}, "", [], [], {}
        self.wall_used, self.started_at, self.replans = 0.0, time.time(), 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "EvidenceState | None":
        if not isinstance(d, dict) or not d.get("phase_id"):
            return None
        kw = _kw(cls, d)
        for k in ("discovery", "plan", "bundle"):
            kw[k] = dict(kw.get(k) or {})
        kw["steps"] = {str(k): dict(v) for k, v in (kw.get("steps") or {}).items() if isinstance(v, dict)}
        kw["reusable"] = {str(k): dict(v) for k, v in (kw.get("reusable") or {}).items() if isinstance(v, dict)}
        for k in ("resources", "artifacts", "history", "replan_log"):
            kw[k] = [dict(x) for x in kw.get(k) or [] if isinstance(x, dict)]
        return cls(**kw)
