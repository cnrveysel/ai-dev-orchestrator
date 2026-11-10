"""Pipeline V2 repair models: the persisted repair state of a phase and Claude's
DIAGNOSE_BLOCKER answer.

The repair state lives in run.pipeline_state["repair"][<phase id>] so it survives a
restart together with the rest of the V2 run state. It holds only what can be resumed
safely (counters, fingerprints, snapshot ids, feedback); nothing process-local.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any

MAX_ROUNDS_KEPT = 20
MAX_DIAGNOSES_KEPT = 6

DIAGNOSIS_STATUSES = ("REPAIR", "BLOCKED")

_STR_LIST = {"type": "array", "items": {"type": "string"}}

DIAGNOSIS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "diagnosis": {"type": "string", "description": "The most likely root cause, grounded in the evidence."},
        "recommended_actions": _STR_LIST,
        "avoid": {**_STR_LIST, "description": "Approaches the worker already tried or must not take."},
        "target_files": _STR_LIST,
        "target_tests": {**_STR_LIST, "description": "Targeted commands/tests to run after each change."},
        "status": {"type": "string", "enum": list(DIAGNOSIS_STATUSES)},
        "blocked_reason": {"type": "string", "description": "Only with BLOCKED: what the user must decide or fix."},
        "authorized_test_changes": {
            **_STR_LIST,
            "description": "Only when an existing acceptance test itself is wrong: test files (or globs) the "
                           "repair may modify. Leave empty otherwise."},
    },
    "required": ["diagnosis", "recommended_actions", "avoid", "target_files", "target_tests", "status"],
}


def _strs(v: Any, limit: int = 20) -> list[str]:
    return [str(x) for x in v][:limit] if isinstance(v, list) else []


@dataclass
class Diagnosis:
    """Structured guidance from Claude for a stuck repair. Claude never edits files."""
    status: str  # REPAIR | BLOCKED
    diagnosis: str = ""
    recommended_actions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    target_tests: list[str] = field(default_factory=list)
    blocked_reason: str = ""
    authorized_test_changes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "Diagnosis":
        if not isinstance(d, dict):
            raise ValueError("diagnosis must be a JSON object")
        status = str(d.get("status", "")).strip().upper()
        if status not in DIAGNOSIS_STATUSES:
            raise ValueError(f"invalid diagnosis status {status!r}")
        return cls(status=status, diagnosis=str(d.get("diagnosis") or ""),
                   recommended_actions=_strs(d.get("recommended_actions")), avoid=_strs(d.get("avoid")),
                   target_files=_strs(d.get("target_files"), 40), target_tests=_strs(d.get("target_tests")),
                   blocked_reason=str(d.get("blocked_reason") or ""),
                   authorized_test_changes=_strs(d.get("authorized_test_changes"), 40))


@dataclass
class RepairGuardState:
    """Counters of the CURRENT repair session (a fresh session gets a fresh budget; the number
    of sessions per repair attempt is bounded by repair_max_diagnoses)."""
    wall_used: float = 0.0  # seconds of repair turns in this session
    tool_calls: int = 0
    same_rounds: int = 0  # consecutive rounds whose failure fingerprint did not change
    # Consecutive rounds WITHOUT an objective improvement (a score strictly better than the best-known
    # state). A merely different fingerprint (same or worse score) is not progress and counts here.
    no_improvement_rounds: int = 0
    worse_rounds: int = 0  # consecutive rounds worse than best-known (regression grace)
    restores: int = 0  # best-known restorations in this session
    probe_files: list[str] = field(default_factory=list)  # new probe files since the last progress
    trip: str = ""  # why the guard stopped the session ("" = not tripped)
    # Usage of the turn in flight, persisted at every tool call: a turn killed by a crash is still
    # counted (fold_inflight) instead of silently getting its budget back after a restart.
    inflight_tool_calls: int = 0
    inflight_wall: float = 0.0

    def fold_inflight(self) -> None:
        self.tool_calls += self.inflight_tool_calls
        self.wall_used += self.inflight_wall
        self.inflight_tool_calls, self.inflight_wall = 0, 0.0

    @classmethod
    def from_dict(cls, d: Any) -> "RepairGuardState":
        d = d if isinstance(d, dict) else {}
        return cls(wall_used=float(d.get("wall_used") or 0.0), tool_calls=int(d.get("tool_calls") or 0),
                   same_rounds=int(d.get("same_rounds") or 0),
                   no_improvement_rounds=int(d.get("no_improvement_rounds") or 0),
                   worse_rounds=int(d.get("worse_rounds") or 0),
                   restores=int(d.get("restores") or 0), probe_files=_strs(d.get("probe_files"), 200),
                   trip=str(d.get("trip") or ""), inflight_tool_calls=int(d.get("inflight_tool_calls") or 0),
                   inflight_wall=float(d.get("inflight_wall") or 0.0))


@dataclass
class RepairState:
    """Repair of ONE phase. Attempt-level fields are reset when a new checkpoint RETRY starts a
    new repair attempt; rounds and diagnoses are kept (bounded) as the phase's repair history."""
    phase_id: str
    checkpoint_attempt: int = 0  # Attempt.number whose checkpoint RETRY started this repair attempt
    repair_attempt: int = 0  # n-th repair attempt of this phase
    status: str = "working"  # working | blocked
    # --- checkpoint feedback that started the repair ---
    checkpoint: dict[str, Any] = field(default_factory=dict)  # summary, problems, corrective_instructions
    authorized: list[str] = field(default_factory=list)  # test paths/globs Claude allowed to change
    # --- repair session ---
    session_id: str = ""  # "" = the next round starts a NEW Qwen session
    session_no: int = 0  # repair sessions started in this attempt
    fresh_reason: str = ""
    guard: RepairGuardState = field(default_factory=RepairGuardState)
    # --- verification evidence ---
    round_no: int = 0
    rounds: list[dict[str, Any]] = field(default_factory=list)  # bounded history
    current: dict[str, Any] = field(default_factory=dict)  # evidence of the last verification
    best: dict[str, Any] = field(default_factory=dict)  # best-known evidence + "tree" + "round"
    base_tree: str = ""  # snapshot tree when this repair attempt began
    base_paths: list[str] = field(default_factory=list)  # changed/untracked paths at that point
    protected: dict[str, str] = field(default_factory=dict)  # acceptance file -> blob id (exact bytes)
    rejected_tests: list[str] = field(default_factory=list)  # protected files whose edits were reverted
    # --- escalation ---
    pending_escalation: str = ""  # guard reason waiting for a DIAGNOSE_BLOCKER call
    diagnoses: list[dict[str, Any]] = field(default_factory=list)
    diagnoses_this_attempt: int = 0
    # --- one-shot notes for the next repair prompt (regression, restore, rejected test edits) ---
    notes: list[str] = field(default_factory=list)
    restore_log: list[dict[str, Any]] = field(default_factory=list)

    @property
    def last_diagnosis(self) -> Diagnosis | None:
        if not self.diagnoses:
            return None
        try:
            return Diagnosis.from_dict(self.diagnoses[-1])
        except ValueError:
            return None

    def add_round(self, rec: dict[str, Any]) -> None:
        self.rounds = (self.rounds + [rec])[-MAX_ROUNDS_KEPT:]

    def add_diagnosis(self, rec: dict[str, Any]) -> None:
        self.diagnoses = (self.diagnoses + [rec])[-MAX_DIAGNOSES_KEPT:]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "RepairState":
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in (d or {}).items() if k in known}
        kw["guard"] = RepairGuardState.from_dict(kw.get("guard"))
        for k in ("rounds", "diagnoses", "restore_log"):
            kw[k] = [x for x in kw.get(k) or [] if isinstance(x, dict)]
        for k in ("authorized", "base_paths", "notes", "rejected_tests"):
            kw[k] = _strs(kw.get(k), 2000)
        kw["protected"] = {str(p): str(b) for p, b in (kw.get("protected") or {}).items()}
        for k in ("checkpoint", "current", "best"):
            kw[k] = dict(kw.get(k) or {})
        return cls(**kw)
