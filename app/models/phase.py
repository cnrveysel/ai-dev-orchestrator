"""Pipeline V2 phase model.

A phase is one coherent engineering objective executed by ONE Qwen session and
reviewed by Claude only at its boundary (checkpoint). Its internal subtasks are
the worker's own breakdown and are never reviewed individually.

Phases are stored in the same `tasks` table as V1 tasks (a Phase is a Task with
extra fields), but they are only ever parsed as phases for runs whose
pipeline_version is "v2" - V1 task rows are never reinterpreted.
"""
from __future__ import annotations

import re
import time
from dataclasses import dataclass, field, fields
from typing import Any

from app.models.human import HumanCriterion
from app.models.plan import TASK_ID_RE, Plan, PlanValidationError
from app.models.task import Task, TaskResult, TaskStatus
from app.models.verification import VERIFICATION_CONTRACT_SCHEMA, EvidenceRequirement, VerificationContract

MAX_PHASES = 8
MAX_SUBTASKS = 12

# Phases that are closed for good: never re-run, never modified by a replan.
TERMINAL_PHASE_STATUSES = frozenset({TaskStatus.PASSED, TaskStatus.SUPERSEDED})

_STR_LIST = {"type": "array", "items": {"type": "string"}}

_PHASE_ITEM_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "title": {"type": "string"},
        "description": {"type": "string", "description": "The coherent engineering objective of the phase."},
        "instructions": {"type": "string"},
        "subtasks": {
            "type": "array",
            "minItems": 1,
            "maxItems": MAX_SUBTASKS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                },
                "required": ["id", "title"],
            },
        },
        "acceptance_criteria": _STR_LIST,
        "files_likely_involved": _STR_LIST,
        "verification_commands": _STR_LIST,
        "integration_expectations": _STR_LIST,
        "estimated_minutes": {"type": "integer"},
        "verification_contract": VERIFICATION_CONTRACT_SCHEMA,  # V4 (optional)
    },
    "required": ["id", "title", "description", "subtasks", "acceptance_criteria"],
}

PHASE_PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "summary": {"type": "string"},
        "assumptions": _STR_LIST,
        "architecture_notes": _STR_LIST,
        "phases": {"type": "array", "minItems": 1, "maxItems": MAX_PHASES, "items": _PHASE_ITEM_SCHEMA},
    },
    "required": ["goal", "summary", "phases"],
}

PHASE_REPLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "summary": {"type": "string"},
        "architecture_notes": _STR_LIST,
        "phases": {"type": "array", "minItems": 1, "maxItems": MAX_PHASES, "items": _PHASE_ITEM_SCHEMA},
    },
    "required": ["summary", "phases"],
}


@dataclass
class Subtask:
    id: str
    title: str
    description: str = ""
    done: bool = False  # set only by a structured completion signal (worker todo list) or a checkpoint PASS
    completed_at: float = 0.0  # when the completion signal arrived
    completed_via: str = ""  # "todo" (structured worker event) | "checkpoint" (accepted with the phase)

    @classmethod
    def from_any(cls, raw: Any, index: int) -> "Subtask":
        if isinstance(raw, str):
            return cls(id=f"S{index + 1}", title=raw.strip())
        if not isinstance(raw, dict):
            raise PlanValidationError(f"subtask #{index + 1} must be an object or a string")
        try:
            completed_at = float(raw.get("completed_at") or 0.0)
        except (TypeError, ValueError):
            completed_at = 0.0
        return cls(id=str(raw.get("id") or f"S{index + 1}").strip(), title=str(raw.get("title") or "").strip(),
                   description=str(raw.get("description") or "").strip(), done=bool(raw.get("done", False)),
                   completed_at=completed_at, completed_via=str(raw.get("completed_via") or ""))


@dataclass
class Phase(Task):
    kind: str = "phase"
    subtasks: list[Subtask] = field(default_factory=list)
    integration_expectations: list[str] = field(default_factory=list)
    estimated_minutes: int = 0
    # --- Qwen session (persisted after every turn and as soon as the session id is known) ---
    session_id: str = ""
    session_turns: int = 0  # turns sent to the current session
    sessions_started: int = 0
    fresh_session_requested: bool = False  # set by the user; honoured at the next turn
    # --- progress / checkpoint bookkeeping ---
    todo: list[dict[str, Any]] = field(default_factory=list)  # worker's live todo list [{text, completed}]
    reported_status: str = ""  # COMPLETE | INCOMPLETE | BLOCKED from the worker's last final message
    last_checkpoint: str = ""  # PASS | RETRY | BLOCKED | REPLAN | ASK_USER
    superseded_reason: str = ""
    # --- V4 verification semantics (None / empty = the phase behaves exactly as in V2/V3) ---
    verification_contract: VerificationContract | None = None  # WHAT evidence shows success (from the plan)
    # Criteria derived from human feedback (HF-<phase>-NN): separate from acceptance_criteria, binding for
    # every later repair and checkpoint of this phase.
    human_criteria: list[HumanCriterion] = field(default_factory=list)

    @property
    def subtasks_done(self) -> int:
        return sum(1 for s in self.subtasks if s.done)

    def evidence_requirements(self) -> list[EvidenceRequirement]:
        """V4 pass 3: what evidence collection plans for - the contract's requirements plus one required
        requirement per human-derived criterion (HF-*). The contract and the acceptance criteria themselves
        are never modified; the two stay separate and traceable."""
        reqs = list(self.verification_contract.evidence_requirements) if self.verification_contract else []
        for c in self.human_criteria:
            reqs.append(EvidenceRequirement(
                c.id, c.text, c.kind or "unspecified", True,
                f"derived from the user's verification ({c.verdict or 'answer'}) of attempt {c.attempt}"
                + (f"; {c.rationale}" if c.rationale else "")))
        return reqs

    @property
    def todo_done(self) -> int:
        return sum(1 for t in self.todo if t.get("completed"))

    def apply_todo(self, todo: list[dict[str, Any]] | None) -> list[str]:
        """Mark planned subtasks done from the worker's structured todo list. Only ever marks
        done: an unchecked or missing entry (e.g. a new session's fresh list) never undoes
        progress. Returns the ids that became done."""
        done = todo_done_ids(todo, [s.id for s in self.subtasks])
        newly = [s.id for s in self.subtasks if s.id in done and not s.done]
        now = time.time()
        for s in self.subtasks:
            if s.id in newly:
                s.done, s.completed_at, s.completed_via = True, now, "todo"
        return newly

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Phase":
        base = Task.from_dict(d)
        kw = {f.name: getattr(base, f.name) for f in fields(Task)}
        subtasks = [Subtask.from_any(s, i) for i, s in enumerate(d.get("subtasks") or [])]
        todo = [{"text": str(t.get("text", "")), "completed": bool(t.get("completed"))}
                for t in (d.get("todo") or []) if isinstance(t, dict)]
        try:
            minutes = int(d.get("estimated_minutes") or 0)
        except (TypeError, ValueError):
            minutes = 0
        return cls(
            **kw,
            subtasks=subtasks,
            integration_expectations=[str(x) for x in d.get("integration_expectations", []) or []],
            estimated_minutes=minutes,
            session_id=str(d.get("session_id") or ""),
            session_turns=int(d.get("session_turns", 0) or 0),
            sessions_started=int(d.get("sessions_started", 0) or 0),
            fresh_session_requested=bool(d.get("fresh_session_requested", False)),
            todo=todo,
            reported_status=str(d.get("reported_status") or ""),
            last_checkpoint=str(d.get("last_checkpoint") or ""),
            superseded_reason=str(d.get("superseded_reason") or ""),
            verification_contract=VerificationContract.from_any(d.get("verification_contract")),
            human_criteria=[c for c in (HumanCriterion.from_any(x) for x in d.get("human_criteria") or []) if c],
        )


# ------------------------------------------------------------ plan parsing

def parse_phases(raw_phases: Any, *, start_index: int = 0) -> list[Phase]:
    """Validate Claude's phase list (plan or replan). Statuses are reset to PENDING."""
    if not isinstance(raw_phases, list) or not raw_phases:
        raise PlanValidationError("plan contains no phases")
    if len(raw_phases) > MAX_PHASES:
        raise PlanValidationError(f"plan has {len(raw_phases)} phases; at most {MAX_PHASES} coherent phases "
                                  "are allowed (merge closely related work)")
    phases: list[Phase] = []
    seen: set[str] = set()
    for i, raw in enumerate(raw_phases):
        if not isinstance(raw, dict):
            raise PlanValidationError(f"phase #{i + 1} must be an object")
        raw = dict(raw)
        raw["id"] = str(raw.get("id") or f"P{start_index + i + 1}").strip()
        raw["status"] = TaskStatus.PENDING.value
        raw["dependencies"] = []  # phases run strictly in plan order
        for k in ("attempts", "extra_attempts", "started_at", "finished_at", "result", "session_id",
                  "session_turns", "sessions_started", "todo", "reported_status", "last_checkpoint",
                  "superseded_reason", "user_guidance", "fresh_session_requested", "human_criteria"):
            raw.pop(k, None)
        try:
            phase = Phase.from_dict(raw)
        except PlanValidationError:
            raise
        except (ValueError, KeyError, TypeError) as exc:
            raise PlanValidationError(f"phase #{i + 1} invalid: {exc}") from exc
        if not TASK_ID_RE.match(phase.id):
            raise PlanValidationError(f"invalid phase id {phase.id!r}")
        if phase.id in seen:
            raise PlanValidationError(f"duplicate phase id {phase.id}")
        seen.add(phase.id)
        if not phase.title.strip():
            raise PlanValidationError(f"phase {phase.id} has no title")
        if not phase.acceptance_criteria:
            raise PlanValidationError(f"phase {phase.id} has no acceptance criteria")
        if not phase.subtasks:
            raise PlanValidationError(f"phase {phase.id} has no subtasks")
        if len(phase.subtasks) > MAX_SUBTASKS:
            raise PlanValidationError(f"phase {phase.id} has {len(phase.subtasks)} subtasks (max {MAX_SUBTASKS}); "
                                      "split it into two phases")
        sub_ids: set[str] = set()
        for j, s in enumerate(phase.subtasks):
            if not s.title:
                raise PlanValidationError(f"{phase.id} subtask #{j + 1} has no title")
            if not s.id or s.id in sub_ids:
                s.id = f"S{j + 1}"
            s.done, s.completed_at, s.completed_via = False, 0.0, ""
            sub_ids.add(s.id)
        phase.position = start_index + i
        phases.append(phase)
    return phases


def parse_phase_plan(d: dict[str, Any]) -> Plan:
    """Parse and validate a V2 phase plan from Claude."""
    if not isinstance(d, dict):
        raise PlanValidationError("plan must be a JSON object")
    phases = parse_phases(d.get("phases"))
    return Plan(
        goal=str(d.get("goal", "")),
        summary=str(d.get("summary", "")),
        assumptions=[str(x) for x in d.get("assumptions", []) or []],
        architecture_notes=[str(x) for x in d.get("architecture_notes", []) or []],
        tasks=list(phases),
    )


def unique_phase_id(base: str, used: set[str]) -> str:
    base = base[:28] or "P"
    for k in range(1, 1000):
        cand = f"{base}r{k}"
        if cand not in used and TASK_ID_RE.match(cand):
            return cand
    raise PlanValidationError(f"cannot derive a unique id from {base!r}")


def merge_phase_replan(old: list[Phase], new: list[Phase]) -> list[Phase]:
    """Completed (PASSED) and SUPERSEDED phases are history: kept unchanged, in place.
    Every other phase is replaced by the revised list. A returned phase identical to a
    PASSED phase (same id and title) is Claude echoing history and is dropped; any other
    id collision is renamed so history is never overwritten and no new work is lost."""
    kept = [p for p in sorted(old, key=lambda p: p.position) if p.status in TERMINAL_PHASE_STATUSES]
    passed = {p.id: p for p in kept if p.status == TaskStatus.PASSED}
    used = {p.id for p in kept}
    merged: list[Phase] = list(kept)
    for n in new:
        echo = passed.get(n.id)
        if echo is not None and echo.title.strip().lower() == n.title.strip().lower():
            continue
        if n.id in used:
            n.id = unique_phase_id(n.id, used)
        n.status = TaskStatus.PENDING
        n.attempts = n.extra_attempts = 0
        n.started_at = n.finished_at = None
        n.result = None
        used.add(n.id)
        merged.append(n)
    for i, p in enumerate(merged):
        p.position = i
    return merged


def phase_result_line(p: Phase | Task) -> str:
    r: TaskResult | None = p.result
    tag = " [SUPERSEDED]" if p.status == TaskStatus.SUPERSEDED else ""
    return f"{p.id} {p.title}{tag}: {r.result if r else '(no result recorded)'}"


# ------------------------------------------------------------ worker report parsing

_STATUS_RE = re.compile(r"^[\s>#*_-]*\**\s*phase\s+status\s*\**\s*:\s*\**\s*(complete|incomplete|blocked)\b",
                        re.IGNORECASE | re.MULTILINE)
_SUBTASK_RE = re.compile(r"^[\s>*-]*\[(x|X|✓|✔| |-|~|/)\]\s*\**\s*([A-Za-z0-9_.-]{1,32})\b", re.MULTILINE)
_TODO_LEAD_RE = re.compile(r"^[\s\[(*#>•-]*")  # bullets, brackets and bold/heading markers before the id
# What may follow the id: whitespace or a separator; "." "-" "/" only when not continuing the token (S1.2, S1/S2).
_TODO_BOUNDARY_RE = re.compile(r"\s|[:)\]}>*,;|–—]|[./-](?![A-Za-z0-9])")


def todo_subtask_id(text: str, subtask_ids: list[str]) -> str:
    """The planned subtask a worker todo entry belongs to, or "". The entry must START with
    the subtask id as a whole token ("S1: ...", "S1 - ...", "[S1] ...", "**S1** ..."); an id
    mentioned later ("Verify S1"), a longer token ("S10", "S1.2", "S1a") or a combined
    entry ("S1/S2") never matches."""
    s = _TODO_LEAD_RE.sub("", str(text or ""), count=1)
    for sid in subtask_ids:
        if sid and s[:len(sid)].upper() == sid.upper():
            rest = s[len(sid):]
            if not rest or _TODO_BOUNDARY_RE.match(rest):
                return sid
    return ""


def todo_done_ids(todo: list[dict[str, Any]] | None, subtask_ids: list[str]) -> set[str]:
    """Planned subtasks whose todo entries are all completed. Entries without a planned id
    (exploring, verifying, cleanup) are the worker's own and never count."""
    state: dict[str, bool] = {}
    for item in todo or []:
        sid = todo_subtask_id(str(item.get("text", "")), subtask_ids)
        if sid:
            state[sid] = state.get(sid, True) and bool(item.get("completed"))
    return {sid for sid, completed in state.items() if completed}


@dataclass
class PhaseReport:
    status: str  # COMPLETE | INCOMPLETE | BLOCKED | "" (not reported)
    done: set[str] = field(default_factory=set)
    open: set[str] = field(default_factory=set)
    source: str = ""  # "report" | "todo" | "default"

    @property
    def effective_status(self) -> str:
        return self.status or "COMPLETE"


def parse_phase_report(final_message: str, subtask_ids: list[str],
                       todo: list[dict[str, Any]] | None = None) -> PhaseReport:
    """Status and subtask progress from the worker's final message. Falls back to its live
    todo list when the message does not state PHASE STATUS."""
    text = final_message or ""
    m = _STATUS_RE.search(text)
    status = m.group(1).upper() if m else ""
    known = {s.upper(): s for s in subtask_ids}
    done: set[str] = set()
    opened: set[str] = set()
    for mark, sid in _SUBTASK_RE.findall(text):
        real = known.get(sid.upper())
        if real is None:
            continue
        (done if mark in "xX✓✔" else opened).add(real)
    done |= todo_done_ids(todo, subtask_ids) - opened  # todo entries are prefixed with the subtask id
    source = "report" if status else ""
    if not status and todo:
        status = "INCOMPLETE" if any(not t.get("completed") for t in todo) else "COMPLETE"
        source = "todo"
    if status == "COMPLETE" and not done and not opened:
        done = set(subtask_ids)  # a plain "complete" report covers every subtask
    return PhaseReport(status, done, opened - done, source or "default")
