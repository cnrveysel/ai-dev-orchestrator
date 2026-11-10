"""Claude review decision model (V1 task reviews, V2 phase checkpoints, V4 human follow-ups)."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Iterable

from app.models.verification import (
    EVIDENCE_ASSESSMENT_SCHEMA, ISSUES_SCHEMA, normalize_assessment, normalize_issues,
)


class ReviewDecision(str, Enum):
    PASS = "PASS"
    RETRY = "RETRY"
    BLOCKED = "BLOCKED"
    REPLAN = "REPLAN"  # V2 checkpoints only; the V1 review schema never offers it
    # V4 checkpoints only, and only when the phase's human gate is open: the run waits for the
    # user (AWAITING_HUMAN). Never offered by the V1 review or the HUMAN_FOLLOWUP schema.
    ASK_USER = "ASK_USER"


CHECKPOINT_DECISIONS = ("PASS", "RETRY", "BLOCKED", "REPLAN")
FOLLOWUP_DECISIONS = ("PASS", "RETRY", "BLOCKED")  # no ASK_USER recursion


REVIEW_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["PASS", "RETRY", "BLOCKED"]},
        "summary": {"type": "string"},
        "problems": {"type": "array", "items": {"type": "string"}},
        "corrective_instructions": {"type": "array", "items": {"type": "string"}},
        "verification_notes": {"type": "array", "items": {"type": "string"}},
        "plan_changes_required": {"type": "boolean"},
        "task_result": {
            "type": "object",
            "description": "Compact result for future tasks. Fill when decision is PASS.",
            "properties": {
                "result": {"type": "string"},
                "important_decisions": {"type": "array", "items": {"type": "string"}},
                "remaining_risks": {"type": "array", "items": {"type": "string"}},
            },
        },
    },
    "required": ["decision", "summary", "problems", "corrective_instructions",
                 "verification_notes", "plan_changes_required"],
}

_STR_LIST = {"type": "array", "items": {"type": "string"}}

# V2: Claude reviews a whole phase at its boundary (and the final phase as the final audit).
CHECKPOINT_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": ["PASS", "RETRY", "BLOCKED", "REPLAN"]},
        "summary": {"type": "string"},
        "problems": _STR_LIST,
        "corrective_instructions": _STR_LIST,
        "verification_notes": _STR_LIST,
        "integration_notes": _STR_LIST,
        "regressions": _STR_LIST,
        "remaining_risks": _STR_LIST,
        "plan_changes_required": {"type": "boolean"},
        "replan_reason": {"type": "string"},
        "authorized_test_changes": {
            "type": "array", "items": {"type": "string"},
            "description": "Only with RETRY, and only when an existing acceptance test itself is wrong: the "
                           "repository-relative test files (or globs) the repair may modify. Leave empty otherwise: "
                           "existing acceptance tests are protected during repair."},
        "phase_result": {
            "type": "object",
            "description": "Compact record of this phase for later phases. Fill when decision is PASS.",
            "properties": {
                "result": {"type": "string"},
                "important_decisions": _STR_LIST,
                "remaining_risks": _STR_LIST,
            },
        },
        # V4 (optional, descriptive metadata)
        "issues": ISSUES_SCHEMA,
        "evidence_assessment": EVIDENCE_ASSESSMENT_SCHEMA,
        # V4 pass 2: only when EVIDENCE RESULTS were provided
        "evidence_sufficient": {"type": "boolean",
                                "description": "Only with EVIDENCE RESULTS: whether the evidence suffices to accept "
                                               "the phase (green builds/tests alone do not prove runtime, visual or "
                                               "interaction properties)."},
        "missing_evidence": {**_STR_LIST, "description": "Required evidence that is still missing or unconvincing."},
    },
    "required": ["decision", "summary", "problems", "corrective_instructions",
                 "verification_notes", "plan_changes_required"],
}

# V4: the ASK_USER payload. Offered only when the phase's human gate is open.
HUMAN_REQUEST_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "Only with ASK_USER: what the user must observe in the running result of THIS phase.",
    "properties": {
        "reason": {"type": "string", "description": "Why automated evidence cannot settle acceptance."},
        "questions": {**_STR_LIST, "minItems": 1, "maxItems": 6,
                      "description": "Short, specific questions a person answers by OBSERVING the running result, each "
                                     "about one unresolved criterion or evidence gap (never 'does everything work?', "
                                     "never about what the evidence already proved)."},
        "question_targets": {**_STR_LIST, "description": "For each question (same order): the requirement id it "
                                                         "resolves (E1, HF-P2-01, ...), or \"\" if none."},
        "observations": {**_STR_LIST, "description": "What the automated evidence already established."},
        "checklist": _STR_LIST,
    },
    "required": ["reason", "questions", "observations"],
}


def checkpoint_schema(allow_ask_user: bool) -> dict[str, Any]:
    """The CHECKPOINT schema; ASK_USER (and its payload) only when the phase's human gate is open."""
    if not allow_ask_user:
        return CHECKPOINT_JSON_SCHEMA
    s = copy.deepcopy(CHECKPOINT_JSON_SCHEMA)
    s["properties"]["decision"]["enum"] = [*CHECKPOINT_DECISIONS, "ASK_USER"]
    s["properties"]["human_request"] = HUMAN_REQUEST_SCHEMA
    return s


# V4: Claude adjudicates the user's answer to an ASK_USER request. PASS / RETRY / BLOCKED only.
HUMAN_FOLLOWUP_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(FOLLOWUP_DECISIONS)},
        "summary": {"type": "string"},
        "problems": _STR_LIST,
        "corrective_instructions": {
            **_STR_LIST, "description": "Only with RETRY: sent verbatim to a fresh, bounded repair session."},
        "derived_criteria": {
            "type": "array", "maxItems": 6,
            "description": "Phase-scoped acceptance criteria derived from the user's feedback (only requirements the "
                           "feedback revealed that the existing criteria do not already state). Each becomes an "
                           "evidence requirement (HF-<phase>-NN) of the phase's later attempts.",
            "items": {"type": "object", "properties": {
                "text": {"type": "string", "description": "A checkable, observable criterion."},
                "rationale": {"type": "string"},
                "kind": {"type": "string", "description": "Semantic evidence kind that would demonstrate it, e.g. "
                                                          "runtime_behavior, visual_evidence, interaction, performance."}},
                "required": ["text"]},
        },
        "do_not_redo": {**_STR_LIST, "description": "Work confirmed working that a repair must not redo."},
        "verification_notes": _STR_LIST,
        "remaining_risks": _STR_LIST,
        "issues": ISSUES_SCHEMA,
        "authorized_test_changes": _STR_LIST,
        "plan_changes_required": {"type": "boolean"},
        "replan_reason": {"type": "string"},
        "phase_result": {
            "type": "object",
            "description": "Fill when decision is PASS: the compact record of this phase for later phases.",
            "properties": {"result": {"type": "string"}, "important_decisions": _STR_LIST,
                           "remaining_risks": _STR_LIST},
        },
    },
    "required": ["decision", "summary", "problems", "corrective_instructions", "derived_criteria", "do_not_redo"],
}


def _strs(v: Any) -> list[str]:
    return [str(x) for x in v or []] if isinstance(v, list) else []


def _human_request(v: Any) -> dict[str, Any]:
    if not isinstance(v, dict):
        return {}
    out = {"reason": str(v.get("reason") or ""), "questions": [q for q in _strs(v.get("questions")) if q.strip()],
           "observations": _strs(v.get("observations")), "checklist": _strs(v.get("checklist"))}
    if isinstance(v.get("question_targets"), list):
        out["question_targets"] = [str(x or "") for x in v["question_targets"]]
    for k in ("phase_id", "phase_title", "attempt", "evidence", "created_at", "missing",  # a persisted request
              "established", "targets", "dropped"):
        if k in v:
            out[k] = v[k]
    return out


def _criteria(v: Any) -> list[dict[str, Any]]:
    out = []
    for x in v if isinstance(v, list) else []:
        d = x if isinstance(x, dict) else {"text": x}
        if str(d.get("text") or "").strip():
            out.append({k: d[k] for k in d if k in ("id", "text", "rationale", "verdict", "note", "attempt", "kind")})
    return out


@dataclass
class Review:
    decision: ReviewDecision
    summary: str = ""
    problems: list[str] = field(default_factory=list)
    corrective_instructions: list[str] = field(default_factory=list)
    verification_notes: list[str] = field(default_factory=list)
    plan_changes_required: bool = False
    task_result: dict[str, Any] = field(default_factory=dict)
    # --- V2 checkpoint fields (empty for V1 reviews) ---
    kind: str = "REVIEW"  # REVIEW | CHECKPOINT | FINAL_AUDIT | HUMAN_FOLLOWUP
    integration_notes: list[str] = field(default_factory=list)
    regressions: list[str] = field(default_factory=list)
    remaining_risks: list[str] = field(default_factory=list)
    replan_reason: str = ""
    authorized_test_changes: list[str] = field(default_factory=list)  # V2 repair: tests Claude allows to change
    # --- V4 (empty unless used) ---
    issues: list[dict[str, Any]] = field(default_factory=list)  # {description, severity, disposition, evidence_id}
    evidence_assessment: list[dict[str, Any]] = field(default_factory=list)  # {id, status, note} per contract item
    # ASK_USER: Claude's request. HUMAN_FOLLOWUP: the persisted request that was answered.
    human_request: dict[str, Any] = field(default_factory=dict)
    human_feedback: dict[str, Any] = field(default_factory=dict)  # HUMAN_FOLLOWUP: the user's structured answer
    derived_criteria: list[dict[str, Any]] = field(default_factory=list)  # HUMAN_FOLLOWUP: {id, text}
    do_not_redo: list[str] = field(default_factory=list)  # HUMAN_FOLLOWUP: confirmed-working work
    # --- V4 pass 2: Claude's judgment of the evidence bundle (None = no evidence was provided) ---
    evidence_sufficient: bool | None = None
    missing_evidence: list[str] = field(default_factory=list)
    # --- V4 pass 3: CONSISTENCY_REVIEW adjudication of an internally inconsistent checkpoint (kind
    # CONSISTENCY_REVIEW): trigger, the original checkpoint (compact), justification, dispositions ---
    consistency: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["decision"] = self.decision.value
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any], allowed: Iterable[str] | None = None) -> "Review":
        """`allowed` restricts the decision for a FRESH answer from Claude (a checkpoint without an open
        human gate, a HUMAN_FOLLOWUP); persisted reviews are parsed without it."""
        if not isinstance(d, dict):
            raise ValueError("review must be a JSON object")
        decision = str(d.get("decision", "")).strip().upper()
        if decision not in ReviewDecision.__members__:
            raise ValueError(f"invalid review decision {decision!r}")
        if allowed is not None and decision not in set(allowed):
            raise ValueError(f"decision {decision} is not allowed here (expected {', '.join(allowed)})")
        human_request = _human_request(d.get("human_request"))
        if allowed is not None and decision == "ASK_USER" and not human_request.get("questions"):
            raise ValueError("ASK_USER without any question for the user")
        result = d.get("task_result") or d.get("phase_result") or {}
        return cls(
            decision=ReviewDecision(decision),
            summary=str(d.get("summary", "")),
            problems=_strs(d.get("problems")),
            corrective_instructions=_strs(d.get("corrective_instructions")),
            verification_notes=_strs(d.get("verification_notes")),
            plan_changes_required=bool(d.get("plan_changes_required", False)),
            task_result=dict(result) if isinstance(result, dict) else {},
            kind=str(d.get("kind") or "REVIEW"),
            integration_notes=_strs(d.get("integration_notes")),
            regressions=_strs(d.get("regressions")),
            remaining_risks=_strs(d.get("remaining_risks")),
            replan_reason=str(d.get("replan_reason") or ""),
            authorized_test_changes=_strs(d.get("authorized_test_changes")),
            issues=normalize_issues(d.get("issues")),
            evidence_assessment=normalize_assessment(d.get("evidence_assessment")),
            human_request=human_request,
            human_feedback=dict(d.get("human_feedback") or {}) if isinstance(d.get("human_feedback"), dict) else {},
            derived_criteria=_criteria(d.get("derived_criteria")),
            do_not_redo=_strs(d.get("do_not_redo")),
            evidence_sufficient=d["evidence_sufficient"] if isinstance(d.get("evidence_sufficient"), bool) else None,
            missing_evidence=_strs(d.get("missing_evidence")),
            consistency=dict(d.get("consistency") or {}) if isinstance(d.get("consistency"), dict) else {},
        )
