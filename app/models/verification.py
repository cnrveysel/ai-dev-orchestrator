"""Pipeline V4 verification semantics: what evidence convincingly shows that a phase succeeded.

A VerificationContract describes WHAT evidence a phase needs - never HOW a tool obtains it.
Evidence kinds and success dimensions are semantic, project-independent vocabulary that Claude
chooses per phase when it plans; the lists below are suggestions, not an enum. Concrete
evidence executors are not part of this module.

Checkpoint issues may carry severity / disposition metadata. It is descriptive only: the
orchestrator never derives a decision from it - Claude decides from the phase's actual
requirements and evidence.

Phases without a contract behave exactly as before (no human gate, no evidence assessment).
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

# Suggested semantic vocabulary (Claude may use other snake_case values).
EVIDENCE_KINDS = ("static_analysis", "build", "automated_test", "runtime_behavior", "visual_evidence", "interaction",
                  "performance", "data_integrity", "integration", "documentation", "human_judgment")
SUCCESS_DIMENSIONS = ("functional", "runtime", "visual", "interaction", "performance", "integration", "robustness",
                      "usability", "content")

ISSUE_SEVERITIES = ("critical", "functional", "regression", "polish", "subjective", "micro_deviation")
ISSUE_DISPOSITIONS = ("repair", "ask_user", "accept", "blocked")
EVIDENCE_STATUSES = ("satisfied", "unsatisfied", "insufficient", "not_applicable")

MAX_DIMENSIONS = 10
MAX_REQUIREMENTS = 12
MAX_CHECKLIST = 12
MAX_ISSUES = 20

_STR_LIST = {"type": "array", "items": {"type": "string"}}

VERIFICATION_CONTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "description": "WHAT evidence would convincingly show that this phase succeeded (not how a tool obtains it).",
    "properties": {
        "success_dimensions": {**_STR_LIST, "description": "Semantic dimensions of success, e.g. "
                                                           + ", ".join(SUCCESS_DIMENSIONS)},
        "evidence_requirements": {
            "type": "array", "maxItems": MAX_REQUIREMENTS,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string", "description": "E1, E2, ..."},
                    "goal": {"type": "string", "description": "What the evidence must demonstrate."},
                    "kind": {"type": "string", "description": "Semantic evidence kind, e.g. " + ", ".join(EVIDENCE_KINDS)},
                    "required": {"type": "boolean"},
                    "reason": {"type": "string", "description": "Why this evidence is needed (e.g. why static tests are "
                                                                "insufficient)."},
                },
                "required": ["goal", "kind", "required"],
            },
        },
        "human_gate": {
            "type": "object",
            "description": "Whether the checkpoint may ask the user to verify something only a person can judge.",
            "properties": {
                "allowed": {"type": "boolean"},
                "reason": {"type": "string"},
                "checklist": {**_STR_LIST, "description": "Optional things a person would check in the running result."},
            },
            "required": ["allowed"],
        },
    },
    "required": ["success_dimensions", "evidence_requirements", "human_gate"],
}

ISSUES_SCHEMA: dict[str, Any] = {
    "type": "array", "maxItems": MAX_ISSUES,
    "description": "Optional metadata for each finding (descriptive only; it does not decide anything by itself).",
    "items": {
        "type": "object",
        "properties": {
            "description": {"type": "string"},
            "severity": {"type": "string", "enum": list(ISSUE_SEVERITIES)},
            "disposition": {"type": "string", "enum": list(ISSUE_DISPOSITIONS)},
            "evidence_id": {"type": "string", "description": "Related evidence requirement id, if any."},
        },
        "required": ["description", "severity", "disposition"],
    },
}

EVIDENCE_ASSESSMENT_SCHEMA: dict[str, Any] = {
    "type": "array", "maxItems": MAX_REQUIREMENTS,
    "description": "Only when the phase has a verification contract: one entry per evidence requirement.",
    "items": {
        "type": "object",
        "properties": {
            "id": {"type": "string"},
            "status": {"type": "string", "enum": list(EVIDENCE_STATUSES)},
            "note": {"type": "string"},
        },
        "required": ["id", "status"],
    },
}


def _token(v: Any, limit: int = 40) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(v or "").strip().lower()).strip("_")[:limit]


def _text(v: Any, limit: int = 600) -> str:
    return str(v or "").strip()[:limit] if v is not None else ""


def _texts(v: Any, n: int, limit: int = 400) -> list[str]:
    out: list[str] = []
    for x in v if isinstance(v, list) else []:
        t = _text(x, limit)
        if t and t not in out:
            out.append(t)
    return out[:n]


def _bool(v: Any, default: bool) -> bool:
    if isinstance(v, bool):
        return v
    if isinstance(v, str) and v.strip():
        return v.strip().lower() not in ("false", "no", "0", "off")
    return default if v is None else bool(v)


@dataclass
class EvidenceRequirement:
    """One evidence requirement. Contract requirements have ids E1, E2, ...; requirements derived from
    the user's verification (V4 pass 3) are HF-<phase>-NN and are never part of the contract itself."""
    id: str
    goal: str
    kind: str = "unspecified"
    required: bool = True
    reason: str = ""

    @property
    def human_derived(self) -> bool:
        return self.id.upper().startswith("HF-")


@dataclass
class HumanGate:
    allowed: bool = False
    reason: str = ""
    checklist: list[str] = field(default_factory=list)


@dataclass
class VerificationContract:
    success_dimensions: list[str] = field(default_factory=list)
    evidence_requirements: list[EvidenceRequirement] = field(default_factory=list)
    human_gate: HumanGate = field(default_factory=HumanGate)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_any(cls, raw: Any) -> "VerificationContract | None":
        """Tolerant parse of Claude's (or a persisted) contract; None when absent or empty.
        Optional metadata never fails a plan: malformed entries are dropped."""
        if isinstance(raw, VerificationContract):
            return raw
        if not isinstance(raw, dict):
            return None
        dims: list[str] = []
        raw_dims = raw.get("success_dimensions")
        for d in raw_dims if isinstance(raw_dims, list) else []:
            t = _token(d)
            if t and t not in dims:
                dims.append(t)
        reqs: list[EvidenceRequirement] = []
        ids: set[str] = set()
        raw_reqs = raw.get("evidence_requirements")
        for r in (raw_reqs if isinstance(raw_reqs, list) else [])[:MAX_REQUIREMENTS]:
            if isinstance(r, str):
                r = {"goal": r}
            if not isinstance(r, dict):
                continue
            goal = _text(r.get("goal") or r.get("description"))
            if not goal:
                continue
            rid = _text(r.get("id"), 16)
            if not rid or rid in ids or rid.upper().startswith("HF-"):  # HF-* is reserved for human-derived criteria
                rid = next(f"E{k}" for k in range(len(reqs) + 1, 1000) if f"E{k}" not in ids)
            ids.add(rid)
            reqs.append(EvidenceRequirement(rid, goal, _token(r.get("kind")) or "unspecified",
                                            _bool(r.get("required"), True), _text(r.get("reason"))))
        g = raw.get("human_gate")
        gate = (HumanGate(_bool(g.get("allowed"), False), _text(g.get("reason")), _texts(g.get("checklist"), MAX_CHECKLIST))
                if isinstance(g, dict) else HumanGate())
        if not dims and not reqs and not gate.allowed and not gate.reason and not gate.checklist:
            return None
        return cls(dims[:MAX_DIMENSIONS], reqs, gate)


def contract_lines(c: VerificationContract | None) -> list[str]:
    """Compact rendering for Claude prompts."""
    if c is None:
        return []
    lines: list[str] = []
    if c.success_dimensions:
        lines.append("success dimensions: " + ", ".join(c.success_dimensions))
    for r in c.evidence_requirements:
        lines.append(f"{r.id} [{r.kind}, {'required' if r.required else 'optional'}] {r.goal}"
                     + (f" (why: {r.reason})" if r.reason else ""))
    g = c.human_gate
    lines.append("human gate: " + ("allowed" if g.allowed else "not allowed") + (f" - {g.reason}" if g.reason else ""))
    lines += [f"  human checklist: {x}" for x in g.checklist]
    return lines


def normalize_issue(raw: Any) -> dict[str, str] | None:
    if isinstance(raw, str):
        raw = {"description": raw}
    if not isinstance(raw, dict):
        return None
    desc = _text(raw.get("description") or raw.get("issue") or raw.get("summary"))
    if not desc:
        return None
    sev, disp = _token(raw.get("severity")), _token(raw.get("disposition"))
    return {"description": desc, "severity": sev if sev in ISSUE_SEVERITIES else "",
            "disposition": disp if disp in ISSUE_DISPOSITIONS else "", "evidence_id": _text(raw.get("evidence_id"), 16)}


def normalize_issues(v: Any) -> list[dict[str, str]]:
    out = [normalize_issue(x) for x in (v if isinstance(v, list) else [])]
    return [x for x in out if x][:MAX_ISSUES]


def normalize_assessment(v: Any) -> list[dict[str, str]]:
    out: list[dict[str, str]] = []
    for x in v if isinstance(v, list) else []:
        if not isinstance(x, dict) or not _text(x.get("id"), 16):
            continue
        st = _token(x.get("status"))
        out.append({"id": _text(x.get("id"), 16), "status": st if st in EVIDENCE_STATUSES else "",
                    "note": _text(x.get("note"), 400)})
    return out[:MAX_REQUIREMENTS]


def issue_line(i: dict[str, Any]) -> str:
    tags = "/".join(x for x in (i.get("severity"), i.get("disposition")) if x)
    return (f"[{tags}] " if tags else "") + str(i.get("description", "")) + \
        (f" ({i['evidence_id']})" if i.get("evidence_id") else "")
