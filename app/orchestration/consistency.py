"""V4 pass 3 CONSISTENCY_REVIEW: a bounded, tool-free Claude adjudication of a checkpoint whose structured
answer contradicts itself in a way that matters for acceptance.

Triggers (only these; Claude remains the decision-maker - nothing here decides the outcome):
  - decision PASS, evidence_sufficient == false, and required evidence is unresolved
    (not SATISFIED, or stale)
  - decision PASS although a required evidence requirement FAILED (a demonstrated defect)

The review sees only a compact packet: the phase objective, acceptance criteria, human-derived
criteria, the verification contract, every required evidence status, missing_evidence, the original
checkpoint verdict and reason, human evidence (the user's answers) and remaining risks. It may answer
PASS / RETRY / BLOCKED (and ASK_USER when the phase's human gate is open). A PASS must justify EVERY
unresolved required requirement (not needed for acceptance, or named substitute evidence suffices);
an unjustified PASS is rejected as an invalid answer. Its outcome is final: a consistency review is
never itself reviewed again (no recursion). The requirements it accepts stay recorded as UNVERIFIED.
"""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

from app.evidence.model import FAILED, result_line, unresolved_required
from app.models.human import criteria_lines
from app.models.phase import Phase
from app.models.review import CHECKPOINT_JSON_SCHEMA, HUMAN_REQUEST_SCHEMA, Review, ReviewDecision
from app.models.verification import contract_lines
from app.orchestration.checkpoint import human_history_section
from app.orchestration.reviewer import _cut, utf8_len

CONSISTENCY_DECISIONS = ("PASS", "RETRY", "BLOCKED")
DISPOSITIONS = ("not_needed", "substitute_evidence", "still_required")
_STR_LIST = {"type": "array", "items": {"type": "string"}}

CONSISTENCY_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": list(CONSISTENCY_DECISIONS)},
        "summary": {"type": "string"},
        "justification": {"type": "string",
                          "description": "Required with PASS: why the unresolved required evidence is not necessary "
                                         "for accepting this phase, or which available substitute evidence suffices."},
        "evidence_dispositions": {
            "type": "array",
            "description": "One entry per unresolved required requirement.",
            "items": {"type": "object", "properties": {
                "requirement_id": {"type": "string"},
                "disposition": {"type": "string", "enum": list(DISPOSITIONS)},
                "reason": {"type": "string"}},
                "required": ["requirement_id", "disposition", "reason"]}},
        "problems": _STR_LIST,
        "corrective_instructions": {**_STR_LIST, "description": "Only with RETRY: sent verbatim to a fresh, bounded "
                                                                "repair session."},
        "remaining_risks": _STR_LIST,
        "authorized_test_changes": _STR_LIST,
        "phase_result": CHECKPOINT_JSON_SCHEMA["properties"]["phase_result"],
    },
    "required": ["decision", "summary", "justification", "evidence_dispositions", "problems",
                 "corrective_instructions"],
}


def consistency_schema(allow_ask_user: bool) -> dict[str, Any]:
    if not allow_ask_user:
        return CONSISTENCY_JSON_SCHEMA
    s = copy.deepcopy(CONSISTENCY_JSON_SCHEMA)
    s["properties"]["decision"]["enum"] = [*CONSISTENCY_DECISIONS, "ASK_USER"]
    s["properties"]["human_request"] = HUMAN_REQUEST_SCHEMA
    return s


def find_conflict(review: Review, evidence: dict[str, Any] | None) -> dict[str, Any] | None:
    """The meaningful internal inconsistency of a checkpoint answer, or None (then nothing changes)."""
    if review.decision != ReviewDecision.PASS or not evidence:
        return None
    unresolved = unresolved_required(evidence.get("results") or [])
    if not unresolved:
        return None
    ids = [str(r.get("requirement_id")) for r in unresolved]
    lines = [f"{r.get('requirement_id')} [{r.get('kind')}] {r.get('status')}"
             + (" (stale)" if r.get("freshness") == "stale" else "") for r in unresolved]
    failed = [r for r in unresolved if r.get("status") == FAILED]
    if review.evidence_sufficient is False:
        return {"kind": "insufficient_evidence", "unresolved": ids,
                "reason": f"PASS while evidence_sufficient=false and {len(ids)} required evidence requirement(s) are "
                          f"unresolved: {', '.join(lines)}"}
    if failed:
        return {"kind": "failed_evidence", "unresolved": ids,
                "reason": "PASS although required evidence FAILED: " + ", ".join(
                    f"{r.get('requirement_id')} [{r.get('kind')}]" for r in failed)}
    return None


@dataclass
class ConsistencyPacket:
    phase: Phase
    checkpoint: dict[str, Any]  # the original checkpoint (Review.to_dict())
    evidence: dict[str, Any]  # the evidence bundle the checkpoint judged (freshness re-checked)
    trigger: dict[str, Any]
    human_rounds: list[dict[str, Any]] = field(default_factory=list)
    human_gate_open: bool = False
    final: bool = False
    goal: str = ""
    repo_path: str = ""
    user_guidance: str = ""


@dataclass(frozen=True)
class ConsistencyLimits:
    max_prompt_bytes: int = 24000

    @classmethod
    def from_settings(cls, s: Any) -> "ConsistencyLimits":
        return cls(max(6000, int(getattr(s, "consistency_max_prompt_bytes", 24000))))


_DECIDE = """DECIDE (structured answer)
- PASS only if the unresolved required evidence is genuinely not necessary to accept THIS phase, or available
  substitute evidence (name it) is sufficient. Then give `justification` and an `evidence_dispositions` entry for
  EVERY unresolved required requirement: not_needed or substitute_evidence, with the reason. A PASS that leaves any
  of them unjustified is rejected. Accepted requirements stay recorded as UNVERIFIED - never as verified.
- RETRY if the worker can fix what FAILED evidence shows, or can make the missing evidence obtainable (e.g. a runnable
  check or an observable report): `corrective_instructions` go verbatim to a fresh, bounded repair session.
- BLOCKED if acceptance needs a user decision, missing information or an environment change (say what in `summary`).
- Mark requirements you do not waive as still_required. Your answer is final for this checkpoint."""

_ASK = """- ASK_USER if a person observing the running result can close the gap: ask ONLY about the unresolved
  requirements (short, specific, observable questions; `question_targets` = their requirement ids) and list what the
  evidence already established in `observations`."""


def build_consistency_prompt(pk: ConsistencyPacket, limits: ConsistencyLimits) -> tuple[str, dict[str, int]]:
    p, ck = pk.phase, pk.checkpoint or {}
    results = list((pk.evidence or {}).get("results") or [])
    results.sort(key=lambda r: (0 if r.get("required", True) else 1, str(r.get("requirement_id"))))
    head = [f"CONSISTENCY_REVIEW - your checkpoint of phase {p.id} ({p.title}) is internally inconsistent; resolve it"
            + (" (last phase of the run: final audit)" if pk.final else "") + ".",
            "This session has no tools. The orchestrator does not decide for you: it only asks you to make the "
            "decision consistent with the evidence.",
            "", "THE INCONSISTENCY", pk.trigger.get("reason", ""),
            "", "OBJECTIVE", _cut(p.description.strip() or "(none)", 1200),
            "", "ACCEPTANCE CRITERIA", *[f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)]]
    if p.human_criteria:
        head += ["", "HUMAN-DERIVED ACCEPTANCE CRITERIA (from the user's verification; equally binding)",
                 *[f"- {x}" for x in criteria_lines(p.human_criteria, with_status=True)]]
    if p.verification_contract is not None:
        head += ["", "VERIFICATION CONTRACT", *[_cut(x, 400) for x in contract_lines(p.verification_contract)]]
    head += ["", "EVIDENCE STATUS (what actually ran; only SATISFIED and fresh verifies a requirement)"]
    head += [result_line(r, 240) for r in results] or ["(no evidence results)"]
    head += ["UNRESOLVED REQUIRED EVIDENCE: " + ", ".join(pk.trigger.get("unresolved") or [])]
    head += ["", "YOUR CHECKPOINT", f"decision: {ck.get('decision')}; evidence_sufficient: "
             f"{str(ck.get('evidence_sufficient')).lower()}", f"summary: {_cut(str(ck.get('summary', '')), 1000)}"]
    head += [f"missing_evidence: {_cut(str(x), 300)}" for x in (ck.get("missing_evidence") or [])[:8]]
    head += [f"remaining risk: {_cut(str(x), 300)}" for x in (ck.get("remaining_risks") or [])[:8]]
    if pk.user_guidance.strip():
        head += ["", "USER GUIDANCE", _cut(pk.user_guidance.strip(), 800)]
    mandatory = "\n".join(head)
    decide = _DECIDE + ("\n" + _ASK if pk.human_gate_open else "\n- Human verification is not available: decide "
                                                                 "from the evidence.")

    def compose(opt: int) -> str:
        parts = [mandatory]
        notes = [f"- verification note: {_cut(str(x), 300)}" for x in (ck.get("verification_notes") or [])[:6]]
        notes += [f"- your assessment {a.get('id')}: {a.get('status')}" + (f" - {_cut(str(a.get('note')), 200)}"
                                                                          if a.get("note") else "")
                  for a in (ck.get("evidence_assessment") or [])[:10]]
        notes += [f"- problem: {_cut(str(x), 300)}" for x in (ck.get("problems") or [])[:6]]
        if notes and opt:
            parts.append(_cut("CHECKPOINT DETAILS\n" + "\n".join(notes), opt))
        if pk.human_rounds and opt:
            parts.append(_cut(human_history_section(pk.human_rounds), opt))
        parts.append(decide)
        return "\n\n".join(parts) + "\n"

    prompt, truncated = "", False
    for opt in (4000, 1500, 0):
        prompt = compose(opt)
        if utf8_len(prompt) <= limits.max_prompt_bytes:
            truncated = opt < 4000
            break
    else:
        mandatory = _cut(mandatory, max(2000, limits.max_prompt_bytes - utf8_len(decide) - 200))
        prompt = compose(0)
        truncated = True
    return prompt, {"prompt_bytes": utf8_len(prompt), "limit_bytes": limits.max_prompt_bytes,
                    "truncated": int(truncated)}


def _dispositions(v: Any) -> list[dict[str, str]]:
    out = []
    for d in v if isinstance(v, list) else []:
        if not isinstance(d, dict):
            continue
        rid = str(d.get("requirement_id") or "").strip()[:40]
        disp = str(d.get("disposition") or "").strip().lower()
        if rid:
            out.append({"requirement_id": rid, "disposition": disp if disp in DISPOSITIONS else "still_required",
                        "reason": str(d.get("reason") or "").strip()[:600]})
    return out


def check_justified(review: Review, unresolved_ids: list[str]) -> str:
    """"" if the answer is acceptable; otherwise why it is invalid (an unjustified PASS)."""
    if review.decision != ReviewDecision.PASS:
        return ""
    c = review.consistency or {}
    if not str(c.get("justification") or "").strip():
        return "PASS without a justification for the unresolved required evidence"
    ok = {d["requirement_id"] for d in c.get("dispositions") or []
          if d.get("disposition") in ("not_needed", "substitute_evidence") and str(d.get("reason") or "").strip()}
    missing = [rid for rid in unresolved_ids if rid not in ok]
    if missing:
        return "PASS without an accepted reason for the unresolved required evidence " + ", ".join(missing)
    return ""


def parse_consistency(data: Any, *, unresolved_ids: list[str], allow_ask_user: bool) -> Review:
    """Claude's answer as a Review of kind CONSISTENCY_REVIEW. Raises ValueError for an invalid answer (a
    disallowed decision, ASK_USER without questions, or an unjustified PASS)."""
    allowed = (*CONSISTENCY_DECISIONS, "ASK_USER") if allow_ask_user else CONSISTENCY_DECISIONS
    review = Review.from_dict(data, allowed=allowed)
    review.kind = "CONSISTENCY_REVIEW"
    review.consistency = {"justification": str(data.get("justification") or "").strip()[:2000],
                          "dispositions": _dispositions(data.get("evidence_dispositions"))}
    why = check_justified(review, unresolved_ids)
    if why:
        raise ValueError(why)
    return review
