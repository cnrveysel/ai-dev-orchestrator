"""Pipeline V4 HUMAN_FOLLOWUP: the compact, byte-bounded packet with which Claude adjudicates
the user's answer to an ASK_USER checkpoint. No tools; the answer schema offers PASS / RETRY /
BLOCKED only (no ASK_USER recursion).

Mandatory: the phase (objective, criteria, human-derived criteria, verification contract,
completed subtasks), the ASK_USER checkpoint, the request and the user's answer, and the
decision guidance. Optional sections - automated evidence detail, repository changes, earlier
human rounds, prior repair diagnoses and the diff - are compacted level by level to fit.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.attempt import VerificationResult
from app.models.human import HumanFeedback, HumanRequest, criteria_lines
from app.models.phase import Phase
from app.models.verification import contract_lines, issue_line
from app.orchestration.checkpoint import evidence_results_section, human_history_section
from app.orchestration.reviewer import _cut, _filter_stat, format_verification, utf8_len


@dataclass
class HumanFollowupPacket:
    phase: Phase
    request: HumanRequest
    feedback: HumanFeedback
    goal: str = ""
    repo_path: str = ""
    checkpoint: dict[str, Any] = field(default_factory=dict)  # the ASK_USER checkpoint (Review.to_dict())
    verification: list[VerificationResult] = field(default_factory=list)  # of the reviewed attempt
    diff_stat: str = ""
    changed_files: list[str] = field(default_factory=list)
    diff: str = ""
    human_history: list[dict[str, Any]] = field(default_factory=list)  # EARLIER rounds of this phase
    prior_diagnoses: list[dict[str, Any]] = field(default_factory=list)  # DIAGNOSE_BLOCKER results of this phase
    final: bool = False
    evidence: dict[str, Any] = field(default_factory=dict)  # V4 pass 2: the evidence bundle of the attempt


@dataclass(frozen=True)
class HumanFollowupLimits:
    max_prompt_bytes: int = 40000
    max_diff_bytes: int = 16000

    @classmethod
    def from_settings(cls, s: Any) -> "HumanFollowupLimits":
        prompt = max(6000, int(getattr(s, "human_followup_max_prompt_bytes", 40000)))
        return cls(prompt, max(0, min(int(getattr(s, "human_followup_max_diff_bytes", 16000)), prompt // 2)))


_VERDICT_TEXT = {
    "LOOKS_GOOD": "the user observed the running result and confirms it looks right",
    "ISSUE_FOUND": "the user observed a problem in the running result",
    "CANNOT_VERIFY": "the user could not verify it",
}

_DECIDE = """DECIDE (structured answer: PASS / RETRY / BLOCKED)
- The user's answer is authoritative about what they observed; judge its cause and scope yourself from the evidence
  above. The note is the user's own words.
- LOOKS_GOOD: normally PASS, unless something from your checkpoint is still unresolved (then RETRY for that).
- ISSUE_FOUND: decide whether the observed issue fails this phase's objective or criteria. Usually RETRY with
  corrective instructions aimed at exactly what the user observed; PASS (recording it in remaining_risks) if it is
  immaterial to this phase; BLOCKED if resolving it needs a user decision or is beyond what the worker can do.
- CANNOT_VERIFY: decide from the automated evidence alone: PASS if it is sufficient; RETRY if the worker can produce
  stronger evidence (e.g. an observable check or report); BLOCKED if acceptance depends on an observation nobody can
  make now.
- RETRY: `corrective_instructions` go verbatim to a fresh, time- and tool-bounded repair session (the work stays on
  disk): specific, concise, actionable. `do_not_redo`: what is confirmed working (by the user or the evidence) and must
  be kept as it is.
- `derived_criteria`: turn what the answer revealed into phase-scoped, checkable acceptance criteria that the existing
  criteria do not already state (none if nothing new). They are kept separately from the original criteria and
  checked at every later checkpoint of this phase.
- Optionally classify findings in `issues` (severity / disposition: descriptive metadata only).
- PASS: fill `phase_result` compactly (result, important_decisions, remaining_risks).
- BLOCKED: say in `summary` what the user must decide or provide."""


def _phase_part(pk: HumanFollowupPacket) -> str:
    p = pk.phase
    lines = [f"HUMAN_FOLLOWUP - adjudicate the user's verification of phase {p.id}: {p.title}"
             + (" (last phase of the run: final audit)" if pk.final else ""),
             "Your checkpoint asked the user to verify what the automated evidence could not settle. Decide PASS, RETRY "
             "or BLOCKED; ASK_USER is not available in this follow-up. This session has no tools."]
    if pk.goal.strip():
        lines += ["", "OVERALL GOAL", _cut(pk.goal.strip(), 1000)]
    lines += ["", "OBJECTIVE", _cut(p.description.strip() or "(none)", 1500), "", "ACCEPTANCE CRITERIA"]
    lines += [f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)] or ["(none given)"]
    if p.human_criteria:
        lines += ["", "HUMAN-DERIVED ACCEPTANCE CRITERIA ALREADY RECORDED"] + [f"- {x}" for x in
                                                                           criteria_lines(p.human_criteria)]
    if p.verification_contract is not None:
        lines += ["", "VERIFICATION CONTRACT"] + [_cut(x, 600) for x in contract_lines(p.verification_contract)]
    done = [f"{s.id} {s.title}" for s in p.subtasks if s.done]
    lines += ["", "COMPLETED SUBTASKS: " + (", ".join(done) or "(none reported)")]
    return "\n".join(lines)


def _answer_part(pk: HumanFollowupPacket) -> str:
    req, fb, ck = pk.request, pk.feedback, pk.checkpoint or {}
    lines = [f"YOUR CHECKPOINT (phase attempt {req.attempt}): ASK_USER",
             f"summary: {_cut(str(ck.get('summary') or req.checkpoint_summary), 1200)}"]
    lines += [f"- problem: {_cut(str(x), 400)}" for x in (ck.get("problems") or [])[:8]]
    lines += [f"- issue: {_cut(issue_line(i), 400)}" for i in (ck.get("issues") or [])[:8]]
    lines += ["", "HUMAN VERIFICATION REQUEST", f"reason: {_cut(req.reason, 800)}"]
    lines += [f"{i}. {q}" for i, q in enumerate(req.questions, 1)]
    lines += [f"- checklist: {x}" for x in req.checklist]
    lines += [f"- already established by automated evidence: {x}" for x in req.observations]
    lines += [f"- not established by automated evidence: {x}" for x in req.missing]
    lines += ["", "THE USER'S ANSWER", f"verdict: {fb.verdict} ({_VERDICT_TEXT.get(fb.verdict, '')})",
              "note: " + (fb.note or "(no note)")]
    return "\n".join(lines)


def _evidence(pk: HumanFollowupPacket, ok_lines: int, fail_bytes: int) -> str:
    ck = pk.checkpoint or {}
    parts = [evidence_results_section(pk.evidence)] if pk.evidence else []
    parts.append(format_verification(pk.verification, ok_lines, fail_bytes).replace(
        "(no verification commands defined for this task)", "(no verification commands ran for this phase)"))
    assess = ck.get("evidence_assessment") or []
    if assess:
        parts.append("YOUR EVIDENCE ASSESSMENT AT THE CHECKPOINT\n" + "\n".join(
            f"- {a.get('id')}: {a.get('status') or '?'}" + (f" - {a.get('note')}" if a.get("note") else "")
            for a in assess))
    notes = [f"- {x}" for x in (ck.get("verification_notes") or [])[:6]] + \
            [f"- risk: {x}" for x in (ck.get("remaining_risks") or [])[:6]]
    if notes:
        parts.append("CHECKPOINT NOTES\n" + "\n".join(notes))
    return "\n\n".join(parts)


def _repo(pk: HumanFollowupPacket, stat_bytes: int) -> str:
    lines = [f"REPOSITORY CHANGES OF THIS PHASE ({len(pk.changed_files)} authored file(s))"]
    stat = _filter_stat(pk.diff_stat)
    if stat:
        lines.append(_cut("\n".join(stat), stat_bytes))
    elif pk.changed_files:
        lines.append(_cut(", ".join(pk.changed_files[:60]), stat_bytes))
    return "\n".join(lines)


def _diagnoses(pk: HumanFollowupPacket) -> str:
    lines = ["PRIOR REPAIR DIAGNOSES OF THIS PHASE"]
    for d in pk.prior_diagnoses[-3:]:
        lines.append(f"- {d.get('status')}: {_cut(str(d.get('diagnosis', '')), 500)}")
        lines += [f"  action: {x}" for x in (d.get("recommended_actions") or [])[:4]]
    return "\n".join(lines)


def build_human_followup_prompt(pk: HumanFollowupPacket, limits: HumanFollowupLimits) -> tuple[str, dict[str, int]]:
    phase_part, answer = _phase_part(pk), _answer_part(pk)

    def compose(ok: int, fail: int, stat: int, opt: int, diff_bytes: int) -> str:
        parts = [phase_part, answer, _cut(_evidence(pk, ok, fail), opt * 2), _repo(pk, stat)]
        if pk.human_history:
            parts.append(_cut(human_history_section(pk.human_history), opt))
        if pk.prior_diagnoses:
            parts.append(_cut(_diagnoses(pk), opt))
        if pk.diff and diff_bytes > 0:
            d = _cut(pk.diff, diff_bytes)
            parts.append("RELEVANT DIFF (this phase" + (", truncated)" if d != pk.diff else ")") + "\n" + d)
        return "\n\n".join(parts)

    body = ""
    room = limits.max_prompt_bytes - utf8_len(_DECIDE) - 4
    for level in ((10, 2500, 3000, 3000, limits.max_diff_bytes), (3, 800, 1000, 1200, limits.max_diff_bytes // 3),
                  (1, 300, 300, 400, 0)):
        body = compose(*level)
        if utf8_len(body) <= room:
            break
    else:  # even the minimal packet is too large: shorten the phase part; the answer and guidance survive
        phase_part = _cut(phase_part, max(1000, room - utf8_len(compose(1, 200, 200, 200, 0)) + utf8_len(phase_part)))
        body = _cut(compose(1, 200, 200, 200, 0), room)
    prompt = body + "\n\n" + _DECIDE + "\n"
    return prompt, {"prompt_bytes": utf8_len(prompt), "limit_bytes": limits.max_prompt_bytes,
                    "diff_original_bytes": utf8_len(pk.diff)}
