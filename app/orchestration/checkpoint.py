"""Compact, hard-bounded context for the Claude phase CHECKPOINT (Pipeline V2).

Claude reviews a whole phase at its boundary (the last phase doubles as the final
audit). It receives ONLY:
  A. the phase (objective, instructions, subtasks with the worker's status, acceptance
     criteria, integration expectations)
  B. compact results of earlier phases + this phase's previous checkpoints
  C. the worker's final report and a summary of its session turns (never the transcript)
  D. independent verification results
  E. the phase's AUTHORED changes since the phase started (generated/temporary paths,
     earlier phases' and the user's pre-existing changes excluded) and a relevant diff
     within a byte budget
  F. final audit only: a run-level diff stat over all phases
  G. V4: the phase's verification contract, criteria derived from human feedback and the
     human verification rounds so far; ASK_USER is offered only when the human gate is open
  H. V4 pass 2: the evidence bundle - required evidence that is not SATISFIED is mandatory
     (right after the criteria); capabilities, plan, other results and artifacts are compacted

Same guarantees as the V1 reviewer (reviewer.py, whose helpers are reused): optional
sections are compacted level by level; if the prompt still exceeds its hard limit,
ReviewContextTooLarge is raised and Claude is not launched.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.evidence.model import SATISFIED, is_satisfied, result_line
from app.integrations import path_filter
from app.models.attempt import Attempt
from app.models.human import criteria_lines
from app.models.phase import Phase
from app.models.verification import contract_lines
from app.orchestration.reviewer import (
    _DIFF_MARKER_RESERVE, REVIEW_PROMPT_BYTES_CEILING, ReviewContext, ReviewContextTooLarge, _assemble_diff, _cut,
    _filter_stat, format_verification, parse_worker_result, rank_paths, split_diff, utf8_len,
)


@dataclass(frozen=True)
class CheckpointContextLimits:
    max_prompt_bytes: int = 60000
    max_diff_bytes: int = 40000
    max_changed_files: int = 60

    @classmethod
    def from_settings(cls, s: Any) -> "CheckpointContextLimits":
        prompt = max(4000, min(int(s.checkpoint_max_prompt_bytes), REVIEW_PROMPT_BYTES_CEILING))
        diff = max(0, min(int(s.checkpoint_max_diff_bytes), prompt))
        return cls(prompt, diff, max(1, int(s.checkpoint_max_changed_files)))


@dataclass
class CheckpointPackage:
    phase: Phase
    attempt: Attempt
    max_attempts: int
    phase_index: int = 1
    phase_count: int = 1
    final: bool = False
    goal: str = ""
    completed: list[str] = field(default_factory=list)  # compact result lines of earlier phases
    previous_checkpoints: list[dict[str, Any]] = field(default_factory=list)
    run_diff_stat: str = ""  # final audit: authored changes of the whole run
    user_preexisting: list[str] = field(default_factory=list)  # dirty before the run started
    degraded_notes: list[str] = field(default_factory=list)
    repo_path: str = ""
    repair_notes: list[str] = field(default_factory=list)  # V2 REPAIRING summary of this attempt
    # V4: ASK_USER is offered only when this is True (setting on AND the contract's human gate open).
    human_verification_allowed: bool = False
    human_history: list[dict[str, Any]] = field(default_factory=list)  # earlier human rounds of this phase
    # V4 pass 2: the evidence bundle of this attempt (+ "discovered": capability lines); {} = no evidence
    evidence: dict[str, Any] = field(default_factory=dict)
    max_questions: int = 4  # V4 pass 3: the most questions an ASK_USER request may ask


@dataclass
class CheckpointContext(ReviewContext):
    def log_lines(self) -> list[str]:
        return ["Claude CHECKPOINT context:"] + [f"{k}={self.stats[k]}" for k in self.LOG_KEYS]


@dataclass(frozen=True)
class _Opt:
    ok_lines: int
    fail_bytes: int
    worker_bytes: int
    stat_bytes: int
    history_bytes: int
    previous_bytes: int
    run_stat_bytes: int


_LEVELS = (
    _Opt(10, 2500, 4500, 3000, 3000, 2000, 4000),
    _Opt(3, 800, 2000, 1000, 1500, 800, 1500),
    _Opt(1, 300, 800, 300, 500, 300, 500),
)


def build_checkpoint_context(package: CheckpointPackage, limits: CheckpointContextLimits) -> CheckpointContext:
    phase, a = package.phase, package.attempt
    limit = min(limits.max_prompt_bytes, REVIEW_PROMPT_BYTES_CEILING)

    gen_by_dir: Counter = Counter(a.generated_by_dir or {})
    authored: dict[str, str] = {}
    for c in a.changed_files:
        if path_filter.is_generated(c.path):
            gen_by_dir[path_filter.generated_dir(c.path) or "generated files"] += 1
        else:
            authored[c.path] = c.change
    generated_total = sum(gen_by_dir.values())
    user_dirty = {p for p in package.user_preexisting if not path_filter.is_generated(p)}
    earlier = [p for p in a.preexisting_dirty if not path_filter.is_generated(p)]
    n_user = len([p for p in earlier if p in user_dirty])
    n_earlier = len(earlier) - n_user

    fields = parse_worker_result(a.worker_summary)
    ranked = rank_paths(list(authored), phase, fields)
    listed = ranked[:limits.max_changed_files]
    diffable = [p for p in listed if path_filter.classify(p) == path_filter.DIFFABLE]
    sections = split_diff(a.diff)
    mandatory = _mandatory_section(package)
    decide = decide_text(package)

    def compose(opt: _Opt, diff_budget: int) -> tuple[str, dict[str, int]]:
        parts = [mandatory]
        if package.completed:
            parts.append(_cut("EARLIER PHASES (compact results)\n" + "\n".join(f"- {x}" for x in package.completed),
                              opt.history_bytes))
        if package.previous_checkpoints:
            parts.append(_cut(_previous_section(package.previous_checkpoints), opt.previous_bytes))
        if package.human_history:
            parts.append(_cut(human_history_section(package.human_history), opt.previous_bytes))
        if package.evidence:
            parts.append(_cut(evidence_detail_section(package.evidence), opt.history_bytes))
        parts.append(_worker_section(package, fields, opt.worker_bytes))
        if package.repair_notes:
            parts.append(_cut("REPAIR (bounded repair after the previous checkpoint)\n"
                              + "\n".join(f"- {x}" for x in package.repair_notes), opt.previous_bytes))
        verification = format_verification(a.verification, opt.ok_lines, opt.fail_bytes)
        parts.append(verification)
        parts.append(_git_section(a, authored, listed, ranked, generated_total, gen_by_dir, n_earlier, n_user,
                                  opt.stat_bytes))
        if package.final and package.run_diff_stat:
            stat = _filter_stat(package.run_diff_stat, max_lines=80)
            if stat:
                parts.append("RUN-LEVEL CHANGES (authored files across all phases)\n"
                             + _cut("\n".join(stat), opt.run_stat_bytes))
        diff_text, original, used, with_diff = _assemble_diff(diffable, sections, diff_budget)
        parts += ["RELEVANT DIFF\n" + diff_text, decide]
        prompt = "\n\n".join(parts) + "\n"
        return prompt, {
            "changed_paths_total": len(authored) + generated_total,
            "generated_paths_excluded": generated_total,
            "authored_paths": len(authored),
            "files_included": len(with_diff),
            "files_listed": len(listed),
            "verification_bytes": utf8_len(verification),
            "diff_original_bytes": original,
            "diff_included_bytes": used,
            "final_prompt_bytes": utf8_len(prompt),
            "prompt_limit_bytes": limit,
        }

    prompt, stats = "", {}
    for opt in _LEVELS:
        without_diff, _ = compose(opt, 0)
        room = limit - utf8_len(without_diff) - _DIFF_MARKER_RESERVE
        prompt, stats = compose(opt, max(0, min(limits.max_diff_bytes, room)))
        if stats["final_prompt_bytes"] <= limit:
            return CheckpointContext(prompt, stats)
    raise ReviewContextTooLarge(stats.get("final_prompt_bytes", utf8_len(prompt)), limit)


def _mandatory_section(pk: CheckpointPackage) -> str:
    p, a = pk.phase, pk.attempt
    kind = "FINAL AUDIT (last phase of the run)" if pk.final else "PHASE CHECKPOINT"
    lines = [kind, f"{p.id} - {p.title} (phase {pk.phase_index} of {pk.phase_count}; "
                   f"attempt {max(1, p.attempts)} of {pk.max_attempts}, {a.kind})"]
    if pk.goal.strip():
        lines += ["", "OVERALL GOAL", _cut(pk.goal.strip(), 1500)]
    lines += ["", "OBJECTIVE", p.description.strip() or "(none)"]
    if p.instructions.strip():
        lines += ["", "INSTRUCTIONS", _cut(p.instructions.strip(), 3000)]
    lines += ["", "SUBTASKS (the worker's own breakdown; not reviewed individually)"]
    lines += [f"[{'x' if s.done else ' '}] {s.id} {s.title}" for s in p.subtasks] or ["(none)"]
    if p.user_guidance.strip():
        lines += ["", "USER GUIDANCE", p.user_guidance.strip()]
    lines += ["", "ACCEPTANCE CRITERIA"]
    lines += [f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)] or ["(none given)"]
    if p.human_criteria:
        lines += ["", "HUMAN-DERIVED ACCEPTANCE CRITERIA (derived from the user's verification of the running result; "
                      "separate from the criteria above and equally binding; each is also an evidence requirement)"]
        lines += [f"- {x}" for x in criteria_lines(p.human_criteria, with_status=True)]
    if p.integration_expectations:
        lines += ["", "INTEGRATION EXPECTATIONS"] + [f"- {c}" for c in p.integration_expectations]
    if p.verification_contract is not None:
        lines += ["", "VERIFICATION CONTRACT (what convincing evidence of success looks like for this phase)"]
        lines += [_cut(x, 600) for x in contract_lines(p.verification_contract)]
        if p.verification_contract.human_gate.allowed and not pk.human_verification_allowed:
            lines.append("(human verification is currently disabled by the user: decide from the evidence)")
    if pk.evidence:
        lines += ["", _cut(evidence_results_section(pk.evidence), 6000)]
    return "\n".join(lines)


def evidence_results_section(ev: dict[str, Any]) -> str:
    """Mandatory: every requirement's result, required gaps first (what actually ran - nothing inferred)."""
    results = list(ev.get("results") or [])
    rank = {"FAILED": 0, "UNAVAILABLE": 1, "INCONCLUSIVE": 2, "SKIPPED": 3, SATISFIED: 5}
    results.sort(key=lambda r: (0 if r.get("required") else 1, rank.get(r.get("status"), 4)))
    lines = ["EVIDENCE RESULTS (executed evidence steps; a property is verified ONLY when its result is SATISFIED and "
             "not stale)"]
    for r in results:
        lines.append(result_line(r, 300))
        if r.get("status") != SATISFIED and r.get("failure_reason") and r.get("failure_reason") not in r.get("summary", ""):
            lines.append(f"  reason: {_cut(str(r['failure_reason']), 300)}")
        lines += [f"  | {_cut(str(x), 240)}" for x in (r.get("diagnostics") or [])[:4]]  # reduced, never whole logs
    gaps = [r.get("requirement_id") for r in results if r.get("required") and not is_satisfied(r)]
    lines.append("required evidence not satisfied: " + (", ".join(gaps) if gaps else "none"))
    return "\n".join(lines)


def evidence_detail_section(ev: dict[str, Any]) -> str:
    """Optional (compacted): capabilities, the plan, replans, protection, notes and artifact references."""
    lines = [f"EVIDENCE COLLECTION DETAILS - discovered: {ev.get('capability_summary', '')}"]
    lines += [f"- {x}" for x in (ev.get("discovered") or [])[:20]]
    lines.append(f"plan ({ev.get('plan_source') or 'claude'}): {ev.get('plan_summary') or '(no summary)'}")
    for r in ev.get("results") or []:
        lines.append(f"- {r.get('requirement_id')}: strategy {r.get('strategy')}; capabilities "
                     f"{', '.join(r.get('capability_ids') or []) or 'none'}; confidence {r.get('confidence')}"
                     + (f"; cause {r.get('cause')}" if r.get("cause") else ""))
    for rp in (ev.get("replans") or [])[:3]:
        lines.append(f"replan {rp.get('generation')} of {', '.join(rp.get('requirements') or [])}: "
                     + "; ".join(str(x) for x in (rp.get("reasons") or [])[:2]))
    guard = ev.get("guard") or {}
    if guard:
        lines.append(f"repository protection: {guard.get('detail') or guard.get('kind')}")
    lines += [f"limitation: {x}" for x in (ev.get("limitations") or [])[:6]]
    lines += [f"note: {x}" for x in (ev.get("notes") or [])[:6]]
    arts = ev.get("artifacts") or []
    if arts:
        lines.append("artifacts (references; contents not included):")
        lines += [f"- {a.get('id')} [{a.get('kind')}{', ' + a['media_type'] if a.get('media_type') else ''}] "
                  f"{a.get('description')}" for a in arts[:12]]
        if len(arts) > 12:
            lines.append(f"- ... {len(arts) - 12} more artifact(s) not listed")
    return "\n".join(lines)


def _previous_section(prev: list[dict[str, Any]]) -> str:
    lines = ["PREVIOUS CHECKPOINTS OF THIS PHASE (check that these problems are fixed)"]
    for c in prev:
        kind = " (human follow-up)" if c.get("kind") == "HUMAN_FOLLOWUP" else ""
        lines.append(f"- attempt {c.get('number')}: {c.get('decision')}{kind} - {c.get('summary', '')}")
        lines += [f"  problem: {x}" for x in (c.get("problems") or [])[:8]]
    return "\n".join(lines)


def human_history_section(rounds: list[dict[str, Any]]) -> str:
    """Earlier human verification rounds of the phase: what was asked and what the user answered."""
    lines = ["HUMAN VERIFICATION OF THIS PHASE SO FAR (the user's answers are authoritative observations; do not ask "
             "again what was already answered unless the work changed it)"]
    for r in rounds:
        req, fb, fu = r.get("request") or {}, r.get("feedback") or {}, r.get("followup") or {}
        lines.append(f"- attempt {req.get('attempt', '?')}: asked because {_cut(str(req.get('reason', '')), 300)}")
        lines += [f"  question: {_cut(str(q), 300)}" for q in (req.get("questions") or [])[:6]]
        if fb:
            lines.append(f"  user answered {fb.get('verdict')}" + (f": {_cut(str(fb.get('note', '')), 600)}"
                                                                   if fb.get("note") else ""))
        else:
            lines.append(f"  not answered ({r.get('status', '')})")
        if fu:
            lines.append(f"  follow-up decision: {fu.get('decision')} - {_cut(str(fu.get('summary', '')), 300)}")
    return "\n".join(lines)


def _worker_section(pk: CheckpointPackage, fields: dict[str, str], budget: int) -> str:
    a = pk.attempt
    new = sum(1 for t in a.turns if t.get("mode") == "new")
    resumed = len(a.turns) - new
    elapsed = sum(float(t.get("duration") or 0) for t in a.turns)
    head = (f"exit code {a.exit_code}, {elapsed:.0f}s over {len(a.turns)} worker turn(s): {new} new session, "
            f"{resumed} resumed; local verification fix rounds: {a.local_fix_rounds}; "
            f"reported PHASE STATUS: {pk.phase.reported_status or 'not stated'}")
    lines = [f"WORKER RESULT ({head})"]
    if a.error:
        lines.append(f"Worker error: {_cut(a.error, 400)}")
    if pk.degraded_notes:
        lines.append("Degraded tooling: " + _cut("; ".join(pk.degraded_notes), 300))
    if not fields:
        lines.append("(the worker produced no final message)")
    per_field = max(150, budget // max(1, len(fields)))
    for name, value in fields.items():
        lines.append(f"{name}: {_cut(value, per_field)}")
    return _cut("\n".join(lines), budget)


def _git_section(a: Attempt, authored: dict[str, str], listed: list[str], ranked: list[str], generated_total: int,
                 gen_by_dir: Counter, n_earlier: int, n_user: int, stat_bytes: int) -> str:
    scope = ("authored changes of this phase so far (all attempts)" if a.diff_scope == "phase"
             else "authored changes of this attempt only (phase-level diff unavailable)")
    summary = f"{len(authored)} authored file(s) changed"
    if generated_total:
        top = ", ".join(f"{d}: {n}" for d, n in gen_by_dir.most_common(6))
        summary += f"; {generated_total} generated/temporary path(s) excluded ({top})"
    if n_earlier:
        summary += f"; {n_earlier} path(s) changed by earlier phases are not part of this diff"
    if n_user:
        summary += f"; {n_user} pre-existing user change(s) not attributed to the worker"
    lines = [f"GIT - {scope}", summary]
    stat = _filter_stat(a.diff_stat)
    if stat:
        lines += ["diff --stat (authored files):", _cut("\n".join(stat), stat_bytes)]
    if listed:
        lines.append(f"Files (showing {len(listed)} of {len(authored)}, most relevant first):")
        for p in listed:
            cls = path_filter.classify(p)
            note = f"  [diff omitted: {path_filter.metadata_reason(p)}]" if cls == path_filter.METADATA else ""
            lines.append(f"{authored[p]} {p}{note}")
        if len(ranked) > len(listed):
            lines.append(f"... {len(ranked) - len(listed)} more authored path(s) omitted")
    return "\n".join(lines)


_DECIDE = """DECIDE
Return the structured checkpoint result: PASS / RETRY / BLOCKED / REPLAN.
The evidence above is what the orchestrator collected for this phase (this session has no tools).
- Judge the phase as a whole: acceptance criteria, architecture, integration with earlier phases and the
  existing code, verification results, regressions. Subtasks are the worker's own breakdown; do not fail the
  phase over subtask bookkeeping alone.
- PASS if the acceptance criteria are met and the work integrates correctly. Minor, non-blocking issues belong
  in remaining_risks, not in a RETRY.
- RETRY if the phase is incomplete or wrong but fixable by the worker. A fresh, time- and tool-bounded repair
  session receives `corrective_instructions` verbatim (the work stays on disk): make them specific, concise and
  actionable; do not restate the whole phase. Failed verification normally means RETRY. During repair the
  existing acceptance tests are protected; only if a test itself is wrong, list it in `authorized_test_changes`.
- BLOCKED if progress needs a user decision, missing information or an environment fix.
- REPLAN if the remaining plan no longer fits the actual implementation (this phase's objective must be
  re-scoped, or later phases must be restructured before continuing). The work done so far stays in the
  repository; explain why in `replan_reason`.
- If the evidence is genuinely insufficient to judge a criterion, name it in `problems` rather than guessing.
- With PASS, set `plan_changes_required` only if later phases must change (explain in `replan_reason`).
- On PASS fill `phase_result` compactly (result, important_decisions, remaining_risks): it is the only record of
  this phase that later phases see."""

_FINAL = """
FINAL AUDIT: this is the last phase of the run. Additionally audit the integrated result against the overall
goal (RUN-LEVEL CHANGES above): cross-phase integration, regressions, leftover temporary artifacts, and whether
the product works end to end. Use RETRY for integration defects the worker can fix now, and REPLAN only if a
substantial follow-up phase is required."""

_ISSUES = """
- Optionally classify each finding in `issues` (severity: critical | functional | regression | polish | subjective |
  micro_deviation; disposition: repair | ask_user | accept | blocked). The labels are descriptive metadata for the
  record: decide from this phase's actual requirements and evidence, not from the labels."""

_CONTRACT = """
- VERIFICATION CONTRACT: judge every evidence requirement against the evidence above and report it in
  `evidence_assessment` (satisfied | unsatisfied | insufficient | not_applicable, with a short note). Do not assume a
  runtime or perceptual requirement is met because static checks pass - say what the evidence does and does not show.
  Missing required evidence is a finding: when the worker can produce it (e.g. a runnable check or a clear
  observable report), a RETRY can ask for it."""

_ASK_USER = """
- ASK_USER (human verification is enabled for this phase): use it ONLY when a person's observation of the running
  result would materially change the acceptance decision AND the automated evidence cannot settle it reliably or
  cheaply (e.g. how the result looks, moves, sounds or feels). Never ask about what the evidence already settles, and
  do not ask by default - most checkpoints need no human. Fill `human_request`: `reason`; `questions` - few (at most
  {max_q}), short, specific and OBSERVABLE, each about ONE unresolved criterion or evidence gap of THIS phase (bad: "does
  everything work?"; good: a concrete thing to watch for in the running result), never about what automation already
  proved; `question_targets` - the requirement id each question resolves (E1, HF-P2-01, ... or ""); and
  `observations` (what the automated evidence already established). You will adjudicate the user's answer in a
  follow-up. If you ask, still list the problems you already found."""

_NO_HUMAN = """
- Human verification is not available for this phase: decide from the available evidence. If a criterion can only
  be confirmed by a person, say so in `problems` or `remaining_risks` and weigh it in your decision."""


_EVIDENCE = """
- EVIDENCE: the orchestrator discovered what this environment can actually do, you planned evidence with those
  capabilities, and EVIDENCE RESULTS show what really ran. Fill `evidence_sufficient` (does the evidence suffice to
  accept this phase?) and `missing_evidence` (required evidence still missing or unconvincing). Green builds or tests do
  not by themselves prove runtime, visual or interaction properties, and INCONCLUSIVE, UNAVAILABLE or SKIPPED results
  prove nothing; never treat a property as verified without a SATISFIED result. A FAILED required result demonstrates
  a problem; a STALE result was collected before a later change and proves nothing now; cause "unobserved" means the
  check ran cleanly (exit 0) but never showed the expected output - the product may lack the property or the check
  may not report it. "(reused ...)" means no step of that result ran now; "(ran now; ...)" means it was produced in
  this collection. What the worker reports it ran is not evidence. Weigh the required evidence,
  the criteria, alternative evidence and the risk: missing evidence is not by itself a reason to block, and green
  checks are not by themselves a reason to pass. Keep your answer consistent: a PASS with evidence_sufficient=false
  while required evidence is unresolved is reviewed again before it counts."""

_EVIDENCE_HUMAN = """
- If what is missing is something a person can observe in the running result, ASK_USER can close that gap: ask
  exactly about the unverified part, and list what the evidence already established in `observations`."""


def decide_text(pk: CheckpointPackage) -> str:
    text = _DECIDE
    if pk.human_verification_allowed:
        text = text.replace("PASS / RETRY / BLOCKED / REPLAN.", "PASS / RETRY / BLOCKED / REPLAN / ASK_USER.", 1)
    text += _ISSUES
    if pk.phase.verification_contract is not None:
        text += _CONTRACT + (_ASK_USER.replace("{max_q}", str(max(1, min(6, pk.max_questions))))
                             if pk.human_verification_allowed else _NO_HUMAN)
    if pk.evidence:
        text += _EVIDENCE + (_EVIDENCE_HUMAN if pk.human_verification_allowed else "")
    return text + (_FINAL if pk.final else "")
