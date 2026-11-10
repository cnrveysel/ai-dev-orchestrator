"""Pipeline V2 worker prompts.

A NEW session gets the full phase prompt (context, subtasks, tools, rules, report
format). A RESUMED session already holds all of that, so it only gets a short,
purpose-specific message (continue / checkpoint feedback / verification failure /
interruption) - the worker never has to rediscover the repository.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.attempt import VerificationResult
from app.models.human import criteria_lines
from app.models.phase import Phase
from app.models.task import TaskResult
from app.workers.base import (
    CODEBASE_MEMORY_DEGRADED_LINE, COMMON_RULE_LINES, RetryContext, unreal_lines,
)
from app.workers.process_runner import tail

SCRATCH_DIR = "scratch_smoke"

REPORT_FORMAT = [
    "PHASE STATUS: COMPLETE | INCOMPLETE | BLOCKED",
    "SUBTASKS: one line per subtask, e.g. [x] S1 <title>  /  [ ] S2 <title> - <why not done>",
    "SUMMARY: <what you did in this phase>",
    "FILES CHANGED: <list>",
    "DECISIONS: <important decisions>",
    "RISKS: <remaining risks / anything unfinished>",
    "VERIFICATION: <what you built/tested and the result>",
    "BLOCKERS: <only when PHASE STATUS is BLOCKED: what you need>",
]
# Subtask progress is synchronised from the worker's structured todo list, never from prose.
TODO_PROGRESS_RULE = ("Progress is tracked ONLY from your todo list: as soon as a subtask is finished, update the plan "
                      "tool to mark its entry completed, BEFORE you start the next subtask. Writing \"S1 is complete\" "
                      "in a message does not count.")


@dataclass
class Handoff:
    """Context for a NEW session that continues earlier work of the same phase."""
    reason: str
    previous_summary: str = ""
    files_changed: list[str] = field(default_factory=list)
    subtasks_done: list[str] = field(default_factory=list)
    # V4 pass 3: the lead's MID_PHASE_DIAGNOSE after the implementation guard stopped the previous session
    guidance: list[str] = field(default_factory=list)


@dataclass
class PhaseWorkSpec:
    phase: Phase
    repo_path: str
    phase_index: int = 1
    phase_count: int = 1
    goal: str = ""
    plan_summary: str = ""
    architecture_notes: list[str] = field(default_factory=list)
    completed: list[tuple[str, str, TaskResult | None, bool]] = field(default_factory=list)  # id, title, result, superseded
    upcoming: list[tuple[str, str]] = field(default_factory=list)
    codebase_memory_available: bool = False
    codebase_memory_project: str = ""
    unreal_project: bool = False
    unreal_mcp_reachable: bool | None = None
    retry: RetryContext | None = None
    handoff: Handoff | None = None


# HOW TO WORK: the discovery procedure when the repository graph is available.
GRAPH_FIRST_WORK_LINES = [
    "- Discovery: use the indexed codebase-memory graph as your primary map of the repository. Before broad "
    "exploration or reading multiple source files, query the graph (see TOOLS) to find the relevant subsystems, "
    "symbols, dependencies, callers/callees and event flow. Then read only the targeted source you need to "
    "implement and verify the change.",
    "- Avoid \"to understand the system I will read these five files in full\". Instead: graph -> the relevant "
    "symbols and relationships -> read only those implementations (a function, a member, a line range). The same "
    "applies to FILES LIKELY INVOLVED: outline them first, then read the parts that matter.",
]


def _human_criteria_lines(phase: Phase, heading: str) -> list[str]:
    """V4: criteria derived from the user's verification of the running result (none for older phases)."""
    if not phase.human_criteria:
        return []
    return ["", heading] + [f"- {x}" for x in criteria_lines(phase.human_criteria)]


def _evidence_expectation_lines(phase: Phase) -> list[str]:
    """V4: what the lead will look for at the checkpoint (the verification contract; WHAT, not how)."""
    c = phase.verification_contract
    if c is None or not c.evidence_requirements:
        return []
    return ["", "EVIDENCE THE LEAD WILL LOOK FOR AT THE CHECKPOINT (make it observable in your verification and "
                "report - do not claim what you did not check):"] + \
        [f"- {r.id}{' (required)' if r.required else ''}: {r.goal}" for r in c.evidence_requirements] + \
        ["AI Studio collects this evidence itself and trusts only what it runs: your report, and logs your own runs "
         "left behind, are not evidence. A verification helper script is run by AI Studio only when it is declared "
         "in .aistudio/evidence.json (\"commands\": name, command, kind) - it then runs it and reads its output itself."]


def _codebase_memory_lines(spec: PhaseWorkSpec) -> list[str]:
    """codebase-memory is the PRIMARY map of a phase session: graph queries narrow things down,
    then targeted reads / rg. Tool names and arguments are those of codebase-memory-mcp 0.11.0."""
    if not spec.codebase_memory_available:
        return [CODEBASE_MEMORY_DEGRADED_LINE]
    lines = [
        "- codebase-memory MCP is your PRIMARY map for repository-wide discovery in this phase.",
        f"- This repository's exact absolute root path is: {spec.repo_path}",
    ]
    if spec.codebase_memory_project:
        lines.append(f"- Project: the orchestrator already matched this exact root to project "
                     f"\"{spec.codebase_memory_project}\". Use exactly that name and do not call list_projects; call "
                     "index_status once at the start of this session to confirm it is ready (its root_path must be "
                     "this root).")
    else:
        lines.append("- Project: resolve it ONCE at the start of this session as described below, then call "
                     "index_status once to confirm it is ready.")
    lines += [
        "- Resolving the project (only when no project is given above, or codebase-memory reports it unknown or "
        "with another root): call list_projects and take the entry whose root_path is exactly this root (same "
        "absolute path; only slash direction and letter case may differ); if there is none, call index_repository "
        "with repo_path set to this exact absolute root. Use the exact project name codebase-memory returns. NEVER "
        "guess the project name or derive it from the repository folder name.",
        "- Remember the resolved project name for this whole session and reuse it in every codebase-memory call. "
        "Do NOT repeat list_projects / index_status / index_repository for later subtasks, continuations or "
        "checkpoint fixes.",
        "- Never use or query a codebase-memory index belonging to another repository.",
        "- Which codebase-memory tool answers which question (ask the graph FIRST for these):",
        "  - subsystems, layers, entry points, hotspots: get_architecture (aspects e.g. [\"overview\"]; path narrows "
        "it to a directory)",
        "  - where is X / which symbols handle Y: search_graph (query, name_pattern / qn_pattern, label, file_pattern); "
        "rows give the qualified name, file, lines and CALLS/USAGE counts",
        "  - who calls X, what X calls, how data reaches Y: trace_path (function_name, direction inbound / outbound / "
        "both, depth; mode data_flow follows arguments)",
        "  - what is in a file: get_file_outline (exact repository-relative file_path) - before reading the file",
        "  - a string or pattern across the code (e.g. an event name that links emitters and handlers): search_code "
        "(pattern; mode compact or files)",
        "  - multi-hop relationships or aggregations: query_graph (read-only Cypher; get_graph_schema lists the labels "
        "and edge types)",
        "  - impact of your own edits: detect_changes (maps the git diff to the affected symbols and callers)",
        "- get_code_snippet reads ONE symbol: pass the qualified name of the specific function or method (from "
        "search_graph or an outline) and keep source_mode auto - a large class comes back as an outline, then fetch "
        "the member you need. Do not use source_mode full / a large max_lines to dump whole classes or files.",
        "- After the graph has narrowed things down, read the targeted implementation directly (that function or line "
        "range) and use rg for exact, targeted text searches. Direct reads are also fine when you already know the "
        "exact file and function. Shell/file tools remain right for exact source details, editing context, tests, "
        "builds, generated/runtime files the graph does not cover, and checking a graph answer against the source.",
        "- Do not recursively enumerate the repository (e.g. Get-ChildItem -Recurse, dir /s, rg --files over the "
        "whole tree) or read several whole files just to rediscover structure the graph already has.",
        "- Query the graph when it saves exploration; do not call graph tools performatively just to satisfy these "
        "instructions.",
        "- Explicit CALLS edges are strong evidence; REFERENCE / USAGE / semantic links are weaker. Check the source "
        "when graph coverage has gaps (index_status / check_index_coverage), when Unreal macros obscure behaviour, or "
        "for async / callback / event-bus behaviour.",
        "- If codebase-memory is unavailable, stale, returns errors or cannot answer the question, fall back to rg and "
        "targeted source reads - do not block on it.",
    ]
    return lines


def build_phase_prompt(spec: PhaseWorkSpec) -> str:
    """Full prompt for a NEW Qwen session working on a phase."""
    p = spec.phase
    lines: list[str] = [
        "You are the implementation worker for ONE PHASE of a larger plan.",
        "A technical lead owns the global plan and reviews your work only at the END of this phase (checkpoint).",
        "You own this phase: keep your own todo list for its subtasks, implement ALL of them in this session,",
        "verify your work, and only then write your final message. Do not implement later phases and do not",
        "redesign the overall plan.",
        "",
        f"REPOSITORY (your workspace root): {spec.repo_path}",
    ]
    if spec.goal:
        lines += ["", "OVERALL PROJECT GOAL (context only):", spec.goal]
    if spec.architecture_notes:
        lines += ["", "ARCHITECTURE NOTES FROM THE LEAD:"] + [f"- {n}" for n in spec.architecture_notes]
    if spec.completed:
        lines += ["", "COMPLETED PHASES (their changes are already in the repository; integrate with them):"]
        for pid, title, r, superseded in spec.completed:
            tag = " [superseded by a replan; its changes were kept]" if superseded else ""
            lines.append(f"- {pid} {title}{tag}: {r.result if r else ''}".rstrip(": "))
            if r and r.files_changed:
                lines.append(f"  files: {', '.join(r.files_changed[:15])}")
            for d in (r.important_decisions[:4] if r else []):
                lines.append(f"  decision: {d}")
    if spec.upcoming:
        lines += ["", "LATER PHASES (context only - do NOT implement them now):"]
        lines += [f"- {pid} {title}" for pid, title in spec.upcoming]
    lines += ["", f"CURRENT PHASE {p.id} ({spec.phase_index} of {spec.phase_count}): {p.title}",
              "", "OBJECTIVE:", p.description or "(none)"]
    if p.instructions:
        lines += ["", "LEAD'S INSTRUCTIONS:", p.instructions]
    lines += ["", "SUBTASKS (implement all of them, in a sensible order):"]
    for s in p.subtasks:
        lines.append(f"- {s.id}: {s.title}" + (f" - {s.description}" if s.description else ""))
    if p.acceptance_criteria:
        lines += ["", "ACCEPTANCE CRITERIA (checked at the phase checkpoint):"] + [f"- {c}" for c in p.acceptance_criteria]
    lines += _human_criteria_lines(p, "HUMAN-DERIVED ACCEPTANCE CRITERIA (from the user's verification of the running "
                                      "result; also checked at the checkpoint):")
    lines += _evidence_expectation_lines(p)
    if p.integration_expectations:
        lines += ["", "INTEGRATION EXPECTATIONS:"] + [f"- {c}" for c in p.integration_expectations]
    if p.files_likely_involved:
        lines += ["", "FILES LIKELY INVOLVED:"] + [f"- {f}" for f in p.files_likely_involved]
    if p.verification_commands:
        lines += ["", "AFTER YOU FINISH THE SYSTEM WILL RUN:"] + [f"- {c}" for c in p.verification_commands]
    if p.user_guidance:
        lines += ["", "GUIDANCE FROM THE USER:", p.user_guidance]
    if spec.handoff:
        h = spec.handoff
        lines += ["", f"HANDOFF - this is a NEW session continuing work on this phase ({h.reason}).",
                  "The earlier session's changes are already in the repository. Do NOT start over and do NOT",
                  "re-explore the whole project: inspect only the files below (and `git status` / `git diff --stat`;",
                  "note that earlier phases' changes are uncommitted too), then continue."]
        if h.subtasks_done:
            lines.append(f"Subtasks already reported done: {', '.join(h.subtasks_done)}")
        if h.files_changed:
            lines.append("Files changed so far in this phase: " + ", ".join(h.files_changed[:40]))
        if h.previous_summary:
            lines += ["Previous session's last report:", tail(h.previous_summary, 2500)]
        if h.guidance:
            lines += ["", "LEAD'S DIAGNOSIS OF WHY THE PREVIOUS SESSION DID NOT PROGRESS (follow it; do not repeat what it "
                          "says to avoid):", *h.guidance]
    if spec.retry:
        r = spec.retry
        lines += ["", f"CHECKPOINT FEEDBACK - phase attempt {r.attempt_number} of {r.max_attempts}. The lead reviewed "
                      "this phase and requested changes. Fix these issues; keep the good parts of the existing work."]
        if r.problems:
            lines += ["PROBLEMS FOUND BY THE LEAD:"] + [f"- {x}" for x in r.problems]
        if r.corrective_instructions:
            lines += ["CORRECTIVE INSTRUCTIONS (follow these):"] + [f"- {x}" for x in r.corrective_instructions]
    lines += [
        "", "HOW TO WORK:",
        "- Start by creating your todo list (plan tool) with one entry per subtask, each prefixed with its id "
        "(e.g. \"S1: ...\"). Entries of your own (exploring, verifying, cleanup) must not start with a subtask id.",
        f"- {TODO_PROGRESS_RULE}",
        *(GRAPH_FIRST_WORK_LINES if spec.codebase_memory_available else []),
        "- Explore once: after you have located the affected systems, do not re-scan the repository for each subtask.",
        "- Work through ALL subtasks before your final message; do not stop after the first one.",
        "- Integrate with the existing code and with completed phases: reuse their modules and APIs instead of duplicating them.",
        "- Run the verification commands (or equivalent quick checks) yourself before finishing and fix failures.",
        f"- Put temporary verification artifacts (smoke scripts, scratch bundles, test output) under {SCRATCH_DIR}/ "
        "at the repository root and DELETE them before your final message. Do not leave build outputs or bundles "
        "you only created for testing.",
        "- If you cannot continue without a decision from the lead or the user, stop and report PHASE STATUS: BLOCKED.",
        "", "TOOLS:",
        *_codebase_memory_lines(spec),
        *unreal_lines(spec.unreal_project, spec.unreal_mcp_reachable),
        "", "RULES:",
        *COMMON_RULE_LINES,
        "- No destructive commands: no git reset/clean/push/rebase/checkout of other branches, no deleting "
        f"files or directories unrelated to this phase (except your own {SCRATCH_DIR}/ artifacts). Do not commit.",
        "", "WHEN DONE, finish with a final message in this format:",
        *REPORT_FORMAT,
    ]
    return "\n".join(lines)


def _format_failures(failures: list[VerificationResult], budget: int = 3500) -> list[str]:
    lines: list[str] = []
    per = max(400, budget // max(1, len(failures)))
    for v in failures:
        status = "TIMEOUT" if v.exit_code is None else f"exit {v.exit_code}"
        lines.append(f"- {v.command}: FAILED ({status})")
        out = tail(v.output_tail or "", per).strip()
        if out:
            lines += [f"  | {x}" for x in out.splitlines()[-40:]]
    return lines


def build_turn_prompt(purpose: str, phase: Phase, *, retry: RetryContext | None = None,
                      failures: list[VerificationResult] | None = None, fix_round: int = 0,
                      fix_rounds: int = 0, note: str = "", codebase_memory_available: bool = False,
                      codebase_memory_project: str = "") -> str:
    """Short prompt for a RESUMED session (the session already holds the full phase context)."""
    open_ids = [s.id for s in phase.subtasks if not s.done]
    lines: list[str]
    if purpose == "retry" and retry is not None:
        lines = [f"PHASE {phase.id} CHECKPOINT: the technical lead reviewed this phase and requested changes "
                 f"(phase attempt {retry.attempt_number} of {retry.max_attempts}).",
                 "Your work is in the repository and you still have this session's context: fix the specific issues "
                 "below instead of starting over, and do not re-explore the project."]
        if retry.problems:
            lines += ["", "PROBLEMS:"] + [f"- {x}" for x in retry.problems]
        if retry.corrective_instructions:
            lines += ["", "CORRECTIVE INSTRUCTIONS (follow these):"] + [f"- {x}" for x in retry.corrective_instructions]
    elif purpose == "verify_fix":
        lines = [f"PHASE {phase.id}: the orchestrator's verification FAILED after your last turn "
                 f"(local fix round {fix_round} of {fix_rounds}, before the lead's checkpoint).",
                 "Fix the cause - do not weaken, skip or delete the checks - and re-run the failing command yourself.",
                 "", "FAILED CHECKS:", *_format_failures(failures or [])]
    elif purpose == "resume":
        lines = [f"RESUMING PHASE {phase.id} after an interruption (the orchestrator was stopped, paused or restarted, "
                 "or your process ended unexpectedly).",
                 "Your work so far is on disk. Re-check the file you were editing last (it may be partially edited), "
                 "update your todo list, then continue with the remaining subtasks."]
        if retry is not None and (retry.problems or retry.corrective_instructions):
            lines += ["", f"You were addressing the lead's checkpoint feedback (phase attempt {retry.attempt_number} of "
                          f"{retry.max_attempts}):"]
            lines += [f"- problem: {x}" for x in retry.problems]
            lines += [f"- fix: {x}" for x in retry.corrective_instructions]
    else:  # continue
        lines = [f"CONTINUE PHASE {phase.id} in this session: your earlier context is still valid - do not re-explore "
                 "the repository.",
                 "Your last message did not complete the phase."]
    lines += ["", TODO_PROGRESS_RULE]
    if codebase_memory_available or codebase_memory_project:
        project = f"project \"{codebase_memory_project}\"" if codebase_memory_project else "the project"
        lines += ["", f"codebase-memory: keep using {project} already resolved in this session (do not repeat "
                      "list_projects / index_status / index_repository). For code you have not located yet, ask the "
                      "graph first (search_graph / trace_path / get_file_outline), then read only the targeted "
                      "functions or line ranges."]
    if purpose in ("retry", "resume"):
        lines += _human_criteria_lines(phase, "HUMAN-DERIVED ACCEPTANCE CRITERIA (from the user's verification; must "
                                              "hold):")
    if note:
        lines += ["", note]
    if open_ids and purpose in ("continue", "resume"):
        lines += ["", "Subtasks not reported done yet: " + ", ".join(open_ids)]
    if phase.user_guidance:
        lines += ["", "GUIDANCE FROM THE USER:", phase.user_guidance]
    lines += ["", f"Remove temporary {SCRATCH_DIR}/ artifacts before finishing. When done, end with the same final "
                  "message format:", *REPORT_FORMAT]
    return "\n".join(lines)


def turn_record(n: int, purpose: str, mode: str, outcome: Any, report_status: str) -> dict[str, Any]:
    return {
        "n": n, "purpose": purpose, "mode": mode, "session_id": outcome.session_id,
        "exit_code": outcome.exit_code, "duration": round(outcome.duration, 1), "tool_calls": outcome.tool_calls,
        "timed_out": outcome.timed_out, "error": (outcome.error or "")[:500], "reported_status": report_status,
        "summary": tail(outcome.summary or "", 1500),
    }


# ------------------------------------------------------------ V2 REPAIRING (after a checkpoint RETRY)

@dataclass
class RepairSpec:
    """Everything a repair prompt needs. A NEW repair session gets a concise reconstruction of
    the phase (never the historical transcript); the repository itself is the state."""
    phase: Phase
    repo_path: str
    repair_attempt: int = 1
    attempt_number: int = 2
    max_attempts: int = 3
    reason: str = "the checkpoint requested changes"
    checkpoint: dict[str, Any] = field(default_factory=dict)  # summary, problems, corrective_instructions
    current: Any = None  # VerificationEvidence of the current state
    best: Any = None  # VerificationEvidence of the best-known state
    files_changed: list[str] = field(default_factory=list)
    protected_tests: list[str] = field(default_factory=list)
    authorized_tests: list[str] = field(default_factory=list)
    diagnosis: Any = None  # Diagnosis from Claude (DIAGNOSE_BLOCKER)
    notes: list[str] = field(default_factory=list)  # regression / restore / rejected test edits
    wall_seconds: int = 1200
    tool_calls: int = 30
    max_probe_files: int = 3
    codebase_memory_available: bool = False
    codebase_memory_project: str = ""


def _evidence(ev: Any, n: int = 12) -> list[str]:
    if ev is None or not ev.has_evidence:
        return ["(no verification evidence yet)"]
    if ev.passed:
        return ["verification passed"]
    return [ev.summary()] + [f"  | {x}" for x in (ev.lines or ev.failures)[:n]]


def _best_differs(s: RepairSpec) -> bool:
    return (s.best is not None and s.best.has_evidence and s.current is not None
            and s.best.fingerprint != s.current.fingerprint)


def _diagnosis_lines(d: Any) -> list[str]:
    if d is None:
        return []
    lines = ["", "LEAD'S DIAGNOSIS OF THE BLOCKER (follow it):", f"- diagnosis: {d.diagnosis}"]
    lines += [f"- do: {x}" for x in d.recommended_actions]
    lines += [f"- avoid: {x}" for x in d.avoid]
    if d.target_files:
        lines.append("- target files: " + ", ".join(d.target_files[:15]))
    if d.target_tests:
        lines.append("- targeted checks after each change: " + "; ".join(d.target_tests[:6]))
    return lines


def _repair_rules(s: RepairSpec) -> list[str]:
    targeted = "the failing verification command"
    if s.diagnosis is not None and s.diagnosis.target_tests:
        targeted = "the targeted checks above"
    return [
        "- Your earlier work is on disk. First inspect the CURRENT state: `git status`, `git diff --stat`, the failing "
        "output above and the files it points to. Do not start over and do not re-explore the whole project.",
        "- Completed subtasks are complete: do NOT redo them. Address ONLY the checkpoint findings and the currently "
        "failing verification.",
        f"- Work evidence-first: form ONE hypothesis from the failing output, make the smallest change, then run "
        f"{targeted}. When it passes, run the full phase verification below before you declare success.",
        "- Progress is measured ONLY by verification results. Announcing a root cause does not count; if a hypothesis "
        "does not change the failing output, drop it and use the new evidence.",
        f"- Diagnostic probes: at most {s.max_probe_files} new probe/debug scripts in this session, under "
        f"{SCRATCH_DIR}/; prefer one proper regression test. Delete probes before finishing.",
        f"- Budget: this session is stopped after about {max(1, s.wall_seconds // 60)} minutes or {s.tool_calls} tool "
        "calls. The orchestrator re-runs the phase verification after your turn and keeps the best verified state.",
        "- If you cannot make progress without a decision from the lead or the user, report PHASE STATUS: BLOCKED.",
    ]


def _protected_lines(s: RepairSpec) -> list[str]:
    if not s.protected_tests and not s.authorized_tests:
        return []
    lines = ["", "PROTECTED ACCEPTANCE TESTS (read them; do NOT modify, delete, skip or relax them - they define "
                 "done; changes are reverted automatically). You may ADD new diagnostic/regression tests:",
             "- " + (", ".join(s.protected_tests[:30]) or "(none besides the ones below)")]
    if s.authorized_tests:
        lines.append("- The lead explicitly allowed changes to: " + ", ".join(s.authorized_tests[:15]))
    return lines


def _human_feedback_lines(ck: dict[str, Any]) -> list[str]:
    """V4: the user's verification that led to this repair (their own words, never paraphrased)."""
    fb = ck.get("human_feedback") or {}
    if not fb:
        return []
    what = {"ISSUE_FOUND": "found an issue", "CANNOT_VERIFY": "could not verify the result",
            "LOOKS_GOOD": "confirmed the result looks right"}.get(fb.get("verdict"), str(fb.get("verdict")))
    lines = ["", f"HUMAN VERIFICATION (authoritative - the user observed the running result and {what}):"]
    if fb.get("note"):
        lines.append(f"- the user's note: \"{fb['note']}\"")
    lines += [f"- the user was asked: {q}" for q in (fb.get("questions") or [])[:6]]
    return lines


def _do_not_redo_lines(ck: dict[str, Any]) -> list[str]:
    keep = ck.get("do_not_redo") or []
    return (["", "DO NOT REDO (confirmed working - keep it exactly as it is):"] + [f"- {x}" for x in keep]) if keep else []


def build_repair_prompt(s: RepairSpec) -> str:
    """Full prompt of a NEW repair session."""
    p = s.phase
    done = [f"{x.id} {x.title}" for x in p.subtasks if x.done]
    open_ = [f"{x.id} {x.title}" for x in p.subtasks if not x.done]
    ck = s.checkpoint or {}
    human = bool(ck.get("human_feedback"))
    found = ("the user verified the running result and the technical lead's follow-up found problems that you must "
             "fix now" if human else "the technical lead's checkpoint found problems that you must fix now")
    lines = [
        f"You are the REPAIR worker for phase {p.id} of a larger plan (repair attempt {s.repair_attempt}; phase attempt "
        f"{s.attempt_number} of {s.max_attempts}).",
        f"This is a NEW, bounded session ({s.reason}). The phase was implemented earlier and its work is in the "
        f"repository; {found}.",
        "", f"REPOSITORY (your workspace root): {s.repo_path}",
        "", f"PHASE {p.id}: {p.title}", "OBJECTIVE:", p.description or "(none)",
    ]
    if p.acceptance_criteria:
        lines += ["", "ACCEPTANCE CRITERIA:"] + [f"- {c}" for c in p.acceptance_criteria]
    lines += _human_criteria_lines(p, "HUMAN-DERIVED ACCEPTANCE CRITERIA (from the user's verification; they must hold "
                                      "after this repair):")
    lines += ["", "COMPLETED SUBTASKS (done - do NOT redo): " + (", ".join(done) or "(none reported)")]
    if open_:
        lines.append("SUBTASKS NOT REPORTED DONE: " + ", ".join(open_))
    lines += _do_not_redo_lines(ck)
    lines += _human_feedback_lines(ck)
    lines += ["", ("FINDINGS FROM THE HUMAN VERIFICATION FOLLOW-UP" if human else "CHECKPOINT FINDINGS")
              + " (address ONLY these and the failing verification):"]
    if ck.get("summary"):
        lines.append(f"- summary: {ck['summary']}")
    lines += [f"- problem: {x}" for x in ck.get("problems") or []]
    lines += [f"- fix: {x}" for x in ck.get("corrective_instructions") or []]
    lines += _diagnosis_lines(s.diagnosis)
    lines += ["", "CURRENT VERIFICATION FAILURES:", *_evidence(s.current)]
    if _best_differs(s):
        lines += ["BEST-KNOWN STATE IN THIS REPAIR:", *_evidence(s.best, 6)]
    if s.notes:
        lines += ["", "ORCHESTRATOR NOTES:"] + [f"- {x}" for x in s.notes]
    if s.files_changed:
        lines += ["", "FILES CHANGED IN THIS PHASE (inspect these first): " + ", ".join(s.files_changed[:40])]
    lines += _protected_lines(s)
    if p.verification_commands:
        lines += ["", "FULL PHASE VERIFICATION (run by the orchestrator after your turn):"]
        lines += [f"- {c}" for c in p.verification_commands]
    if p.user_guidance:
        lines += ["", "GUIDANCE FROM THE USER:", p.user_guidance]
    lines += ["", "HOW TO REPAIR:", *_repair_rules(s), f"- {TODO_PROGRESS_RULE}"]
    lines += ["", "TOOLS:"]
    if s.codebase_memory_available:
        project = f"project \"{s.codebase_memory_project}\"" if s.codebase_memory_project else "the project"
        lines += [f"- codebase-memory: use {project} for this exact repository root ({s.repo_path}); never guess a "
                  "project name. Ask the graph (search_graph / trace_path / get_file_outline / detect_changes) before "
                  "reading more than the targeted functions; fall back to rg when it cannot help."]
    else:
        lines.append(CODEBASE_MEMORY_DEGRADED_LINE)
    lines += ["", "RULES:", *COMMON_RULE_LINES,
              "- No destructive commands: no git reset/clean/push/rebase/checkout/stash. Do not commit.",
              "", "WHEN DONE, finish with a final message in this format:", *REPORT_FORMAT]
    return "\n".join(lines)


def build_repair_round_prompt(s: RepairSpec, round_no: int, trend_line: str, *, resumed_after_interrupt: bool = False,
                              wall_left: float = 0, tools_left: int = 0) -> str:
    """Short prompt for the next round in the SAME repair session."""
    p = s.phase
    if resumed_after_interrupt:
        lines = [f"RESUMING THE REPAIR of phase {p.id} after an interruption (the orchestrator was stopped, paused or "
                 "restarted, or your process ended). Your work so far is on disk; re-check the file you were editing "
                 "last (it may be partially edited), then continue the repair."]
        ck = s.checkpoint or {}
        if ck.get("problems") or ck.get("corrective_instructions"):
            lines += ["", "You are repairing these checkpoint findings:"]
            lines += [f"- problem: {x}" for x in ck.get("problems") or []]
            lines += [f"- fix: {x}" for x in ck.get("corrective_instructions") or []]
        lines += _human_feedback_lines(ck)
        lines += _human_criteria_lines(p, "HUMAN-DERIVED ACCEPTANCE CRITERIA (must hold after this repair):")
        lines += _do_not_redo_lines(ck)
    else:
        lines = [f"REPAIR ROUND {round_no} for phase {p.id}: the orchestrator re-ran the full phase verification after "
                 "your last turn.", f"RESULT: {trend_line}"]
    lines += ["", "CURRENT VERIFICATION FAILURES:", *_evidence(s.current)]
    if _best_differs(s):
        lines += ["BEST-KNOWN STATE:", *_evidence(s.best, 6)]
    if s.notes:
        lines += ["", "ORCHESTRATOR NOTES:"] + [f"- {x}" for x in s.notes]
    lines += _diagnosis_lines(s.diagnosis) if resumed_after_interrupt else []
    lines += ["", f"Budget left in this session: about {max(0, int(wall_left // 60))} min, {max(0, tools_left)} tool "
                  "calls.",
              "Next: drop hypotheses the evidence refuted, make ONE targeted change, run the failing check, then the "
              "full phase verification. Do not modify the protected acceptance tests. Completed subtasks stay done.",
              "", TODO_PROGRESS_RULE]
    if s.codebase_memory_available or s.codebase_memory_project:
        lines += ["", "codebase-memory: keep using the project already resolved in this session."]
    if p.user_guidance:
        lines += ["", "GUIDANCE FROM THE USER:", p.user_guidance]
    lines += ["", f"Remove temporary {SCRATCH_DIR}/ artifacts before finishing. End with the same final message "
                  "format:", *REPORT_FORMAT]
    return "\n".join(lines)
