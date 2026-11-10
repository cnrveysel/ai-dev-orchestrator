"""V4 pass 3 MID_PHASE_DIAGNOSE: a cheap, tool-free Claude diagnosis when the normal implementation
guard stops a worker session (wall time, tool calls, or no planned-subtask progress).

The packet is short and bounded: the phase (objective, subtasks with their done state, criteria), why
the guard stopped the session, the budget used, the completed subtasks, the current verification
state (run by the orchestrator right now), the authored changes (a diff stat, never the whole diff),
the recent worker turns and earlier diagnoses. The answer (CONTINUE with a precise blocker and next
actions, or BLOCKED) is handed to a FRESH worker session; it never edits anything itself.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.attempt import VerificationResult
from app.models.phase import Phase
from app.orchestration.reviewer import _cut, _filter_stat, format_verification, utf8_len


@dataclass
class MidPhasePacket:
    phase: Phase
    reason: str  # why the guard stopped the session
    repo_path: str = ""
    goal: str = ""
    usage: dict[str, Any] = field(default_factory=dict)  # wall_used, tool_calls, no_progress, limits
    turns: list[dict[str, Any]] = field(default_factory=list)  # recent worker turn records of this phase
    worker_summary: str = ""
    verification: list[VerificationResult] = field(default_factory=list)  # the current state
    changed_files: list[str] = field(default_factory=list)
    diff_stat: str = ""
    previous: list[dict[str, Any]] = field(default_factory=list)  # earlier MID_PHASE_DIAGNOSE answers
    user_guidance: str = ""


@dataclass(frozen=True)
class MidPhaseLimits:
    max_prompt_bytes: int = 16000

    @classmethod
    def from_settings(cls, s: Any) -> "MidPhaseLimits":
        return cls(max(4000, int(getattr(s, "mid_phase_max_prompt_bytes", 16000))))


_DECIDE = """ANSWER (structured): status CONTINUE with `blocker` (what the worker is stuck on, from the evidence
above), `diagnosis`, `next_actions` (precise and ordered - a fresh worker session receives them verbatim),
`avoid` (what already failed or wasted the budget), `target_files` and `target_checks` (fast checks after each
change); or status BLOCKED with `blocked_reason` when progress needs a user decision, missing information or an
environment fix. Do not redesign the phase and do not ask for work that is already done."""


def build_mid_phase_prompt(pk: MidPhasePacket, limits: MidPhaseLimits) -> tuple[str, dict[str, int]]:
    p, u = pk.phase, pk.usage or {}
    done = [f"{s.id} {s.title}" for s in p.subtasks if s.done]
    open_ = [f"{s.id} {s.title}" for s in p.subtasks if not s.done]
    head = ["MID_PHASE_DIAGNOSE - a normal implementation session of this phase was stopped by the orchestrator's "
            "guard before its checkpoint. Diagnose the blocker; do NOT edit files (this session has no tools).",
            f"Guard reason: {pk.reason}",
            f"Budget of this session: {u.get('wall_used', 0):.0f}s of {u.get('wall_limit', 0):.0f}s, "
            f"{u.get('tool_calls', 0)} of {u.get('tool_limit', 0)} tool calls; no planned subtask finished for "
            f"{u.get('no_progress', 0):.0f}s (limit {u.get('no_progress_limit', 0):.0f}s)",
            "", f"PHASE {p.id}: {p.title}", "OBJECTIVE", _cut(p.description.strip() or "(none)", 1200),
            "", "ACCEPTANCE CRITERIA", *[f"{i}. {c}" for i, c in enumerate(p.acceptance_criteria, 1)],
            "", "PLANNED SUBTASKS DONE (structured progress): " + (", ".join(done) or "(none)"),
            "NOT DONE YET: " + (", ".join(open_) or "(none)")]
    if pk.user_guidance.strip():
        head += ["", "USER GUIDANCE", _cut(pk.user_guidance.strip(), 600)]
    head += ["", "CURRENT VERIFICATION (run by the orchestrator now)",
             format_verification(pk.verification, 3, 1500).replace("VERIFICATION\n", "", 1)
             if pk.verification else "(no verification commands defined)"]
    mandatory = "\n".join(head)

    def compose(opt: int) -> str:
        parts = [mandatory]
        stat = _filter_stat(pk.diff_stat)
        changes = "\n".join(stat) if stat else ", ".join(pk.changed_files[:40])
        parts.append(_cut(f"AUTHORED CHANGES OF THIS PHASE SO FAR ({len(pk.changed_files)} file(s))\n"
                          + (changes or "(none)"), max(300, opt // 2)))
        if pk.turns and opt:
            lines = ["RECENT WORKER TURNS (oldest first)"]
            for t in pk.turns[-6:]:
                lines.append(f"- turn {t.get('n')}: {t.get('purpose')} ({t.get('mode')} session), "
                             f"{t.get('duration', 0)}s, {t.get('tool_calls', 0)} tool call(s), status "
                             f"{t.get('reported_status') or 'none'}")
            parts.append(_cut("\n".join(lines), opt))
        if pk.worker_summary.strip() and opt:
            parts.append(_cut("THE WORKER'S LAST REPORT\n" + pk.worker_summary.strip(), opt))
        if pk.previous and opt:
            lines = ["EARLIER DIAGNOSES OF THIS PHASE (do not repeat advice that did not work)"]
            for d in pk.previous[-3:]:
                lines.append(f"- {d.get('reason', '')}: {_cut(str(d.get('blocker') or d.get('diagnosis') or ''), 400)}")
                lines += [f"  do: {x}" for x in (d.get("next_actions") or [])[:4]]
            parts.append(_cut("\n".join(lines), opt))
        parts.append(_DECIDE)
        return "\n\n".join(parts) + "\n"

    prompt = ""
    for opt in (3000, 1200, 0):
        prompt = compose(opt)
        if utf8_len(prompt) <= limits.max_prompt_bytes:
            break
    else:
        mandatory = _cut(mandatory, max(1500, limits.max_prompt_bytes - utf8_len(_DECIDE) - 600))
        prompt = _cut(compose(0), limits.max_prompt_bytes)
    return prompt, {"prompt_bytes": utf8_len(prompt), "limit_bytes": limits.max_prompt_bytes}
