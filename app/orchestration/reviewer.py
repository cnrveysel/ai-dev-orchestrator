"""Compact, hard-bounded context for Claude REVIEW (runs after every task).

The reviewer receives ONLY:
  A. the task (id, title, description/instructions, acceptance criteria)
  B. the worker's FINAL result (parsed from its last message; never the transcript,
     tool calls or intermediate reasoning)
  C. concise independent verification results
  D. a git summary of AUTHORED changes (generated/dependency paths excluded)
  E. a relevant source diff within a byte budget

Hard limits (UTF-8 bytes): the final prompt never exceeds max_prompt_bytes. If it
does, optional context is compacted; if it still does, ReviewContextTooLarge is
raised and Claude is not launched.
"""
from __future__ import annotations

import posixpath
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

from app.integrations import path_filter
from app.models.attempt import Attempt, VerificationResult
from app.models.task import Task, TaskResult
from app.orchestration.providers.base import OrchestratorError
from app.workers.process_runner import tail

# Absolute ceiling: no configuration can let a larger review prompt reach Claude.
REVIEW_PROMPT_BYTES_CEILING = 80_000
_DIFF_MARKER_RESERVE = 700  # bytes kept free for truncation markers / omitted-file notes


class ReviewContextTooLarge(OrchestratorError):
    def __init__(self, size: int, limit: int):
        super().__init__(f"Reviewer context exceeded hard limit: {size} > {limit} bytes")
        self.size = size
        self.limit = limit


@dataclass(frozen=True)
class ReviewContextLimits:
    max_prompt_bytes: int = 40000
    max_diff_bytes: int = 25000
    max_changed_files: int = 30

    @classmethod
    def from_settings(cls, s: Any) -> "ReviewContextLimits":
        prompt = max(2000, min(int(s.review_max_prompt_bytes), REVIEW_PROMPT_BYTES_CEILING))
        diff = max(0, min(int(s.review_max_diff_bytes), prompt))
        return cls(prompt, diff, max(1, int(s.review_max_changed_files)))


@dataclass(frozen=True)
class _Optional:
    """Sizes of the optional sections; later levels are progressively smaller."""
    ok_lines: int
    fail_bytes: int
    worker_bytes: int
    stat_bytes: int


_LEVELS = (
    _Optional(ok_lines=10, fail_bytes=2500, worker_bytes=3500, stat_bytes=3000),
    _Optional(ok_lines=3, fail_bytes=800, worker_bytes=1500, stat_bytes=1000),
    _Optional(ok_lines=1, fail_bytes=300, worker_bytes=600, stat_bytes=300),
)


@dataclass
class ReviewPackage:
    """Input from the scheduler. Only the fields listed in the module docstring are
    rendered into the prompt; the rest is kept for the mock reviewer / history."""
    task: Task
    attempt: Attempt
    max_attempts: int
    previous_attempts: list[dict[str, Any]] = field(default_factory=list)
    dependency_results: list[TaskResult] = field(default_factory=list)
    degraded_notes: list[str] = field(default_factory=list)
    repo_path: str = ""


@dataclass
class ReviewContext:
    prompt: str
    stats: dict[str, int]

    LOG_KEYS = ("changed_paths_total", "generated_paths_excluded", "authored_paths", "files_included",
                "files_listed", "verification_bytes", "diff_original_bytes", "diff_included_bytes",
                "final_prompt_bytes", "prompt_limit_bytes")

    def log_lines(self) -> list[str]:
        return ["Claude REVIEW context:"] + [f"{k}={self.stats[k]}" for k in self.LOG_KEYS]


# ------------------------------------------------------------ small helpers

def utf8_len(s: str) -> int:
    return len(s.encode("utf-8"))


def _cut(s: str, max_bytes: int) -> str:
    """Truncate to <= max_bytes UTF-8 bytes, preferably at a line boundary."""
    if max_bytes <= 0:
        return ""
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    cut = b[:max_bytes].decode("utf-8", errors="ignore")
    nl = cut.rfind("\n")
    return cut[:nl + 1] if nl > len(cut) // 2 else cut


_ANSI = re.compile(r"\x1b\[[0-9;?]*[ -/]*[@-~]")


def _useful_lines(text: str) -> list[str]:
    out = []
    for line in _ANSI.sub("", text or "").splitlines():
        line = line.rstrip()
        if line.strip():
            out.append(line if len(line) <= 200 else line[:200] + "…")
    return out


# ------------------------------------------------------------ B. worker final result

_LABELS = {"summary": "Summary", "files changed": "Files changed", "decisions": "Decisions",
           "risks": "Risks", "blockers": "Blockers", "verification": "Verification"}
_LABEL_RE = re.compile(r"^[\s>#*_-]*\**\s*(summary|files changed|decisions|risks|blockers|verification)"
                       r"\s*\**\s*:\s*\**\s*(.*)$", re.IGNORECASE)


def parse_worker_result(final_message: str) -> dict[str, str]:
    """Labelled fields of the worker's final message (SUMMARY:, FILES CHANGED:, ...)."""
    fields: dict[str, list[str]] = {}
    current: str | None = None
    for line in (final_message or "").splitlines():
        m = _LABEL_RE.match(line)
        if m:
            current = _LABELS[m.group(1).lower()]
            fields.setdefault(current, [])
            if m.group(2).strip():
                fields[current].append(m.group(2).strip())
        elif current and line.strip():
            fields[current].append(line.strip())
    if not fields:
        text = (final_message or "").strip()
        return {"Final message": text} if text else {}
    return {k: "\n".join(v) for k, v in fields.items() if v}


def _worker_section(a: Attempt, fields: dict[str, str], budget: int) -> str:
    head = f"exit code {a.exit_code}" + (f", {a.elapsed:.0f}s" if a.elapsed is not None else "")
    lines = [f"WORKER FINAL RESULT ({head})"]
    if a.error:
        lines.append(f"Worker error: {_cut(a.error, 400)}")
    if not fields:
        lines.append("(the worker produced no final message)")
    per_field = max(150, budget // max(1, len(fields)))
    for name, value in fields.items():
        lines.append(f"{name}: {_cut(value, per_field)}")
    return _cut("\n".join(lines), budget)


# ------------------------------------------------------------ C. verification

def format_verification(results: list[VerificationResult], ok_lines: int, fail_bytes: int) -> str:
    if not results:
        return "VERIFICATION\n(no verification commands defined for this task)"
    lines = ["VERIFICATION"]
    for v in results:
        cmd = v.command if len(v.command) <= 160 else v.command[:160] + "…"
        dur = f"{v.duration:.1f}s"
        if v.skipped_reason:
            lines.append(f"{cmd}: SKIPPED ({v.skipped_reason})")
            continue
        out = _useful_lines(v.output_tail)
        if v.passed:
            lines.append(f"{cmd}: PASS ({dur})")
            lines += [f"  | {x}" for x in out[-ok_lines:]] if ok_lines > 0 else []
        else:
            status = "TIMEOUT" if v.exit_code is None else f"FAIL (exit {v.exit_code}, {dur})"
            lines.append(f"{cmd}: {status}")
            err_tail = _cut_tail("\n".join(out), fail_bytes)
            lines += [f"  | {x}" for x in err_tail.splitlines()]
    return "\n".join(lines)


def _cut_tail(s: str, max_bytes: int) -> str:
    b = s.encode("utf-8")
    if len(b) <= max_bytes:
        return s
    t = b[-max_bytes:].decode("utf-8", errors="ignore")
    nl = t.find("\n")
    return t[nl + 1:] if 0 <= nl < len(t) // 2 else t


# ------------------------------------------------------------ D/E. git + diff

_DIFF_HEADER = re.compile(r"^diff --git a/(.*?) b/(.*)$")


def split_diff(diff: str) -> dict[str, str]:
    """Per-file sections of a unified git diff, keyed by (new) path."""
    sections: dict[str, str] = {}
    cur: str | None = None
    buf: list[str] = []
    for line in (diff or "").splitlines(keepends=True):
        m = _DIFF_HEADER.match(line.rstrip("\r\n"))
        if m:
            if cur is not None:
                sections[cur] = "".join(buf)
            cur = m.group(2).strip('"')
            buf = [line]
        elif cur is not None:
            buf.append(line)
    if cur is not None:
        sections[cur] = "".join(buf)
    return sections


_SOURCE_EXTS = {".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs", ".py", ".cpp", ".cc", ".c", ".h", ".hpp", ".cs"}


def rank_paths(paths: list[str], task: Task, worker_fields: dict[str, str]) -> list[str]:
    """Most task-relevant first: files the plan named, files mentioned in the task,
    files the worker reported, then source > config/docs > metadata."""
    likely = [p.replace("\\", "/").strip().lstrip("./").lower() for p in task.files_likely_involved if p.strip()]
    task_text = " ".join([task.title, task.description, task.instructions, *task.acceptance_criteria]).lower()
    worker_text = worker_fields.get("Files changed", "").lower()

    def score(p: str) -> int:
        pl = p.lower()
        base = posixpath.basename(pl)
        s = 0
        if any(pl == l or pl.endswith("/" + l) or l.endswith("/" + pl) or posixpath.basename(l) == base
               for l in likely):
            s += 100
        if pl in task_text or (len(base) > 3 and base in task_text):
            s += 40
        if pl in worker_text or (len(base) > 3 and base in worker_text):
            s += 20
        cls = path_filter.classify(p)
        if cls == path_filter.DIFFABLE:
            s += 10 + (5 if posixpath.splitext(pl)[1] in _SOURCE_EXTS else 0)
        return s

    return sorted(paths, key=lambda p: (-score(p), p))


def _filter_stat(diff_stat: str, max_lines: int = 40) -> list[str]:
    out = []
    for line in (diff_stat or "").splitlines():
        if "|" not in line:
            continue  # the "N files changed" summary may include generated paths in old data
        path = line.split("|", 1)[0].strip()
        if path_filter.is_generated(path.replace(".../", "x/")):
            continue
        out.append(line.rstrip())
    return out[:max_lines]


def _assemble_diff(ranked: list[str], sections: dict[str, str], budget: int) -> tuple[str, int, int, list[str]]:
    """Returns (diff text incl. markers, original bytes, included bytes, files with diff included)."""
    relevant = [(p, sections[p]) for p in ranked if p in sections]
    original = sum(utf8_len(s) for _, s in relevant)
    parts: list[str] = []
    used = 0
    included_files: list[str] = []
    omitted: list[str] = []
    for p, sec in relevant:
        size = utf8_len(sec)
        if used + size <= budget:
            parts.append(sec)
            used += size
            included_files.append(p)
        elif budget - used >= 600:
            piece = _cut(sec, budget - used - 100)
            parts.append(piece + f"[... {p}: rest of this file's diff not included]\n")
            used += utf8_len(piece)
            included_files.append(p)
        else:
            omitted.append(p)
    text = "".join(parts)
    notes = []
    if omitted:
        shown = ", ".join(omitted[:15]) + (f", … {len(omitted) - 15} more" if len(omitted) > 15 else "")
        notes.append(f"[diff omitted for: {shown}]")
    if used < original:
        notes.append(f"[DIFF TRUNCATED: original {original} bytes, included {used} bytes]")
    if not relevant:
        notes.append("(no source/config/doc diff for the listed files)")
    return (text + ("\n" if text and not text.endswith("\n") else "") + "\n".join(notes)).rstrip(), original, used, included_files


# ------------------------------------------------------------ builder

def build_review_context(package: ReviewPackage, limits: ReviewContextLimits) -> ReviewContext:
    task, a = package.task, package.attempt
    limit = min(limits.max_prompt_bytes, REVIEW_PROMPT_BYTES_CEILING)

    # Attribution, defensively re-filtered (also covers attempts stored before this filter).
    gen_by_dir: Counter = Counter(a.generated_by_dir or {})
    authored: dict[str, str] = {}
    for c in a.changed_files:
        if path_filter.is_generated(c.path):
            gen_by_dir[path_filter.generated_dir(c.path) or "generated files"] += 1
        else:
            authored[c.path] = c.change
    generated_total = sum(gen_by_dir.values())
    preexisting = [p for p in a.preexisting_dirty if not path_filter.is_generated(p)]

    fields = parse_worker_result(a.worker_summary)
    ranked = rank_paths(list(authored), task, fields)
    listed = ranked[:limits.max_changed_files]
    diffable = [p for p in listed if path_filter.classify(p) == path_filter.DIFFABLE]
    sections = split_diff(a.diff)

    mandatory = _mandatory_task_section(task, a, package.max_attempts)
    decide = _DECIDE

    def compose(opt: _Optional, diff_budget: int) -> tuple[str, dict[str, int]]:
        worker = _worker_section(a, fields, opt.worker_bytes)
        verification = format_verification(a.verification, opt.ok_lines, opt.fail_bytes)
        git = _git_section(authored, listed, ranked, generated_total, gen_by_dir, preexisting,
                           a.diff_stat, opt.stat_bytes)
        diff_text, original, used, with_diff = _assemble_diff(diffable, sections, diff_budget)
        prompt = "\n\n".join([mandatory, worker, verification, git, "RELEVANT DIFF\n" + diff_text, decide]) + "\n"
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
            return ReviewContext(prompt, stats)
    raise ReviewContextTooLarge(stats.get("final_prompt_bytes", utf8_len(prompt)), limit)


def _mandatory_task_section(task: Task, a: Attempt, max_attempts: int) -> str:
    lines = ["TASK", f"{task.id} - {task.title} (attempt {a.number} of {max_attempts}, {a.kind})"]
    if task.description.strip():
        lines += ["", "DESCRIPTION", task.description.strip()]
    if task.instructions.strip():
        lines += ["", "INSTRUCTIONS", task.instructions.strip()]
    if task.user_guidance.strip():
        lines += ["", "USER GUIDANCE", task.user_guidance.strip()]
    lines += ["", "ACCEPTANCE CRITERIA"]
    lines += [f"{i}. {c}" for i, c in enumerate(task.acceptance_criteria, 1)] or ["(none given)"]
    return "\n".join(lines)


def _git_section(authored: dict[str, str], listed: list[str], ranked: list[str], generated_total: int,
                 gen_by_dir: Counter, preexisting: list[str], diff_stat: str, stat_bytes: int) -> str:
    summary = f"{len(authored)} authored file(s) changed by this attempt"
    if generated_total:
        top = ", ".join(f"{d}: {n}" for d, n in gen_by_dir.most_common(6))
        summary += f"; {generated_total} generated/dependency path(s) excluded ({top})"
    if preexisting:
        summary += f"; {len(preexisting)} pre-existing user change(s) not attributed to the worker"
    lines = ["GIT", summary]
    stat = _filter_stat(diff_stat)
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
Return the structured review result: PASS / RETRY / BLOCKED.
The evidence above is what the orchestrator collected for this attempt (this session has no tools).
- PASS only if every acceptance criterion is met by the changes and verification shown.
- RETRY if the work is incomplete or wrong but fixable by the worker; failed verification normally
  means RETRY. `corrective_instructions` are sent to the worker verbatim - make them specific.
- BLOCKED if progress needs a user decision, missing information or an environment fix.
- If the evidence is genuinely insufficient to judge a criterion, name it in `problems` rather than
  guessing.
- Set `plan_changes_required` only if later tasks must change.
- On PASS fill `task_result` compactly (result, important_decisions, remaining_risks)."""


def summarize_previous_attempts(attempts: list[Attempt]) -> list[dict[str, Any]]:
    """Compact history kept on the package (not rendered into the review prompt)."""
    out = []
    for a in attempts:
        out.append({
            "number": a.number,
            "exit_code": a.exit_code,
            "interrupted": a.interrupted,
            "summary": tail(a.worker_summary, 800),
            "files_changed": a.worker_files()[:30],
            "review_decision": a.review.decision.value if a.review else None,
            "review_problems": a.review.problems if a.review else [],
            "corrective_instructions": a.review.corrective_instructions if a.review else [],
        })
    return out
