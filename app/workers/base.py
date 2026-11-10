"""Worker provider abstraction. There is exactly one worker at a time; the only
real implementation is CodexNInferWorker (local Qwen via Codex NInfer)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.plan import Plan
from app.models.task import Task, TaskResult

LogFn = Callable[[str, str], None]  # (source, text)


class WorkerError(RuntimeError):
    """The worker could not be launched at all (misconfiguration). Never
    triggers a fallback to another executable/provider."""


@dataclass
class RetryContext:
    attempt_number: int
    max_attempts: int
    previous_summary: str
    problems: list[str]
    corrective_instructions: list[str]


@dataclass
class WorkerTaskSpec:
    task: Task
    repo_path: str
    goal: str = ""
    plan_summary: str = ""
    architecture_notes: list[str] = field(default_factory=list)
    dependency_results: list[TaskResult] = field(default_factory=list)
    retry: RetryContext | None = None
    codebase_memory_available: bool = False
    unreal_project: bool = False
    unreal_mcp_reachable: bool | None = None

    @classmethod
    def from_plan(cls, plan: Plan, task: Task, repo_path: str, **kw: Any) -> "WorkerTaskSpec":
        return cls(task=task, repo_path=repo_path, goal=plan.goal, plan_summary=plan.summary,
                   architecture_notes=list(plan.architecture_notes), **kw)


@dataclass
class PhaseTurnRequest:
    """V2: one Qwen turn inside a phase. An empty session_id starts a NEW session; otherwise
    the existing session is resumed so the worker keeps its context (no rediscovery)."""
    prompt: str
    repo_path: str
    session_id: str = ""
    phase_id: str = ""
    purpose: str = "start"  # start | continue | retry | resume | verify_fix | repair | repair_round | repair_resume


@dataclass
class WorkerOutcome:
    executable: str
    argv: list[str]
    exit_code: int | None
    summary: str
    stdout_tail: str
    stderr_tail: str
    duration: float
    tool_calls: int = 0
    tool_call_log: list[str] = field(default_factory=list)
    usage: dict[str, Any] = field(default_factory=dict)
    timed_out: bool = False
    cancelled: bool = False
    error: str = ""
    degraded_notes: list[str] = field(default_factory=list)
    env_overrides: dict[str, str] = field(default_factory=dict)
    # --- V2 session information ---
    session_id: str = ""  # Codex thread id reported by the worker
    resumed: bool = False  # this turn continued an existing session
    resume_failed: bool = False  # the session could not be resumed (nothing was done)
    context_exhausted: bool = False  # the worker reported running out of context
    todo: list[dict[str, Any]] = field(default_factory=list)  # last todo list reported in the turn
    # --- V5: the worker profile this call ran with (recorded in worker_calls; "" = legacy / mock) ---
    worker_profile_id: str = ""
    worker_profile_revision: str = ""


@dataclass(frozen=True)
class WorkerCapabilities:
    """V5.1: what a worker runtime can report, so guards never fake precision a runtime cannot provide.
    The defaults are the Codex worker's (and the V4 behaviour): exact tool-call counts and structured todo-list
    progress of planned subtasks. Session resume is asked through session_resume_supported() (it may depend on
    the installed CLI)."""
    tool_call_count: bool = True  # exact tool-call events: tool budgets of the implementation / repair guards
    subtask_progress: bool = True  # structured todo progress: the "no planned-subtask progress" guard limit
    sandbox: bool = True  # the runtime confines the worker's commands (e.g. Codex --sandbox workspace-write)


class WorkerProvider(ABC):
    name = "worker"
    capabilities = WorkerCapabilities()

    def __init__(self, log: LogFn | None = None):
        self.log: LogFn = log or (lambda s, t: None)
        self.on_tool_call: Callable[[int], None] | None = None
        # V2: called as soon as the worker reports its session id / updates its todo list.
        self.on_session: Callable[[str], None] | None = None
        self.on_todo: Callable[[list[dict[str, Any]]], None] | None = None
        # V2 repair: (kind, path) of every structured file change the worker reports (add/update/delete).
        self.on_file_change: Callable[[str, str], None] | None = None
        # V5.1: optional normalized events (app.workers.events.WorkerEvent) for diagnostics; unused by the schedulers.
        self.on_event: Callable[[Any], None] | None = None

    @abstractmethod
    async def execute_task(self, spec: WorkerTaskSpec) -> WorkerOutcome: ...

    async def session_resume_supported(self) -> bool:
        """V2: whether run_turn() can continue an existing session."""
        return False

    async def run_turn(self, request: PhaseTurnRequest) -> WorkerOutcome:
        """V2: run one turn (new or resumed session)."""
        raise WorkerError(f"{self.name} does not support phase-based (V2) execution")

    async def fix_task(self, spec: WorkerTaskSpec) -> WorkerOutcome:
        if spec.retry is None:
            raise ValueError("fix_task requires retry context")
        return await self.execute_task(spec)

    @abstractmethod
    async def cancel(self) -> None: ...


# Shared by the V1 task prompt and the V2 phase prompt.
CODEBASE_MEMORY_LINES = [
    "- Prefer the codebase-memory MCP tools (search_code, search_graph, get_architecture, trace_path,",
    "  get_file_outline, get_code_snippet, query_graph) to locate systems and analyse impact instead",
    "  of recursively reading the repository. Explicit CALLS edges are strong evidence; REFERENCE /",
    "  USAGE / semantic links are weaker. Open the source file when you need exact details, when graph",
    "  coverage has gaps, when Unreal macros obscure behaviour, or for async/callback behaviour.",
    "- For codebase-memory, NEVER guess the project name from the repository folder name.",
    "- First call list_projects and match the current repository by its exact absolute root path.",
    "- If the current repository root is not indexed, call index_repository with the absolute repository path.",
    "- After indexing, use the exact project name returned by codebase-memory for all subsequent queries.",
    "- Never use or query a codebase-memory index belonging to another repository.",
]
CODEBASE_MEMORY_DEGRADED_LINE = "- codebase-memory MCP is not available (degraded mode): use normal repository tools."

COMMON_RULE_LINES = [
    "- Stay inside the repository. Do not modify files outside it.",
    "- On Windows, prefer direct native commands instead of wrapping commands in PowerShell when possible.",
    "- Use rg for repository text/file searches; do not recursively scan node_modules or dist.",
    "- Prefer apply_patch for editing existing files.",
    "- If a patch or text anchor fails once, reread the exact target region before retrying; never retry the same failed patch unchanged.",
    "- When PowerShell is necessary, prefer pwsh.exe -NoProfile -NonInteractive over Windows PowerShell 5.1.",
    "- If forced to use Windows PowerShell 5.1, NEVER use Set-Content -Encoding UTF8NoBOM.",
    "- For BOM-free UTF-8 writes use [System.IO.File]::WriteAllText(path, content, (New-Object System.Text.UTF8Encoding($false))).",
    "- Before creating a new file, ensure its parent directory exists.",
    "- If node/esbuild/test execution times out, check for and terminate leftover child processes before retrying.",
]


def unreal_lines(unreal_project: bool, reachable: bool | None) -> list[str]:
    if not unreal_project:
        return []
    if reachable:
        return ["- This is an Unreal project; the Unreal MCP server is available for editor/Blueprint/asset work."]
    return ["- This is an Unreal project but the Unreal MCP server is NOT reachable; work with source files."]


def build_worker_prompt(spec: WorkerTaskSpec) -> str:
    t = spec.task
    lines: list[str] = [
        "You are the implementation worker for ONE task of a larger plan.",
        "A technical lead owns the global plan and will review your work after you finish.",
        "Work ONLY on the task below. Do not start other tasks and do not redesign the plan.",
        "",
        f"REPOSITORY (your workspace root): {spec.repo_path}",
    ]
    if spec.goal:
        lines += ["", "OVERALL PROJECT GOAL (context only):", spec.goal]
    if spec.architecture_notes:
        lines += ["", "ARCHITECTURE NOTES FROM THE LEAD:"] + [f"- {n}" for n in spec.architecture_notes]
    if spec.dependency_results:
        lines += ["", "RESULTS OF PREREQUISITE TASKS (compact):"]
        for r in spec.dependency_results:
            lines.append(f"- {r.task_id}: {r.result}")
            if r.files_changed:
                lines.append(f"  files: {', '.join(r.files_changed[:15])}")
            for d in r.important_decisions[:5]:
                lines.append(f"  decision: {d}")
    lines += ["", f"CURRENT TASK {t.id}: {t.title}", "", "DESCRIPTION:", t.description or "(none)"]
    if t.instructions:
        lines += ["", "LEAD'S INSTRUCTIONS:", t.instructions]
    if t.acceptance_criteria:
        lines += ["", "ACCEPTANCE CRITERIA:"] + [f"- {c}" for c in t.acceptance_criteria]
    if t.files_likely_involved:
        lines += ["", "FILES LIKELY INVOLVED:"] + [f"- {f}" for f in t.files_likely_involved]
    if t.verification_commands:
        lines += ["", "AFTER YOU FINISH THE SYSTEM WILL RUN:"] + [f"- {c}" for c in t.verification_commands]
    if t.user_guidance:
        lines += ["", "GUIDANCE FROM THE USER:", t.user_guidance]
    if spec.retry:
        r = spec.retry
        lines += ["", f"THIS IS ATTEMPT {r.attempt_number} OF {r.max_attempts}. "
                      "The previous attempt was reviewed and REJECTED. The repository already contains "
                      "the previous attempt's changes; fix them rather than starting over.",
                  "", "PREVIOUS ATTEMPT SUMMARY:", r.previous_summary or "(none)"]
        if r.problems:
            lines += ["", "PROBLEMS FOUND BY THE REVIEWER:"] + [f"- {p}" for p in r.problems]
        if r.corrective_instructions:
            lines += ["", "CORRECTIVE INSTRUCTIONS (follow these):"] + [f"- {c}" for c in r.corrective_instructions]
    lines += ["", "TOOLS:"]
    if spec.codebase_memory_available:
        lines += CODEBASE_MEMORY_LINES
    else:
        lines.append(CODEBASE_MEMORY_DEGRADED_LINE)
    lines += unreal_lines(spec.unreal_project, spec.unreal_mcp_reachable)
    lines += [
        "", "RULES:",
        *COMMON_RULE_LINES,
        "- No destructive commands: no git reset/clean/push/rebase/checkout of other branches, no deleting "
        "files or directories unrelated to this task. Do not commit.",
        "- Build/test when appropriate for the task.",
        "", "WHEN DONE, finish with a final message in this format:",
        "SUMMARY: <what you did>",
        "FILES CHANGED: <list>",
        "DECISIONS: <important decisions>",
        "RISKS: <remaining risks / anything unfinished>",
        "VERIFICATION: <what you built/tested and the result>",
    ]
    return "\n".join(lines)
