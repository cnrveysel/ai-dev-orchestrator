"""Application settings, stored as JSON under %APPDATA%\\AIDevOrchestrator.

No secrets are stored here: Claude uses the user's existing Claude Code auth,
and the worker uses its own CODEX_HOME profile.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict, fields, is_dataclass
from pathlib import Path
from typing import Any

APP_NAME = "AIDevOrchestrator"


def app_config_dir() -> Path:
    base = os.environ.get("APPDATA") or str(Path.home() / ".config")
    return Path(base) / APP_NAME


def app_data_dir() -> Path:
    base = os.environ.get("LOCALAPPDATA") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def _same_path(a: str, b: str) -> bool:
    if not a or not b:
        return False
    norm = lambda p: os.path.normcase(os.path.abspath(os.path.expandvars(p)))  # noqa: E731
    return norm(a) == norm(b)


def default_codex_home() -> str:
    return str(Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".codex-ninfer")


@dataclass
class ClaudeSettings:
    executable: str = ""
    model: str = "opus"  # alias accepted by `claude --model`; empty = CLI default
    effort: str = ""  # low|medium|high|xhigh|max; empty = CLI default
    extra_args: list[str] = field(default_factory=list)
    allow_repo_read_tools: bool = True  # planner only (Read/Glob/Grep); the reviewer never gets tools
    verified_executable: str = ""  # set by a successful smoke test
    verified_at: float = 0.0
    # Reviewer model (runs after every task); empty = same as `model`.
    review_model: str = "claude-opus-5-5"
    # --- usage guards (per call) ---
    # Hard wall-clock limits; on expiry Claude is terminated, then killed.
    # Planning (PLAN/REPLAN, V1 and V2) inspects the repository and may repair a structured-output
    # shape error, which did not fit in 120 s on a real V2 run.
    plan_timeout_seconds: int = 240
    review_timeout_seconds: int = 60
    smoke_timeout_seconds: int = 45
    # The installed CLI has no --max-turns flag, so turns/tool calls are counted from the
    # stream and Claude is terminated when a limit is exceeded. The reviewer has no tools.
    plan_max_turns: int = 6
    review_max_turns: int = 2
    plan_max_tool_calls: int = 8
    # Passed as --max-budget-usd (supported by the installed CLI); 0 = not passed.
    plan_max_budget_usd: float = 0.25
    review_max_budget_usd: float = 0.20
    # Reviewer context hard limits (UTF-8 bytes / files). The prompt limit can never be
    # configured above reviewer.REVIEW_PROMPT_BYTES_CEILING.
    review_max_prompt_bytes: int = 40000
    review_max_diff_bytes: int = 25000
    review_max_changed_files: int = 30
    # Automatic retries of a failed plan/review call: 0 (default) or 1. Never more.
    auto_retries: int = 0
    # --- Pipeline V2 phase checkpoints (one Claude review per phase; uses the reviewer model) ---
    checkpoint_timeout_seconds: int = 240
    checkpoint_max_budget_usd: float = 0.60
    # Hard limits for the checkpoint context; the prompt limit is also capped by
    # reviewer.REVIEW_PROMPT_BYTES_CEILING.
    checkpoint_max_prompt_bytes: int = 60000
    checkpoint_max_diff_bytes: int = 40000
    checkpoint_max_changed_files: int = 60
    # --- Pipeline V2 repair escalation (DIAGNOSE_BLOCKER): cheap advice, no tools, bounded packet ---
    diagnose_timeout_seconds: int = 180
    diagnose_max_budget_usd: float = 0.30
    diagnose_max_prompt_bytes: int = 30000
    diagnose_max_diff_bytes: int = 12000
    # --- V4 HUMAN_FOLLOWUP (Claude adjudicates the user's answer to ASK_USER): no tools, bounded packet ---
    human_followup_timeout_seconds: int = 180
    human_followup_max_budget_usd: float = 0.40
    human_followup_max_prompt_bytes: int = 40000
    human_followup_max_diff_bytes: int = 16000
    # --- V4 EVIDENCE_PLAN (Claude plans evidence with the discovered capabilities): no tools, bounded ---
    evidence_plan_timeout_seconds: int = 180
    evidence_plan_max_budget_usd: float = 0.30
    evidence_plan_max_prompt_bytes: int = 30000
    # --- V4 pass 3 CONSISTENCY_REVIEW (a PASS that contradicts its own evidence judgment): no tools, bounded ---
    consistency_timeout_seconds: int = 180
    consistency_max_turns: int = 2
    consistency_max_budget_usd: float = 0.30
    consistency_max_prompt_bytes: int = 24000
    # --- V4 pass 3 MID_PHASE_DIAGNOSE (the implementation guard stopped a worker session): no tools, bounded ---
    mid_phase_timeout_seconds: int = 180
    mid_phase_max_budget_usd: float = 0.30
    mid_phase_max_prompt_bytes: int = 16000


@dataclass
class WorkerSettings:
    executable: str = ""
    codex_home: str = field(default_factory=default_codex_home)
    extra_env: dict[str, str] = field(default_factory=lambda: {"RUST_MIN_STACK": "33554432"})
    model: str = "qwen3.8-27b"
    pass_model_flag: bool = True
    sandbox: str = "workspace-write"  # read-only | workspace-write | danger-full-access
    timeout_seconds: int = 3600
    extra_args: list[str] = field(default_factory=list)
    ninfer_base_url: str = "http://127.0.0.1:8080/v1"
    # Recorded by a passing worker smoke test; REAL worker mode requires both to
    # match the current executable and CODEX_HOME.
    verified_executable: str = ""
    verified_codex_home: str = ""
    verified_at: float = 0.0

    def verified_for(self, executable: str, codex_home: str) -> bool:
        return (_same_path(self.verified_executable, executable)
                and _same_path(self.verified_codex_home, codex_home))

    def is_verified(self) -> bool:
        return self.verified_for(self.executable, self.codex_home)


@dataclass
class IntegrationSettings:
    codebase_memory_executable: str = ""
    metrics_url: str = "http://127.0.0.1:8080/metrics"
    metrics_enabled: bool = False
    metrics_interval_seconds: float = 2.0
    unreal_mcp_url: str = "http://127.0.0.1:8003/mcp"
    # V2: if the repository root is not in codebase-memory yet, index it (by its exact absolute
    # path) before the first phase so the worker does not fall back to broad file reads.
    auto_index_codebase_memory: bool = True
    codebase_memory_index_timeout_seconds: int = 900


@dataclass
class OrchestrationSettings:
    orchestrator_mode: str = "mock"  # mock | claude
    worker_mode: str = "mock"  # mock | ninfer
    max_attempts: int = 3
    run_verification_commands: bool = True
    verification_timeout_seconds: int = 900
    create_working_branch: bool = True
    working_branch_prefix: str = "ai-orchestrator/run-"
    auto_commit: bool = False  # OFF by default, per spec
    precise_git_snapshots: bool = True
    mock_step_delay: float = 0.6
    mock_retry_task_ids: list[str] = field(default_factory=lambda: ["T002", "P2"])
    # --- Pipeline selection: "v1" task-by-task (default) | "v2" phases (applies to new PLANs) ---
    pipeline_version: str = "v1"
    # V2: Qwen turns per phase attempt (the first turn plus continuations in the SAME session
    # when the worker reports the phase INCOMPLETE).
    v2_max_turns_per_attempt: int = 4
    # V2: when verification fails after the work turns, Qwen fixes it in the same session this
    # many times before the Claude checkpoint (0 = hand failures straight to the checkpoint).
    v2_local_fix_rounds: int = 1
    # V2: after this many turns in one Qwen session, continue in a fresh session with a compact
    # handoff (bounds context growth). Sessions are never shared across phases.
    v2_max_session_turns: int = 8
    # V2: REPLAN checkpoints allowed per run before the run blocks for the user.
    v2_max_replans: int = 3
    # V2 REPAIRING (after a checkpoint RETRY): a fresh, bounded Qwen repair session. A repair
    # session is stopped as soon as ANY limit is hit, then the blocker is escalated to Claude.
    repair_max_wall_seconds: int = 1200  # wall-clock time of one repair session
    repair_max_tool_calls: int = 30  # worker tool calls in one repair session
    repair_max_same_failure_rounds: int = 3  # verification rounds with an unchanged failure fingerprint
    # rounds without a score strictly better than best-known (a changed fingerprint is not progress)
    repair_max_no_improvement_rounds: int = 4
    repair_regression_grace_rounds: int = 1  # rounds allowed worse than best-known before it is restored
    repair_max_new_probe_files: int = 3  # new scratch/probe files without verification progress
    repair_max_diagnoses: int = 2  # Claude DIAGNOSE_BLOCKER calls per repair attempt
    # V4: "Allow Claude to request human verification". A checkpoint may answer ASK_USER only when this
    # is on AND the phase's verification contract opens its human gate; otherwise Claude decides from the
    # available evidence. Phases without a contract are never affected.
    human_verification_enabled: bool = True
    # V4 pass 2: evidence collection for phases whose verification contract lists evidence requirements
    # (capability discovery -> Claude EVIDENCE_PLAN -> bounded execution -> evidence bundle for the checkpoint).
    # Phases without such a contract never collect evidence.
    evidence_enabled: bool = True
    evidence_max_wall_seconds: int = 600  # all evidence steps of one phase attempt
    evidence_step_timeout_seconds: int = 180  # one step (a capability may declare less)
    evidence_max_steps: int = 12  # steps per phase attempt
    evidence_max_output_bytes: int = 16000  # output captured per step (tail kept)
    evidence_max_artifacts: int = 24  # artifact files per phase attempt
    evidence_max_artifact_bytes: int = 1_000_000  # one artifact file
    evidence_max_retries: int = 0  # re-runs of a step that timed out or errored (never of a failure)
    evidence_service_ready_timeout_seconds: int = 60  # a declared service must become ready within this
    evidence_discovery_timeout_seconds: int = 20  # capability discovery, all providers together
    evidence_max_capabilities: int = 40
    # Explicitly configured evidence commands (one command line per entry; offered to the evidence planner).
    evidence_commands: list[str] = field(default_factory=list)
    # --- V4 pass 3 ---
    # A checkpoint PASS that contradicts its own evidence judgment gets a bounded Claude CONSISTENCY_REVIEW.
    consistency_review_enabled: bool = True
    # Bounded re-planning when an evidence STRATEGY cannot be executed (never for an ordinary failed check).
    evidence_max_replans: int = 1
    # Non-git projects: bounded manifest of the files evidence steps might change (exact snapshot within bytes).
    evidence_guard_max_files: int = 5000
    evidence_guard_max_snapshot_bytes: int = 64_000_000
    # Artifact retention under AI Studio's data folder (artifacts of runs that can still resume are kept).
    evidence_artifact_cleanup_enabled: bool = True
    evidence_artifact_max_age_days: int = 14
    evidence_artifact_max_run_bytes: int = 200_000_000
    evidence_artifact_max_total_bytes: int = 1_000_000_000
    # At most this many questions per human verification request (Claude's questions are deduplicated too).
    human_max_questions: int = 4
    # Provider-specific keys (generic key=value): unreal.engine_root, unreal.test_filter, unreal.ready_marker,
    # unreal.editor_ready_marker, browser.enabled, ... Unknown keys are ignored.
    evidence_provider_settings: dict[str, str] = field(default_factory=dict)
    # Normal implementation guard (WORKER_RUNNING, before a checkpoint): a limit <= 0 disables it. On a trip
    # Claude MID_PHASE_DIAGNOSE guides a fresh worker session; after the maximum diagnoses the run blocks.
    implementation_guard_enabled: bool = True
    implementation_max_wall_seconds: int = 2400
    implementation_max_tool_calls: int = 50
    implementation_max_no_subtask_progress_seconds: int = 1200
    implementation_max_diagnoses: int = 2


LEGACY_WORKER_PROFILE_ID = "legacy-ninfer-qwen"


@dataclass
class WorkerProfilesSettings:
    """V5 worker profiles registered with AI Studio (WorkerProfile.to_dict() records; see app.workers.profiles)
    and the profile NEW runs use. `worker` above is always available as the virtual profile
    LEGACY_WORKER_PROFILE_ID, so an installation without registered profiles behaves exactly like V4.
    No secret values are stored: a profile can only name environment variables that must exist."""
    default_profile_id: str = LEGACY_WORKER_PROFILE_ID
    profiles: list[dict[str, Any]] = field(default_factory=list, metadata={"records": True})
    # Parent folder for CODEX_HOMEs of profiles AI Studio creates (managed profiles); empty = ~/.codex-profiles.
    managed_root: str = ""


@dataclass
class Settings:
    claude: ClaudeSettings = field(default_factory=ClaudeSettings)
    worker: WorkerSettings = field(default_factory=WorkerSettings)
    integrations: IntegrationSettings = field(default_factory=IntegrationSettings)
    orchestration: OrchestrationSettings = field(default_factory=OrchestrationSettings)
    worker_profiles: WorkerProfilesSettings = field(default_factory=WorkerProfilesSettings)
    last_repo_path: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Settings":
        return _fill(cls(), d)


def _fill(obj: Any, data: dict[str, Any]) -> Any:
    """Apply known keys from data onto a dataclass instance (ignores unknown keys)."""
    if not isinstance(data, dict):
        return obj
    for f in fields(obj):
        if f.name not in data:
            continue
        cur = getattr(obj, f.name)
        val = data[f.name]
        if is_dataclass(cur):
            _fill(cur, val)
        elif isinstance(cur, bool):
            setattr(obj, f.name, bool(val))
        elif isinstance(cur, (int, float)) and not isinstance(cur, bool):
            try:
                setattr(obj, f.name, type(cur)(val))
            except (TypeError, ValueError):
                pass
        elif isinstance(cur, list) and f.metadata.get("records"):
            if isinstance(val, list):  # a list of JSON records (dicts), kept as they are
                setattr(obj, f.name, [dict(x) for x in val if isinstance(x, dict)])
        elif isinstance(cur, list):
            if isinstance(val, list):
                setattr(obj, f.name, [str(x) for x in val])
        elif isinstance(cur, dict):
            if isinstance(val, dict):
                setattr(obj, f.name, {str(k): str(v) for k, v in val.items()})
        else:
            setattr(obj, f.name, "" if val is None else str(val))
    return obj


class SettingsStore:
    def __init__(self, path: Path | str | None = None):
        self.path = Path(path) if path else (app_config_dir() / "settings.json")

    def load(self) -> Settings:
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            return Settings()
        except (OSError, json.JSONDecodeError):
            return Settings()
        return Settings.from_dict(data)

    def save(self, settings: Settings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(settings.to_dict(), indent=2), encoding="utf-8")
        os.replace(tmp, self.path)
