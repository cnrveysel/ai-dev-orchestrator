"""Orchestration run model."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from app.models.plan import Plan


@dataclass
class Run:
    id: int | None
    goal: str
    repo_path: str
    state: str = "IDLE"
    resume_state: str | None = None  # state to re-enter on RESUME
    status_message: str = ""
    created_at: float = 0.0
    updated_at: float = 0.0
    started_at: float | None = None
    finished_at: float | None = None
    interrupted: bool = False
    git_head: str | None = None
    git_branch: str | None = None
    working_branch: str | None = None
    initial_dirty: list[str] = field(default_factory=list)
    current_task_id: str | None = None
    plan: Plan | None = None
    settings_snapshot: dict[str, Any] = field(default_factory=dict)
    orchestrator_mode: str = "mock"
    worker_mode: str = "mock"
    # "v1" = task-by-task (every task reviewed); "v2" = phase-based (checkpoint per phase).
    # Rows written before V2 existed default to "v1" in the database.
    pipeline_version: str = "v1"
    # V2 run-level recovery state (replan bookkeeping, base snapshots, codebase-memory project).
    pipeline_state: dict[str, Any] = field(default_factory=dict)
    # V5: RunWorkerProfiles.to_dict() - the worker-profile snapshot locked before the first worker launch (never
    # changed afterwards; resume uses it, not the current default profile). {} = not locked yet / a V4 run.
    worker_snapshot: dict[str, Any] = field(default_factory=dict)
