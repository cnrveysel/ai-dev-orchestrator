"""One worker attempt at a task, plus everything collected about it."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

from app.models.review import Review


@dataclass
class VerificationResult:
    command: str
    exit_code: int | None
    output_tail: str = ""
    duration: float = 0.0
    skipped_reason: str = ""  # non-empty when not executed (e.g. needs approval)

    @property
    def passed(self) -> bool:
        return not self.skipped_reason and self.exit_code == 0

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["passed"] = self.passed
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "VerificationResult":
        return cls(command=d.get("command", ""), exit_code=d.get("exit_code"),
                   output_tail=d.get("output_tail", ""), duration=d.get("duration", 0.0),
                   skipped_reason=d.get("skipped_reason", ""))


@dataclass
class ChangedFile:
    path: str
    change: str  # A / M / D / R ...
    attributed_to_worker: bool
    note: str = ""


@dataclass
class Attempt:
    task_id: str
    number: int
    kind: str = "execute"  # execute | fix
    started_at: float = 0.0
    finished_at: float | None = None
    exit_code: int | None = None
    interrupted: bool = False
    executable: str = ""
    command_line: list[str] = field(default_factory=list)
    worker_summary: str = ""
    stdout_tail: str = ""
    stderr_tail: str = ""
    tool_calls: int = 0
    tool_call_log: list[str] = field(default_factory=list)
    worker_usage: dict[str, Any] = field(default_factory=dict)
    changed_files: list[ChangedFile] = field(default_factory=list)
    preexisting_dirty: list[str] = field(default_factory=list)
    generated_excluded: int = 0  # generated/dependency paths changed but not attributed
    generated_by_dir: dict[str, int] = field(default_factory=dict)
    diff_stat: str = ""
    diff: str = ""
    verification: list[VerificationResult] = field(default_factory=list)
    review: Review | None = None
    error: str = ""
    db_id: int | None = None
    # --- V2 (phase attempts); defaults keep V1 attempts unchanged ---
    turns: list[dict[str, Any]] = field(default_factory=list)  # one entry per Qwen turn in this attempt
    session_id: str = ""  # Qwen/Codex session used by the last turn
    diff_scope: str = "attempt"  # "phase" when changed_files/diff cover the whole phase so far
    verified_at_turn: int = -1  # len(turns) when `verification` was last produced
    local_fix_rounds: int = 0
    # --- V4 pass 2: the evidence bundle collected for this attempt (compact; outputs are artifact files) ---
    evidence: dict[str, Any] = field(default_factory=dict)

    @property
    def elapsed(self) -> float | None:
        if self.finished_at is None:
            return None
        return self.finished_at - self.started_at

    def worker_files(self) -> list[str]:
        return [c.path for c in self.changed_files if c.attributed_to_worker]

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["verification"] = [v.to_dict() for v in self.verification]
        d["review"] = self.review.to_dict() if self.review else None
        return d
