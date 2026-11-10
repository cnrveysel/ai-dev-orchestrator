"""Task model: one focused unit of work executed by the single worker."""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any


class TaskStatus(str, Enum):
    PENDING = "PENDING"
    READY = "READY"
    RUNNING = "RUNNING"
    VERIFYING = "VERIFYING"
    REVIEWING = "REVIEWING"
    RETRY = "RETRY"
    PASSED = "PASSED"
    FAILED = "FAILED"
    BLOCKED = "BLOCKED"
    PAUSED = "PAUSED"
    SUPERSEDED = "SUPERSEDED"  # V2 only: phase closed by a REPLAN checkpoint (its work is kept)
    AWAITING_HUMAN = "AWAITING_HUMAN"  # V4 only: the checkpoint asked the user to verify the phase


# Statuses meaning "this task currently owns the worker slot / is in flight".
ACTIVE_STATUSES = {TaskStatus.RUNNING, TaskStatus.VERIFYING, TaskStatus.REVIEWING}


@dataclass
class TaskResult:
    """Compact structured result stored after PASS and fed to later tasks."""
    task_id: str
    result: str = ""
    files_changed: list[str] = field(default_factory=list)
    important_decisions: list[str] = field(default_factory=list)
    remaining_risks: list[str] = field(default_factory=list)
    verification: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "TaskResult":
        return cls(
            task_id=str(d.get("task_id", "")),
            result=str(d.get("result", "")),
            files_changed=list(d.get("files_changed", []) or []),
            important_decisions=list(d.get("important_decisions", []) or []),
            remaining_risks=list(d.get("remaining_risks", []) or []),
            verification=list(d.get("verification", []) or []),
        )


@dataclass
class Task:
    id: str
    title: str
    description: str = ""
    dependencies: list[str] = field(default_factory=list)
    files_likely_involved: list[str] = field(default_factory=list)
    acceptance_criteria: list[str] = field(default_factory=list)
    verification_commands: list[str] = field(default_factory=list)
    instructions: str = ""  # Claude's implementation guidance for the worker
    user_guidance: str = ""  # provided by the user when resuming a BLOCKED task
    status: TaskStatus = TaskStatus.PENDING
    attempts: int = 0
    extra_attempts: int = 0  # granted by the user after the retry limit was hit
    started_at: float | None = None
    finished_at: float | None = None
    result: TaskResult | None = None
    position: int = 0

    @property
    def duration(self) -> float | None:
        if self.started_at is None:
            return None
        import time
        end = self.finished_at if self.finished_at is not None else time.time()
        return max(0.0, end - self.started_at)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["status"] = self.status.value
        d["result"] = self.result.to_dict() if self.result else None
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Task":
        result = d.get("result")
        return cls(
            id=str(d["id"]),
            title=str(d.get("title", d["id"])),
            description=str(d.get("description", "")),
            dependencies=[str(x) for x in d.get("dependencies", []) or []],
            files_likely_involved=[str(x) for x in d.get("files_likely_involved", []) or []],
            acceptance_criteria=[str(x) for x in d.get("acceptance_criteria", []) or []],
            verification_commands=[str(x) for x in d.get("verification_commands", []) or []],
            instructions=str(d.get("instructions", "") or ""),
            user_guidance=str(d.get("user_guidance", "") or ""),
            status=TaskStatus(d.get("status", "PENDING")),
            attempts=int(d.get("attempts", 0) or 0),
            extra_attempts=int(d.get("extra_attempts", 0) or 0),
            started_at=d.get("started_at"),
            finished_at=d.get("finished_at"),
            result=TaskResult.from_dict(result) if isinstance(result, dict) else None,
            position=int(d.get("position", 0) or 0),
        )
