"""Plan model and the JSON schema Claude must satisfy when planning."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

from app.models.task import Task, TaskStatus

TASK_ID_RE = re.compile(r"^[A-Za-z0-9_.-]{1,32}$")


class PlanValidationError(ValueError):
    pass


PLAN_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "goal": {"type": "string"},
        "summary": {"type": "string"},
        "assumptions": {"type": "array", "items": {"type": "string"}},
        "architecture_notes": {"type": "array", "items": {"type": "string"}},
        "tasks": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "instructions": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                    "files_likely_involved": {"type": "array", "items": {"type": "string"}},
                    "acceptance_criteria": {"type": "array", "items": {"type": "string"}},
                    "verification_commands": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["id", "title", "description", "acceptance_criteria"],
            },
        },
    },
    "required": ["goal", "summary", "tasks"],
}


@dataclass
class Plan:
    goal: str
    summary: str = ""
    assumptions: list[str] = field(default_factory=list)
    architecture_notes: list[str] = field(default_factory=list)
    tasks: list[Task] = field(default_factory=list)

    def task(self, task_id: str) -> Task | None:
        return next((t for t in self.tasks if t.id == task_id), None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "summary": self.summary,
            "assumptions": list(self.assumptions),
            "architecture_notes": list(self.architecture_notes),
            "tasks": [t.to_dict() for t in self.tasks],
        }

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "Plan":
        """Parse and validate a plan (from Claude or from persistence)."""
        if not isinstance(d, dict):
            raise PlanValidationError("plan must be a JSON object")
        raw_tasks = d.get("tasks")
        if not isinstance(raw_tasks, list) or not raw_tasks:
            raise PlanValidationError("plan contains no tasks")
        tasks: list[Task] = []
        seen: set[str] = set()
        for i, raw in enumerate(raw_tasks):
            if not isinstance(raw, dict) or "id" not in raw:
                raise PlanValidationError(f"task #{i} is missing an id")
            raw = dict(raw)
            raw.setdefault("status", TaskStatus.PENDING.value)
            try:
                task = Task.from_dict(raw)
            except (ValueError, KeyError) as exc:
                raise PlanValidationError(f"task #{i} invalid: {exc}") from exc
            if not TASK_ID_RE.match(task.id):
                raise PlanValidationError(f"invalid task id {task.id!r}")
            if task.id in seen:
                raise PlanValidationError(f"duplicate task id {task.id}")
            seen.add(task.id)
            task.position = i
            tasks.append(task)
        for t in tasks:
            for dep in t.dependencies:
                if dep not in seen:
                    raise PlanValidationError(f"{t.id} depends on unknown task {dep}")
                if dep == t.id:
                    raise PlanValidationError(f"{t.id} depends on itself")
        plan = cls(
            goal=str(d.get("goal", "")),
            summary=str(d.get("summary", "")),
            assumptions=[str(x) for x in d.get("assumptions", []) or []],
            architecture_notes=[str(x) for x in d.get("architecture_notes", []) or []],
            tasks=tasks,
        )
        # Imported lazily to avoid a cycle; raises PlanValidationError on cycles.
        from app.orchestration.task_graph import TaskGraph
        TaskGraph(plan.tasks).topological_order()
        return plan
