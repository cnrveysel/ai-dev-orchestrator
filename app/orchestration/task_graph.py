"""Dependency graph used ONLY to order tasks. Execution is always sequential:
the graph never yields more than one task to run at a time."""
from __future__ import annotations

from app.models.task import Task, TaskStatus, ACTIVE_STATUSES


class TaskGraph:
    def __init__(self, tasks: list[Task]):
        self.tasks = tasks
        self._by_id = {t.id: t for t in tasks}

    def get(self, task_id: str) -> Task | None:
        return self._by_id.get(task_id)

    def topological_order(self) -> list[Task]:
        """Stable topological order (plan order is the tie-breaker)."""
        from app.models.plan import PlanValidationError
        order: list[Task] = []
        state: dict[str, int] = {}  # 1 = visiting, 2 = done

        def visit(t: Task, stack: list[str]) -> None:
            s = state.get(t.id)
            if s == 2:
                return
            if s == 1:
                raise PlanValidationError("dependency cycle: " + " -> ".join(stack + [t.id]))
            state[t.id] = 1
            for dep in t.dependencies:
                dt = self._by_id.get(dep)
                if dt is None:
                    raise PlanValidationError(f"{t.id} depends on unknown task {dep}")
                visit(dt, stack + [t.id])
            state[t.id] = 2
            order.append(t)

        for t in sorted(self.tasks, key=lambda x: x.position):
            visit(t, [])
        return order

    def deps_satisfied(self, task: Task) -> bool:
        return all(self._by_id[d].status == TaskStatus.PASSED for d in task.dependencies)

    def active_task(self) -> Task | None:
        active = [t for t in self.tasks if t.status in ACTIVE_STATUSES or t.status == TaskStatus.RETRY]
        if len(active) > 1:
            raise RuntimeError("invariant violated: more than one active task: "
                               + ", ".join(t.id for t in active))
        return active[0] if active else None

    def next_ready(self) -> Task | None:
        """The single next task to run, or None."""
        for t in self.topological_order():
            if t.status in (TaskStatus.PENDING, TaskStatus.READY, TaskStatus.PAUSED) and self.deps_satisfied(t):
                return t
        return None

    def refresh_ready_flags(self) -> None:
        """PENDING <-> READY display flag. Only one task is ever marked READY."""
        nxt = None if self.active_task() else self.next_ready()
        for t in self.tasks:
            if t.status == TaskStatus.READY and t is not nxt:
                t.status = TaskStatus.PENDING
        if nxt is not None and nxt.status == TaskStatus.PENDING:
            nxt.status = TaskStatus.READY

    def all_passed(self) -> bool:
        return all(t.status == TaskStatus.PASSED for t in self.tasks)

    def dependency_results(self, task: Task) -> list:
        """Compact results of the transitive dependencies of a task."""
        seen: set[str] = set()
        out = []

        def walk(tid: str) -> None:
            for dep in self._by_id[tid].dependencies:
                if dep in seen:
                    continue
                seen.add(dep)
                walk(dep)
                r = self._by_id[dep].result
                if r is not None:
                    out.append(r)

        walk(task.id)
        return out
