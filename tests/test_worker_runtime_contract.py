"""V5.1 milestone 1: the worker contract the pipeline actually uses, pinned before runtimes were added.

The schedulers (V1 task scheduler, V2 phase scheduler incl. repair) talk to a worker ONLY through the
WorkerProvider members listed here, and never mention a runtime, provider or CODEX_HOME. A new worker runtime
(Cline, ...) only has to implement this contract; the schedulers stay runtime-agnostic.
"""
import ast
import inspect
import re
from pathlib import Path

from app.workers.base import WorkerProvider

ROOT = Path(__file__).resolve().parent.parent
SCHEDULERS = ["app/orchestration/scheduler.py", "app/orchestration/phase_scheduler.py", "app/orchestration/repair.py"]
CONTRACT = {"run_turn", "execute_task", "fix_task", "session_resume_supported", "cancel", "on_tool_call", "on_session",
            "on_todo", "on_file_change", "capabilities"}


def _worker_members_used(path: Path) -> set[str]:
    """Attributes accessed on names that hold the worker (`self.d.worker`, `worker`, `w`)."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    used: set[str] = set()
    for node in ast.walk(tree):
        if not isinstance(node, ast.Attribute):
            continue
        v = node.value
        if isinstance(v, ast.Attribute) and v.attr == "worker":  # self.d.worker.X
            used.add(node.attr)
        elif isinstance(v, ast.Name) and v.id in ("worker", "w") and node.attr != "worker":
            used.add(node.attr)
    return used


def test_schedulers_use_only_the_worker_contract():
    used = set()
    for rel in SCHEDULERS:
        used |= _worker_members_used(ROOT / rel)
    unexpected = used - CONTRACT
    assert not unexpected, f"schedulers reach past the WorkerProvider contract: {sorted(unexpected)}"
    assert {"run_turn", "session_resume_supported", "cancel", "on_session", "on_tool_call"} <= used


def test_schedulers_contain_no_runtime_or_provider_specifics():
    pattern = re.compile(r"codex|cline|ninfer|CODEX_HOME|openrouter|lmstudio|runtime\s*==", re.I)
    for rel in SCHEDULERS:
        for n, line in enumerate((ROOT / rel).read_text(encoding="utf-8").splitlines(), 1):
            assert not pattern.search(line), f"{rel}:{n} is runtime-specific: {line.strip()}"


def test_worker_provider_exposes_the_contract():
    members = {name for name, _ in inspect.getmembers(WorkerProvider)}
    for name in ("run_turn", "execute_task", "fix_task", "session_resume_supported", "cancel"):
        assert name in members
    w = type("W", (WorkerProvider,), {"execute_task": lambda self, s: None, "cancel": lambda self: None})()
    for cb in ("on_tool_call", "on_session", "on_todo", "on_file_change"):
        assert getattr(w, cb) is None, "callbacks are wired by the scheduler, per turn"
