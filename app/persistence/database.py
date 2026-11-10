"""SQLite persistence. One Database object per thread (sqlite connections are
not shared across threads). WAL mode lets the GUI read while the engine writes."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from app.models.attempt import Attempt, ChangedFile, VerificationResult
from app.models.phase import Phase
from app.models.plan import Plan
from app.models.review import Review
from app.models.run import Run
from app.models.task import Task
from app.persistence.migrations import migrate
from app.workers.profiles import redact_env

# Run states that mean "something was in progress" - after a restart such runs
# are marked interrupted and can be resumed from the recorded state.
_IN_PROGRESS = {"STARTING_TASK", "WORKER_RUNNING", "VERIFYING", "REVIEWING",
                "RETRY_PENDING", "ADVANCING", "REPLANNING", "REPAIRING", "COLLECTING_EVIDENCE",
                "CONSISTENCY_REVIEW", "STOPPING"}


def _j(v: Any) -> str:
    return json.dumps(v, ensure_ascii=False)


def attempt_from_dict(d: dict[str, Any]) -> Attempt:
    d = dict(d)
    d["changed_files"] = [ChangedFile(**c) for c in d.get("changed_files", [])]
    d["verification"] = [VerificationResult.from_dict(v) for v in d.get("verification", [])]
    d["review"] = Review.from_dict(d["review"]) if d.get("review") else None
    known = set(Attempt.__dataclass_fields__)
    return Attempt(**{k: v for k, v in d.items() if k in known})


class Database:
    def __init__(self, path: Path | str):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.path), timeout=30)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")
        self._lock = threading.Lock()
        migrate(self.conn)

    def close(self) -> None:
        self.conn.close()

    def commit(self) -> None:
        with self._lock:
            self.conn.commit()

    def _exec(self, sql: str, params: tuple = (), commit: bool = True) -> sqlite3.Cursor:
        with self._lock:
            cur = self.conn.execute(sql, params)
            if commit:
                self.conn.commit()
            return cur

    # ------------------------------------------------------------ runs

    def create_run(self, run: Run) -> int:
        now = time.time()
        run.created_at = run.created_at or now
        run.updated_at = now
        cur = self._exec(
            "INSERT INTO runs (goal, repo_path, state, created_at, updated_at, orchestrator_mode, worker_mode, "
            "settings_snapshot_json, pipeline_version, pipeline_state_json) VALUES (?,?,?,?,?,?,?,?,?,?)",
            (run.goal, run.repo_path, run.state, run.created_at, run.updated_at,
             run.orchestrator_mode, run.worker_mode, _j(run.settings_snapshot), run.pipeline_version,
             _j(run.pipeline_state)))
        run.id = int(cur.lastrowid)
        return run.id

    def update_run(self, run: Run) -> None:
        run.updated_at = time.time()
        plan_meta = None
        if run.plan is not None:
            plan_meta = {k: v for k, v in run.plan.to_dict().items() if k != "tasks"}
        self._exec(
            "UPDATE runs SET goal=?, state=?, resume_state=?, status_message=?, updated_at=?, started_at=?, "
            "finished_at=?, interrupted=?, git_head=?, git_branch=?, working_branch=?, initial_dirty_json=?, "
            "current_task_id=?, plan_json=?, orchestrator_mode=?, worker_mode=?, pipeline_state_json=? WHERE id=?",
            (run.goal, run.state, run.resume_state, run.status_message, run.updated_at, run.started_at,
             run.finished_at, int(run.interrupted), run.git_head, run.git_branch, run.working_branch,
             _j(run.initial_dirty), run.current_task_id, _j(plan_meta) if plan_meta else None,
             run.orchestrator_mode, run.worker_mode, _j(run.pipeline_state), run.id))

    def list_runs(self, limit: int = 50) -> list[dict[str, Any]]:
        rows = self._exec(
            "SELECT r.id, r.goal, r.repo_path, r.state, r.created_at, r.interrupted, r.pipeline_version, "
            "(SELECT COUNT(*) FROM tasks t WHERE t.run_id=r.id) AS n_tasks, "
            "(SELECT COUNT(*) FROM tasks t WHERE t.run_id=r.id AND t.status IN ('PASSED', 'SUPERSEDED')) AS n_passed "
            "FROM runs r ORDER BY r.id DESC LIMIT ?", (limit,), commit=False).fetchall()
        return [dict(r) for r in rows]

    def load_run(self, run_id: int) -> Run | None:
        r = self._exec("SELECT * FROM runs WHERE id=?", (run_id,), commit=False).fetchone()
        if r is None:
            return None
        run = Run(
            id=r["id"], goal=r["goal"], repo_path=r["repo_path"], state=r["state"],
            resume_state=r["resume_state"], status_message=r["status_message"] or "",
            created_at=r["created_at"], updated_at=r["updated_at"], started_at=r["started_at"],
            finished_at=r["finished_at"], interrupted=bool(r["interrupted"]), git_head=r["git_head"],
            git_branch=r["git_branch"], working_branch=r["working_branch"],
            initial_dirty=json.loads(r["initial_dirty_json"] or "[]"), current_task_id=r["current_task_id"],
            settings_snapshot=json.loads(r["settings_snapshot_json"] or "{}"),
            orchestrator_mode=r["orchestrator_mode"] or "mock", worker_mode=r["worker_mode"] or "mock",
            pipeline_version=(r["pipeline_version"] or "v1").lower(),
            pipeline_state=json.loads(r["pipeline_state_json"] or "{}"),
            worker_snapshot=json.loads(r["worker_snapshot_json"] or "{}"),
        )
        # The run's own pipeline decides how its rows are parsed: V1 task rows are never read as phases.
        tasks = self.load_tasks(run_id, phases=run.pipeline_version == "v2")
        if r["plan_json"] and tasks:
            meta = json.loads(r["plan_json"])
            run.plan = Plan(goal=meta.get("goal", run.goal), summary=meta.get("summary", ""),
                            assumptions=meta.get("assumptions", []),
                            architecture_notes=meta.get("architecture_notes", []), tasks=tasks)
        return run

    def lock_worker_snapshot(self, run_id: int, snapshot: dict[str, Any]) -> bool:
        """V5: store a run's worker-profile snapshot. Write-once: an existing snapshot is never replaced (returns
        False). Committed immediately - it must survive a crash before the first worker launch."""
        cur = self._exec("UPDATE runs SET worker_snapshot_json=? WHERE id=? AND "
                         "(worker_snapshot_json IS NULL OR worker_snapshot_json IN ('', '{}'))",
                         (_j(snapshot), run_id))
        return cur.rowcount == 1

    def worker_profiles_in_use(self) -> dict[str, list[int]]:
        """V5: profile id -> ids of runs that can still be resumed (not COMPLETED) and are bound to it."""
        rows = self._exec("SELECT id, worker_snapshot_json FROM runs WHERE state != 'COMPLETED'",
                          commit=False).fetchall()
        out: dict[str, list[int]] = {}
        for r in rows:
            try:
                d = json.loads(r["worker_snapshot_json"] or "{}")
            except json.JSONDecodeError:
                continue
            snaps = [d.get("default")] + list((d.get("phase_overrides") or {}).values())
            for s in snaps:
                if isinstance(s, dict) and s.get("profile_id"):
                    ids = out.setdefault(str(s["profile_id"]), [])
                    if r["id"] not in ids:
                        ids.append(r["id"])
        return out

    def mark_interrupted_runs(self) -> list[int]:
        """Called at startup: runs that were mid-flight when the app exited."""
        rows = self._exec("SELECT id, state, resume_state FROM runs", commit=False).fetchall()
        ids = []
        for r in rows:
            if r["state"] in _IN_PROGRESS or r["state"] == "PLANNING":
                resume = r["state"] if r["state"] in _IN_PROGRESS - {"STOPPING"} else r["resume_state"]
                new_state = "STOPPED" if r["state"] != "PLANNING" else "ERROR"
                self._exec("UPDATE runs SET state=?, resume_state=?, interrupted=1, "
                           "status_message='interrupted: application exited' WHERE id=?",
                           (new_state, resume, r["id"]))
                self._exec("UPDATE task_attempts SET interrupted=1 WHERE run_id=? AND finished_at IS NULL",
                           (r["id"],))
                ids.append(r["id"])
        return ids

    # ------------------------------------------------------------ tasks

    def save_tasks(self, run_id: int, tasks: list[Task]) -> None:
        with self._lock:
            ids = [t.id for t in tasks]
            if ids:
                self.conn.execute(
                    f"DELETE FROM tasks WHERE run_id=? AND task_id NOT IN ({','.join('?' * len(ids))})",
                    (run_id, *ids))
            else:
                self.conn.execute("DELETE FROM tasks WHERE run_id=?", (run_id,))
            for t in tasks:
                self.conn.execute(
                    "INSERT INTO tasks (run_id, task_id, position, status, attempts, data_json) VALUES (?,?,?,?,?,?) "
                    "ON CONFLICT(run_id, task_id) DO UPDATE SET position=excluded.position, status=excluded.status, "
                    "attempts=excluded.attempts, data_json=excluded.data_json",
                    (run_id, t.id, t.position, t.status.value, t.attempts, _j(t.to_dict())))
            self.conn.commit()

    def save_task(self, run_id: int, task: Task) -> None:
        self._exec(
            "UPDATE tasks SET status=?, attempts=?, data_json=? WHERE run_id=? AND task_id=?",
            (task.status.value, task.attempts, _j(task.to_dict()), run_id, task.id))

    def load_tasks(self, run_id: int, phases: bool = False) -> list[Task]:
        rows = self._exec("SELECT data_json FROM tasks WHERE run_id=? ORDER BY position",
                          (run_id,), commit=False).fetchall()
        parse = Phase.from_dict if phases else Task.from_dict
        return [parse(json.loads(r["data_json"])) for r in rows]

    # ------------------------------------------------------------ attempts

    def insert_attempt(self, run_id: int, a: Attempt) -> int:
        cur = self._exec(
            "INSERT INTO task_attempts (run_id, task_id, number, kind, started_at, data_json) VALUES (?,?,?,?,?,?)",
            (run_id, a.task_id, a.number, a.kind, a.started_at, _j(a.to_dict())))
        a.db_id = int(cur.lastrowid)
        return a.db_id

    def update_attempt(self, a: Attempt) -> None:
        assert a.db_id is not None
        with self._lock:
            self.conn.execute(
                "UPDATE task_attempts SET finished_at=?, exit_code=?, interrupted=?, data_json=? WHERE id=?",
                (a.finished_at, a.exit_code, int(a.interrupted), _j(a.to_dict()), a.db_id))
            self.conn.execute("DELETE FROM changed_files WHERE attempt_id=?", (a.db_id,))
            for c in a.changed_files:
                self.conn.execute(
                    "INSERT INTO changed_files (attempt_id, path, change, attributed_to_worker, note) VALUES (?,?,?,?,?)",
                    (a.db_id, c.path, c.change, int(c.attributed_to_worker), c.note))
            self.conn.execute("DELETE FROM verification_results WHERE attempt_id=?", (a.db_id,))
            for v in a.verification:
                self.conn.execute(
                    "INSERT INTO verification_results (attempt_id, command, exit_code, passed, duration, "
                    "skipped_reason, output_tail) VALUES (?,?,?,?,?,?,?)",
                    (a.db_id, v.command, v.exit_code, int(v.passed), v.duration, v.skipped_reason, v.output_tail))
            self.conn.commit()

    def load_attempts(self, run_id: int) -> dict[str, list[Attempt]]:
        rows = self._exec("SELECT id, data_json, interrupted FROM task_attempts WHERE run_id=? ORDER BY task_id, number, id",
                          (run_id,), commit=False).fetchall()
        out: dict[str, list[Attempt]] = {}
        for r in rows:
            a = attempt_from_dict(json.loads(r["data_json"]))
            a.db_id = r["id"]
            a.interrupted = a.interrupted or bool(r["interrupted"])
            out.setdefault(a.task_id, []).append(a)
        return out

    def add_review(self, run_id: int, a: Attempt) -> None:
        assert a.review is not None
        self._exec(
            "INSERT INTO reviews (run_id, task_id, attempt_id, decision, summary, data_json, created_at) "
            "VALUES (?,?,?,?,?,?,?)",
            (run_id, a.task_id, a.db_id, a.review.decision.value, a.review.summary, _j(a.review.to_dict()), time.time()))

    # ------------------------------------------------------------ calls / logs

    def add_orchestrator_call(self, run_id: int | None, task_id: str | None, c: Any) -> None:
        self._exec(
            "INSERT INTO orchestrator_calls (run_id, task_id, kind, executable, model, argv_json, started_at, "
            "duration, exit_code, cost_usd, usage_json, session_id, error) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, task_id, c.kind, c.executable, c.model, _j(c.argv), c.started_at, c.duration,
             c.exit_code, c.cost_usd,
             _j({"cli_usage": c.usage, "num_turns": getattr(c, "num_turns", None),
                 "assistant_turns": getattr(c, "assistant_turns", 0), "tool_calls": getattr(c, "tool_calls", 0),
                 "attempt": getattr(c, "attempt", 1)}),
             c.session_id, c.error))

    def add_worker_call(self, run_id: int, attempt_id: int | None, o: Any, started_at: float) -> None:
        self._exec(
            "INSERT INTO worker_calls (run_id, attempt_id, executable, argv_json, env_overrides_json, started_at, "
            "duration, exit_code, timed_out, cancelled, tool_calls, usage_json, worker_profile_id, "
            "worker_profile_revision) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (run_id, attempt_id, o.executable, _j(o.argv), _j(redact_env(o.env_overrides)), started_at, o.duration,
             o.exit_code, int(o.timed_out), int(o.cancelled), o.tool_calls, _j(o.usage),
             getattr(o, "worker_profile_id", "") or None, getattr(o, "worker_profile_revision", "") or None))

    def add_log(self, run_id: int | None, source: str, message: str, ts: float | None = None) -> None:
        """Not committed immediately; the engine commits periodically."""
        with self._lock:
            self.conn.execute("INSERT INTO logs (run_id, ts, source, message) VALUES (?,?,?,?)",
                              (run_id, ts or time.time(), source, message))

    def load_logs(self, run_id: int, limit: int = 2000) -> list[dict[str, Any]]:
        rows = self._exec("SELECT ts, source, message FROM (SELECT * FROM logs WHERE run_id=? ORDER BY id DESC LIMIT ?) "
                          "ORDER BY ts", (run_id, limit), commit=False).fetchall()
        return [dict(r) for r in rows]

    def add_transition(self, run_id: int | None, old: str, new: str, reason: str) -> None:
        self._exec("INSERT INTO state_transitions (run_id, ts, from_state, to_state, reason) VALUES (?,?,?,?,?)",
                   (run_id, time.time(), old, new, reason))

    def add_settings_refs(self, run_id: int, refs: dict[str, str]) -> None:
        with self._lock:
            for k, v in refs.items():
                self.conn.execute("INSERT INTO settings_refs (run_id, key, value) VALUES (?,?,?)", (run_id, k, v))
            self.conn.commit()

    def total_orchestrator_cost(self, run_id: int) -> float | None:
        r = self._exec("SELECT SUM(cost_usd) AS c, COUNT(cost_usd) AS n FROM orchestrator_calls WHERE run_id=?",
                       (run_id,), commit=False).fetchone()
        return float(r["c"]) if r and r["n"] else None
