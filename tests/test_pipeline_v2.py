"""Pipeline V2 (phase-based execution) tests, plus V1 compatibility.

Mock orchestrator / mock worker only; the conftest guard refuses any real Claude or
Codex process. Git repositories are throwaway temp repos.
"""
import asyncio
import json
import sqlite3
import subprocess
import sys
import time

import pytest

from app.config.settings import ClaudeSettings, Settings, WorkerSettings
from app.integrations import path_filter
from app.models.attempt import Attempt, ChangedFile, VerificationResult
from app.models.phase import Phase, merge_phase_replan, parse_phase_plan, parse_phase_report, parse_phases
from app.models.plan import Plan, PlanValidationError
from app.models.review import ReviewDecision
from app.models.task import Task, TaskResult, TaskStatus
from app.orchestration.checkpoint import CheckpointContextLimits, CheckpointPackage, build_checkpoint_context
from app.orchestration.orchestrator import Engine
from app.orchestration.phase_scheduler import PhaseScheduler
from app.orchestration.providers.base import OrchestratorError, PlanRequest
from app.orchestration.providers.mock import MOCK_PHASE_PLAN, MockOrchestrator
from app.orchestration.reviewer import ReviewContextTooLarge
from app.orchestration.scheduler import Scheduler
from app.orchestration.state_machine import RunState
from app.persistence.database import Database
from app.persistence.migrations import MIGRATIONS
from app.workers.base import PhaseTurnRequest, RetryContext
from app.workers.codex_ninfer import CodexNInferWorker
from app.workers.mock import MockWorker
from app.workers.phase_prompts import Handoff, PhaseWorkSpec, build_phase_prompt, build_turn_prompt

PY = sys.executable
S = RunState


# ------------------------------------------------------------ helpers

def make_settings(**orch):
    s = Settings()
    s.integrations.metrics_enabled = False
    s.integrations.codebase_memory_executable = ""
    s.orchestration.mock_step_delay = 0.0
    s.orchestration.mock_retry_task_ids = []
    for k, v in orch.items():
        setattr(s.orchestration, k, v)
    return s


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@e", "-c", "user.name=t", *args], cwd=repo, check=True,
                   capture_output=True)


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


class V2Harness:
    """Engine on a temp DB whose providers are shared, scriptable mocks."""

    def __init__(self, tmp_path, settings, orch=None, worker=None, db="v2.db"):
        self.events = []
        self.orch = orch or MockOrchestrator(delay=0.0)
        self.worker = worker or MockWorker(delay=0.0, steps=2)
        self.engine = Engine(settings, lambda k, p: self.events.append((k, p)), str(tmp_path / db))
        self.engine._make_orchestrator = lambda mode: self.orch
        self.engine._make_worker = lambda mode: self.worker

    async def wait_idle(self, timeout=60):
        t0 = time.monotonic()
        while self.engine._busy is not None and not self.engine._busy.done():
            await asyncio.sleep(0.01)
            assert time.monotonic() - t0 < timeout, "engine did not settle"

    async def plan_and_start(self, repo, before_start=None):
        e = self.engine
        await e.plan("Build the feature", str(repo), pipeline="v2")
        await self.wait_idle()
        assert e.sm.state == S.PLAN_READY
        if before_start:
            before_start(e)
        await e.start()
        await self.wait_idle()
        return e

    def turns(self, phase_id=None):
        return [(t["phase"], t["purpose"], t["resumed"]) for t in self.worker.turns
                if phase_id is None or t["phase"] == phase_id]


def phase(e, pid) -> Phase:
    return e.run.plan.task(pid)


# ------------------------------------------------------------ plan parsing

def test_parse_phase_plan_valid_defaults_and_limits():
    plan = parse_phase_plan({"goal": "g", "summary": "s", "phases": [
        {"id": "P1", "title": "Core", "description": "obj", "subtasks": ["a", {"title": "b"}, {"id": "X9", "title": "c"}],
         "acceptance_criteria": ["works"], "verification_commands": ["npm test"], "estimated_minutes": "40",
         "status": "PASSED", "attempts": 5},
        {"title": "Second", "description": "obj2", "subtasks": [{"id": "S1", "title": "x"}, {"id": "S1", "title": "y"}],
         "acceptance_criteria": ["ok"], "integration_expectations": ["uses P1"]},
    ]})
    p1, p2 = plan.tasks
    assert isinstance(p1, Phase) and p1.kind == "phase"
    assert [s.id for s in p1.subtasks] == ["S1", "S2", "X9"]
    assert p1.status == TaskStatus.PENDING and p1.attempts == 0, "Claude cannot inject run state"
    assert p1.estimated_minutes == 40 and p1.dependencies == []
    assert p2.id == "P2" and [s.id for s in p2.subtasks] == ["S1", "S2"] and p2.position == 1
    # round trip through persistence form
    again = Phase.from_dict(json.loads(json.dumps(p2.to_dict())))
    assert again.integration_expectations == ["uses P1"] and again.subtasks[1].title == "y"

    base = {"title": "t", "description": "d", "subtasks": ["a"], "acceptance_criteria": ["c"]}
    for bad, msg in [({"phases": []}, "no phases"),
                     ({"phases": [dict(base, id=f"P{i}") for i in range(9)]}, "at most 8"),
                     ({"phases": [dict(base, id="P1"), dict(base, id="P1")]}, "duplicate"),
                     ({"phases": [dict(base, id="P1", acceptance_criteria=[])]}, "acceptance"),
                     ({"phases": [dict(base, id="P1", subtasks=[])]}, "subtasks"),
                     ({"phases": [dict(base, id="bad id!")]}, "invalid phase id")]:
        with pytest.raises(PlanValidationError, match=msg):
            parse_phase_plan({"goal": "g", "summary": "s", **bad})


def test_parse_phase_report_status_subtasks_and_todo_fallback():
    ids = ["S1", "S2", "S3"]
    r = parse_phase_report("**PHASE STATUS:** incomplete\nSUBTASKS:\n[x] S1 model\n- [ ] S2 ui - later\n[x] s3 x", ids)
    assert r.status == "INCOMPLETE" and r.done == {"S1", "S3"} and r.open == {"S2"}
    r = parse_phase_report("PHASE STATUS: COMPLETE\nSUMMARY: all done", ids)
    assert r.status == "COMPLETE" and r.done == set(ids), "plain COMPLETE covers every subtask"
    todo = [{"text": "S1: model", "completed": True}, {"text": "S2: ui", "completed": False}]
    r = parse_phase_report("SUMMARY: stopped midway", ids, todo)
    assert r.status == "INCOMPLETE" and r.source == "todo" and r.done == {"S1"}
    r = parse_phase_report("SUMMARY: no status", ids)
    assert r.status == "" and r.effective_status == "COMPLETE"


# ------------------------------------------------------------ V2.1.1 planned-subtask progress from the todo list

def _three_subtask_phase() -> Phase:
    return parse_phases([{"id": "P1", "title": "Chase", "description": "d", "acceptance_criteria": ["ok"],
                          "subtasks": [{"id": "S1", "title": "Fix chase"}, {"id": "S2", "title": "Search/return"},
                                       {"id": "S3", "title": "Pedestrians"}]}])[0]


LIVE_TODO = [{"text": "Explore repo: Owner.ts, Pedestrians.ts", "completed": True},
             {"text": "S1: Fix chase give-up", "completed": True},
             {"text": "S2: Search/return", "completed": False},
             {"text": "S3: Pedestrians", "completed": False},
             {"text": "Verify: npm run build", "completed": False}]


def test_todo_entries_map_to_planned_subtasks_by_leading_id_only():
    from app.models.phase import todo_subtask_id
    ids = ["S1", "S2", "S3"]
    for text in ["S1: Fix chase give-up", "S1 - fix", "S1 — fix", "S1. fix", "S1) fix", "s1: fix", "[S1] fix",
                 "**S1**: fix", "- S1: fix", "S1 Fix chase", "S1"]:
        assert todo_subtask_id(text, ids) == "S1", text
    for text in ["Explore repo: S1 files", "Verify S1 and S2", "Fix chase", "S10: other", "S1.2 sub-step", "S1a: x",
                 "S1/S2: both", "S1-2 sub-step", "PS1: x", ""]:
        assert todo_subtask_id(text, ids) == "", text


def test_structured_todo_marks_only_the_completed_planned_subtask():
    ph = _three_subtask_phase()
    assert ph.apply_todo(LIVE_TODO) == ["S1"]
    assert [(s.id, s.done) for s in ph.subtasks] == [("S1", True), ("S2", False), ("S3", False)]
    ph.todo = LIVE_TODO
    assert (ph.subtasks_done, len(ph.subtasks), ph.todo_done, len(ph.todo)) == (1, 3, 2, 5), \
        "Explore/Verify are the worker's own items: they count in the todo, never as planned subtasks"
    # an id-prefixed item split in two only counts when every part is completed
    split = [{"text": "S2: implement", "completed": True}, {"text": "S2: test", "completed": False}]
    assert ph.apply_todo(split) == [] and not ph.subtasks[1].done
    assert ph.apply_todo([dict(split[1], completed=True), split[0]]) == ["S2"]


def test_later_todo_lists_never_regress_a_completed_subtask():
    ph = _three_subtask_phase()
    ph.apply_todo(LIVE_TODO)
    fresh_session_list = [{"text": "S1: Fix chase give-up", "completed": False},
                          {"text": "S2: Search/return", "completed": False}]
    for todo in (fresh_session_list, [], [{"text": "Explore again", "completed": False}]):
        assert ph.apply_todo(todo) == [] and ph.subtasks[0].done, todo
    report = parse_phase_report("PHASE STATUS: INCOMPLETE\nSUMMARY: s", ["S1", "S2", "S3"], fresh_session_list)
    assert report.open == set() and report.done == set(), "an unchecked todo entry never reopens a subtask"


def test_subtask_progress_survives_serialization_and_the_v2_task_table(tmp_path):
    ph = _three_subtask_phase()
    ph.apply_todo(LIVE_TODO)
    ph.todo = LIVE_TODO
    again = Phase.from_dict(json.loads(json.dumps(ph.to_dict())))
    assert [(s.id, s.done) for s in again.subtasks] == [("S1", True), ("S2", False), ("S3", False)]
    assert again.todo == LIVE_TODO and again.subtasks_done == 1
    from app.models.run import Run
    db = Database(tmp_path / "subtasks.db")  # the scheduler's path: save_tasks, then save_task per update
    run_id = db.create_run(Run(id=None, goal="g", repo_path="C:/r", pipeline_version="v2"))
    db.save_tasks(run_id, [_three_subtask_phase()])
    db.save_task(run_id, ph)
    db.close()
    db = Database(tmp_path / "subtasks.db")  # restart
    loaded = db.load_tasks(run_id, phases=True)[0]
    db.close()
    assert isinstance(loaded, Phase) and [s.done for s in loaded.subtasks] == [True, False, False]
    assert loaded.todo == LIVE_TODO and loaded.subtasks_done == 1


def test_v2_run_syncs_subtasks_live_from_the_worker_todo_and_reconciles_on_pass(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=0)
    worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}
    h = V2Harness(tmp_path, make_settings(), worker=worker)
    snapshots = []

    def p1_state():
        t = [p["task"] for k, p in h.events if k == "task" and p["task"]["id"] == "P1"][-1]
        return [s["done"] for s in t["subtasks"]], sum(x["completed"] for x in t["todo"]), len(t["todo"])

    def todo(explore, s1, s2, s3, verify=False):
        return [{"text": "Explore repo: models", "completed": explore},
                {"text": "S1: Survey architecture", "completed": s1},
                {"text": "S2: Add data model", "completed": s2},
                {"text": "S3: Add interfaces", "completed": s3},
                {"text": "Verify: run the tests", "completed": verify}]

    def writer(req):  # the worker's plan-tool updates, as Codex streams them mid-turn
        if req.phase_id != "P1":
            return
        if req.purpose == "start":
            worker.on_todo(todo(False, False, False, False))
            worker.on_todo(todo(True, False, False, False))
            snapshots.append(p1_state())
            worker.on_todo(todo(True, True, False, False))
            snapshots.append(p1_state())
        else:  # the continuation re-sends its list with S1 unchecked: must not regress
            worker.on_todo([{"text": "S1: Survey architecture", "completed": False},
                            {"text": "S2: Add data model", "completed": True},
                            {"text": "S3: Add interfaces", "completed": False}])
            snapshots.append(p1_state())
    worker.file_writer = writer
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert snapshots[0] == ([False, False, False], 1, 5), "Explore done: todo 1/5, subtasks still 0/3"
    assert snapshots[1] == ([True, False, False], 2, 5), "S1 checked in the todo -> subtask S1 done, 1/3"
    assert snapshots[2] == ([True, True, False], 1, 3), "S1 stays done although the new list has it unchecked"
    assert [t["purpose"] for t in worker.turns if t["phase"] == "P1"] == ["start", "continue"]
    logs = [p["text"] for k, p in h.events if k == "log"]
    assert "P1: subtask S1 done (worker todo list) - subtasks 1/3" in logs
    assert any("PASSED phase P1" in t and "subtask S3 not reported done by the worker" in t for t in logs), logs
    persisted = {p.id: p for p in h.engine.db.load_tasks(e.run.id, phases=True)}
    assert all(p.status == TaskStatus.PASSED and p.subtasks_done == len(p.subtasks) for p in persisted.values())


def test_v1_tasks_are_unaffected_by_subtask_sync(tmp_path, repo):
    from app.workers.base import WorkerTaskSpec, build_worker_prompt
    from app.workers.phase_prompts import TODO_PROGRESS_RULE
    h = V2Harness(tmp_path, make_settings(), worker=MockWorker(delay=0.0, steps=1))

    async def main():
        await h.engine.plan("Build the feature", str(repo))
        await h.wait_idle()
        await h.engine.start()
        await h.wait_idle()
        return h.engine
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and e.run.pipeline_version == "v1"
    tasks = e.db.load_tasks(e.run.id)
    assert tasks and all(type(t) is Task and t.status == TaskStatus.PASSED for t in tasks)
    assert all("subtasks" not in t.to_dict() and "todo" not in t.to_dict() for t in tasks)
    assert not any("worker todo list" in p["text"] for k, p in h.events if k == "log")
    v1 = build_worker_prompt(WorkerTaskSpec(task=tasks[0], repo_path=str(repo)))
    assert TODO_PROGRESS_RULE not in v1 and "plan tool" not in v1


def test_merge_phase_replan_keeps_history_and_never_overwrites_ids():
    old = parse_phases([dict(id=f"P{i}", title=f"t{i}", description="d", subtasks=["a"], acceptance_criteria=["c"])
                        for i in (1, 2, 3)])
    old[0].status, old[0].result = TaskStatus.PASSED, TaskResult("P1", "done")
    old[1].status = TaskStatus.SUPERSEDED
    new = parse_phases([dict(id="P1", title="T1", description="d", subtasks=["a"], acceptance_criteria=["c"]),
                        dict(id="P2", title="re-scoped", description="d", subtasks=["a"], acceptance_criteria=["c"]),
                        dict(id="P4", title="new", description="d", subtasks=["a"], acceptance_criteria=["c"])])
    new += parse_phases([dict(id="P1", title="renumbered new work", description="d", subtasks=["a"],
                              acceptance_criteria=["c"])])  # e.g. Claude renumbered the revised phases from P1
    merged = merge_phase_replan(old, new)
    assert [p.id for p in merged] == ["P1", "P2", "P2r1", "P4", "P1r1"], "echo dropped; new work never dropped"
    assert merged[0] is old[0] and merged[0].result.result == "done", "PASSED phase untouched"
    assert merged[1] is old[1] and merged[1].status == TaskStatus.SUPERSEDED
    assert [p.position for p in merged] == [0, 1, 2, 3, 4]
    assert all(p.status == TaskStatus.PENDING for p in merged[2:]) and merged[4].title == "renumbered new work"


# ------------------------------------------------------------ phase progression / sessions

def test_v2_run_passes_phases_in_order_with_one_new_session_per_phase(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings())
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and e.run.status_message.startswith("all phases passed")
    assert isinstance(e.scheduler, PhaseScheduler) and e.run.pipeline_version == "v2"
    assert h.turns() == [("P1", "start", False), ("P2", "start", False), ("P3", "start", False)]
    sessions = [p.session_id for p in e.run.plan.tasks]
    assert all(sessions) and len(set(sessions)) == 3, "a NEW Qwen session at every phase boundary"
    assert h.worker.max_active == 1
    assert [(c.phase.id, c.final) for c in h.orch.checkpoints] == [("P1", False), ("P2", False), ("P3", True)]
    assert all(p.status == TaskStatus.PASSED and p.subtasks_done == len(p.subtasks) for p in e.run.plan.tasks)
    assert "COMPLETED PHASES" in h.worker.turns[1]["prompt"] and "P1 Core model" in h.worker.turns[1]["prompt"]
    assert "LATER PHASES" in h.worker.turns[0]["prompt"] and "P3 Tests and polish" in h.worker.turns[0]["prompt"]
    run = e.db.load_run(e.run.id)
    assert run.pipeline_version == "v2" and all(isinstance(p, Phase) for p in run.plan.tasks)
    assert all(p.status == TaskStatus.PASSED and p.session_id for p in run.plan.tasks)
    attempts = e.db.load_attempts(e.run.id)
    assert [a.review.kind for a in attempts["P1"]] == ["CHECKPOINT"]
    assert [a.review.kind for a in attempts["P3"]] == ["FINAL_AUDIT"]
    assert attempts["P1"][0].turns[0]["mode"] == "new" and attempts["P1"][0].diff_scope == "phase"
    assert e.db.list_runs()[0]["pipeline_version"] == "v2"


def test_v2_checkpoint_retry_repairs_in_a_fresh_session_with_concise_feedback(tmp_path, repo):
    # V3: a RETRY no longer extends the (large) implementation session - it enters REPAIRING.
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert h.turns("P2") == [("P2", "start", False), ("P2", "repair", False)]
    first, repair = [t for t in h.worker.turns if t["phase"] == "P2"]
    assert repair["session"] != first["session"], "RETRY repairs in a FRESH Qwen session"
    assert "REPAIR worker" in repair["prompt"] and "Mock fix: wire the feature" in repair["prompt"]
    assert "do NOT redo" in repair["prompt"] and "HOW TO WORK" not in repair["prompt"]
    assert len(h.turns("P1")) == 1, "completed phase not repeated"
    p2 = phase(e, "P2")
    assert p2.attempts == 2 and p2.status == TaskStatus.PASSED
    atts = e.db.load_attempts(e.run.id)["P2"]
    assert [a.review.decision.value for a in atts] == ["RETRY", "PASS"] and atts[1].kind == "repair"
    states = [p["state"] for k, p in h.events if k == "state"]
    i = states.index("REPAIRING")
    assert states[i - 1:i + 3] == ["RETRY_PENDING", "REPAIRING", "VERIFYING", "REVIEWING"]


def test_v2_incomplete_report_continues_in_the_same_session(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings())
    h.worker.status_script = {"P1": ["INCOMPLETE", "INCOMPLETE", "COMPLETE"]}
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert h.turns("P1") == [("P1", "start", False), ("P1", "continue", True), ("P1", "continue", True)]
    assert len({t["session"] for t in h.worker.turns if t["phase"] == "P1"}) == 1
    assert [c.phase.id for c in h.orch.checkpoints].count("P1") == 1, "internal turns are not reviewed"
    a = e.db.load_attempts(e.run.id)["P1"][0]
    assert [t["purpose"] for t in a.turns] == ["start", "continue", "continue"]
    assert phase(e, "P1").session_turns == 3


def test_v2_blocked_checkpoint_then_resume_with_guidance(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["BLOCKED"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        e = await h.plan_and_start(repo)
        assert e.sm.state == S.BLOCKED and e.run.resume_state == "RETRY_PENDING"
        assert phase(e, "P1").status == TaskStatus.BLOCKED and phase(e, "P2").status != TaskStatus.RUNNING
        await e.resume("Use approach A for the storage layer")
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    assert h.turns("P1") == [("P1", "start", False), ("P1", "retry", True)]
    assert "Use approach A for the storage layer" in h.worker.turns[1]["prompt"]
    assert phase(e, "P1").user_guidance == "" and phase(e, "P1").attempts == 2


def test_v2_replan_supersedes_the_phase_and_revises_only_future_phases(tmp_path, repo):
    new_phases = [
        {"id": "P2", "title": "Feature, re-scoped", "description": "d", "subtasks": ["a", "b"],
         "acceptance_criteria": ["c"]},
        {"id": "P4", "title": "Integration", "description": "d", "subtasks": ["a"], "acceptance_criteria": ["c"]},
    ]
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["REPLAN"]}, replan_result=new_phases)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    p1_result = {}

    def remember(e):
        p1_result["plan_ids"] = [p.id for p in e.run.plan.tasks]

    e = asyncio.run(h.plan_and_start(repo, remember))
    assert e.sm.state == S.COMPLETED
    ids = [p.id for p in sorted(e.run.plan.tasks, key=lambda p: p.position)]
    assert ids == ["P1", "P2", "P2r1", "P4"], "old P3 replaced; reused id renamed, history kept"
    st = {p.id: p.status for p in e.run.plan.tasks}
    assert st == {"P1": TaskStatus.PASSED, "P2": TaskStatus.SUPERSEDED, "P2r1": TaskStatus.PASSED,
                  "P4": TaskStatus.PASSED}
    assert [t[0] for t in h.turns()] == ["P1", "P2", "P2r1", "P4"], "P3 never ran; nothing re-ran"
    p2 = phase(e, "P2")
    assert p2.attempts == 1 and p2.result is not None and "REPLAN" in p2.result.result
    req = orch.replan_requests[0]
    assert req.superseded_phase_id == "P2" and "diverged" in req.reason
    assert e.run.pipeline_state["replans"] == 1 and "pending_replan" not in e.run.pipeline_state
    assert h.orch.checkpoints[-1].phase.id == "P4" and h.orch.checkpoints[-1].final
    run = e.db.load_run(e.run.id)
    assert {p.id: p.status for p in run.plan.tasks} == st
    assert e.db.load_attempts(e.run.id)["P2"][0].review.decision == ReviewDecision.REPLAN


def test_v2_replan_limit_blocks_then_resume_grants_one(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["REPLAN"]})
    h = V2Harness(tmp_path, make_settings(v2_max_replans=0), orch=orch)

    async def main():
        e = await h.plan_and_start(repo)
        assert e.sm.state == S.BLOCKED and e.run.resume_state == "REPLANNING"
        assert not orch.replan_requests
        await e.resume("merge P2 and P3")
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    assert orch.replan_requests[0].user_guidance == "merge P2 and P3"
    assert phase(e, "P1").status == TaskStatus.SUPERSEDED
    assert e.run.pipeline_state["extra_replans"] == 1


# ------------------------------------------------------------ pause / stop / crash recovery

def test_v2_pause_between_turns_resumes_the_same_session(tmp_path, repo):
    worker = MockWorker(delay=0.05, steps=2)
    worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}
    h = V2Harness(tmp_path, make_settings(), worker=worker)

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while worker.active == 0:
            await asyncio.sleep(0.005)
        await e.pause()  # the running turn finishes; the next turn does not start
        await h.wait_idle()
        assert e.sm.state == S.PAUSED and e.run.resume_state == "WORKER_RUNNING"
        p1 = phase(e, "P1")
        assert p1.attempts == 0 and p1.session_id and p1.status == TaskStatus.PAUSED
        a = e.db.load_attempts(e.run.id)["P1"][0]
        assert a.interrupted and len(a.turns) == 1 and "paused" in a.error
        assert not h.orch.checkpoints
        session = p1.session_id
        await e.resume()
        await h.wait_idle()
        return e, session

    e, session = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    assert h.turns("P1") == [("P1", "start", False), ("P1", "resume", True)]
    assert h.worker.turns[1]["session"] == session
    assert phase(e, "P1").attempts == 1, "the paused attempt did not consume the retry budget"


class SlowPhaseWorker(MockWorker):
    """P2's turn blocks (after reporting its session id) until the test stops the run."""

    async def run_turn(self, request):
        self.delay = 30 if request.phase_id == "P2" else 0.0
        return await super().run_turn(request)


def test_v2_stop_and_hard_kill_recovery_continue_the_persisted_session(tmp_path, repo):
    s = make_settings()
    worker = SlowPhaseWorker(delay=0.0, steps=2)
    h = V2Harness(tmp_path, s, worker=worker, db="crash.db")

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while not any(t["phase"] == "P2" and t.get("session") for t in worker.turns):
            await asyncio.sleep(0.005)
        await e.stop()
        await h.wait_idle()
        assert e.sm.state == S.STOPPED and e.run.resume_state == "WORKER_RUNNING"
        assert phase(e, "P1").status == TaskStatus.PASSED and phase(e, "P2").status == TaskStatus.PAUSED
        session = phase(e, "P2").session_id
        assert session, "session id persisted as soon as the worker reported it"
        run_id = e.run.id
        await e.shutdown()
        return run_id, session

    run_id, session = asyncio.run(main())
    # Simulate the app being KILLED mid-turn instead: state/attempt/phase look "in flight".
    conn = sqlite3.connect(tmp_path / "crash.db")
    conn.execute("UPDATE runs SET state='WORKER_RUNNING', resume_state=NULL WHERE id=?", (run_id,))
    conn.execute("UPDATE task_attempts SET finished_at=NULL, interrupted=0 WHERE run_id=? AND task_id='P2'", (run_id,))
    row = conn.execute("SELECT data_json FROM tasks WHERE run_id=? AND task_id='P2'", (run_id,)).fetchone()
    d = json.loads(row[0])
    d["status"] = "RUNNING"
    conn.execute("UPDATE tasks SET data_json=?, status='RUNNING' WHERE run_id=? AND task_id='P2'",
                 (json.dumps(d), run_id))
    conn.commit()
    conn.close()

    worker2 = MockWorker(delay=0.0, steps=2)
    h2 = V2Harness(tmp_path, s, worker=worker2, db="crash.db")

    async def main2():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.STOPPED and e.run.resume_state == "WORKER_RUNNING"
        assert e.run.pipeline_version == "v2" and isinstance(phase(e, "P2"), Phase)
        assert phase(e, "P2").status == TaskStatus.PAUSED and phase(e, "P2").attempts == 0
        assert phase(e, "P2").session_id == session
        await e.resume()
        await h2.wait_idle()
        return e

    e = asyncio.run(main2())
    assert e.sm.state == S.COMPLETED
    assert h2.turns() == [("P2", "resume", True), ("P3", "start", False)], "P1 not repeated; P2 session continued"
    assert worker2.turns[0]["session"] == session and "RESUMING PHASE P2" in worker2.turns[0]["prompt"]
    assert phase(e, "P2").attempts == 1
    assert [a.interrupted for a in e.db.load_attempts(e.run.id)["P2"]] == [True, False]


def test_v2_session_resume_failure_falls_back_to_new_session_with_handoff(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=2)
    worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}
    worker.fail_resume = True
    h = V2Harness(tmp_path, make_settings(), worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert h.turns("P1") == [("P1", "start", False), ("P1", "continue", True), ("P1", "continue", False)]
    fresh = [t for t in worker.turns if t["phase"] == "P1"][2]
    assert "HANDOFF" in fresh["prompt"] and "could not be resumed" in fresh["prompt"]
    p1 = phase(e, "P1")
    assert p1.sessions_started == 2 and p1.session_id == fresh["session"]
    a = e.db.load_attempts(e.run.id)["P1"][0]
    assert [t["mode"] for t in a.turns] == ["new", "resumed", "new"]


def test_v2_session_turn_limit_and_user_request_start_a_fresh_session(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=2)
    worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED", "PASS"]})
    h = V2Harness(tmp_path, make_settings(v2_max_session_turns=1), orch=orch, worker=worker)

    async def main():
        e = await h.plan_and_start(repo)
        assert h.turns("P1") == [("P1", "start", False), ("P1", "continue", False)]
        assert "turns (bounding context growth)" in worker.turns[1]["prompt"]
        assert e.sm.state == S.BLOCKED and e.run.current_task_id == "P2"
        e.settings.orchestration.v2_max_session_turns = 8  # from now on only the user request forces a new session
        await e.request_fresh_session()
        assert e.db.load_run(e.run.id).plan.task("P2").fresh_session_requested
        await e.resume()
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    p2_turns = [t for t in worker.turns if t["phase"] == "P2"]
    assert [(t["purpose"], t["resumed"]) for t in p2_turns] == [("start", False), ("retry", False)]
    assert "requested by the user" in p2_turns[1]["prompt"] and "CHECKPOINT FEEDBACK" in p2_turns[1]["prompt"]
    assert not phase(e, "P2").fresh_session_requested


def test_v2_local_verification_fix_loop_runs_before_any_checkpoint(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=2)
    worker.file_writer = lambda req: (repo / "fixed.txt").write_text("ok") if req.purpose == "verify_fix" else None
    cmd = (f'"{PY}" -c "open(\'vcount.txt\',\'a\').write(\'x\'); import os,sys; '
           f'sys.exit(0 if os.path.exists(\'fixed.txt\') else 1)"')
    h = V2Harness(tmp_path, make_settings(v2_local_fix_rounds=1), worker=worker)

    def set_verification(e):
        phase(e, "P1").verification_commands = [cmd]

    e = asyncio.run(h.plan_and_start(repo, set_verification))
    assert e.sm.state == S.COMPLETED
    assert h.turns("P1") == [("P1", "start", False), ("P1", "verify_fix", True)]
    fix_prompt = worker.turns[1]["prompt"]
    assert "verification FAILED" in fix_prompt and "exit 1" in fix_prompt
    assert [c.phase.id for c in h.orch.checkpoints].count("P1") == 1
    a = e.db.load_attempts(e.run.id)["P1"][0]
    assert a.local_fix_rounds == 1 and all(v.passed for v in a.verification)
    assert (repo / "vcount.txt").read_text() == "xx", "VERIFYING reused the current self-check result"
    assert a.review.decision == ReviewDecision.PASS


def test_v2_worker_infrastructure_failure_halts_without_a_claude_call(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=2)
    worker.infra_failure = True
    h = V2Harness(tmp_path, make_settings(), worker=worker)

    async def main():
        e = await h.plan_and_start(repo)
        assert e.sm.state == S.ERROR and e.run.resume_state == "WORKER_RUNNING"
        assert not h.orch.checkpoints and phase(e, "P1").attempts == 0
        assert all(a.interrupted for a in e.db.load_attempts(e.run.id)["P1"])
        worker.infra_failure = False
        await e.resume()
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and phase(e, "P1").attempts == 1


# ------------------------------------------------------------ attribution / generated files

def test_v2_phase_diff_is_cumulative_and_generated_artifacts_never_reach_review(tmp_path, repo):
    def write(rel, text="x\n"):
        p = repo / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(text)

    def writer(req):
        if req.phase_id == "P1" and req.purpose == "start":
            write("src/a.py", "A = 1\n")
            for junk in ("scratch_smoke/smoke.js", "dist/bundle.js", "node_modules/x/index.js", "src/app.js.map",
                         "tmp/out.txt", "build/app.bundle.js", ".pytest_cache/v/cache"):
                write(junk)
        elif req.phase_id == "P1" and req.purpose == "repair":  # the checkpoint RETRY's repair session
            write("src/b.py", "B = 2\n")
        elif req.phase_id == "P2":
            write("src/c.py", "C = 3\n")
            write("src/a.py", "A = 10\n")

    worker = MockWorker(delay=0.0, steps=1)
    worker.file_writer = writer
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    atts = e.db.load_attempts(e.run.id)
    a1, a2 = atts["P1"]
    assert sorted(c.path for c in a1.changed_files) == ["src/a.py"]
    assert sorted(c.path for c in a2.changed_files) == ["src/a.py", "src/b.py"], "retry sees the whole phase"
    assert a2.diff_scope == "phase" and a1.generated_excluded >= 5
    p2 = atts["P2"][0]
    assert sorted(c.path for c in p2.changed_files) == ["src/a.py", "src/c.py"]
    assert "src/b.py" not in p2.diff and "A = 10" in p2.diff
    note = next(c.note for c in p2.changed_files if c.path == "src/a.py")
    assert note == "file was also changed by an earlier phase"

    ctx1, ctx2 = (build_checkpoint_context(next(c for c in orch.checkpoints if c.phase.id == "P1"
                                                and c.attempt.number == n), CheckpointContextLimits()) for n in (1, 2))
    assert "src/a.py" in ctx2.prompt and "src/b.py" in ctx2.prompt and "B = 2" in ctx2.prompt
    # generated/temporary artifacts are only COUNTED by directory; no path or content is sent
    for junk in ("scratch_smoke/smoke.js", "dist/bundle.js", "node_modules/x", "app.js.map", "tmp/out.txt",
                 "app.bundle.js", ".pytest_cache/v"):
        assert junk not in ctx1.prompt and junk not in ctx2.prompt
    assert "generated/temporary path(s) excluded" in ctx1.prompt, "junk created in attempt 1 is only counted"
    assert "PREVIOUS CHECKPOINTS" in ctx2.prompt and "Mock problem" in ctx2.prompt
    p2_pkg = next(c for c in orch.checkpoints if c.phase.id == "P2")
    p2_ctx = build_checkpoint_context(p2_pkg, CheckpointContextLimits())
    assert "changed by earlier phases are not part of this diff" in p2_ctx.prompt
    final = next(c for c in orch.checkpoints if c.final)
    assert "src/a.py" in final.run_diff_stat and "node_modules" not in final.run_diff_stat
    assert "RUN-LEVEL CHANGES" in build_checkpoint_context(final, CheckpointContextLimits()).prompt


@pytest.mark.parametrize("path,kind", [
    ("scratch_smoke/run.js", "generated"), ("app/scratch_smoke/x/y.txt", "generated"),
    (".pytest_cache/v/x", "generated"), (".mypy_cache/a.json", "generated"), (".turbo/x", "generated"),
    ("tmp/bundle.js", "generated"), ("Temp/x.txt", "generated"), ("src/out.tmp", "generated"),
    ("web/app.bundle.js", "generated"), ("src/templates/page.html", "diffable"), ("src/temperature.py", "diffable"),
    ("src/tmpl.ts", "diffable"), ("docs/build.md", "diffable"), ("src/bundle.ts", "diffable"),
])
def test_path_filter_temporary_artifacts(path, kind):
    assert path_filter.classify(path) == kind


def _package(diff_size=0, final=False, description="Implement it.", repo_path=""):
    ph = parse_phases([{"id": "P2", "title": "Feature", "description": description,
                        "subtasks": ["logic", "wire"], "acceptance_criteria": ["works"],
                        "integration_expectations": ["uses P1"]}])[0]
    diff = "".join(f"diff --git a/src/f{i}.ts b/src/f{i}.ts\n+{'y' * 900}\n" for i in range(diff_size))
    a = Attempt(task_id="P2", number=1, worker_summary="PHASE STATUS: COMPLETE\nSUMMARY: done\nFILES CHANGED: src/f0.ts",
                changed_files=[ChangedFile(f"src/f{i}.ts", "M", True) for i in range(diff_size)]
                + [ChangedFile("node_modules/z/i.js", "A", True)], diff=diff, diff_scope="phase",
                verification=[VerificationResult("npm test", 0, "ok", 1.0)],
                turns=[{"n": 1, "mode": "new", "duration": 30}, {"n": 2, "mode": "resumed", "duration": 10}])
    return CheckpointPackage(phase=ph, attempt=a, max_attempts=3, phase_index=2, phase_count=3, final=final,
                             goal="Make a game", completed=["P1 Core: done"],
                             run_diff_stat=" src/f0.ts | 2 +\n node_modules/z/i.js | 9 +\n 2 files changed",
                             repo_path=repo_path)


def test_checkpoint_context_is_bounded_and_the_last_phase_is_a_final_audit():
    ctx = build_checkpoint_context(_package(diff_size=200, final=True), CheckpointContextLimits())
    assert ctx.stats["final_prompt_bytes"] <= 60000 and "DIFF TRUNCATED" in ctx.prompt
    for part in ["FINAL AUDIT", "RUN-LEVEL CHANGES", "src/f0.ts | 2 +", "EARLIER PHASES", "P1 Core: done",
                 "2 worker turn(s): 1 new session, 1 resumed", "INTEGRATION EXPECTATIONS", "REPLAN"]:
        assert part in ctx.prompt
    assert "node_modules/z" not in ctx.prompt and "(node_modules: 1)" in ctx.prompt  # counted, never listed
    assert "FINAL AUDIT" not in build_checkpoint_context(_package(diff_size=2), CheckpointContextLimits()).prompt
    small = CheckpointContextLimits(max_prompt_bytes=5000, max_diff_bytes=4000)
    assert build_checkpoint_context(_package(diff_size=50), small).stats["final_prompt_bytes"] <= 5000
    with pytest.raises(ReviewContextTooLarge):
        build_checkpoint_context(_package(description="x" * 9000), small)


# ------------------------------------------------------------ V1 compatibility / persistence

def test_v1_run_saved_by_pre_v2_schema_loads_resumes_and_finishes_as_v1(tmp_path, repo):
    db_path = tmp_path / "old.db"
    conn = sqlite3.connect(db_path)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (1)")
    conn.executescript(MIGRATIONS[0])  # exactly the pre-V2 schema
    tasks = [
        {"id": "T001", "title": "Survey", "description": "d", "acceptance_criteria": ["a"], "status": "PASSED",
         "attempts": 1, "result": {"task_id": "T001", "result": "surveyed"}},
        {"id": "T002", "title": "Iface", "description": "d", "acceptance_criteria": ["a"], "dependencies": ["T001"]},
        {"id": "T003", "title": "Impl", "description": "d", "acceptance_criteria": ["a"], "dependencies": ["T002"]},
    ]
    now = time.time()
    conn.execute("INSERT INTO runs (goal, repo_path, state, resume_state, created_at, updated_at, started_at, "
                 "plan_json, orchestrator_mode, worker_mode, settings_snapshot_json) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                 ("old goal", str(repo), "STOPPED", "STARTING_TASK", now, now, now,
                  json.dumps({"goal": "old goal", "summary": "s"}), "mock", "mock", "{}"))
    for i, t in enumerate(tasks):
        full = Task.from_dict(dict(t, status=t.get("status", "PENDING"))).to_dict()
        conn.execute("INSERT INTO tasks (run_id, task_id, position, status, attempts, data_json) VALUES (1,?,?,?,?,?)",
                     (t["id"], i, full["status"], full["attempts"], json.dumps(full)))
    conn.commit()
    conn.close()

    h = V2Harness(tmp_path, make_settings(), worker=MockWorker(delay=0.0, steps=1), db="old.db")
    e = h.engine
    cols = {r[1]: r[4] for r in e.db.conn.execute("PRAGMA table_info(runs)")}
    assert cols["pipeline_version"] == "'v1'" and "pipeline_state_json" in cols
    assert e.db.conn.execute("SELECT version FROM schema_version").fetchone()[0] == len(MIGRATIONS) == 3  # V5: +1

    async def main():
        await e.load_run(1)
        assert e.run.pipeline_version == "v1"
        assert all(type(t) is Task for t in e.run.plan.tasks), "V1 rows are never parsed as phases"
        await e.resume()
        await h.wait_idle()

    asyncio.run(main())
    assert type(e.scheduler) is Scheduler and not isinstance(e.scheduler, PhaseScheduler)
    assert e.sm.state == S.COMPLETED
    assert h.worker.invocations == ["T002", "T003"], "PASSED V1 task not repeated"
    assert not h.worker.turns and not h.orch.checkpoints, "no V2 machinery used for a V1 run"
    assert e.db.list_runs()[0]["pipeline_version"] == "v1"


def test_migration_is_additive_and_idempotent(tmp_path):
    p = tmp_path / "m.db"
    db = Database(p)
    from app.models.run import Run
    run = Run(id=None, goal="g", repo_path="C:/r")
    db.create_run(run)
    db.close()
    db = Database(p)  # re-open: nothing re-applied, data intact
    assert db.conn.execute("SELECT version FROM schema_version").fetchone()[0] == len(MIGRATIONS)
    assert db.load_run(run.id).pipeline_version == "v1" and db.load_run(run.id).pipeline_state == {}
    db.close()


def test_v1_pipeline_is_still_the_default(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings(), worker=MockWorker(delay=0.0, steps=1))

    async def main():
        await h.engine.plan("g", str(repo))
        await h.wait_idle()
        await h.engine.start()
        await h.wait_idle()

    asyncio.run(main())
    e = h.engine
    assert e.run.pipeline_version == "v1" and type(e.scheduler) is Scheduler and e.sm.state == S.COMPLETED
    assert all(type(t) is Task for t in e.run.plan.tasks)


# ------------------------------------------------------------ worker adapter (session reuse)

FAKE_CODEX_V2 = r'''
import json, sys
prompt = sys.stdin.read()
last, mode = sys.argv[1], sys.argv[2]
def out(o): print(json.dumps(o), flush=True)
if mode == "gone":
    sys.stderr.write("Error: no rollout found for thread id th-9\n")
    sys.exit(1)
out({"type": "thread.started", "thread_id": "th-9"})
out({"type": "turn.started"})
out({"type": "item.started", "item": {"id": "t1", "type": "todo_list",
     "items": [{"text": "S1: model", "completed": True}, {"text": "S2: ui", "completed": False}]}})
out({"type": "item.completed", "item": {"id": "c1", "type": "command_execution", "command": "rg x", "exit_code": 0}})
out({"type": "item.completed", "item": {"id": "m1", "type": "agent_message", "text": "PHASE STATUS: COMPLETE"}})
out({"type": "turn.completed", "usage": {"input_tokens": 10, "output_tokens": 2}})
open(last, "w", encoding="utf-8").write("PHASE STATUS: COMPLETE\nSUMMARY: " + mode)
'''


def _fake_worker(tmp_path):
    fake = tmp_path / "fake_codex_v2.py"
    fake.write_text(FAKE_CODEX_V2)
    home = tmp_path / ".codex-ninfer"
    home.mkdir(exist_ok=True)
    w = CodexNInferWorker(WorkerSettings(executable="unused", codex_home=str(home), timeout_seconds=30))
    seen = []

    async def argv(repo, last, sandbox, ephemeral=False, skip_git=False):
        seen.append(("exec", None))
        return [PY, str(fake), last, "fresh"]

    async def resume_argv(session, last, sandbox):
        seen.append(("resume", session))
        return [PY, str(fake), last, "gone" if session == "missing" else "resumed"]

    async def supported():
        return True
    w._build_argv, w._build_resume_argv, w.session_resume_supported = argv, resume_argv, supported
    return w, seen


def test_codex_worker_session_capture_resume_and_resume_failure(tmp_path):
    w, seen = _fake_worker(tmp_path)
    sessions, todos = [], []
    w.on_session = sessions.append
    w.on_todo = todos.append
    out = asyncio.run(w.run_turn(PhaseTurnRequest("do P1", str(tmp_path), "", "P1", "start")))
    assert out.exit_code == 0 and out.session_id == "th-9" and not out.resumed and sessions == ["th-9"]
    assert out.todo == [{"text": "S1: model", "completed": True}, {"text": "S2: ui", "completed": False}]
    assert todos and out.summary.endswith("fresh")
    out = asyncio.run(w.run_turn(PhaseTurnRequest("continue", str(tmp_path), "th-9", "P1", "continue")))
    assert out.resumed and out.session_id == "th-9" and out.summary.endswith("resumed")
    out = asyncio.run(w.run_turn(PhaseTurnRequest("continue", str(tmp_path), "missing", "P1", "continue")))
    assert out.resume_failed and out.session_id == "" and not out.resumed and "could not resume" in out.error
    assert seen == [("exec", None), ("resume", "th-9"), ("resume", "missing")]


def test_codex_resume_argv_keeps_sandbox_and_never_uses_exec_only_flags(tmp_path):
    from app.workers.codex_ninfer import OPTIONAL_EXEC_FLAGS, REQUIRED_EXEC_FLAGS
    exe = tmp_path / "codex-ninfer.exe"
    exe.write_bytes(b"MZ")
    w = CodexNInferWorker(WorkerSettings(executable=str(exe), codex_home=str(tmp_path)))
    w._flags = set(REQUIRED_EXEC_FLAGS + OPTIONAL_EXEC_FLAGS)
    w._resume_flags = {"--json", "--output-last-message", "--model"}
    argv = asyncio.run(w._build_resume_argv("0199-abc", "C:/t/last.txt", "workspace-write"))
    assert argv[:3] == [str(exe), "exec", "resume"] and argv[-2:] == ["0199-abc", "-"]
    assert "--cd" not in argv and "--sandbox" not in argv and "--ephemeral" not in argv
    i = argv.index("-c")
    assert argv[i + 1] == 'sandbox_mode="workspace-write"'
    assert argv[argv.index("--model") + 1] == "qwen3.8-27b" and "--json" in argv
    assert asyncio.run(w.session_resume_supported())


def test_phase_prompts_full_for_new_sessions_and_concise_for_resumed_ones():
    ph = parse_phases([{"id": "P2", "title": "Inventory", "description": "Build inventory",
                        "subtasks": [{"id": "S1", "title": "model"}, {"id": "S2", "title": "ui"}],
                        "acceptance_criteria": ["items stack"], "verification_commands": ["npm test"],
                        "integration_expectations": ["uses the P1 save system"]}])[0]
    spec = PhaseWorkSpec(phase=ph, repo_path="C:/Users/u/SampleGame", phase_index=2, phase_count=4, goal="Game",
                         completed=[("P1", "Save system", TaskResult("P1", "saves work", ["src/save.ts"]), False)],
                         upcoming=[("P3", "Crafting")], codebase_memory_available=True,
                         codebase_memory_project="C-Users-u-SampleGame",
                         handoff=Handoff("the session ran out of context", "SUMMARY: half", ["src/inv.ts"], ["S1"]))
    fresh = build_phase_prompt(spec)
    for part in ["CURRENT PHASE P2 (2 of 4): Inventory", "S1: model", "S2: ui", "items stack", "uses the P1 save system",
                 "P1 Save system: saves work", "src/save.ts", "P3 Crafting", "do NOT implement them now",
                 "list_projects", "index_repository", "exact absolute root path is: C:/Users/u/SampleGame",
                 '"C-Users-u-SampleGame"', "NEVER guess the project name", "scratch_smoke/", "rg", "apply_patch",
                 "pwsh.exe", "reread the exact target region", "PHASE STATUS: COMPLETE | INCOMPLETE | BLOCKED",
                 "HANDOFF", "ran out of context", "src/inv.ts", "todo list"]:
        assert part in fresh, part
    retry = build_turn_prompt("retry", ph, retry=RetryContext(2, 3, "", ["no stacking"], ["merge stacks in add()"]))
    assert "no stacking" in retry and "merge stacks in add()" in retry and "do not re-explore" in retry
    assert "RULES:" not in retry and len(retry) < len(fresh) / 3
    fix = build_turn_prompt("verify_fix", ph, failures=[VerificationResult("npm test", 1, "FAIL inv.test.ts")],
                            fix_round=1, fix_rounds=2)
    assert "FAIL inv.test.ts" in fix and "round 1 of 2" in fix and "do not weaken" in fix
    cont = build_turn_prompt("continue", ph)
    assert "S1, S2" in cont and "PHASE STATUS" in cont


# ------------------------------------------------------------ Claude adapter (V2 calls)

FAKE_CLAUDE_V2 = r'''
import json, sys
sys.stdin.read()
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": []}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.02,
                  "usage": {"input_tokens": 3, "output_tokens": 9}, "structured_output": payload}), flush=True)
'''


def _claude(tmp_path, payload):
    from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
    fake = tmp_path / "fake_claude_v2.py"
    fake.write_text(FAKE_CLAUDE_V2)
    pf = tmp_path / "payload.json"
    pf.write_text(json.dumps(payload))
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused"))
    seen = []

    async def argv(limits, schema):
        seen.append((limits, schema))
        return [PY, str(fake), str(pf)]
    prov._build_argv = argv
    return prov, seen


def test_claude_v2_plan_checkpoint_and_replan_use_their_own_schemas_and_limits(tmp_path):
    prov, seen = _claude(tmp_path, dict(MOCK_PHASE_PLAN, goal="g"))
    plan = asyncio.run(prov.plan_phases(PlanRequest("g", str(tmp_path))))
    assert [p.id for p in plan.tasks] == ["P1", "P2", "P3"] and isinstance(plan.tasks[0], Phase)
    limits, schema = seen[-1]
    assert limits.purpose == "PLAN_V2" and "phases" in schema["properties"] and limits.tools == "Read,Glob,Grep"

    prov, seen = _claude(tmp_path, {"decision": "REPLAN", "summary": "s", "problems": ["p"],
                                    "corrective_instructions": [], "verification_notes": [],
                                    "plan_changes_required": True, "replan_reason": "restructure"})
    review = asyncio.run(prov.checkpoint(_package(diff_size=3, repo_path=str(tmp_path))))
    assert review.decision == ReviewDecision.REPLAN and review.replan_reason == "restructure"
    limits, schema = seen[-1]
    assert limits.purpose == "CHECKPOINT" and limits.tools == "" and limits.max_tool_calls == 0
    assert limits.timeout == 240 and "REPLAN" in schema["properties"]["decision"]["enum"]
    assert prov.calls[-1].kind == "CHECKPOINT"

    prov, seen = _claude(tmp_path, {"summary": "s", "phases": []})
    with pytest.raises(OrchestratorError, match="invalid revised phase plan"):
        from app.orchestration.providers.base import PhaseReplanRequest
        asyncio.run(prov.replan_phases(PhaseReplanRequest("g", Plan(goal="g", tasks=[]), "r", str(tmp_path))))


def test_claude_v2_argv_uses_v2_role_and_checkpoint_budget(tmp_path):
    from app.orchestration.providers.claude_code_cli import (
        OPTIONAL_FLAGS, REQUIRED_FLAGS, SYSTEM_ROLE, SYSTEM_ROLE_V2, ClaudeCodeCLIProvider, limits_for,
    )
    exe = tmp_path / "claude.exe"
    exe.write_bytes(b"MZ")
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable=str(exe)))
    prov._flags = set(REQUIRED_FLAGS + OPTIONAL_FLAGS) - {"--max-turns"}

    def argv(purpose):
        a = asyncio.run(prov._build_argv(limits_for(purpose, prov.settings), {}))
        return a, a[a.index("--append-system-prompt") + 1]

    ck, role = argv("CHECKPOINT")
    assert role == SYSTEM_ROLE_V2 and ck[ck.index("--tools") + 1] == "" and "--allowedTools" not in ck
    assert ck[ck.index("--max-budget-usd") + 1] == "0.60" and ck[ck.index("--model") + 1] == "claude-opus-5-5"
    assert argv("PLAN_V2")[1] == SYSTEM_ROLE_V2
    assert argv("PLAN")[1] == SYSTEM_ROLE and argv("REVIEW")[1] == SYSTEM_ROLE, "V1 calls unchanged"


# ------------------------------------------------------------ GUI (headless)

def test_gui_v2_mock_run_shows_phases_subtasks_and_progress(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from app.config.settings import SettingsStore
    from app.gui.bridge import EventBridge
    from app.gui.main_window import MainWindow
    from app.orchestration.orchestrator import EngineHost
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def pump(cond, timeout=60):
        t0 = time.time()
        while not cond():
            app.processEvents()
            time.sleep(0.01)
            assert time.time() - t0 < timeout, "condition not reached"

    repo = tmp_path / "proj"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    s = make_settings()
    s.orchestration.mock_step_delay = 0.01
    s.last_repo_path = str(repo)
    store = SettingsStore(tmp_path / "settings.json")
    bridge = EventBridge()
    host = EngineHost(s, bridge.emit, str(tmp_path / "gui_v2.db"))
    win = MainWindow(s, store, bridge, host)
    host.start()
    win.engine_started()
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: None)
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    try:
        win.pipeline_combo.setCurrentIndex(win.pipeline_combo.findData("v2"))
        assert store.load().orchestration.pipeline_version == "v2"
        win.task_input.setPlainText("Implement an inventory system")
        win.btn_plan.click()
        pump(lambda: win.state == "PLAN_READY" and win.tasks.topLevelItemCount() == 3)
        assert [win.tasks.topLevelItem(i).childCount() for i in range(3)] == [3, 3, 3]
        assert "V2" in win.mode_label.text()
        win.btn_start.click()
        pump(lambda: win.state == "COMPLETED")
        pump(lambda: "Phases: 3/3 done" in win.m_phase.text())
        assert [win.tasks.topLevelItem(i).text(2) for i in range(3)] == ["PASSED"] * 3
        win.tasks.setCurrentItem(win.tasks.topLevelItem(0))
        app.processEvents()
        text = win.details.toPlainText()
        assert "Subtasks" in text and "Qwen turns" in text and "Claude checkpoint" in text
        pump(lambda: win.project.history.count() >= 1)
        assert "V2" in win.project.history.item(0).text()
    finally:
        win.close()


# ------------------------------------------------------------ audit regressions

def test_v2_pass_with_plan_changes_replans_future_phases_and_keeps_the_phase_passed(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["PASS_CHANGES"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert phase(e, "P1").status == TaskStatus.PASSED and phase(e, "P1").result.result.startswith("[MOCK]")
    req = orch.replan_requests[0]
    assert req.superseded_phase_id == "" and "restructuring" in req.reason
    assert [t[0] for t in h.turns()] == ["P1", "P2", "P3"] and e.run.pipeline_state["replans"] == 1


def test_v2_advancing_is_idempotent_after_a_crash_between_phase_and_run_commits(tmp_path, repo):
    s = make_settings()
    orch = MockOrchestrator(delay=0.2, checkpoint_script={"P1": ["PASS_CHANGES"]})
    h = V2Harness(tmp_path, s, orch=orch, db="adv.db")

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while e.sm.state != S.REVIEWING:
            await asyncio.sleep(0.005)
        await e.pause()  # the checkpoint finishes, the run pauses before ADVANCING
        await h.wait_idle()
        assert e.sm.state == S.PAUSED and e.run.resume_state == "ADVANCING"
        run_id = e.run.id
        await e.shutdown()
        return run_id

    run_id = asyncio.run(main())
    # Crash window: the phase row says PASSED, the run row was not updated yet.
    conn = sqlite3.connect(tmp_path / "adv.db")
    row = conn.execute("SELECT data_json FROM tasks WHERE run_id=? AND task_id='P1'", (run_id,)).fetchone()
    d = json.loads(row[0])
    d["status"] = "PASSED"
    conn.execute("UPDATE tasks SET data_json=?, status='PASSED' WHERE run_id=? AND task_id='P1'", (json.dumps(d), run_id))
    conn.commit()
    conn.close()
    orch2 = MockOrchestrator(delay=0.0)
    h2 = V2Harness(tmp_path, s, orch=orch2, db="adv.db")

    async def main2():
        e = h2.engine
        await e.load_run(run_id)
        assert e.run.current_task_id == "P1" and phase(e, "P1").status == TaskStatus.PASSED
        await e.resume()
        await h2.wait_idle()
        return e

    e = asyncio.run(main2())
    assert e.sm.state == S.COMPLETED
    assert [t[0] for t in h2.turns()] == ["P2", "P3"], "a PASSED phase never runs again"
    assert len(orch2.replan_requests) == 1, "the suggested replan is not lost"


def test_v2_auto_indexes_codebase_memory_and_gives_the_worker_the_exact_project(tmp_path, repo, monkeypatch):
    from app.integrations import codebase_memory
    from app.orchestration import orchestrator as orch_mod
    calls = []

    async def probe(exe, path):
        return codebase_memory.CodebaseMemoryStatus(True, version="0.11.0", repo_indexed=False)

    async def ensure(exe, path, timeout):
        calls.append((exe, path))
        return "C-tmp-proj", "indexed now"
    monkeypatch.setattr(orch_mod.codebase_memory, "probe", probe)
    monkeypatch.setattr(codebase_memory, "ensure_indexed", ensure)
    s = make_settings()
    s.integrations.codebase_memory_executable = str(tmp_path / "codebase-memory-mcp.exe")
    h = V2Harness(tmp_path, s)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert calls == [(s.integrations.codebase_memory_executable, str(repo))], "indexed once, by exact root path"
    assert e.run.pipeline_state["codebase_memory_project"] == "C-tmp-proj"
    assert '"C-tmp-proj"' in h.worker.turns[0]["prompt"] and str(repo) in h.worker.turns[0]["prompt"]


def _fake_codebase_memory_cli(monkeypatch, tmp_path, repo, project, *, listed_before):
    """codebase-memory-mcp 0.11.0 stand-in: text-form list_projects (root quoted only when it has
    spaces) and an index_repository result of exit 0 + status=indexed - the first real V2 run."""
    from app.integrations import codebase_memory
    from app.workers.process_runner import ProcessResult
    root = str(repo).replace("\\", "/")
    row = f'  {project} {root if " " not in root else chr(34) + root + chr(34)} main\n'
    state = {"indexed": listed_before, "calls": []}

    class Runner:
        async def run(self, argv, *, timeout=None, **kw):
            state["calls"].append(argv[1:])
            if argv[1] == "--version":
                return ProcessResult(argv, 0, "codebase-memory-mcp 0.11.0\n", "", 0.0)
            if argv[3] == "index_repository":
                state["indexed"] = True
                out = json.dumps({"nodes": 546, "edges": 1676, "adr_present": False, "status": "indexed"})
                return ProcessResult(argv, 0, out, "", 0.0)
            rows = row if state["indexed"] else ""
            return ProcessResult(argv, 0, f"projects: 1  (cols: name root_path branch)\n{rows}total: 1\n", "", 0.0)
    monkeypatch.setattr(codebase_memory, "ProcessRunner", Runner)
    exe = tmp_path / "codebase-memory-mcp.exe"
    exe.write_bytes(b"MZ")
    return str(exe), state


def test_v2_preflight_recognises_a_successful_index_and_the_session_reuses_the_project(tmp_path, repo, monkeypatch):
    exe, state = _fake_codebase_memory_cli(monkeypatch, tmp_path, repo, "C-tmp-proj", listed_before=False)
    s = make_settings()
    s.integrations.codebase_memory_executable = exe
    h = V2Harness(tmp_path, s)
    h.worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}  # P1 continues in its session
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    logs = [p["text"] for k, p in h.events if k == "log" and p["source"] == "MCP"]
    assert "codebase-memory: repository indexed successfully as C-tmp-proj (546 nodes, 1676 edges)" in logs, logs
    assert not any("did not succeed" in t for t in logs)
    assert sum(c[2] == "index_repository" for c in state["calls"] if c[0] == "cli") == 1
    assert e.run.pipeline_state["codebase_memory_project"] == "C-tmp-proj"
    p1 = [t for t in h.worker.turns if t["phase"] == "P1"]
    assert not p1[0]["resumed"] and '"C-tmp-proj"' in p1[0]["prompt"] and "get_architecture" in p1[0]["prompt"]
    assert p1[1]["resumed"] and p1[1]["purpose"] == "continue"
    assert '"C-tmp-proj" already resolved in this session' in p1[1]["prompt"]
    assert "list_projects" not in p1[1]["prompt"].replace("do not repeat list_projects", "")
    assert "search_graph" in p1[1]["prompt"], "the resumed turn keeps the graph-first reminder"


def test_v2_resumed_turns_keep_graph_first_reminder_when_the_worker_resolved_the_project(tmp_path, repo, monkeypatch):
    exe, state = _fake_codebase_memory_cli(monkeypatch, tmp_path, repo, "C-tmp-proj", listed_before=False)
    s = make_settings()
    s.integrations.codebase_memory_executable = exe
    s.integrations.auto_index_codebase_memory = False  # the orchestrator does not know the project name
    h = V2Harness(tmp_path, s)
    h.worker.status_script = {"P1": ["INCOMPLETE", "COMPLETE"]}
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and "codebase_memory_project" not in e.run.pipeline_state
    p1 = [t for t in h.worker.turns if t["phase"] == "P1"]
    assert "list_projects" in p1[0]["prompt"] and "C-tmp-proj" not in p1[0]["prompt"]
    assert p1[1]["resumed"] and "keep using the project already resolved in this session" in p1[1]["prompt"]
    assert "search_graph" in p1[1]["prompt"]


def test_v2_preflight_already_indexed_repository_is_used_without_reindexing(tmp_path, repo, monkeypatch):
    exe, state = _fake_codebase_memory_cli(monkeypatch, tmp_path, repo, "C-tmp-proj", listed_before=True)
    s = make_settings()
    s.integrations.codebase_memory_executable = exe
    h = V2Harness(tmp_path, s)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    logs = [p["text"] for k, p in h.events if k == "log" and p["source"] == "MCP"]
    assert "codebase-memory: repository already indexed as C-tmp-proj" in logs, logs
    assert not any(c[2] == "index_repository" for c in state["calls"] if c[0] == "cli")
    assert e.run.pipeline_state["codebase_memory_project"] == "C-tmp-proj"
    assert '"C-tmp-proj"' in h.worker.turns[0]["prompt"]


def test_list_projects_parsing_matches_the_exact_root_never_the_folder_name():
    from app.integrations.codebase_memory import parse_list_projects, project_for_root
    out = ('projects: 2  (cols: name root_path branch)\n'
           '  C-Users-V-Documents-Unreal-Projects-OtherGame "C:/Users/V/Documents/Unreal Projects/OtherGame" master\n'
           '  SampleGame "C:/Users/V/Other/SampleGame" main\n'
           'total: 2\n')
    projects = parse_list_projects(out)
    assert projects["C-Users-V-Documents-Unreal-Projects-OtherGame"] == "C:/Users/V/Documents/Unreal Projects/OtherGame"
    assert project_for_root(projects, r"C:\Users\V\Documents\Unreal Projects\OtherGame") == \
        "C-Users-V-Documents-Unreal-Projects-OtherGame"
    assert project_for_root(projects, r"C:\Users\V\Documents\SampleGame") == "", \
        "a same-named project at another root is never used"


class SlowAttributionScheduler(PhaseScheduler):
    async def _attribute(self, phase, attempt, pre):
        await asyncio.sleep(30)


def test_v2_stop_during_attribution_abandons_the_attempt_without_consuming_budget(tmp_path, repo, monkeypatch):
    from app.orchestration import orchestrator as orch_mod
    monkeypatch.setattr(orch_mod, "PhaseScheduler", SlowAttributionScheduler)
    h = V2Harness(tmp_path, make_settings())

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while not h.worker.turns or h.worker.active:
            await asyncio.sleep(0.005)
        await asyncio.sleep(0.05)  # now inside the (slow) attribution
        await e.stop()
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.STOPPED and e.run.resume_state == "WORKER_RUNNING"
    a = e.db.load_attempts(e.run.id)["P1"][0]
    assert a.interrupted and a.finished_at is not None and phase(e, "P1").attempts == 0


def test_v2_stale_pending_replan_from_an_interrupted_checkpoint_never_supersedes_a_passed_phase(tmp_path, repo):
    s = make_settings()
    h = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.2), db="stale.db")

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while e.sm.state != S.REVIEWING:
            await asyncio.sleep(0.005)
        await e.stop()  # killed during P1's checkpoint
        await h.wait_idle()
        run_id = e.run.id
        await e.shutdown()
        return run_id

    run_id = asyncio.run(main())
    # The interrupted checkpoint had decided REPLAN and persisted it just before the kill.
    conn = sqlite3.connect(tmp_path / "stale.db")
    st = json.loads(conn.execute("SELECT pipeline_state_json FROM runs WHERE id=?", (run_id,)).fetchone()[0])
    st["pending_replan"] = {"phase_id": "P1", "supersede": True, "reason": "stale"}
    conn.execute("UPDATE runs SET pipeline_state_json=? WHERE id=?", (json.dumps(st), run_id))
    conn.commit()
    conn.close()
    orch2 = MockOrchestrator(delay=0.0)  # the repeated checkpoint now says PASS
    h2 = V2Harness(tmp_path, s, orch=orch2, db="stale.db")

    async def main2():
        e = h2.engine
        await e.load_run(run_id)
        assert e.run.resume_state == "REVIEWING"
        await e.resume()
        await h2.wait_idle()
        return e

    e = asyncio.run(main2())
    assert e.sm.state == S.COMPLETED and phase(e, "P1").status == TaskStatus.PASSED
    assert not orch2.replan_requests and phase(e, "P1").last_checkpoint == "PASS"


def test_v2_replan_that_leaves_no_phase_to_run_blocks_instead_of_completing(tmp_path, repo):
    echo = [{"id": "P1", "title": "Core model and interfaces", "description": "d", "subtasks": ["a"],
             "acceptance_criteria": ["c"]}]
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P3": ["REPLAN"]}, replan_result=echo)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.BLOCKED and e.run.resume_state == "REPLANNING"
    assert "no phase to run" in e.run.status_message and "final audit passed" not in e.run.status_message
    assert phase(e, "P3").status == TaskStatus.SUPERSEDED and e.run.pipeline_state["pending_replan"]
    assert e.run.pipeline_state["replans"] == 0


def test_v2_final_audit_pass_with_plan_changes_does_not_invent_phases(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P3": ["PASS_CHANGES"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and not orch.replan_requests
    assert [p.id for p in e.run.plan.tasks] == ["P1", "P2", "P3"]


def test_v2_worker_status_is_per_attempt_so_a_stale_blocked_never_skips_the_self_check(tmp_path, repo):
    worker = MockWorker(delay=0.0, steps=1)
    worker.file_writer = lambda req: (repo / "fixed.txt").write_text("ok") \
        if req.purpose in ("verify_fix", "repair_round") else None
    cmd = f'"{PY}" -c "import os,sys; sys.exit(0 if os.path.exists(\'fixed.txt\') else 1)"'
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, make_settings(v2_local_fix_rounds=1), orch=orch, worker=worker)
    real_run_turn = worker.run_turn

    async def run_turn(req):  # attempt 1 reports BLOCKED; later turns state no PHASE STATUS at all
        out = await real_run_turn(req)
        out.summary = ("PHASE STATUS: BLOCKED\n" if req.purpose == "start" else "") + "SUMMARY: worked"
        return out
    worker.run_turn = run_turn
    e = asyncio.run(h.plan_and_start(repo, lambda e: setattr(phase(e, "P1"), "verification_commands", [cmd])))
    assert e.sm.state == S.COMPLETED
    a1, a2 = e.db.load_attempts(e.run.id)["P1"][:2]
    assert a1.local_fix_rounds == 0, "a BLOCKED report skips local fixes"
    # The RETRY attempt is a repair: its verification rounds run despite attempt 1's stale BLOCKED.
    assert a2.kind == "repair" and [t["purpose"] for t in a2.turns] == ["repair", "repair_round"]
    assert all(v.passed for v in a2.verification), "the next attempt starts with a clean worker status"
    pkg = next(c for c in orch.checkpoints if c.phase.id == "P1" and c.attempt.number == 2)
    assert "reported PHASE STATUS: BLOCKED" not in build_checkpoint_context(pkg, CheckpointContextLimits()).prompt


def test_v2_interrupted_retry_resumes_with_the_feedback_in_a_short_prompt(tmp_path, repo):
    worker = MockWorker(delay=0.05, steps=2)
    worker.status_script = {"P1": ["COMPLETE", "INCOMPLETE", "COMPLETE"]}
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)

    async def main():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while not any(t["purpose"] == "repair" for t in worker.turns):
            await asyncio.sleep(0.005)
        await e.pause()
        await h.wait_idle()
        assert e.sm.state == S.PAUSED and e.run.resume_state == "REPAIRING"
        await e.resume()
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    # the interrupted repair continues ITS OWN (fresh, bounded) session, not the implementation session
    assert h.turns("P1") == [("P1", "start", False), ("P1", "repair", False), ("P1", "repair_resume", True)]
    assert worker.turns[2]["session"] == worker.turns[1]["session"] != worker.turns[0]["session"]
    resume_prompt = worker.turns[2]["prompt"]
    assert "RESUMING THE REPAIR of phase P1" in resume_prompt and "Mock fix: wire the feature" in resume_prompt
    assert "HOW TO WORK" not in resume_prompt
    pkg = orch.checkpoints[1]
    assert "attempt 2 of 3" in build_checkpoint_context(pkg, CheckpointContextLimits()).prompt, \
        "the interrupted attempt is not counted"


def test_codex_worker_never_resumes_a_session_missing_from_codex_home(tmp_path):
    w, seen = _fake_worker(tmp_path)
    day = tmp_path / ".codex-ninfer" / "sessions" / "2026" / "09" / "25"
    day.mkdir(parents=True)
    (day / "rollout-2026-09-25T09-22-26-th-9.jsonl").write_text("{}")
    assert w.session_exists("th-9") and not w.session_exists("th-404")
    out = asyncio.run(w.run_turn(PhaseTurnRequest("continue", str(tmp_path), "th-404", "P1", "continue")))
    assert out.resume_failed and seen == [], "no process was started for a missing session"
    out = asyncio.run(w.run_turn(PhaseTurnRequest("continue", str(tmp_path), "th-9", "P1", "continue")))
    assert out.resumed and seen == [("resume", "th-9")]
