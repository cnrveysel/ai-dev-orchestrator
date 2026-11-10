"""V4 pass 3: the NORMAL implementation guard (WORKER_RUNNING, before any checkpoint).

A runaway implementation session is stopped by wall time, tool calls or time without PLANNED-subtask
progress; Claude MID_PHASE_DIAGNOSE (mock) guides a FRESH worker session; after the maximum diagnoses the
phase BLOCKS instead of looping. When every planned subtask is done and the orchestrator's verification is
green, an INCOMPLETE report / non-zero worker exit does not keep the worker going.

Deterministic: mock orchestrator, a scripted looping worker, throwaway git repositories.
"""
import asyncio
import sys
import time

import pytest

from app.models.guard import ImplGuardState, MidPhaseDiagnosis, impl_guard_trip
from app.orchestration.impl_guard import MidPhaseLimits, MidPhasePacket, build_mid_phase_prompt
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.state_machine import RunState
from app.workers.base import WorkerOutcome
from app.workers.mock import MockWorker
from tests.test_pipeline_v2 import V2Harness, git, make_settings, phase

PY = sys.executable
S = RunState


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


class GuardWorker(MockWorker):
    """P1 turns follow a script (one dict per turn); every other turn is the plain mock.
        seconds  how long the turn keeps working unless cancelled
        tick     seconds between tool calls
        todo     [(at_seconds, [{"text", "completed"}, ...]), ...] structured todo updates
        status   PHASE STATUS of the final message; exit - the process exit code"""

    def __init__(self, script):
        super().__init__(delay=0.0, steps=1)
        self.script = list(script)
        self.cancelled = False
        self.p1_turns: list[dict] = []

    async def cancel(self):
        self.cancelled = True

    async def run_turn(self, req):
        if req.phase_id != "P1" or not self.script:
            return await super().run_turn(req)
        b = self.script.pop(0)
        self.cancelled = False
        sid = req.session_id or f"guard-{len(self.p1_turns) + 1}"
        rec = {"purpose": req.purpose, "session": req.session_id, "prompt": req.prompt}
        self.p1_turns.append(rec)
        self.turns.append({"phase": "P1", "purpose": req.purpose, "resumed": bool(req.session_id), "session": sid,
                           "prompt": req.prompt})
        if self.on_session:
            self.on_session(sid)
        start, n = time.monotonic(), 0
        todos = list(b.get("todo") or [])
        last_todo = []
        while not self.cancelled and time.monotonic() - start < b.get("seconds", 0.0):
            await asyncio.sleep(b.get("tick", 0.05))
            n += 1
            if self.on_tool_call:
                self.on_tool_call(n)
            while todos and time.monotonic() - start >= todos[0][0]:
                last_todo = todos.pop(0)[1]
                if self.on_todo:
                    self.on_todo(last_todo)
        for _at, items in todos:  # updates scheduled after a quick turn
            last_todo = items
            if self.on_todo:
                self.on_todo(items)
        rec["cancelled"] = self.cancelled
        rec["duration"] = time.monotonic() - start
        status = b.get("status", "INCOMPLETE" if self.cancelled else "COMPLETE")
        return WorkerOutcome(executable="(guard)", argv=["(guard)"], exit_code=None if self.cancelled else
                             b.get("exit", 0), summary=f"PHASE STATUS: {status}\nSUMMARY: turn", stdout_tail="",
                             stderr_tail="", duration=time.monotonic() - start, tool_calls=n,
                             cancelled=self.cancelled, session_id=sid, todo=last_todo)


def todo(*done, helper=False):
    items = [{"text": f"S{i}: step", "completed": f"S{i}" in done} for i in (1, 2, 3)]
    return [{"text": "exploring the repository", "completed": True}] if helper else items


def guard_settings(**kw):
    base = dict(implementation_max_wall_seconds=60, implementation_max_tool_calls=10_000,
                implementation_max_no_subtask_progress_seconds=60, implementation_max_diagnoses=2,
                v2_local_fix_rounds=0)
    base.update(kw)
    return make_settings(**base)


def logs(h, source=None):
    return [p["text"] for k, p in h.events if k == "log" and (source is None or p["source"] == source)]


def states(h):
    return [p["state"] for k, p in h.events if k == "state"]


def p1_guard(e):
    return e.run.pipeline_state.get("impl_guard", {}).get("P1")


def test_the_guard_trips_on_each_limit_and_only_structured_progress_counts():
    o = guard_settings(implementation_max_wall_seconds=100, implementation_max_tool_calls=5,
                       implementation_max_no_subtask_progress_seconds=30).orchestration
    g = ImplGuardState("P1")
    assert impl_guard_trip(g, o) == ""
    assert "wall-clock budget reached (100s/100s)" in impl_guard_trip(ImplGuardState("P1", wall_used=100), o)
    assert "tool budget reached 5/5" == impl_guard_trip(ImplGuardState("P1", tool_calls=5), o)
    assert "no planned subtask progress for 30s" in impl_guard_trip(ImplGuardState("P1", wall_used=40,
                                                                                   progress_wall=10), o)
    assert impl_guard_trip(ImplGuardState("P1", wall_used=40, progress_wall=15), o) == ""
    assert "tool budget" in impl_guard_trip(g, o, extra_tools=5), "the turn in flight counts"
    g2 = ImplGuardState("P1", tool_calls=2, wall_used=5.0, inflight_tool_calls=3, inflight_wall=4.0)
    g2.fold_inflight()
    assert (g2.tool_calls, g2.wall_used, g2.inflight_tool_calls) == (5, 9.0, 0), "a crash never refunds a budget"
    assert ImplGuardState.from_dict(g2.to_dict()) == g2
    off = guard_settings(implementation_max_wall_seconds=0, implementation_max_tool_calls=0,
                         implementation_max_no_subtask_progress_seconds=0).orchestration
    assert impl_guard_trip(ImplGuardState("P1", wall_used=1e6, tool_calls=10**6), off) == "", "0 disables a limit"


def test_wall_limit_escalates_to_mid_phase_diagnose_and_a_fresh_session(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.05}, {"seconds": 0.0, "todo": [(0, todo("S1", "S2", "S3"))]}])
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, guard_settings(implementation_max_wall_seconds=0.6), orch=orch, worker=worker)
    t0 = time.monotonic()
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and time.monotonic() - t0 < 20
    first, second = worker.p1_turns
    assert first["cancelled"] and first["duration"] < 5, "the runaway turn was stopped"
    assert second["purpose"] == "continue" and not second["session"], "a FRESH continuation session"
    assert "LEAD'S DIAGNOSIS OF WHY THE PREVIOUS SESSION DID NOT PROGRESS" in second["prompt"]
    assert "[MOCK] wire the parser into the entry point" in second["prompt"]
    assert "wall-clock budget reached" in second["prompt"]
    [pk] = orch.mid_phase_packets
    assert "wall-clock budget reached" in pk.reason and pk.usage["wall_limit"] == 0.6
    prompt = orch.mid_phase_prompts[0]
    for part in ["MID_PHASE_DIAGNOSE", "Guard reason: wall-clock budget reached", "PLANNED SUBTASKS DONE",
                 "CURRENT VERIFICATION", "no tools"]:
        assert part in prompt, part
    assert any(x.startswith("P1: implementation guard: wall-clock budget reached") for x in logs(h, "SYSTEM"))
    assert any(x.startswith("MID_PHASE_DIAGNOSE P1: CONTINUE") for x in logs(h, "CLAUDE"))
    assert "impl_guard" in e.run.pipeline_state and p1_guard(e) is None, "cleared when the phase passes"


def test_tool_limit_stops_the_turn(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.02}, {"seconds": 0.0}])
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, guard_settings(implementation_max_tool_calls=8), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and worker.p1_turns[0]["cancelled"]
    assert orch.mid_phase_packets[0].reason == "tool budget reached 8/8"


def test_structured_subtask_progress_resets_the_no_progress_timer(tmp_path, repo):
    # 1.5s of work with a planned subtask finished every 0.4s: no gap reaches the 0.7s no-progress limit
    worker = GuardWorker([{"seconds": 1.5, "tick": 0.05, "todo": [(0.4, todo("S1")), (0.8, todo("S1", "S2")),
                                                                  (1.2, todo("S1", "S2", "S3"))]}])
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, guard_settings(implementation_max_no_subtask_progress_seconds=0.7), orch=orch,
                  worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and not worker.p1_turns[0]["cancelled"], "real progress is never punished"
    assert not orch.mid_phase_packets
    assert [s.done for s in phase(e, "P1").subtasks] == [True, True, True]


def test_helper_todo_updates_are_not_progress(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.05, "todo": [(0.1, todo(helper=True)), (0.3, todo(helper=True)),
                                                                 (0.5, todo(helper=True))]},
                          {"seconds": 0.0}])
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, guard_settings(implementation_max_no_subtask_progress_seconds=0.7), orch=orch,
                  worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED and worker.p1_turns[0]["cancelled"]
    assert "no planned subtask progress" in orch.mid_phase_packets[0].reason


def test_the_maximum_diagnoses_block_instead_of_looping_and_resume_gets_a_fresh_budget(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.05}] * 2 + [{"seconds": 0.0}])
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, guard_settings(implementation_max_wall_seconds=0.5, implementation_max_diagnoses=1),
                  orch=orch, worker=worker)

    async def main():
        e = await h.plan_and_start(repo)
        assert e.sm.state == S.BLOCKED and e.run.resume_state == "WORKER_RUNNING"
        assert "after 1 MID_PHASE_DIAGNOSE call(s)" in e.run.status_message
        assert len(orch.mid_phase_packets) == 1 and len(worker.p1_turns) == 2
        g = p1_guard(e)
        assert g["status"] == "blocked" and len(g["trips"]) == 2 and g["diagnoses_used"] == 1
        await e.resume("Use the existing parser module")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    third = worker.p1_turns[2]
    assert not third["session"] and "Use the existing parser module" in third["prompt"], "fresh session + guidance"
    assert any("fresh session and budget" in x for x in logs(h, "SYSTEM"))


def test_a_blocked_diagnosis_blocks_immediately(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.05}])
    orch = MockOrchestrator(delay=0.0, mid_phase_script=["BLOCKED"])
    h = V2Harness(tmp_path, guard_settings(implementation_max_wall_seconds=0.5), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.BLOCKED and "choose the persistence format" in e.run.status_message
    assert len(worker.p1_turns) == 1 and e.run.resume_state == "WORKER_RUNNING"


def test_completed_subtasks_and_green_verification_advance_despite_an_incomplete_worker(tmp_path, repo):
    worker = GuardWorker([{"seconds": 0.0, "todo": [(0, todo("S1", "S2", "S3"))], "status": "INCOMPLETE",
                           "exit": 1}])
    h = V2Harness(tmp_path, guard_settings(), worker=worker)

    def before(e):
        phase(e, "P1").verification_commands = [f'"{PY}" -c "print(1)"']
    e = asyncio.run(h.plan_and_start(repo, before))
    assert e.sm.state == S.COMPLETED
    assert len(worker.p1_turns) == 1, "the worker is not resumed just because it said INCOMPLETE / exited 1"
    assert any("all planned subtasks are done and the phase verification is green" in x for x in logs(h, "SYSTEM"))
    seq = states(h)
    assert seq[seq.index("WORKER_RUNNING") + 1] == "VERIFYING"
    # with a failing verification the worker does continue (bounded as before)
    worker2 = GuardWorker([{"seconds": 0.0, "todo": [(0, todo("S1", "S2", "S3"))], "status": "INCOMPLETE",
                            "exit": 1}, {"seconds": 0.0}])
    h2 = V2Harness(tmp_path, guard_settings(), worker=worker2, db="red.db")

    def failing(e):
        phase(e, "P1").verification_commands = [f'"{PY}" -c "import sys; sys.exit(1)"']
    asyncio.run(h2.plan_and_start(repo, failing))
    assert len(worker2.p1_turns) == 2 and worker2.p1_turns[1]["purpose"] == "resume"


def test_the_guard_budget_survives_a_restart(tmp_path, repo):
    worker = GuardWorker([{"seconds": 30, "tick": 0.05}])
    orch = MockOrchestrator(delay=0.0)
    s = guard_settings(implementation_max_wall_seconds=100)
    h = V2Harness(tmp_path, s, orch=orch, worker=worker, db="g.db")

    async def first():
        e = h.engine
        await e.plan("g", str(repo), pipeline="v2")
        await h.wait_idle()
        await e.start()
        while not (p1_guard(e) or {}).get("inflight_tool_calls"):
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await e.stop()
        await h.wait_idle()
        g = p1_guard(e)
        assert g["inflight_tool_calls"] > 0 or g["tool_calls"] > 0
        run_id = e.run.id
        await e.shutdown()
        return run_id, g
    run_id, g_before = asyncio.run(first())
    used = g_before["tool_calls"] + g_before["inflight_tool_calls"]
    worker2 = GuardWorker([{"seconds": 0.0}])
    h2 = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0), worker=worker2, db="g.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(second())
    assert e.sm.state == S.COMPLETED
    g = e.db.load_run(e.run.id).pipeline_state.get("impl_guard", {})
    assert "P1" not in g, "cleared once the phase passed"
    assert used > 0, "the interrupted turn's usage was persisted and folded in, never refunded"


def test_mid_phase_prompt_is_short_and_bounded(repo):
    from tests.test_evidence import make_phase, req
    ph = make_phase([req("E1")])
    pk = MidPhasePacket(phase=ph, reason="wall-clock budget reached (2400s/2400s)", usage={
        "wall_used": 2400, "wall_limit": 2400, "tool_calls": 12, "tool_limit": 50, "no_progress": 900,
        "no_progress_limit": 1200}, turns=[{"n": i, "purpose": "continue", "mode": "resumed", "duration": 600,
                                            "tool_calls": 10} for i in range(20)],
        worker_summary="w" * 20000, diff_stat=" src/a.ts | 3 +-\n" * 500, changed_files=[f"f{i}" for i in range(200)],
        previous=[{"reason": "r", "blocker": "b" * 3000, "next_actions": ["x"] * 10}] * 5)
    prompt, stats = build_mid_phase_prompt(pk, MidPhaseLimits(6000))
    assert stats["prompt_bytes"] <= 6000 and "ANSWER (structured)" in prompt and "Guard reason:" in prompt
    d = MidPhaseDiagnosis.from_dict({"status": "continue", "blocker": "b", "diagnosis": "d", "next_actions": ["a"],
                                     "avoid": []})
    assert d.status == "CONTINUE" and d.guidance_lines()[0] == "- blocker: b"
    with pytest.raises(ValueError):
        MidPhaseDiagnosis.from_dict({"status": "RETRY"})
