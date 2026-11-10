"""V3 orchestration hardening: bounded, evidence-driven REPAIRING after a checkpoint RETRY.

Mock orchestrator / scripted mock workers only (the conftest guard refuses any real Claude
or Codex process). The phase verification is a tiny acceptance script in a throwaway git
repository whose failing assertions are driven by impl.txt.
"""
import asyncio
import json
import sqlite3
import subprocess
import sys
import time

import pytest

from app.integrations.git import GitRepo
from app.models.attempt import VerificationResult
from app.models.phase import parse_phases
from app.models.repair import DIAGNOSIS_JSON_SCHEMA, Diagnosis, RepairGuardState, RepairState
from app.models.review import ReviewDecision
from app.models.task import TaskStatus
from app.orchestration import repair as rp
from app.orchestration.providers.base import OrchestratorError
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.state_machine import RunState
from app.workers.base import WorkerOutcome
from app.workers.mock import MockWorker
from app.workers.phase_prompts import RepairSpec, build_repair_prompt
from tests.test_pipeline_v2 import V2Harness, git, make_settings, phase

PY = sys.executable
S = RunState
VERIFY = f'"{PY}" tests/acceptance_check.py'
ACCEPTANCE = (
    "import pathlib, sys\n"
    "names = [l.strip() for l in pathlib.Path('impl.txt').read_text().splitlines() if l.strip()]\n"
    "for n in names:\n"
    "    print('FAIL: ' + n)\n"
    "print('all assertions passed' if not names else str(len(names)) + ' failing')\n"
    "sys.exit(1 if names else 0)\n"
)


def impl(repo, *names):
    (repo / "impl.txt").write_text("".join(n + "\n" for n in names))


@pytest.fixture
def arepo(tmp_path):
    r = tmp_path / "proj"
    (r / "tests").mkdir(parents=True)
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    (r / ".gitignore").write_text("ignored.log\n")
    (r / "tests" / "acceptance_check.py").write_text(ACCEPTANCE)
    impl(r, "no car stood still longer than 20 s (max 34.2 s)", "cars honked during the run (3 honks)",
         "intersection cleared")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


def repair_settings(**kw):
    base = dict(v2_local_fix_rounds=0, repair_max_diagnoses=1)
    base.update(kw)
    return make_settings(**base)


def with_verification(e):
    phase(e, "P1").verification_commands = [VERIFY]


class ScriptedWorker(MockWorker):
    """Repair turns run the next scripted step (edits the repository), everything else is the
    plain mock. `summary` can make the worker CLAIM progress without changing anything."""

    def __init__(self, repo, steps=None, summary=""):
        super().__init__(delay=0.0, steps=1)
        self.repo = repo
        self.steps = 1
        self.script = list(steps or [])
        self.claim = summary
        self.seen_impl: list[str] = []

    async def run_turn(self, req):
        if req.purpose.startswith("repair"):
            self.seen_impl.append((self.repo / "impl.txt").read_text())
            step = self.script.pop(0) if self.script else None
            self.file_writer = (lambda r: step(self, r)) if step else None
        try:
            out = await super().run_turn(req)
        finally:
            self.file_writer = None
        if req.purpose.startswith("repair") and self.claim:
            out.summary = "PHASE STATUS: COMPLETE\nSUMMARY: " + self.claim
        return out


class LoopingWorker(MockWorker):
    """Repair turns keep working (tool calls / probe files) until the orchestrator cancels them."""

    def __init__(self, repo, mode, limit_seconds=20.0):
        super().__init__(delay=0.0, steps=1)
        self.repo, self.mode, self.limit = repo, mode, limit_seconds
        self.cancelled = False
        self.durations: list[float] = []
        self.calls: list[int] = []

    async def cancel(self):
        self.cancelled = True

    async def run_turn(self, req):
        if not req.purpose.startswith("repair"):
            return await super().run_turn(req)
        self.cancelled = False
        sid = req.session_id or f"loop-session-{len(self.turns) + 1}"
        self.turns.append({"phase": req.phase_id, "purpose": req.purpose, "resumed": bool(req.session_id),
                           "session": sid, "prompt": req.prompt})
        if self.on_session:
            self.on_session(sid)
        start, n = time.time(), 0
        while not self.cancelled and time.time() - start < self.limit:
            await asyncio.sleep(0.02)
            if self.mode in ("tools", "probes", "block"):
                n += 1
                if self.mode == "probes":
                    p = self.repo / f"probe{n}.js"
                    p.write_text("console.log('probe')\n")
                    if self.on_file_change:
                        self.on_file_change("add", str(p))
                if self.on_tool_call:
                    self.on_tool_call(n)
                if self.mode == "block" and n >= 3:
                    await asyncio.sleep(60)  # killed here by STOP
        self.durations.append(time.time() - start)
        self.calls.append(n)
        return WorkerOutcome(executable="(mock)", argv=["(mock)"], exit_code=0 if not self.cancelled else None,
                             summary="PHASE STATUS: INCOMPLETE\nSUMMARY: I found the root cause (the final deadlock)",
                             stdout_tail="", stderr_tail="", duration=time.time() - start, tool_calls=n,
                             cancelled=self.cancelled, session_id=sid)


def logs(h, source=None):
    return [p["text"] for k, p in h.events if k == "log" and (source is None or p["source"] == source)]


def repair_turns(worker):
    return [t for t in worker.turns if t["phase"] == "P1" and t["purpose"].startswith("repair")]


def run(h, repo, before=with_verification):
    return asyncio.run(h.plan_and_start(repo, before))


# ------------------------------------------------------------ failure fingerprinting (unit)

def test_fingerprint_normalizes_noise_but_keeps_assertion_identity():
    n = rp.normalize_failure_line
    assert n("\x1b[31mFAIL\x1b[0m: no car stood still longer than 20 s (max 34.7 s, car #12)") == \
        n("FAIL: no car stood still longer than 20 s (max 41.2 s, car #3)") == \
        "FAIL: no car stood still longer than 20 s (max <n> s, car #<n>)", "measurements vary, the threshold stays"
    assert n("[12:03:44] FAIL: cars honked during the run") == n("2026-09-26T10:00:01Z FAIL: cars honked during the run")
    assert n(r"FAIL C:\Users\V\AppData\Local\Temp\tmpab12\smoke.test.ts:12:5 deadlock") == \
        n("FAIL /tmp/pytest-of-v/tmp9/smoke.test.ts:98 deadlock") == "FAIL <abs>/smoke.test.ts:<n> deadlock"
    assert n("FAILED tests/test_x.py::test_y - AssertionError: assert 3 == 1") == "FAILED tests/test_x.py::test_y"
    assert n("not ok 7 - intersection clears") == "not ok - intersection clears"
    assert n("FAIL: took 123 ms at 0x7ffde0 id 3f9a2c1be4d7") == "FAIL: took <n> ms at <hex> id <id>"
    assert n("FAIL: no car stood still longer than 20 s") != n("FAIL: no car stood still longer than 30 s")


def test_same_underlying_failures_give_the_same_fingerprint_across_rounds():
    def ev(out, code=1, cmd="node smoke.mjs"):
        return rp.evaluate_verification([VerificationResult(cmd, code, out)])
    r1 = ev("ok 1\nFAIL: no car stood still longer than 20 s (max 34.7 s)\nFAIL: cars honked during the run (3)\n")
    r2 = ev("\x1b[31mFAIL: cars honked during the run (7)\x1b[0m\nFAIL: no car stood still longer than 20 s (max 51 s)\n")
    r3 = ev("FAIL: cars honked during the run (1)\n")
    r4 = ev("FAIL: a new assertion\n")
    assert r1.fingerprint == r2.fingerprint and r1.score.failing == 2
    assert rp.trend(r1, r2) == "same" and rp.trend(r2, r3) == "improved" and rp.trend(r3, r1) == "worse"
    assert rp.trend(r3, r4) == "changed" and rp.trend(None, r1) == "first"
    fixed, new = rp.fixed_and_new(r1, r3)
    assert len(fixed) == 1 and "no car stood still" in fixed[0] and not new
    clean = ev("all good", code=0)
    assert clean.passed and clean.fingerprint == "" and clean.score.better_than(r3.score)


def test_score_orders_build_failure_then_critical_then_failing_count():
    def ev(cmd, code, out):
        return rp.evaluate_verification([VerificationResult(cmd, code, out)])
    build = ev("npm run build", 2, "src/a.ts(3,1): error TS2304: Cannot find name 'x'.")
    hang = ev("npm test", None, "running...")
    three = ev("npm test", 1, "FAIL: a\nFAIL: b\nFAIL: c\n")
    one = ev("npm test", 1, "FAIL: a\n")
    assert build.score.build_failed and hang.score.critical and not three.score.critical
    assert one.score.better_than(three.score) and three.score.better_than(hang.score)
    assert hang.score.better_than(build.score)
    skipped = rp.evaluate_verification([VerificationResult("npm test", None, skipped_reason="disabled")])
    assert not skipped.has_evidence and not skipped.passed


def test_guard_trips_on_each_limit_and_never_on_prose():
    o = make_settings().orchestration
    assert (o.repair_max_wall_seconds, o.repair_max_tool_calls, o.repair_max_same_failure_rounds,
            o.repair_regression_grace_rounds, o.repair_max_new_probe_files) == (1200, 30, 3, 1, 3)
    assert rp.guard_trip(RepairGuardState(), o) == ""
    assert "wall-clock" in rp.guard_trip(RepairGuardState(wall_used=1200), o)
    assert "tool budget reached 30/30" in rp.guard_trip(RepairGuardState(tool_calls=30), o)
    assert "unchanged for 3/3" in rp.guard_trip(RepairGuardState(same_rounds=3), o)
    assert rp.guard_trip(RepairGuardState(same_rounds=2), o) == ""
    assert "probe" in rp.guard_trip(RepairGuardState(probe_files=["p1.js", "p2.js", "p3.js", "p4.js"]), o)
    assert rp.guard_trip(RepairGuardState(probe_files=["p1.js", "p2.js", "p3.js"]), o) == ""
    assert "regression" in rp.guard_trip(RepairGuardState(restores=2), o)
    g = RepairGuardState(tool_calls=10, wall_used=5.0, inflight_tool_calls=4, inflight_wall=2.5)
    g.fold_inflight()
    assert (g.tool_calls, g.wall_used, g.inflight_tool_calls) == (14, 7.5, 0)


def test_probe_and_acceptance_path_rules(tmp_path):
    for p in ["probe2.js", "debug_deadlock.ts", "src/deadlockProbe.js", "scratch_smoke/x.mjs", "repro-honk.py",
              "tmp/check.js"]:
        assert rp.is_probe_path(p), p
    for p in ["tests/test_traffic.py", "src/traffic.ts", "src/latest.ts", "tests/regression_honk.test.ts"]:
        assert not rp.is_probe_path(p), p
    assert rp.is_test_path("tests/test_x.py") and rp.is_test_path("src/a.test.ts")
    assert rp.is_test_path("scripts/smoke-traffic.mjs") and not rp.is_test_path("src/traffic.ts")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("")
    (tmp_path / "tests" / "helper.txt").write_text("")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "smoke-traffic.mjs").write_text("")
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "car.test.ts").write_text("")
    found = rp.acceptance_files([f'"{PY}" -m pytest tests/ -q', "node scripts/smoke-traffic.mjs && npm run build"],
                                str(tmp_path), ["src/car.test.ts", "src/car.ts", "probe1.test.js"], ["src/missing.test.ts"])
    assert found == ["tests/helper.txt", "tests/test_a.py", "scripts/smoke-traffic.mjs", "src/car.test.ts"], \
        "everything under a test directory named by a command is part of the acceptance suite"
    assert rp.is_authorized("tests/test_a.py", ["tests/*.py"]) and rp.is_authorized("src/car.test.ts", ["car.test.ts"])
    assert rp.is_authorized("x/y.py", ["*"]) and not rp.is_authorized("tests/test_a.py", ["tests/test_b.py"])


def test_repair_state_round_trips_for_restart_recovery():
    rs = RepairState(phase_id="P2", checkpoint_attempt=1, repair_attempt=1, session_id="s-1",
                     guard=RepairGuardState(tool_calls=7, same_rounds=2, probe_files=["probe1.js"]),
                     best={"score": {"commands": 1, "failed_commands": 1, "failing": 1}, "tree": "abc", "round": 2},
                     protected={"tests/a.py": "deadbeef"}, pending_escalation="tool budget reached 30/30")
    again = RepairState.from_dict(json.loads(json.dumps(rs.to_dict())))
    assert again == rs


# ------------------------------------------------------------ git restore semantics (unit)

def test_restore_tree_preserves_user_changes_and_never_touches_ignored_or_generated_files(arepo):
    async def main():
        g = GitRepo(str(arepo))
        (arepo / "README.md").write_text("USER EDIT\n")  # pre-existing user change (tracked)
        (arepo / "notes_user.md").write_text("mine\n")  # pre-existing untracked user file
        impl(arepo, "a")
        best = await g.snapshot(True)
        impl(arepo, "a", "b", "c")  # the regression
        (arepo / "junk.py").write_text("J = 1\n")  # created after the best-known state
        (arepo / "ignored.log").write_text("log\n")
        (arepo / "scratch_smoke").mkdir()
        (arepo / "scratch_smoke" / "p.js").write_text("p\n")
        cur = await g.snapshot(True)
        restored = await g.restore_tree(best.tree, cur.tree)
        blob = subprocess.run(["git", "cat-file", "-p", f"{cur.tree}:junk.py"], cwd=arepo, capture_output=True,
                              text=True).stdout
        return restored, blob
    restored, blob = asyncio.run(main())
    assert sorted(restored) == [("D", "junk.py"), ("W", "impl.txt")]
    assert (arepo / "impl.txt").read_text() == "a\n" and not (arepo / "junk.py").exists()
    assert (arepo / "README.md").read_text() == "USER EDIT\n" and (arepo / "notes_user.md").read_text() == "mine\n"
    assert (arepo / "ignored.log").exists() and (arepo / "scratch_smoke" / "p.js").exists()
    assert blob == "J = 1\n", "the regressed state stays recoverable from its snapshot tree"
    head = subprocess.run(["git", "status", "--porcelain"], cwd=arepo, capture_output=True, text=True).stdout
    assert " M README.md" in head and "?? notes_user.md" in head, "the real index is untouched"


# ------------------------------------------------------------ the REPAIRING workflow (scheduler)

def test_v2_happy_path_pass_never_enters_repair(tmp_path, arepo):
    impl(arepo)  # every assertion passes from the start
    h = V2Harness(tmp_path, repair_settings())
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert "REPAIRING" not in [p["state"] for k, p in h.events if k == "state"]
    assert not h.orch.diagnoses and not e.run.pipeline_state.get("repair")
    assert not logs(h, "REPAIR") and all(not t["purpose"].startswith("repair") for t in h.worker.turns)


def test_retry_enters_repair_mode_with_a_fresh_session_and_bounded_context(tmp_path, arepo):
    worker = ScriptedWorker(arepo, [lambda w, r: impl(arepo)])  # the first repair turn fixes everything
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    impl_turn, repair = [t for t in worker.turns if t["phase"] == "P1"]
    assert impl_turn["purpose"] == "start" and repair["purpose"] == "repair"
    assert not repair["resumed"] and repair["session"] != impl_turn["session"], "never the implementation session"
    p = repair["prompt"]
    for part in ["REPAIR worker for phase P1", "COMPLETED SUBTASKS (done - do NOT redo)", "Mock fix: wire the feature",
                 "CURRENT VERIFICATION FAILURES", "FAIL: cars honked during the run", "tests/acceptance_check.py",
                 "git status", "run the failing verification command", "full phase verification",
                 "stopped after about 20 minutes or 30 tool calls", "at most 3 new probe"]:
        assert part in p, part
    assert "LATER PHASES" not in p and "HOW TO WORK" not in p and len(p) < len(impl_turn["prompt"]) * 2
    rl = logs(h, "REPAIR")
    assert any("entering repair mode after checkpoint RETRY" in x for x in rl)
    assert any(x.startswith("P1: starting fresh repair session 1") for x in rl)
    assert any("full phase verification passed - handing the repair to the checkpoint" in x for x in rl)
    atts = e.db.load_attempts(e.run.id)["P1"]
    assert [a.kind for a in atts] == ["execute", "repair"] and atts[1].review.decision == ReviewDecision.PASS
    assert all(v.passed for v in atts[1].verification) and not orch.diagnoses


def test_same_failure_fingerprint_three_times_escalates_even_when_the_worker_claims_progress(tmp_path, arepo):
    worker = ScriptedWorker(arepo, summary="Found the root cause - the final deadlock is fixed now.")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    purposes = [t["purpose"] for t in repair_turns(worker)]
    assert purposes == ["repair", "repair_round", "repair_round", "repair", "repair_round", "repair_round"], purposes
    assert len(orch.diagnoses) == 1 and "failure fingerprint unchanged for 3/3" in orch.diagnoses[0].reason
    rl = logs(h, "REPAIR")
    assert "P1: failure fingerprint unchanged 2/3 (3 failing assertion(s)/test(s) in 1 command(s))" in rl
    assert any("escalating blocker to Claude" in x for x in rl)
    assert any("DIAGNOSE_BLOCKER P1: REPAIR" in x for x in logs(h, "CLAUDE"))
    assert any("escalation budget used (1/1 diagnoses)" in x for x in rl), "bounded: no endless diagnose loop"


def test_repair_after_diagnosis_starts_fresh_with_concise_corrective_context(tmp_path, arepo):
    worker = ScriptedWorker(arepo, [None, None, None, lambda w, r: impl(arepo)])
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    turns = repair_turns(worker)
    after = turns[3]
    assert after["purpose"] == "repair" and not after["resumed"] and after["session"] != turns[2]["session"]
    for part in ["LEAD'S DIAGNOSIS OF THE BLOCKER", "the intersection lock is never released",
                 "avoid: [MOCK] adding more probe scripts", "src/junction.ts", "targeted checks after each change",
                 "Mock fix: wire the feature", "FAIL: intersection cleared"]:
        assert part in after["prompt"], part
    assert "LATER PHASES" not in after["prompt"] and turns[0]["prompt"] not in after["prompt"], "no transcript replay"
    packet = orch.diagnoses[0]
    assert packet.current.score.failing == 3 and packet.checkpoint["corrective_instructions"]
    assert "tests/acceptance_check.py" in packet.protected_tests and len(packet.rounds) == 3


def test_improvement_resets_both_counters_but_a_changed_fingerprint_only_the_unchanged_one(tmp_path, arepo):
    a, b = "no car stood still longer than 20 s (max 34.2 s)", "cars honked during the run (3 honks)"
    steps = [None, None,  # unchanged 1/3, 2/3; no improvement 1/4, 2/4
             lambda w, r: impl(arepo, a, b),  # improved 3 -> 2 (new best-known): both reset
             None,  # unchanged 1/3; no improvement 1/4
             lambda w, r: impl(arepo, a, "a different failure"),  # changed: unchanged-counter reset, 2/4
             None,  # unchanged 1/3; no improvement 3/4
             lambda w, r: impl(arepo)]
    worker = ScriptedWorker(arepo, steps, summary="found it")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(repair_max_tool_calls=100, repair_max_wall_seconds=3600), orch=orch,
                  worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert not orch.diagnoses and len(repair_turns(worker)) == 7
    rl = logs(h, "REPAIR")
    assert any("verification improved: 3 -> 2 failing" in x for x in rl)
    noimp = [x.split("state ")[1].split(" ")[0] for x in rl if "no improvement over the best-known state" in x]
    assert noimp == ["1/4", "2/4", "1/4", "2/4", "3/4"], "only the 3 -> 2 improvement reset it; 'changed' did not"
    unchanged = [x.split("unchanged ")[1].split(" ")[0] for x in rl if "failure fingerprint unchanged" in x]
    assert unchanged == ["1/3", "2/3", "1/3", "1/3"]


def test_wall_clock_guard_stops_a_repair_turn(tmp_path, arepo):
    worker = LoopingWorker(arepo, "wall")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(repair_max_wall_seconds=1), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert worker.durations and all(d < 6 for d in worker.durations), worker.durations
    assert any("wall-clock budget reached" in x and "stopping the repair turn" in x for x in logs(h, "REPAIR"))
    assert "wall-clock budget reached" in orch.diagnoses[0].reason


def test_tool_call_guard_stops_a_repair_turn(tmp_path, arepo):
    worker = LoopingWorker(arepo, "tools")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(repair_max_tool_calls=5), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert worker.calls and all(5 <= n <= 7 for n in worker.calls), worker.calls
    assert any("tool budget reached 5/5 - stopping the repair turn" in x for x in logs(h, "REPAIR"))
    assert "tool budget reached" in orch.diagnoses[0].reason


def test_best_known_state_survives_a_regression_and_is_restored(tmp_path, arepo):
    a = "no car stood still longer than 20 s (max 34.2 s)"
    three = ["no car stood still longer than 20 s (max 34.2 s)", "cars honked during the run (3 honks)",
             "intersection cleared"]
    steps = [lambda w, r: impl(arepo, a),  # 3 -> 1: new best-known state
             lambda w, r: (impl(arepo, *three), (arepo / "regress.py").write_text("R = 1\n")),  # 1 -> 3 (grace)
             None,  # still 3: restored to the 1-failure state
             lambda w, r: impl(arepo)]
    worker = ScriptedWorker(arepo, steps)
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    rl = logs(h, "REPAIR")
    assert any("new best-known state recorded (round 1)" in x for x in rl)
    reg = [x for x in rl if "regression detected: 1 -> 3 failing" in x]
    assert len(reg) == 2 and "grace 1/1" in reg[0], reg
    assert any("restoring best-known state (round 1: 1 failing" in x for x in rl)
    assert worker.seen_impl[3] == a + "\n", "the round after the restore starts from the best-known state"
    assert not (arepo / "regress.py").exists()
    prompts = [t["prompt"] for t in repair_turns(worker)]
    assert "REGRESSION: verification is now worse than the best-known state" in prompts[2]
    assert "REGRESSION RESTORED" in prompts[3] and "do not repeat the approach that regressed" in prompts[3]


def test_regression_restore_keeps_pre_existing_user_changes(tmp_path, arepo):
    (arepo / "README.md").write_text("user edit before the run\n")
    (arepo / "user_notes.md").write_text("untracked user file\n")
    (arepo / "ignored.log").write_text("ignored\n")
    a = "cars honked during the run (3 honks)"
    steps = [lambda w, r: impl(arepo, a),
             lambda w, r: (impl(arepo, a, "x", "y"), (arepo / "src_new.py").write_text("N = 1\n")),
             None,
             lambda w, r: impl(arepo)]
    worker = ScriptedWorker(arepo, steps)
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert (arepo / "README.md").read_text() == "user edit before the run\n"
    assert (arepo / "user_notes.md").read_text() == "untracked user file\n"
    assert (arepo / "ignored.log").read_text() == "ignored\n" and not (arepo / "src_new.py").exists()
    kept = next(x for x in logs(h, "REPAIR") if "the regressed state is kept as snapshot tree" in x)
    tree = kept.rsplit(" ", 1)[1]
    blob = subprocess.run(["git", "cat-file", "-p", f"{tree}:src_new.py"], cwd=arepo, capture_output=True, text=True)
    assert blob.stdout == "N = 1\n", "nothing is lost: the regressed files stay recoverable"


def test_protected_acceptance_test_modification_is_rejected_and_restored(tmp_path, arepo):
    def weaken(w, r):
        (arepo / "tests" / "acceptance_check.py").write_text("print('all assertions passed')\n")
    worker = ScriptedWorker(arepo, [weaken, lambda w, r: impl(arepo)])
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert (arepo / "tests" / "acceptance_check.py").read_text() == ACCEPTANCE
    assert any("protected acceptance test modification rejected: tests/acceptance_check.py" in x
               for x in logs(h, "REPAIR"))
    turns = repair_turns(worker)
    assert len(turns) == 2, "the weakened test never 'passed' the repair"
    assert "were REJECTED and reverted" in turns[1]["prompt"]
    assert "PROTECTED ACCEPTANCE TESTS" in turns[0]["prompt"] and "tests/acceptance_check.py" in turns[0]["prompt"]
    ck = next(c for c in orch.checkpoints if c.attempt.kind == "repair")
    assert any("rejected (reverted) edits of protected acceptance tests" in x for x in ck.repair_notes)


def test_explicit_claude_authorization_allows_an_acceptance_test_change(tmp_path, arepo):
    class AuthorizingOrchestrator(MockOrchestrator):
        async def checkpoint(self, package):
            r = await super().checkpoint(package)
            if r.decision == ReviewDecision.RETRY:
                r.authorized_test_changes = ["tests/acceptance_check.py"]
            return r
    fixed = ACCEPTANCE.replace("if l.strip()]", "if l.strip() and 'honked' not in l]")

    def fix_test_and_impl(w, r):
        (arepo / "tests" / "acceptance_check.py").write_text(fixed)
        impl(arepo, "cars honked during the run (3 honks)")  # now legitimately not an assertion
    worker = ScriptedWorker(arepo, [fix_test_and_impl])
    orch = AuthorizingOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert (arepo / "tests" / "acceptance_check.py").read_text() == fixed
    assert any("acceptance test change authorized by the lead: tests/acceptance_check.py" in x
               for x in logs(h, "REPAIR"))
    assert "The lead explicitly allowed changes to: tests/acceptance_check.py" in repair_turns(worker)[0]["prompt"]


def test_probe_file_proliferation_without_progress_escalates(tmp_path, arepo):
    worker = LoopingWorker(arepo, "probes")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(repair_max_tool_calls=100), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    assert worker.calls[0] in (4, 5), worker.calls
    assert "new probe/debug files without verification progress (max 3)" in orch.diagnoses[0].reason
    assert (arepo / "probe1.js").exists(), "probe files are only counted, never deleted"


def test_diagnose_blocked_stops_the_repair_loop_until_the_user_resumes(tmp_path, arepo):
    worker = ScriptedWorker(arepo)
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]}, diagnose_script=["BLOCKED"])
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)

    async def main():
        e = await h.plan_and_start(arepo, with_verification)
        assert e.sm.state == S.BLOCKED and e.run.resume_state == "REPAIRING"
        assert "diagnosed the stuck repair of P1 as BLOCKED" in e.run.status_message
        assert phase(e, "P1").status == TaskStatus.BLOCKED and len(repair_turns(worker)) == 3
        assert e.run.pipeline_state["repair"]["P1"]["status"] == "blocked"
        worker.script = [lambda w, r: impl(arepo)]
        await e.resume("Cars must yield to the right; honking is allowed")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    resumed = repair_turns(worker)[3]
    assert resumed["purpose"] == "repair" and not resumed["resumed"], "fresh session after the user's decision"
    assert "Cars must yield to the right" in resumed["prompt"]
    assert len(orch.diagnoses) == 1


def test_restart_during_repairing_keeps_the_safe_repair_state(tmp_path, arepo):
    s = repair_settings()
    worker = LoopingWorker(arepo, "block")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, s, orch=orch, worker=worker, db="repair_crash.db")

    def s1_done(req):  # the implementation turn reports S1 through its todo list
        if req.purpose == "start" and req.phase_id == "P1":
            worker.on_todo([{"text": "S1: Survey architecture", "completed": True}])
    worker.file_writer = s1_done

    async def main():
        e = h.engine
        await e.plan("g", str(arepo), pipeline="v2")
        await h.wait_idle()
        with_verification(e)
        await e.start()
        while not (repair_turns(worker) and worker.calls == [] and
                   e.run.pipeline_state.get("repair", {}).get("P1", {}).get("guard", {}).get("inflight_tool_calls", 0)
                   >= 3):
            await asyncio.sleep(0.01)
        await e.stop()  # the repair turn is killed mid-flight
        await h.wait_idle()
        assert e.sm.state == S.STOPPED and e.run.resume_state == "REPAIRING"
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(main())
    conn = sqlite3.connect(tmp_path / "repair_crash.db")  # make it look like a hard kill mid-turn
    conn.execute("UPDATE runs SET state='REPAIRING', resume_state=NULL WHERE id=?", (run_id,))
    conn.execute("UPDATE task_attempts SET finished_at=NULL, interrupted=0 WHERE run_id=? AND task_id='P1' "
                 "AND kind='repair'", (run_id,))
    conn.commit()
    conn.close()

    worker2 = ScriptedWorker(arepo, [lambda w, r: impl(arepo)])
    h2 = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0), worker=worker2, db="repair_crash.db")

    async def main2():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.STOPPED and e.run.resume_state == "REPAIRING"
        rs = RepairState.from_dict(e.run.pipeline_state["repair"]["P1"])
        assert rs.checkpoint_attempt == 1 and rs.session_id == "loop-session-2" and rs.round_no == 0
        assert rs.best["score"]["failing"] == 3 and rs.best["tree"], "best-known state survives the restart"
        assert "tests/acceptance_check.py" in rs.protected and rs.guard.inflight_tool_calls >= 3
        assert phase(e, "P1").subtasks[0].done, "completed subtasks survive"
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(main2())
    assert e.sm.state == S.COMPLETED
    first = worker2.turns[0]
    assert (first["phase"], first["purpose"], first["resumed"]) == ("P1", "repair_resume", True), \
        "the dead turn is resumed in the repair session - neither the phase nor the repair restarts"
    assert first["session"] == "loop-session-2" and "RESUMING THE REPAIR of phase P1" in first["prompt"]
    assert not any(t["purpose"] == "start" and t["phase"] == "P1" for t in worker2.turns)
    assert any("continuing repair attempt 1 after an interruption" in x and "3 tool call(s)" in x
               for x in logs(h2, "REPAIR")), "the killed turn's tool calls still count"
    assert phase(e, "P1").status == TaskStatus.PASSED


# ------------------------------------------------------------ DIAGNOSE_BLOCKER (packet + Claude adapter)

def _phase():
    return parse_phases([{"id": "P2", "title": "Traffic", "description": "Cars never deadlock",
                          "subtasks": [{"id": "S1", "title": "Junction"}, {"id": "S2", "title": "Honk"}],
                          "acceptance_criteria": ["no car stands still > 20 s"],
                          "verification_commands": ["node smoke.mjs"]}])[0]


def _packet(diff_kb=200, rounds=40):
    cur = rp.evaluate_verification([VerificationResult("node smoke.mjs", 1, "FAIL: no car stood still\n" * 3)])
    return rp.DiagnosisPacket(
        phase=_phase(), reason="tool budget reached 30/30", repo_path="C:/r", checkpoint={
            "summary": "deadlock remains", "problems": ["cars freeze at junction"], "corrective_instructions": ["fix"]},
        current=cur, best=cur, changed_files=[f"src/f{i}.ts" for i in range(100)], diff="+x\n" * (diff_kb * 350),
        rounds=[{"n": i, "trend": "UNCHANGED", "summary": "1 failing", "worker": "found the root cause " * 50}
                for i in range(rounds)], protected_tests=["tests/smoke.mjs"])


def test_diagnosis_packet_is_bounded_and_keeps_the_evidence():
    prompt, stats = rp.build_diagnosis_prompt(_packet(), rp.DiagnosisLimits(30000, 12000))
    assert stats["prompt_bytes"] <= 30000 and stats["diff_original_bytes"] > 60000
    for part in ["DIAGNOSE_BLOCKER", "Do NOT edit files", "tool budget reached 30/30", "cars freeze at junction",
                 "CURRENT FAILING ASSERTIONS", "BEST-KNOWN STATE", "tests/smoke.mjs", "RELEVANT DIFF", "truncated",
                 "ANSWER with the structured schema"]:
        assert part in prompt, part
    tiny, st = rp.build_diagnosis_prompt(_packet(), rp.DiagnosisLimits(5000, 2000))
    assert st["prompt_bytes"] <= 5000 and "ANSWER with the structured schema" in tiny


FAKE_CLAUDE = r'''
import json, sys
sys.stdin.read()
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.03,
                  "usage": {}, "structured_output": payload}), flush=True)
'''


def _claude(tmp_path, payload):
    from app.config.settings import ClaudeSettings
    from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
    (tmp_path / "fake_claude.py").write_text(FAKE_CLAUDE)
    (tmp_path / "payload.json").write_text(json.dumps(payload))
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused"))
    seen = []

    async def argv(limits, schema):
        seen.append((limits, schema))
        return [PY, str(tmp_path / "fake_claude.py"), str(tmp_path / "payload.json")]
    prov._build_argv = argv
    return prov, seen


def test_claude_diagnose_blocker_returns_strict_structured_guidance(tmp_path):
    prov, seen = _claude(tmp_path, {"diagnosis": "lock never released", "recommended_actions": ["release it"],
                                    "avoid": ["probe scripts"], "target_files": ["src/junction.ts"],
                                    "target_tests": ["node smoke.mjs"], "status": "REPAIR"})
    pk = _packet(diff_kb=5, rounds=3)
    pk.repo_path = str(tmp_path)
    d = asyncio.run(prov.diagnose_blocker(pk))
    assert d == Diagnosis("REPAIR", "lock never released", ["release it"], ["probe scripts"], ["src/junction.ts"],
                          ["node smoke.mjs"])
    limits, schema = seen[-1]
    assert limits.purpose == "DIAGNOSE_BLOCKER" and limits.tools == "" and limits.max_tool_calls == 0
    assert limits.timeout == 180 and limits.max_budget_usd == 0.30 and limits.max_turns == 2
    assert schema is DIAGNOSIS_JSON_SCHEMA and schema["additionalProperties"] is False
    assert schema["properties"]["status"]["enum"] == ["REPAIR", "BLOCKED"]
    assert prov.calls[-1].kind == "DIAGNOSE_BLOCKER"
    prov, _ = _claude(tmp_path, {"diagnosis": "x", "recommended_actions": [], "avoid": [], "target_files": [],
                                 "target_tests": [], "status": "MAYBE"})
    with pytest.raises(OrchestratorError, match="invalid diagnosis"):
        asyncio.run(prov.diagnose_blocker(pk))


def test_repair_prompt_uses_the_exact_codebase_memory_project():
    spec = RepairSpec(phase=_phase(), repo_path="C:/Users/u/SampleGame", codebase_memory_available=True,
                      codebase_memory_project="C-Users-u-SampleGame")
    p = build_repair_prompt(spec)
    assert '"C-Users-u-SampleGame"' in p and "never guess a project name" in p and "C:/Users/u/SampleGame" in p
    assert "not available (degraded mode)" in build_repair_prompt(RepairSpec(phase=_phase(), repo_path="C:/r"))


# ------------------------------------------------------------ structured subtask completion

def test_subtasks_complete_only_from_structured_events_and_persist_immediately(tmp_path):
    r = tmp_path / "plain"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "a.txt").write_text("a")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    worker = MockWorker(delay=0.0, steps=0)
    h = V2Harness(tmp_path, repair_settings(), worker=worker)
    persisted = []

    def writer(req):
        if req.phase_id != "P1":
            return
        worker.on_todo([{"text": "Explore repo", "completed": True}, {"text": "S1: Survey architecture",
                                                                      "completed": True},
                        {"text": "Verify: build", "completed": True}, {"text": "S2: Add data model", "completed": False}])
        row = h.engine.db.load_tasks(h.engine.run.id, phases=True)[0]  # the database, not memory
        persisted.append([(s.id, s.done, s.completed_via) for s in row.subtasks])
    worker.file_writer = writer
    real = worker.run_turn

    async def run_turn(req):
        out = await real(req)
        if req.phase_id == "P1":  # the final report CLAIMS everything, without structured signals
            out.summary = "PHASE STATUS: COMPLETE\nSUBTASKS:\n[x] S1\n[x] S2 done\n[x] S3 done\nSUMMARY: S2 is done"
            out.todo = []
        return out
    worker.run_turn = run_turn
    e = asyncio.run(h.plan_and_start(r))
    assert e.sm.state == S.COMPLETED
    assert persisted[0] == [("S1", True, "todo"), ("S2", False, ""), ("S3", False, "")], \
        "S1 persisted at once; Explore/Verify helpers never count as planned subtasks"
    text = logs(h)
    assert "P1: subtask S1 done (worker todo list) - subtasks 1/3" in text
    assert any("inconsistent progress - the worker reported COMPLETE but sent no structured completion (todo list) "
               "for S2, S3; not marking them done (subtasks 1/3); its final message claims them done" in x for x in text)
    p1 = e.db.load_run(e.run.id).plan.task("P1")
    assert [(s.id, s.completed_via) for s in p1.subtasks] == [("S1", "todo"), ("S2", "checkpoint"),
                                                              ("S3", "checkpoint")], "PASS reconciles, marked as such"


# ------------------------------------------------------------ edge case 1: "changed" is not progress

def test_no_improvement_guard_limit_and_default():
    o = make_settings().orchestration
    assert o.repair_max_no_improvement_rounds == 4
    assert rp.guard_trip(RepairGuardState(no_improvement_rounds=3), o) == ""
    assert rp.guard_trip(RepairGuardState(no_improvement_rounds=4), o) == \
        "no verification improvement over the best-known state for 4/4 rounds"
    again = RepairGuardState.from_dict(json.loads(json.dumps(RepairGuardState(no_improvement_rounds=2).__dict__)))
    assert again.no_improvement_rounds == 2, "persisted for restart recovery"


def _no_improvement_run(tmp_path, arepo, rounds):
    """Baseline 3 failing; round 1 improves to a new best-known state; `rounds` then never beat it."""
    steps = [lambda w, r: impl(arepo, "stuck at junction", "x")] + [
        (lambda names: (lambda w, r: impl(arepo, *names)))(names) for names in rounds] + [lambda w, r: impl(arepo)]
    worker = ScriptedWorker(arepo, steps, summary="Found the final deadlock - fixed now.")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(repair_max_tool_calls=100, repair_max_wall_seconds=3600), orch=orch,
                  worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    return h, orch, worker


def test_alternating_failure_fingerprints_cannot_evade_the_repair_guard(tmp_path, arepo):
    # Every round differs from the previous one, so the unchanged-fingerprint guard never fires.
    f1, f2 = ("stuck at junction", "y"), ("stuck at junction", "x")
    h, orch, worker = _no_improvement_run(tmp_path, arepo, [f1, f2, f1, f2])
    rl = logs(h, "REPAIR")
    assert not any("failure fingerprint unchanged" in x for x in rl), "the old guard is evaded by alternation"
    assert len(orch.diagnoses) == 1, "... but the no-improvement guard escalates"
    assert orch.diagnoses[0].reason == "no verification improvement over the best-known state for 4/4 rounds"
    assert len(orch.diagnoses[0].rounds) == 5, "escalated right after 1 improving + 4 non-improving rounds"
    assert [x.split("state ")[1] for x in rl if "no improvement over the best-known state" in x] == \
        [f"{n}/4 (different failures, score not better)" for n in (1, 2, 3, 4)]
    turns = repair_turns(worker)
    assert [t["purpose"] for t in turns] == ["repair"] + ["repair_round"] * 4 + ["repair"], \
        "the escalation starts a fresh session with the diagnosis"
    assert "no improvement over the best-known state for 3/4 rounds" in turns[4]["prompt"]


def test_unique_new_fingerprints_every_round_cannot_evade_the_repair_guard(tmp_path, arepo):
    # "Found the root cause" with a new, incompatible failure set every round.
    rounds = [("stuck at junction", f"new hypothesis {i} failed") for i in range(1, 5)]
    h, orch, worker = _no_improvement_run(tmp_path, arepo, rounds)
    assert len(orch.diagnoses) == 1 and "no verification improvement" in orch.diagnoses[0].reason
    assert not any("failure fingerprint unchanged" in x for x in logs(h, "REPAIR"))


def test_score_oscillation_does_not_count_as_improvement(tmp_path, arepo):
    # 2 -> 3 -> 2 -> 3 -> 2: every other round "improves" on the previous round, but never beats the
    # best-known 2-failure state; the regression grace (1 round) never expires either.
    worse = ("stuck at junction", "x", "honk")
    best = ("stuck at junction", "x")
    h, orch, worker = _no_improvement_run(tmp_path, arepo, [worse, best, worse, best])
    rl = logs(h, "REPAIR")
    assert not any("restoring best-known state" in x for x in rl), "the regression restore is evaded"
    assert len(orch.diagnoses) == 1 and orch.diagnoses[0].reason.endswith("for 4/4 rounds")
    assert [x.split("(")[-1] for x in rl if "no improvement over the best-known state" in x] == \
        ["worse)", "better than the last round, not than the best-known state)"] * 2


# ------------------------------------------------------------ edge case 2: worker-created files

def test_git_restore_is_byte_exact_only_with_the_exact_blobs(tmp_path):
    r = tmp_path / "eol"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    git(r, "config", "core.autocrlf", "true")  # deterministic: EOL filters active on every machine
    (r / "a.py").write_bytes(b"A = 1\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    lf = b"def helper():\n    return 1\n"

    async def main():
        g = GitRepo(str(r))
        (r / "helper.py").write_bytes(lf)
        best = await g.snapshot(True)
        exact = await g.hash_files(["helper.py"], write=True)
        (r / "helper.py").unlink()
        cur = await g.snapshot(True)
        await g.restore_tree(best.tree, cur.tree)
        from_tree = (r / "helper.py").read_bytes()
        (r / "helper.py").unlink()
        restored = await g.restore_tree(best.tree, cur.tree, exact=exact)
        return from_tree, restored
    from_tree, restored = asyncio.run(main())
    assert from_tree == b"def helper():\r\n    return 1\r\n", "a snapshot tree alone re-applies EOL filters"
    assert restored == [("W", "helper.py")] and (r / "helper.py").read_bytes() == lf


def test_best_known_restore_recreates_a_worker_created_file_exactly(tmp_path, arepo):
    git(arepo, "config", "core.autocrlf", "true")
    user = arepo / "user_scratch.md"
    user.write_bytes(b"my own notes\nnot tracked\n")  # pre-existing untracked user file
    user_before = (user.read_bytes(), user.stat().st_mtime_ns)
    helper = arepo / "src" / "junction_helper.py"
    v1 = "def release(lock):\n    lock.owner = None  # verified fix ✓\n    return True".encode("utf-8")  # LF, no EOF NL
    seen = []

    def create_helper(w, r):  # a legitimate new source file; this state verifies best (1 failing)
        helper.parent.mkdir(exist_ok=True)
        helper.write_bytes(v1)
        impl(arepo, "cars honked during the run (3 honks)")

    def change_helper(w, r):  # changes it and regresses (grace round)
        helper.write_bytes(b"def release(lock):\n    return False\n")
        impl(arepo, "a", "b", "c")

    def delete_helper(w, r):  # deletes it, still regressed -> the best-known state is restored
        helper.unlink()

    def after_restore(w, r):
        seen.append(helper.read_bytes() if helper.exists() else None)
        impl(arepo)
    worker = ScriptedWorker(arepo, [create_helper, change_helper, delete_helper, after_restore])
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, repair_settings(), orch=orch, worker=worker)
    e = run(h, arepo)
    assert e.sm.state == S.COMPLETED
    rl = logs(h, "REPAIR")
    assert any("new best-known state recorded (round 1)" in x for x in rl)
    restored = next(x for x in rl if x.startswith("P1: restored "))
    assert "src/junction_helper.py" in restored and "user_scratch.md" not in restored
    assert seen == [v1], "the deleted file is recreated with the exact bytes of the best-known version"
    assert helper.read_bytes() == v1
    assert (user.read_bytes(), user.stat().st_mtime_ns) == user_before, "the user's untracked file is untouched"
