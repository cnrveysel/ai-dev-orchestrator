import asyncio
import os
import sys
import textwrap
import time

import pytest

from app.config.discovery import ExecutableRejected, validate_worker_executable
from app.integrations.ninfer_metrics import DecodeRateTracker, parse_prometheus
from app.models.plan import Plan, PlanValidationError
from app.models.review import Review, ReviewDecision
from app.models.task import TaskStatus
from app.orchestration.safety import destructive_reason
from app.orchestration.state_machine import IllegalTransition, RunState, StateMachine
from app.orchestration.task_graph import TaskGraph
from app.workers.process_runner import ProcessRunner, ProcessLaunchError


def plan_dict(**over):
    d = {"goal": "g", "summary": "s", "tasks": [
        {"id": "T001", "title": "a", "description": "", "acceptance_criteria": []},
        {"id": "T002", "title": "b", "description": "", "acceptance_criteria": [], "dependencies": ["T003"]},
        {"id": "T003", "title": "c", "description": "", "acceptance_criteria": [], "dependencies": ["T001"]},
    ]}
    d.update(over)
    return d


# ------------------------------------------------------------ plan / graph

def test_plan_topological_order_and_single_ready():
    plan = Plan.from_dict(plan_dict())
    g = TaskGraph(plan.tasks)
    assert [t.id for t in g.topological_order()] == ["T001", "T003", "T002"]
    assert g.next_ready().id == "T001"
    g.refresh_ready_flags()
    assert [t.status for t in plan.tasks].count(TaskStatus.READY) == 1
    plan.task("T001").status = TaskStatus.PASSED
    assert g.next_ready().id == "T003"


def test_plan_rejects_cycles_unknown_deps_duplicates():
    d = plan_dict()
    d["tasks"][0]["dependencies"] = ["T002"]
    with pytest.raises(PlanValidationError):
        Plan.from_dict(d)
    d = plan_dict()
    d["tasks"][0]["dependencies"] = ["T999"]
    with pytest.raises(PlanValidationError):
        Plan.from_dict(d)
    d = plan_dict()
    d["tasks"][1]["id"] = "T001"
    with pytest.raises(PlanValidationError):
        Plan.from_dict(d)
    with pytest.raises(PlanValidationError):
        Plan.from_dict({"goal": "g", "tasks": []})


def test_active_task_invariant():
    plan = Plan.from_dict(plan_dict())
    plan.tasks[0].status = TaskStatus.RUNNING
    plan.tasks[1].status = TaskStatus.REVIEWING
    with pytest.raises(RuntimeError):
        TaskGraph(plan.tasks).active_task()


def test_review_parsing():
    r = Review.from_dict({"decision": "retry", "summary": "x", "problems": ["p"]})
    assert r.decision == ReviewDecision.RETRY
    with pytest.raises(ValueError):
        Review.from_dict({"decision": "MAYBE"})


# ------------------------------------------------------------ state machine

def test_state_machine_transitions():
    seen = []
    sm = StateMachine(on_transition=lambda o, n, r: seen.append((o, n)))
    sm.transition(RunState.PLANNING)
    sm.transition(RunState.PLAN_READY)
    sm.transition(RunState.STARTING_TASK)
    with pytest.raises(IllegalTransition):
        sm.transition(RunState.ADVANCING)  # must go through worker/verify/review
    sm.transition(RunState.WORKER_RUNNING)
    sm.transition(RunState.PAUSED)
    sm.transition(RunState.WORKER_RUNNING)  # resume
    assert len(seen) == 6


# ------------------------------------------------------------ discovery

def test_worker_executable_validation(tmp_path):
    for bad in ["codex.cmd", "codex.exe", "codex-menu.cmd", "codex-lmstudio.exe"]:
        p = tmp_path / bad
        p.write_bytes(b"MZ")
        with pytest.raises(ExecutableRejected):
            validate_worker_executable(str(p))
    npm = tmp_path / "npm"
    npm.mkdir()
    (npm / "codex-ninfer.exe").write_bytes(b"MZ")
    with pytest.raises(ExecutableRejected):
        validate_worker_executable(str(npm / "codex-ninfer.exe"))
    with pytest.raises(ExecutableRejected):
        validate_worker_executable("")
    with pytest.raises(ExecutableRejected):
        validate_worker_executable(str(tmp_path / "missing" / "codex-ninfer.exe"))
    good = tmp_path / "codex-ninfer.exe"
    good.write_bytes(b"MZ")
    assert validate_worker_executable(str(good)) == str(good)


def test_normal_codex_on_path_is_rejected():
    import shutil
    which = shutil.which("codex")
    if not which:
        pytest.skip("no codex on PATH")
    with pytest.raises(ExecutableRejected):
        validate_worker_executable(which)


# ------------------------------------------------------------ metrics

def test_metrics_rate_uses_deltas():
    text = textwrap.dedent("""
        # HELP x
        llamacpp:tokens_predicted_total 1000
        llamacpp:tokens_predicted_seconds_total 20.0
    """)
    v = parse_prometheus(text)
    assert v["llamacpp:tokens_predicted_total"] == 1000
    t = DecodeRateTracker()
    assert t.update(1000, 20.0) is None  # first sample: no delta yet (lifetime avg would be 50)
    assert t.update(1800, 30.0) == pytest.approx(80.0)
    assert t.update(1800, 30.0) is None  # idle
    assert t.update(10, 0.5) is None  # counter reset


# ------------------------------------------------------------ safety

@pytest.mark.parametrize("cmd", ["git reset --hard", "git clean -fdx", "git push --force origin x",
                                 "rm -rf build", "rmdir /s /q Binaries", "del /q *.obj",
                                 "Remove-Item -Recurse x", "setx PATH foo"])
def test_destructive_commands_flagged(cmd):
    assert destructive_reason(cmd)


@pytest.mark.parametrize("cmd", ["python -m pytest", "dotnet build", "git status", "git diff --stat",
                                 "msbuild Foo.sln /p:Configuration=Development"])
def test_safe_commands_allowed(cmd):
    assert destructive_reason(cmd) is None


# ------------------------------------------------------------ process runner

PY = sys.executable


def test_runner_streams_live_before_exit():
    script = "import sys,time; print('first', flush=True); time.sleep(1.5); print('second', flush=True); " \
             "sys.stderr.write('err\\n'); sys.exit(3)"
    got = []

    async def main():
        t0 = time.monotonic()
        res = await ProcessRunner().run([PY, "-c", script], on_stdout=lambda l: got.append((l, time.monotonic() - t0)),
                                        on_stderr=lambda l: got.append(("E:" + l, 0)))
        return res

    res = asyncio.run(main())
    assert res.exit_code == 3
    assert res.stdout.splitlines() == ["first", "second"]
    first_time = next(t for l, t in got if l == "first")
    assert first_time < 1.2, "stdout must be delivered live, not after exit"
    assert ("E:err", 0) in got


def test_runner_stdin_env_and_no_global_env_change():
    script = "import os,sys; print(sys.stdin.read().strip()); print(os.environ.get('CODEX_HOME'))"
    before = os.environ.get("CODEX_HOME")
    res = asyncio.run(ProcessRunner().run([PY, "-c", script], stdin_data="héllo prompt\n",
                                          env_overrides={"CODEX_HOME": r"C:\x\.codex-ninfer"}))
    assert res.stdout.splitlines() == ["héllo prompt", r"C:\x\.codex-ninfer"]
    assert os.environ.get("CODEX_HOME") == before


def test_runner_timeout_kills():
    res = asyncio.run(ProcessRunner(grace_seconds=1).run([PY, "-c", "import time; time.sleep(60)"], timeout=1))
    assert res.timed_out and res.duration < 20


def test_runner_cancel_kills_process():
    async def main():
        r = ProcessRunner(grace_seconds=1)
        task = asyncio.ensure_future(r.run([PY, "-c", "import time; print('up', flush=True); time.sleep(60)"],
                                           on_stdout=lambda l: None))
        await asyncio.sleep(1.0)
        assert r.running
        t0 = time.monotonic()
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
        return time.monotonic() - t0

    assert asyncio.run(main()) < 20


def test_runner_refuses_batch_files(tmp_path):
    with pytest.raises(ProcessLaunchError):
        asyncio.run(ProcessRunner().run([str(tmp_path / "launcher.cmd")]))


def test_run_shell_preserves_quotes(tmp_path):
    d = tmp_path / "dir with space"
    d.mkdir()
    (d / "f.txt").write_text("x")
    res = asyncio.run(ProcessRunner().run_shell(f'dir /b "{d}"' if sys.platform == "win32" else f'ls "{d}"'))
    assert res.exit_code == 0 and "f.txt" in res.stdout


def test_runner_closes_stdin_before_waiting_for_exit():
    # The child only exits after it sees EOF on stdin (like `claude --print` reading its prompt).
    script = "import sys; data = sys.stdin.buffer.read(); print('got', len(data), flush=True)"
    closed = []
    res = asyncio.run(ProcessRunner().run([PY, "-c", script], stdin_data="x" * 300_000,
                                          on_stdin_closed=closed.append, timeout=30))
    assert res.exit_code == 0 and not res.timed_out
    assert res.stdout.strip() == "got 300000"
    assert closed == [300_000]


def test_runner_returns_when_descendant_holds_stdout(tmp_path):
    # Child spawns a long-lived grandchild that inherits stdout, then exits at once.
    pidfile = tmp_path / "gc.pid"
    script = ("import subprocess, sys; "
              "p = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(60)'], stdout=sys.stdout); "
              f"open(r'{pidfile}', 'w').write(str(p.pid)); print('child done', flush=True)")
    try:
        t0 = time.monotonic()
        res = asyncio.run(ProcessRunner(drain_seconds=1).run([PY, "-c", script], stdin_data="", timeout=30))
        elapsed = time.monotonic() - t0
        assert res.exit_code == 0 and not res.timed_out
        assert "child done" in res.stdout
        assert res.pipes_held_open
        assert elapsed < 15, f"run() waited {elapsed:.1f}s on a pipe held by a descendant"
    finally:
        if pidfile.exists():
            import subprocess as sp
            sp.run(["taskkill", "/PID", pidfile.read_text(), "/F"], capture_output=True)


def test_runner_stdin_write_cannot_hang_past_timeout():
    # Child never reads stdin; a large payload would block the write forever without a deadline.
    t0 = time.monotonic()
    res = asyncio.run(ProcessRunner(grace_seconds=1).run([PY, "-c", "import time; time.sleep(60)"],
                                                         stdin_data="y" * 20_000_000, timeout=3))
    assert res.timed_out and time.monotonic() - t0 < 25


def test_runner_terminate_with_reason():
    async def main():
        r = ProcessRunner(grace_seconds=1)

        def on_line(line):
            if line == "up":
                asyncio.get_running_loop().create_task(r.terminate("usage limit"))
        return await r.run([PY, "-c", "import time; print('up', flush=True); time.sleep(60)"],
                           on_stdout=on_line, timeout=30)

    res = asyncio.run(main())
    assert res.terminated_reason == "usage limit" and not res.cancelled and not res.timed_out
