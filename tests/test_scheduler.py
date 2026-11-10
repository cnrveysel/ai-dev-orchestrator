"""End-to-end engine tests with the mock orchestrator/worker."""
import asyncio
import subprocess
import time

import pytest

from app.config.settings import Settings
from app.models.task import TaskStatus
from app.orchestration.orchestrator import Engine
from app.orchestration.state_machine import RunState


def make_settings(**orch):
    s = Settings()
    s.integrations.metrics_enabled = False
    s.integrations.codebase_memory_executable = ""
    s.orchestration.mock_step_delay = 0.01
    for k, v in orch.items():
        setattr(s.orchestration, k, v)
    return s


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=r, check=True)
    (r / "README.md").write_text("x\n")
    subprocess.run(["git", "add", "-A"], cwd=r, check=True)
    subprocess.run(["git", "-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "i"], cwd=r, check=True)
    return r


class Harness:
    def __init__(self, tmp_path, settings):
        self.events = []
        self.engine = Engine(settings, lambda k, p: self.events.append((k, p)), str(tmp_path / "t.db"))

    def states(self):
        return [p["state"] for k, p in self.events if k == "state"]

    async def wait_idle(self, timeout=30):
        t0 = time.monotonic()
        while self.engine._busy is not None and not self.engine._busy.done():
            await asyncio.sleep(0.02)
            assert time.monotonic() - t0 < timeout, "engine did not settle"


def test_full_run_with_retry_loop_is_sequential(tmp_path, repo):
    h = Harness(tmp_path, make_settings(mock_retry_task_ids=["T002"]))

    async def main():
        e = h.engine
        await e.plan("Build inventory", str(repo))
        await h.wait_idle()
        assert e.sm.state == RunState.PLAN_READY
        worker_calls_before_start = [k for k, _ in h.events if k == "attempt"]
        assert not worker_calls_before_start, "PLAN must not start coding"
        await e.start()
        await h.wait_idle()
        return e

    e = asyncio.run(main())
    assert e.sm.state == RunState.COMPLETED
    worker = e.scheduler.d.worker
    # T002 is retried (RETRY on attempt 1) BEFORE T003 starts
    assert worker.invocations == ["T001", "T002", "T002", "T003", "T004"]
    assert worker.max_active == 1
    assert all(t.status == TaskStatus.PASSED for t in e.run.plan.tasks)
    assert e.run.plan.task("T002").attempts == 2
    assert e.run.plan.task("T001").result is not None
    st = h.states()
    assert "RETRY_PENDING" in st and st[-1] == "COMPLETED"
    # persisted and reloadable
    run = e.db.load_run(e.run.id)
    assert run.state == "COMPLETED" and all(t.status == TaskStatus.PASSED for t in run.plan.tasks)
    attempts = e.db.load_attempts(e.run.id)
    assert [a.review.decision.value for a in attempts["T002"]] == ["RETRY", "PASS"]


def test_retry_limit_blocks_and_does_not_skip(tmp_path, repo):
    h = Harness(tmp_path, make_settings(max_attempts=2, mock_retry_task_ids=[]))

    async def main():
        e = h.engine
        await e.plan("g", str(repo))
        await h.wait_idle()
        # Make T001 always fail verification so review keeps saying RETRY
        e.run.plan.task("T001").verification_commands = ["exit 1"]
        await e.start()
        await h.wait_idle()
        assert e.sm.state == RunState.BLOCKED
        t1 = e.run.plan.task("T001")
        assert t1.status == TaskStatus.FAILED and t1.attempts == 2
        assert e.run.plan.task("T002").status in (TaskStatus.PENDING, TaskStatus.READY)
        assert e.scheduler.d.worker.invocations == ["T001", "T001"]
        # User resumes: exactly one more attempt is granted, still no skipping
        await e.resume("try harder")
        await h.wait_idle()
        assert e.sm.state == RunState.BLOCKED
        assert e.scheduler.d.worker.invocations == ["T001"]  # new worker instance per resume
        assert e.run.plan.task("T001").attempts == 3

    asyncio.run(main())


def test_pause_then_resume(tmp_path, repo):
    s = make_settings(mock_retry_task_ids=[])
    s.orchestration.mock_step_delay = 0.05
    h = Harness(tmp_path, s)

    async def main():
        e = h.engine
        await e.plan("g", str(repo))
        await h.wait_idle()
        await e.start()
        while e.scheduler.d.worker.active == 0:
            await asyncio.sleep(0.01)
        await e.pause()  # while worker is running: it must finish its stage, not be killed
        await h.wait_idle()
        assert e.sm.state == RunState.PAUSED
        assert e.run.resume_state == "VERIFYING"
        passed_before = [t.id for t in e.run.plan.tasks if t.status == TaskStatus.PASSED]
        assert passed_before == []
        attempts = e.db.load_attempts(e.run.id)["T001"]
        assert attempts[0].exit_code == 0 and not attempts[0].interrupted
        await e.resume()
        await h.wait_idle()
        assert e.sm.state == RunState.COMPLETED
        # T001 was not re-executed on resume
        assert e.scheduler.d.worker.invocations == ["T002", "T003", "T004"]

    asyncio.run(main())


def test_stop_interrupts_and_resume_continues_without_repeating_passed(tmp_path, repo):
    s = make_settings(mock_retry_task_ids=[])
    h = Harness(tmp_path, s)

    async def main():
        e = h.engine
        await e.plan("g", str(repo))
        await h.wait_idle()
        e.settings.orchestration.mock_step_delay = 0.01
        await e.start()
        while e.run.plan.task("T002").status != TaskStatus.RUNNING:
            await asyncio.sleep(0.005)
        e.scheduler.d.worker.delay = 5  # make T002 long-running
        await asyncio.sleep(0.05)
        await e.stop()
        await h.wait_idle()
        assert e.sm.state == RunState.STOPPED and e.run.interrupted
        assert e.run.resume_state == "WORKER_RUNNING"
        assert e.run.plan.task("T001").status == TaskStatus.PASSED
        assert e.run.plan.task("T002").status == TaskStatus.PAUSED
        a = e.db.load_attempts(e.run.id)["T002"][-1]
        assert a.interrupted
        run_id = e.run.id
        await e.shutdown()
        return run_id

    run_id = asyncio.run(main())

    # "Restart the application": a new engine on the same database.
    h2 = Harness(tmp_path, s)

    async def main2():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == RunState.STOPPED
        await e.resume()
        await h2.wait_idle()
        assert e.sm.state == RunState.COMPLETED
        assert e.scheduler.d.worker.invocations == ["T002", "T003", "T004"]  # T001 not repeated
        assert e.run.plan.task("T002").attempts == 1  # interrupted attempt not counted

    asyncio.run(main2())


def test_real_worker_requires_verified_adapter(tmp_path, repo):
    s = make_settings(worker_mode="ninfer")
    s.worker.executable = ""
    h = Harness(tmp_path, s)

    async def main():
        e = h.engine
        await e.plan("g", str(repo))
        await h.wait_idle()
        with pytest.raises(Exception, match="(?i)no fallback|not configured"):
            await e.start()
        assert e.sm.state == RunState.PLAN_READY

    asyncio.run(main())


def test_claude_mode_requires_smoke_test(tmp_path, repo):
    s = make_settings(orchestrator_mode="claude")
    s.claude.executable = __file__  # exists but not a verified claude exe
    h = Harness(tmp_path, s)

    async def main():
        with pytest.raises(Exception):
            await h.engine.plan("g", str(repo))

    asyncio.run(main())


def test_interrupted_runs_marked_on_startup(tmp_path, repo):
    h = Harness(tmp_path, make_settings())
    e = h.engine
    from app.models.run import Run
    run = Run(id=None, goal="g", repo_path=str(repo), state="WORKER_RUNNING")
    e.db.create_run(run)
    e.db.update_run(run)
    e.db.close()
    h2 = Harness(tmp_path, make_settings())
    r = h2.engine.db.load_run(run.id)
    assert r.state == "STOPPED" and r.resume_state == "WORKER_RUNNING" and r.interrupted
