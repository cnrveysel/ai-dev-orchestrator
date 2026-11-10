"""Characterization of the V4 worker launch (written BEFORE the V5 worker-profile refactor).

These tests pin exactly how the legacy NInfer/Qwen worker is started today - executable, argv for a new and
for a resumed session, the process-local environment (CODEX_HOME, RUST_MIN_STACK), cwd, stdin prompt and
timeout - so that the V5 profile/launcher layers can be proven to change nothing for the legacy worker.
No real process is started: ProcessRunner.run is replaced by a recorder.
"""
import asyncio
import os

import pytest

from app.config.settings import Settings, WorkerSettings
from app.workers import process_runner
from app.workers.base import PhaseTurnRequest
from app.workers.codex_ninfer import CodexNInferWorker
from app.workers.process_runner import ProcessResult

EXEC_HELP = "Usage: codex exec --json --cd --output-last-message --sandbox --model --skip-git-repo-check --ephemeral"
RESUME_HELP = "Usage: codex exec resume --json --output-last-message --model"


@pytest.fixture
def legacy_install(tmp_path):
    """A V4-shaped installation: bin/ninfer/codex-ninfer.exe and a dedicated ~/.codex-ninfer."""
    exe = tmp_path / "bin" / "ninfer" / "codex-ninfer.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    home = tmp_path / ".codex-ninfer"
    home.mkdir()
    (home / "config.toml").write_text('model = "qwen3.8-27b"\nmodel_provider = "ninfer_lan"\n', encoding="utf-8")
    return str(exe), str(home)


class Recorder:
    """Stands in for ProcessRunner.run: answers capability probes, records every worker launch."""

    def __init__(self):
        self.launches = []

    def install(self, monkeypatch):
        rec = self

        async def fake_run(self, argv, *, cwd=None, env_overrides=None, stdin_data=None, timeout=None,
                           on_stdout=None, on_stderr=None, **kw):
            tail = argv[1:]
            if tail == ["--help"]:
                return ProcessResult(argv, 0, "Commands: exec", "", 0.0)
            if tail == ["--version"]:
                return ProcessResult(argv, 0, "codex-cli 0.0-test", "", 0.0)
            if tail == ["exec", "--help"]:
                return ProcessResult(argv, 0, EXEC_HELP, "", 0.0)
            if tail == ["exec", "resume", "--help"]:
                return ProcessResult(argv, 0, RESUME_HELP, "", 0.0)
            rec.launches.append({"argv": list(argv), "cwd": cwd, "env": dict(env_overrides or {}),
                                 "stdin": stdin_data, "timeout": timeout})
            if on_stdout:
                on_stdout('{"type": "thread.started", "thread_id": "thread-123"}')
                on_stdout('{"type": "item.completed", "item": {"id": "a", "type": "agent_message", "text": "ok"}}')
            return ProcessResult(argv, 0, "", "", 0.1)

        monkeypatch.setattr(process_runner.ProcessRunner, "run", fake_run)
        return self


def legacy_settings(exe: str, home: str) -> WorkerSettings:
    return WorkerSettings(executable=exe, codex_home=home)


def _last_msg_index(argv):
    return argv.index("--output-last-message") + 1


def test_legacy_defaults_are_the_v4_ninfer_profile():
    w = WorkerSettings()
    assert w.model == "qwen3.8-27b" and w.pass_model_flag is True
    assert w.extra_env == {"RUST_MIN_STACK": "33554432"}
    assert w.codex_home.endswith(".codex-ninfer")
    assert w.sandbox == "workspace-write" and w.timeout_seconds == 3600 and w.extra_args == []


def test_legacy_new_session_launch_is_pinned(legacy_install, tmp_path, monkeypatch):
    exe, home = legacy_install
    rec = Recorder().install(monkeypatch)
    w = CodexNInferWorker(legacy_settings(exe, home))
    sessions = []
    w.on_session = sessions.append
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    out = asyncio.run(w.run_turn(PhaseTurnRequest("DO THE PHASE", repo, "", "P1", "start")))
    assert len(rec.launches) == 1
    launch = rec.launches[0]
    argv = launch["argv"]
    last = argv[_last_msg_index(argv)]
    assert argv == [exe, "exec", "--json", "--cd", repo, "--output-last-message", last, "--sandbox",
                    "workspace-write", "--model", "qwen3.8-27b", "-"]
    assert launch["env"] == {"RUST_MIN_STACK": "33554432", "CODEX_HOME": home}
    assert launch["cwd"] == repo and launch["stdin"] == "DO THE PHASE" and launch["timeout"] == 3600
    assert out.session_id == "thread-123" and sessions == ["thread-123"], "session id reported at once"
    assert out.env_overrides["CODEX_HOME"] == home and out.executable == exe


def test_legacy_resumed_session_launch_is_pinned(legacy_install, tmp_path, monkeypatch):
    exe, home = legacy_install
    rec = Recorder().install(monkeypatch)
    w = CodexNInferWorker(legacy_settings(exe, home))
    repo = str(tmp_path / "repo")
    os.makedirs(repo)
    out = asyncio.run(w.run_turn(PhaseTurnRequest("CONTINUE", repo, "sess-42", "P1", "continue")))
    argv = rec.launches[0]["argv"]
    last = argv[_last_msg_index(argv)]
    assert argv == [exe, "exec", "resume", "--json", "--output-last-message", last, "-c",
                    'sandbox_mode="workspace-write"', "--model", "qwen3.8-27b", "sess-42", "-"]
    assert rec.launches[0]["env"] == {"RUST_MIN_STACK": "33554432", "CODEX_HOME": home}
    assert rec.launches[0]["cwd"] == repo
    assert out.resumed and out.session_id == "thread-123"


def test_legacy_launch_never_touches_parent_environment(legacy_install, tmp_path, monkeypatch):
    exe, home = legacy_install
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.delenv("RUST_MIN_STACK", raising=False)
    Recorder().install(monkeypatch)
    w = CodexNInferWorker(legacy_settings(exe, home))
    asyncio.run(w.run_turn(PhaseTurnRequest("x", str(tmp_path), "", "P1", "start")))
    assert "CODEX_HOME" not in os.environ and "RUST_MIN_STACK" not in os.environ


def test_engine_builds_the_legacy_worker_from_settings(legacy_install, tmp_path, monkeypatch):
    """Engine._make_worker('ninfer') with a verified legacy configuration yields the pinned launch."""
    from app.orchestration.orchestrator import Engine
    exe, home = legacy_install
    s = Settings()
    s.integrations.metrics_enabled = False
    s.worker.executable, s.worker.codex_home = exe, home
    s.worker.verified_executable, s.worker.verified_codex_home = exe, home
    e = Engine(s, lambda k, p: None, str(tmp_path / "c.db"))
    try:
        from app.models.run import Run
        e.run = Run(id=None, goal="g", repo_path=str(tmp_path), worker_mode="ninfer")
        e.db.create_run(e.run)
        w = e._make_worker("ninfer")
        assert isinstance(w, CodexNInferWorker)
        rec = Recorder().install(monkeypatch)
        asyncio.run(w.run_turn(PhaseTurnRequest("p", str(tmp_path), "", "P1", "start")))
        argv = rec.launches[0]["argv"]
        assert argv[:2] == [exe, "exec"] and argv[-3:] == ["--model", "qwen3.8-27b", "-"]
        assert rec.launches[0]["env"] == {"RUST_MIN_STACK": "33554432", "CODEX_HOME": home}
    finally:
        e.db.close()


def test_engine_refuses_unverified_legacy_worker(legacy_install, tmp_path):
    from app.orchestration.orchestrator import Engine, EngineError
    exe, home = legacy_install
    s = Settings()
    s.integrations.metrics_enabled = False
    s.worker.executable, s.worker.codex_home = exe, home
    e = Engine(s, lambda k, p: None, str(tmp_path / "c.db"))
    try:
        from app.models.run import Run
        e.run = Run(id=None, goal="g", repo_path=str(tmp_path), worker_mode="ninfer")
        e.db.create_run(e.run)
        with pytest.raises(EngineError, match="not verified"):
            e._make_worker("ninfer")
    finally:
        e.db.close()


# ------------------------------------------------------------ V5 milestone 1: the profile path is identical

def _launch(worker, repo, session=""):
    asyncio.run(worker.run_turn(PhaseTurnRequest("PROMPT", repo, session, "P1", "start")))


def _normalized(launch):
    argv = list(launch["argv"])
    argv[_last_msg_index(argv)] = "<last-message-file>"  # a fresh temp file per launch
    return {**launch, "argv": argv}


@pytest.mark.parametrize("session", ["", "sess-9"])
def test_profile_launch_spec_is_identical_to_the_v4_launch(legacy_install, tmp_path, monkeypatch, session):
    from app.workers.profiles import WorkerLaunchSpec, WorkerSnapshot, legacy_profile
    exe, home = legacy_install
    ws = legacy_settings(exe, home)
    ws.extra_args = ["-c", "model_reasoning_effort=high"]
    rec = Recorder().install(monkeypatch)
    _launch(CodexNInferWorker(ws), str(tmp_path), session)
    profile = legacy_profile(ws)
    spec = WorkerLaunchSpec.from_snapshot(WorkerSnapshot.of(profile), ws, secret_values=profile.extra_env)
    _launch(CodexNInferWorker(spec), str(tmp_path), session)
    v4, v5 = (_normalized(x) for x in rec.launches)
    assert v4 == v5
    assert v5["env"] == {"RUST_MIN_STACK": "33554432", "CODEX_HOME": home}


def test_codex_normalized_event_hook_does_not_change_the_launch(legacy_install, tmp_path, monkeypatch):
    """V5.1: the optional on_event hook sees normalized events; argv/env/logs of the legacy launch are unchanged."""
    exe, home = legacy_install
    rec = Recorder().install(monkeypatch)
    plain_logs, hooked_logs, events = [], [], []
    w1 = CodexNInferWorker(legacy_settings(exe, home), lambda s, t: plain_logs.append((s, t)))
    _launch(w1, str(tmp_path))
    w2 = CodexNInferWorker(legacy_settings(exe, home), lambda s, t: hooked_logs.append((s, t)))
    w2.on_event = events.append
    _launch(w2, str(tmp_path))
    a, b = (_normalized(x) for x in rec.launches)
    assert a == b and len(plain_logs) == len(hooked_logs)
    assert [e.kind for e in events] == ["session_started", "assistant_text"]
    assert events[0].session_id == "thread-123"
