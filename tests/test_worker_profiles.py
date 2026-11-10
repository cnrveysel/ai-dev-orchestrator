"""V5 worker profiles: profile core, launcher, registry, validation, run snapshots and resume safety.

No real worker is launched: ProcessRunner.run is replaced by a recorder (and conftest refuses real CLIs anyway).
"""
import asyncio
import json
import os
import sqlite3
import time

import pytest

from app.config import codex_config
from app.config.discovery import ExecutableRejected, validate_codex_executable, validate_executable_for_policy
from app.config.settings import Settings, SettingsStore, WorkerSettings
from app.workers import process_runner
from app.workers.base import PhaseTurnRequest
from app.workers.codex_ninfer import CodexNInferWorker
from app.workers.launcher import (
    RUNTIME_ADAPTERS, WorkerProfileUnavailable, build_launch_spec, launch_worker, snapshot_problems,
)
from app.workers.process_runner import ProcessResult
from app.workers.profile_registry import ProfileError, WorkerProfileRegistry
from app.workers.profiles import (
    LEGACY_PROFILE_ID, REDACTED, RunWorkerProfiles, WorkerLaunchSpec, WorkerProfile, WorkerSnapshot,
    legacy_profile, redact_env,
)

EXEC_HELP = "Usage: codex exec --json --cd --output-last-message --sandbox --model --skip-git-repo-check --ephemeral"
RESUME_HELP = "Usage: codex exec resume --json --output-last-message --model"


class Recorder:
    def __init__(self):
        self.launches = []

    def install(self, monkeypatch):
        rec = self

        async def fake_run(self, argv, *, cwd=None, env_overrides=None, stdin_data=None, timeout=None,
                           on_stdout=None, on_stderr=None, **kw):
            tail = argv[1:]
            answers = {("--help",): "Commands: exec", ("--version",): "codex-cli test",
                       ("exec", "--help"): EXEC_HELP, ("exec", "resume", "--help"): RESUME_HELP}
            if tuple(tail) in answers:
                return ProcessResult(argv, 0, answers[tuple(tail)], "", 0.0)
            rec.launches.append({"argv": list(argv), "cwd": cwd, "env": dict(env_overrides or {}),
                                 "stdin": stdin_data, "timeout": timeout})
            if on_stdout:
                on_stdout('{"type": "thread.started", "thread_id": "thread-%d"}' % len(rec.launches))
            if on_stderr:
                on_stderr("worker started")
            return ProcessResult(argv, 0, "", "", 0.1)

        monkeypatch.setattr(process_runner.ProcessRunner, "run", fake_run)
        return self


def make_exe(tmp_path, name, sub="bin"):
    p = tmp_path / sub / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_bytes(b"MZ")
    return str(p)


def make_home(tmp_path, name, config='model = "m"\nmodel_provider = "p"\n'):
    h = tmp_path / name
    h.mkdir(parents=True, exist_ok=True)
    if config is not None:
        (h / "config.toml").write_text(config, encoding="utf-8")
    return str(h)


OPENROUTER_TOML = '''model = "z-ai/glm-5.3-flash"
model_provider = "openrouter"
[model_providers.openrouter]
name = "OpenRouter"
base_url = "https://openrouter.ai/api/v1"
wire_api = "responses"
[model_providers.openrouter.auth]
command = "powershell"
args = ["-NoProfile", "-Command", "Write-Output $env:OPENROUTER_API_KEY"]
'''
NINFER_TOML = '''model = "qwen3.8-27b"
model_provider = "ninfer_lan"
model_context_window = 262144
[model_providers.ninfer_lan]
name = "NInfer LAN"
base_url = "http://192.168.56.20:8080/v1"
wire_api = "responses"
'''


@pytest.fixture
def machine(tmp_path, monkeypatch):
    """A machine like the user's: one codex-ninfer.exe, a codex-lmstudio.exe and three profile homes."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    ninfer_exe = make_exe(tmp_path, "codex-ninfer.exe", "bin/ninfer")
    lm_exe = make_exe(tmp_path, "codex-lmstudio.exe")
    homes = {"ninfer": make_home(tmp_path, ".codex-ninfer", NINFER_TOML),
             "lmstudio": make_home(tmp_path, ".codex-lmstudio"),
             "openrouter": make_home(tmp_path, ".codex-openrouter", OPENROUTER_TOML)}
    s = Settings()
    s.integrations.metrics_enabled = False
    s.worker.executable, s.worker.codex_home = ninfer_exe, homes["ninfer"]
    s.worker.verified_executable, s.worker.verified_codex_home = ninfer_exe, homes["ninfer"]
    return {"settings": s, "ninfer_exe": ninfer_exe, "lm_exe": lm_exe, "homes": homes, "tmp": tmp_path}


def verified(reg, p):
    reg.record_verification(p.id, True, p.codex_executable, p.codex_home)
    return reg.require(p.id)


# ------------------------------------------------------------ profile core (milestone 1)

def test_legacy_profile_mirrors_the_v4_worker_settings():
    w = WorkerSettings(executable=r"C:\b\codex-ninfer.exe", codex_home=r"C:\u\.codex-ninfer")
    p = legacy_profile(w)
    assert p.id == LEGACY_PROFILE_ID and p.runtime == "codex" and p.executable_policy == "ninfer"
    assert p.codex_executable == w.executable and p.codex_home == w.codex_home
    assert p.extra_env == {"RUST_MIN_STACK": "33554432"} and p.model == "qwen3.8-27b" and p.pass_model_flag
    assert p.is_verified() == w.is_verified() is False


def test_profile_round_trip_and_revision():
    p = WorkerProfile(id="x", display_name="X", codex_executable="C:/a.exe", codex_home="C:/h",
                      extra_env={"A": "1"}, required_env=["OPENROUTER_API_KEY"], context_window=5)
    q = WorkerProfile.from_dict(json.loads(json.dumps(p.to_dict())))
    assert q == p and q.revision() == p.revision()
    q.notes = "display only"
    assert q.revision() == p.revision(), "metadata does not change the launch revision"
    q.codex_home = "C:/other"
    assert q.revision() != p.revision()


def test_snapshot_never_persists_secret_values():
    p = WorkerProfile(id="x", display_name="X", codex_executable="e", codex_home="h",
                      extra_env={"RUST_MIN_STACK": "1", "OPENAI_API_KEY": "sk-secret"})
    snap = WorkerSnapshot.of(p)
    dumped = json.dumps(snap.to_dict())
    assert "sk-secret" not in dumped and snap.secret_env_names == ("OPENAI_API_KEY",)
    assert snap.extra_env == {"RUST_MIN_STACK": "1"}
    again = WorkerSnapshot.from_dict(json.loads(dumped))
    assert again == snap
    spec = build_launch_spec(again, WorkerSettings(), p)  # the value is re-read from the live profile, in memory
    assert spec.extra_env["OPENAI_API_KEY"] == "sk-secret"
    assert "sk-secret" not in "\n".join(again.describe())


def test_run_worker_profiles_reserve_per_phase_overrides():
    a = WorkerSnapshot.of(WorkerProfile(id="a", display_name="A"))
    b = WorkerSnapshot.of(WorkerProfile(id="b", display_name="B"))
    rp = RunWorkerProfiles(default=a, phase_overrides={"P2": b})
    back = RunWorkerProfiles.from_dict(json.loads(json.dumps(rp.to_dict())))
    assert back.for_phase("P1").profile_id == "a" and back.for_phase("P2").profile_id == "b"
    assert back.for_phase(None).profile_id == "a"
    assert RunWorkerProfiles.from_dict({}).default is None and RunWorkerProfiles().to_dict() == {}


def test_settings_round_trip_keeps_profile_records(tmp_path):
    s = Settings()
    s.worker_profiles.profiles.append(WorkerProfile(id="or", display_name="OpenRouter").to_dict())
    s.worker_profiles.default_profile_id = "or"
    store = SettingsStore(tmp_path / "settings.json")
    store.save(s)
    back = store.load()
    assert back.worker_profiles.default_profile_id == "or"
    assert back.worker_profiles.profiles[0]["display_name"] == "OpenRouter"
    old = json.loads((tmp_path / "settings.json").read_text())
    del old["worker_profiles"]  # a V4 settings.json
    (tmp_path / "settings.json").write_text(json.dumps(old))
    v4 = store.load()
    assert v4.worker_profiles.default_profile_id == LEGACY_PROFILE_ID and v4.worker_profiles.profiles == []


# ------------------------------------------------------------ executable policy

def test_profile_executable_policy_keeps_the_launcher_protections(tmp_path):
    lm = make_exe(tmp_path, "codex-lmstudio.exe")
    assert validate_codex_executable(lm) == lm
    with pytest.raises(ExecutableRejected):
        validate_executable_for_policy(lm, "ninfer"), "the V4 rule still applies to the legacy worker"
    for bad in ["codex.exe", "codex-menu.exe", "codex.cmd", "codex-local.cmd"]:
        with pytest.raises(ExecutableRejected):
            validate_codex_executable(make_exe(tmp_path, bad, "b2"))
    with pytest.raises(ExecutableRejected):
        validate_codex_executable(make_exe(tmp_path, "codex-ninfer.exe", "npm/node_modules"))
    with pytest.raises(ExecutableRejected):
        validate_codex_executable("")


# ------------------------------------------------------------ launcher (milestone 2)

def test_launcher_dispatches_by_runtime_only(machine):
    reg = WorkerProfileRegistry(machine["settings"])
    p = reg.register_existing(machine["homes"]["lmstudio"], machine["lm_exe"], "LM Studio Qwen")
    spec = build_launch_spec(WorkerSnapshot.of(p), machine["settings"].worker, p)
    w = launch_worker(spec)
    assert isinstance(w, CodexNInferWorker) and w.settings is spec
    bad = WorkerLaunchSpec(**{**spec.__dict__, "runtime": "other"})
    with pytest.raises(WorkerProfileUnavailable, match="WORKER_PROFILE_UNAVAILABLE"):
        launch_worker(bad)
    assert set(RUNTIME_ADAPTERS) == {"codex", "cline"}  # V5.1


def test_two_profiles_launch_with_their_own_codex_home_and_process_local_env(machine, monkeypatch):
    monkeypatch.delenv("CODEX_HOME", raising=False)
    monkeypatch.setenv("OPENROUTER_API_KEY", "or-secret-value")
    reg = WorkerProfileRegistry(machine["settings"])
    lm = reg.register_existing(machine["homes"]["lmstudio"], machine["lm_exe"], "LM Studio Qwen",
                               extra_env={"LMS_TRACE": "1"})
    orp = reg.register_existing(machine["homes"]["openrouter"], machine["ninfer_exe"], "OpenRouter GLM")
    assert orp.required_env == ["OPENROUTER_API_KEY"], "env var NAME discovered from config.toml's auth command"
    rec = Recorder().install(monkeypatch)
    for p in (lm, orp):
        spec = build_launch_spec(WorkerSnapshot.of(p), machine["settings"].worker, p)
        asyncio.run(launch_worker(spec).run_turn(PhaseTurnRequest("x", str(machine["tmp"]), "", "P1", "start")))
    a, b = rec.launches
    assert a["argv"][0] == machine["lm_exe"] and b["argv"][0] == machine["ninfer_exe"]
    assert a["env"] == {"LMS_TRACE": "1", "CODEX_HOME": machine["homes"]["lmstudio"]}
    assert b["env"] == {"CODEX_HOME": machine["homes"]["openrouter"]}, "the key is inherited, never injected"
    assert "--model" not in a["argv"] and "--model" not in b["argv"], "config.toml decides the model"
    assert "CODEX_HOME" not in os.environ and "LMS_TRACE" not in os.environ


def test_secret_values_are_redacted_from_logs_and_outcomes(machine, monkeypatch):
    s = machine["settings"]
    s.worker.extra_env = {"RUST_MIN_STACK": "33554432", "NINFER_API_TOKEN": "tok-12345"}  # legacy allows it
    p = legacy_profile(s.worker)
    rec = Recorder().install(monkeypatch)
    logs = []
    spec = build_launch_spec(WorkerSnapshot.of(p), s.worker, p)
    w = launch_worker(spec, lambda src, text: logs.append(text))
    out = asyncio.run(w.run_turn(PhaseTurnRequest("x", str(machine["tmp"]), "", "P1", "start")))
    assert rec.launches[0]["env"]["NINFER_API_TOKEN"] == "tok-12345", "the child still receives it"
    assert all("tok-12345" not in line for line in logs)
    assert out.env_overrides["NINFER_API_TOKEN"] == REDACTED and out.worker_profile_id == LEGACY_PROFILE_ID
    assert any(line.startswith("Worker profile: ") for line in logs)
    assert redact_env({"A_PASSWORD": "x", "PATH_X": "y"}) == {"A_PASSWORD": REDACTED, "PATH_X": "y"}


# ------------------------------------------------------------ run snapshots and resume (milestones 3 + 4)

def _engine(settings, db_path):
    from app.orchestration.orchestrator import Engine
    return Engine(settings, lambda k, p: None, str(db_path))


def _new_run(e, repo, **kw):
    from app.models.run import Run
    run = Run(id=None, goal="g", repo_path=str(repo), worker_mode="ninfer", **kw)
    e.db.create_run(run)
    e.run = run
    return run


def _with_openrouter(machine, default=False):
    s = machine["settings"]
    reg = WorkerProfileRegistry(s)
    p = verified(reg, reg.register_existing(machine["homes"]["openrouter"], machine["ninfer_exe"], "OpenRouter GLM",
                                            profile_id="openrouter-glm"))
    if default:
        reg.set_default(p.id)
    return p


def test_changing_the_default_profile_does_not_alter_an_existing_run(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = machine["settings"]
    e = _engine(s, machine["tmp"] / "a.db")
    try:
        run = _new_run(e, machine["tmp"])
        w1 = e._make_worker("ninfer")
        assert w1.settings.profile_id == LEGACY_PROFILE_ID
        _with_openrouter(machine, default=True)
        asyncio.run(e.apply_settings(s))  # the user switches the default to OpenRouter
        w2 = e._make_worker("ninfer")
        assert w2.settings.profile_id == LEGACY_PROFILE_ID and w2.settings.codex_home == machine["homes"]["ninfer"]
        stored = e.db.load_run(run.id).worker_snapshot
        assert stored["default"]["profile_id"] == LEGACY_PROFILE_ID
        other = _new_run(e, machine["tmp"])  # a NEW run takes the new default
        w3 = e._make_worker("ninfer")
        assert w3.settings.profile_id == "openrouter-glm" and w3.settings.codex_home == machine["homes"]["openrouter"]
        assert e.db.load_run(other.id).worker_snapshot["default"]["profile_id"] == "openrouter-glm"
    finally:
        e.db.close()


def test_resume_after_restart_restores_the_original_snapshot(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = machine["settings"]
    _with_openrouter(machine, default=True)
    e = _engine(s, machine["tmp"] / "r.db")
    run = _new_run(e, machine["tmp"])
    assert e._make_worker("ninfer").settings.profile_id == "openrouter-glm"
    run_id = run.id
    e.db.close()
    # "restart": new settings (default back to the legacy profile, OpenRouter edited) and a new engine
    s2 = Settings.from_dict(s.to_dict())
    reg = WorkerProfileRegistry(s2)
    reg.set_default(LEGACY_PROFILE_ID)
    reg.update("openrouter-glm", extra_env={"TRACE": "1"})
    e2 = _engine(s2, machine["tmp"] / "r.db")
    try:
        e2.run = e2.db.load_run(run_id)
        logs = []
        e2.log = lambda src, text: logs.append(text)
        spec = e2._make_worker("ninfer").settings
        assert spec.profile_id == "openrouter-glm" and spec.executable == machine["ninfer_exe"]
        assert spec.codex_home == machine["homes"]["openrouter"] and "TRACE" not in spec.extra_env, \
            "the edit applies to new runs only"
        assert any("was edited after this run started" in t for t in logs)
        assert any(t.startswith("Worker profile: OpenRouter GLM") for t in logs)
    finally:
        e2.db.close()


def test_a_v4_run_without_snapshot_keeps_the_legacy_worker_not_the_default(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    _with_openrouter(machine, default=True)
    e = _engine(machine["settings"], machine["tmp"] / "v4.db")
    try:
        run = _new_run(e, machine["tmp"], started_at=time.time() - 3600)  # executed under V4
        spec = e._make_worker("ninfer").settings
        assert spec.profile_id == LEGACY_PROFILE_ID and spec.codex_home == machine["homes"]["ninfer"]
        snap = e.db.load_run(run.id).worker_snapshot["default"]
        assert snap["source"] == "legacy-settings" and snap["model"] == "qwen3.8-27b"
    finally:
        e.db.close()


def test_snapshot_is_write_once(tmp_path):
    from app.models.run import Run
    from app.persistence.database import Database
    db = Database(tmp_path / "w.db")
    run = Run(id=None, goal="g", repo_path="C:/r")
    db.create_run(run)
    assert db.lock_worker_snapshot(run.id, {"default": {"profile_id": "a"}})
    assert not db.lock_worker_snapshot(run.id, {"default": {"profile_id": "b"}})
    run.status_message = "x"
    db.update_run(run)  # ordinary run updates never touch the snapshot
    assert db.load_run(run.id).worker_snapshot == {"default": {"profile_id": "a"}}
    db.close()


@pytest.mark.parametrize("breakage", ["disabled", "deleted", "exe_missing", "home_missing", "env_missing",
                                      "unverified"])
def test_unavailable_profile_never_falls_back(machine, monkeypatch, breakage):
    from app.orchestration.orchestrator import EngineError
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = machine["settings"]
    p = _with_openrouter(machine, default=True)
    e = _engine(s, machine["tmp"] / "u.db")
    try:
        run = _new_run(e, machine["tmp"])
        e._make_worker("ninfer")  # locked to OpenRouter
        reg = WorkerProfileRegistry(s)
        reg.set_default(LEGACY_PROFILE_ID)  # a perfectly usable default exists
        if breakage == "disabled":
            reg.set_enabled(p.id, False)
        elif breakage == "deleted":
            s.worker_profiles.profiles = []  # e.g. settings.json edited by hand
        elif breakage == "exe_missing":
            os.remove(machine["ninfer_exe"])
        elif breakage == "home_missing":
            os.rename(machine["homes"]["openrouter"], machine["homes"]["openrouter"] + "-gone")
        elif breakage == "env_missing":
            monkeypatch.delenv("OPENROUTER_API_KEY")
        elif breakage == "unverified":
            reg.record_verification(p.id, False)  # a failing smoke test revokes it
        asyncio.run(e.apply_settings(s))  # what the GUI does after Settings > Save
        with pytest.raises(EngineError, match="WORKER_PROFILE_UNAVAILABLE") as ei:
            e._make_worker("ninfer")
        assert "No fallback" in str(ei.value) and "openrouter-glm" in str(ei.value)
        stored = e.db.load_run(run.id)
        assert stored.worker_snapshot["default"]["profile_id"] == "openrouter-glm", "never rebound"
        assert stored.status_message.startswith("WORKER_PROFILE_UNAVAILABLE")
    finally:
        e.db.close()


def test_new_run_is_not_locked_when_its_profile_is_not_ready(machine):
    from app.orchestration.orchestrator import EngineError
    s = machine["settings"]
    s.worker.verified_executable = ""  # the legacy worker has not passed its smoke test
    e = _engine(s, machine["tmp"] / "n.db")
    try:
        run = _new_run(e, machine["tmp"])
        with pytest.raises(EngineError, match="not verified"):
            e._make_worker("ninfer")
        assert e.db.load_run(run.id).worker_snapshot == {}, "nothing locked: fixing the profile and START again works"
    finally:
        e.db.close()


def test_profile_used_by_a_resumable_run_cannot_be_deleted(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = machine["settings"]
    p = _with_openrouter(machine, default=True)
    e = _engine(s, machine["tmp"] / "d.db")
    try:
        run = _new_run(e, machine["tmp"])
        e._make_worker("ninfer")
        reg = WorkerProfileRegistry(s)
        reg.set_default(LEGACY_PROFILE_ID)
        in_use = e.worker_profiles_in_use()
        assert in_use == {p.id: [run.id]}
        with pytest.raises(ProfileError, match="can still be resumed"):
            reg.delete(p.id, in_use)
        run.state = "COMPLETED"
        e.db.update_run(run)
        assert e.worker_profiles_in_use() == {}
        reg.delete(p.id, e.worker_profiles_in_use())
        assert reg.get(p.id) is None and os.path.isdir(machine["homes"]["openrouter"]), "the home folder stays"
    finally:
        e.db.close()


@pytest.fixture
def repo_git(tmp_path):
    import subprocess
    r = tmp_path / "proj"
    r.mkdir()
    (r / "README.md").write_text("x\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=r, check=True, capture_output=True)
    return r


def test_snapshot_is_persisted_before_the_first_worker_process(machine, repo_git, monkeypatch):
    """Crash safety: START locks the snapshot in the database before any worker process exists, and every worker
    call records the profile it ran with. Full V1 pipeline with the mock orchestrator and a recorded Codex."""
    s = machine["settings"]
    s.orchestration.worker_mode = "ninfer"
    s.orchestration.mock_step_delay = 0.0
    s.orchestration.mock_retry_task_ids = []
    s.integrations.codebase_memory_executable = ""
    db_path = machine["tmp"] / "c.db"
    seen = []
    original = process_runner.ProcessRunner.run

    async def fake_run(self, argv, *args, **kw):
        if argv and os.path.basename(argv[0]) == "codex-ninfer.exe":
            tail = tuple(argv[1:])
            if tail in (("--help",), ("--version",), ("exec", "--help"), ("exec", "resume", "--help")):
                return ProcessResult(argv, 0, "exec " + EXEC_HELP + " " + RESUME_HELP, "", 0.0)
            with sqlite3.connect(db_path) as c:
                seen.append(json.loads(c.execute("SELECT worker_snapshot_json FROM runs").fetchone()[0]))
            return ProcessResult(argv, 0, "", "", 0.01)
        return await original(self, argv, *args, **kw)

    monkeypatch.setattr(process_runner.ProcessRunner, "run", fake_run)
    e = _engine(s, db_path)

    async def settle():
        while e._busy is not None and not e._busy.done():
            await asyncio.sleep(0.01)

    async def main():
        await e.plan("g", str(repo_git))
        await settle()
        await e.start()
        await settle()

    try:
        asyncio.run(main())
        assert e.sm.state.value == "COMPLETED"
        assert seen and all(x["default"]["profile_id"] == LEGACY_PROFILE_ID for x in seen)
        rows = e.db.conn.execute("SELECT worker_profile_id, worker_profile_revision FROM worker_calls").fetchall()
        assert rows and all(r[0] == LEGACY_PROFILE_ID and r[1] for r in rows)
    finally:
        e.db.close()


def test_v4_database_opens_after_the_additive_migration(tmp_path):
    from app.persistence.database import Database
    from app.persistence.migrations import MIGRATIONS
    p = tmp_path / "v4.db"
    conn = sqlite3.connect(p)
    conn.execute("CREATE TABLE schema_version (version INTEGER NOT NULL)")
    conn.execute("INSERT INTO schema_version (version) VALUES (2)")
    conn.executescript(MIGRATIONS[0])
    conn.executescript(MIGRATIONS[1])  # exactly the V4 schema
    now = time.time()
    conn.execute("INSERT INTO runs (goal, repo_path, state, resume_state, created_at, updated_at, started_at, "
                 "orchestrator_mode, worker_mode, settings_snapshot_json, pipeline_version, pipeline_state_json) "
                 "VALUES ('old', 'C:/r', 'STOPPED', 'WORKER_RUNNING', ?, ?, ?, 'claude', 'ninfer', ?, 'v2', ?)",
                 (now, now, now, json.dumps({"worker_codex_home": "C:/u/.codex-ninfer"}), json.dumps({"replans": 1})))
    conn.execute("INSERT INTO worker_calls (run_id, attempt_id, executable, argv_json, env_overrides_json, "
                 "started_at, duration, exit_code, timed_out, cancelled, tool_calls, usage_json) "
                 "VALUES (1, 1, 'C:/b/codex-ninfer.exe', '[]', '{}', ?, 1.0, 0, 0, 0, 3, '{}')", (now,))
    conn.commit()
    conn.close()
    db = Database(p)
    try:
        assert db.conn.execute("SELECT version FROM schema_version").fetchone()[0] == len(MIGRATIONS) == 3
        run = db.load_run(1)
        assert run.goal == "old" and run.pipeline_version == "v2" and run.pipeline_state == {"replans": 1}
        assert run.worker_snapshot == {} and run.settings_snapshot["worker_codex_home"] == "C:/u/.codex-ninfer"
        row = db.conn.execute("SELECT tool_calls, worker_profile_id FROM worker_calls").fetchone()
        assert row[0] == 3 and row[1] is None, "old worker calls stay readable (profile unknown = pre-V5)"
        assert db.worker_profiles_in_use() == {}
    finally:
        db.close()
    db = Database(p)  # idempotent
    assert db.conn.execute("SELECT version FROM schema_version").fetchone()[0] == 3
    db.close()


def test_worker_call_rows_never_store_secret_env(tmp_path):
    from app.persistence.database import Database
    from app.workers.base import WorkerOutcome
    db = Database(tmp_path / "s.db")
    o = WorkerOutcome(executable="e", argv=[], exit_code=0, summary="", stdout_tail="", stderr_tail="", duration=0,
                      env_overrides={"CODEX_HOME": "h", "MY_API_KEY": "sk-1"}, worker_profile_id="p")
    db.add_worker_call(1, None, o, time.time())
    row = db.conn.execute("SELECT env_overrides_json, worker_profile_id FROM worker_calls").fetchone()
    assert "sk-1" not in row[0] and row[1] == "p"
    db.close()


# ------------------------------------------------------------ registry (milestone 5) and validation (milestone 7)

def _fingerprint(home):
    """Bytes + mtime of every file directly in a profile home."""
    out = {}
    for name in sorted(os.listdir(home)):
        p = os.path.join(home, name)
        if os.path.isfile(p):
            with open(p, "rb") as f:
                out[name] = (f.read(), os.stat(p).st_mtime_ns)
    return out


def test_external_profiles_are_never_mutated(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    home = machine["homes"]["openrouter"]
    before = _fingerprint(home)
    reg = WorkerProfileRegistry(machine["settings"])
    p = reg.register_existing(home, machine["ninfer_exe"], "OpenRouter GLM")
    assert p.mode == "external" and p.provider == "openrouter" and p.model == "z-ai/glm-5.3-flash"
    assert p.base_url == "https://openrouter.ai/api/v1" and not p.pass_model_flag
    reg.validate(p.id)
    reg.update(p.id, notes="edited", extra_env={"TRACE": "1"}, model="display-only")
    reg.rename(p.id, "OpenRouter (renamed)")
    reg.set_enabled(p.id, False)
    reg.set_enabled(p.id, True)
    dup = reg.duplicate(p.id, "OR copy")  # references the same home
    assert dup.codex_home == home and dup.mode == "external"
    reg.delete(dup.id)
    reg.delete(p.id)
    assert _fingerprint(home) == before, "config.toml untouched (bytes and mtime)"


def test_concurrent_profiles_never_rewrite_each_others_config(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    homes = machine["homes"]
    before = {k: _fingerprint(h) for k, h in homes.items()}
    reg = WorkerProfileRegistry(machine["settings"])
    lm = reg.register_existing(homes["lmstudio"], machine["lm_exe"], "LM Studio")
    orp = reg.register_existing(homes["openrouter"], machine["ninfer_exe"], "OpenRouter")
    Recorder().install(monkeypatch)
    specs = [build_launch_spec(WorkerSnapshot.of(p), machine["settings"].worker, p)
             for p in (reg.legacy(), lm, orp)]

    async def all_at_once():
        await asyncio.gather(*(launch_worker(s).run_turn(PhaseTurnRequest("x", str(machine["tmp"]), "", f"P{i}",
                                                                          "start")) for i, s in enumerate(specs)))
    asyncio.run(all_at_once())
    assert {k: _fingerprint(h) for k, h in homes.items()} == before


def test_managed_profile_gets_its_own_new_home(machine):
    s = machine["settings"]
    s.worker_profiles.managed_root = str(machine["tmp"] / "profiles")
    reg = WorkerProfileRegistry(s)
    template = machine["homes"]["ninfer"]
    before = _fingerprint(template)
    swift = reg.create_managed("NInfer Swift Qwen", machine["ninfer_exe"], template_home=template,
                               extra_env={"RUST_MIN_STACK": "33554432"})
    assert swift.mode == "managed" and swift.codex_home == str(machine["tmp"] / "profiles" / "ninfer-swift-qwen")
    assert open(os.path.join(swift.codex_home, "config.toml"), encoding="utf-8").read() == NINFER_TOML
    assert swift.model == "qwen3.8-27b" and swift.provider == "ninfer_lan" and swift.context_window == 262144
    assert _fingerprint(template) == before, "the template is only read"
    assert not swift.is_verified()
    with pytest.raises(codex_config.CodexConfigError, match="not empty"):
        codex_config.create_managed_home(swift.codex_home, template_home=template)
    copy = reg.duplicate(swift.id, "Swift copy", copy_home=True)
    assert copy.codex_home != swift.codex_home and os.path.isfile(os.path.join(copy.codex_home, "config.toml"))
    fresh = reg.create_managed("LM Studio 2", machine["lm_exe"], config_text=codex_config.minimal_config(
        "Qwen3.8 27B", "lmstudio_lan", "LM Studio LAN", "http://192.168.56.20:1234/v1", context_window=190000))
    assert fresh.provider == "lmstudio_lan" and fresh.base_url == "http://192.168.56.20:1234/v1"
    keyed = codex_config.minimal_config("m", "openrouter", "OpenRouter", "https://openrouter.ai/api/v1",
                                        api_key_env="OPENROUTER_API_KEY")
    assert "$env:OPENROUTER_API_KEY" in keyed and "sk-" not in keyed


def test_create_managed_home_refuses_to_overwrite(tmp_path):
    home = tmp_path / "h"
    home.mkdir()
    (home / "config.toml").write_text("model = 'x'\n")
    with pytest.raises(codex_config.CodexConfigError, match="not empty"):
        codex_config.create_managed_home(str(home), config_text="model = 'y'\n")
    assert (home / "config.toml").read_text() == "model = 'x'\n"
    with pytest.raises(Exception):
        codex_config.create_managed_home(str(tmp_path / "bad"), config_text="this is = = not toml")
    assert not (tmp_path / "bad" / "config.toml").exists()


def test_registry_rules(machine):
    s = machine["settings"]
    reg = WorkerProfileRegistry(s)
    assert [p.id for p in reg.profiles()] == [LEGACY_PROFILE_ID] and reg.default_id() == LEGACY_PROFILE_ID
    for op in (lambda: reg.delete(LEGACY_PROFILE_ID), lambda: reg.rename(LEGACY_PROFILE_ID, "x"),
               lambda: reg.set_enabled(LEGACY_PROFILE_ID, False)):
        with pytest.raises(ProfileError):
            op()
    p = reg.register_existing(machine["homes"]["lmstudio"], machine["lm_exe"], "LM Studio Qwen")
    assert p.id == "lm-studio-qwen"
    with pytest.raises(ProfileError, match="already exists"):
        reg.register_existing(machine["homes"]["lmstudio"], machine["lm_exe"], "x", profile_id="lm-studio-qwen")
    with pytest.raises(ProfileError, match="looks like a secret"):
        reg.update(p.id, extra_env={"OPENROUTER_API_KEY": "sk-plain"})
    assert "sk-plain" not in json.dumps(s.to_dict())
    with pytest.raises(ProfileError, match="unsafe"):
        reg.update(p.id, extra_args=["--dangerously-bypass-approvals-and-sandbox"])
    with pytest.raises(ProfileError, match="normal ~/.codex"):
        os.makedirs(machine["tmp"] / ".codex", exist_ok=True)
        reg.register_existing(str(machine["tmp"] / ".codex"), machine["lm_exe"], "normal")
    reg.set_default(p.id)
    with pytest.raises(ProfileError, match="default"):
        reg.delete(p.id)
    with pytest.raises(ProfileError, match="default"):
        reg.set_enabled(p.id, False)
    reg.set_default(LEGACY_PROFILE_ID)
    reg.set_enabled(p.id, False)
    with pytest.raises(ProfileError, match="disabled"):
        reg.set_default(p.id)
    verified(reg, reg.set_enabled(p.id, True))
    assert reg.require(p.id).is_verified()
    reg.update(p.id, codex_home=machine["homes"]["openrouter"])
    assert not reg.require(p.id).is_verified(), "a verification never survives a changed launch"
    legacy_dup = reg.duplicate(LEGACY_PROFILE_ID, "NInfer copy")
    assert legacy_dup.executable_policy == "ninfer" and legacy_dup.pass_model_flag and legacy_dup.mode == "external"
    assert legacy_dup.is_verified(), "same executable + home as the verified legacy worker"


def test_quick_validation(machine, monkeypatch):
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    reg = WorkerProfileRegistry(machine["settings"])
    orp = reg.register_existing(machine["homes"]["openrouter"], machine["ninfer_exe"], "OpenRouter")
    rep = reg.validate(orp.id)
    assert not rep.ok and any("OPENROUTER_API_KEY" in e for e in rep.errors)
    assert rep.info["provider"] == "openrouter" and rep.info["env_refs"] == ["OPENROUTER_API_KEY"]
    assert reg.validate(orp.id, environ={"OPENROUTER_API_KEY": "set"}).ok
    with open(os.path.join(machine["homes"]["openrouter"], "config.toml"), "a", encoding="utf-8") as f:
        f.write("\nthis is not toml = = =\n")
    assert any("could not be parsed" in e for e in reg.validate(orp.id, environ={"OPENROUTER_API_KEY": "1"}).errors)
    assert reg.validate(LEGACY_PROFILE_ID).ok
    assert not reg.validate("nope").ok
    shared = reg.duplicate(LEGACY_PROFILE_ID, "same home")
    assert any("shares its CODEX_HOME" in w for w in reg.validate(shared.id).warnings)
    os.remove(machine["lm_exe"])
    lm = reg.register_existing(machine["homes"]["lmstudio"], machine["lm_exe"], "LM")
    assert any("not found" in e for e in reg.validate(lm.id).errors)


def test_discover_codex_homes(machine):
    found = codex_config.discover_codex_homes(str(machine["tmp"]))
    assert found == sorted(machine["homes"].values())
    assert codex_config.inspect_home(machine["homes"]["ninfer"]).base_url == "http://192.168.56.20:8080/v1"


def test_full_test_runs_the_chosen_profile_and_records_its_verification(machine, monkeypatch):
    monkeypatch.setenv("OPENROUTER_API_KEY", "x")
    s = machine["settings"]
    reg = WorkerProfileRegistry(s)
    orp = reg.register_existing(machine["homes"]["openrouter"], machine["ninfer_exe"], "OpenRouter")
    calls = []

    async def smoke(self, metrics_url=""):
        from app.orchestration.providers.base import SmokeResult
        calls.append((self.settings.profile_id, self.settings.codex_home, metrics_url))
        return SmokeResult(True, self.settings.executable, "ok", extra={"codex_home": self.settings.codex_home})

    monkeypatch.setattr(CodexNInferWorker, "smoke_test", smoke)
    s.integrations.metrics_enabled = True
    e = _engine(s, machine["tmp"] / "t.db")
    try:
        out = asyncio.run(e.smoke_test_worker(orp.id))
        assert out["ok"] and out["profile_id"] == orp.id and out["verified_codex_home"] == orp.codex_home
        assert calls == [(orp.id, orp.codex_home, "")], "no NInfer metrics claim for a non-NInfer profile"
        assert WorkerProfileRegistry(e.settings).require(orp.id).is_verified()
        out = asyncio.run(e.smoke_test_worker())  # default = legacy: exactly the V4 smoke test target
        assert out["profile_id"] == LEGACY_PROFILE_ID and calls[-1][2] == s.integrations.metrics_url
        monkeypatch.delenv("OPENROUTER_API_KEY")
        out = asyncio.run(e.smoke_test_worker(orp.id))
        assert not out["ok"] and "OPENROUTER_API_KEY" in out["detail"] and len(calls) == 2, "nothing launched"
        assert not WorkerProfileRegistry(e.settings).require(orp.id).is_verified(), "a failure revokes it"
    finally:
        e.db.close()
