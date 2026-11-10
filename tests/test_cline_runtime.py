"""V5.1 Cline worker runtime (tests 3-20 of the V5.1 list; 1-2 live in test_worker_launch_characterization.py).

A fake Cline CLI (a Python script) replays the NDJSON observed from the installed Cline CLI 3.0.65 (`--json`), and
answers `history --json` with the pid of the process that ran the session. The conftest guard refuses any real
cline.exe / codex / claude process.
"""
import asyncio
import ctypes
import json
import os
import subprocess
import sys
import time

import pytest

from app.config.discovery import ExecutableRejected, validate_cline_executable
from app.config.settings import Settings, SettingsStore, WorkerSettings
from app.workers import cline_cli
from app.workers.base import PhaseTurnRequest, WorkerCapabilities
from app.workers.cline_cli import HEADER, ClineWorker
from app.workers.codex_ninfer import CodexNInferWorker
from app.workers.events import normalize_cline_event, normalize_codex_event, redact
from app.workers.launcher import build_launch_spec, launch_worker, snapshot_problems
from app.workers.profile_registry import ProfileError, WorkerProfileRegistry
from app.workers.profiles import (
    LEGACY_PROFILE_ID, RUNTIME_CLINE, RUNTIME_CODEX, WorkerLaunchSpec, WorkerProfile, WorkerSnapshot,
)
from app.workers.runtimes import WORKER_RUNTIMES, cline_providers, get_runtime

PY = sys.executable

FAKE_CLINE = r'''
import json, os, subprocess, sys, time
mode, state = sys.argv[1], sys.argv[2]
def out(o, stream=sys.stdout):
    print(json.dumps(o), file=stream, flush=True)
if mode == "history":
    st = json.load(open(state))
    # the launched pid is ours, or our parent's when sys.executable is a venv redirector
    rows = [{"sessionId": "1790000000000_other", "pid": 1, "cwd": st["cwd"], "source": "cli"}] + [
        {"sessionId": st["session"], "pid": pid, "cwd": st["cwd"].replace("\\", "/"), "source": "cli"}
        for pid in (st["pid"], st["ppid"])]
    print(json.dumps(rows)); sys.exit(0)
prompt = sys.stdin.read()
st = {"pid": os.getpid(), "ppid": os.getppid(), "cwd": os.getcwd(), "argv": sys.argv[3:], "prompt": prompt,
      "env": {k: os.environ.get(k) for k in ("CLINE_TEST_FLAG", "CODEX_HOME", "SOME_API_KEY")},
      "session": "1790572420096_fake" + str(os.getpid() % 1000)}
if mode == "hang":
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(120)"])
    st["child"] = child.pid
json.dump(st, open(state, "w"))
out({"ts": "t", "type": "hook_event", "hookEventName": "agent_start", "agentId": "a1", "taskId": "conv_1"})
out({"ts": "t", "type": "agent_event", "event": {"type": "iteration_start", "iteration": 1}})
if mode == "fail":
    out({"ts": "t", "type": "error", "message": "authentication failed for provider openrouter"}, sys.stderr)
    sys.exit(1)
if mode == "hang":
    time.sleep(120); sys.exit(0)
for chunk in ("Work", "ing"):
    out({"ts": "t", "type": "agent_event", "event": {"type": "content_start", "contentType": "text", "text": chunk}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_start", "contentType": "tool", "toolName": "editor",
     "toolCallId": "call_1", "input": {"path": "src/a.txt", "new_text": "x", "apiKey": "sk-should-not-leak"}}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_update", "contentType": "tool", "toolName": "editor",
     "toolCallId": "call_1", "update": "writing"}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_end", "contentType": "tool", "toolName": "editor",
     "toolCallId": "call_1", "output": "ok"}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_start", "contentType": "tool", "toolName": "apply_patch",
     "toolCallId": "call_2", "input": {"input": "*** Begin Patch\n*** Add File: src/b.txt\n+b\n*** End Patch"}}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_end", "contentType": "tool", "toolName": "apply_patch",
     "toolCallId": "call_2", "error": "patch failed"}})
out({"ts": "t", "type": "agent_event", "event": {"type": "error", "error": {"message": "1 tool call(s) failed: "
     "[submit_and_exit] Too small: expected string to have >=10 characters"}, "recoverable": True, "iteration": 1}})
final = "MERHABA" if mode.startswith("smoke") else "SUMMARY: done\nSTATUS: COMPLETE"
if mode != "smoke_submit_only":  # yolo models may answer only inside submit_and_exit
    out({"ts": "t", "type": "agent_event", "event": {"type": "content_end", "contentType": "text", "text": final}})
submitted = final + (" - replied" if mode.startswith("smoke") else "\nFILES CHANGED: src/a.txt")
out({"ts": "t", "type": "agent_event", "event": {"type": "content_start", "contentType": "tool", "toolName": "submit_and_exit",
     "toolCallId": "call_3", "input": {"summary": submitted}}})
out({"ts": "t", "type": "agent_event", "event": {"type": "content_end", "contentType": "tool", "toolName": "submit_and_exit",
     "toolCallId": "call_3", "output": "Submission recorded (verified): " + submitted}})
out({"ts": "t", "type": "agent_event", "event": {"type": "usage", "inputTokens": 5362, "outputTokens": 71,
     "totalInputTokens": 5362, "totalOutputTokens": 71, "totalCost": 0}})
ack = "Submission recorded (verified): " + submitted
out({"ts": "t", "type": "agent_event", "event": {"type": "done", "reason": "completed", "text": ack, "iterations": 1}})
out({"ts": "t", "type": "run_result", "finishReason": "completed", "iterations": 1,
     "usage": {"inputTokens": 5362, "outputTokens": 71, "totalCost": 0}, "text": ack,
     "model": {"id": "mimo-v2.6-pro", "provider": "openai-compatible"}})
'''


def alive(pid: int) -> bool:
    """Liveness without signalling (os.kill on Windows would TERMINATE the process)."""
    h = ctypes.windll.kernel32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return False
    code = ctypes.c_ulong()
    ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
    ctypes.windll.kernel32.CloseHandle(h)
    return code.value == 259  # STILL_ACTIVE


@pytest.fixture
def cline_env(tmp_path, monkeypatch):
    """A machine with a native cline.exe (fake bytes), a Cline configuration with two saved providers, the V5 legacy
    NInfer worker verified, and the fake CLI wired in for every ClineWorker."""
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    monkeypatch.delenv("CLINE_DATA_DIR", raising=False)
    exe = tmp_path / "npm" / "node_modules" / "cline" / "node_modules" / "@cline" / "cli-windows-x64" / "bin" / "cline.exe"
    exe.parent.mkdir(parents=True)
    exe.write_bytes(b"MZ")
    settings_dir = tmp_path / ".cline" / "data" / "settings"
    settings_dir.mkdir(parents=True)
    (settings_dir / "providers.json").write_text(json.dumps({"version": 1, "lastUsedProvider": "openai-compatible",
        "providers": {"openai-compatible": {"settings": {"provider": "openai-compatible", "apiKey": "sk-secret-1",
                                                         "model": "mimo-v2.6-pro", "baseUrl": "https://x/v1"}},
                      "cline": {"settings": {"provider": "cline", "auth": {"accessToken": "tok"},
                                             "model": "stealth/pixel-canary"}}}}), encoding="utf-8")
    ninfer_exe = tmp_path / "bin" / "ninfer" / "codex-ninfer.exe"
    ninfer_exe.parent.mkdir(parents=True)
    ninfer_exe.write_bytes(b"MZ")
    ninfer_home = tmp_path / ".codex-ninfer"
    ninfer_home.mkdir()
    (ninfer_home / "config.toml").write_text('model = "qwen3.8-27b"\n', encoding="utf-8")
    fake = tmp_path / "fake_cline.py"
    fake.write_text(FAKE_CLINE, encoding="utf-8")
    state = tmp_path / "cline_state.json"
    mode = {"value": "ok"}

    def build_argv(self, cwd):
        real = ClineWorker.__dict__["_real_build_argv"](self, cwd)  # validates like production
        return [PY, str(fake), mode["value"], str(state), *real[1:]]

    def history_argv(self):
        return [PY, str(fake), "history", str(state)]

    monkeypatch.setattr(ClineWorker, "_real_build_argv", ClineWorker.build_argv, raising=False)
    monkeypatch.setattr(ClineWorker, "build_argv", build_argv)
    monkeypatch.setattr(ClineWorker, "history_argv", history_argv)
    s = Settings()
    s.integrations.metrics_enabled = False
    s.worker.executable, s.worker.codex_home = str(ninfer_exe), str(ninfer_home)
    s.worker.verified_executable, s.worker.verified_codex_home = str(ninfer_exe), str(ninfer_home)
    return {"settings": s, "exe": str(exe), "state": state, "mode": mode, "tmp": tmp_path}


def cline_profile(env, pid="cline-mimo", verified=True, **kw):
    reg = WorkerProfileRegistry(env["settings"])
    p = reg.create_cline(kw.pop("name", "Cline MiMo"), env["exe"], kw.pop("provider", "openai-compatible"),
                         kw.pop("model", "mimo-v2.6-pro"), profile_id=pid, **kw)
    if verified:
        reg.record_verification(p.id, True, p.executable_path, p.launch_target)
    return reg.require(p.id)


def spec_of(env, p, timeout=120):
    policy = WorkerSettings(timeout_seconds=timeout)
    return build_launch_spec(WorkerSnapshot.of(p), policy, p)


def state(env):
    return json.loads(env["state"].read_text())


# ------------------------------------------------------------ profile model (3, 4)

def test_profiles_without_runtime_default_to_codex():
    assert WorkerProfile.from_dict({"id": "old", "codex_home": "C:/h"}).runtime == RUNTIME_CODEX
    assert WorkerProfile.from_dict({"id": "old", "runtime": ""}).runtime == RUNTIME_CODEX
    snap = WorkerSnapshot.from_dict({"profile_id": "x", "codex_executable": "e", "codex_home": "h"})  # a V5 snapshot
    assert snap.runtime == RUNTIME_CODEX and snap.executable_path == "e" and snap.launch_target == "h"
    assert snap.cline_executable == "" and snap.cline_config_dir == ""


def test_codex_revisions_are_unchanged_by_v51():
    """The V5 revision formula, re-implemented: snapshots of V5 runs must not look 'edited'."""
    import hashlib
    p = WorkerProfile(id="or", display_name="OR", codex_executable="C:/b/codex-ninfer.exe", codex_home="C:/h",
                      extra_env={"RUST_MIN_STACK": "33554432"}, required_env=["OPENROUTER_API_KEY"], model="m")
    n = lambda x: os.path.normcase(os.path.abspath(os.path.expandvars(x))) if x else ""  # noqa: E731
    key = {"runtime": "codex", "exe": n(p.codex_executable), "home": n(p.codex_home),
           "env": sorted(p.extra_env.items()), "required_env": sorted(p.required_env), "model": "",
           "pass_model": False, "args": [], "policy": "codex"}
    assert p.revision() == hashlib.sha256(json.dumps(key, sort_keys=True).encode()).hexdigest()[:12]


def test_the_users_v5_settings_keep_loading(tmp_path):
    """settings.json as V5 wrote it (records carry runtime=codex, no Cline fields)."""
    rec = WorkerProfile(id="openrouter-glm", display_name="OpenRouter GLM", codex_executable="C:/e.exe",
                        codex_home="C:/h").to_dict()
    for k in ("cline_executable", "cline_config_dir"):
        rec.pop(k)
    data = Settings().to_dict()
    data["worker_profiles"]["profiles"] = [rec]
    (tmp_path / "settings.json").write_text(json.dumps(data))
    loaded = WorkerProfileRegistry(SettingsStore(tmp_path / "settings.json").load())
    p = loaded.require("openrouter-glm")
    assert p.runtime == RUNTIME_CODEX and p.executable_path == "C:/e.exe" and p.launch_target == "C:/h"


def test_cline_profile_resolves_to_the_cline_runtime(cline_env):
    p = cline_profile(cline_env)
    assert p.runtime == RUNTIME_CLINE and get_runtime(p.runtime) is WORKER_RUNTIMES["cline"]
    assert p.codex_home == "" and p.executable_path == cline_env["exe"]
    assert p.launch_target == f"cline:provider=openai-compatible;model=mimo-v2.6-pro;config="
    assert p.is_verified()
    w = launch_worker(spec_of(cline_env, p))
    assert isinstance(w, ClineWorker) and not isinstance(w, CodexNInferWorker)
    assert w.capabilities == WorkerCapabilities(tool_call_count=True, subtask_progress=False, sandbox=False)
    assert CodexNInferWorker.capabilities == WorkerCapabilities()


def test_cline_profile_rules(cline_env):
    reg = WorkerProfileRegistry(cline_env["settings"])
    p = cline_profile(cline_env)
    with pytest.raises(ProfileError, match="looks like a secret"):
        reg.update(p.id, extra_env={"OPENROUTER_API_KEY": "sk-x"})
    with pytest.raises(ProfileError, match="only Codex profiles"):
        reg.duplicate(p.id, "copy", copy_home=True)
    reg.update(p.id, model="other-model")
    assert not reg.require(p.id).is_verified(), "a verification never survives a different model"
    copy = reg.duplicate(p.id, "Cline copy")
    assert copy.runtime == RUNTIME_CLINE and copy.provider == "openai-compatible"
    with pytest.raises(ProfileError, match="not editable"):
        reg.update(p.id, runtime="codex")


def test_cline_quick_validation(cline_env, monkeypatch):
    reg = WorkerProfileRegistry(cline_env["settings"])
    p = cline_profile(cline_env, verified=False)
    rep = reg.validate(p.id)
    assert rep.ok and rep.info["runtime"] == "cline" and rep.info["saved_model"] == "mimo-v2.6-pro"
    assert rep.info["cline_providers"] == ["cline", "openai-compatible"]
    assert any("no command sandbox" in w for w in rep.warnings)
    assert any("no headless session resume" in w for w in rep.warnings)
    assert "sk-secret-1" not in json.dumps(rep.info) + "".join(rep.warnings)
    bad = cline_profile(cline_env, pid="cline-bad", verified=False, provider="", model="")
    errs = reg.validate(bad.id).errors
    assert any("no provider" in e for e in errs) and any("no model" in e for e in errs)
    assert cline_providers() == {"openai-compatible": "mimo-v2.6-pro", "cline": "stealth/pixel-canary"}
    for bad_exe in ("cline.cmd", "cline", "cline.ps1"):
        path = cline_env["tmp"] / bad_exe
        path.write_text("@echo off")
        with pytest.raises(ExecutableRejected):
            validate_cline_executable(str(path))


# ------------------------------------------------------------ the adapter (6-10, 15, 16)

def test_cline_argv_cwd_env_and_prompt(cline_env, monkeypatch):
    monkeypatch.delenv("CLINE_TEST_FLAG", raising=False)
    p = cline_profile(cline_env, extra_env={"CLINE_TEST_FLAG": "1"}, extra_args=["--thinking", "low"])
    repo = cline_env["tmp"] / "repo"
    repo.mkdir()
    w = launch_worker(spec_of(cline_env, p))
    real = ClineWorker.__dict__["_real_build_argv"](w, str(repo))
    assert real == [cline_env["exe"], "-y", "--json", "-c", str(repo), "-P", "openai-compatible", "-m", "mimo-v2.6-pro",
                    "--thinking", "low", HEADER], "-y keeps the agent in this process (no shared hub daemon)"
    out = asyncio.run(w.run_turn(PhaseTurnRequest("FULL PHASE PROMPT\nline 2", str(repo), "", "P1", "start")))
    st = state(cline_env)
    assert os.path.normcase(st["cwd"]) == os.path.normcase(str(repo)), "the process runs in the repository"
    assert st["prompt"] == "FULL PHASE PROMPT\nline 2", "the prompt goes through stdin, not the command line"
    assert "FULL PHASE PROMPT" not in " ".join(out.argv)
    assert st["env"] == {"CLINE_TEST_FLAG": "1", "CODEX_HOME": os.environ.get("CODEX_HOME"), "SOME_API_KEY": None}
    assert "CLINE_TEST_FLAG" not in os.environ, "process-local only"
    assert out.env_overrides == {"CLINE_TEST_FLAG": "1"} and "CODEX_HOME" not in out.env_overrides
    assert out.exit_code == 0
    assert out.summary == "SUMMARY: done\nSTATUS: COMPLETE\n\nSUMMARY: done\nSTATUS: COMPLETE\nFILES CHANGED: src/a.txt"
    assert out.tool_calls == 3 and out.usage["inputTokens"] == 5362 and not out.error
    assert out.worker_profile_id == "cline-mimo" and out.worker_profile_revision == p.revision()


def test_cline_events_session_capture_and_callbacks(cline_env):
    p = cline_profile(cline_env)
    w = launch_worker(spec_of(cline_env, p))
    seen = {"tools": [], "files": [], "sessions": [], "events": []}
    w.on_tool_call = seen["tools"].append
    w.on_file_change = lambda kind, path: seen["files"].append((kind, path))
    w.on_session = seen["sessions"].append
    w.on_event = lambda e: seen["events"].append(e.kind)
    logs = []
    w.log = lambda src, text: logs.append((src, text))
    out = asyncio.run(w.run_turn(PhaseTurnRequest("x", str(cline_env["tmp"]), "", "P1", "start")))
    sid = state(cline_env)["session"]
    assert out.session_id == sid and seen["sessions"] == [sid], "captured by exact pid match, persisted at once"
    assert seen["tools"] == [1, 2, 3]
    assert seen["files"] == [("update", "src/a.txt"), ("add", "src/b.txt")]
    assert seen["events"][:5] == ["tool_started", "command_output", "tool_finished", "tool_started", "tool_finished"]
    assert "assistant_text" in seen["events"] and "session_started" in seen["events"]
    assert all(src in ("CLINE", "SYSTEM") for src, _ in logs)
    text = "\n".join(t for _, t in logs)
    assert "▶ editor" in text and "failed: patch failed" in text and "Working" not in text, "chunks are not logged"
    assert "sk-should-not-leak" not in text, "tool inputs are logged from the redacted event"
    assert not out.resumed and not out.resume_failed


def test_cline_never_resumes_headlessly(cline_env):
    p = cline_profile(cline_env)
    w = launch_worker(spec_of(cline_env, p))
    assert asyncio.run(w.session_resume_supported()) is False
    out = asyncio.run(w.run_turn(PhaseTurnRequest("x", str(cline_env["tmp"]), "1790572420096_prev", "P1", "continue")))
    assert out.resume_failed and out.argv == [] and not cline_env["state"].exists(), "nothing was started"


def test_cline_failure_and_exit_codes(cline_env):
    cline_env["mode"]["value"] = "fail"
    w = launch_worker(spec_of(cline_env, cline_profile(cline_env)))
    out = asyncio.run(w.run_turn(PhaseTurnRequest("x", str(cline_env["tmp"]), "", "P1", "start")))
    assert out.exit_code == 1 and "authentication failed" in out.error and "exited with code 1" in out.error
    assert out.tool_calls == 0 and out.summary == ""


def test_cline_timeout_kills_the_worker_tree(cline_env):
    cline_env["mode"]["value"] = "hang"
    w = launch_worker(spec_of(cline_env, cline_profile(cline_env), timeout=3))
    t0 = time.monotonic()
    out = asyncio.run(w.run_turn(PhaseTurnRequest("x", str(cline_env["tmp"]), "", "P1", "start")))
    st = state(cline_env)
    assert out.timed_out and "worker timed out after 3" in out.error and time.monotonic() - t0 < 40
    assert not alive(st["pid"]) and not alive(st["child"]), "the whole worker tree is gone"


def test_cancel_kills_only_the_workers_process_tree(cline_env):
    bystander = subprocess.Popen([PY, "-c", "import time; time.sleep(120)"],
                                 creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)  # e.g. the user's own Cline
    try:
        cline_env["mode"]["value"] = "hang"
        w = launch_worker(spec_of(cline_env, cline_profile(cline_env)))

        async def main():
            task = asyncio.ensure_future(w.run_turn(PhaseTurnRequest("x", str(cline_env["tmp"]), "", "P1", "start")))
            while not cline_env["state"].exists() or "child" not in cline_env["state"].read_text():
                await asyncio.sleep(0.05)
            await asyncio.sleep(0.3)
            await w.cancel()
            return await task

        out = asyncio.run(main())
        st = state(cline_env)
        assert out.cancelled
        assert not alive(st["pid"]) and not alive(st["child"])
        assert alive(bystander.pid), "an unrelated process is never killed (no image-name kill)"
    finally:
        bystander.kill()


def test_events_are_normalized_and_redacted():
    ev = {"type": "agent_event", "event": {"type": "content_start", "contentType": "tool", "toolName": "run_commands",
                                           "toolCallId": "c1", "input": {"commands": ["dir"], "apiKey": "sk-1"}}}
    (n,) = normalize_cline_event(ev)
    assert n.kind == "tool_started" and n.tool_name == "run_commands" and n.tool_id == "c1"
    assert n.raw["event"]["input"]["apiKey"] == "***redacted***" and "sk-1" not in json.dumps(n.raw)
    usage = normalize_cline_event({"type": "agent_event", "event": {"type": "usage", "inputTokens": 5}})[0]
    assert usage.raw["event"]["inputTokens"] == 5, "counts are not secrets"
    assert normalize_cline_event({"type": "agent_event", "event": {"type": "content_start", "contentType": "text",
                                                                   "text": "chunk"}}) == []
    codex = normalize_codex_event({"type": "item.started", "item": {"id": "i1", "type": "command_execution",
                                                                   "command": "dir"}})
    assert codex[0].kind == "tool_started" and codex[0].tool_id == "i1"
    assert normalize_codex_event({"type": "thread.started", "thread_id": "t9"})[0].session_id == "t9"
    assert redact({"headers": {"Authorization": "Bearer x"}})["headers"]["Authorization"] == "***redacted***"


# ------------------------------------------------------------ engine: snapshots, resume, no fallback (11-13, 17, 18)

def _engine(settings, db):
    from app.orchestration.orchestrator import Engine
    return Engine(settings, lambda k, p: None, str(db))


def _run(e, repo, **kw):
    from app.models.run import Run
    run = Run(id=None, goal="g", repo_path=str(repo), worker_mode="ninfer", **kw)
    e.db.create_run(run)
    e.run = run
    return run


def test_active_cline_run_keeps_its_runtime_and_profile(cline_env):
    s = cline_env["settings"]
    p = cline_profile(cline_env)
    WorkerProfileRegistry(s).set_default(p.id)
    e = _engine(s, cline_env["tmp"] / "a.db")
    try:
        run = _run(e, cline_env["tmp"])
        assert isinstance(e._make_worker("ninfer"), ClineWorker)
        snap = e.db.load_run(run.id).worker_snapshot["default"]
        assert snap["runtime"] == "cline" and snap["provider"] == "openai-compatible" and snap["model"] == "mimo-v2.6-pro"
        assert snap["cline_executable"] == cline_env["exe"] and snap["codex_home"] == ""
        reg = WorkerProfileRegistry(s)
        reg.set_default(LEGACY_PROFILE_ID)  # default runtime back to Codex
        reg.update(p.id, model="another-model")
        asyncio.run(e.apply_settings(s))
        w = e._make_worker("ninfer")
        assert isinstance(w, ClineWorker) and w.settings.model == "mimo-v2.6-pro", "the run keeps its snapshot"
        other = _run(e, cline_env["tmp"])
        assert isinstance(e._make_worker("ninfer"), CodexNInferWorker), "a NEW run takes the new default"
        assert e.db.load_run(other.id).worker_snapshot["default"]["runtime"] == "codex"
        assert e.worker_profiles_in_use() == {p.id: [run.id], LEGACY_PROFILE_ID: [other.id]}
    finally:
        e.db.close()


def test_cline_run_resumes_after_restart_with_its_snapshot(cline_env):
    s = cline_env["settings"]
    p = cline_profile(cline_env)
    WorkerProfileRegistry(s).set_default(p.id)
    e = _engine(s, cline_env["tmp"] / "r.db")
    run_id = _run(e, cline_env["tmp"]).id
    e._make_worker("ninfer")
    e.db.close()
    s2 = Settings.from_dict(s.to_dict())
    WorkerProfileRegistry(s2).set_default(LEGACY_PROFILE_ID)
    e2 = _engine(s2, cline_env["tmp"] / "r.db")
    try:
        e2.run = e2.db.load_run(run_id)
        logs = []
        e2.log = lambda src, text: logs.append(text)
        w = e2._make_worker("ninfer")
        assert isinstance(w, ClineWorker) and w.settings.profile_id == p.id and w.settings.provider == "openai-compatible"
        assert "Worker runtime: Cline" in logs and any("no structured subtask progress" in t for t in logs)
        assert any("NO command sandbox" in t for t in logs)
    finally:
        e2.db.close()


@pytest.mark.parametrize("breakage", ["exe_missing", "disabled", "deleted", "unverified", "config_dir_missing"])
def test_unavailable_cline_profile_never_falls_back_to_codex(cline_env, breakage):
    from app.orchestration.orchestrator import EngineError
    s = cline_env["settings"]
    cfg = cline_env["tmp"] / "cline-cfg"
    cfg.mkdir()
    p = cline_profile(cline_env, cline_config_dir=str(cfg))
    reg = WorkerProfileRegistry(s)
    reg.set_default(p.id)
    e = _engine(s, cline_env["tmp"] / "u.db")
    try:
        run = _run(e, cline_env["tmp"])
        e._make_worker("ninfer")
        reg.set_default(LEGACY_PROFILE_ID)  # a perfectly usable Codex default exists
        if breakage == "exe_missing":
            os.remove(cline_env["exe"])
        elif breakage == "disabled":
            reg.set_enabled(p.id, False)
        elif breakage == "deleted":
            s.worker_profiles.profiles = []
        elif breakage == "unverified":
            reg.record_verification(p.id, False)
        elif breakage == "config_dir_missing":
            os.rmdir(cfg)
        asyncio.run(e.apply_settings(s))
        with pytest.raises(EngineError, match="WORKER_PROFILE_UNAVAILABLE") as ei:
            e._make_worker("ninfer")
        assert "No fallback" in str(ei.value) and p.id in str(ei.value)
        assert e.db.load_run(run.id).worker_snapshot["default"]["runtime"] == "cline", "never rebound to Codex"
    finally:
        e.db.close()


def test_snapshot_runtime_mismatch_is_unavailable(cline_env):
    p = cline_profile(cline_env)
    snap = WorkerSnapshot.of(p)
    live = WorkerProfile.from_dict({**p.to_dict(), "runtime": "codex"})
    assert any("bound to cline" in x for x in snapshot_problems(snap, live))


def test_paid_full_test_is_never_triggered_automatically(cline_env, monkeypatch):
    calls = []

    async def smoke(self, metrics_url=""):
        calls.append(self.settings.profile_id)
        raise AssertionError("the paid full test ran without being asked")

    monkeypatch.setattr(ClineWorker, "smoke_test", smoke)
    s = cline_env["settings"]
    p = cline_profile(cline_env)
    reg = WorkerProfileRegistry(s)
    reg.set_default(p.id)
    reg.validate(p.id)
    reg.ready_problems(p)
    e = _engine(s, cline_env["tmp"] / "p.db")
    try:
        asyncio.run(e.refresh_status(None))
        _run(e, cline_env["tmp"])
        e._make_worker("ninfer")
    finally:
        e.db.close()
    assert calls == []


@pytest.mark.parametrize("mode", ["smoke", "smoke_submit_only"])
def test_explicit_full_test_of_a_cline_profile(cline_env, mode):
    cline_env["mode"]["value"] = mode
    s = cline_env["settings"]
    p = cline_profile(cline_env, verified=False)

    async def version(self):
        return "3.0.65"

    ClineWorker.version = version  # restored below
    e = _engine(s, cline_env["tmp"] / "s.db")
    try:
        out = asyncio.run(e.smoke_test_worker(p.id))
        assert out["ok"], out["detail"]
        assert out["runtime"] == "cline" and out["verified_target"] == p.launch_target
        assert out["extra"]["argv"][:2] == [PY, str(cline_env["tmp"] / "fake_cline.py")] and "-y" in out["extra"]["argv"]
        assert out["extra"]["reply"].startswith("MERHABA") and out["extra"]["files_created"] == []
        assert out["extra"]["session_id"] == state(cline_env)["session"]
        assert WorkerProfileRegistry(e.settings).require(p.id).is_verified()
    finally:
        del ClineWorker.version
        e.db.close()


def test_cline_snapshot_round_trips_through_the_v5_database(cline_env):
    from app.persistence.database import Database
    from app.models.run import Run
    from app.workers.profiles import RunWorkerProfiles
    p = cline_profile(cline_env)
    db = Database(cline_env["tmp"] / "d.db")
    run = Run(id=None, goal="g", repo_path="C:/r")
    db.create_run(run)
    assert db.lock_worker_snapshot(run.id, RunWorkerProfiles(default=WorkerSnapshot.of(p)).to_dict())
    back = RunWorkerProfiles.from_dict(db.load_run(run.id).worker_snapshot).default
    assert back == WorkerSnapshot.of(p, now=back.locked_at) and back.runtime == "cline"
    db.close()


# ------------------------------------------------------------ the pipeline with a Cline worker (5)

def test_a_v2_phase_run_executes_with_the_cline_runtime(cline_env, monkeypatch):
    """PLAN -> START with a Cline default profile: the phase scheduler drives Cline through the same contract as
    Codex. Every turn is a new Cline session (explicit, logged); the pipeline completes."""
    from app.orchestration.providers.mock import MockOrchestrator
    s = cline_env["settings"]
    s.orchestration.worker_mode = "ninfer"
    s.orchestration.mock_step_delay = 0.0
    s.orchestration.mock_retry_task_ids = []
    s.integrations.codebase_memory_executable = ""
    p = cline_profile(cline_env)
    WorkerProfileRegistry(s).set_default(p.id)
    repo = cline_env["tmp"] / "proj"
    repo.mkdir()
    (repo / "README.md").write_text("x\n")
    for args in (["init", "-q", "-b", "main"], ["add", "-A"],
                 ["-c", "user.email=t@e", "-c", "user.name=t", "commit", "-qm", "i"]):
        subprocess.run(["git", *args], cwd=repo, check=True, capture_output=True)
    e = _engine(s, cline_env["tmp"] / "v2.db")
    e._make_orchestrator = lambda mode: MockOrchestrator(delay=0.0)
    logs = []
    orig_log = e.log
    e.log = lambda src, text: (logs.append((src, text)), orig_log(src, text))

    async def main():
        await e.plan("Build the feature", str(repo), pipeline="v2")
        while e._busy is not None and not e._busy.done():
            await asyncio.sleep(0.01)
        await e.start()
        while e._busy is not None and not e._busy.done():
            await asyncio.sleep(0.01)

    try:
        asyncio.run(main())
        assert e.sm.state.value == "COMPLETED", e.run.status_message
        texts = [t for _, t in logs]
        assert "Worker runtime: Cline" in texts and "Worker profile: Cline MiMo (cline-mimo, revision " \
            f"{p.revision()})" in texts
        rows = e.db.conn.execute("SELECT worker_profile_id FROM worker_calls").fetchall()
        assert rows and {r[0] for r in rows} == {"cline-mimo"}
        phases = e.run.plan.tasks
        assert all(ph.session_id for ph in phases), "the captured Cline session is recorded on the phase"
    finally:
        e.db.close()
