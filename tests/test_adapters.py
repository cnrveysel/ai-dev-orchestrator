"""Adapter output parsing, exercised with fake CLI scripts that emit the real
event formats observed from the installed Claude Code CLI and codex-ninfer."""
import asyncio
import json
import sys
import time

import pytest

from app.config.settings import ClaudeSettings, WorkerSettings
from app.models.plan import Plan
from app.models.task import Task
from app.orchestration.providers.base import OrchestratorError, PlanRequest
from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
from app.workers.base import RetryContext, WorkerTaskSpec, build_worker_prompt
from app.workers.codex_ninfer import CodexNInferWorker

PY = sys.executable

PLAN = {"goal": "g", "summary": "s", "tasks": [
    {"id": "T001", "title": "A", "description": "d", "acceptance_criteria": ["x"]}]}

FAKE_CLAUDE = r'''
import json, sys, time
prompt = sys.stdin.read()  # like the real CLI: blocks until stdin EOF
mode, payload, counter = sys.argv[1], sys.argv[2], sys.argv[3]
open(counter, "a").write("x")
def out(o): print(json.dumps(o), flush=True)
def assistant(i, blocks): out({"type": "assistant", "message": {"id": f"msg_{i}", "content": blocks}})
def result():
    out({"type": "result", "subtype": "success", "is_error": False, "total_cost_usd": 0.0123, "num_turns": 2,
         "usage": {"input_tokens": 5, "cache_creation_input_tokens": 100, "cache_read_input_tokens": 50,
                   "output_tokens": 7},
         "session_id": "abc", "structured_output": json.loads(payload)})
out({"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": ["Read"]})
assistant(0, [{"type": "text", "text": "Looking at the repo"},
              {"type": "tool_use", "name": "Glob", "input": {"pattern": "*.md"}}])
if mode == "ok":
    result()
elif mode == "result_then_hang":
    result()
    time.sleep(60)
elif mode == "error":
    out({"type": "result", "subtype": "error_during_execution", "is_error": True, "result": "auth failed"})
elif mode == "budget":
    out({"type": "result", "subtype": "error_max_budget_usd", "is_error": True, "result": ""})
elif mode == "many_turns":
    for i in range(1, 20):
        assistant(i, [{"type": "tool_use", "name": "Read", "input": {"file_path": f"f{i}"}}])
        time.sleep(0.2)
    time.sleep(60)
elif mode == "many_tools":
    for i in range(1, 20):
        assistant(i, [{"type": "tool_use", "name": "Grep", "input": {"pattern": str(j)}} for j in range(3)])
        time.sleep(0.2)
    time.sleep(60)
elif mode == "hang":
    time.sleep(60)
else:
    sys.exit(1)
'''

FAKE_CODEX = r'''
import json, sys
prompt = sys.stdin.read()
last = sys.argv[1]
def out(o): print(json.dumps(o), flush=True)
sys.stderr.write("ERROR rmcp::transport::worker: worker quit with fatal: Transport channel closed, http://127.0.0.1:8003/mcp\n")
out({"type": "thread.started", "thread_id": "t1"})
out({"type": "turn.started"})
out({"type": "item.started", "item": {"id": "i1", "type": "command_execution", "command": "dir", "status": "in_progress"}})
out({"type": "item.completed", "item": {"id": "i1", "type": "command_execution", "command": "dir", "aggregated_output": "a.txt", "exit_code": 0}})
out({"type": "item.completed", "item": {"id": "i2", "type": "mcp_tool_call", "server": "codebase_memory", "tool": "search_code", "arguments": {"q": "x"}, "status": "completed"}})
out({"type": "item.completed", "item": {"id": "i3", "type": "file_change", "changes": [{"path": "a.txt", "kind": "update"}]}})
out({"type": "item.completed", "item": {"id": "i4", "type": "agent_message", "text": "SUMMARY: done"}})
out({"type": "turn.completed", "usage": {"input_tokens": 100, "output_tokens": 20}})
open(last, "w", encoding="utf-8").write("SUMMARY: done\nFILES CHANGED: a.txt")
print("PROMPT_LEN", len(prompt), file=sys.stderr)
'''


@pytest.fixture
def fake_claude(tmp_path):
    p = tmp_path / "fake_claude.py"
    p.write_text(FAKE_CLAUDE)
    return p


def make_provider(fake, mode, payload, tmp_path, **settings_kw):
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused", **settings_kw))
    logs = []
    prov.log = lambda s, t: logs.append((s, t))
    counter = tmp_path / f"invocations-{mode}.txt"

    async def argv(limits, schema):
        return [PY, str(fake), mode, json.dumps(payload), str(counter)]
    prov._build_argv = argv
    return prov, logs, counter


def invocations(counter) -> int:
    return len(counter.read_text()) if counter.exists() else 0


def test_claude_plan_parses_structured_output_and_logs_usage(fake_claude, tmp_path):
    prov, logs, _ = make_provider(fake_claude, "ok", PLAN, tmp_path)
    plan = asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))
    assert isinstance(plan, Plan) and plan.tasks[0].id == "T001"
    call = prov.calls[0]
    assert call.cost_usd == pytest.approx(0.0123) and call.session_id == "abc"
    assert call.kind == "PLAN" and call.model == "claude-opus-5-5"
    assert call.tool_calls == 1 and call.assistant_turns == 1 and call.num_turns == 2
    system = [t for s, t in logs if s == "SYSTEM"]
    assert any("stdin closed (EOF)" in t for t in system)
    usage = next(t for t in system if t.startswith("Claude usage [PLAN]"))
    for part in ["model=claude-opus-5-5", "elapsed=", "turns=1 (CLI num_turns=2)", "tool_calls=1",
                 "input=5", "cache_write=100", "cache_read=50", "output=7", "$0.0123 (reported by Claude CLI)"]:
        assert part in usage


def test_claude_error_result_raises(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "error", {}, tmp_path)
    with pytest.raises(OrchestratorError, match="auth failed"):
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))
    prov, _, _ = make_provider(fake_claude, "crash", {}, tmp_path)
    with pytest.raises(OrchestratorError):
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))


def test_claude_invalid_plan_raises(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "ok", {"goal": "g", "summary": "", "tasks": []}, tmp_path)
    with pytest.raises(OrchestratorError, match="invalid plan"):
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))


def _timed_plan(prov, tmp_path):
    t0 = time.monotonic()
    with pytest.raises(OrchestratorError) as ei:
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))
    return str(ei.value), time.monotonic() - t0


def test_claude_turn_limit_terminates(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "many_turns", PLAN, tmp_path)
    msg, elapsed = _timed_plan(prov, tmp_path)
    assert "turn limit exceeded (7 > 6)" in msg and elapsed < 20


def test_claude_tool_call_limit_terminates(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "many_tools", PLAN, tmp_path)
    msg, elapsed = _timed_plan(prov, tmp_path)
    assert "tool-call limit exceeded (9 > 8)" in msg and elapsed < 20


def test_claude_hard_timeout_terminates(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "hang", PLAN, tmp_path, plan_timeout_seconds=2)
    msg, elapsed = _timed_plan(prov, tmp_path)
    assert "hard timeout after 2s" in msg and elapsed < 20


def test_claude_that_lingers_after_result_is_terminated_and_result_kept(fake_claude, tmp_path, monkeypatch):
    from app.orchestration.providers import claude_code_cli
    monkeypatch.setattr(claude_code_cli, "RESULT_EXIT_GRACE_SECONDS", 0.5)
    prov, logs, _ = make_provider(fake_claude, "result_then_hang", PLAN, tmp_path)
    t0 = time.monotonic()
    plan = asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))
    assert plan.tasks[0].id == "T001" and time.monotonic() - t0 < 20
    assert any("did not exit within" in t for s, t in logs)


def test_claude_budget_error_is_reported(fake_claude, tmp_path):
    prov, _, _ = make_provider(fake_claude, "budget", PLAN, tmp_path)
    with pytest.raises(OrchestratorError, match=r"budget limit \$0\.25"):
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))


@pytest.mark.parametrize("configured,expected", [(0, 1), (1, 2), (7, 2)])
def test_claude_is_not_retried_unless_configured_and_at_most_once(fake_claude, tmp_path, configured, expected):
    prov, _, counter = make_provider(fake_claude, "crash", PLAN, tmp_path, auto_retries=configured)
    with pytest.raises(OrchestratorError):
        asyncio.run(prov.plan(PlanRequest("g", str(tmp_path))))
    assert invocations(counter) == expected


def _has(argv, a, b):
    return any(argv[i] == a and argv[i + 1] == b for i in range(len(argv) - 1))


def test_claude_argv_applies_guards_only_with_supported_flags(tmp_path):
    from app.orchestration.providers.claude_code_cli import OPTIONAL_FLAGS, REQUIRED_FLAGS, limits_for
    exe = tmp_path / "claude.exe"
    exe.write_bytes(b"MZ")
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable=str(exe)))
    prov._flags = set(REQUIRED_FLAGS + OPTIONAL_FLAGS) - {"--max-turns"}  # as in the installed CLI

    def argv(purpose):
        return asyncio.run(prov._build_argv(limits_for(purpose, prov.settings), {}))

    plan, review, smoke = argv("PLAN"), argv("REVIEW"), argv("SMOKE")
    for a in (plan, review, smoke):
        assert _has(a, "--permission-prompts", "none") and _has(a, "--permission-mode", "dontAsk")
        assert "--max-turns" not in a
    assert _has(plan, "--tools", "Read,Glob,Grep") and _has(plan, "--max-budget-usd", "0.25")
    assert _has(review, "--tools", "") and "--allowedTools" not in review
    assert _has(review, "--max-budget-usd", "0.20") and _has(review, "--model", "claude-opus-5-5")
    r = limits_for("REVIEW", prov.settings)
    assert r.tools == "" and r.max_tool_calls == 0 and r.max_turns == 2
    assert _has(smoke, "--tools", "") and "--allowedTools" not in smoke
    assert limits_for("PLAN", prov.settings).timeout == 240  # V2.1 planner default
    assert limits_for("REVIEW", prov.settings).timeout == 60
    assert limits_for("SMOKE", prov.settings).timeout == 45
    prov._flags.add("--max-turns")
    assert _has(argv("PLAN"), "--max-turns", "6") and _has(argv("REVIEW"), "--max-turns", "2")
    prov._flags -= {"--max-budget-usd", "--permission-prompts"}
    a = argv("PLAN")
    assert "--max-budget-usd" not in a and "--permission-prompts" not in a


def test_codex_worker_parses_events_and_counts_tools(tmp_path):
    fake = tmp_path / "fake_codex.py"
    fake.write_text(FAKE_CODEX)
    home = tmp_path / ".codex-ninfer"
    home.mkdir()
    w = CodexNInferWorker(WorkerSettings(executable="unused", codex_home=str(home), timeout_seconds=30))
    logs = []
    w.log = lambda s, t: logs.append((s, t))

    async def argv(repo, last, sandbox, ephemeral=False, skip_git=False):
        return [PY, str(fake), last]
    w._build_argv = argv
    spec = WorkerTaskSpec(task=Task(id="T001", title="t", acceptance_criteria=["c"]), repo_path=str(tmp_path))
    out = asyncio.run(w.execute_task(spec))
    assert out.exit_code == 0
    assert out.tool_calls == 3
    assert out.summary.startswith("SUMMARY: done")
    assert out.usage["output_tokens"] == 20
    assert out.env_overrides["CODEX_HOME"] == str(home)
    assert out.degraded_notes, "MCP connection failure must be reported as degraded mode"
    worker_lines = [t for s, t in logs if s != "SYSTEM"]
    assert worker_lines and all(s in ("QWEN", "SYSTEM") for s, _ in logs), "worker output is labelled QWEN"
    assert any("mcp codebase_memory.search_code" in t for t in worker_lines)
    assert any("[stderr]" in t and "rmcp" in t for t in worker_lines)
    assert any(t.startswith("Worker CODEX_HOME (process-local): ") for s, t in logs if s == "SYSTEM")


FAKE_SMOKE = r'''
import json, os, sys
sys.stdin.read()
last, reply, mode = sys.argv[1], sys.argv[2], sys.argv[3]
print(json.dumps({"type": "item.completed", "item": {"id": "i0", "type": "agent_message", "text": reply}}), flush=True)
if mode == "write":
    open("stray.txt", "w").write("x")
open(last, "w", encoding="utf-8").write(reply)
sys.exit(int(mode) if mode.isdigit() else 0)
'''


@pytest.mark.parametrize("reply,mode,ok", [
    ("QWEN_WORKER_OK", "0", True),
    ("QWEN_WORKER_OK\n", "0", True),
    ("QWEN_WORKER_OK.", "0", False),  # must be exact
    ("Sure! QWEN_WORKER_OK", "0", False),
    ("QWEN_WORKER_OK", "1", False),  # exit code must be 0
    ("QWEN_WORKER_OK", "write", False),  # no file edits allowed
])
def test_worker_smoke_test_checks(tmp_path, reply, mode, ok):
    fake = tmp_path / "fake_smoke.py"
    fake.write_text(FAKE_SMOKE)
    exe = tmp_path / "codex-ninfer.exe"
    exe.write_bytes(b"MZ")
    home = tmp_path / ".codex-ninfer"
    home.mkdir()
    w = CodexNInferWorker(WorkerSettings(executable=str(exe), codex_home=str(home)))
    seen = {}

    async def caps():
        return {"version": "codex-cli test", "flags": [], "missing": []}

    async def argv(repo, last, sandbox, ephemeral=False, skip_git=False):
        seen.update(sandbox=sandbox, ephemeral=ephemeral, repo=repo)
        return [PY, str(fake), last, reply, mode]
    w.detect_capabilities = caps
    w._build_argv = argv
    res = asyncio.run(w.smoke_test())
    assert res.ok is ok, res.detail
    assert res.executable == str(exe)
    assert res.extra["codex_home"] == str(home)
    assert seen["sandbox"] == "read-only" and seen["ephemeral"]


def test_worker_verification_is_bound_to_exe_and_codex_home():
    s = WorkerSettings(executable=r"C:\x\codex-ninfer.exe", codex_home=r"C:\u\.codex-ninfer")
    assert not s.is_verified()
    s.verified_executable = r"C:\x\codex-ninfer.exe"
    assert not s.is_verified(), "an executable-only record (older smoke test) does not unlock REAL mode"
    s.verified_codex_home = r"C:\u\.codex-ninfer"
    assert s.is_verified()
    assert not s.verified_for(r"C:\x\codex-ninfer.exe", r"C:\u\.codex")
    assert not s.verified_for(r"C:\y\codex-ninfer.exe", r"C:\u\.codex-ninfer")


def test_worker_rejects_normal_codex_home(tmp_path, monkeypatch):
    monkeypatch.setenv("USERPROFILE", str(tmp_path))
    (tmp_path / ".codex").mkdir()
    w = CodexNInferWorker(WorkerSettings(codex_home=str(tmp_path / ".codex")))
    from app.workers.base import WorkerError
    with pytest.raises(WorkerError):
        w._check_codex_home()


def test_worker_prompt_contains_task_and_retry_feedback_not_transcript():
    t = Task(id="T002", title="Door", description="Implement door", acceptance_criteria=["opens"],
             instructions="Use IInteractable")
    spec = WorkerTaskSpec(task=t, repo_path="C:/repo", retry=RetryContext(2, 3, "prev summary",
                                                                           ["did not compile"], ["fix include"]))
    p = build_worker_prompt(spec)
    for s in ["T002", "Implement door", "opens", "Use IInteractable", "ATTEMPT 2 OF 3", "did not compile",
              "fix include", "prev summary"]:
        assert s in p
