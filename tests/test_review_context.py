"""Regression tests for the compact, hard-bounded Claude REVIEW context.

Reproduces the SampleGame failure: a task that created node_modules/ and dist/
in a repository without .gitignore (~2561 changed paths, ~617 KB review prompt).
No real Claude or Codex process is ever launched (see conftest guard).
"""
import asyncio
import json
import subprocess
import sys
import time

import pytest

from app.config.settings import ClaudeSettings
from app.integrations import path_filter
from app.integrations.git import GitRepo
from app.models.attempt import Attempt, ChangedFile, VerificationResult
from app.models.task import Task
from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
from app.orchestration.reviewer import (
    REVIEW_PROMPT_BYTES_CEILING, ReviewContextLimits, ReviewContextTooLarge, ReviewPackage,
    build_review_context, parse_worker_result, utf8_len,
)
from app.workers.process_runner import ProcessRunner

PY = sys.executable
LIMITS = ReviewContextLimits()  # defaults: 40000 / 25000 / 30

FINAL_MESSAGE = """Done.

SUMMARY: Implemented player movement with WASD input and a fixed-step game loop.
FILES CHANGED: src/Player.ts, src/Game.ts, package-lock.json
DECISIONS: Used requestAnimationFrame with a fixed 60 Hz simulation step.
RISKS: Diagonal speed is not normalised yet.
VERIFICATION: npm run build succeeded; unit tests pass."""


def git(repo, *args):
    subprocess.run(["git", "-c", "user.email=t@example.com", "-c", "user.name=t", *args],
                   cwd=repo, check=True, capture_output=True)


def write(path, text):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def make_task():
    return Task(id="T001", title="Player movement",
                description="Add WASD player movement to the game loop.",
                instructions="Edit src/Player.ts and src/Game.ts. Keep the fixed-step loop.",
                acceptance_criteria=["Player moves with WASD", "npm run build succeeds"],
                files_likely_involved=["src/Player.ts", "src/Game.ts"],
                verification_commands=["npm run build", "python -m unittest"])


def make_verification():
    build_out = "\n".join(f"vite build line {i}" for i in range(500)) + "\n✓ built in 2.70s"
    return [VerificationResult("npm run build", 0, build_out, 2.7),
            VerificationResult("python -m unittest", 0, "....\n" + "-" * 70 + "\nRan 42 tests in 0.120s\n\nOK", 1.2)]


def worker_noise(a: Attempt) -> None:
    """A long Qwen transcript that must never reach the reviewer."""
    a.stdout_tail = ('{"type":"item.completed","item":{"type":"reasoning","text":"QWEN_TRANSCRIPT_MARKER"}}\n'
                     * 2500)
    a.stderr_tail = "QWEN_STDERR_MARKER\n" * 200
    a.tool_call_log = ["$ npm install TOOL_CALL_MARKER"] * 400


@pytest.fixture
def sample_game_repo(tmp_path):
    """Fake repo WITHOUT .gitignore: src/, README, package.json, package-lock.json committed."""
    r = tmp_path / "SampleGame"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    write(r / "src" / "Player.ts", "export class Player {\n  x = 0;\n}\n")
    write(r / "src" / "Game.ts", "export class Game {\n  tick() {}\n}\n")
    write(r / "README.md", "# Sausage City\n")
    write(r / "package.json", '{\n  "name": "sausage-city",\n  "version": "0.0.1"\n}\n')
    write(r / "package-lock.json", "{\n" + "".join(f'  "baseline-{i}": "1.0.0",\n' for i in range(3000)) + "}\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "init")
    return r


def simulate_worker(r):
    """src/Player.ts and src/Game.ts modified; dependencies + bundles + caches generated."""
    write(r / "src" / "Player.ts", "export class Player {\n  x = 0;\n  // PLAYER_CHANGE_MARKER\n"
                                   "  move(dx: number) { this.x += dx; }\n}\n")
    write(r / "src" / "Game.ts", "export class Game {\n  // GAME_CHANGE_MARKER\n  tick() { this.step(); }\n"
                                 "  step() {}\n}\n")
    for i in range(1500):
        write(r / "node_modules" / f"pkg{i}" / "index.js", f"module.exports = 'NODE_MODULES_CONTENT_MARKER {i}';\n")
        write(r / "node_modules" / f"pkg{i}" / "package.json", '{"name": "pkg%d"}\n' % i)
    write(r / "node_modules" / ".package-lock.json", "{}\n")
    write(r / "dist" / "index.html", "<script src=assets/index-abc123.js></script>\n")
    write(r / "dist" / "assets" / "index-abc123.js", "/*DIST_BUNDLE_MARKER*/" + "var a=1;" * 30000 + "\n")
    write(r / ".vite" / "deps" / "_metadata.json", '{"VITE_CACHE_MARKER": true}\n')
    write(r / "package-lock.json", "{\n" + "".join(f'  "PKGLOCK_CHANGE_MARKER-{i}": "2.0.0",\n'
                                                    for i in range(3000)) + "}\n")


def attempt_from_changes(ch) -> Attempt:
    a = Attempt(task_id="T001", number=1, started_at=time.time() - 30, finished_at=time.time(), exit_code=0,
                executable="codex-ninfer.exe", worker_summary=FINAL_MESSAGE, tool_calls=400)
    a.changed_files = ch.changed_files
    a.preexisting_dirty = ch.preexisting_dirty
    a.generated_excluded = ch.generated_excluded
    a.generated_by_dir = ch.generated_by_dir
    a.diff_stat = ch.diff_stat
    a.diff = ch.diff
    a.verification = make_verification()
    worker_noise(a)
    return a


# ------------------------------------------------------------ the SampleGame regression

def test_sample_game_regression_review_context_is_compact(sample_game_repo):
    r = sample_game_repo

    async def main():
        g = GitRepo(str(r))
        pre = await g.snapshot(True)
        simulate_worker(r)
        post = await g.snapshot(True)
        ch = await g.attempt_changes(pre, post)
        listed = subprocess.run(["git", "ls-tree", "-r", "--name-only", post.tree], cwd=r,
                                capture_output=True, text=True).stdout
        return ch, listed

    ch, post_tree_paths = asyncio.run(main())

    # Attribution: generated/dependency paths are excluded inside the orchestrator (no .gitignore).
    assert {c.path for c in ch.changed_files} == {"src/Player.ts", "src/Game.ts", "package-lock.json"}
    assert ch.generated_excluded >= 3000 + 3
    assert ch.generated_by_dir["node_modules"] >= 3000 and ch.generated_by_dir["dist"] == 2
    assert "node_modules" not in post_tree_paths and "dist/" not in post_tree_paths  # never snapshotted
    assert "PKGLOCK_CHANGE_MARKER" not in ch.diff and "NODE_MODULES_CONTENT_MARKER" not in ch.diff

    ctx = build_review_context(ReviewPackage(make_task(), attempt_from_changes(ch), 3), LIMITS)
    p = ctx.prompt
    # excluded
    assert "node_modules/" not in p and "NODE_MODULES_CONTENT_MARKER" not in p
    assert "dist/" not in p and "DIST_BUNDLE_MARKER" not in p
    assert ".vite/" not in p and "VITE_CACHE_MARKER" not in p
    # sources included
    assert "PLAYER_CHANGE_MARKER" in p and "GAME_CHANGE_MARKER" in p
    assert "M src/Player.ts" in p and "M src/Game.ts" in p
    # lockfile listed as metadata, full diff omitted
    assert "M package-lock.json  [diff omitted: lockfile]" in p and "PKGLOCK_CHANGE_MARKER" not in p
    # worker: final result only, never the transcript / tool calls / stderr
    assert "Summary: Implemented player movement" in p and "Risks: Diagonal speed" in p
    for marker in ("QWEN_TRANSCRIPT_MARKER", "TOOL_CALL_MARKER", "QWEN_STDERR_MARKER"):
        assert marker not in p
    # verification summary, successful output capped at ~10 lines
    assert "npm run build: PASS (2.7s)" in p and "python -m unittest: PASS (1.2s)" in p
    assert "Ran 42 tests" in p and "✓ built in 2.70s" in p
    assert p.count("vite build line") <= 10 and "vite build line 0\n" not in p
    # bounded
    assert utf8_len(p) <= 40000 and ctx.stats["final_prompt_bytes"] == utf8_len(p)
    s = ctx.stats
    assert s["authored_paths"] == 3 and s["generated_paths_excluded"] == ch.generated_excluded
    assert s["changed_paths_total"] == 3 + ch.generated_excluded
    assert s["files_included"] == 2 and s["files_listed"] == 3


def test_attempt_stored_before_the_fix_is_filtered_too():
    """SampleGame T001 as stored in the DB: 2561 changed paths incl. node_modules/dist and a
    60 KB diff that is mostly node_modules content."""
    changed = [ChangedFile(f"node_modules/pkg{i}/index.js", "A", True) for i in range(2548)]
    changed += [ChangedFile(p, "A", True) for p in ("dist/index.html", "dist/assets/index-abc.js", "dist/vite.svg")]
    authored = ["README.md", "index.html", "package.json", "package-lock.json", "src/main.ts", "src/Player.ts",
                "src/Game.ts", "src/style.css", "tsconfig.json", "vite.config.ts"]
    changed += [ChangedFile(p, "A", True) for p in authored]
    diff = "".join(f"diff --git a/node_modules/pkg{i}/index.js b/node_modules/pkg{i}/index.js\nnew file mode 100644\n"
                   f"--- /dev/null\n+++ b/node_modules/pkg{i}/index.js\n@@ -0,0 +1 @@\n+NODE_MODULES_CONTENT_MARKER\n"
                   for i in range(400))[:60000] + "\n...[diff truncated]..."
    stat = "\n".join(f" node_modules/pkg{i}/index.js | 1 +" for i in range(2548)) + "\n src/Player.ts | 5 +\n"
    a = Attempt(task_id="T001", number=1, started_at=0.0, finished_at=30.0, exit_code=0,
                worker_summary=FINAL_MESSAGE, changed_files=changed, diff=diff, diff_stat=stat,
                verification=make_verification())
    worker_noise(a)
    ctx = build_review_context(ReviewPackage(make_task(), a, 3), LIMITS)
    assert ctx.stats["changed_paths_total"] == 2561
    assert ctx.stats["generated_paths_excluded"] == 2551 and ctx.stats["authored_paths"] == 10
    assert "node_modules/" not in ctx.prompt and "NODE_MODULES_CONTENT_MARKER" not in ctx.prompt
    assert utf8_len(ctx.prompt) <= 40000
    assert "(no source/config/doc diff for the listed files)" in ctx.prompt  # stored diff had none


# ------------------------------------------------------------ budgets

def _attempt_with_sources(n_files: int, bytes_per_file: int, generated: int = 0) -> Attempt:
    changed, sections = [], []
    for i in range(n_files):
        path = f"src/mod{i:02d}.ts"
        changed.append(ChangedFile(path, "M", True))
        n_lines = max(1, bytes_per_file // 34)
        body = "".join(f"+  const v{j} = {i}; // FILE{i:02d}\n" for j in range(n_lines))
        sections.append(f"diff --git a/{path} b/{path}\n--- a/{path}\n+++ b/{path}\n@@ -1 +1,{n_lines} @@\n{body}")
    changed += [ChangedFile(f"node_modules/x{i}.js", "A", True) for i in range(generated)]
    return Attempt(task_id="T001", number=1, started_at=0.0, finished_at=10.0, exit_code=0,
                   worker_summary=FINAL_MESSAGE, changed_files=changed, diff="".join(sections),
                   verification=make_verification())


def test_diff_budget_prioritises_task_files_and_marks_truncation():
    a = _attempt_with_sources(12, 5000)
    task = make_task()
    task.files_likely_involved = ["src/mod07.ts"]
    ctx = build_review_context(ReviewPackage(task, a, 3), LIMITS)
    s = ctx.stats
    assert s["diff_included_bytes"] <= 25000 < s["diff_original_bytes"]
    assert f"[DIFF TRUNCATED: original {s['diff_original_bytes']} bytes, included {s['diff_included_bytes']} bytes]" in ctx.prompt
    diff_part = ctx.prompt.split("RELEVANT DIFF", 1)[1]
    assert diff_part.index("FILE07") < diff_part.index("FILE00")  # task file first
    assert "[diff omitted for:" in ctx.prompt
    assert utf8_len(ctx.prompt) <= 40000


def test_changed_file_limit_counts_only_authored_paths():
    a = _attempt_with_sources(45, 100, generated=500)
    ctx = build_review_context(ReviewPackage(make_task(), a, 3), LIMITS)
    assert ctx.stats["authored_paths"] == 45 and ctx.stats["files_listed"] == 30
    assert ctx.stats["generated_paths_excluded"] == 500
    assert "Files (showing 30 of 45, most relevant first):" in ctx.prompt
    assert "... 15 more authored path(s) omitted" in ctx.prompt


def test_optional_context_is_compacted_to_fit():
    a = _attempt_with_sources(12, 5000)
    a.verification = [VerificationResult("npm test", 1, "\n".join(f"error line {i} " + "x" * 150 for i in range(400)), 3.0)]
    task = make_task()
    task.description = "Long but mandatory description. " * 700  # ~23 KB
    ctx = build_review_context(ReviewPackage(task, a, 3), LIMITS)
    assert utf8_len(ctx.prompt) <= 40000
    assert ctx.stats["diff_included_bytes"] < 25000  # diff shrank to make room
    assert "npm test: FAIL (exit 1, 3.0s)" in ctx.prompt and "error line 399" in ctx.prompt  # failure tail kept


def test_mandatory_context_over_the_limit_raises():
    task = make_task()
    task.acceptance_criteria = [f"criterion {i}: " + "y" * 1000 for i in range(60)]
    with pytest.raises(ReviewContextTooLarge, match=r"^Reviewer context exceeded hard limit: \d+ > 40000 bytes$"):
        build_review_context(ReviewPackage(task, _attempt_with_sources(1, 100), 3), LIMITS)


def test_prompt_limit_cannot_be_configured_above_the_ceiling():
    s = ClaudeSettings(review_max_prompt_bytes=617_000, review_max_diff_bytes=900_000)
    lim = ReviewContextLimits.from_settings(s)
    assert lim.max_prompt_bytes == REVIEW_PROMPT_BYTES_CEILING == 80_000
    assert lim.max_diff_bytes <= lim.max_prompt_bytes
    assert ReviewContextLimits.from_settings(ClaudeSettings()) == ReviewContextLimits(40000, 25000, 30)


def test_worker_result_parsing_uses_only_labelled_final_fields():
    f = parse_worker_result("**SUMMARY:** did it\nmore detail\nFILES CHANGED: a.ts\nBLOCKERS: none")
    assert f == {"Summary": "did it\nmore detail", "Files changed": "a.ts", "Blockers": "none"}
    assert parse_worker_result("plain final answer") == {"Final message": "plain final answer"}


@pytest.mark.parametrize("path,kind", [
    ("node_modules/a/b.js", "generated"), ("packages/x/node_modules/y.js", "generated"),
    ("dist/assets/app.js", "generated"), ("build/out.txt", "generated"), ("coverage/lcov.info", "generated"),
    (".vite/deps/x.json", "generated"), (".next/cache/a", "generated"), (".cache/x", "generated"),
    (".venv/Lib/site.py", "generated"), ("venv/bin/python", "generated"), ("src/__pycache__/a.pyc", "generated"),
    ("Binaries/Win64/Game.dll", "generated"), ("Plugins/P/Intermediate/x.h", "generated"),
    ("Saved/Logs/Game.log", "generated"), ("DerivedDataCache/a.ddc", "generated"), ("src/app.js.map", "generated"),
    ("src/Player.ts", "diffable"), ("README.md", "diffable"), (".gitignore", "diffable"), ("pyproject.toml", "diffable"),
    ("package-lock.json", "metadata"), ("yarn.lock", "metadata"), ("Content/Hero.uasset", "metadata"),
    ("assets/logo.png", "metadata"),
])
def test_path_classification(path, kind):
    assert path_filter.classify(path) == kind


# ------------------------------------------------------------ adapter: never launch an oversized review

FAKE_REVIEW_CLAUDE = r'''
import json, sys
data = sys.stdin.buffer.read()
open(sys.argv[1], "w").write(str(len(data)))
print(json.dumps({"type": "system", "subtype": "init", "model": "claude-opus-5-5", "tools": []}), flush=True)
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1,
                  "total_cost_usd": 0.01, "usage": {"input_tokens": 3, "output_tokens": 50},
                  "structured_output": {"decision": "PASS", "summary": "ok", "problems": [],
                                        "corrective_instructions": [], "verification_notes": [],
                                        "plan_changes_required": False}}), flush=True)
'''


def _provider(tmp_path):
    fake = tmp_path / "fake_review_claude.py"
    fake.write_text(FAKE_REVIEW_CLAUDE)
    received = tmp_path / "received_bytes.txt"
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused"))
    logs, seen_limits = [], []
    prov.log = lambda s, t: logs.append((s, t))

    async def argv(limits, schema):
        seen_limits.append(limits)
        return [PY, str(fake), str(received)]
    prov._build_argv = argv
    return prov, logs, received, seen_limits


def test_review_is_compact_logged_and_uses_no_tools(tmp_path):
    prov, logs, received, seen = _provider(tmp_path)
    a = _attempt_with_sources(12, 5000, generated=2549)
    review = asyncio.run(prov.review(ReviewPackage(make_task(), a, 3, repo_path=str(tmp_path))))
    assert review.decision.value == "PASS"
    sent = int(received.read_text())
    assert sent <= 40000
    system = [t for s, t in logs if s == "SYSTEM"]
    i = system.index("Claude REVIEW context:")
    block = dict(line.split("=", 1) for line in system[i + 1:i + 11])
    assert int(block["final_prompt_bytes"]) == sent and block["prompt_limit_bytes"] == "40000"
    assert block["generated_paths_excluded"] == "2549" and block["authored_paths"] == "12"
    assert seen[0].tools == "" and seen[0].max_tool_calls == 0 and seen[0].max_turns == 2
    assert seen[0].model == "claude-opus-5-5" and seen[0].timeout == 60 and seen[0].max_budget_usd == 0.20


def test_oversized_review_never_launches_claude(tmp_path):
    prov, logs, received, seen = _provider(tmp_path)
    task = make_task()
    task.acceptance_criteria = ["z" * 1000] * 60
    with pytest.raises(ReviewContextTooLarge):
        asyncio.run(prov.review(ReviewPackage(task, _attempt_with_sources(1, 100), 3)))
    assert not received.exists() and not seen and not prov.calls  # nothing spawned, no call recorded


def test_final_gate_blocks_oversized_review_prompt_even_if_built_elsewhere(tmp_path):
    prov, _, received, seen = _provider(tmp_path)
    with pytest.raises(ReviewContextTooLarge, match="48291 > 40000"):
        asyncio.run(prov._invoke("REVIEW", "x" * 48291, {}, None))
    assert not received.exists() and not seen


def test_scheduler_pauses_with_clear_message_when_review_context_too_large(tmp_path, monkeypatch):
    from app.config.settings import Settings
    from app.orchestration.orchestrator import Engine
    from app.orchestration.providers.mock import MockOrchestrator
    from app.orchestration.state_machine import RunState

    async def too_large(self, package):
        raise ReviewContextTooLarge(48291, 40000)
    monkeypatch.setattr(MockOrchestrator, "review", too_large)
    repo = tmp_path / "proj"
    repo.mkdir()
    subprocess.run(["git", "init", "-q", "-b", "main"], cwd=repo, check=True)
    s = Settings()
    s.integrations.metrics_enabled = False
    s.integrations.codebase_memory_executable = ""
    s.orchestration.mock_step_delay = 0.01
    e = Engine(s, lambda k, p: None, str(tmp_path / "t.db"))

    async def main():
        await e.plan("g", str(repo))
        while not e._busy.done():
            await asyncio.sleep(0.01)
        await e.start()
        while not e._busy.done():
            await asyncio.sleep(0.01)

    asyncio.run(main())
    assert e.sm.state == RunState.PAUSED
    assert e.run.status_message == "Reviewer context exceeded hard limit: 48291 > 40000 bytes"
    assert e.run.resume_state == "REVIEWING"


def test_conftest_blocks_real_model_clis():
    with pytest.raises(AssertionError, match="test tried to launch a real model CLI"):
        asyncio.run(ProcessRunner().run(["claude", "--version"]))
    with pytest.raises(AssertionError, match="test tried to launch a real model CLI"):
        asyncio.run(ProcessRunner().run([r"C:\Users\Example\bin\ninfer\codex-ninfer.exe", "--help"]))
