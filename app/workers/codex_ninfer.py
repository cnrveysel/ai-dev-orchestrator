"""Coding worker: the local Qwen model via the DEDICATED Codex NInfer executable.

Invocation:
    <codex-ninfer.exe> exec --json -C <repo> -o <last-message-file> -s <sandbox> [-m <model>] -
with the prompt on stdin.

Pipeline V2 continues a phase's session with
    <codex-ninfer.exe> exec resume --json -o <last-message-file> -c sandbox_mode="<sandbox>" [-m <model>] <session> -
(`exec resume` has no --cd/--sandbox: the process cwd is the repository and the sandbox is
set through the config override). The session itself is persisted by Codex under the
process-local CODEX_HOME, so no worker process has to stay alive between turns.

CODEX_HOME (and e.g. RUST_MIN_STACK) are set ONLY in the child environment.
The normal `codex` command / interactive launcher is never executed, and there
is no fallback of any kind if the configured executable is unusable.
"""
from __future__ import annotations

import json
import os
import tempfile
import time
from pathlib import Path
from typing import Any

from app.config.discovery import ExecutableRejected, validate_executable_for_policy
from app.config.settings import WorkerSettings
from app.orchestration.providers.base import SmokeResult
from app.workers.base import (
    LogFn, PhaseTurnRequest, WorkerError, WorkerOutcome, WorkerProvider, WorkerTaskSpec, build_worker_prompt,
)
from app.workers.events import normalize_codex_event
from app.workers.process_runner import ProcessLaunchError, ProcessRunner, tail
from app.workers.profiles import WorkerLaunchSpec, redact_env

REQUIRED_EXEC_FLAGS = ["--json", "--cd", "--output-last-message", "--sandbox"]
OPTIONAL_EXEC_FLAGS = ["--model", "--skip-git-repo-check", "--ephemeral"]

SMOKE_TOKEN = "QWEN_WORKER_OK"

TOOL_ITEM_TYPES = {"command_execution", "file_change", "mcp_tool_call", "web_search"}

# `exec resume` flags needed for session reuse (V2); without them every turn is a new session.
REQUIRED_RESUME_FLAGS = ["--json", "--output-last-message"]
CONTEXT_EXHAUSTED_MARKERS = ("context window", "context_length_exceeded", "maximum context length",
                             "context length exceeded", "exceeds the context", "ran out of room")


class CodexNInferWorker(WorkerProvider):
    """The Codex runtime adapter. `settings` is either the legacy WorkerSettings (V4) or a V5 WorkerLaunchSpec
    resolved from a run's worker-profile snapshot; both expose the same launch attributes, so the argv, the
    process-local environment and the session handling are identical for the legacy NInfer worker."""
    name = "codex-ninfer"

    def __init__(self, settings: WorkerSettings | WorkerLaunchSpec, log: LogFn | None = None):
        super().__init__(log)
        self.settings = settings
        self._runner: ProcessRunner | None = None
        self._flags: set[str] | None = None
        self._resume_flags: set[str] | None = None

    @property
    def executable(self) -> str:
        try:
            return validate_executable_for_policy(self.settings.executable,
                                                  getattr(self.settings, "executable_policy", "ninfer"))
        except ExecutableRejected as exc:
            raise WorkerError(str(exc)) from exc

    def env_overrides(self) -> dict[str, str]:
        """Child-process-only environment. os.environ is never modified."""
        env = dict(self.settings.extra_env)
        if self.settings.codex_home:
            env["CODEX_HOME"] = os.path.expandvars(self.settings.codex_home)
        return env

    def _check_codex_home(self) -> None:
        home = self.env_overrides().get("CODEX_HOME")
        if not home:
            return
        if not Path(home).is_dir():
            raise WorkerError(f"CODEX_HOME for the worker does not exist: {home}")
        normal = Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".codex"
        if os.path.normcase(os.path.abspath(home)) == os.path.normcase(os.path.abspath(normal)):
            raise WorkerError("worker CODEX_HOME points at the normal ~/.codex profile; "
                              "use the dedicated NInfer profile (e.g. ~/.codex-ninfer)")

    async def detect_capabilities(self) -> dict[str, Any]:
        exe = self.executable
        env = self.env_overrides()
        top = await ProcessRunner().run([exe, "--help"], env_overrides=env, timeout=60)
        ex = await ProcessRunner().run([exe, "exec", "--help"], env_overrides=env, timeout=60)
        ver = await ProcessRunner().run([exe, "--version"], env_overrides=env, timeout=60)
        text = ex.stdout + ex.stderr
        flags = {f for f in REQUIRED_EXEC_FLAGS + OPTIONAL_EXEC_FLAGS if f in text}
        has_exec = "exec" in (top.stdout + top.stderr)
        self._flags = flags
        missing = [f for f in REQUIRED_EXEC_FLAGS if f not in flags]
        if not has_exec:
            missing.insert(0, "exec subcommand")
        return {"version": ver.stdout.strip(), "flags": sorted(flags), "missing": missing}

    async def _ensure_capabilities(self) -> set[str]:
        if self._flags is None:
            caps = await self.detect_capabilities()
            if caps["missing"]:
                raise WorkerError("codex-ninfer lacks required exec flags: " + ", ".join(caps["missing"]))
        assert self._flags is not None
        return self._flags

    async def _build_argv(self, repo: str, last_msg: str, sandbox: str,
                          ephemeral: bool = False, skip_git: bool = False) -> list[str]:
        flags = await self._ensure_capabilities()
        argv = [self.executable, "exec", "--json", "--cd", repo, "--output-last-message", last_msg,
                "--sandbox", sandbox]
        if self.settings.pass_model_flag and self.settings.model and "--model" in flags:
            argv += ["--model", self.settings.model]
        if ephemeral and "--ephemeral" in flags:
            argv.append("--ephemeral")
        if skip_git and "--skip-git-repo-check" in flags:
            argv.append("--skip-git-repo-check")
        argv += list(self.settings.extra_args)
        argv.append("-")  # prompt from stdin
        return argv

    async def session_resume_supported(self) -> bool:
        if self._resume_flags is None:
            try:
                r = await ProcessRunner().run([self.executable, "exec", "resume", "--help"],
                                              env_overrides=self.env_overrides(), timeout=60)
            except Exception as exc:  # not cached: checked again at the next turn
                self.log("SYSTEM", f"codex-ninfer `exec resume` capability check failed: {exc}")
                return False
            text = r.stdout + r.stderr
            self._resume_flags = ({f for f in REQUIRED_RESUME_FLAGS + ["--model"] if f in text}
                                  if r.exit_code == 0 else set())
            if not all(f in self._resume_flags for f in REQUIRED_RESUME_FLAGS):
                self.log("SYSTEM", "codex-ninfer does not support `exec resume --json -o`: "
                                   "V2 turns will use new sessions with a handoff")
        return all(f in self._resume_flags for f in REQUIRED_RESUME_FLAGS)

    def session_exists(self, session_id: str) -> bool:
        """Codex keeps sessions as CODEX_HOME/sessions/YYYY/MM/DD/rollout-<time>-<id>.jsonl. Only a
        definite miss returns False (unknown layout -> assume it exists)."""
        home = self.env_overrides().get("CODEX_HOME")
        root = Path(home) / "sessions" if home else None
        if not session_id or root is None or not root.is_dir():
            return bool(session_id)
        return any(root.rglob(f"rollout-*{session_id}.jsonl"))

    async def _build_resume_argv(self, session_id: str, last_msg: str, sandbox: str) -> list[str]:
        await self._ensure_capabilities()
        flags = self._resume_flags or set()
        argv = [self.executable, "exec", "resume", "--json", "--output-last-message", last_msg,
                "-c", f'sandbox_mode="{sandbox}"']
        if self.settings.pass_model_flag and self.settings.model and "--model" in flags:
            argv += ["--model", self.settings.model]
        argv += list(self.settings.extra_args)
        argv += [session_id, "-"]  # prompt from stdin
        return argv

    async def execute_task(self, spec: WorkerTaskSpec) -> WorkerOutcome:
        return await self._run(build_worker_prompt(spec), spec.repo_path, self.settings.sandbox,
                               timeout=self.settings.timeout_seconds)

    async def run_turn(self, request: PhaseTurnRequest) -> WorkerOutcome:
        session = request.session_id if request.session_id and await self.session_resume_supported() else ""
        if session and not self.session_exists(session):
            # Never let `exec resume` run without the session: the short turn prompt assumes its context.
            self.log("SYSTEM", f"Qwen session {session} not found under CODEX_HOME/sessions; not resuming")
            return WorkerOutcome(executable=self.settings.executable, argv=[], exit_code=None, summary="", stdout_tail="",
                                 stderr_tail="", duration=0.0, error=f"session {session} not found",
                                 resume_failed=True)
        out = await self._run(request.prompt, request.repo_path, self.settings.sandbox,
                              timeout=self.settings.timeout_seconds, resume_session=session)
        if session and out.session_id and out.session_id != session:
            self.log("SYSTEM", f"WARNING: resuming {session} reported thread {out.session_id}; continuing with it")
        return out

    async def _run(self, prompt: str, cwd: str, sandbox: str, timeout: float,
                   ephemeral: bool = False, skip_git: bool = False, resume_session: str = "") -> WorkerOutcome:
        self._check_codex_home()
        fd, last_msg = tempfile.mkstemp(prefix="aio-worker-last-", suffix=".txt")
        os.close(fd)
        if resume_session:
            argv = await self._build_resume_argv(resume_session, last_msg, sandbox)
        else:
            argv = await self._build_argv(cwd, last_msg, sandbox, ephemeral, skip_git)
        env = self.env_overrides()
        profile_id = getattr(self.settings, "profile_id", "")
        if profile_id:  # V5: which worker profile this process belongs to (sanitized; no secret values)
            s = self.settings
            self.log("SYSTEM", f"Worker profile: {s.profile_name} ({profile_id}, revision {s.profile_revision}) - "
                               f"runtime Codex, provider {s.provider or '(config.toml)'}, model "
                               f"{s.model if s.model and s.pass_model_flag else '(config.toml)'}")
        self.log("SYSTEM", f"Worker executable: {argv[0]}")
        if resume_session:
            self.log("SYSTEM", f"Worker session: resuming Qwen session {resume_session} (context kept)")
        self.log("SYSTEM", f"Worker CODEX_HOME (process-local): {env.get('CODEX_HOME', '(not set)')}")
        other = redact_env({k: v for k, v in env.items() if k != "CODEX_HOME"})
        if other:
            self.log("SYSTEM", "Worker extra env (process-local): " + ", ".join(f"{k}={v}" for k, v in other.items()))
        self.log("SYSTEM", f"Worker cwd: {cwd}")

        tool_calls = 0
        tool_log: list[str] = []
        seen_items: set[str] = set()
        usage: dict[str, Any] = {}
        agent_messages: list[str] = []
        errors: list[str] = []
        degraded: list[str] = []
        session_id = ""
        todo: list[dict[str, Any]] = []

        def on_stdout(line: str) -> None:
            nonlocal tool_calls, usage, session_id, todo
            s = line.strip()
            if not s:
                return
            try:
                ev = json.loads(s)
            except json.JSONDecodeError:
                self.log("QWEN", line)
                return
            if self.on_event:  # V5.1 normalized events (diagnostics only; logging below is unchanged)
                for norm in normalize_codex_event(ev):
                    try:
                        self.on_event(norm)
                    except Exception:
                        pass
            etype = ev.get("type", "")
            item = ev.get("item") or {}
            itype = item.get("type", "")
            if etype in ("item.started", "item.completed") and itype in TOOL_ITEM_TYPES:
                iid = str(item.get("id", ""))
                if iid not in seen_items:  # started/completed pairs count once
                    seen_items.add(iid)
                    tool_calls += 1
                    desc = _describe_item(item)
                    tool_log.append(desc)
                    if self.on_tool_call:
                        self.on_tool_call(tool_calls)
                    self.log("QWEN", f"▶ {desc}")
                if etype == "item.completed":
                    self._log_completed_tool(item)
                    if itype == "file_change" and self.on_file_change:
                        for ch in item.get("changes") or []:
                            if isinstance(ch, dict) and ch.get("path"):
                                self.on_file_change(str(ch.get("kind", "")), str(ch["path"]))
            elif etype == "item.completed" and itype == "agent_message":
                agent_messages.append(item.get("text", ""))
                self.log("QWEN", item.get("text", ""))
            elif etype == "item.completed" and itype == "reasoning":
                txt = item.get("text", "")
                if txt:
                    self.log("QWEN", f"(thinking) {txt}")
            elif etype == "item.completed" and itype == "error":
                errors.append(item.get("message", ""))
                self.log("QWEN", f"error: {item.get('message', '')}")
            elif etype == "turn.completed":
                usage = ev.get("usage") or {}
                self.log("QWEN", f"turn completed; usage={json.dumps(usage)}")
            elif etype in ("turn.failed", "error"):
                msg = (ev.get("error") or {}).get("message") if etype == "turn.failed" else ev.get("message")
                errors.append(str(msg))
                self.log("QWEN", f"{etype}: {msg}")
            elif etype == "thread.started":
                session_id = str(ev.get("thread_id", "") or "")
                self.log("QWEN", f"thread started {session_id}")
                if session_id and self.on_session:
                    self.on_session(session_id)  # persisted at once: a killed process stays resumable
            elif etype == "turn.started":
                self.log("QWEN", "turn started")
            elif itype == "todo_list":
                steps = item.get("items") or []
                self.log("QWEN", "todo: " + "; ".join(
                    f"[{'x' if st.get('completed') else ' '}] {st.get('text', '')}" for st in steps))
                todo = [{"text": str(st.get("text", "")), "completed": bool(st.get("completed"))}
                        for st in steps if isinstance(st, dict)]
                if self.on_todo:
                    self.on_todo(todo)
            elif etype in ("item.started", "item.updated") and itype in ("agent_message", "reasoning"):
                pass  # the text is logged once, on item.completed
            else:
                self.log("QWEN", f"event {s[:400]}")  # never silently drop worker output

        def on_stderr(line: str) -> None:
            if not line.strip():
                return
            low = line.lower()
            if ("rmcp" in low or "mcp" in low) and ("failed" in low or "error" in low):
                note = "an MCP server failed to connect: " + line.strip()[-200:]
                if note not in degraded:
                    degraded.append(note)
            self.log("QWEN", f"[stderr] {line}")

        self._runner = ProcessRunner()
        start = time.time()
        try:
            res = await self._runner.run(argv, cwd=cwd, env_overrides=env, stdin_data=prompt,
                                         timeout=timeout, on_stdout=on_stdout, on_stderr=on_stderr)
        except ProcessLaunchError as exc:
            raise WorkerError(str(exc)) from exc
        finally:
            self._runner = None
        try:
            summary = Path(last_msg).read_text(encoding="utf-8", errors="replace").strip()
        except OSError:
            summary = ""
        finally:
            try:
                os.unlink(last_msg)
            except OSError:
                pass
        if not summary and agent_messages:
            summary = agent_messages[-1]
        err = "; ".join(e for e in errors if e)
        if res.timed_out:
            err = (err + "; " if err else "") + f"worker timed out after {timeout}s"
        # Only a FAILED turn can mean "out of context" (ordinary log lines may mention the context window).
        failed = res.exit_code != 0 or bool(errors) or res.timed_out
        haystack = (err + "\n" + tail(res.stderr, 1500)).lower() if failed else ""
        context_exhausted = any(m in haystack for m in CONTEXT_EXHAUSTED_MARKERS)
        # A resume that never started a thread, did nothing and failed: the session is unusable.
        resume_failed = bool(resume_session) and not session_id and tool_calls == 0 and res.exit_code != 0 \
            and not res.cancelled
        if resume_failed:
            err = (err + "; " if err else "") + f"could not resume session {resume_session}: {tail(res.stderr, 300)}"
        return WorkerOutcome(
            executable=argv[0], argv=argv, exit_code=res.exit_code, summary=summary,
            stdout_tail=tail(res.stdout, 6000), stderr_tail=tail(res.stderr, 4000),
            duration=time.time() - start, tool_calls=tool_calls, tool_call_log=tool_log,
            usage=usage, timed_out=res.timed_out, cancelled=res.cancelled, error=err,
            degraded_notes=degraded, env_overrides=redact_env(env),  # persisted with the call: never a secret
            session_id="" if resume_failed else (session_id or resume_session),
            resumed=bool(resume_session) and not resume_failed, resume_failed=resume_failed,
            context_exhausted=context_exhausted, todo=todo,
            worker_profile_id=profile_id, worker_profile_revision=getattr(self.settings, "profile_revision", ""),
        )

    def _log_completed_tool(self, item: dict[str, Any]) -> None:
        itype = item.get("type")
        if itype == "command_execution":
            out = item.get("aggregated_output") or ""
            self.log("QWEN", f"  exit={item.get('exit_code')} {tail(out, 1500)}".rstrip())
        elif itype == "mcp_tool_call":
            self.log("QWEN", f"  {item.get('status', 'done')}")

    async def cancel(self) -> None:
        if self._runner is not None:
            await self._runner.cancel()

    async def smoke_test(self, metrics_url: str = "") -> SmokeResult:
        """Harmless prompt through the dedicated executable, run in a throwaway
        empty temp directory with a read-only sandbox. Passes only if the reply
        is exactly SMOKE_TOKEN, the exit code is 0 and no file was created."""
        from app.config.discovery import normal_codex_paths
        from app.integrations.ninfer_metrics import NInferMetricsClient
        try:
            exe = self.executable
            self._check_codex_home()
            caps = await self.detect_capabilities()
        except Exception as exc:
            return SmokeResult(False, self.settings.executable, str(exc))
        if caps["missing"]:
            return SmokeResult(False, exe, "missing: " + ", ".join(caps["missing"]))
        if os.path.normcase(os.path.abspath(exe)) in normal_codex_paths():
            return SmokeResult(False, exe, "refused: this is the normal codex launcher")
        before = after = None
        metrics = NInferMetricsClient(metrics_url) if metrics_url else None
        if metrics:
            s = await metrics.sample()
            before = s.tokens_total if s.ok else None
        with tempfile.TemporaryDirectory(prefix="aio-smoke-") as tmp:
            try:
                out = await self._run(f"Reply with exactly: {SMOKE_TOKEN}", tmp, "read-only", timeout=300,
                                      ephemeral=True, skip_git=True)
            except Exception as exc:
                return SmokeResult(False, exe, str(exc))
            created = sorted(os.listdir(tmp))
        if metrics:
            s = await metrics.sample()
            after = s.tokens_total if s.ok else None
        reply = out.summary.strip()
        problems = []
        if out.exit_code != 0:
            problems.append(f"exit code {out.exit_code} (expected 0)")
        if reply != SMOKE_TOKEN:
            problems.append(f"response {reply[:200]!r} is not exactly {SMOKE_TOKEN!r}")
        if created:
            problems.append(f"files were created in the smoke-test directory: {created}")
        if out.error:
            problems.append(out.error)
        ok = not problems
        extra = {"version": caps["version"], "argv": out.argv, "env": out.env_overrides,
                 "codex_home": out.env_overrides.get("CODEX_HOME", ""), "reply": reply[:500],
                 "exit_code": out.exit_code, "elapsed": round(out.duration, 1), "files_created": created,
                 "normal_codex_paths_rejected": sorted(normal_codex_paths())}
        if before is not None and after is not None:
            extra["ninfer_tokens_predicted_delta"] = after - before
        detail = (f"response {reply!r}, exit code 0, no files created" if ok else "; ".join(problems))
        return SmokeResult(ok, exe, detail, model=self.settings.model, extra=extra)


def _describe_item(item: dict[str, Any]) -> str:
    itype = item.get("type")
    if itype == "command_execution":
        return f"$ {item.get('command', '')}"
    if itype == "file_change":
        changes = item.get("changes") or []
        return "edit " + ", ".join(f"{c.get('kind', '?')}:{c.get('path', '?')}" for c in changes[:8])
    if itype == "mcp_tool_call":
        args = json.dumps(item.get("arguments", {}), ensure_ascii=False)
        return f"mcp {item.get('server', '?')}.{item.get('tool', '?')} {args[:200]}"
    if itype == "web_search":
        return f"web_search {item.get('query', '')}"
    return str(itype)
