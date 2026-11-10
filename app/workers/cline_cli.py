"""Coding worker: the Cline CLI in headless JSON mode (V5.1 worker runtime "cline").

Installed contract (Cline CLI 3.0.65, inspected on this machine - see README "V5.1"):
    <cline.exe> -y --json -c <cwd> -P <provider> -m <model> [--config <dir>] [extra args] "<HEADER>"
with the full prompt on stdin. With --json a prompt ARGUMENT is mandatory; piped stdin is appended to it
("<HEADER>\\n<prompt>"), which keeps long AI Studio prompts off the Windows command line (~32K limit) and out of
the recorded argv.
-y (yolo; documented in Cline's README, hidden from --help) is essential: it makes the CLI run the agent in its OWN
process (forceLocalBackend). Without it a one-shot run attaches to the shared background hub daemon and the session
executes inside that daemon (the user's own Cline process), where killing AI Studio's child would not stop it. Yolo
also auto-approves tools, keeps only read_files / run_commands / editor / submit_and_exit (no web fetch, no
ask_question, no sub-agents or teams) and ends the turn through submit_and_exit.
Output: NDJSON on stdout, one {"ts", "type", ...} object per line (agent_event / hook_event / run_result /
run_aborted ...); errors as {"type": "error", "message"} on stderr. Exit code 0 only for finishReason "completed".
Sessions: the stream carries no session id. `cline history --json` lists every session with the pid of the process
that ran it, so the id is looked up by THIS worker process's pid (an exact match, never a guess).
Resume: `--id <session>` forces the interactive TUI in this CLI version (the prompt is dropped), so a session cannot
be continued headlessly. session_resume_supported() is therefore False and the pipeline's existing, logged path
applies (a new session with AI Studio's compact handoff). ACP mode (--acp) can load sessions but authenticates only
Cline-account providers and may start an interactive OAuth login, so it is deliberately not used.
Cancellation / timeout: ProcessRunner's graceful-then-forced stop of THIS process tree (taskkill /PID <pid> /T /F);
never an image-name kill, so other Cline processes of the user are untouched.
Credentials stay in Cline's own configuration: AI Studio passes only -P / -m and the profile's process-local env.
Cline has no command sandbox and no structured todo list: see `capabilities`.
The worker's final message = its last assistant text plus the summary it passed to submit_and_exit.
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import tempfile
import time
from typing import Any

from app.config.discovery import ExecutableRejected, validate_cline_executable
from app.orchestration.providers.base import SmokeResult
from app.workers.base import (
    LogFn, PhaseTurnRequest, WorkerCapabilities, WorkerError, WorkerOutcome, WorkerProvider, WorkerTaskSpec,
    build_worker_prompt,
)
from app.workers.codex_ninfer import CONTEXT_EXHAUSTED_MARKERS
from app.workers.events import WorkerEvent, normalize_cline_event
from app.workers.process_runner import ProcessLaunchError, ProcessRunner, tail
from app.workers.profiles import WorkerLaunchSpec, redact_env

HEADER = "AI Studio worker task. The complete instructions follow below (from standard input)."
SMOKE_TOKEN = "MERHABA"
LOG = "CLINE"
NO_HEADLESS_RESUME = ("the installed Cline CLI cannot continue a session headlessly (--id opens the interactive "
                      "TUI), so each worker turn starts a new Cline session")
# Node/AI-SDK diagnostics Cline prints on stderr: logged once per run, stack frames dropped.
_STACK_FRAME = re.compile(r"^\s+at \S")
SUBMIT_TOOL = "submit_and_exit"


def _norm_dir(p: str) -> str:
    return os.path.normcase(os.path.normpath(p.replace("/", "\\"))) if p else ""


class ClineWorker(WorkerProvider):
    name = "cline"
    # Exact tool-call events (toolCallId); no todo list; commands are not sandboxed.
    capabilities = WorkerCapabilities(tool_call_count=True, subtask_progress=False, sandbox=False)

    def __init__(self, spec: WorkerLaunchSpec, log: LogFn | None = None):
        super().__init__(log)
        self.settings = spec
        self._runner: ProcessRunner | None = None
        self.last_assistant_text = ""  # the last assistant message of the last run (the smoke test checks it)
        self.last_submitted = ""  # what the last run passed to submit_and_exit

    @property
    def executable(self) -> str:
        try:
            return validate_cline_executable(self.settings.executable)
        except ExecutableRejected as exc:
            raise WorkerError(str(exc)) from exc

    def env_overrides(self) -> dict[str, str]:
        """Child-process-only environment (no CODEX_HOME: Cline has its own configuration). os.environ is never
        modified; a provider key named by the profile is inherited from the user's environment."""
        return dict(self.settings.extra_env)

    def _config_args(self) -> list[str]:
        return ["--config", self.settings.config_dir] if self.settings.config_dir else []

    def build_argv(self, cwd: str) -> list[str]:
        s = self.settings
        if not s.provider or not s.model:
            raise WorkerError("the Cline profile needs a provider (-P) and a model (-m)")
        return [self.executable, "-y", "--json", "-c", cwd, "-P", s.provider, "-m", s.model, *self._config_args(),
                *s.extra_args, HEADER]

    def history_argv(self) -> list[str]:
        return [self.executable, "history", "--json", "--limit", "30", *self._config_args()]

    async def session_resume_supported(self) -> bool:
        return False

    # ------------------------------------------------------------ turns

    async def execute_task(self, spec: WorkerTaskSpec) -> WorkerOutcome:
        return await self._run(build_worker_prompt(spec), spec.repo_path, timeout=self.settings.timeout_seconds)

    async def run_turn(self, request: PhaseTurnRequest) -> WorkerOutcome:
        if request.session_id:
            # The pipeline asks session_resume_supported() first, so this only guards against misuse: nothing is
            # started and the caller sees a failed resume (it never silently becomes something else here).
            self.log("SYSTEM", f"Cline session {request.session_id} not resumed: {NO_HEADLESS_RESUME}")
            return WorkerOutcome(executable=self.settings.executable, argv=[], exit_code=None, summary="",
                                 stdout_tail="", stderr_tail="", duration=0.0, error=NO_HEADLESS_RESUME,
                                 resume_failed=True, worker_profile_id=self.settings.profile_id,
                                 worker_profile_revision=self.settings.profile_revision)
        return await self._run(request.prompt, request.repo_path, timeout=self.settings.timeout_seconds)

    async def _lookup_session(self, pid: int | None, cwd: str, started: float) -> str:
        """The session this process runs, from `cline history --json` (exact pid match, same cwd)."""
        if not pid:
            return ""
        try:
            res = await ProcessRunner().run(self.history_argv(), env_overrides=self.env_overrides(), timeout=45)
            rows = json.loads(res.stdout or "[]")
        except Exception as exc:  # the session id is informational; a failed lookup never fails the turn
            self.log("SYSTEM", f"Cline session id lookup failed: {exc}")
            return ""
        want = _norm_dir(cwd)
        for row in rows if isinstance(rows, list) else []:
            if not isinstance(row, dict) or row.get("pid") != pid or row.get("source", "cli") != "cli":
                continue
            if row.get("cwd") and _norm_dir(str(row["cwd"])) != want:
                continue
            return str(row.get("sessionId") or "")
        return ""

    async def _run(self, prompt: str, cwd: str, timeout: float) -> WorkerOutcome:
        argv = self.build_argv(cwd)
        env = self.env_overrides()
        s = self.settings
        if s.profile_id:
            self.log("SYSTEM", f"Worker profile: {s.profile_name} ({s.profile_id}, revision {s.profile_revision}) - "
                               f"runtime Cline, provider {s.provider}, model {s.model}")
        self.log("SYSTEM", f"Worker executable: {argv[0]}")
        self.log("SYSTEM", f"Cline configuration: {s.config_dir or '(Cline default)'}; new session ({NO_HEADLESS_RESUME})")
        other = redact_env(env)
        if other:
            self.log("SYSTEM", "Worker extra env (process-local): " + ", ".join(f"{k}={v}" for k, v in other.items()))
        self.log("SYSTEM", f"Worker cwd: {cwd}")

        seen_tools: set[str] = set()
        tool_log: list[str] = []
        texts: list[str] = []
        errors: list[str] = []
        degraded: list[str] = []
        state: dict[str, Any] = {"session": "", "final": None, "finish": "", "usage": {}, "aborted": "",
                                 "lookup": None, "submitted": "", "noise": set()}
        started = time.time()
        runner = ProcessRunner()
        self._runner = runner

        def session_found(sid: str) -> None:
            if sid and not state["session"]:
                state["session"] = sid
                self.log(LOG, f"session {sid}")
                self._emit(WorkerEvent("session_started", session_id=sid))
                if self.on_session:
                    self.on_session(sid)  # persisted at once by the pipeline (traceability; not resumable)

        async def capture() -> None:
            await asyncio.sleep(0.5)  # the session row exists once the agent has started
            session_found(await self._lookup_session(runner.pid, cwd, started))

        def handle(ev: WorkerEvent) -> None:
            self._emit(ev)
            k = ev.kind
            if k == "tool_started":
                key = ev.tool_id or f"{ev.tool_name}#{len(seen_tools)}"
                if key in seen_tools:
                    return
                seen_tools.add(key)
                if ev.tool_name == SUBMIT_TOOL:
                    tool_input = ((ev.raw.get("event") or {}).get("input")) if isinstance(ev.raw, dict) else None
                    if isinstance(tool_input, dict) and isinstance(tool_input.get("summary"), str):
                        state["submitted"] = tool_input["summary"]
                desc = f"{ev.tool_name} {ev.text}".strip()
                tool_log.append(desc[:300])
                if self.on_tool_call:
                    self.on_tool_call(len(seen_tools))
                self.log(LOG, f"▶ {desc[:600]}")
                if self.on_file_change:
                    for kind, path in ev.files:
                        self.on_file_change(kind, path)
            elif k == "tool_finished":
                self.log(LOG, f"  {'failed: ' + ev.error if ev.error else 'done'} {tail(ev.text, 1200)}".rstrip())
            elif k == "command_output":
                pass  # progress chunks; the complete output arrives with tool_finished
            elif k == "assistant_text":
                texts.append(ev.text)
                self.log(LOG, ev.text)
            elif k == "reasoning":
                self.log(LOG, f"(thinking) {ev.text}")
            elif k == "warning":
                self.log(LOG, f"notice: {ev.text}")
            elif k == "error":
                errors.append(ev.text)
                self.log(LOG, f"error: {ev.text}")

        def on_stdout(line: str) -> None:
            text = line.strip()
            if not text:
                return
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                self.log(LOG, line)
                return
            if state["lookup"] is None and ev.get("type") in ("agent_event", "hook_event"):
                state["lookup"] = asyncio.ensure_future(capture())
            etype = ev.get("type")
            if etype == "agent_event":
                e = ev.get("event") or {}
                if e.get("type") == "usage":
                    state["usage"] = {k: v for k, v in e.items() if k != "type"}
                elif e.get("type") == "done":
                    state["finish"] = state["finish"] or str(e.get("reason") or "")
                    if state["final"] is None:
                        state["final"] = str(e.get("text") or "")
            elif etype == "run_result":
                state["finish"] = str(ev.get("finishReason") or "")
                state["final"] = str(ev.get("text") or "")
                if isinstance(ev.get("usage"), dict):
                    state["usage"] = ev["usage"]
                self.log(LOG, f"run finished: {state['finish']}; usage={json.dumps(state['usage'])}")
            elif etype == "run_aborted":
                state["aborted"] = str(ev.get("reason") or "aborted")
            elif etype == "hook_event":
                if ev.get("hookEventName") in ("agent_start", "agent_end"):
                    self.log(LOG, str(ev.get("hookEventName")).replace("_", " "))
                return
            for n in normalize_cline_event(ev):
                handle(n)

        def on_stderr(line: str) -> None:
            text = line.strip()
            if not text:
                return
            try:
                ev = json.loads(text)
            except json.JSONDecodeError:
                ev = None
            if isinstance(ev, dict) and ev.get("type"):
                for n in normalize_cline_event(ev):
                    handle(n)
                return
            if _STACK_FRAME.match(line):
                return
            if "AI SDK Warning" in text or text.startswith("DeprecationWarning"):
                if text in state["noise"]:
                    return
                state["noise"].add(text)
            low = text.lower()
            if "mcp" in low and ("failed" in low or "error" in low):
                note = "an MCP server failed to connect: " + text[-200:]
                if note not in degraded:
                    degraded.append(note)
            self.log(LOG, f"[stderr] {line}")

        try:
            res = await runner.run(argv, cwd=cwd, env_overrides=env, stdin_data=prompt, timeout=timeout,
                                   on_stdout=on_stdout, on_stderr=on_stderr)
        except ProcessLaunchError as exc:
            raise WorkerError(str(exc)) from exc
        finally:
            self._runner = None
        lookup = state["lookup"]
        if lookup is not None:
            try:
                await asyncio.wait_for(lookup, timeout=60)
            except (asyncio.TimeoutError, Exception):
                pass
        if not state["session"] and res.pid:
            session_found(await self._lookup_session(res.pid, cwd, started))
        # The worker's report: its last assistant message plus what it passed to submit_and_exit (yolo ends every
        # turn with that tool; run_result.text is then only the tool's acknowledgement).
        self.last_assistant_text = texts[-1].strip() if texts else ""
        self.last_submitted = state["submitted"].strip()
        parts = [x for x in (self.last_assistant_text, state["submitted"].strip()) if x]
        if len(parts) == 2 and parts[1] in parts[0]:
            parts = parts[:1]
        summary = "\n\n".join(parts) if parts else (state["final"] or "").strip()
        err = "; ".join(e for e in errors if e)
        if res.timed_out:
            err = (err + "; " if err else "") + f"worker timed out after {timeout}s"
        elif res.exit_code not in (0, None) and not res.cancelled:
            reason = state["finish"] or state["aborted"] or "no result"
            err = (err + "; " if err else "") + f"Cline exited with code {res.exit_code} ({reason})"
        failed = res.exit_code != 0 or bool(errors) or res.timed_out
        haystack = (err + "\n" + tail(res.stderr, 1500)).lower() if failed else ""
        return WorkerOutcome(
            executable=argv[0], argv=argv, exit_code=res.exit_code, summary=summary,
            stdout_tail=tail(res.stdout, 6000), stderr_tail=tail(res.stderr, 4000),
            duration=time.time() - started, tool_calls=len(seen_tools), tool_call_log=tool_log[-200:],
            usage=dict(state["usage"]), timed_out=res.timed_out, cancelled=res.cancelled or bool(state["aborted"]),
            error=err, degraded_notes=degraded, env_overrides=redact_env(env), session_id=state["session"],
            resumed=False, resume_failed=False,
            context_exhausted=any(m in haystack for m in CONTEXT_EXHAUSTED_MARKERS), todo=[],
            worker_profile_id=s.profile_id, worker_profile_revision=s.profile_revision)

    def _emit(self, ev: WorkerEvent) -> None:
        if self.on_event:
            try:
                self.on_event(ev)
            except Exception:
                pass

    async def cancel(self) -> None:
        if self._runner is not None:
            await self._runner.cancel()

    # ------------------------------------------------------------ checks

    async def version(self) -> str:
        res = await ProcessRunner().run([self.executable, "--version"], env_overrides=self.env_overrides(),
                                        timeout=60)
        if res.exit_code != 0:
            raise WorkerError(f"`cline --version` failed (exit {res.exit_code}): {tail(res.stderr, 300)}")
        return res.stdout.strip()

    async def smoke_test(self, metrics_url: str = "") -> SmokeResult:
        """Full test: one tiny prompt in a throwaway empty directory through exactly the launch a run uses (any file
        created there fails the test). May use paid API credits of the provider; it runs only when the user asks."""
        try:
            exe = self.executable
            ver = await self.version()
        except Exception as exc:
            return SmokeResult(False, self.settings.executable, str(exc))
        with tempfile.TemporaryDirectory(prefix="aio-cline-smoke-") as tmp:
            try:
                out = await self._run(f"Reply only with {SMOKE_TOKEN}.", tmp, timeout=300)
            except Exception as exc:
                return SmokeResult(False, exe, str(exc))
            created = sorted(os.listdir(tmp))
        # The model may answer as an assistant message, or only inside its submit_and_exit summary (yolo).
        reply = self.last_assistant_text or self.last_submitted or out.summary.strip()

        def first_word(text: str) -> str:
            words = re.findall(r"[A-Za-z0-9_]+", text or "")
            return words[0].upper() if words else ""
        normalized = SMOKE_TOKEN if SMOKE_TOKEN in (first_word(self.last_assistant_text),
                                                   first_word(self.last_submitted)) else reply
        problems = []
        if out.exit_code != 0:
            problems.append(f"exit code {out.exit_code} (expected 0)")
        if normalized != SMOKE_TOKEN:
            problems.append(f"response {reply[:200]!r} is not {SMOKE_TOKEN!r}")
        if created:
            problems.append(f"files were created in the smoke-test directory: {created}")
        if out.error:
            problems.append(out.error)
        ok = not problems
        extra = {"version": ver, "argv": out.argv, "env": out.env_overrides, "reply": reply[:500],
                 "exit_code": out.exit_code, "elapsed": round(out.duration, 1), "files_created": created,
                 "session_id": out.session_id, "provider": self.settings.provider, "model": self.settings.model,
                 "usage": out.usage}
        detail = (f"response {reply!r}, exit code 0, no files created, session {out.session_id or '(not found)'}"
                  if ok else "; ".join(problems))
        return SmokeResult(ok, exe, detail, model=self.settings.model, extra=extra)
