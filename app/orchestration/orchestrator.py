"""Engine: owns the run lifecycle (PLAN / START / PAUSE / RESUME / STOP),
provider construction, persistence and status probes.

It runs on a dedicated background thread with its own asyncio loop
(EngineHost). The GUI talks to it only through EngineHost.submit() and
receives events through the emit callback - no CLI logic lives in widgets.
"""
from __future__ import annotations

import asyncio
import copy
import os
import threading
import time
from concurrent.futures import Future
from typing import Any, Callable, Coroutine

from app.config.discovery import validate_claude_executable
from app.config.settings import Settings, app_data_dir
from app.evidence.artifacts import CleanupReport, RetentionPolicy
from app.evidence.artifacts import cleanup as cleanup_artifacts
from app.evidence.registry import CapabilityRegistry, default_registry
from app.integrations import codebase_memory, unreal_mcp
from app.integrations.git import GitRepo
from app.integrations.ninfer_metrics import NInferMetricsClient, probe_models
from app.models import human as hm
from app.models.run import Run
from app.models.task import TaskStatus
from app.orchestration.providers.base import OrchestratorError, OrchestratorProvider, PlanRequest, SmokeResult
from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
from app.orchestration.phase_scheduler import PhaseScheduler
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.scheduler import Scheduler, SchedulerDeps, run_payload
from app.orchestration.state_machine import EXECUTION_STATES, HALTED_STATES, RunState, StateMachine
from app.orchestration.task_graph import TaskGraph
from app.orchestration.verifier import Verifier
from app.persistence.database import Database
from app.workers.base import WorkerProvider
from app.workers.mock import MockWorker
from app.workers.process_runner import ProcessRunner
from app.workers.launcher import UNAVAILABLE, build_launch_spec, env_problems, launch_worker, snapshot_problems
from app.workers.profile_registry import WorkerProfileRegistry
from app.workers.profiles import RunWorkerProfiles, WorkerLaunchSpec, WorkerSnapshot

S = RunState
EmitFn = Callable[[str, dict[str, Any]], None]
PIPELINES = ("v1", "v2")


class EngineError(RuntimeError):
    pass


class Engine:
    def __init__(self, settings: Settings, emit: EmitFn, db_path: str | None = None):
        self.settings = copy.deepcopy(settings)
        self._emit = emit
        self.db = Database(db_path or (app_data_dir() / "orchestrator.db"))
        self.run: Run | None = None
        self.sm = StateMachine(S.IDLE, self._on_transition)
        self.scheduler: Scheduler | None = None
        self._busy: asyncio.Task | None = None
        self._planner: OrchestratorProvider | None = None
        self._metrics_task: asyncio.Task | None = None
        self._commit_task: asyncio.Task | None = None
        self.status: dict[str, Any] = {}
        interrupted = self.db.mark_interrupted_runs()
        self._startup_notes = [f"run #{i} was interrupted by application exit; it can be resumed" for i in interrupted]

    # ------------------------------------------------------------ plumbing

    def emit(self, kind: str, payload: dict[str, Any]) -> None:
        try:
            self._emit(kind, payload)
        except Exception:
            pass

    def log(self, source: str, text: str) -> None:
        ts = time.time()
        run_id = self.run.id if self.run else None
        for line in (text.splitlines() or [""]):
            self.db.add_log(run_id, source, line, ts)
            self.emit("log", {"source": source, "text": line, "ts": ts, "run_id": run_id})

    def _on_transition(self, old: RunState, new: RunState, reason: str) -> None:
        run_id = self.run.id if self.run else None
        self.db.add_transition(run_id, old.value, new.value, reason)
        self.log("SYSTEM", f"state {old.value} -> {new.value}" + (f" ({reason})" if reason else ""))
        self.emit("state", {"state": new.value, "previous": old.value, "reason": reason, "run_id": run_id})

    async def start_background(self) -> None:
        for note in self._startup_notes:
            self.log("SYSTEM", note)
        self._commit_task = asyncio.ensure_future(self._commit_loop())
        self._restart_metrics()
        self.emit("runs", {"runs": self.db.list_runs()})
        await self.cleanup_artifacts()

    @property
    def artifact_root(self) -> str:
        return str(self.db.path.parent / "evidence")

    async def cleanup_artifacts(self) -> CleanupReport | None:
        """V4 pass 3 retention of evidence artifacts (only under AI Studio's evidence folder). Artifacts of runs
        that can still continue (anything not COMPLETED) and of the loaded run are never deleted."""
        o = self.settings.orchestration
        if not getattr(o, "evidence_artifact_cleanup_enabled", True):
            return None
        root = self.artifact_root
        if not os.path.isdir(root):
            return None
        protected = {int(r["id"]) for r in self.db.list_runs(limit=100000) if r.get("state") != "COMPLETED"}
        if self.run is not None and self.run.id is not None:
            protected.add(int(self.run.id))
        try:
            rep = await asyncio.to_thread(cleanup_artifacts, root, protected_runs=protected,
                                          policy=RetentionPolicy.from_settings(o))
        except Exception as exc:  # retention must never break the engine
            self.log("SYSTEM", f"evidence artifact cleanup failed: {exc}")
            return None
        if rep.deleted or rep.errors:
            self.log("SYSTEM", rep.summary())
        return rep

    async def shutdown(self) -> None:
        if self.scheduler and self._busy and not self._busy.done():
            self.scheduler.request_stop()
            try:
                await asyncio.wait_for(asyncio.shield(self._busy), timeout=30)
            except Exception:
                pass
        for t in (self._metrics_task, self._commit_task):
            if t:
                t.cancel()
        # Anything else still in flight (planning, smoke tests, status probes) is cancelled so its
        # child process tree is killed; otherwise e.g. a `claude --print` planner is orphaned.
        others = [t for t in asyncio.all_tasks() if t is not asyncio.current_task() and not t.done()]
        for t in others:
            t.cancel()
        if others:
            await asyncio.wait(others, timeout=20)
        self.db.commit()
        self.db.close()

    async def _commit_loop(self) -> None:
        while True:
            await asyncio.sleep(1.0)
            try:
                self.db.commit()
            except Exception:
                pass

    def _restart_metrics(self) -> None:
        if self._metrics_task:
            self._metrics_task.cancel()
        integ = self.settings.integrations
        if integ.metrics_enabled and integ.metrics_url:
            self._metrics_task = asyncio.ensure_future(self._metrics_loop(integ.metrics_url,
                                                                          max(0.5, integ.metrics_interval_seconds)))

    async def _metrics_loop(self, url: str, interval: float) -> None:
        client = NInferMetricsClient(url)
        while True:
            s = await client.sample()
            self.emit("metrics", {"ok": s.ok, "tokens_per_second": s.tokens_per_second,
                                  "tokens_total": s.tokens_total, "error": s.error, "at": s.at})
            await asyncio.sleep(interval)

    def _busy_check(self) -> None:
        if self._busy is not None and not self._busy.done():
            raise EngineError("an operation is already in progress")

    # ------------------------------------------------------------ settings / status

    async def apply_settings(self, settings: Settings) -> None:
        self.settings = copy.deepcopy(settings)
        self._restart_metrics()
        self.log("SYSTEM", "settings applied")

    async def refresh_status(self, repo_path: str | None) -> dict[str, Any]:
        st: dict[str, Any] = {}
        s = self.settings
        # Claude
        try:
            exe = validate_claude_executable(s.claude.executable)
            v = await ProcessRunner().run([exe, "--version"], timeout=30)
            st["claude"] = {"ok": v.exit_code == 0, "path": exe, "version": v.stdout.strip(),
                            "model": s.claude.model or "CLI default",
                            "verified": s.claude.verified_executable == exe}
        except Exception as exc:
            st["claude"] = {"ok": False, "error": str(exc)}
        # Worker: the default profile (what a NEW run would lock; a V4 installation = the legacy worker settings)
        registry = WorkerProfileRegistry(s)
        profile = registry.default_profile() or registry.legacy()
        try:
            w = launch_worker(build_launch_spec(WorkerSnapshot.of(profile), s.worker, profile))
            exe = w.executable
            v = await ProcessRunner().run([exe, "--version"], env_overrides=w.env_overrides(), timeout=30)
            st["worker"] = {"ok": v.exit_code == 0, "path": exe, "version": v.stdout.strip(),
                            "codex_home": w.env_overrides().get("CODEX_HOME", ""),
                            "verified": profile.is_verified(), "profile": profile.name, "profile_id": profile.id,
                            "runtime": profile.runtime}
        except Exception as exc:
            st["worker"] = {"ok": False, "error": str(exc), "profile": profile.name, "profile_id": profile.id,
                            "runtime": profile.runtime}
        # NInfer server / model
        try:
            models = await asyncio.to_thread(probe_models, s.worker.ninfer_base_url)
            st["ninfer"] = {"ok": True, "url": s.worker.ninfer_base_url, "models": models,
                            "model_loaded": s.worker.model in models}
        except Exception as exc:
            st["ninfer"] = {"ok": False, "url": s.worker.ninfer_base_url, "error": str(exc)}
        # Integrations
        cm = await codebase_memory.probe(s.integrations.codebase_memory_executable, repo_path)
        st["codebase_memory"] = {"ok": cm.available, "text": cm.describe(), "indexed": cm.repo_indexed}
        ur = await unreal_mcp.probe(repo_path, s.integrations.unreal_mcp_url)
        st["unreal"] = {"is_unreal": ur.is_unreal_project, "ok": bool(ur.mcp_reachable), "text": ur.describe()}
        # Repo
        if repo_path:
            info = await GitRepo(repo_path).info()
            st["repo"] = {"is_repo": info.is_repo, "branch": info.branch, "head": info.head,
                          "dirty": len(info.dirty), "text": info.describe()}
        self.status = st
        self.emit("status", st)
        return st

    # ------------------------------------------------------------ smoke tests

    async def smoke_test_claude(self) -> dict[str, Any]:
        self._busy_check()
        self.log("SYSTEM", "=== Claude Code CLI smoke test ===")
        prov = ClaudeCodeCLIProvider(self.settings.claude, self.log)
        res = await prov.smoke_test()
        for c in prov.calls:
            self.db.add_orchestrator_call(None, None, c)
        self.log("SYSTEM", f"Claude smoke test {'PASSED' if res.ok else 'FAILED'}: executable={res.executable} "
                           f"model={res.model or 'CLI default'} version={res.extra.get('version', '?')} - {res.detail}")
        if res.ok:
            self.settings.claude.verified_executable = res.executable
            self.settings.claude.verified_at = time.time()
        out = {"target": "claude", "ok": res.ok, "executable": res.executable, "detail": res.detail,
               "extra": res.extra, "verified_at": self.settings.claude.verified_at}
        self.emit("smoke", out)
        return out

    async def smoke_test_worker(self, profile_id: str | None = None) -> dict[str, Any]:
        """Full test of one worker profile (default: the default profile - the legacy worker on a V4 installation):
        a harmless prompt through exactly the launch path a run would use, in a throwaway read-only directory. Only
        run on request, never by a status refresh (it may use paid API credits of the profile's provider). The
        result (bound to the executable + the runtime's launch target: Codex CODEX_HOME, Cline provider/model/
        configuration) gates REAL runs of that profile; a failure revokes it."""
        self._busy_check()
        registry = WorkerProfileRegistry(self.settings)
        profile = registry.get(profile_id or registry.default_id())
        if profile is None:
            raise EngineError(f"unknown worker profile: {profile_id or registry.default_id()}")
        self.log("SYSTEM", f"=== Worker smoke test: {profile.name} ({profile.id}) ===")
        snap = WorkerSnapshot.of(profile)
        for line in snap.describe():
            self.log("SYSTEM", f"  {line}")
        missing = env_problems(profile.required_env)
        if missing:
            res = SmokeResult(False, profile.executable_path, missing[0])
        else:
            w = launch_worker(build_launch_spec(snap, self.settings.worker, profile), self.log)
            # The NInfer metrics delta only proves something for a worker that talks to the NInfer server.
            ninfer = profile.is_legacy or "ninfer" in profile.provider.lower()
            integ = self.settings.integrations
            res = await w.smoke_test(integ.metrics_url if integ.metrics_enabled and ninfer else "")
        ex = res.extra
        self.log("SYSTEM", f"Worker smoke test {'PASSED' if res.ok else 'FAILED'}")
        self.log("SYSTEM", f"  Worker executable: {res.executable}")
        if "session_id" in ex:  # runtimes without CODEX_HOME (Cline) report their session instead
            self.log("SYSTEM", f"  Session: {ex.get('session_id') or '(not found)'}   provider {ex.get('provider')} "
                               f"model {ex.get('model')}")
        else:
            self.log("SYSTEM", f"  CODEX_HOME (process-local): {ex.get('codex_home', '(not set)')}")
        if ex:
            self.log("SYSTEM", f"  Response: {ex.get('reply')!r}")
            self.log("SYSTEM", f"  Exit code: {ex.get('exit_code')}   Elapsed: {ex.get('elapsed')}s   "
                               f"Files created: {ex.get('files_created') or 'none'}")
            self.log("SYSTEM", f"  version: {ex.get('version')}  |  argv: {ex.get('argv')}")
            if "normal_codex_paths_rejected" in ex:
                self.log("SYSTEM", f"  normal codex paths refused: {ex.get('normal_codex_paths_rejected')}")
            if "ninfer_tokens_predicted_delta" in ex:
                self.log("SYSTEM", f"  NInfer server tokens_predicted delta during test: "
                                   f"{ex['ninfer_tokens_predicted_delta']:+.0f} (proves the request hit NInfer)")
        self.log("SYSTEM", f"  {res.detail}")
        # a failing smoke test revokes REAL worker mode for this profile
        registry.record_verification(profile.id, res.ok, res.executable, profile.launch_target)
        p = registry.require(profile.id)
        out = {"target": "worker", "ok": res.ok, "executable": res.executable, "detail": res.detail,
               "extra": ex, "verified_at": p.verified_at, "verified_executable": p.verified_executable,
               "verified_codex_home": p.verified_codex_home, "verified_target": p.verified_codex_home,
               "profile_id": p.id, "profile_name": p.name, "runtime": p.runtime}
        self.emit("smoke", out)
        return out

    # ------------------------------------------------------------ providers

    def _make_orchestrator(self, mode: str) -> OrchestratorProvider:
        if mode == "claude":
            exe = validate_claude_executable(self.settings.claude.executable)
            if self.settings.claude.verified_executable != exe:
                raise EngineError("Claude Code adapter is not verified for this executable. "
                                  "Run the Claude smoke test first (Settings > Claude > Smoke test).")
            return ClaudeCodeCLIProvider(self.settings.claude, self.log)
        o = self.settings.orchestration
        return MockOrchestrator(self.log, delay=o.mock_step_delay, retry_task_ids=o.mock_retry_task_ids)

    def _make_worker(self, mode: str) -> WorkerProvider:
        if mode == "ninfer":
            worker = launch_worker(self._run_launch_spec(), self.log)
            caps = worker.capabilities
            if not caps.subtask_progress:
                self.log("SYSTEM", "Worker capabilities: no structured subtask progress - the implementation guard's "
                                   "no-progress limit is not applied (wall-clock and tool-call limits are)")
            if not caps.tool_call_count:
                self.log("SYSTEM", "Worker capabilities: no exact tool-call count - tool budgets are not applied")
            if not caps.sandbox:
                self.log("SYSTEM", "Worker capabilities: NO command sandbox - the worker's commands are not confined "
                                   "to the repository")
            return worker
        return MockWorker(self.log, delay=self.settings.orchestration.mock_step_delay)

    # ------------------------------------------------------------ V5 worker profiles

    def _run_launch_spec(self) -> WorkerLaunchSpec:
        """The worker launch of the loaded run, always from the run's own worker-profile snapshot: locked (and
        committed) before the run's first worker launch, reused unchanged by every resume - whatever the default
        profile is today. An unusable snapshot stops here (WORKER_PROFILE_UNAVAILABLE); there is no fallback."""
        run = self.run
        registry = WorkerProfileRegistry(self.settings)
        snap = RunWorkerProfiles.from_dict(run.worker_snapshot).default
        if snap is None:
            snap = self._lock_worker_profile(run, registry)
        live = registry.get(snap.profile_id)
        problems = snapshot_problems(snap, live)
        if not problems and live is not None and not live.verified_for(snap.executable_path, snap.launch_target) \
                and not live.verified_at:
            problems.append("the profile is not verified (its last worker smoke test failed or none passed) - run "
                            "its full Test first")
        if problems:
            self._worker_profile_unavailable(run, snap.display_name, snap.profile_id, problems)
        if live is not None and live.revision() != snap.revision:
            self.log("SYSTEM", f"note: worker profile {snap.profile_id} was edited after this run started (revision "
                               f"{snap.revision} -> {live.revision()}); this run keeps its original snapshot")
        for line in snap.describe():
            self.log("SYSTEM", line)
        return build_launch_spec(snap, self.settings.worker, live)

    def _lock_worker_profile(self, run: Run, registry: WorkerProfileRegistry) -> WorkerSnapshot:
        # A run that already executed before V5 has no snapshot: it keeps the V4 worker (the legacy profile),
        # never the current default. A new run takes the default profile at its first START.
        v4_run = run.started_at is not None
        profile = registry.legacy() if v4_run else registry.default_profile()
        if profile is None:
            self._worker_profile_unavailable(run, registry.default_id(), registry.default_id(),
                                             ["the default worker profile does not exist"])
        problems = registry.ready_problems(profile)
        if problems:
            self._worker_profile_unavailable(run, profile.name, profile.id, problems)
        snap = WorkerSnapshot.of(profile, source="legacy-settings" if v4_run else "profile")
        data = RunWorkerProfiles(default=snap).to_dict()
        if not self.db.lock_worker_snapshot(run.id, data):  # never overwrite: reuse what is stored
            stored = self.db.load_run(run.id)
            existing = RunWorkerProfiles.from_dict(stored.worker_snapshot if stored else {}).default
            if existing is None:
                raise EngineError(f"could not record the worker profile of run #{run.id}")
            run.worker_snapshot = stored.worker_snapshot
            return existing
        run.worker_snapshot = data
        self.log("SYSTEM", f"Worker profile locked for run #{run.id}: {snap.display_name} ({snap.profile_id}, "
                           f"revision {snap.revision})" + (" - a V4 run keeps the legacy worker settings" if v4_run
                                                           else ""))
        return snap

    def _worker_profile_unavailable(self, run: Run, name: str, profile_id: str, problems: list[str]) -> None:
        msg = (f"{UNAVAILABLE}: worker profile '{name}' ({profile_id}) cannot be used: " + "; ".join(problems)
               + ". No fallback will be used.")
        run.status_message = msg
        self._persist_run()
        raise EngineError(msg)

    def worker_profiles_in_use(self) -> dict[str, list[int]]:
        """profile id -> resumable runs bound to it (a profile in use cannot be deleted)."""
        return self.db.worker_profiles_in_use()

    async def profiles_in_use(self) -> dict[str, list[int]]:  # for EngineHost.submit (GUI thread)
        return self.worker_profiles_in_use()

    def _make_registry(self) -> CapabilityRegistry:
        """V4 pass 2: the capability providers used for evidence collection."""
        return default_registry()

    # ------------------------------------------------------------ PLAN

    async def plan(self, goal: str, repo_path: str, pipeline: str | None = None) -> None:
        self._busy_check()
        pipeline = (pipeline or self.settings.orchestration.pipeline_version or "v1").strip().lower()
        if pipeline not in PIPELINES:
            raise EngineError(f"unknown pipeline {pipeline!r} (expected v1 or v2)")
        goal = goal.strip()
        if not goal:
            raise EngineError("enter a development request first")
        if not repo_path or not os.path.isdir(repo_path):
            raise EngineError("select a project directory first")
        if self.sm.state in EXECUTION_STATES or self.sm.state == S.STOPPING:
            raise EngineError("a run is executing; stop it before planning a new one")
        o = self.settings.orchestration
        planner = self._make_orchestrator(o.orchestrator_mode)
        run = Run(id=None, goal=goal, repo_path=os.path.abspath(repo_path), state=S.IDLE.value,
                  orchestrator_mode=o.orchestrator_mode, worker_mode=o.worker_mode,
                  settings_snapshot=self._settings_refs(pipeline), pipeline_version=pipeline)
        self.db.create_run(run)
        self.db.add_settings_refs(run.id, self._settings_refs(pipeline))
        self.run = run
        self.scheduler = None
        self.sm = StateMachine(S.IDLE, self._on_transition)
        self.emit("run_loaded", {"run_id": run.id, "goal": goal, "repo_path": run.repo_path,
                                 "pipeline_version": pipeline})
        self.sm.transition(S.PLANNING, f"PLAN clicked (pipeline {pipeline.upper()})")
        self._persist_run()
        self._planner = planner
        self._busy = asyncio.ensure_future(self._do_plan(planner, run))

    async def _do_plan(self, planner: OrchestratorProvider, run: Run) -> None:
        try:
            summary = await self._repo_summary(run.repo_path)
            request = PlanRequest(run.goal, run.repo_path, summary)
            plan = await (planner.plan_phases(request) if run.pipeline_version == "v2" else planner.plan(request))
        except asyncio.CancelledError:
            self.sm.transition(S.STOPPED, "planning cancelled")
            run.status_message = "planning cancelled"
            self._persist_run()
            return
        except (OrchestratorError, Exception) as exc:
            run.status_message = f"planning failed: {exc}"
            self.log("SYSTEM", f"PLAN FAILED - Claude did not return a usable plan: {exc}")
            self.sm.transition(S.ERROR, "planning failed")
            self._persist_run()
            return
        finally:
            for c in planner.calls:
                self.db.add_orchestrator_call(run.id, None, c)
                self.emit("orchestrator_call", {"kind": c.kind, "cost_usd": c.cost_usd, "usage": c.usage,
                                                "duration": c.duration})
            planner.calls.clear()
            self._planner = None
        TaskGraph(plan.tasks).refresh_ready_flags()
        run.plan = plan
        self.db.save_tasks(run.id, plan.tasks)
        unit = "phase" if run.pipeline_version == "v2" else "task"
        run.status_message = f"plan ready: {len(plan.tasks)} {unit}(s). Review it, then press START."
        self.sm.transition(S.PLAN_READY, f"{len(plan.tasks)} {unit}s")
        self._persist_run()
        self.emit("plan", {"plan": plan.to_dict()})
        self.log("CLAUDE", f"Plan ({run.pipeline_version.upper()}): {plan.summary}")
        for t in plan.tasks:
            if run.pipeline_version == "v2":
                self.log("CLAUDE", f"  {t.id} {t.title}  ({len(t.subtasks)} subtasks"
                                   + (f", ~{t.estimated_minutes} min" if t.estimated_minutes else "") + ")")
                for st in t.subtasks:
                    self.log("CLAUDE", f"      {st.id} {st.title}")
            else:
                self.log("CLAUDE", f"  {t.id} {t.title}" + (f"  (after {', '.join(t.dependencies)})" if t.dependencies else ""))

    async def _repo_summary(self, repo_path: str) -> str:
        info = await GitRepo(repo_path).info()
        lines = [f"Git: {info.describe()}"]
        ur = await unreal_mcp.probe(repo_path, self.settings.integrations.unreal_mcp_url)
        if ur.is_unreal_project:
            lines.append(f"Unreal Engine project ({ur.uproject}); worker Unreal MCP: {ur.describe()}")
        cm = await codebase_memory.probe(self.settings.integrations.codebase_memory_executable, repo_path)
        lines.append(f"Worker has codebase-memory-mcp: {cm.describe()}")
        return "\n".join(lines)

    # ------------------------------------------------------------ START / RESUME

    async def start(self) -> None:
        self._busy_check()
        if self.run is None or self.run.plan is None:
            raise EngineError("no plan: press PLAN first")
        if self.sm.state != S.PLAN_READY:
            raise EngineError(f"START is only valid after PLAN (state is {self.sm.state.value}); use RESUME")
        await self._begin(S.STARTING_TASK)

    async def resume(self, guidance: str = "") -> None:
        self._busy_check()
        if self.run is None or self.run.plan is None:
            raise EngineError("no run loaded")
        if self.sm.state == S.AWAITING_HUMAN:
            # Only an answer continues a waiting run. RESUME just re-runs a HUMAN_FOLLOWUP whose answer was
            # recorded but not adjudicated (e.g. the app exited during the Claude call).
            hs = self._human_gate()
            if hs is None or hs.feedback is None or hs.status == hm.AWAITING:
                raise EngineError("the run is waiting for human verification: answer it with Looks good, Issue found "
                                  "or Cannot verify")
            self.log("SYSTEM", f"{hs.phase_id}: running the human verification follow-up for the recorded answer")
            await self._begin(S.AWAITING_HUMAN)
            return
        if self.sm.state not in HALTED_STATES:
            raise EngineError(f"nothing to resume (state {self.sm.state.value})")
        target = S(self.run.resume_state) if self.run.resume_state else S.STARTING_TASK
        v2 = self.run.pipeline_version == "v2"
        if target not in EXECUTION_STATES and not (v2 and target == S.AWAITING_HUMAN):
            target = S.STARTING_TASK
        if target in (S.REPLANNING, S.REPAIRING, S.COLLECTING_EVIDENCE, S.CONSISTENCY_REVIEW) and not v2:
            target = S.STARTING_TASK
        if self.run.pipeline_version == "v2" and target == S.REPLANNING:
            st = self.run.pipeline_state
            if guidance.strip() and isinstance(st.get("pending_replan"), dict):
                st["pending_replan"]["user_guidance"] = guidance.strip()
                self.log("SYSTEM", "user guidance attached to the pending replan")
                guidance = ""
            allowed = self.settings.orchestration.v2_max_replans + int(st.get("extra_replans", 0))
            if self.sm.state == S.BLOCKED and int(st.get("replans", 0)) >= allowed:
                st["extra_replans"] = int(st.get("extra_replans", 0)) + 1
                self.log("SYSTEM", "user granted one additional replan")
            self._persist_run()
        task = self.run.plan.task(self.run.current_task_id) if self.run.current_task_id else None
        if task is not None:
            if guidance.strip():
                task.user_guidance = guidance.strip()
                self.log("SYSTEM", f"user guidance attached to {task.id}")
            if self.sm.state == S.BLOCKED and task.attempts >= self.settings.orchestration.max_attempts + task.extra_attempts:
                task.extra_attempts += 1
                self.log("SYSTEM", f"user granted one additional attempt for {task.id}")
            self.db.save_task(self.run.id, task)
        await self._begin(target)

    # ------------------------------------------------------------ V4 human verification

    def _human_gate(self) -> hm.HumanGateState | None:
        run = self.run
        if run is None or run.pipeline_version != "v2" or not run.current_task_id:
            return None
        return hm.HumanGateState.from_dict((run.pipeline_state.get("human") or {}).get(run.current_task_id))

    async def submit_human_feedback(self, verdict: str, note: str = "") -> None:
        """The user's structured answer to an ASK_USER request (LOOKS_GOOD | ISSUE_FOUND | CANNOT_VERIFY plus a
        free-text note, stored as given). It is persisted first, then Claude HUMAN_FOLLOWUP adjudicates it."""
        self._busy_check()
        if self.run is None or self.run.plan is None:
            raise EngineError("no run loaded")
        if self.sm.state != S.AWAITING_HUMAN:
            raise EngineError(f"the run is not waiting for human verification (state {self.sm.state.value})")
        hs = self._human_gate()
        if hs is None or hs.request is None:
            raise EngineError("no human verification request is recorded for the current phase")
        if hs.status != hm.AWAITING:
            raise EngineError("an answer to this request is already recorded; press RESUME to run its follow-up")
        try:
            fb = hm.HumanFeedback.create(hs.request, verdict, note)
        except ValueError as exc:
            raise EngineError(str(exc)) from exc
        hs.record_feedback(fb)
        self.run.pipeline_state.setdefault("human", {})[hs.phase_id] = hs.to_dict()
        self._persist_run()
        self.log("HUMAN", f"{hs.phase_id}: {fb.verdict}" + (f' - "{hm.short(fb.note, 160)}"' if fb.note else ""))
        await self._begin(S.AWAITING_HUMAN)

    async def _begin(self, entry: RunState) -> None:
        run = self.run
        first_start = run.started_at is None
        o = self.settings.orchestration
        orchestrator = self._make_orchestrator(run.orchestrator_mode)
        worker = self._make_worker(run.worker_mode)
        git = GitRepo(run.repo_path, lambda s: self.log("GIT", s))
        info = await git.info()
        if run.worker_mode == "ninfer" and not info.is_repo:
            raise EngineError("real worker runs require the project folder to be a git repository root "
                              f"(needed to attribute changes safely): {info.describe()}")
        if first_start:
            run.git_head, run.git_branch = info.head, info.branch
            run.initial_dirty = [e.path for e in info.dirty]
            self.log("GIT", f"pre-run: {info.describe()}")
            if info.dirty:
                self.log("GIT", f"{len(info.dirty)} pre-existing change(s) recorded; they will never be attributed to the worker")
            if info.is_repo and o.create_working_branch:
                if run.worker_mode != "ninfer":
                    self.log("GIT", "working branch not created in mock worker mode (no edits happen)")
                elif not info.head:
                    self.log("GIT", "working branch skipped: repository has no commits yet")
                else:
                    name = f"{o.working_branch_prefix}{run.id}"
                    await git.create_branch(name)
                    run.working_branch = name
                    self.log("GIT", f"created and switched to working branch {name} (uncommitted changes carried over)")
        cm = await codebase_memory.probe(self.settings.integrations.codebase_memory_executable, run.repo_path)
        ur = await unreal_mcp.probe(run.repo_path, self.settings.integrations.unreal_mcp_url)
        if not cm.available:
            self.log("MCP", "codebase-memory-mcp unavailable: worker runs in DEGRADED mode (plain repository tools)")
        if ur.is_unreal_project and not ur.mcp_reachable:
            self.log("MCP", f"Unreal MCP not reachable ({ur.detail}): DEGRADED mode for editor operations")
        integ = self.settings.integrations
        deps = SchedulerDeps(
            db=self.db, orchestrator=orchestrator, worker=worker, git=git if info.is_repo else None,
            verifier=Verifier(self.log, o.verification_timeout_seconds), settings=o,
            emit=self.emit, log=self.log, codebase_memory_available=cm.available,
            unreal_project=ur.is_unreal_project, unreal_mcp_reachable=ur.mcp_reachable,
            codebase_memory_executable=integ.codebase_memory_executable, codebase_memory_project=cm.project,
            auto_index_codebase_memory=integ.auto_index_codebase_memory,
            codebase_memory_index_timeout=integ.codebase_memory_index_timeout_seconds,
            evidence_registry=self._make_registry(), artifact_root=self.artifact_root)
        scheduler_cls = PhaseScheduler if run.pipeline_version == "v2" else Scheduler
        if run.pipeline_version == "v2":
            self.log("SYSTEM", "Pipeline V2: phase-based execution (one Qwen session per phase, Claude checkpoints "
                               "at phase boundaries)")
        self.scheduler = scheduler_cls(run, self.sm, deps, self.db.load_attempts(run.id))
        self._busy = asyncio.ensure_future(self._drive(entry))

    async def _drive(self, entry: RunState) -> None:
        try:
            await self.scheduler.drive(entry)
        finally:
            self.emit("runs", {"runs": self.db.list_runs()})
        if self.sm.state == S.COMPLETED:  # a finished run's artifacts become subject to retention
            await self.cleanup_artifacts()

    # ------------------------------------------------------------ PAUSE / STOP

    async def pause(self) -> None:
        if self.scheduler and self._busy and not self._busy.done():
            self.scheduler.request_pause()
        else:
            raise EngineError("nothing is running")

    async def stop(self) -> None:
        if self._planner is not None and self._busy and not self._busy.done():
            self.log("SYSTEM", "STOP requested during planning")
            self._busy.cancel()  # propagates into the process runner, which kills the CLI
            return
        if self.scheduler and self._busy and not self._busy.done():
            self.scheduler.request_stop()
            return
        if self.run and self.sm.can(S.STOPPED) and self.sm.state != S.STOPPED:
            self.run.interrupted = True
            self.sm.transition(S.STOPPED, "stopped by user")
            self._persist_run()
            return
        raise EngineError("nothing to stop")

    # ------------------------------------------------------------ history

    async def load_run(self, run_id: int) -> None:
        self._busy_check()
        if self.sm.state in EXECUTION_STATES:
            raise EngineError("stop the active run first")
        run = self.db.load_run(run_id)
        if run is None:
            raise EngineError(f"run {run_id} not found")
        attempts = self.db.load_attempts(run_id)
        if run.plan:
            for t in run.plan.tasks:  # repair counters after a crash
                t.attempts = len([a for a in attempts.get(t.id, []) if not a.interrupted])
                if t.status.value in ("RUNNING", "VERIFYING", "REVIEWING"):
                    t.status = TaskStatus.PAUSED
            self.db.save_tasks(run_id, run.plan.tasks)
        self.run = run
        self.scheduler = None
        try:
            st = S(run.state)
        except ValueError:
            st = S.ERROR
        self.sm = StateMachine(S.IDLE, self._on_transition)
        self.sm.force(st, "loaded from history")
        self.emit("run_loaded", {"run_id": run.id, "goal": run.goal, "repo_path": run.repo_path,
                                 "pipeline_version": run.pipeline_version})
        self.emit("run", run_payload(run))
        self.emit("state", {"state": run.state, "previous": None, "reason": "loaded", "run_id": run.id})
        if run.plan:
            self.emit("plan", {"plan": run.plan.to_dict()})
        for lst in attempts.values():
            for a in lst:
                self.emit("attempt", {"attempt": a.to_dict()})
        self.emit("history_logs", {"logs": self.db.load_logs(run_id)})
        cost = self.db.total_orchestrator_cost(run_id)
        self.emit("run_cost", {"cost_usd": cost})

    async def list_runs(self) -> None:
        self.emit("runs", {"runs": self.db.list_runs()})

    async def request_fresh_session(self) -> None:
        """V2: the current phase continues in a NEW Qwen session (with a compact handoff) at its
        next turn. The running turn, if any, is not interrupted."""
        run = self.run
        if run is None or run.plan is None or run.pipeline_version != "v2":
            raise EngineError("a fresh Qwen session can only be requested for a loaded V2 (phase) run")
        phase = run.plan.task(run.current_task_id) if run.current_task_id else None
        if phase is None:
            phase = next((p for p in sorted(run.plan.tasks, key=lambda p: p.position)
                          if p.status.value not in ("PASSED", "SUPERSEDED")), None)
        if phase is None:
            raise EngineError("no phase left to run")
        phase.fresh_session_requested = True
        self.db.save_task(run.id, phase)
        self.emit("task", {"task": phase.to_dict()})
        self.log("SYSTEM", f"{phase.id}: the next Qwen turn will start a NEW session (requested by the user)")

    # ------------------------------------------------------------ misc

    def _persist_run(self) -> None:
        if self.run is None:
            return
        self.run.state = self.sm.state.value
        self.db.update_run(self.run)
        self.emit("run", run_payload(self.run))

    def _settings_refs(self, pipeline: str = "v1") -> dict[str, str]:
        """Non-secret references recorded with each run."""
        s = self.settings
        return {
            "pipeline_version": pipeline,
            "orchestrator_mode": s.orchestration.orchestrator_mode,
            "worker_mode": s.orchestration.worker_mode,
            "claude_executable": s.claude.executable,
            "claude_model": s.claude.model,
            "worker_profile_default": s.worker_profiles.default_profile_id,  # the run locks it at its first START
            "worker_executable": s.worker.executable,
            "worker_codex_home": s.worker.codex_home,
            "worker_model": s.worker.model,
            "worker_sandbox": s.worker.sandbox,
            "max_attempts": str(s.orchestration.max_attempts),
            "human_verification_enabled": str(s.orchestration.human_verification_enabled),
            "evidence_enabled": str(s.orchestration.evidence_enabled),
            "consistency_review_enabled": str(s.orchestration.consistency_review_enabled),
            "implementation_guard_enabled": str(s.orchestration.implementation_guard_enabled),
        }


class EngineHost:
    """Runs the Engine on a dedicated thread with its own asyncio loop."""

    def __init__(self, settings: Settings, emit: EmitFn, db_path: str | None = None):
        self._settings = settings
        self._emit = emit
        self._db_path = db_path
        self.loop: asyncio.AbstractEventLoop | None = None
        self.engine: Engine | None = None
        self._ready = threading.Event()
        self._thread = threading.Thread(target=self._main, name="orchestrator-engine", daemon=True)

    def start(self) -> None:
        self._thread.start()
        self._ready.wait(timeout=30)
        if self.engine is None:
            raise RuntimeError("engine failed to start")

    def _main(self) -> None:
        # On Windows the default (Proactor) loop supports subprocesses in any thread.
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.engine = Engine(self._settings, self._emit, self._db_path)
            self.loop.run_until_complete(self.engine.start_background())
        finally:
            self._ready.set()
        self.loop.run_forever()

    def submit(self, fn: Callable[[Engine], Coroutine[Any, Any, Any]]) -> Future:
        """Schedule engine work from any thread; errors are reported as events."""
        assert self.loop is not None and self.engine is not None
        engine = self.engine

        async def wrapper() -> Any:
            try:
                return await fn(engine)
            except Exception as exc:
                engine.emit("error", {"message": str(exc)})
                engine.log("SYSTEM", f"error: {exc}")
                return None

        return asyncio.run_coroutine_threadsafe(wrapper(), self.loop)

    def shutdown(self, timeout: float = 40) -> None:
        if self.loop is None or self.engine is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self.engine.shutdown(), self.loop)
        try:
            fut.result(timeout=timeout)
        except Exception:
            pass
        self.loop.call_soon_threadsafe(self.loop.stop)
        self._thread.join(timeout=5)
