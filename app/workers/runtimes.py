"""V5.1 worker runtimes: the ONE place that knows which kinds of worker CLI exist.

    WorkerRuntime        how profiles of one runtime are validated, checked before a resume, probed and launched
    CodexWorkerRuntime   the V5 Codex worker (codex-ninfer.exe, codex-lmstudio.exe, ...) - behaviour unchanged
    ClineWorkerRuntime   the Cline CLI in headless --json mode (app.workers.cline_cli)
    WORKER_RUNTIMES      runtime id -> runtime;  get_runtime()

A profile's `runtime` selects its WorkerRuntime; the engine turns a run's snapshot into a WorkerProvider here and
the schedulers only ever see that WorkerProvider. Adding a runtime = one class + one registry entry.
"""
from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
from typing import Any, Mapping

from app.config import codex_config
from app.config.discovery import ExecutableRejected, validate_cline_executable, validate_executable_for_policy
from app.workers.base import LogFn, WorkerCapabilities, WorkerProvider
from app.workers.profiles import (
    LEGACY_PROFILE_ID, RUNTIME_CLINE, RUNTIME_CODEX, WorkerLaunchSpec, WorkerProfile, WorkerSnapshot, same_path,
)


def normal_codex_home() -> str:
    return str(Path(os.environ.get("USERPROFILE", str(Path.home()))) / ".codex")


def codex_home_problems(home: str) -> list[str]:
    if not home:
        return ["no CODEX_HOME is configured"]
    expanded = os.path.expandvars(home)
    if not Path(expanded).is_dir():
        return [f"CODEX_HOME does not exist: {expanded}"]
    if same_path(expanded, normal_codex_home()):
        return ["CODEX_HOME points at the normal ~/.codex profile; a worker needs a dedicated profile home"]
    return []


def executable_problems(executable: str, policy: str) -> list[str]:
    try:
        validate_executable_for_policy(executable, policy)
    except ExecutableRejected as exc:
        return [str(exc)]
    return []


class WorkerRuntime:
    id = ""
    display_name = ""
    capabilities = WorkerCapabilities()
    # Shown with every profile of this runtime (what the pipeline does differently for it).
    notes: tuple[str, ...] = ()

    def profile_problems(self, p: WorkerProfile, others: list[WorkerProfile]) -> tuple[list[str], list[str], dict]:
        """Runtime-specific quick checks of a profile (errors, warnings, display info). Never launches anything."""
        raise NotImplementedError

    def snapshot_problems(self, snap: WorkerSnapshot) -> list[str]:
        """Why a run's locked snapshot cannot be launched now (the snapshot's own executable/configuration)."""
        raise NotImplementedError

    def create_worker(self, spec: WorkerLaunchSpec, log: LogFn | None = None) -> WorkerProvider:
        raise NotImplementedError

    async def probe(self, p: WorkerProfile) -> str:
        """Explicit quick check that the CLI responds (no model call). "" = not applicable for this runtime."""
        return ""


class CodexWorkerRuntime(WorkerRuntime):
    id = RUNTIME_CODEX
    display_name = "Codex"

    def profile_problems(self, p, others):
        errors: list[str] = []
        warnings: list[str] = []
        errors += executable_problems(p.codex_executable, p.executable_policy)
        if p.is_legacy and not p.codex_home:  # V4 allowed it: the worker then uses its own default profile
            warnings.append("no CODEX_HOME configured: the worker uses its built-in default profile")
        else:
            errors += codex_home_problems(p.codex_home)
        info = codex_config.inspect_home(p.codex_home)
        if info.exists and not info.has_config:
            (warnings if p.is_legacy else errors).append(f"{codex_config.config_path(p.codex_home)} does not exist")
        if info.parse_error:
            errors.append(info.parse_error)
        for name in info.env_refs:
            if name not in p.required_env:
                warnings.append(f"config.toml reads ${name}; add it to the required env names so a missing value is "
                                "detected before a run")
        if p.pass_model_flag and p.model and info.model and p.model != info.model:
            warnings.append(f"--model {p.model} overrides config.toml's model {info.model}")
        shared = [o.name for o in others if o.id != p.id and o.runtime == RUNTIME_CODEX
                  and same_path(o.codex_home, p.codex_home)]
        if shared:
            warnings.append(f"shares its CODEX_HOME with {', '.join(shared)} (sessions and config.toml are shared)")
        info_d = {"model": info.model, "provider": info.provider, "base_url": info.base_url,
                  "context_window": info.context_window, "env_refs": info.env_refs, "mcp_servers": info.mcp_servers,
                  "config": str(codex_config.config_path(p.codex_home)) if p.codex_home else ""}
        return errors, warnings, info_d

    def snapshot_problems(self, snap):
        problems = executable_problems(snap.codex_executable, snap.executable_policy)
        if snap.codex_home or snap.profile_id != LEGACY_PROFILE_ID:  # V4 allowed a legacy worker without CODEX_HOME
            problems += codex_home_problems(snap.codex_home)
        return problems

    def create_worker(self, spec, log=None):
        from app.workers.codex_ninfer import CodexNInferWorker
        return CodexNInferWorker(spec, log)


def cline_settings_dir(config_dir: str = "", environ: Mapping[str, str] | None = None) -> Path:
    """Where Cline keeps its provider settings (<data dir>/settings). CLINE_DATA_DIR wins, as in the CLI."""
    env = os.environ if environ is None else environ
    if env.get("CLINE_DATA_DIR"):
        return Path(env["CLINE_DATA_DIR"]) / "settings"
    base = Path(os.path.expandvars(config_dir)) if config_dir else Path(os.environ.get("USERPROFILE",
                                                                                         str(Path.home()))) / ".cline"
    return base / "data" / "settings"


def cline_providers(config_dir: str = "", environ: Mapping[str, str] | None = None) -> dict[str, str] | None:
    """Provider id -> saved model from Cline's providers.json. ONLY ids and model names are read (never keys or
    tokens). None = the file is missing or unreadable."""
    path = cline_settings_dir(config_dir, environ) / "providers.json"
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    out: dict[str, str] = {}
    for pid, rec in ((data or {}).get("providers") or {}).items():
        settings = rec.get("settings") if isinstance(rec, dict) else None
        model = settings.get("model") if isinstance(settings, dict) else ""
        out[str(pid)] = str(model or "")
    return out


class ClineWorkerRuntime(WorkerRuntime):
    id = RUNTIME_CLINE
    display_name = "Cline"
    capabilities = WorkerCapabilities(tool_call_count=True, subtask_progress=False, sandbox=False)
    notes = (
        "no command sandbox: Cline's tools are not confined to the repository (Codex --sandbox has no Cline "
        "equivalent)",
        "no headless session resume in the installed Cline CLI: every worker turn is a new Cline session that gets "
        "AI Studio's compact handoff",
        "no structured todo list: the implementation guard's no-subtask-progress limit is not applied (wall-clock and "
        "tool-call limits are)",
    )

    def profile_problems(self, p, others):
        errors: list[str] = []
        warnings: list[str] = list(self.notes)
        try:
            validate_cline_executable(p.cline_executable)
        except ExecutableRejected as exc:
            errors.append(str(exc))
        if not p.provider:
            errors.append("no provider set (it is passed as -P; e.g. the provider configured with `cline auth`)")
        if not p.model:
            errors.append("no model set (it is passed as -m and locked with every run)")
        if p.cline_config_dir and not Path(os.path.expandvars(p.cline_config_dir)).is_dir():
            errors.append(f"Cline configuration directory does not exist: {p.cline_config_dir}")
        if p.codex_home:
            warnings.append("CODEX_HOME is ignored by the Cline runtime")
        env = dict(os.environ) | dict(p.extra_env)
        saved = cline_providers(p.cline_config_dir, env)
        if saved is None:
            warnings.append(f"Cline provider settings not found under {cline_settings_dir(p.cline_config_dir, env)}")
        elif p.provider and p.provider not in saved and not p.required_env:
            warnings.append(f"Cline has no saved settings for provider {p.provider!r}: configure it with "
                            f"`cline auth {p.provider}` or list the API key's environment variable NAME")
        info = {"provider": p.provider, "model": p.model, "cline_providers": sorted(saved or {}),
                "saved_model": (saved or {}).get(p.provider, ""),
                "settings_dir": str(cline_settings_dir(p.cline_config_dir, env))}
        return errors, warnings, info

    def snapshot_problems(self, snap):
        problems: list[str] = []
        try:
            validate_cline_executable(snap.cline_executable)
        except ExecutableRejected as exc:
            problems.append(str(exc))
        if not snap.provider or not snap.model:
            problems.append("the snapshot has no provider/model")
        if snap.cline_config_dir and not Path(os.path.expandvars(snap.cline_config_dir)).is_dir():
            problems.append(f"Cline configuration directory does not exist: {snap.cline_config_dir}")
        return problems

    def create_worker(self, spec, log=None):
        from app.workers.cline_cli import ClineWorker
        return ClineWorker(spec, log)

    async def probe(self, p):
        from app.workers.cline_cli import ClineWorker
        spec = WorkerLaunchSpec.from_snapshot(WorkerSnapshot.of(p), _NoPolicy())
        return await ClineWorker(spec).version()


class _NoPolicy:
    sandbox = ""
    timeout_seconds = 60


WORKER_RUNTIMES: dict[str, WorkerRuntime] = {r.id: r for r in (CodexWorkerRuntime(), ClineWorkerRuntime())}


def get_runtime(runtime_id: str) -> WorkerRuntime | None:
    return WORKER_RUNTIMES.get(runtime_id or RUNTIME_CODEX)


def probe_sync(p: WorkerProfile) -> str:
    """For the GUI thread (no running event loop): the runtime's explicit CLI probe."""
    r = get_runtime(p.runtime)
    if r is None:
        return ""
    return asyncio.run(r.probe(p))


def describe_capabilities(runtime_id: str) -> dict[str, Any]:
    r = get_runtime(runtime_id)
    c = r.capabilities if r else WorkerCapabilities()
    return {"tool_call_count": c.tool_call_count, "subtask_progress": c.subtask_progress, "sandbox": c.sandbox}
