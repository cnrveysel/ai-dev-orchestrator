"""V5 worker launcher: turns a worker-profile snapshot into a ready worker adapter.

The schedulers never see runtimes, providers, models or CODEX_HOMEs - they get a WorkerProvider from the engine.
V5.1: which adapter starts which process is decided by the snapshot's runtime through app.workers.runtimes (the one
runtime registry); provider-specific configuration lives in the runtime's own configuration (a Codex profile's
CODEX_HOME/config.toml, Cline's provider settings), never in code. There is no fallback of any kind - not to another
profile and not to another runtime: an unusable snapshot is reported as WORKER_PROFILE_UNAVAILABLE.
"""
from __future__ import annotations

import os
from typing import Callable, Mapping

from app.workers.base import LogFn, WorkerProvider
from app.workers.profiles import WorkerLaunchSpec, WorkerProfile, WorkerSnapshot
from app.workers.runtimes import (  # noqa: F401  (re-exported: V5 callers import them from here)
    WORKER_RUNTIMES, codex_home_problems, executable_problems, get_runtime, normal_codex_home,
)

UNAVAILABLE = "WORKER_PROFILE_UNAVAILABLE"

# runtime -> adapter factory (derived from the runtime registry; kept for V5 callers).
RUNTIME_ADAPTERS: dict[str, Callable[[WorkerLaunchSpec, LogFn | None], WorkerProvider]] = {
    rid: r.create_worker for rid, r in WORKER_RUNTIMES.items()
}


class WorkerProfileUnavailable(RuntimeError):
    """The worker profile a run is bound to cannot be used right now. Never answered with another profile."""

    def __init__(self, profile_id: str, problems: list[str]):
        self.profile_id = profile_id
        self.problems = list(problems)
        super().__init__(f"{UNAVAILABLE}: worker profile '{profile_id}' - " + "; ".join(problems)
                         + ". No other profile is used as a fallback.")


def env_problems(required: tuple[str, ...] | list[str], environ: Mapping[str, str] | None = None) -> list[str]:
    env = os.environ if environ is None else environ
    missing = [n for n in required if not env.get(n)]
    return [f"required environment variable(s) not set: {', '.join(missing)} (values are never stored by AI Studio)"
            ] if missing else []


def snapshot_problems(snap: WorkerSnapshot, live: WorkerProfile | None,
                      environ: Mapping[str, str] | None = None) -> list[str]:
    """Why the run's locked snapshot cannot be launched now (empty = launchable). The snapshot's own runtime,
    executable, configuration and environment are checked - not whatever the profile says today."""
    problems: list[str] = []
    runtime = get_runtime(snap.runtime)
    if runtime is None:
        problems.append(f"unknown worker runtime {snap.runtime!r}")
    if live is None:
        problems.append("the profile no longer exists in Settings > Worker Profiles (re-register it with this "
                        "profile id to continue the run)")
    elif not live.enabled:
        problems.append("the profile is disabled")
    elif live.runtime != snap.runtime:
        problems.append(f"the profile now uses the {live.runtime} runtime but this run is bound to {snap.runtime}")
    if runtime is not None:
        problems += runtime.snapshot_problems(snap)
    problems += env_problems(snap.required_env, environ)
    if live is not None:
        missing = [k for k in snap.secret_env_names if k not in live.extra_env]
        if missing:
            problems.append(f"secret env value(s) {', '.join(missing)} are no longer configured in the profile")
    return problems


def build_launch_spec(snap: WorkerSnapshot, policy, live: WorkerProfile | None) -> WorkerLaunchSpec:
    """`policy` = the global worker execution policy (Settings.worker: sandbox, timeout)."""
    secrets = {k: live.extra_env[k] for k in snap.secret_env_names if live is not None and k in live.extra_env}
    return WorkerLaunchSpec.from_snapshot(snap, policy, secret_values=secrets)


def launch_worker(spec: WorkerLaunchSpec, log: LogFn | None = None) -> WorkerProvider:
    """The worker adapter for `spec`, chosen by its runtime only. The process is started per turn by the adapter."""
    runtime = get_runtime(spec.runtime)
    if runtime is None:
        raise WorkerProfileUnavailable(spec.profile_id or "?", [f"unknown worker runtime {spec.runtime!r}"])
    return runtime.create_worker(spec, log)
