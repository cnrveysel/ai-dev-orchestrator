"""V5 worker profiles: WHICH worker runtime, build, configuration and process-local environment a run's worker uses.

    WorkerProfile     a named worker configuration (a registry entry). Provider / model / base URL are display
                      metadata: the profile's CODEX_HOME/config.toml stays the authoritative source. `--model` is
                      only passed when the profile asks for it (the legacy V4 profile does).
    WorkerSnapshot    the immutable copy of a profile's execution-relevant fields, persisted with a run BEFORE the
                      run's first worker launch. Resume always uses it - never the profile selected later.
    WorkerLaunchSpec  a snapshot + the global worker execution policy (sandbox, timeout): exactly what the runtime
                      adapter (app.workers.launcher) needs to start the process.

The V4 worker configuration (Settings.worker) is always available as the virtual profile LEGACY_PROFILE_ID, so an
existing installation keeps behaving exactly as before. Secrets are never stored: a profile may name environment
variables that must exist (e.g. OPENROUTER_API_KEY) but never holds their values.

V5.1: a profile belongs to one worker RUNTIME (app.workers.runtimes). "codex" (the default, and every profile saved
before V5.1) uses codex_executable + CODEX_HOME; "cline" uses cline_executable, provider + model (passed as -P / -m;
Cline keeps the credentials in its own configuration) and an optional Cline configuration directory. Runtime-specific
fields are simply unused by the other runtime; CODEX_HOME never applies to Cline.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any

from app.config.settings import LEGACY_WORKER_PROFILE_ID

RUNTIME_CODEX = "codex"
RUNTIME_CLINE = "cline"
LEGACY_PROFILE_ID = LEGACY_WORKER_PROFILE_ID
LEGACY_DISPLAY_NAME = "NInfer Qwen (legacy settings)"
MODE_EXTERNAL = "external"  # AI Studio references the CODEX_HOME; it never rewrites its config.toml
MODE_MANAGED = "managed"  # AI Studio created the CODEX_HOME and may edit its configuration
# "ninfer": the V4 executable rule (the file name must contain "ninfer"); "codex": any dedicated Codex build that
# is not the normal `codex` launcher / npm shim.
EXECUTABLE_POLICIES = ("ninfer", "codex")

SNAPSHOT_SCHEMA = 1
REDACTED = "***redacted***"
_SECRET_NAME = re.compile(r"(KEY|TOKEN|SECRET|PASSWORD|PASSWD|AUTH|BEARER|COOKIE|CREDENTIAL|PRIVATE)", re.I)
_ENV_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")


def is_secret_name(name: str) -> bool:
    """Heuristic: environment variable names that usually hold credentials."""
    return bool(_SECRET_NAME.search(name or ""))


def valid_env_name(name: str) -> bool:
    return bool(_ENV_NAME.match(name or ""))


def redact_env(env: dict[str, str] | None) -> dict[str, str]:
    """A copy safe for logs and persistence: secret-looking values are replaced."""
    return {k: (REDACTED if is_secret_name(k) else v) for k, v in (env or {}).items()}


def _norm_path(p: str) -> str:
    return os.path.normcase(os.path.abspath(os.path.expandvars(p))) if p else ""


def _executable_path(runtime: str, codex_executable: str, cline_executable: str) -> str:
    return cline_executable if runtime == RUNTIME_CLINE else codex_executable


def _launch_target(runtime: str, codex_home: str, provider: str, model: str, cline_config_dir: str) -> str:
    """Besides the executable, what a smoke-test verification is bound to: Codex = CODEX_HOME (as in V5); Cline = the
    provider/model selection and the Cline configuration directory."""
    if runtime == RUNTIME_CLINE:
        return f"cline:provider={provider};model={model};config={_norm_path(cline_config_dir)}"
    return codex_home


def same_path(a: str, b: str) -> bool:
    return bool(a) and bool(b) and _norm_path(a) == _norm_path(b)


@dataclass
class WorkerProfile:
    id: str
    display_name: str = ""
    runtime: str = RUNTIME_CODEX
    codex_executable: str = ""
    codex_home: str = ""
    enabled: bool = True
    mode: str = MODE_EXTERNAL
    extra_env: dict[str, str] = field(default_factory=dict)  # process-local, non-secret
    required_env: list[str] = field(default_factory=list)  # NAMES that must exist in the environment; never values
    pass_model_flag: bool = False  # config.toml is authoritative unless the profile asks for --model
    model: str = ""
    extra_args: list[str] = field(default_factory=list)
    executable_policy: str = "codex"
    # --- V5.1 Cline runtime (unused by Codex profiles) ---
    cline_executable: str = ""
    cline_config_dir: str = ""  # passed as --config; empty = Cline's default configuration (~/.cline)
    # --- display metadata (optional; config.toml stays authoritative). Cline: provider + model are the -P / -m
    #     selection of the run. ---
    provider: str = ""
    base_url: str = ""
    context_window: int = 0
    notes: str = ""
    # --- smoke-test verification, bound to the exact executable + CODEX_HOME that passed ---
    verified_executable: str = ""
    verified_codex_home: str = ""
    verified_at: float = 0.0
    created_at: float = 0.0
    updated_at: float = 0.0

    @property
    def is_legacy(self) -> bool:
        return self.id == LEGACY_PROFILE_ID

    @property
    def name(self) -> str:
        return self.display_name or self.id

    @property
    def executable_path(self) -> str:
        return _executable_path(self.runtime, self.codex_executable, self.cline_executable)

    @property
    def launch_target(self) -> str:
        return _launch_target(self.runtime, self.codex_home, self.provider, self.model, self.cline_config_dir)

    def revision(self) -> str:
        """Short hash of the execution-relevant configuration (changes when the launch would change)."""
        key = {
            "runtime": self.runtime, "exe": _norm_path(self.codex_executable), "home": _norm_path(self.codex_home),
            "env": sorted(self.extra_env.items()), "required_env": sorted(self.required_env),
            "model": self.model if self.pass_model_flag else "", "pass_model": self.pass_model_flag,
            "args": list(self.extra_args), "policy": self.executable_policy,
        }
        if self.runtime == RUNTIME_CLINE:  # Codex keys (and so every V5 revision) are unchanged
            key.update(cline_exe=_norm_path(self.cline_executable), provider=self.provider, model=self.model,
                       cline_config=_norm_path(self.cline_config_dir))
        return hashlib.sha256(json.dumps(key, sort_keys=True).encode("utf-8")).hexdigest()[:12]

    def verified_for(self, executable: str, target: str) -> bool:
        """`target` is the runtime's launch target (Codex: CODEX_HOME). For Cline it is stored in
        verified_codex_home too (the field keeps its V5 name so saved settings stay valid)."""
        if self.runtime == RUNTIME_CLINE:
            return (same_path(self.verified_executable, executable) and bool(target)
                    and self.verified_codex_home == target)
        return same_path(self.verified_executable, executable) and same_path(self.verified_codex_home, target)

    def is_verified(self) -> bool:
        return self.verified_for(self.executable_path, self.launch_target)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> "WorkerProfile":
        known = {f.name: f for f in fields(cls)}
        kw: dict[str, Any] = {}
        for k, v in (d or {}).items():
            if k not in known:
                continue
            if k in ("extra_env",):
                v = {str(a): str(b) for a, b in (v or {}).items()} if isinstance(v, dict) else {}
            elif k in ("required_env", "extra_args"):
                v = [str(x) for x in v] if isinstance(v, list) else []
            elif k in ("enabled", "pass_model_flag"):
                v = bool(v)
            elif k in ("context_window",):
                try:
                    v = int(v or 0)
                except (TypeError, ValueError):
                    v = 0
            elif k in ("verified_at", "created_at", "updated_at"):
                try:
                    v = float(v or 0.0)
                except (TypeError, ValueError):
                    v = 0.0
            else:
                v = "" if v is None else str(v)
            kw[k] = v
        kw.setdefault("id", "")
        if not kw.get("runtime"):
            kw["runtime"] = RUNTIME_CODEX  # every profile saved before V5.1 is a Codex profile
        return cls(**kw)


def legacy_profile(worker: Any) -> WorkerProfile:
    """The V4 worker configuration (Settings.worker) as a profile. Nothing is copied or migrated: the legacy
    profile is re-derived from the settings, so the existing Worker settings keep working unchanged."""
    return WorkerProfile(
        id=LEGACY_PROFILE_ID, display_name=LEGACY_DISPLAY_NAME, runtime=RUNTIME_CODEX,
        codex_executable=worker.executable, codex_home=worker.codex_home, enabled=True, mode=MODE_EXTERNAL,
        extra_env=dict(worker.extra_env), required_env=[], pass_model_flag=worker.pass_model_flag,
        model=worker.model, extra_args=list(worker.extra_args), executable_policy="ninfer",
        provider="ninfer", base_url=worker.ninfer_base_url,
        verified_executable=worker.verified_executable, verified_codex_home=worker.verified_codex_home,
        verified_at=worker.verified_at)


@dataclass(frozen=True)
class WorkerSnapshot:
    """Immutable, secret-free record of the worker configuration a run executes with."""
    profile_id: str
    display_name: str
    revision: str
    runtime: str
    codex_executable: str
    codex_home: str
    extra_env: dict[str, str]  # non-secret values only
    secret_env_names: tuple[str, ...]  # extra_env keys whose values were NOT persisted (re-read from the profile)
    required_env: tuple[str, ...]
    pass_model_flag: bool
    model: str
    extra_args: tuple[str, ...]
    executable_policy: str
    mode: str = MODE_EXTERNAL
    provider: str = ""
    base_url: str = ""
    context_window: int = 0
    locked_at: float = 0.0
    source: str = "profile"  # profile | legacy-settings (a V4 run first resumed under V5)
    schema: int = SNAPSHOT_SCHEMA
    # V5.1 Cline runtime (absent from V5 snapshots -> "")
    cline_executable: str = ""
    cline_config_dir: str = ""

    @property
    def executable_path(self) -> str:
        return _executable_path(self.runtime, self.codex_executable, self.cline_executable)

    @property
    def launch_target(self) -> str:
        return _launch_target(self.runtime, self.codex_home, self.provider, self.model, self.cline_config_dir)

    @classmethod
    def of(cls, p: WorkerProfile, *, source: str = "profile", now: float | None = None) -> "WorkerSnapshot":
        secret = tuple(sorted(k for k in p.extra_env if is_secret_name(k)))
        return cls(
            profile_id=p.id, display_name=p.name, revision=p.revision(), runtime=p.runtime,
            codex_executable=p.codex_executable, codex_home=p.codex_home,
            extra_env={k: v for k, v in p.extra_env.items() if k not in secret}, secret_env_names=secret,
            required_env=tuple(p.required_env), pass_model_flag=p.pass_model_flag, model=p.model,
            extra_args=tuple(p.extra_args), executable_policy=p.executable_policy, mode=p.mode,
            provider=p.provider, base_url=p.base_url, context_window=p.context_window,
            locked_at=time.time() if now is None else now, source=source,
            cline_executable=p.cline_executable, cline_config_dir=p.cline_config_dir)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k in ("secret_env_names", "required_env", "extra_args"):
            d[k] = list(d[k])
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "WorkerSnapshot | None":
        if not isinstance(d, dict) or not d.get("profile_id"):
            return None
        known = {f.name for f in fields(cls)}
        kw = {k: v for k, v in d.items() if k in known}
        kw["extra_env"] = {str(a): str(b) for a, b in (kw.get("extra_env") or {}).items()}
        for k in ("secret_env_names", "required_env", "extra_args"):
            kw[k] = tuple(str(x) for x in (kw.get(k) or []))
        kw["pass_model_flag"] = bool(kw.get("pass_model_flag", False))
        for k in ("display_name", "revision", "runtime", "codex_executable", "codex_home", "model",
                  "executable_policy", "mode", "provider", "base_url", "source", "cline_executable",
                  "cline_config_dir"):
            kw[k] = str(kw.get(k) or "")
        kw["runtime"] = kw["runtime"] or RUNTIME_CODEX
        kw["executable_policy"] = kw["executable_policy"] or "codex"
        kw["context_window"] = int(kw.get("context_window") or 0)
        kw["locked_at"] = float(kw.get("locked_at") or 0.0)
        kw["schema"] = int(kw.get("schema") or SNAPSHOT_SCHEMA)
        return cls(**kw)

    def describe(self) -> list[str]:
        """Sanitized, human-readable identity lines (logs / UI). Never contains secret values."""
        env = ", ".join(f"{k}={v}" for k, v in redact_env(self.extra_env).items())
        secret = ", ".join(f"{k}=<from profile, not stored>" for k in self.secret_env_names)
        lines = [f"Worker runtime: {self.runtime.capitalize()}",
                 f"Worker profile: {self.display_name} ({self.profile_id}, revision {self.revision})"]
        if self.runtime == RUNTIME_CLINE:
            lines += [f"Provider: {self.provider} (-P)", f"Model: {self.model} (-m)",
                      f"Executable: {self.cline_executable}",
                      f"Cline configuration: {self.cline_config_dir or '(Cline default, ~/.cline)'}"]
        else:
            lines += [f"Provider: {self.provider or '(from config.toml)'}",
                      f"Model: {self.model or '(from config.toml)'}" + ("" if self.pass_model_flag or not self.model
                                                                        else " (display only; config.toml decides)"),
                      f"Executable: {self.codex_executable}",
                      f"CODEX_HOME: {self.codex_home}"]
        if env or secret:
            lines.append("Extra env (process-local): " + ", ".join(x for x in (env, secret) if x))
        if self.required_env:
            lines.append("Required env (names only): " + ", ".join(self.required_env))
        return lines


@dataclass(frozen=True)
class RunWorkerProfiles:
    """What a run's worker executes with: the run default (locked at the first START) plus a reserved map for
    future per-phase overrides. V5 always uses the run default; nothing ever switches profiles automatically."""
    default: WorkerSnapshot | None = None
    phase_overrides: dict[str, WorkerSnapshot] = field(default_factory=dict)

    def for_phase(self, phase_id: str | None) -> WorkerSnapshot | None:
        return self.phase_overrides.get(phase_id or "") or self.default

    def to_dict(self) -> dict[str, Any]:
        if self.default is None and not self.phase_overrides:
            return {}
        return {"schema": SNAPSHOT_SCHEMA, "default": self.default.to_dict() if self.default else None,
                "phase_overrides": {k: v.to_dict() for k, v in self.phase_overrides.items()}}

    @classmethod
    def from_dict(cls, d: dict[str, Any] | None) -> "RunWorkerProfiles":
        if not isinstance(d, dict) or not d:
            return cls()
        overrides = {}
        for k, v in (d.get("phase_overrides") or {}).items():
            snap = WorkerSnapshot.from_dict(v)
            if snap is not None:
                overrides[str(k)] = snap
        return cls(default=WorkerSnapshot.from_dict(d.get("default")), phase_overrides=overrides)


@dataclass(frozen=True)
class WorkerLaunchSpec:
    """Everything the runtime adapter needs to start one worker process. The attribute names used by the Codex
    adapter (executable, codex_home, extra_env, model, pass_model_flag, sandbox, timeout_seconds, extra_args) are
    the same as WorkerSettings', so the adapter launches a legacy WorkerSettings and a spec identically."""
    runtime: str
    executable: str
    codex_home: str
    extra_env: dict[str, str]  # resolved, in memory only (may hold a legacy secret re-read from the profile)
    model: str
    pass_model_flag: bool
    extra_args: list[str]
    sandbox: str
    timeout_seconds: int
    executable_policy: str = "ninfer"
    profile_id: str = ""
    profile_name: str = ""
    profile_revision: str = ""
    provider: str = ""
    required_env: tuple[str, ...] = ()
    snapshot: WorkerSnapshot | None = None
    config_dir: str = ""  # V5.1 Cline: --config (empty = Cline default)

    @classmethod
    def from_snapshot(cls, snap: WorkerSnapshot, policy: Any, secret_values: dict[str, str] | None = None
                      ) -> "WorkerLaunchSpec":
        """`policy` is the global worker execution policy (Settings.worker): sandbox and timeout are not part of a
        profile, so their V4 semantics are unchanged."""
        env = dict(snap.extra_env)
        for k in snap.secret_env_names:
            if secret_values and k in secret_values:
                env[k] = secret_values[k]
        return cls(runtime=snap.runtime, executable=snap.executable_path, codex_home=snap.codex_home,
                   extra_env=env, model=snap.model, pass_model_flag=snap.pass_model_flag,
                   extra_args=list(snap.extra_args), sandbox=policy.sandbox, timeout_seconds=policy.timeout_seconds,
                   executable_policy=snap.executable_policy, profile_id=snap.profile_id,
                   profile_name=snap.display_name, profile_revision=snap.revision, provider=snap.provider,
                   required_env=tuple(snap.required_env), snapshot=snap, config_dir=snap.cline_config_dir)
