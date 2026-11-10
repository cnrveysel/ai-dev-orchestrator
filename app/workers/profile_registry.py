"""V5 worker profile registry (Settings > Worker Profiles).

Registered profiles live in Settings.worker_profiles (settings.json); the V4 worker settings (Settings.worker) are
always present as the virtual legacy profile. The registry only edits AI Studio's own records:

  - an EXTERNAL profile references an existing CODEX_HOME that AI Studio never rewrites;
  - a MANAGED profile's CODEX_HOME was created by AI Studio (a new folder, see app.config.codex_config);
  - a profile that a resumable run is bound to cannot be deleted, a CODEX_HOME folder is never deleted, and
    changes only ever apply to NEW runs (running/resumable runs keep their snapshot);
  - secret values are never stored: extra env entries whose names look like credentials are refused - list the
    variable NAME under required env instead.
  - V5.1: every profile has a worker runtime (codex | cline, app.workers.runtimes). Runtime-specific checks live in
    the runtime; a profile's runtime is fixed at creation (moving to another runtime = a new, explicit profile).

Mutations change the Settings object; the caller persists it (SettingsStore.save) and applies it to the engine.
"""
from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from app.config import codex_config
from app.config.settings import Settings
from app.workers import launcher
from app.workers.profiles import (
    EXECUTABLE_POLICIES, LEGACY_PROFILE_ID, MODE_EXTERNAL, MODE_MANAGED, RUNTIME_CLINE, RUNTIME_CODEX, WorkerProfile,
    is_secret_name, legacy_profile, same_path, valid_env_name,
)
from app.workers.runtimes import get_runtime

EDITABLE_FIELDS = ("display_name", "codex_executable", "codex_home", "extra_env", "required_env", "pass_model_flag",
                   "model", "extra_args", "provider", "base_url", "context_window", "notes", "enabled",
                   "cline_executable", "cline_config_dir")
# Fields whose change alters the launch: a smoke-test verification does not survive such an edit.
_LAUNCH_FIELDS = ("codex_executable", "codex_home", "cline_executable", "cline_config_dir", "provider", "model")
_UNSAFE_ARG_MARKERS = ("dangerously", "bypass", "--yolo", "skip-permissions")


class ProfileError(ValueError):
    pass


@dataclass
class ValidationReport:
    profile_id: str
    ok: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        head = "OK" if self.ok else "NOT USABLE"
        parts = [f"{self.profile_id}: {head}"] + [f"error: {e}" for e in self.errors] + \
                [f"warning: {w}" for w in self.warnings]
        return "\n".join(parts)


def slugify(text: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", (text or "").lower()).strip("-")
    return s[:48] or "profile"


class WorkerProfileRegistry:
    def __init__(self, settings: Settings):
        self.settings = settings

    # ------------------------------------------------------------ queries

    @property
    def _records(self) -> list[dict[str, Any]]:
        return self.settings.worker_profiles.profiles

    def legacy(self) -> WorkerProfile:
        return legacy_profile(self.settings.worker)

    def registered(self) -> list[WorkerProfile]:
        return [WorkerProfile.from_dict(r) for r in self._records if r.get("id")]

    def profiles(self) -> list[WorkerProfile]:
        return [self.legacy()] + self.registered()

    def get(self, profile_id: str) -> WorkerProfile | None:
        if profile_id == LEGACY_PROFILE_ID:
            return self.legacy()
        return next((p for p in self.registered() if p.id == profile_id), None)

    def require(self, profile_id: str) -> WorkerProfile:
        p = self.get(profile_id)
        if p is None:
            raise ProfileError(f"unknown worker profile: {profile_id}")
        return p

    def default_id(self) -> str:
        return self.settings.worker_profiles.default_profile_id or LEGACY_PROFILE_ID

    def default_profile(self) -> WorkerProfile | None:
        return self.get(self.default_id())

    def managed_root(self) -> str:
        return self.settings.worker_profiles.managed_root or codex_config.default_managed_root()

    # ------------------------------------------------------------ mutations

    def _new_id(self, wanted: str) -> str:
        base = slugify(wanted)
        if base == LEGACY_PROFILE_ID:
            base += "-copy"
        taken = {p.id for p in self.profiles()}
        pid, n = base, 2
        while pid in taken:
            pid, n = f"{base}-{n}", n + 1
        return pid

    def _store(self, p: WorkerProfile) -> WorkerProfile:
        p.updated_at = time.time()
        rec = p.to_dict()
        for i, r in enumerate(self._records):
            if r.get("id") == p.id:
                self._records[i] = rec
                return p
        self._records.append(rec)
        return p

    @staticmethod
    def _check_fields(p: WorkerProfile) -> None:
        if not p.display_name.strip():
            raise ProfileError("a profile needs a display name")
        if get_runtime(p.runtime) is None or p.runtime not in launcher.RUNTIME_ADAPTERS:
            raise ProfileError(f"unknown worker runtime {p.runtime!r}")
        if p.executable_policy not in EXECUTABLE_POLICIES:
            raise ProfileError(f"unknown executable policy {p.executable_policy!r}")
        for k in p.extra_env:
            if not valid_env_name(k):
                raise ProfileError(f"invalid environment variable name: {k!r}")
            if k.upper() == "CODEX_HOME":
                raise ProfileError("CODEX_HOME is the profile's own field, not an extra env entry")
            if is_secret_name(k):
                raise ProfileError(f"{k} looks like a secret. AI Studio never stores secret values: set it in your "
                                   f"user environment and list only its NAME under required env variables")
        for k in p.required_env:
            if not valid_env_name(k):
                raise ProfileError(f"invalid required environment variable name: {k!r}")
        for a in p.extra_args:
            if any(m in a.lower() for m in _UNSAFE_ARG_MARKERS):
                raise ProfileError(f"refusing unsafe extra argument: {a}")

    def _fill_metadata(self, p: WorkerProfile) -> None:
        """Display metadata from config.toml where the profile has none (config.toml stays authoritative)."""
        info = codex_config.inspect_home(p.codex_home)
        p.provider = p.provider or info.provider
        p.model = p.model or info.model
        p.base_url = p.base_url or info.base_url
        p.context_window = p.context_window or info.context_window
        for name in info.env_refs:
            if name not in p.required_env:
                p.required_env.append(name)

    def register_existing(self, codex_home: str, codex_executable: str, display_name: str = "", *,
                          profile_id: str = "", extra_env: Mapping[str, str] | None = None,
                          required_env: Iterable[str] = (), notes: str = "") -> WorkerProfile:
        """Reference an existing CODEX_HOME (EXTERNAL: AI Studio never rewrites its config.toml)."""
        home = os.path.expandvars(codex_home.strip())
        if not Path(home).is_dir():
            raise ProfileError(f"CODEX_HOME does not exist: {home}")
        if same_path(home, launcher.normal_codex_home()):
            raise ProfileError("the normal ~/.codex profile cannot be used as a worker profile")
        name = display_name.strip() or Path(home).name.lstrip(".")
        pid = profile_id.strip() or self._new_id(name)
        if self.get(pid) is not None:
            raise ProfileError(f"a profile with id {pid!r} already exists")
        if slugify(pid) != pid:
            raise ProfileError(f"invalid profile id {pid!r} (use lowercase letters, digits and dashes)")
        now = time.time()
        p = WorkerProfile(id=pid, display_name=name, runtime=RUNTIME_CODEX, codex_executable=codex_executable.strip(),
                          codex_home=home, mode=MODE_EXTERNAL, extra_env=dict(extra_env or {}),
                          required_env=[n for n in required_env if n], notes=notes, executable_policy="codex",
                          created_at=now)
        self._fill_metadata(p)
        self._check_fields(p)
        return self._store(p)

    def create_managed(self, display_name: str, codex_executable: str, *, template_home: str = "",
                       config_text: str = "", extra_env: Mapping[str, str] | None = None,
                       required_env: Iterable[str] = (), notes: str = "") -> WorkerProfile:
        """A NEW profile with its own new CODEX_HOME under the managed root (copied from a template home's
        config.toml, or written from `config_text`)."""
        pid = self._new_id(display_name)
        home = str(Path(self.managed_root()) / pid)
        p = WorkerProfile(id=pid, display_name=display_name.strip(), codex_executable=codex_executable.strip(),
                          codex_home=home, mode=MODE_MANAGED, extra_env=dict(extra_env or {}),
                          required_env=[n for n in required_env if n], notes=notes, executable_policy="codex",
                          created_at=time.time())
        self._check_fields(p)
        try:
            codex_config.create_managed_home(home, template_home=template_home, config_text=config_text)
        except codex_config.CodexConfigError as exc:
            raise ProfileError(str(exc)) from exc
        self._fill_metadata(p)
        return self._store(p)

    def create_cline(self, display_name: str, cline_executable: str, provider: str, model: str, *,
                     profile_id: str = "", cline_config_dir: str = "", extra_env: Mapping[str, str] | None = None,
                     required_env: Iterable[str] = (), extra_args: Iterable[str] = (), notes: str = "") -> WorkerProfile:
        """A Cline profile (runtime "cline"). It has no CODEX_HOME: Cline keeps its provider credentials in its own
        configuration (`cline auth`), the profile selects the provider and model (-P / -m) and may point at another
        Cline configuration directory (--config). Nothing of Cline's configuration is written."""
        name = display_name.strip()
        pid = profile_id.strip() or self._new_id(name or "cline")
        if self.get(pid) is not None:
            raise ProfileError(f"a profile with id {pid!r} already exists")
        if slugify(pid) != pid:
            raise ProfileError(f"invalid profile id {pid!r} (use lowercase letters, digits and dashes)")
        p = WorkerProfile(id=pid, display_name=name, runtime=RUNTIME_CLINE, cline_executable=cline_executable.strip(),
                          cline_config_dir=cline_config_dir.strip(), provider=provider.strip(), model=model.strip(),
                          pass_model_flag=True, extra_env=dict(extra_env or {}),
                          required_env=[n for n in required_env if n], extra_args=[a for a in extra_args if a],
                          notes=notes, mode=MODE_EXTERNAL, executable_policy="codex", created_at=time.time())
        self._check_fields(p)
        return self._store(p)

    def duplicate(self, profile_id: str, display_name: str = "", *, copy_home: bool = False) -> WorkerProfile:
        """A new profile from an existing one. copy_home=False references the same CODEX_HOME (as EXTERNAL);
        copy_home=True creates a new MANAGED home with a copy of its config.toml (never sessions or logs)."""
        src = self.require(profile_id)
        name = display_name.strip() or f"{src.name} (copy)"
        if copy_home and src.runtime != RUNTIME_CODEX:
            raise ProfileError("only Codex profiles have a CODEX_HOME to copy")
        if copy_home:
            p = self.create_managed(name, src.codex_executable, template_home=src.codex_home,
                                    extra_env=src.extra_env, required_env=src.required_env, notes=src.notes)
        else:
            p = WorkerProfile.from_dict({**src.to_dict(), "id": self._new_id(name), "display_name": name,
                                         "mode": MODE_EXTERNAL, "created_at": time.time()})
            if src.is_legacy:
                # The legacy secret rule is not enforced retroactively on V4 settings, but a new profile never
                # carries a secret value: such entries must become required env NAMES.
                p.extra_env = {k: v for k, v in src.extra_env.items() if not is_secret_name(k)}
                p.required_env += [k for k in src.extra_env if is_secret_name(k) and k not in p.required_env]
            self._check_fields(p)
            self._store(p)
        # A copy starts unverified unless it launches exactly what was verified.
        if not p.is_verified():
            p.verified_executable, p.verified_codex_home, p.verified_at = "", "", 0.0
            self._store(p)
        return p

    def update(self, profile_id: str, **changes: Any) -> WorkerProfile:
        """Edit a profile's registry record. Never touches its config.toml. Existing runs keep their snapshot."""
        unknown = set(changes) - set(EDITABLE_FIELDS)
        if unknown:
            raise ProfileError(f"not editable: {', '.join(sorted(unknown))}")
        if profile_id == LEGACY_PROFILE_ID:
            return self._update_legacy(changes)
        p = self.require(profile_id)
        if "enabled" in changes and not changes["enabled"] and p.id == self.default_id():
            raise ProfileError("the default profile cannot be disabled; choose another default first")
        for k, v in changes.items():
            setattr(p, k, v)
        self._check_fields(p)
        # The full-test record is kept: it is bound to the executable + launch target that passed, so after an edit
        # of a launch field (_LAUNCH_FIELDS) the profile is simply unverified for NEW runs, while runs locked to the
        # old snapshot stay resumable. Only a failing full test revokes it (record_verification).
        return self._store(p)

    def _update_legacy(self, changes: dict[str, Any]) -> WorkerProfile:
        """The legacy profile IS Settings.worker: editing it edits the V4 worker settings."""
        w = self.settings.worker
        mapping = {"codex_executable": "executable", "codex_home": "codex_home", "extra_env": "extra_env",
                   "pass_model_flag": "pass_model_flag", "model": "model", "extra_args": "extra_args",
                   "base_url": "ninfer_base_url"}
        refused = [k for k in changes if k not in mapping]
        if refused:
            raise ProfileError("the legacy profile mirrors the V4 worker settings; these cannot be changed on it: "
                               + ", ".join(sorted(refused)))
        for k, v in changes.items():
            setattr(w, mapping[k], v)
        return self.legacy()

    def rename(self, profile_id: str, display_name: str) -> WorkerProfile:
        if profile_id == LEGACY_PROFILE_ID:
            raise ProfileError("the legacy profile cannot be renamed")
        return self.update(profile_id, display_name=display_name)

    def set_enabled(self, profile_id: str, enabled: bool) -> WorkerProfile:
        if profile_id == LEGACY_PROFILE_ID:
            raise ProfileError("the legacy profile is always available (V4 runs may depend on it)")
        return self.update(profile_id, enabled=enabled)

    def set_default(self, profile_id: str) -> None:
        """The profile NEW runs lock at their first START. Existing runs are never affected."""
        p = self.require(profile_id)
        if not p.enabled:
            raise ProfileError(f"{p.name} is disabled")
        self.settings.worker_profiles.default_profile_id = p.id

    def delete(self, profile_id: str, in_use: Iterable[str] = ()) -> WorkerProfile:
        """Remove the registry record only (the CODEX_HOME folder is never deleted). Refused for the legacy and the
        default profile and while a resumable run is bound to the profile."""
        if profile_id == LEGACY_PROFILE_ID:
            raise ProfileError("the legacy profile cannot be deleted")
        p = self.require(profile_id)
        if p.id == self.default_id():
            raise ProfileError("the default profile cannot be deleted; choose another default first")
        if p.id in set(in_use):
            raise ProfileError(f"{p.name} is used by a run that can still be resumed; finish or abandon that run "
                               "first (its worker would otherwise become unavailable)")
        self.settings.worker_profiles.profiles = [r for r in self._records if r.get("id") != p.id]
        return p

    def record_verification(self, profile_id: str, ok: bool, executable: str = "", codex_home: str = "",
                            at: float | None = None) -> None:
        """A passing full test binds the verification to the exact executable + launch target (Codex: CODEX_HOME;
        Cline: provider/model/configuration - see WorkerProfile.launch_target); a failing one revokes it (like the V4
        worker smoke test)."""
        if profile_id == LEGACY_PROFILE_ID:
            w = self.settings.worker
            w.verified_executable, w.verified_codex_home, w.verified_at = (
                (executable, codex_home, at or time.time()) if ok else ("", "", 0.0))
            return
        p = self.require(profile_id)
        p.verified_executable, p.verified_codex_home, p.verified_at = (
            (executable, codex_home, at or time.time()) if ok else ("", "", 0.0))
        self._store(p)

    # ------------------------------------------------------------ validation (quick; nothing is launched)

    def validate(self, profile_id: str, environ: Mapping[str, str] | None = None) -> ValidationReport:
        p = self.get(profile_id)
        if p is None:
            return ValidationReport(profile_id, False, [f"unknown worker profile: {profile_id}"])
        errors: list[str] = []
        warnings: list[str] = []
        if not p.enabled:
            errors.append("the profile is disabled")
        runtime = get_runtime(p.runtime)
        if runtime is None:
            errors.append(f"unknown worker runtime {p.runtime!r}")
            return ValidationReport(p.id, False, errors, warnings, {})
        r_errors, r_warnings, info_d = runtime.profile_problems(p, self.profiles())
        errors += r_errors
        errors += launcher.env_problems(p.required_env, environ)
        warnings += r_warnings
        secret_keys = [k for k in p.extra_env if is_secret_name(k)]
        if secret_keys:
            warnings.append(f"extra env {', '.join(secret_keys)} looks like a secret; move it to your user "
                            "environment and list only the name (values are redacted from logs and snapshots)")
        if not p.is_verified():
            warnings.append("not verified yet: run the full Test before using it for a REAL run")
        info_d = {"runtime": p.runtime, **info_d}
        return ValidationReport(p.id, not errors, errors, warnings, info_d)

    def ready_problems(self, p: WorkerProfile, environ: Mapping[str, str] | None = None) -> list[str]:
        """What prevents locking `p` for a NEW run (quick checks plus the smoke-test verification gate)."""
        rep = self.validate(p.id, environ)
        problems = list(rep.errors)
        if not problems and not p.is_verified():
            target = "CODEX_HOME" if p.runtime == RUNTIME_CODEX else "configuration"
            problems.append(f"the profile is not verified for this executable and {target} - run its full Test "
                            "(worker smoke test) first. REAL worker mode stays disabled until then")
        return problems
