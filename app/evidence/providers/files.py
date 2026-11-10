"""Read-only repository file inspection: does a file exist, does it contain a literal text.

This only ever shows that a file exists or contains some text. It never observes runtime, visual
or interaction behaviour (file_inspection is a non-observational kind), so for such requirements
its success is supporting evidence at most. Its results depend on exactly the file they read
(freshness scope "files"): they stay fresh while that file is unchanged, whatever else changes.

Generated files (build output, logs, caches - a generated directory, or ignored by the repository) are
different: their content comes from whatever process wrote them last - the worker's own run, an earlier
attempt, an earlier evidence collection. Such a check only counts when the file was written DURING this
evidence collection (i.e. by an evidence step that ran now); an older one is not an observation of this
run and the step is not trusted (a stale "Result={Success}" in a log proves nothing). Its freshness scope
is the whole repository: what a generated file says depends on everything that produced it.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time

from app.evidence.freshness import SCOPE_FILES, SCOPE_REPOSITORY
from app.evidence.model import CONFIRMED, ERROR, FAILED, PASSED, UNAVAILABLE, Capability, PlanStep
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext, ExecContext, StepOutcome
from app.integrations import path_filter

MAX_READ_BYTES = 2 * 1024 * 1024
ACTIONS = {"exists": 2, "contains": 3}


def _relative_ok(rel: str) -> bool:
    p = rel.replace("\\", "/")
    if not p or p.startswith("/") or re.match(r"^[A-Za-z]:", p):
        return False
    parts = [x for x in p.split("/") if x not in ("", ".")]
    return bool(parts) and ".." not in parts and parts[0].lower() != ".git"


def _inside(repo: str, rel: str) -> str:
    """Absolute path of a repository-relative path, or "" if it escapes the repository or is git metadata."""
    if not _relative_ok(rel):
        return ""
    root = os.path.abspath(repo)
    full = os.path.abspath(os.path.join(root, rel))
    return full if full.startswith(root + os.sep) else ""


def generated_reason(repo: str, rel: str) -> str:
    """Why a repository path holds generated content ("" = authored content)."""
    rel = rel.replace("\\", "/")
    if path_filter.is_generated(rel):
        return "generated output (a build, log or cache location)"
    if not repo or not os.path.exists(os.path.join(repo, ".git")):
        return ""
    try:  # read-only; exit 0 = the repository ignores the path
        r = subprocess.run(["git", "-C", repo, "check-ignore", "-q", "--", rel], capture_output=True, timeout=5,
                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0) if sys.platform == "win32" else 0)
    except (OSError, subprocess.SubprocessError):
        return ""
    return "ignored by the repository (not authored content)" if r.returncode == 0 else ""


class RepositoryFileProvider(CapabilityProvider):
    name = "repository_files"

    async def discover(self, ctx: DiscoveryContext) -> list[Capability]:
        readable = os.path.isdir(ctx.repo_path) and os.access(ctx.repo_path, os.R_OK)
        return [Capability(
            id="", kind="file_inspection", name="repository file inspection",
            description="read-only check inside the repository: args [\"exists\", <path>] or "
                        "[\"contains\", <path>, <literal text>]",
            source="environment", confidence=CONFIRMED if readable else UNAVAILABLE, provider=self.name,
            key="files:repository", constraints={"read_only": True, "accepts_args": True, "max_args": 3,
                                                 "lifecycle": "oneshot", "timeout_seconds": 10},
            reason="probed: the repository is readable" if readable else "the repository is not readable")]

    def validate(self, cap: Capability, step: PlanStep) -> str:
        a = step.args
        if not a or a[0] not in ACTIONS or len(a) != ACTIONS[a[0]]:
            return "file inspection takes [\"exists\", <path>] or [\"contains\", <path>, <literal text>]"
        if len(a[1]) > 300 or not _relative_ok(a[1]):
            return f"path {a[1]!r} must be a repository-relative path outside .git"
        if len(a) == 3 and not 0 < len(a[2]) <= 200:
            return "the literal text must be 1-200 characters"
        return ""

    def input_scope(self, cap: Capability, step: PlanStep, repo_path: str = "") -> str:
        if len(step.args) >= 2 and generated_reason(repo_path, step.args[1]):
            return SCOPE_REPOSITORY
        return SCOPE_FILES

    def input_files(self, cap: Capability, step: PlanStep, repo_path: str = "") -> list[str]:
        return [step.args[1].replace("\\", "/")] if len(step.args) >= 2 else []

    async def execute(self, cap: Capability, step: PlanStep, ctx: ExecContext) -> StepOutcome:
        start = time.monotonic()
        why = self.validate(cap, step)
        full = _inside(ctx.repo_path, step.args[1]) if not why else ""
        if why or not full:
            return StepOutcome(UNAVAILABLE, reason=why or "path outside the repository")
        action, rel = step.args[0], step.args[1]
        if not os.path.isfile(full):
            return StepOutcome(FAILED, summary=f"{rel} does not exist", reason="missing file",
                               duration=time.monotonic() - start)
        generated = generated_reason(ctx.repo_path, rel)
        written = os.path.getmtime(full)
        if generated and ctx.collection_started and written < ctx.collection_started:
            when = time.strftime("%H:%M:%S", time.localtime(written))
            return StepOutcome(ERROR, summary=f"{rel} is {generated}, last written at {when} - before this evidence "
                                              "collection started (by the worker or an earlier run): not an observation "
                                              "of this run", reason="stale generated file", cause="strategy",
                               duration=time.monotonic() - start)
        note = " (written during this evidence collection)" if generated else ""
        if action == "exists":
            return StepOutcome(PASSED, summary=f"{rel} exists{note}", duration=time.monotonic() - start)
        with open(full, "rb") as f:
            text = f.read(MAX_READ_BYTES).decode("utf-8", errors="replace")
        found = step.args[2] in text
        return StepOutcome(PASSED if found else FAILED,
                           summary=f"{rel} {'contains' if found else 'does not contain'} {step.args[2]!r}{note}",
                           reason="" if found else "text not found", duration=time.monotonic() - start)
