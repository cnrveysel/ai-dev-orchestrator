"""Evidence freshness: what a step ran against, and whether that still holds.

Every executed step is stamped with its provenance:
  signature      capability definition + arguments + markers + the resources it ran with
  capability_fp  the capability's definition (provider, key, kind, declared details/constraints, provider
                 version): a changed definition makes earlier evidence stale
  scope          repository  - the step may depend on ANY authored file (commands, runtimes, engines): it
                               is fresh only while the whole repository fingerprint is unchanged
                 files       - the provider knows exactly which files the step read (file inspection): it
                               is fresh while those files' contents are unchanged
  repo_fp        the repository fingerprint when it ran (git snapshot tree, or a bounded manifest)

Conservative policy (correctness over reuse): a step is FRESH only when this can be shown; with an
unknown fingerprint a step is never reused across attempts. Stale evidence never counts as SATISFIED.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any

SCOPE_REPOSITORY, SCOPE_FILES = "repository", "files"
MAX_INPUT_BYTES = 8 * 1024 * 1024


def _sha(obj: Any) -> str:
    return hashlib.sha1(json.dumps(obj, sort_keys=True, default=str).encode("utf-8")).hexdigest()[:16]


def capability_fp(cap: Any) -> str:
    return _sha({"provider": cap.provider, "key": cap.key, "kind": cap.kind, "details": cap.details,
                 "constraints": cap.constraints, "version": (cap.provenance or {}).get("provider_version", "")})


def step_signature(cap_fp: str, step: Any, resource_fps: list[str]) -> str:
    return _sha({"cap": cap_fp, "args": list(step.args), "expect": step.expect_output, "fail": step.fail_output,
                 "resources": sorted(resource_fps)})


def file_inputs(repo_path: str, rels: list[str]) -> dict[str, str]:
    """Content hashes of the files a step reads (missing / unreadable are recorded as such)."""
    out: dict[str, str] = {}
    root = os.path.abspath(repo_path)
    for rel in rels:
        full = os.path.abspath(os.path.join(root, *str(rel).replace("\\", "/").split("/")))
        if not full.startswith(root + os.sep):
            out[rel] = "outside"
            continue
        try:
            if not os.path.isfile(full):
                out[rel] = "missing"
                continue
            h = hashlib.sha1()
            with open(full, "rb") as f:
                h.update(f.read(MAX_INPUT_BYTES))
            out[rel] = h.hexdigest()[:16]
        except OSError:
            out[rel] = "unreadable"
    return out


def make_provenance(*, cap: Any, cap_fp: str, signature: str, scope: str, repo_fp: str, repo_fp_kind: str,
                    inputs: dict[str, str] | None, attempt: int) -> dict[str, Any]:
    prov: dict[str, Any] = {"signature": signature, "capability_fp": cap_fp, "capability_key": cap.key,
                            "capability_id": cap.id, "provider": cap.provider,
                            "provider_version": (cap.provenance or {}).get("provider_version", ""),
                            "scope": scope, "repo_fp": repo_fp, "repo_fp_kind": repo_fp_kind,
                            "attempt": attempt, "created_at": time.time()}
    if inputs is not None:
        prov["inputs"] = dict(inputs)
    return prov


def check(prov: dict[str, Any] | None, *, cap_fp: str, repo_fp: str,
          inputs: dict[str, str] | None) -> tuple[bool | None, str]:
    """True = fresh, False = stale (with the reason), None = unknown (nothing proves it either way)."""
    if not prov:
        return None, "recorded without provenance"
    if prov.get("capability_fp") and cap_fp and prov["capability_fp"] != cap_fp:
        return False, "the capability's definition changed"
    if prov.get("scope") == SCOPE_FILES:
        want = prov.get("inputs") or {}
        if inputs is None:
            return None, "the step's inputs are unknown"
        changed = sorted(k for k in set(want) | set(inputs) if want.get(k) != inputs.get(k))
        return (False, "input changed: " + ", ".join(changed[:4])) if changed else (True, "")
    old = str(prov.get("repo_fp") or "")
    if not old or not repo_fp:
        return None, "no repository fingerprint"
    return (True, "") if old == repo_fp else (False, "the repository changed after it ran")


def result_provenance(steps: list[Any], *, repo_fp: str, repo_fp_kind: str, generation: int = 0) -> dict[str, Any]:
    """Provenance of a requirement result, aggregated from the steps that produced it."""
    scopes = {(s.provenance or {}).get("scope") for s in steps if s.provenance}
    inputs: dict[str, str] = {}
    for s in steps:
        inputs.update((s.provenance or {}).get("inputs") or {})
    scope = SCOPE_FILES if scopes == {SCOPE_FILES} else SCOPE_REPOSITORY
    reused = sorted({int(s.reused_from) for s in steps if s.reused_from})
    reused_steps = {s.key: int(s.reused_from) for s in steps if s.reused_from}
    prov: dict[str, Any] = {"repo_fp": repo_fp, "repo_fp_kind": repo_fp_kind, "scope": scope,
                            "collected_at": time.time(), "plan_generation": generation,
                            "capability_fps": sorted({(s.provenance or {}).get("capability_fp", "")
                                                      for s in steps if s.provenance} - {""})}
    if scope == SCOPE_FILES:
        prov["inputs"] = inputs
    if reused:
        prov["reused_from"] = reused[0]
        prov["reused_steps"] = reused_steps  # which steps were NOT run now (and the attempt they ran in)
    return prov


def refresh(results: list[dict[str, Any]], *, repo_fp: str, repo_path: str) -> list[tuple[str, str]]:
    """Re-check stored results against the CURRENT repository. Stale ones are demoted in place (see
    model.mark_stale). Returns [(requirement id, reason)] of the results that went stale now."""
    from app.evidence.model import mark_stale  # local: model must stay import-light
    went: list[tuple[str, str]] = []
    for r in results:
        prov = r.get("provenance") or {}
        if not prov or r.get("source") == "plan" or r.get("freshness") == "stale":
            continue
        inputs = file_inputs(repo_path, sorted(prov.get("inputs") or {})) if prov.get("scope") == SCOPE_FILES else None
        fresh, why = check({**prov, "capability_fp": ""}, cap_fp="", repo_fp=repo_fp, inputs=inputs)
        if fresh is False and mark_stale(r, why):
            went.append((str(r.get("requirement_id")), why))
    return went
