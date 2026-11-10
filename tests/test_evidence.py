"""V4 pass 2: capability discovery, EVIDENCE_PLAN validation, bounded evidence execution, the
COLLECTING_EVIDENCE state, checkpoint / human-gate integration and crash safety.

Deterministic: mock orchestrator / mock workers (the conftest guard refuses any real Claude or
Codex process). Evidence commands are tiny Python scripts in throwaway git repositories.
"""
import asyncio
import json
import os
import shutil
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

import pytest

from app.evidence import procutil
from app.evidence.artifacts import ArtifactStore
from app.evidence.executor import EvidenceExecutor, EvidenceLimits
from app.evidence.lifecycle import ResourceManager
from app.evidence.model import (
    CONFIRMED, FAILED, INCONCLUSIVE, INFERRED, PASSED, RUNNING, SATISFIED, SKIPPED, TIMEOUT, UNAVAILABLE, Capability,
    CapabilitySnapshot, EvidencePlan, EvidenceState,
)
from app.evidence.planning import (
    EVIDENCE_PLAN_JSON_SCHEMA, EvidencePlanPacket, deterministic_plan, plan_fingerprint, validate_plan,
)
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext
from app.evidence.providers.command import classify_command
from app.evidence.registry import CapabilityRegistry, default_registry
from app.integrations.git import GitRepo
from app.models.attempt import VerificationResult
from app.models.phase import parse_phases
from app.models.task import Task, TaskStatus
from app.models.verification import VerificationContract
from app.orchestration.checkpoint import CheckpointContextLimits, build_checkpoint_context
from app.orchestration.providers.base import OrchestratorError
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.scheduler import Scheduler
from app.orchestration.state_machine import RunState
from app.persistence.database import Database
from app.workers.mock import MockWorker
from tests.test_pipeline_v2 import V2Harness, git, make_settings, phase

PY = sys.executable
S = RunState

PROBE = r'''
import os, pathlib, sys, time
mode = sys.argv[1] if len(sys.argv) > 1 else "ok"
counter = os.environ.get("EVIDENCE_COUNTER")
if counter:
    with open(counter, "a") as f:
        f.write(mode + "\n")
root = pathlib.Path(__file__).resolve().parent.parent
if mode == "ok":
    state = (root / "state.txt").read_text().strip()
    if state == "ok":
        print("interaction exercised")
        print("RUNTIME OK")
        sys.exit(0)
    print("FAIL: runtime broken (state=" + state + ")")
    sys.exit(1)
if mode == "fail":
    print("FAIL: runtime broken")
    sys.exit(1)
if mode == "slow":
    time.sleep(float(os.environ.get("EVIDENCE_SLOW") or (root / "slow.txt").read_text()))
    print("RUNTIME OK")
    sys.exit(0)
if mode == "sleepy":
    time.sleep(1.2)
    print("RUNTIME OK")
    sys.exit(0)
if mode == "noisy":
    for i in range(4000):
        print("line %05d %s" % (i, "x" * 50))
    print("RUNTIME OK")
    sys.exit(0)
if mode == "mutate":
    (root / "README.md").write_text("mutated by an evidence step\n")
    (root / "created_by_evidence.txt").write_text("x\n")
    print("RUNTIME OK")
    sys.exit(0)
print("unknown mode")
sys.exit(2)
'''
SERVICE = 'import time\nprint("booting", flush=True)\nprint("READY", flush=True)\nwhile True:\n    time.sleep(0.2)\n'
SERVICE_SILENT = 'import time\nprint("booting", flush=True)\nwhile True:\n    time.sleep(0.2)\n'
SERVICE_CRASH = 'import sys\nprint("boom", flush=True)\nsys.exit(3)\n'
BUILD_CMD = f'"{PY}" checks/build.py'
PROBE_CMD = f'"{PY}" checks/probe.py'
MANIFEST = {
    "commands": [
        {"name": "runtime probe", "command": PROBE_CMD, "kind": "runtime_behavior", "accepts_args": True,
         "arg_pattern": "[a-z]+", "max_args": 1, "timeout_seconds": 60,
         "description": "exercises the primary interaction in the running program"},
        {"name": "build", "command": BUILD_CMD, "kind": "build"}],
    "services": [
        {"name": "app service", "command": f'"{PY}" checks/service.py', "ready_output": "READY",
         "ready_timeout_seconds": 10},
        {"name": "silent service", "command": f'"{PY}" checks/service_silent.py', "ready_output": "READY",
         "ready_timeout_seconds": 1},
        {"name": "crashing service", "command": f'"{PY}" checks/service_crash.py', "ready_output": "READY",
         "ready_timeout_seconds": 5}],
}


@pytest.fixture
def erepo(tmp_path):
    r = tmp_path / "proj"
    (r / "checks").mkdir(parents=True)
    (r / ".aistudio").mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("readme\n")
    (r / "state.txt").write_text("ok\n")
    (r / "slow.txt").write_text("30\n")
    (r / "checks" / "probe.py").write_text(PROBE)
    (r / "checks" / "build.py").write_text('print("build ok")\n')
    (r / "checks" / "service.py").write_text(SERVICE)
    (r / "checks" / "service_silent.py").write_text(SERVICE_SILENT)
    (r / "checks" / "service_crash.py").write_text(SERVICE_CRASH)
    (r / ".aistudio" / "evidence.json").write_text(json.dumps(MANIFEST))
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


def req(rid, kind="runtime_behavior", required=True, goal=None):
    return {"id": rid, "kind": kind, "required": required, "goal": goal or f"{rid}: the feature works at runtime"}


def contract(reqs, gate=False):
    return {"success_dimensions": ["functional"], "evidence_requirements": reqs,
            "human_gate": {"allowed": gate, "reason": "perception" if gate else ""}}


def make_phase(reqs, gate=False, verification=None, pid="P1"):
    return parse_phases([{"id": pid, "title": "Feature", "description": "Build it", "subtasks": ["a"],
                          "acceptance_criteria": ["works"], "verification_commands": verification or [],
                          "verification_contract": contract(reqs, gate)}])[0]


def cap_id(snap, name):
    return next(c.id for c in snap.capabilities if c.name == name)


async def discover(repo, ph, ran=None, which=shutil.which, settings=None, **kw):
    reg = default_registry(which)
    ctx = DiscoveryContext(str(repo), ph, [ph], ran or {}, settings or make_settings().orchestration)
    return reg, await reg.discover(ctx, **kw)


def make_executor(reg, repo, tmp_path, git_repo=None, log=None, **limits):
    base = dict(wall_seconds=60, step_timeout=30, max_steps=12, max_output_bytes=16000, max_artifacts=24,
                max_artifact_bytes=1_000_000, max_retries=0, ready_timeout=10)
    base.update(limits)
    lim = EvidenceLimits(**base)
    store = ArtifactStore(str(tmp_path / "artifacts"), 1, "P1", 1, max_artifacts=lim.max_artifacts,
                          max_bytes=lim.max_artifact_bytes)
    return EvidenceExecutor(reg, lim, store, repo_path=str(repo), persist=lambda: None,
                            log=(log.append if log is not None else (lambda s: None)), git=git_repo)


def steps_of(state):
    return {k: v["status"] for k, v in state.steps.items()}


def statuses(bundle):
    return {r.requirement_id: r.status for r in bundle.results}


# ------------------------------------------------------------ capability discovery

def test_discovery_reports_confirmed_inferred_and_unavailable_honestly(erepo):
    (erepo / "package.json").write_text(json.dumps({"scripts": {
        "test": "jest", "build": "tsc", "lint": "eslint .", "start": "node server.js", "deploy": "firebase deploy"}}))
    ran_cmd, missing_cmd = BUILD_CMD, "definitely-missing-tool --check"
    ph = make_phase([req("E1")], verification=[ran_cmd, missing_cmd])
    s = make_settings(evidence_commands=["git status", "git clean -fdx"]).orchestration
    which = {"npm": "C:/fake/npm.cmd", "git": "C:/fake/git.exe"}.get
    ran = {ran_cmd: VerificationResult(ran_cmd, 0, "build ok")}
    reg, snap = asyncio.run(discover(erepo, ph, ran, which, s))
    by = {c.name: c for c in snap.capabilities}
    # The display name is shortened for long checkout paths. The key keeps the full command.
    ran_cap = next(c for c in snap.capabilities if c.key == f"command:{ran_cmd}")
    assert ran_cap.source == "phase_verification"
    assert (ran_cap.confidence, ran_cap.kind, ran_cap.reason) == (CONFIRMED, "build", "ran in this run: exit 0"), \
        "CONFIRMED only because it actually ran"
    assert by[f"phase verification: {missing_cmd}"].confidence == UNAVAILABLE
    assert "not found" in by[f"phase verification: {missing_cmd}"].reason
    probe = by["runtime probe"]
    assert (probe.confidence, probe.kind, probe.source) == (INFERRED, "runtime_behavior", "repository"), \
        "declared, program found, never run: INFERRED, not confirmed"
    svc = by["app service"]
    assert svc.confidence == INFERRED and svc.constraints["lifecycle"] == "service" and svc.kind == "runtime_launch"
    assert by["package script: test"].confidence == INFERRED and by["package script: test"].kind == "automated_test"
    assert by["package script: lint"].kind == "static_analysis" and by["package script: build"].kind == "build"
    assert by["package script: start"].confidence == UNAVAILABLE and "long-running" in by["package script: start"].reason
    assert by["package script: deploy"].confidence == UNAVAILABLE and "side effects" in by["package script: deploy"].reason
    assert by["configured: git status"].confidence == INFERRED
    assert by["configured: git clean -fdx"].confidence == UNAVAILABLE and "approval" in by["configured: git clean -fdx"].reason
    assert by["repository file inspection"].confidence == CONFIRMED
    assert "build" not in [c.name for c in snap.capabilities], "the manifest's build command is the same capability"
    assert len({c.id for c in snap.capabilities}) == len(snap.capabilities) and snap.capabilities[0].id == "cap-001"
    n = snap.counts()
    assert snap.summary().startswith(f"{len(snap.capabilities)} capabilities found ({n[CONFIRMED]} confirmed, "
                                     f"{n[INFERRED]} inferred, {n[UNAVAILABLE]} unavailable)")
    assert CapabilitySnapshot.from_dict(json.loads(json.dumps(snap.to_dict()))) == snap
    # no runner at all: declared scripts are explicitly unavailable, never assumed
    _, snap2 = asyncio.run(discover(erepo, ph, ran, lambda name: None, s))
    assert {c.name: c.reason for c in snap2.capabilities}["package script: test"] == "program 'npm' not found"
    assert classify_command("npx tsc --noEmit") == "static_analysis" and classify_command("node smoke.mjs") == \
        "automated_test" and classify_command("python tool.py") == "command_execution"


def test_discovery_is_bounded_and_isolates_broken_providers(erepo):
    class Slow(CapabilityProvider):
        name = "slow"

        async def discover(self, ctx):
            await asyncio.sleep(5)
            return []

        async def execute(self, cap, step, ctx):
            raise AssertionError

    class Broken(Slow):
        name = "broken"

        async def discover(self, ctx):
            raise RuntimeError("probe crashed")

    class Many(Slow):
        name = "many"

        async def discover(self, ctx):
            return [Capability("", "log_inspection", f"log {i}", confidence=CONFIRMED, key=f"many:{i}")
                    for i in range(5)]

    reg = CapabilityRegistry([Many(), Broken(), Slow()])
    ph = make_phase([req("E1")])
    t0 = time.monotonic()
    snap = asyncio.run(reg.discover(DiscoveryContext(str(erepo), ph), timeout=0.5, max_capabilities=3))
    assert time.monotonic() - t0 < 3, "one shared, hard time budget"
    assert snap.providers["many"] == "ok (5)" and snap.providers["broken"].startswith("error: probe crashed")
    assert snap.providers["slow"].startswith("timeout")
    assert len(snap.capabilities) == 3 and snap.truncated == 2 and "2 more not listed" in snap.summary()
    known = {"many:0": "confirmed earlier in this run (P1)"}

    class Inferred(Many):
        async def discover(self, ctx):
            return [Capability("", "log_inspection", "log 0", confidence=INFERRED, key="many:0")]
    reg2 = CapabilityRegistry([Inferred()])
    snap2 = asyncio.run(reg2.discover(DiscoveryContext(str(erepo), ph), known_confirmed=known))
    assert snap2.capabilities[0].confidence == CONFIRMED and "earlier in this run" in snap2.capabilities[0].reason


def _tree_state(repo: Path):
    files = {}
    for p in repo.rglob("*"):
        if ".git" in p.relative_to(repo).parts:
            continue
        if p.is_file():
            st = p.stat()
            files[str(p.relative_to(repo))] = (st.st_size, st.st_mtime_ns)
    status = subprocess.run(["git", "status", "--porcelain", "--untracked-files=all"], cwd=repo, capture_output=True,
                            text=True).stdout
    return files, status


def test_discovery_does_not_mutate_the_repository(erepo):
    (erepo / "package.json").write_text(json.dumps({"scripts": {"test": "node checks/probe.py mutate"}}))
    before = _tree_state(erepo)
    ph = make_phase([req("E1")], verification=[f"{PROBE_CMD} mutate"])  # would mutate IF it were run
    asyncio.run(discover(erepo, ph))
    assert _tree_state(erepo) == before, "discovery reads metadata only; it never runs or writes anything"


# ------------------------------------------------------------ evidence planning (validation)

def test_plan_can_only_use_discovered_capability_ids_and_keeps_required_evidence(erepo):
    ph = make_phase([req("E1"), req("E2"), req("E3", required=False), req("E4")])
    reg, snap = asyncio.run(discover(erepo, ph))
    probe, svc = cap_id(snap, "runtime probe"), cap_id(snap, "app service")
    files = cap_id(snap, "repository file inspection")
    raw = {"summary": "s", "limitations": ["no display"], "requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g1", "coverage": "direct", "resources": [svc, "cap-999"],
         "steps": [{"capability_id": probe, "args": ["ok"], "expect_output": "RUNTIME OK"},
                   {"capability_id": "cap-999"},  # invented
                   {"capability_id": probe, "command": "del /s /q C:\\"},  # free-form shell
                   {"capability_id": probe, "args": ["ok; del x"]},  # not the declared argument pattern
                   {"capability_id": svc},  # a service is not a step
                   {"capability_id": files, "args": ["exists", "../outside.txt"]}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g2", "required": False,  # cannot downgrade
         "steps": [{"capability_id": "cap-998"}]},
        {"requirement_id": "E9", "strategy": "execute", "goal": "invented requirement", "steps": []},
        {"requirement_id": "E3", "strategy": "human", "goal": "g3", "reason": "only a person can see it"}]}
    plan = validate_plan(raw, ph, snap, reg, max_steps=12)
    e1, e2, e3, e4 = (plan.get(x) for x in ("E1", "E2", "E3", "E4"))
    assert [s.capability_id for s in e1.steps] == [probe] and e1.steps[0].key == "E1.s1" and e1.resources == [svc]
    notes = " | ".join(e1.notes)
    for part in ["resource cap-999: unknown capability", "step 2: unknown capability 'cap-999'",
                 "step 3: unsupported field(s) command", "does not match the pattern",
                 "is a service: list it under `resources`", "must be a repository-relative path"]:
        assert part in notes, part
    assert e2.strategy == UNAVAILABLE.lower() and e2.required and "no valid step" in e2.reason, \
        "no usable step: unavailable, and still REQUIRED (the planner cannot downgrade it)"
    assert e3.strategy == "human" and not e3.required
    assert e4.strategy == "unavailable" and e4.required and e4.reason == "not planned by the evidence planner"
    assert any("unknown requirement 'E9'" in x for x in plan.validation_errors)
    assert plan.limitations == ["no display"] and plan.step_count() == 1
    budget = validate_plan({"requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": probe, "args": ["ok"]}, {"capability_id": probe, "args": ["fail"]}]},
        {"requirement_id": "E4", "strategy": "execute", "goal": "g", "steps": [{"capability_id": probe}]}]},
        ph, snap, reg, max_steps=1)
    assert budget.step_count() == 1 and any("budget" in n for n in budget.get("E1").notes)
    assert budget.get("E4").strategy == "unavailable" and "budget" in budget.get("E4").reason
    again = EvidencePlan.from_dict(json.loads(json.dumps(plan.to_dict())))
    assert again == plan, "the validated plan is persisted as-is"
    none = deterministic_plan(ph, "nothing usable")
    assert none.source == "deterministic" and {r.strategy for r in none.requirements} == {"unavailable"}
    assert all(r.required for r in none.requirements if r.requirement_id != "E3")
    assert plan_fingerprint(ph, snap) == plan_fingerprint(ph, snap)
    ph2 = make_phase([req("E1")])
    assert plan_fingerprint(ph2, snap) != plan_fingerprint(ph, snap), "a different contract needs a new plan"
    steps_schema = EVIDENCE_PLAN_JSON_SCHEMA["properties"]["requirements"]["items"]["properties"]["steps"]["items"]
    assert steps_schema["additionalProperties"] is False and "command" not in steps_schema["properties"]


def test_a_build_can_never_directly_prove_runtime_behaviour(erepo, tmp_path):
    ph = make_phase([req("E1")], verification=[BUILD_CMD])
    ran = {BUILD_CMD: VerificationResult(BUILD_CMD, 0, "build ok")}
    reg, snap = asyncio.run(discover(erepo, ph, ran))
    build = next(c.id for c in snap.capabilities if c.kind == "build")
    plan = validate_plan({"requirements": [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                            "coverage": "direct", "steps": [{"capability_id": build}]}]},
                         ph, snap, reg, 12)
    assert plan.get("E1").coverage == "indirect" and "does not observe runtime_behavior" in plan.get("E1").notes[0]
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path).run(state, plan, snap))
    r = bundle.results[0]
    assert steps_of(state) == {"E1.s1": PASSED} and r.status == INCONCLUSIVE and r.confidence == "proxy"
    assert "supporting evidence only" in r.summary and bundle.missing_required() == [r], \
        "a green build is recorded as what it is - never as runtime evidence"


# ------------------------------------------------------------ evidence execution

def _plan(ph, snap, reg, entries, max_steps=12):
    return validate_plan({"summary": "s", "requirements": entries}, ph, snap, reg, max_steps)


def _probe_step(snap, mode, **kw):
    return dict({"capability_id": cap_id(snap, "runtime probe"), "args": [mode]}, **kw)


def test_executor_distinguishes_satisfied_failed_unavailable_inconclusive_and_timeout(erepo, tmp_path):
    ph = make_phase([req(f"E{i}") for i in range(1, 7)], gate=True)
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "ok", expect_output="RUNTIME OK")]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "fail")]},
        {"requirement_id": "E3", "strategy": "human", "goal": "g", "reason": "needs eyes"},
        {"requirement_id": "E4", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "slow")]},
        {"requirement_id": "E5", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "ok", expect_output="NOT THERE")]},
        {"requirement_id": "E6", "strategy": "execute", "goal": "g", "coverage": "indirect",
         "steps": [_probe_step(snap, "ok")]}])
    state = EvidenceState("P1", attempt=1)
    t0 = time.monotonic()
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, step_timeout=1).run(state, plan, snap))
    assert time.monotonic() - t0 < 20
    assert statuses(bundle) == {"E1": SATISFIED, "E2": FAILED, "E3": UNAVAILABLE, "E4": INCONCLUSIVE,
                                "E5": FAILED, "E6": INCONCLUSIVE}
    assert steps_of(state)["E4.s1"] == TIMEOUT and "timed out after 1s" in state.steps["E4.s1"]["summary"]
    by = {r.requirement_id: r for r in bundle.results}
    assert by["E1"].confidence == "observed" and "RUNTIME OK" in by["E1"].summary
    assert by["E3"].source == "plan" and "human observation needed" in by["E3"].summary
    assert "expected output 'NOT THERE' did not appear" in by["E5"].summary
    assert state.steps["E2.s1"]["exit_code"] == 1 and "FAIL: runtime broken" in state.steps["E2.s1"]["output_tail"]
    arts = {a.id: a for a in bundle.artifacts}
    assert arts and all(Path(a.path_or_ref).is_file() for a in arts.values())
    assert all(str(tmp_path / "artifacts") in a.path_or_ref and str(erepo) not in a.path_or_ref for a in arts.values()), \
        "artifacts live outside the target repository"
    assert by["E1"].artifacts and by["E1"].artifacts[0] in arts
    probe = snap.get(cap_id(snap, "runtime probe"))
    assert probe.confidence == CONFIRMED and "executed during evidence collection" in probe.reason
    assert bundle.missing_required() and not (erepo / "created_by_evidence.txt").exists()


def test_step_output_is_truncated_and_bounded(erepo, tmp_path):
    ph = make_phase([req("E1")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                  "steps": [_probe_step(snap, "noisy", expect_output="RUNTIME OK")]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, max_output_bytes=4000, max_artifact_bytes=3000)
                         .run(state, plan, snap))
    rec = state.steps["E1.s1"]
    assert rec["status"] == PASSED and rec["truncated"], "the tail (with the success marker) is kept"
    assert len(rec["output_tail"]) <= 1500 and rec["output_tail"].rstrip().endswith("RUNTIME OK")
    art = bundle.artifacts[0]
    assert art.truncated and os.path.getsize(art.path_or_ref) <= 3000 + 100
    assert len(json.dumps(state.to_dict())) < 20000, "the run state holds summaries, never megabytes of output"


def test_step_count_and_wall_time_budgets(erepo, tmp_path):
    ph = make_phase([req("E1"), req("E2")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "ok"), _probe_step(snap, "ok"), _probe_step(snap, "ok")]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "ok")]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, max_steps=2).run(state, plan, snap))
    assert steps_of(state) == {"E1.s1": PASSED, "E1.s2": PASSED, "E1.s3": SKIPPED, "E2.s1": SKIPPED}
    assert statuses(bundle) == {"E1": INCONCLUSIVE, "E2": SKIPPED} and "step budget" in state.steps["E2.s1"]["reason"]
    plan = _plan(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                  "steps": [_probe_step(snap, "sleepy"), _probe_step(snap, "sleepy")]}])
    state = EvidenceState("P1", attempt=1)
    t0 = time.monotonic()
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, wall_seconds=1.0).run(state, plan, snap))
    assert time.monotonic() - t0 < 10
    assert steps_of(state) == {"E1.s1": TIMEOUT, "E1.s2": SKIPPED}
    assert "wall-time budget" in state.steps["E1.s1"]["reason"] and "wall-time budget" in state.steps["E1.s2"]["reason"]
    assert statuses(bundle)["E1"] == INCONCLUSIVE and statuses(bundle)["E2"] == UNAVAILABLE


def test_service_lifecycle_starts_probes_captures_and_always_stops(erepo, tmp_path):
    ph = make_phase([req("E1"), req("E2"), req("E3")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "resources": [cap_id(snap, "app service")], "steps": [_probe_step(snap, "ok", expect_output="RUNTIME OK")]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "resources": [cap_id(snap, "silent service")], "steps": [_probe_step(snap, "ok")]},
        {"requirement_id": "E3", "strategy": "execute", "goal": "g", "coverage": "direct",
         "resources": [cap_id(snap, "crashing service")], "steps": [_probe_step(snap, "ok")]}])
    state = EvidenceState("P1", attempt=1)
    logs = []
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, log=logs).run(state, plan, snap))
    assert statuses(bundle) == {"E1": SATISFIED, "E2": INCONCLUSIVE, "E3": FAILED}
    assert steps_of(state) == {"E1.s1": PASSED, "E2.s1": SKIPPED, "E3.s1": SKIPPED}
    by = {r.requirement_id: r for r in bundle.results}
    assert "not ready within 1s" in by["E2"].summary and "exited with code 3" in by["E3"].summary
    assert [r["state"] for r in state.resources] == ["STOPPED"] * 3
    assert all(r["pid"] and not procutil.is_alive(r["pid"]) for r in state.resources), "nothing is leaked"
    runtime = [a for a in bundle.artifacts if a.id in by["E1"].artifacts and "while E1 was observed" in a.description]
    assert runtime and "READY" in Path(runtime[0].path_or_ref).read_text()
    assert any("resource res-1" in x and "ready" in x for x in logs)


def test_a_cancelled_collection_stops_its_resources_and_the_interrupted_step_is_never_trusted(erepo, tmp_path,
                                                                                               monkeypatch):
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("EVIDENCE_COUNTER", str(counter))
    ph = make_phase([req("E1"), req("E2")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [_probe_step(snap, "ok", expect_output="RUNTIME OK")]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "resources": [cap_id(snap, "app service")], "steps": [_probe_step(snap, "slow", expect_output="RUNTIME OK")]}])
    state = EvidenceState("P1", attempt=1)

    async def interrupted():
        task = asyncio.ensure_future(make_executor(reg, erepo, tmp_path).run(state, plan, snap))
        while state.inflight != "E2.s1":
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await task
    asyncio.run(interrupted())
    assert state.steps["E2.s1"]["status"] == RUNNING and state.inflight == "E2.s1"
    assert state.resources[0]["state"] == "STOPPED" and not procutil.is_alive(state.resources[0]["pid"])
    assert not state.bundle, "an interrupted collection produces no evidence"
    (erepo / "slow.txt").write_text("0.1\n")
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path).run(state, plan, snap))  # after the "restart"
    assert statuses(bundle) == {"E1": SATISFIED, "E2": SATISFIED}
    assert counter.read_text().split() == ["ok", "slow", "slow"], "E1 not repeated; the interrupted E2 step re-ran"
    assert any("was interrupted" in n for n in bundle.notes)
    (erepo / "state.txt").write_text("broken\n")  # the step that was interrupted would now fail
    plan2 = _plan(ph, snap, reg, [{"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
                                   "steps": [_probe_step(snap, "ok")]}])
    state2 = EvidenceState("P1", attempt=1, steps={"E2.s1": {"key": "E2.s1", "requirement_id": "E2",
                                                              "capability_id": "x", "status": RUNNING}},
                           inflight="E2.s1")
    bundle2 = asyncio.run(make_executor(reg, erepo, tmp_path).run(state2, plan2, snap))
    assert statuses(bundle2)["E2"] == FAILED, "a re-run that fails is a failure, never the interrupted 'success'"


def test_stale_resources_of_a_crashed_collection_are_reaped_only_when_provably_ours(tmp_path):
    sleeper = [PY, "-c", "import time; time.sleep(60)"]
    ours, other = subprocess.Popen(sleeper), subprocess.Popen(sleeper)
    gone = subprocess.Popen([PY, "-c", "pass"])
    gone.wait()
    try:
        time.sleep(0.3)
        state = EvidenceState("P1", resources=[
            {"id": "res-1", "capability_id": "cap-001", "pid": ours.pid, "start_time": procutil.start_time(ours.pid),
             "state": "READY"},
            {"id": "res-2", "capability_id": "cap-001", "pid": other.pid,
             "start_time": procutil.start_time(other.pid) - 3600, "state": "OBSERVING"},
            {"id": "res-3", "capability_id": "cap-001", "pid": gone.pid, "start_time": 1.0, "state": "STARTING"},
            {"id": "res-4", "capability_id": "cap-001", "pid": ours.pid, "start_time": 1.0, "state": "STOPPED"}])
        notes = ResourceManager(state, lambda: None, lambda s: None).reap_stale()
        assert [r["state"] for r in state.resources] == ["REAPED", "ABANDONED", "GONE", "STOPPED"]
        assert ours.wait(timeout=10) is not None and not procutil.is_alive(ours.pid)
        assert procutil.is_alive(other.pid), "a pid that now belongs to another process is never killed"
        assert len(notes) == 3 and "killed" in notes[0] and "reused" in notes[1]
    finally:
        for p in (ours, other):
            if p.poll() is None:
                p.kill()


def test_evidence_steps_can_never_change_the_repository(erepo, tmp_path):
    ph = make_phase([req("E1")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _plan(ph, snap, reg, [{"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
                                  "steps": [_probe_step(snap, "mutate", expect_output="RUNTIME OK")]}])
    state = EvidenceState("P1", attempt=1)
    logs = []
    bundle = asyncio.run(make_executor(reg, erepo, tmp_path, git_repo=GitRepo(str(erepo)), log=logs)
                         .run(state, plan, snap))
    assert (erepo / "README.md").read_text() == "readme\n" and not (erepo / "created_by_evidence.txt").exists()
    r = bundle.results[0]
    assert steps_of(state) == {"E1.s1": PASSED} and r.status == INCONCLUSIVE
    assert "modified repository files" in r.failure_reason
    assert any("restored to the pre-evidence state" in n and "README.md" in n for n in bundle.notes)
    assert subprocess.run(["git", "status", "--porcelain"], cwd=erepo, capture_output=True, text=True).stdout == ""


# ------------------------------------------------------------ scheduler integration (COLLECTING_EVIDENCE)

def with_p2(reqs, gate=False, verification=None):
    def before(e):
        p2 = phase(e, "P2")
        p2.verification_contract = VerificationContract.from_any(contract(reqs, gate))
        p2.verification_commands = list(verification or [])
    return before


def logs(h, source=None):
    return [p["text"] for k, p in h.events if k == "log" and (source is None or p["source"] == source)]


def states(h):
    return [p["state"] for k, p in h.events if k == "state"]


def p2_bundle(e, number=None):
    atts = e.db.load_attempts(e.run.id)["P2"]
    a = atts[-1] if number is None else next(x for x in atts if x.number == number)
    return a.evidence


def test_green_build_with_missing_runtime_evidence_is_not_passed(tmp_path, erepo):
    (erepo / ".aistudio" / "evidence.json").unlink()  # no runtime capability in this environment
    git(erepo, "commit", "-qam", "no manifest")
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")], verification=[BUILD_CMD])))
    assert e.sm.state == S.BLOCKED and "required evidence for P2 cannot be obtained" in e.run.status_message
    seq = states(h)
    i = seq.index("COLLECTING_EVIDENCE")
    assert seq[i - 1:i + 2] == ["VERIFYING", "COLLECTING_EVIDENCE", "REVIEWING"]
    a = e.db.load_attempts(e.run.id)["P2"][-1]
    assert all(v.passed for v in a.verification), "the build is green"
    assert [(r["requirement_id"], r["status"]) for r in a.evidence["results"]] == [("E1", UNAVAILABLE)]
    ck = orch.checkpoints[-1]
    prompt = build_checkpoint_context(ck, CheckpointContextLimits()).prompt
    for part in ["EVIDENCE RESULTS", "? E1 [runtime_behavior] UNAVAILABLE", "required evidence not satisfied: E1",
                 "not by themselves prove runtime, visual or interaction properties", "evidence_sufficient",
                 "EVIDENCE COLLECTION DETAILS", "[build] CONFIRMED"]:
        assert part in prompt, part
    assert "ASK_USER" not in prompt, "the human gate is closed"
    text = logs(h)
    assert any(x.startswith("P2: ") and "capabilities found (" in x for x in logs(h, "DISCOVERY"))
    assert "P2: plan created for 1 requirement(s) (0 execute, 1 unavailable)" in logs(h, "EVIDENCE")
    assert any(x.startswith("P2 E1 runtime_behavior: UNAVAILABLE") for x in logs(h, "EVIDENCE"))
    assert "evidence sufficient: false - 1 missing: E1 (runtime_behavior): UNAVAILABLE" in logs(h, "CLAUDE")
    assert text.index("evidence sufficient: false - 1 missing: E1 (runtime_behavior): UNAVAILABLE") < \
        next(i for i, x in enumerate(text) if x.startswith("CHECKPOINT P2 attempt 1: BLOCKED"))
    # Claude "proving" runtime behaviour with the green build: validated to indirect, still not satisfied
    orch2 = MockOrchestrator(delay=0.0, evidence_aware=True, evidence_plan_script={"P2": [lambda pk: {
        "summary": "use the build", "requirements": [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                                      "coverage": "direct", "reason": "",
                                                      "steps": [{"capability_id": next(
                                                          c.id for c in pk.snapshot.usable() if c.kind == "build")}]}]}]})
    h2 = V2Harness(tmp_path, make_settings(), orch=orch2, db="v2b.db")
    e2 = asyncio.run(h2.plan_and_start(erepo, with_p2([req("E1")], verification=[BUILD_CMD])))
    assert e2.sm.state == S.BLOCKED
    r = p2_bundle(e2)["results"][0]
    assert r["status"] == INCONCLUSIVE and r["confidence"] == "proxy" and "supporting evidence only" in r["summary"]


def test_sufficient_evidence_passes_and_is_recorded(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED
    ev = p2_bundle(e)
    assert [(r["requirement_id"], r["status"]) for r in ev["results"]] == [("E1", SATISFIED)]
    assert ev["artifacts"] and Path(ev["artifacts"][0]["path_or_ref"]).is_file()
    assert str(tmp_path) in ev["artifacts"][0]["path_or_ref"] and str(erepo) not in ev["artifacts"][0]["path_or_ref"]
    p2 = e.db.load_run(e.run.id).plan.task("P2")
    assert "evidence E1 [runtime_behavior]: SATISFIED" in p2.result.verification
    assert not any(x.startswith("UNVERIFIED") for x in p2.result.remaining_risks)
    assert "evidence sufficient: true" in logs(h, "CLAUDE")
    assert len(orch.evidence_plans) == 1 and "AVAILABLE CAPABILITIES" in orch.evidence_plan_prompts[0]
    assert "runtime probe" in orch.evidence_plan_prompts[0] and "Never invent a command" in orch.evidence_plan_prompts[0]
    assert [c.phase.id for c in orch.checkpoints if c.evidence] == ["P2"], "legacy phases P1/P3 collect nothing"


def test_proven_defect_retries_through_repair_and_the_plan_is_reused(tmp_path, erepo):
    (erepo / "state.txt").write_text("broken\n")
    git(erepo, "commit", "-qam", "broken")
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    worker = MockWorker(delay=0.0, steps=1)

    def fix(req):
        if req.purpose == "repair":
            (erepo / "state.txt").write_text("ok\n")
    worker.file_writer = fix
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED
    assert p2_bundle(e, 1)["results"][0]["status"] == FAILED and p2_bundle(e, 2)["results"][0]["status"] == SATISFIED
    atts = e.db.load_attempts(e.run.id)["P2"]
    assert [a.kind for a in atts] == ["execute", "repair"] and atts[0].review.decision.value == "RETRY"
    assert atts[0].review.evidence_sufficient is False
    rep = next(t for t in worker.turns if t["purpose"] == "repair")
    assert "Make E1 pass" in rep["prompt"] and "evidence E1 FAILED" in rep["prompt"]
    assert len(orch.evidence_plans) == 1, "the unchanged plan is reused for the repaired attempt"
    assert any("P2: plan reused" in x for x in logs(h, "EVIDENCE"))
    st = e.run.pipeline_state["evidence"]["P2"]
    assert st["attempt"] == 2 and [r["status"] for r in st["history"][0]["results"]] == [FAILED]


def test_unobservable_evidence_asks_the_user_with_the_evidence_summary(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, evidence_aware=True, checkpoint_script={"P2": ["EVIDENCE", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    reqs = [req("E1"), req("E2", kind="visual_evidence", goal="the result has an intentional visual hierarchy")]

    async def main():
        e = await h.plan_and_start(erepo, with_p2(reqs, gate=True))
        assert e.sm.state == S.AWAITING_HUMAN
        rq = e.run.pipeline_state["human"]["P2"]["request"]
        assert rq["questions"] == ["In the running result, can you confirm: the result has an intentional visual "
                                   "hierarchy?"]
        assert rq["evidence"][0].startswith("✓ E1 [runtime_behavior] SATISFIED")
        assert rq["evidence"][1].startswith("? E2 [visual_evidence] UNAVAILABLE - human observation needed")
        assert rq["missing"] == ["E2 (visual_evidence): UNAVAILABLE"]
        assert any("E1 runtime_behavior" in x for x in rq["observations"])
        await e.submit_human_feedback("ISSUE_FOUND", "The heading and the body text have the same weight.")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    packet = orch.followups[0]
    assert packet.evidence["results"][0]["requirement_id"] == "E1"
    for part in ["EVIDENCE RESULTS", "✓ E1 [runtime_behavior] SATISFIED", "not established by automated evidence: E2"]:
        assert part in orch.followup_prompts[0], part
    assert "REPAIRING" in states(h), "the human RETRY goes through the existing V3 repair"
    rep = next(t for t in h.worker.turns if t["purpose"] == "repair")
    assert "The heading and the body text have the same weight." in rep["prompt"]
    assert p2_bundle(e, 2)["results"][0]["status"] == SATISFIED, "evidence is collected again after the repair"


def test_objective_evidence_that_nobody_can_provide_blocks(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1", kind="performance", goal="p95 latency < 50 ms")],
                                                     gate=True)))
    assert e.sm.state == S.BLOCKED and "AWAITING_HUMAN" not in states(h)
    assert p2_bundle(e)["results"][0]["strategy"] == "unavailable"
    assert e.db.load_attempts(e.run.id)["P2"][-1].review.problems == ["missing evidence: E1 (performance): UNAVAILABLE"]


def test_legacy_phases_and_disabled_evidence_keep_the_old_flow(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo))  # the mock plan: no evidence requirements anywhere
    assert e.sm.state == S.COMPLETED and "COLLECTING_EVIDENCE" not in states(h)
    assert not orch.evidence_plans and "evidence" not in e.run.pipeline_state
    assert all(not c.evidence for c in orch.checkpoints) and not logs(h, "EVIDENCE") and not logs(h, "DISCOVERY")
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, make_settings(evidence_enabled=False), orch=orch, db="off.db")
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED and "COLLECTING_EVIDENCE" not in states(h) and not orch.evidence_plans
    assert "evidence sufficient" not in " ".join(logs(h, "CLAUDE"))


def test_crash_during_evidence_resumes_without_replanning_or_repeating_completed_steps(tmp_path, erepo, monkeypatch):
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("EVIDENCE_COUNTER", str(counter))

    def plan(pk):
        probe = next(c.id for c in pk.snapshot.usable() if c.name == "runtime probe")
        return {"summary": "two probes", "requirements": [
            {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct", "reason": "",
             "steps": [{"capability_id": probe, "args": ["ok"], "expect_output": "RUNTIME OK"}]},
            {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct", "reason": "",
             "steps": [{"capability_id": probe, "args": ["slow"], "expect_output": "RUNTIME OK"}]}]}
    s = make_settings()
    orch = MockOrchestrator(delay=0.0, evidence_aware=True, evidence_plan_script={"P2": [plan]})
    h = V2Harness(tmp_path, s, orch=orch, db="ecrash.db")

    async def first():
        e = h.engine
        await e.plan("g", str(erepo), pipeline="v2")
        await h.wait_idle()
        with_p2([req("E1"), req("E2")])(e)
        await e.start()
        while (e.run.pipeline_state.get("evidence", {}).get("P2", {}).get("inflight") != "E2.s1"):
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await e.stop()
        await h.wait_idle()
        assert e.sm.state == S.STOPPED and e.run.resume_state == "COLLECTING_EVIDENCE"
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(first())
    conn = sqlite3.connect(tmp_path / "ecrash.db")  # make it look like a hard kill during the step
    conn.execute("UPDATE runs SET state='COLLECTING_EVIDENCE', resume_state=NULL WHERE id=?", (run_id,))
    conn.commit()
    conn.close()
    # V4 pass 3: the repository itself stays unchanged (a changed file would make E1 stale - see
    # test_evidence_pass3); only the probe's delay changes, through its environment.
    monkeypatch.setenv("EVIDENCE_SLOW", "0.1")
    orch2 = MockOrchestrator(delay=0.0, evidence_aware=True)
    h2 = V2Harness(tmp_path, s, orch=orch2, db="ecrash.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.STOPPED and e.run.resume_state == "COLLECTING_EVIDENCE"
        st = e.run.pipeline_state["evidence"]["P2"]
        assert st["discovery"]["capabilities"] and st["plan"]["requirements"], "discovery and plan survived"
        assert st["steps"]["E1.s1"]["status"] == PASSED and st["steps"]["E2.s1"]["status"] == RUNNING
        first_art = st["artifacts"][0]
        await e.resume()
        await h2.wait_idle()
        return e, first_art
    e, first_art = asyncio.run(second())
    assert e.sm.state == S.COMPLETED
    assert not orch2.evidence_plans and not logs(h2, "DISCOVERY"), "no re-discovery, no re-planning"
    assert counter.read_text().split() == ["ok", "slow", "slow"], "E1 not repeated; the interrupted step re-ran"
    assert any("continuing the interrupted evidence collection" in x for x in logs(h2, "EVIDENCE"))
    assert any("E2.s1 was interrupted" in x for x in logs(h2, "EVIDENCE"))
    ev = p2_bundle(e)
    assert {r["requirement_id"]: r["status"] for r in ev["results"]} == {"E1": SATISFIED, "E2": SATISFIED}
    assert first_art["id"] in [a["id"] for a in ev["artifacts"]] and Path(first_art["path_or_ref"]).is_file(), \
        "artifact metadata from before the crash survives"


def test_v4_pass1_database_without_evidence_loads_and_resumes(tmp_path, erepo):
    """A run persisted by V4 pass 1: a phase contract WITH evidence requirements, but no evidence state and an
    attempt without an evidence bundle, resumed at the checkpoint."""
    from app.models.run import Run
    from app.orchestration.providers.mock import MOCK_PHASE_PLAN
    from app.models.phase import parse_phase_plan
    db_path = tmp_path / "p1.db"
    db = Database(db_path)
    plan = parse_phase_plan(dict(MOCK_PHASE_PLAN, goal="g"))
    plan.tasks[1].verification_contract = VerificationContract.from_any(contract([req("E1")]))
    run = Run(id=None, goal="g", repo_path=str(erepo), state="STOPPED", resume_state="REVIEWING",
              pipeline_version="v2", pipeline_state={"replans": 0, "human": {}}, current_task_id="P2")
    run.plan = plan
    db.create_run(run)
    db.update_run(run)
    rows = []
    for p in plan.tasks:
        d = p.to_dict()
        if p.id == "P1":
            d.update(status="PASSED", attempts=1, result={"task_id": "P1", "result": "done"})
        if p.id == "P2":
            d.update(status="PAUSED", attempts=1)
        rows.append((p.id, p.position, d))
    db.close()
    conn = sqlite3.connect(db_path)
    for pid, pos, d in rows:
        conn.execute("INSERT INTO tasks (run_id, task_id, position, status, attempts, data_json) VALUES (?,?,?,?,?,?)",
                     (run.id, pid, pos, d["status"], d.get("attempts", 0), json.dumps(d)))
    now = time.time()
    for pid in ("P1", "P2"):
        conn.execute("INSERT INTO task_attempts (run_id, task_id, number, kind, started_at, finished_at, data_json) "
                     "VALUES (?,?,?,?,?,?,?)", (run.id, pid, 1, "execute", now, now,
                                               json.dumps({"task_id": pid, "number": 1, "finished_at": now})))
    conn.commit()
    conn.close()
    orch = MockOrchestrator(delay=0.0)
    h = V2Harness(tmp_path, make_settings(), orch=orch, db="p1.db")

    async def main():
        e = h.engine
        await e.load_run(run.id)
        assert e.db.load_attempts(run.id)["P2"][0].evidence == {}
        await e.resume()
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and orch.checkpoints[0].phase.id == "P2" and orch.checkpoints[0].evidence == {}
    assert [t[0] for t in h.turns()] == ["P3"]


def test_claude_evidence_plan_call_is_bounded_and_returns_the_raw_plan(tmp_path, erepo):
    from app.config.settings import ClaudeSettings
    from app.orchestration.providers.claude_code_cli import V2_PURPOSES, ClaudeCodeCLIProvider, limits_for
    fake = tmp_path / "fake_claude_ev.py"
    fake.write_text("import json, sys\nsys.stdin.read()\npayload = json.loads(open(sys.argv[1]).read())\n"
                    "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1, "
                    "'usage': {}, 'structured_output': payload}), flush=True)\n")
    ph = make_phase([req("E1")])
    reg, snap = asyncio.run(discover(erepo, ph))
    packet = EvidencePlanPacket(phase=ph, snapshot=snap, repo_path=str(erepo))
    seen = []

    def prov_for(payload):
        (tmp_path / "ev_payload.json").write_text(json.dumps(payload))
        prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused"))

        async def argv(limits, schema):
            seen.append((limits, schema))
            return [PY, str(fake), str(tmp_path / "ev_payload.json")]
        prov._build_argv = argv
        return prov
    raw = {"summary": "s", "requirements": [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                             "reason": "", "steps": [{"capability_id": "cap-999"}]}]}
    out = asyncio.run(prov_for(raw).evidence_plan(packet))
    assert out == raw, "raw: the orchestrator validates it"
    limits, schema = seen[-1]
    assert schema is EVIDENCE_PLAN_JSON_SCHEMA and limits.purpose == "EVIDENCE_PLAN"
    assert limits.tools == "" and limits.max_tool_calls == 0 and limits.timeout == 180 and limits.max_budget_usd == 0.30
    assert "EVIDENCE_PLAN" in V2_PURPOSES and limits_for("EVIDENCE_PLAN", ClaudeSettings()).tools == ""
    with pytest.raises(OrchestratorError, match="invalid evidence plan"):
        asyncio.run(prov_for({"summary": "no list"}).evidence_plan(packet))


def test_v1_pipeline_is_unaffected(tmp_path, erepo):
    h = V2Harness(tmp_path, make_settings(), worker=MockWorker(delay=0.0, steps=1))

    async def main():
        await h.engine.plan("g", str(erepo))
        await h.wait_idle()
        await h.engine.start()
        await h.wait_idle()
    asyncio.run(main())
    e = h.engine
    assert type(e.scheduler) is Scheduler and e.sm.state == S.COMPLETED
    assert S.COLLECTING_EVIDENCE not in e.scheduler._handlers and all(type(t) is Task for t in e.run.plan.tasks)
    assert "evidence" not in e.run.pipeline_state and "COLLECTING_EVIDENCE" not in states(h)


def test_phase_details_show_a_compact_evidence_section():
    import os as _os
    _os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from app.gui.task_details import TaskDetails
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ph = make_phase([req("E1"), req("E2", kind="visual_evidence")], gate=True)
    ev = {"results": [
        {"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": SATISFIED,
         "summary": "probe exited 0 and printed 'RUNTIME OK'", "capability_ids": ["cap-003"], "strategy": "execute"},
        {"requirement_id": "E2", "kind": "visual_evidence", "required": True, "status": UNAVAILABLE,
         "summary": "human observation needed: no capture capability", "capability_ids": [], "strategy": "human"}],
        "artifacts": [{"id": "art-P2-a1-01", "kind": "log", "path_or_ref": "C:/x/art.log",
                       "description": "output of E1.s1"}],
        "capabilities": [{"id": "cap-003", "kind": "runtime_behavior", "name": "runtime probe",
                          "confidence": CONFIRMED}],
        "capability_summary": "7 capabilities found (2 confirmed, 4 inferred, 1 unavailable)", "plan_source": "claude",
        "steps_run": 1, "wall_used": 0.4, "notes": []}
    w = TaskDetails()
    w.show_task(ph.to_dict(), [{"number": 1, "kind": "execute", "started_at": 0, "evidence": ev,
                                "review": {"decision": "ASK_USER", "kind": "CHECKPOINT", "evidence_sufficient": False,
                                           "missing_evidence": ["E2 (visual_evidence): UNAVAILABLE"]}}])
    text = w.toPlainText()
    for part in ["Evidence (attempt 1)", "SATISFIED", "UNAVAILABLE", "runtime probe", "not sufficient",
                 "Missing evidence", "E2 (visual_evidence): UNAVAILABLE", "art-P2-a1-01", "7 capabilities found",
                 "Evidence requirements", "E1 [runtime_behavior, required]"]:
        assert part in text, part
    w.show_task(make_phase([req("E1")]).to_dict(), [])
    assert "(not collected yet)" in w.toPlainText()
