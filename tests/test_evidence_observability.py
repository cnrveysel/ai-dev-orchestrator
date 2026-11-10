"""Regression: required automation evidence must be observed by AI Studio itself (V4 real-Unreal validation).

The real failure: a phase verification command (UnrealEditor-Cmd ... -ExecCmds="Automation RunTests X; Quit"
without -stdout) exited 0 without printing Result={Success}; E2 stayed FAILED ("reused", although it had run
again), the worker's new verification script was never executed by the harness, a file check trusted a
leftover log, and PASS -> CONSISTENCY_REVIEW -> RETRY -> repair repeated until the retry limit.

Covered here (deterministic: mocked engine tools, throwaway git repositories, mock orchestrator/worker):
  A stdout success   B exit 0 without a matching result   C unrelated test success   D stale logs
  E fresh-log fallback   F invalidation by a changed verification input   G unrelated evidence stays reused
  H worker self-report   I the consistency/retry loop (same-attempt replan, and repair of the mechanism)
"""
import asyncio
import json
import os
import sys
import time
from pathlib import Path


from app.evidence import freshness as fr
from app.evidence import model as evm
from app.evidence.freshlog import FreshLog
from app.evidence.model import (
    FAILED, INCONCLUSIVE, SATISFIED, EvidencePlan, EvidenceState, RequirementPlan,
)
from app.evidence.mutation import repo_fingerprint
from app.evidence.planning import merge_replan, strategy_failures, validate_plan
from app.evidence.providers import unreal as ue
from app.evidence.providers.files import generated_reason
from app.evidence.registry import CapabilityRegistry
from app.integrations.git import GitRepo
from app.orchestration.checkpoint import CheckpointContextLimits, build_checkpoint_context
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.state_machine import RunState
from app.workers.mock import MockWorker
from app.workers.phase_prompts import _evidence_expectation_lines
from tests.test_evidence import MANIFEST, discover, erepo, make_phase, req, with_p2  # noqa: F401 - fixtures
from tests.test_evidence_pass3 import contradictory_pass, executor_for, logs, p2_attempts, states
from tests.test_pipeline_v2 import V2Harness, git, make_settings
from tests.test_providers_pass3 import (  # noqa: F401 - fixtures
    discover_with, engine, executor, plan_of, settings, unreal_provider, uproject,
)

PY = sys.executable
S = RunState
OLD = time.time() - 3600
LOG_SUCCESS = ("LogAutomationController: Display: Test Completed. Result={Success} Name={Wheels} "
               "Path={MyGame.Car.Wheels}\n")


# ============================================================ native Unreal automation (A-E)

def _automation(uproject, engine, tmp_path, flt="MyGame", **step):
    root, fakes = engine
    reg = CapabilityRegistry([unreal_provider(fakes)])
    ph = make_phase([req("E2", kind="automated_test")])
    snap = asyncio.run(discover_with(reg, uproject, ph, settings(**{"unreal.engine_root": str(root)})))
    auto = next(c for c in snap.capabilities if c.key.endswith(":automation"))
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "steps": [{"capability_id": auto.id, "args": [flt], **step}]}])
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor(reg, uproject, tmp_path).run(state, plan, snap))
    return bundle.results[0], state.steps["E2.s1"]


def _project_log(uproject, text, mtime=OLD):
    log = uproject / "Saved" / "Logs" / "MyGame.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    log.write_text(text)
    os.utime(log, (mtime, mtime))
    return log


def test_a_stdout_success_of_a_matching_test_satisfies_the_requirement(uproject, engine, tmp_path, monkeypatch):
    argv = tmp_path / "argv.json"
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "stdout")
    monkeypatch.setenv("FAKE_ARGV_FILE", str(argv))
    r, step = _automation(uproject, engine, tmp_path)
    assert r.status == SATISFIED and step["exit_code"] == 0
    assert "2 automation test(s) matching 'MyGame' passed (observed in stdout)" in r.summary
    auto = step["details"]["automation"]
    assert auto["found"] == 2 and auto["ran"] == 2 and auto["sources"] == ["stdout"]
    args = json.loads(argv.read_text())
    for flag in ("-ExecCmds=Automation RunTests MyGame;Quit", "-TestExit=Automation Test Queue Empty", "-stdout",
                 "-FullStdOutLogOutput", "-NullRHI", "-unattended"):
        assert flag in args, flag
    report = next(a.split("=", 1)[1] for a in args if a.startswith("-ReportExportPath="))
    assert str(uproject) not in report and not Path(report).exists(), "the run's own report dir, removed afterwards"


def test_the_windows_command_line_quotes_values_the_way_unreal_parses_them():
    cmd = ue.automation_command([r"C:\Program Files\Epic Games\UE_5.7\Engine\Binaries\Win64\UnrealEditor-Cmd.exe"],
                                r"C:\Projects\My Game\MyGame.uproject", "MyGame.Car+MyGame.Menu", r"C:\T\r")
    if sys.platform == "win32":
        assert isinstance(cmd, str) and cmd.startswith('"C:\\Program Files\\Epic Games\\')
        for part in ('"C:\\Projects\\My Game\\MyGame.uproject"', '-ExecCmds="Automation RunTests MyGame.Car+MyGame.Menu;Quit"',
                     '-TestExit="Automation Test Queue Empty"', '-ReportExportPath="C:\\T\\r"',
                     " -stdout", " -FullStdOutLogOutput"):
            assert part in cmd, part
    else:
        assert isinstance(cmd, list) and "-ExecCmds=Automation RunTests MyGame.Car+MyGame.Menu;Quit" in cmd


def test_b_exit_code_zero_without_a_matching_result_never_passes(uproject, engine, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "silent")
    r, step = _automation(uproject, engine, tmp_path)
    assert step["exit_code"] == 0 and r.status == INCONCLUSIVE and r.status != SATISFIED
    assert r.cause == "strategy" and "an exit code alone is not evidence" in r.summary
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "none")
    r2, _ = _automation(uproject, engine, tmp_path)
    assert r2.status == FAILED and "found 0 tests" in r2.summary
    # a planner's marker is checked against what THIS run produced - not satisfied by the exit code
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "stdout")
    r3, _ = _automation(uproject, engine, tmp_path, expect_output="Result={Success} Name={NoSuchTest}")
    assert r3.status == FAILED and r3.cause == "unobserved"


def test_c_a_success_of_an_unrelated_test_does_not_satisfy_the_requested_filter(uproject, engine, tmp_path,
                                                                                  monkeypatch):
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "unrelated")
    r, step = _automation(uproject, engine, tmp_path)
    assert r.status == FAILED and "only unrelated test(s) reported a result: OtherGame.Menu.Opens" in r.summary
    assert step["details"]["automation"]["ran"] == 0
    assert ue.filter_matches("MyGame.Car.Wheels", "MyGame") and ue.filter_matches("Tests.MyGame.X", "MyGame")
    assert not ue.filter_matches("MyGameOther.X", "MyGame") and not ue.filter_matches("OtherGame.Menu", "MyGame")
    assert ue.filter_matches("B.Y", "A+B") and not ue.filter_matches("MyGame.Carx", "MyGame.Car")


def test_d_a_stale_log_success_never_satisfies_the_current_run(uproject, engine, tmp_path, monkeypatch):
    log = _project_log(uproject, "Log file open\n" + LOG_SUCCESS)
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "silent")  # the new run produces no result at all
    r, step = _automation(uproject, engine, tmp_path)
    assert r.status == INCONCLUSIVE and step["details"]["automation"]["ran"] == 0, \
        "the old Result={Success} in the project log is not this run's"
    assert log.read_text().count("Result={Success}") == 1, "read-only: the project log is never rotated or deleted"
    # the new run appends a FAILURE for the same test: the old success above it is ignored
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "project_log_fail")
    r2, _ = _automation(uproject, engine, tmp_path)
    assert r2.status == FAILED and "MyGame.Car.Wheels" in r2.summary


def test_d_a_file_check_of_a_leftover_generated_log_is_not_evidence(uproject, tmp_path):
    _project_log(uproject, LOG_SUCCESS)
    assert generated_reason(str(uproject), "Saved/Logs/MyGame.log")
    assert not generated_reason(str(uproject), "Source/MyGame/MyGame.Build.cs")
    ph = make_phase([req("E2", kind="file_inspection")])  # a file check could satisfy it - if it were fresh
    reg, snap = asyncio.run(discover(uproject, ph))
    files = next(c.id for c in snap.capabilities if c.kind == "file_inspection")
    plan = plan_of(ph, snap, reg, [{"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
                                    "steps": [{"capability_id": files,
                                               "args": ["contains", "Saved/Logs/MyGame.log", "Result={Success}"]}]}])
    state = EvidenceState("P1", attempt=1, started_at=time.time())
    r = asyncio.run(executor(reg, uproject, tmp_path).run(state, plan, snap)).results[0]
    assert r.status == INCONCLUSIVE and "before this evidence collection started" in r.summary
    assert state.steps["E2.s1"]["provenance"]["scope"] == "repository", \
        "a generated file depends on whatever produced it, not on its own bytes"
    _project_log(uproject, LOG_SUCCESS, mtime=time.time() + 1)  # written during this collection (by a step)
    state2 = EvidenceState("P1", attempt=1, started_at=time.time() - 1)
    r2 = asyncio.run(executor(reg, uproject, tmp_path).run(state2, plan, snap)).results[0]
    assert r2.status == SATISFIED and "written during this evidence collection" in r2.summary


def test_e_fresh_log_fallback_when_stdout_lacks_the_result(uproject, engine, tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "project_log")  # stdout lacks the results; the new log has them
    r, step = _automation(uproject, engine, tmp_path)
    assert r.status == SATISFIED and step["details"]["automation"]["sources"] == ["project log"]
    _project_log(uproject, "Log file open, old run\n" + LOG_SUCCESS.replace("Success", "Fail"))  # an OLD failure
    r2, step2 = _automation(uproject, engine, tmp_path)  # appended below it
    assert r2.status == SATISFIED and step2["details"]["automation"]["sources"] == ["project log"], \
        "only the part this run appended counts - the old failure above it is not this run's either"
    _project_log(uproject, "Log file open, old run\n" + LOG_SUCCESS.replace("Success", "Fail"))
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "project_log_rotate")  # Unreal renames the old log, starts anew
    r3, _ = _automation(uproject, engine, tmp_path)
    assert r3.status == SATISFIED
    assert (uproject / "Saved" / "Logs" / "MyGame-backup-2026.01.01-00.00.00.log").is_file()


def test_freshlog_reads_only_what_the_current_run_wrote(tmp_path):
    log = tmp_path / "x.log"
    assert FreshLog.snapshot(str(log)).fresh_text()[0] == ""
    snap = FreshLog.snapshot(str(log))
    log.write_bytes(b"created\n")
    assert snap.fresh_text() == ("created\n", "created by this run")
    log.write_bytes(b"old line\n")
    os.utime(log, (OLD, OLD))
    snap = FreshLog.snapshot(str(log))
    assert snap.fresh_text()[1].endswith("(stale)"), "unchanged since before the run"
    with open(log, "ab") as f:
        f.write(b"new line\n")
    assert snap.fresh_text() == ("new line\n", "appended by this run")
    snap = FreshLog.snapshot(str(log))
    log.write_bytes(b"rotated: a new log\n")
    assert snap.fresh_text() == ("rotated: a new log\n", "replaced by this run")


# ============================================================ invalidation and reuse (F, G)

RUNNER = {"name": "automation runner", "command": f'"{PY}" checks/verify.py', "kind": "automated_test",
          "description": "runs the automation tests and prints their results"}
VERIFY_FAIL = "import sys\nprint('Found 0 automation tests')\nsys.exit(1)\n"
VERIFY_OK = ("print('Test Completed. Result={Success} Name={Wheels} Path={MyGame.Car.Wheels}')\n"
             "print('**** TEST COMPLETE. EXIT CODE: 0 ****')\n")


def add_runner(repo, body=VERIFY_FAIL, commit=True):
    (repo / "checks" / "verify.py").write_text(body)
    (repo / ".aistudio" / "evidence.json").write_text(json.dumps(dict(MANIFEST, commands=MANIFEST["commands"]
                                                                      + [RUNNER])))
    if commit:
        git(repo, "add", "-A")
        git(repo, "commit", "-qm", "runner")


def _cid(snap, name):
    return next(c.id for c in snap.capabilities if c.name == name)


def _three_plan(ph, snap, reg):
    files = _cid(snap, "repository file inspection")
    return validate_plan({"summary": "s", "requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": files, "args": ["contains", "README.md", "readme"]}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": _cid(snap, "automation runner"), "expect_output": "Result={Success}"},
                   {"capability_id": files, "args": ["exists", "checks/probe.py"]}]},
        {"requirement_id": "E3", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": files, "args": ["contains", "state.txt", "ok"]}]}]}, ph, snap, reg, 12)


def test_f_a_changed_verification_script_makes_the_failed_result_stale_and_it_runs_again(erepo, tmp_path):
    add_runner(erepo)
    ph = make_phase([req("E2", kind="automated_test")])
    reg, snap = asyncio.run(discover(erepo, ph))
    runner = next(c for c in snap.capabilities if c.name == "automation runner")
    assert set(runner.details["scripts"]) == {"checks/verify.py"}, "the script is part of the definition"
    plan = validate_plan({"summary": "s", "requirements": [
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": runner.id}]}]}, ph, snap, reg, 12)
    fp = ("git:same-on-purpose", "git_tree")  # isolate the definition: even an unchanged fingerprint cannot save it
    state = EvidenceState("P1", attempt=1)
    b1 = asyncio.run(executor_for(reg, erepo, tmp_path, fp).run(state, plan, snap))
    assert b1.results[0].status == FAILED and b1.results[0].cause == "product"
    state.begin_attempt(2)
    b_same = asyncio.run(executor_for(reg, erepo, tmp_path, fp).run(state, plan, snap))
    assert b_same.results[0].freshness == "reused", "nothing changed: the FAILED result may be reused"
    state.begin_attempt(3)
    (erepo / "checks" / "verify.py").write_text(VERIFY_OK)  # the repair changed the verification mechanism
    reg2, snap2 = asyncio.run(discover(erepo, ph))
    new = next(c for c in snap2.capabilities if c.name == "automation runner")
    old_prov = state.reusable[next(iter(state.reusable))]["provenance"]
    assert fr.check(old_prov, cap_fp=fr.capability_fp(new), repo_fp=fp[0], inputs=None) == \
        (False, "the capability's definition changed")
    b3 = asyncio.run(executor_for(reg2, erepo, tmp_path, fp).run(state, plan, snap2))
    assert b3.results[0].status == SATISFIED and b3.results[0].freshness == "fresh"
    assert state.steps["E2.s1"].get("reused_from", 0) == 0 and state.steps["E2.s1"]["provenance"]["attempt"] == 3


def test_f_a_changed_provider_setting_invalidates_earlier_unreal_evidence(uproject, engine):
    root, fakes = engine
    reg = CapabilityRegistry([unreal_provider(fakes)])
    ph = make_phase([req("E2", kind="automated_test")])

    def auto(flt):
        snap = asyncio.run(discover_with(reg, uproject, ph, settings(**{"unreal.engine_root": str(root),
                                                                         "unreal.test_filter": flt})))
        return next(c for c in snap.capabilities if c.key.endswith(":automation"))
    a, b = auto("MyGame"), auto("MyGame.Car")
    assert fr.capability_fp(a) != fr.capability_fp(b)
    assert fr.check({"capability_fp": fr.capability_fp(a), "repo_fp": "git:x"}, cap_fp=fr.capability_fp(b),
                    repo_fp="git:x", inputs=None)[0] is False


def test_g_only_the_affected_requirement_runs_again_and_labels_say_what_ran(erepo, tmp_path):
    add_runner(erepo)
    g = GitRepo(str(erepo))
    ph = make_phase([req("E1", kind="file_inspection"), req("E2", kind="automated_test"),
                     req("E3", kind="file_inspection")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _three_plan(ph, snap, reg)
    fp1 = asyncio.run(repo_fingerprint(str(erepo), g))
    state = EvidenceState("P1", attempt=1)
    b1 = asyncio.run(executor_for(reg, erepo, tmp_path, fp1, g).run(state, plan, snap))
    assert [r.status for r in b1.results] == [SATISFIED, FAILED, SATISFIED]
    (erepo / "checks" / "verify.py").write_text(VERIFY_OK)  # only E2's mechanism changes
    reg2, snap2 = asyncio.run(discover(erepo, ph))
    fp2 = asyncio.run(repo_fingerprint(str(erepo), g))
    state.begin_attempt(2)
    b2 = asyncio.run(executor_for(reg2, erepo, tmp_path, fp2, g).run(state, plan, snap2))
    by = {r.requirement_id: r for r in b2.results}
    assert {k: (r.status, r.freshness) for k, r in by.items()} == {
        "E1": (SATISFIED, "reused"), "E2": (SATISFIED, "fresh"), "E3": (SATISFIED, "reused")}
    assert state.steps["E2.s1"]["provenance"]["attempt"] == 2 and not state.steps["E2.s1"].get("reused_from")
    assert state.steps["E2.s2"]["reused_from"] == 1, "its unchanged file check is reused"
    assert by["E2"].provenance["reused_steps"] == {"E2.s2": 1}
    assert "(ran now; unchanged step(s) reused: E2.s2)" in evm.result_line(by["E2"]), \
        "a result decided by a step that ran now is never labelled 'reused'"


# ============================================================ H. worker self-report

def test_h_a_worker_report_or_its_leftover_log_never_satisfies_required_evidence(uproject, engine, tmp_path,
                                                                                   monkeypatch):
    # the worker ran the tests itself, left a log saying Result={Success} and reported "all tests pass"
    _project_log(uproject, "Log file open\n" + LOG_SUCCESS, mtime=time.time() - 5)
    monkeypatch.setenv("FAKE_AUTOMATION_MODE", "silent")  # but AI Studio's own run observes nothing
    r, _ = _automation(uproject, engine, tmp_path)
    assert r.status == INCONCLUSIVE and not evm.is_satisfied(r)
    lines = "\n".join(_evidence_expectation_lines(make_phase([req("E2", kind="automated_test")])))
    assert "your report, and logs your own runs left behind, are not evidence" in lines
    assert ".aistudio/evidence.json" in lines, "the worker is told how a helper script can become harness evidence"


def test_h_a_worker_written_helper_script_is_run_by_the_harness_and_flagged(erepo):
    add_runner(erepo, VERIFY_OK, commit=False)
    ph = make_phase([req("E2", kind="automated_test")])
    from app.evidence.providers.base import DiscoveryContext
    from app.evidence.registry import default_registry
    reg = default_registry()
    snap = asyncio.run(reg.discover(DiscoveryContext(str(erepo), ph, [ph], {}, make_settings().orchestration,
                                                     changed_files=["checks/verify.py"])))
    runner = next(c for c in snap.capabilities if c.name == "automation runner")
    assert runner.provenance["changed_in_phase"] is True
    assert runner.provenance["scripts_changed_in_phase"] == ["checks/verify.py"]


# ============================================================ I. the consistency / retry loop

def _p2_plan(first):
    """EVIDENCE_PLAN stand-in: E1 a file check; E2 with the given capability (by name prefix)."""
    def plan(pk):
        caps = pk.snapshot.usable()
        files = next(c.id for c in caps if c.kind == "file_inspection")
        e2 = next(c for c in caps if c.name.startswith(first))
        step = {"capability_id": e2.id}
        if e2.source == "phase_verification":
            step["expect_output"] = "Result={Success}"
        return {"summary": "p", "requirements": [
            {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct", "reason": "r",
             "steps": [{"capability_id": files, "args": ["contains", "README.md", "readme"]}]},
            {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct", "reason": "r",
             "steps": [step]}]}
    return plan


def _replan_e2(name):
    def plan(pk):
        assert pk.replan["requirements"] == ["E2"], pk.replan
        e2 = next(c for c in pk.snapshot.usable() if c.name == name)
        return {"summary": "r", "requirements": [{"requirement_id": "E2", "strategy": "execute", "goal": "g",
                                                  "coverage": "direct", "reason": "reports the results itself",
                                                  "steps": [{"capability_id": e2.id}]}]}
    return plan


def test_i_an_unobserved_marker_is_replanned_to_a_capability_that_observes_it(tmp_path, erepo):
    """The real failure, attempt 1: the phase's own bare command exits 0 without the result on stdout."""
    add_runner(erepo, VERIFY_OK)
    (erepo / "checks" / "bare.py").write_text("print('LogInit: engine started')\n")
    git(erepo, "add", "-A")
    git(erepo, "commit", "-qm", "bare")
    bare = f'"{PY}" checks/bare.py'
    orch = MockOrchestrator(delay=0.0, evidence_aware=True, evidence_plan_script={
        "P2": [_p2_plan("phase verification"), _replan_e2("automation runner")]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1", kind="file_inspection"),
                                                      req("E2", kind="automated_test")], verification=[bare])))
    assert e.sm.state == S.COMPLETED
    atts = p2_attempts(e)
    assert len(atts) == 1 and "CONSISTENCY_REVIEW" not in states(h) and not orch.consistency_packets
    res = {r["requirement_id"]: r for r in atts[0].evidence["results"]}
    assert res["E2"]["status"] == SATISFIED and res["E2"]["steps"] == ["E2.r1.s1"]
    assert any("could not observe the property" in x for x in logs(h, "EVIDENCE"))
    replan = orch.evidence_plans[1]
    assert not any(c.name.startswith("phase verification") for c in replan.snapshot.usable()), \
        "the command that could not show the result is excluded from the replan"
    assert "could not OBSERVE the property" in orch.evidence_plan_prompts[1]


def test_i_an_unobserved_failure_without_a_better_strategy_stays_failed():
    bad = RequirementPlan("E2", "automated_test", True, "execute", "g")
    plan = EvidencePlan([bad], "s")
    none = EvidencePlan([RequirementPlan("E2", "automated_test", True, "unavailable", "g", reason="nothing")], "r",
                        generation=1)
    assert merge_replan(plan, none, ["E2"], {"E2"}).requirements[0] is bad, \
        "a replan never turns a failed check into 'no evidence'"
    assert merge_replan(plan, none, ["E2"]).requirements[0].strategy == "unavailable", "strategy failures unchanged"
    res = evm.EvidenceResult("E2", "automated_test", True, FAILED, cause="product", failed_capability="")
    b = evm.EvidenceBundle("P1", 1, results=[res])
    snap = evm.CapabilitySnapshot([])
    assert strategy_failures(b, plan, snap, set()) == [], "an ordinary product failure is never replanned"
    res.cause, res.failed_capability = "unobserved", "cap-002"
    assert strategy_failures(b, plan, snap, set())[0]["kind"] == "unobserved"


def test_i_a_repaired_verification_mechanism_reruns_e2_and_ends_the_consistency_loop(tmp_path, erepo):
    """The real failure, attempts 2-3: PASS with E2 FAILED -> CONSISTENCY_REVIEW -> RETRY -> the repair changes the
    verification script -> the next collection must run E2 again (not reuse the old FAILED) -> PASS, no second review."""
    add_runner(erepo)
    orch = MockOrchestrator(delay=0.0, evidence_aware=True, evidence_plan_script={"P2": [_p2_plan("automation runner")]},
                            checkpoint_script={"P2": [contradictory_pass("E2 (automated_test): FAILED"), "EVIDENCE"]})
    worker = MockWorker(delay=0.0, steps=1)

    def repair(r):
        if r.purpose == "repair":
            (erepo / "checks" / "verify.py").write_text(VERIFY_OK)
    worker.file_writer = repair
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1", kind="file_inspection"),
                                                      req("E2", kind="automated_test")])))
    assert e.sm.state == S.COMPLETED
    assert len(orch.consistency_packets) == 1, "exactly one review: the repaired evidence settles it"
    atts = p2_attempts(e)
    assert len(atts) == 2 and atts[0].review.kind == "CONSISTENCY_REVIEW"
    first = {r["requirement_id"]: r for r in atts[0].evidence["results"]}
    second = {r["requirement_id"]: r for r in atts[1].evidence["results"]}
    assert first["E2"]["status"] == FAILED
    assert (second["E2"]["status"], second["E2"]["freshness"]) == (SATISFIED, "fresh"), "E2 ran again"
    assert (second["E1"]["status"], second["E1"]["freshness"]) == (SATISFIED, "reused"), "E1 is unaffected"
    assert len(orch.evidence_plans) == 1, "the plan is reused - its E2 step simply cannot reuse the old result"
    ck = build_checkpoint_context([c for c in orch.checkpoints if c.phase.id == "P2"][-1],
                                  CheckpointContextLimits()).prompt
    assert "E2 [automated_test] SATISFIED" in ck and "E2 [automated_test] FAILED (reused" not in ck
