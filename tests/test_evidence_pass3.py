"""V4 pass 3: consistency review, first-class human-derived criteria, evidence freshness, bounded replanning,
readiness probes, artifact retention, non-git mutation safety, provenance, human-gate quality, prompt control
and backward compatibility.

Deterministic: mock orchestrator / mock workers (the conftest guard refuses any real Claude or Codex process);
throwaway fixture repositories (git and non-git); tiny local Python services.
"""
import asyncio
import json
import os
import socket
import sqlite3
import sys
import time
from pathlib import Path

import pytest

from app.evidence import freshness as efresh
from app.evidence import model as evm
from app.evidence import procutil
from app.evidence import readiness as rd
from app.evidence.artifacts import ArtifactStore, RetentionPolicy, cleanup, inside
from app.evidence.executor import EvidenceExecutor, EvidenceLimits
from app.evidence.lifecycle import ServiceProcess
from app.evidence.model import (
    CONFIRMED, FAILED, INCONCLUSIVE, PASSED, SATISFIED, STALE, UNAVAILABLE, EvidenceState,
)
from app.evidence.mutation import ManifestMutationGuard, manifest_scan, repo_fingerprint
from app.evidence.planning import capability_line, validate_plan
from app.evidence.providers.base import DiscoveryContext
from app.evidence.registry import default_registry
from app.integrations.git import GitRepo
from app.models import human as hm
from app.models.review import Review, ReviewDecision
from app.models.task import TaskStatus
from app.models.verification import VerificationContract
from app.orchestration.checkpoint import CheckpointContextLimits, build_checkpoint_context
from app.orchestration.consistency import (
    ConsistencyLimits, ConsistencyPacket, build_consistency_prompt, find_conflict, parse_consistency,
)
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.state_machine import RunState
from app.persistence.database import Database
from app.workers.mock import MockWorker
from tests.test_evidence import (  # noqa: F401 - fixtures and helpers of the pass 2 suite
    MANIFEST, PROBE_CMD, cap_id, contract, discover, erepo, make_phase, req, with_p2,
)
from tests.test_pipeline_v2 import V2Harness, git, make_settings, phase

PY = sys.executable
S = RunState


def logs(h, source=None):
    return [p["text"] for k, p in h.events if k == "log" and (source is None or p["source"] == source)]


def states(h):
    return [p["state"] for k, p in h.events if k == "state"]


def p2_attempts(e):
    return e.db.load_attempts(e.run.id)["P2"]


def reviews(e, pid="P2"):
    return [(r[0], json.loads(r[1]).get("kind")) for r in e.db.conn.execute(
        "SELECT decision, data_json FROM reviews WHERE task_id=? ORDER BY id", (pid,))]


def contradictory_pass(*missing):
    return Review(ReviewDecision.PASS, "[TEST] accepted although evidence is missing", evidence_sufficient=False,
                  missing_evidence=list(missing), task_result={"result": "[TEST] done", "important_decisions": [],
                                                               "remaining_risks": []})


def no_manifest(repo):
    (repo / ".aistudio" / "evidence.json").unlink()
    git(repo, "commit", "-qam", "no manifest")


# ============================================================ A. CONSISTENCY REVIEW

def test_find_conflict_only_for_a_meaningful_contradiction():
    ev = {"results": [{"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": UNAVAILABLE},
                      {"requirement_id": "E2", "kind": "build", "required": True, "status": SATISFIED}]}
    assert find_conflict(contradictory_pass("E1"), ev)["unresolved"] == ["E1"]
    ok = Review(ReviewDecision.PASS, "s", evidence_sufficient=True)
    assert find_conflict(ok, ev) is None, "Claude explicitly judged the evidence sufficient: no contradiction"
    assert find_conflict(Review(ReviewDecision.PASS, "s"), ev) is None, "no judgment: nothing to contradict"
    assert find_conflict(Review(ReviewDecision.RETRY, "s", evidence_sufficient=False), ev) is None
    assert find_conflict(contradictory_pass(), {}) is None, "no evidence: legacy behaviour"
    satisfied = {"results": [{"requirement_id": "E1", "kind": "x", "required": True, "status": SATISFIED}]}
    assert find_conflict(contradictory_pass(), satisfied) is None, "nothing is unresolved"
    failed = {"results": [{"requirement_id": "E1", "kind": "x", "required": True, "status": FAILED}]}
    c = find_conflict(Review(ReviewDecision.PASS, "s", evidence_sufficient=True), failed)
    assert c["kind"] == "failed_evidence", "PASS over a demonstrated defect is a contradiction too"
    stale = {"results": [{"requirement_id": "E1", "kind": "x", "required": True, "status": INCONCLUSIVE,
                          "freshness": STALE, "stale_status": SATISFIED}]}
    assert find_conflict(contradictory_pass(), stale)["unresolved"] == ["E1"]
    optional = {"results": [{"requirement_id": "E1", "kind": "x", "required": False, "status": UNAVAILABLE}]}
    assert find_conflict(contradictory_pass(), optional) is None


def test_ordinary_pass_with_sufficient_evidence_is_unchanged(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED and "CONSISTENCY_REVIEW" not in states(h) and not orch.consistency_packets
    assert "consistency" not in e.run.pipeline_state
    assert [d for d, k in reviews(e)] == ["PASS"]


def test_contradictory_pass_triggers_a_bounded_consistency_review(tmp_path, erepo):
    no_manifest(erepo)  # nothing can observe the runtime behaviour here
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1 (runtime_behavior)")]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.BLOCKED and "Claude CONSISTENCY_REVIEW marked phase P2 BLOCKED" in e.run.status_message
    seq = states(h)
    i = seq.index("CONSISTENCY_REVIEW")
    assert seq[i - 1] == "REVIEWING" and seq.count("CONSISTENCY_REVIEW") == 1
    assert reviews(e) == [("PASS", "CHECKPOINT"), ("BLOCKED", "CONSISTENCY_REVIEW")], "both are on record"
    rec = e.run.pipeline_state["consistency"]["P2"]
    assert rec["status"] == "resolved" and rec["trigger"]["kind"] == "insufficient_evidence"
    assert rec["trigger"]["unresolved"] == ["E1"] and rec["result"]["decision"] == "BLOCKED"
    assert rec["checkpoint"]["decision"] == "PASS" and rec["checkpoint"]["evidence_sufficient"] is False
    packet = orch.consistency_packets[0]
    assert packet.trigger["unresolved"] == ["E1"] and not packet.human_gate_open
    prompt = orch.consistency_prompts[0]
    for part in ["CONSISTENCY_REVIEW", "THE INCONSISTENCY", "evidence_sufficient=false", "ACCEPTANCE CRITERIA",
                 "VERIFICATION CONTRACT", "? E1 [runtime_behavior] UNAVAILABLE", "UNRESOLVED REQUIRED EVIDENCE: E1",
                 "YOUR CHECKPOINT", "decision: PASS; evidence_sufficient: false", "missing_evidence: E1",
                 "EVERY unresolved required requirement", "Human verification is not available"]:
        assert part in prompt, part
    assert "ASK_USER" not in prompt.split("DECIDE")[1]
    a = p2_attempts(e)[-1]
    assert a.review.kind == "CONSISTENCY_REVIEW" and a.review.consistency["trigger"]["unresolved"] == ["E1"]
    assert a.review.evidence_sufficient is False, "the checkpoint's evidence judgment stays on record"
    assert any("inconsistent checkpoint" in x for x in logs(h, "CLAUDE"))


def test_consistency_pass_requires_an_explicit_justification_and_is_recorded(tmp_path, erepo):
    no_manifest(erepo)
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1")]},
                            consistency_script={"P2": ["PASS_UNJUSTIFIED", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        e = await h.plan_and_start(erepo, with_p2([req("E1")]))
        assert e.sm.state == S.PAUSED and e.run.resume_state == "CONSISTENCY_REVIEW"
        assert "PASS without a justification" in e.run.status_message
        assert e.run.pipeline_state["consistency"]["P2"]["status"] == "pending", "an unjustified PASS never counts"
        assert phase(e, "P2").status != TaskStatus.PASSED
        await e.resume()
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    assert len([c for c in orch.checkpoints if c.phase.id == "P2"]) == 1, "the checkpoint itself is never repeated"
    assert len(orch.consistency_packets) == 2
    p2 = e.db.load_run(e.run.id).plan.task("P2")
    assert any(d.startswith("accepted by a consistency review despite unresolved required evidence: [MOCK]")
               and "E1 substitute_evidence" in d for d in p2.result.important_decisions)
    assert any(r.startswith("UNVERIFIED E1 [runtime_behavior]") for r in p2.result.remaining_risks), \
        "accepted is not verified: NO FALSE EVIDENCE"
    rec = e.run.pipeline_state["consistency"]["P2"]
    assert rec["result"]["decision"] == "PASS" and rec["result"]["justification"]
    assert rec["result"]["dispositions"] == [{"requirement_id": "E1", "disposition": "substitute_evidence",
                                              "reason": "[MOCK] covered by the passing phase verification"}]
    # the parser rejects the same unjustified answers a provider could return
    for raw, why in [({"decision": "PASS", "summary": "s", "justification": "", "evidence_dispositions": [],
                       "problems": [], "corrective_instructions": []}, "without a justification"),
                     ({"decision": "PASS", "summary": "s", "justification": "fine", "evidence_dispositions": [
                         {"requirement_id": "E1", "disposition": "still_required", "reason": "x"}], "problems": [],
                       "corrective_instructions": []}, "E1"),
                     ({"decision": "REPLAN", "summary": "s", "justification": "", "evidence_dispositions": [],
                       "problems": [], "corrective_instructions": []}, "not allowed")]:
        with pytest.raises(ValueError, match=why):
            parse_consistency(raw, unresolved_ids=["E1"], allow_ask_user=False)


def test_consistency_retry_goes_through_the_existing_repair(tmp_path, erepo):
    (erepo / "state.txt").write_text("broken\n")
    git(erepo, "commit", "-qam", "broken")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1"), "EVIDENCE"]})
    worker = MockWorker(delay=0.0, steps=1)

    def fix(r):
        if r.purpose == "repair":
            (erepo / "state.txt").write_text("ok\n")
    worker.file_writer = fix
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED
    seq = states(h)
    i = seq.index("CONSISTENCY_REVIEW")
    assert seq[i:i + 3] == ["CONSISTENCY_REVIEW", "RETRY_PENDING", "REPAIRING"]
    assert orch.consistency_packets[0].trigger["kind"] == "insufficient_evidence"
    rep = next(t for t in worker.turns if t["purpose"] == "repair")
    assert "[MOCK] make E1 pass" in rep["prompt"]
    assert any("entering repair mode after consistency review RETRY" in x for x in logs(h, "REPAIR"))
    atts = p2_attempts(e)
    assert atts[0].review.kind == "CONSISTENCY_REVIEW" and atts[0].review.decision == ReviewDecision.RETRY
    assert atts[-1].evidence["results"][0]["status"] == SATISFIED


def test_consistency_review_can_ask_the_user_about_the_unresolved_evidence(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E2")]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    reqs = [req("E1"), req("E2", kind="visual_evidence", goal="the wheels visibly rotate")]

    async def main():
        e = await h.plan_and_start(erepo, with_p2(reqs, gate=True))
        assert e.sm.state == S.AWAITING_HUMAN
        rq = e.run.pipeline_state["human"]["P2"]["request"]
        assert rq["questions"] == ["In the running result, can you confirm: the wheels visibly rotate?"]
        assert rq["targets"] == ["E2"] and rq["missing"] == ["E2"]
        assert rq["established"] and rq["established"][0].startswith("E1 [runtime_behavior]")
        await e.submit_human_feedback("LOOKS_GOOD", "They rotate.")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and orch.consistency_packets[0].human_gate_open
    seq = states(h)
    assert seq[seq.index("CONSISTENCY_REVIEW") + 1] == "AWAITING_HUMAN"
    assert [k for _d, k in reviews(e)] == ["CHECKPOINT", "CONSISTENCY_REVIEW", "HUMAN_FOLLOWUP"]


def test_consistency_review_is_never_recursive(tmp_path, erepo):
    no_manifest(erepo)
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1")]},
                            consistency_script={"P2": ["PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    # The accepted review still says evidence_sufficient=false with E1 unresolved - it is final, never re-reviewed.
    assert e.sm.state == S.COMPLETED and len(orch.consistency_packets) == 1
    assert states(h).count("CONSISTENCY_REVIEW") == 1
    assert p2_attempts(e)[-1].review.evidence_sufficient is False

    class Replanning(MockOrchestrator):
        async def consistency_review(self, packet):
            self.consistency_packets.append(packet)
            return Review(ReviewDecision.REPLAN, "restructure", kind="CONSISTENCY_REVIEW")
    orch2 = Replanning(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1")]})
    h2 = V2Harness(tmp_path, make_settings(), orch=orch2, db="cr2.db")
    e2 = asyncio.run(h2.plan_and_start(erepo, with_p2([req("E1")])))
    assert e2.sm.state == S.PAUSED and "only PASS, RETRY, BLOCKED are allowed" in e2.run.status_message


def test_consistency_review_state_survives_a_restart(tmp_path, erepo):
    no_manifest(erepo)

    class Hanging(MockOrchestrator):
        async def consistency_review(self, packet):
            self.consistency_packets.append(packet)
            await asyncio.sleep(60)
    orch = Hanging(delay=0.0, checkpoint_script={"P2": [contradictory_pass("E1")]})
    s = make_settings()
    h = V2Harness(tmp_path, s, orch=orch, db="crash.db")

    async def first():
        e = h.engine
        await e.plan("g", str(erepo), pipeline="v2")
        await h.wait_idle()
        with_p2([req("E1")])(e)
        await e.start()
        while not orch.consistency_packets:
            await asyncio.sleep(0.02)
        await e.stop()
        await h.wait_idle()
        assert e.sm.state == S.STOPPED and e.run.resume_state == "CONSISTENCY_REVIEW"
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(first())
    conn = sqlite3.connect(tmp_path / "crash.db")  # a hard kill during the Claude call
    conn.execute("UPDATE runs SET state='CONSISTENCY_REVIEW', resume_state=NULL WHERE id=?", (run_id,))
    conn.commit()
    conn.close()
    orch2 = MockOrchestrator(delay=0.0, consistency_script={"P2": ["PASS"]})
    h2 = V2Harness(tmp_path, s, orch=orch2, db="crash.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.STOPPED and e.run.resume_state == "CONSISTENCY_REVIEW"
        assert e.run.pipeline_state["consistency"]["P2"]["status"] == "pending"
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(second())
    assert e.sm.state == S.COMPLETED
    assert not [c for c in orch2.checkpoints if c.phase.id == "P2"], "no second checkpoint after the restart"
    assert len(orch2.consistency_packets) == 1
    assert e.run.pipeline_state["consistency"]["P2"]["status"] == "resolved"


def test_consistency_prompt_is_bounded():
    ph = make_phase([req("E1")], gate=True)
    ck = contradictory_pass("E1").to_dict()
    ck["verification_notes"] = ["x" * 3000] * 6
    rounds = [{"request": {"attempt": 1, "reason": "r" * 900, "questions": ["q" * 400] * 6},
               "feedback": {"verdict": "ISSUE_FOUND", "note": "n" * 4000}, "followup": {}}] * 10
    pk = ConsistencyPacket(phase=ph, checkpoint=ck, evidence={"results": [
        {"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": UNAVAILABLE}]},
        trigger={"kind": "insufficient_evidence", "reason": "PASS while ...", "unresolved": ["E1"]},
        human_rounds=rounds, human_gate_open=True)
    prompt, stats = build_consistency_prompt(pk, ConsistencyLimits(6000))
    assert stats["prompt_bytes"] <= 6000 and stats["truncated"]
    assert "UNRESOLVED REQUIRED EVIDENCE: E1" in prompt and "DECIDE" in prompt and "question_targets" in prompt


def test_claude_consistency_and_mid_phase_calls_are_bounded_tool_free_and_validated(tmp_path):
    from app.config.settings import ClaudeSettings
    from app.models.guard import MID_PHASE_JSON_SCHEMA
    from app.orchestration.consistency import CONSISTENCY_JSON_SCHEMA
    from app.orchestration.impl_guard import MidPhasePacket
    from app.orchestration.providers.base import OrchestratorError
    from app.orchestration.providers.claude_code_cli import V2_PURPOSES, ClaudeCodeCLIProvider, limits_for
    fake = tmp_path / "fake_claude.py"
    fake.write_text("import json, sys\nsys.stdin.read()\npayload = json.loads(open(sys.argv[1]).read())\n"
                    "print(json.dumps({'type': 'result', 'subtype': 'success', 'is_error': False, 'num_turns': 1, "
                    "'usage': {}, 'structured_output': payload}), flush=True)\n")
    seen = []

    def prov_for(payload, s=None):
        (tmp_path / "payload.json").write_text(json.dumps(payload))
        prov = ClaudeCodeCLIProvider(s or ClaudeSettings(executable="unused"))

        async def argv(limits, schema):
            seen.append((limits, schema))
            return [PY, str(fake), str(tmp_path / "payload.json")]
        prov._build_argv = argv
        return prov
    ph = make_phase([req("E1")])
    pk = ConsistencyPacket(phase=ph, checkpoint=contradictory_pass("E1").to_dict(), evidence={"results": [
        {"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": UNAVAILABLE}]},
        trigger={"kind": "insufficient_evidence", "reason": "r", "unresolved": ["E1"]})
    ok = {"decision": "PASS", "summary": "s", "justification": "the unit test covers it", "evidence_dispositions": [
        {"requirement_id": "E1", "disposition": "substitute_evidence", "reason": "tests/test_car.py covers it"}],
        "problems": [], "corrective_instructions": []}
    s = ClaudeSettings(executable="unused", consistency_timeout_seconds=77, consistency_max_turns=3,
                       consistency_max_budget_usd=0.12)
    review = asyncio.run(prov_for(ok, s).consistency_review(pk))
    assert review.kind == "CONSISTENCY_REVIEW" and review.consistency["justification"] == "the unit test covers it"
    limits, schema = seen[-1]
    assert schema is CONSISTENCY_JSON_SCHEMA and "ASK_USER" not in schema["properties"]["decision"]["enum"]
    assert (limits.purpose, limits.tools, limits.max_tool_calls, limits.timeout, limits.max_turns,
            limits.max_budget_usd) == ("CONSISTENCY_REVIEW", "", 0, 77, 3, 0.12)
    with pytest.raises(OrchestratorError, match="invalid consistency review"):
        asyncio.run(prov_for(dict(ok, justification="", evidence_dispositions=[])).consistency_review(pk))
    mp = MidPhasePacket(phase=ph, reason="tool budget reached 50/50")
    d = asyncio.run(prov_for({"status": "CONTINUE", "blocker": "b", "diagnosis": "d", "next_actions": ["x"],
                              "avoid": []}).mid_phase_diagnose(mp))
    limits, schema = seen[-1]
    assert d.status == "CONTINUE" and schema is MID_PHASE_JSON_SCHEMA
    assert (limits.purpose, limits.tools, limits.max_tool_calls) == ("MID_PHASE_DIAGNOSE", "", 0)
    assert {"CONSISTENCY_REVIEW", "MID_PHASE_DIAGNOSE"} <= V2_PURPOSES
    assert limits_for("MID_PHASE_DIAGNOSE", ClaudeSettings()).timeout == 180
    with pytest.raises(OrchestratorError, match="invalid mid-phase diagnosis"):
        asyncio.run(prov_for({"status": "REPAIR"}).mid_phase_diagnose(mp))


# ============================================================ B. DERIVED HUMAN CRITERIA (first-class)

WHEELS = "Cars move but wheels do not visibly rotate."


def followup_retry(text, kind="runtime_behavior"):
    return Review(ReviewDecision.RETRY, "[TEST] the wheels must rotate", problems=[f"User observed: {WHEELS}"],
                  corrective_instructions=["Make the wheels rotate with the vehicle speed"],
                  derived_criteria=[{"text": text, "kind": kind, "rationale": "the user saw static wheels"}])


def test_human_issue_becomes_an_evidence_requirement_of_the_repair(tmp_path, erepo):
    text = "Vehicle wheels visibly rotate while the vehicle is moving."
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER", "EVIDENCE"]},
                            followup_script={"P2": [followup_retry(text)]}, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    original = {}

    def before(e):
        with_p2([req("E1")], gate=True)(e)
        original["criteria"] = list(phase(e, "P2").acceptance_criteria)

    async def main():
        e = await h.plan_and_start(erepo, before)
        assert e.sm.state == S.AWAITING_HUMAN
        await e.submit_human_feedback("ISSUE_FOUND", WHEELS)
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    p2 = e.db.load_run(e.run.id).plan.task("P2")  # persisted: survives a restart
    assert p2.acceptance_criteria == original["criteria"], "the user's original criteria are never mutated"
    [c] = p2.human_criteria
    assert (c.id, c.text, c.kind, c.verdict) == ("HF-P2-01", text, "runtime_behavior", "ISSUE_FOUND")
    assert c.request_id == "P2-a1" and c.feedback_ref.startswith("P2-a1@") and c.created_at > 0
    assert c.note == WHEELS and c.rationale == "the user saw static wheels"
    assert c.status == "verified" and c.evidence["status"] == SATISFIED and c.evidence["attempt"] == 2
    assert [x["event"] for x in c.history] == ["derived", "evidence"]
    assert [r.id for r in p2.evidence_requirements()] == ["E1", "HF-P2-01"]
    assert len(orch.evidence_plans) == 2, "the new requirement makes the next attempt plan again"
    replan_prompt = orch.evidence_plan_prompts[1]
    assert "HF-P2-01 [runtime_behavior, required, human-derived] Vehicle wheels visibly rotate" in replan_prompt
    ev = p2_attempts(e)[-1].evidence
    by = {r["requirement_id"]: r for r in ev["results"]}
    assert by["HF-P2-01"]["status"] == SATISFIED and by["HF-P2-01"]["human_derived"]
    later = [pk for pk in orch.checkpoints if pk.phase.id == "P2"][-1]
    prompt = build_checkpoint_context(later, CheckpointContextLimits()).prompt
    assert "HF-P2-01: Vehicle wheels visibly rotate while the vehicle is moving. [verified" in prompt
    assert any("human-derived criteria: HF-P2-01 verified" in x for x in logs(h, "EVIDENCE"))


def test_the_same_human_issue_reported_twice_is_not_duplicated():
    fb1 = hm.HumanFeedback("P2", "ISSUE_FOUND", "wheels static", [], 1.0, "P2-a1")
    fb2 = hm.HumanFeedback("P2", "ISSUE_FOUND", "still static", [], 2.0, "P2-a3")
    crits: list[hm.HumanCriterion] = []
    added, again = hm.merge_derived_criteria(crits, "P2", [{"text": "Vehicle wheels visibly rotate while the vehicle "
                                                                    "is moving."}], fb1, 1, "P2-a1")
    assert [c.id for c in added] == ["HF-P2-01"] and not again
    added, again = hm.merge_derived_criteria(crits, "P2", [
        {"text": "The vehicle's wheels visibly ROTATE while the vehicle moves!"},  # same criterion, other words
        {"text": "Front wheels visibly rotate while the vehicle is moving."},  # genuinely different
        {"text": "Vehicle wheels do not visibly rotate while the vehicle is moving."},  # the negation matters
    ], fb2, 3, "P2-a3")
    assert [c.id for c in added] == ["HF-P2-02", "HF-P2-03"] and [c.id for c in again] == ["HF-P2-01"]
    assert crits[0].reports == 2 and crits[0].feedback_refs == ["P2-a1@1.000", "P2-a3@2.000"]
    assert crits[0].history[-1]["event"] == "reported again"
    assert hm.criterion_key("A happens before B") != hm.criterion_key("B happens before A"), "order is meaning"
    assert hm.criterion_key("Wheels don't rotate") == hm.criterion_key("wheels do not rotate") != \
        hm.criterion_key("wheels rotate")
    again_json = hm.HumanCriterion.from_any(json.loads(json.dumps(crits[0].to_dict())))
    assert again_json == crits[0], "every provenance field survives persistence"
    old = hm.HumanCriterion.from_any({"id": "HF-P2-01", "text": "t", "verdict": "ISSUE_FOUND", "attempt": 1})
    assert (old.status, old.kind, old.reports, old.history) == ("open", "", 1, []), "pass 1 criteria load"


# ============================================================ C. EVIDENCE FRESHNESS

def executor_for(reg, repo, tmp_path, fp=("", ""), git_repo=None, **limits):
    base = dict(wall_seconds=60, step_timeout=30, max_steps=12, max_output_bytes=16000, max_artifacts=24,
                max_artifact_bytes=1_000_000, max_retries=0, ready_timeout=10)
    base.update(limits)
    lim = EvidenceLimits(**base)
    store = ArtifactStore(str(tmp_path / "artifacts"), 1, "P1", 1, max_artifacts=lim.max_artifacts,
                          max_bytes=lim.max_artifact_bytes)
    return EvidenceExecutor(reg, lim, store, repo_path=str(repo), persist=lambda: None, log=lambda s: None,
                            git=git_repo, repo_fp=fp[0], repo_fp_kind=fp[1])


def _fresh_plan(ph, snap, reg):
    probe, files = cap_id(snap, "runtime probe"), cap_id(snap, "repository file inspection")
    return validate_plan({"summary": "s", "requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": probe, "args": ["ok"], "expect_output": "RUNTIME OK"}]},
        {"requirement_id": "E2", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": files, "args": ["contains", "README.md", "readme"]}]}]}, ph, snap, reg, 12)


def test_evidence_is_reused_only_while_its_inputs_are_unchanged(erepo, tmp_path, monkeypatch):
    counter = tmp_path / "counter.txt"
    monkeypatch.setenv("EVIDENCE_COUNTER", str(counter))
    g = GitRepo(str(erepo))
    ph = make_phase([req("E1"), req("E2", kind="file_inspection")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = _fresh_plan(ph, snap, reg)
    fp1 = asyncio.run(repo_fingerprint(str(erepo), g))
    assert fp1[1] == "git_tree" and fp1[0].startswith("git:")
    state = EvidenceState("P1", attempt=1)
    b1 = asyncio.run(executor_for(reg, erepo, tmp_path, fp1, g).run(state, plan, snap))
    assert {r.requirement_id: (r.status, r.freshness) for r in b1.results} == \
        {"E1": (SATISFIED, "fresh"), "E2": (SATISFIED, "fresh")}
    prov = state.steps["E1.s1"]["provenance"]
    assert prov["repo_fp"] == fp1[0] and prov["scope"] == "repository" and prov["signature"]
    assert state.steps["E2.s1"]["provenance"]["scope"] == "files" and "README.md" in \
        state.steps["E2.s1"]["provenance"]["inputs"]
    assert b1.repo_fp == fp1[0] and b1.guard["kind"] == "git"
    # the next attempt with an unchanged repository: both results are reused, nothing runs again
    state.begin_attempt(2)
    assert len(state.reusable) == 2
    b2 = asyncio.run(executor_for(reg, erepo, tmp_path, fp1, g).run(state, plan, snap))
    assert {r.requirement_id: r.freshness for r in b2.results} == {"E1": "reused", "E2": "reused"}
    assert all(r.status == SATISFIED for r in b2.results) and counter.read_text().split() == ["ok"]
    assert state.steps["E1.s1"]["reused_from"] == 1 and b2.results[0].provenance["reused_from"] == 1
    assert "(reused from attempt 1)" in evm.result_line(b2.results[0])
    # an unrelated source change: the runtime evidence is stale and runs again; the README check stays valid
    (erepo / "slow.txt").write_text("31\n")
    fp3 = asyncio.run(repo_fingerprint(str(erepo), g))
    assert fp3[0] != fp1[0]
    state.begin_attempt(3)
    b3 = asyncio.run(executor_for(reg, erepo, tmp_path, fp3, g).run(state, plan, snap))
    assert {r.requirement_id: r.freshness for r in b3.results} == {"E1": "fresh", "E2": "reused"}
    assert counter.read_text().split() == ["ok", "ok"]
    # the file the check reads changes: it runs again and now shows the defect
    (erepo / "README.md").write_text("changed\n")
    fp4 = asyncio.run(repo_fingerprint(str(erepo), g))
    state.begin_attempt(4)
    b4 = asyncio.run(executor_for(reg, erepo, tmp_path, fp4, g).run(state, plan, snap))
    assert {r.requirement_id: (r.status, r.freshness) for r in b4.results}["E2"] == (FAILED, "fresh")


def test_stored_evidence_goes_stale_after_a_relevant_change_and_never_satisfies(erepo, tmp_path):
    g = GitRepo(str(erepo))
    ph = make_phase([req("E1"), req("E2", kind="file_inspection")])
    reg, snap = asyncio.run(discover(erepo, ph))
    fp = asyncio.run(repo_fingerprint(str(erepo), g))
    bundle = asyncio.run(executor_for(reg, erepo, tmp_path, fp, g).run(EvidenceState("P1", attempt=1),
                                                                       _fresh_plan(ph, snap, reg), snap))
    results = [r.to_dict() for r in bundle.results]
    assert efresh.refresh(results, repo_fp=fp[0], repo_path=str(erepo)) == [], "unchanged: still fresh"
    (erepo / "state.txt").write_text("changed\n")
    fp2 = asyncio.run(repo_fingerprint(str(erepo), g))
    went = efresh.refresh(results, repo_fp=fp2[0], repo_path=str(erepo))
    assert went == [("E1", "the repository changed after it ran")], "the README check read nothing that changed"
    e1, e2 = results
    assert (e1["status"], e1["freshness"], e1["stale_status"], e1["cause"]) == (INCONCLUSIVE, STALE, SATISFIED, "stale")
    assert not evm.is_satisfied(e1) and evm.is_satisfied(e2)
    assert [r["requirement_id"] for r in evm.unresolved_required(results)] == ["E1"]
    assert "(stale)" in evm.result_line(e1) and "STALE (the repository changed after it ran)" in evm.result_line(e1)
    restored = evm.EvidenceBundle.from_dict(json.loads(json.dumps(dict(bundle.to_dict(), results=results))))
    assert restored.results[0].freshness == STALE and restored.results[0].stale_status == SATISFIED, \
        "freshness state is persisted"


def test_stale_evidence_after_a_change_while_waiting_for_the_user_is_not_recorded_as_verified(tmp_path, erepo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        e = await h.plan_and_start(erepo, with_p2([req("E1")], gate=True))
        assert e.sm.state == S.AWAITING_HUMAN
        assert p2_attempts(e)[-1].evidence["results"][0]["status"] == SATISFIED
        (erepo / "state.txt").write_text("edited while the user was looking\n")  # the repository changes
        await e.submit_human_feedback("LOOKS_GOOD", "fine")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    r = p2_attempts(e)[-1].evidence["results"][0]
    assert (r["status"], r["freshness"], r["stale_status"]) == (INCONCLUSIVE, STALE, SATISFIED)
    assert orch.followups[0].evidence["results"][0]["freshness"] == STALE, "the follow-up saw the stale evidence"
    p2 = e.db.load_run(e.run.id).plan.task("P2")
    assert any(x.startswith("UNVERIFIED E1 [runtime_behavior]: evidence INCONCLUSIVE (stale)")
               for x in p2.result.remaining_risks)
    assert "evidence E1 [runtime_behavior]: INCONCLUSIVE (stale)" in p2.result.verification
    assert any("evidence went stale before the human follow-up" in x for x in logs(h, "EVIDENCE"))


def test_a_crash_with_a_changed_repository_reruns_the_completed_steps(tmp_path, erepo, monkeypatch):
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
    h = V2Harness(tmp_path, s, orch=orch, db="stale.db")

    async def first():
        e = h.engine
        await e.plan("g", str(erepo), pipeline="v2")
        await h.wait_idle()
        with_p2([req("E1"), req("E2")])(e)
        await e.start()
        while e.run.pipeline_state.get("evidence", {}).get("P2", {}).get("inflight") != "E2.s1":
            await asyncio.sleep(0.02)
        await asyncio.sleep(0.3)
        await e.stop()
        await h.wait_idle()
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(first())
    (erepo / "slow.txt").write_text("0.1\n")  # a tracked file changes while the app is down
    h2 = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0, evidence_aware=True), db="stale.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(second())
    assert e.sm.state == S.COMPLETED
    assert counter.read_text().split() == ["ok", "slow", "ok", "slow"], "the completed E1 step was stale: re-run"
    assert any("E1.s1 ran before a change (the repository changed after it ran)" in x
               for x in p2_attempts(e)[-1].evidence["notes"])


# ============================================================ D. BOUNDED REPLANNING

REPLAN_MANIFEST = dict(MANIFEST, services=MANIFEST["services"] + [
    {"name": "silent service 2", "command": f'"{PY}" checks/service_silent.py second', "ready_output": "READY",
     "ready_timeout_seconds": 1}])


def _service_plan(service_name):
    def plan(pk):
        return {"summary": f"observe with {service_name}", "requirements": [
            {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct", "reason": "",
             "resources": [next(c.id for c in pk.snapshot.capabilities if c.name == service_name)],
             "steps": [{"capability_id": next(c.id for c in pk.snapshot.usable() if c.name == "runtime probe"),
                        "args": ["ok"], "expect_output": "RUNTIME OK"}]}]}
    return plan


def test_a_strategy_that_cannot_be_executed_is_replanned_once(tmp_path, erepo):
    (erepo / ".aistudio" / "evidence.json").write_text(json.dumps(REPLAN_MANIFEST))
    git(erepo, "commit", "-qam", "manifest")
    orch = MockOrchestrator(delay=0.0, evidence_aware=True, evidence_plan_script={"P2": [
        _service_plan("silent service"), _service_plan("app service")]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED
    assert len(orch.evidence_plans) == 2 and orch.evidence_plans[1].replan["requirements"] == ["E1"]
    assert "never ready" in " ".join(orch.evidence_plans[1].replan["failures"]) or \
        "not ready within 1s" in " ".join(orch.evidence_plans[1].replan["failures"])
    assert "REPLAN 1" in orch.evidence_plan_prompts[1]
    silent = next(c for c in orch.evidence_plans[1].snapshot.capabilities if c.name == "silent service")
    assert silent.confidence == UNAVAILABLE and "failed during evidence collection" in silent.reason, \
        "the failed strategy cannot be chosen again"
    st = e.run.pipeline_state["evidence"]["P2"]
    assert st["replans"] == 1 and st["replan_log"][0]["requirements"] == ["E1"]
    ev = p2_attempts(e)[-1].evidence
    assert ev["results"][0]["status"] == SATISFIED and ev["results"][0]["steps"] == ["E1.r1.s1"]
    assert ev["replans"][0]["generation"] == 1 and ev["plan_generation"] == 1
    assert any("replan 1/1 for E1" in x for x in logs(h, "EVIDENCE"))


def test_a_functional_failure_is_evidence_of_a_defect_not_a_reason_to_replan(tmp_path, erepo):
    (erepo / "state.txt").write_text("broken\n")
    git(erepo, "commit", "-qam", "broken")
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED"]}, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert len(orch.evidence_plans) == 1 and e.run.pipeline_state["evidence"]["P2"]["replans"] == 0
    r = p2_attempts(e)[-1].evidence["results"][0]
    assert (r["status"], r["cause"]) == (FAILED, "product")


def test_the_replan_limit_is_enforced_and_unknown_capabilities_stay_rejected(tmp_path, erepo):
    (erepo / ".aistudio" / "evidence.json").write_text(json.dumps(REPLAN_MANIFEST))
    git(erepo, "commit", "-qam", "manifest")

    def invented(pk):
        return {"summary": "invented", "requirements": [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                                         "coverage": "direct", "reason": "",
                                                         "steps": [{"capability_id": "cap-999"}]}]}
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED"]}, evidence_plan_script={"P2": [
        _service_plan("silent service"), _service_plan("silent service 2")]})
    h = V2Harness(tmp_path, make_settings(evidence_max_replans=1), orch=orch)
    e = asyncio.run(h.plan_and_start(erepo, with_p2([req("E1")])))
    assert len(orch.evidence_plans) == 2, "exactly one replan, never a loop"
    assert e.run.pipeline_state["evidence"]["P2"]["replans"] == 1
    assert any("replan limit (1) reached" in x for x in logs(h, "EVIDENCE"))
    r = p2_attempts(e)[-1].evidence["results"][0]
    assert r["status"] == INCONCLUSIVE and r["cause"] == "strategy"
    orch2 = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED"]}, evidence_plan_script={"P2": [
        _service_plan("silent service"), invented]})
    h2 = V2Harness(tmp_path, make_settings(), orch=orch2, db="inv.db")
    e2 = asyncio.run(h2.plan_and_start(erepo, with_p2([req("E1")])))
    r2 = p2_attempts(e2)[-1].evidence["results"][0]
    assert r2["status"] == UNAVAILABLE and r2["source"] == "plan" and "unknown capability 'cap-999'" in \
        " ".join(e2.run.pipeline_state["evidence"]["P2"]["plan"]["requirements"][0]["notes"])
    orch3 = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED"]}, evidence_plan_script={"P2": [
        _service_plan("silent service")]})
    h3 = V2Harness(tmp_path, make_settings(evidence_max_replans=0), orch=orch3, db="zero.db")
    asyncio.run(h3.plan_and_start(erepo, with_p2([req("E1")])))
    assert len(orch3.evidence_plans) == 1, "replanning disabled"


# ============================================================ E. READINESS PROBES

TCP_SERVICE = ("import socket, sys\n"
               "s = socket.socket()\n"
               "s.bind(('127.0.0.1', int(sys.argv[1])))\n"
               "s.listen(5)\n"
               "print('listening', flush=True)\n"
               "while True:\n"
               "    c, _ = s.accept()\n"
               "    c.close()\n")
MARKER_SERVICE = "import time\nprint('booting', flush=True)\nprint('APP READY', flush=True)\ntime.sleep(30)\n"
FLAG_SERVICE = "import pathlib, time\ntime.sleep(0.3)\npathlib.Path('ready.flag').write_text('x')\ntime.sleep(30)\n"
SLEEPER = "import time\ntime.sleep(30)\n"
CRASHER = "import sys\nsys.exit(3)\n"


def free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


def test_readiness_probes_are_generic_bounded_and_always_cleaned_up(tmp_path):
    for name, body in (("tcp_service.py", TCP_SERVICE), ("marker.py", MARKER_SERVICE), ("flag.py", FLAG_SERVICE),
                       ("sleeper.py", SLEEPER), ("crasher.py", CRASHER)):
        (tmp_path / name).write_text(body)
    tcp_port, http_port, unused = free_port(), free_port(), free_port()
    cases = [
        ("process_alive", f'"{PY}" sleeper.py', rd.parse_readiness(None)[0], "ready"),
        ("stdout", f'"{PY}" marker.py', rd.parse_readiness({"kind": "stdout_contains", "text": "APP READY"})[0],
         "ready"),
        ("tcp", f'"{PY}" tcp_service.py {tcp_port}', rd.parse_readiness({"kind": "tcp_port", "port": tcp_port})[0],
         "ready"),
        ("http", f'"{PY}" -m http.server {http_port} --bind 127.0.0.1',
         rd.parse_readiness({"kind": "http_status", "url": f"http://127.0.0.1:{http_port}/", "expected_status": 200})[0],
         "ready"),
        ("file", f'"{PY}" flag.py', rd.parse_readiness({"kind": "file_exists", "path": "ready.flag"})[0], "ready"),
        ("timeout", f'"{PY}" sleeper.py', rd.parse_readiness({"kind": "tcp_port", "port": unused,
                                                             "timeout_seconds": 1})[0], "timeout"),
        ("exited", f'"{PY}" crasher.py', rd.parse_readiness({"kind": "tcp_port", "port": unused})[0], "exited"),
    ]

    async def main():
        out = {}
        for name, cmd, spec, _want in cases:
            sp = ServiceProcess(cmd, str(tmp_path))
            await sp.start([spec["text"]] if spec["kind"] == "stdout_contains" else [])
            try:
                out[name] = (await rd.wait_ready(sp, spec, 15, str(tmp_path)), sp.pid)
            finally:
                await sp.stop()
        return out
    out = asyncio.run(main())
    for name, _cmd, _spec, want in cases:
        (status, detail), pid = out[name]
        assert status == want, (name, status, detail)
        assert not procutil.is_alive(pid), f"{name}: the process is always stopped"
    assert "accepts connections" in out["tcp"][0][1] and "answered HTTP 200" in out["http"][0][1]
    assert "not ready within 1s" in out["timeout"][0][1] and "exited with code 3" in out["exited"][0][1]
    assert "ready.flag exists" in out["file"][0][1]


def test_unsafe_or_invalid_readiness_declarations_are_rejected():
    bad = [({"kind": "tcp_port", "host": "example.com", "port": 80}, "limited to this machine"),
           ({"kind": "tcp_port", "host": "10.0.0.5", "port": 80}, "limited to this machine"),
           ({"kind": "tcp_port", "port": 0}, "between 1 and 65535"),
           ({"kind": "tcp_port", "port": 70000}, "between 1 and 65535"),
           ({"kind": "http_status", "url": "http://10.0.0.5:80/"}, "limited to this machine"),
           ({"kind": "http_status", "url": "http://user:secret@127.0.0.1:8080/"}, "credentials"),
           ({"kind": "http_status", "url": "ftp://127.0.0.1/"}, "only http"),
           ({"kind": "http_status", "url": "http://127.0.0.1:8080/", "expected_status": 99}, "status code"),
           ({"kind": "file_exists", "path": "../outside.flag"}, "repository-relative"),
           ({"kind": "file_exists", "path": ".git/HEAD"}, "repository-relative"),
           ({"kind": "shell", "command": "curl http://x"}, "unknown readiness kind"),
           ("READY", "must be an object")]
    for raw, why in bad:
        spec, err = rd.parse_readiness(raw)
        assert not spec and why in err, (raw, err)
    status, _body, err = rd.http_get("http://example.com/", 1)
    assert status is None and "limited to this machine" in err, "never a request to another host"
    assert rd.redact_url("http://127.0.0.1:5173/health?token=abc#x") == "http://127.0.0.1:5173/health"
    spec, _ = rd.parse_readiness({"kind": "http_status", "url": "http://localhost:5173/health", "timeout_seconds": 9999})
    assert spec["timeout_seconds"] == 600, "timeouts are bounded"


def test_a_service_with_an_invalid_readiness_declaration_is_unavailable(erepo):
    m = dict(MANIFEST, services=[{"name": "remote probe", "command": f'"{PY}" checks/service.py',
                                  "readiness": {"kind": "http_status", "url": "http://192.168.1.10/"}},
                                 {"name": "port service", "command": f'"{PY}" checks/service.py port',
                                  "readiness": {"kind": "tcp_port", "port": 5173}}])
    (erepo / ".aistudio" / "evidence.json").write_text(json.dumps(m))
    _reg, snap = asyncio.run(discover(erepo, make_phase([req("E1")])))
    by = {c.name: c for c in snap.capabilities}
    assert by["remote probe"].confidence == UNAVAILABLE and "invalid readiness declaration" in by["remote probe"].reason
    port = by["port service"]
    assert port.confidence != UNAVAILABLE and port.constraints["readiness"]["kind"] == "tcp_port"
    assert "accepts TCP connections on 127.0.0.1:5173" in capability_line(port, snap)


# ============================================================ F. ARTIFACT RETENTION

def test_artifact_paths_are_contained_and_size_limits_hold(tmp_path):
    root = tmp_path / "evidence"
    store = ArtifactStore(str(root), 7, "../../escape", 1, max_artifacts=3, max_bytes=2000, max_run_bytes=3000)
    a = store.save_text("log", "../../../x", "y" * 1500, "d", "cap-001")
    assert a is not None and inside(root, a.path_or_ref) and Path(a.path_or_ref).parent.parent.parent.name == "run-7"
    assert not inside(root, tmp_path / "outside.txt") and not inside(root, root / ".." / "x") and not inside(root, root)
    assert store.save_bytes("image", "big", b"x" * 2001, "image/png", "d", "cap-002") is None, \
        "binary artifacts are never truncated: too large means not stored"
    img = store.save_bytes("image", "shot", b"\x89PNG" + b"0" * 996, "image/png", "d", "cap-002",
                           {"width": 4, "height": 2})
    assert img is not None and img.path_or_ref.endswith(".png") and img.media_type == "image/png"
    assert img.metadata == {"width": 4, "height": 2}
    assert store.save_text("log", "third", "z" * 800, "d", "cap-001") is None, "the per-run byte budget is enforced"
    assert store.dropped == 2 and any("per-run artifact budget" in r for r in store.drop_reasons)


def test_retention_keeps_active_runs_cleans_expired_and_never_touches_the_project(tmp_path):
    root, project = tmp_path / "evidence", tmp_path / "proj"
    project.mkdir()
    old = time.time() - 30 * 86400

    def put(rel, size=10, mtime=old):
        p = root.joinpath(*rel.split("/"))
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x" * size)
        os.utime(p, (mtime, mtime))
        return p
    active = put("run-1/P1/a1/art-1.log")
    expired = put("run-2/P1/a1/art-2.log")
    recent = put("run-2/P1/a1/art-3.log", mtime=time.time())
    stray_guard = put(".guard/guard-x/README.md")
    project_file = project / "keep.txt"
    project_file.write_text("mine")
    os.utime(project_file, (old, old))
    rep = cleanup(root, protected_runs={1}, referenced={str(project_file)},
                  policy=RetentionPolicy(max_age_days=14, max_run_bytes=0, max_total_bytes=0))
    assert active.exists() and recent.exists() and not expired.exists() and not stray_guard.exists()
    assert project_file.exists(), "cleanup never looks outside the evidence root"
    assert rep.protected == 1 and len(rep.deleted) == 2 and not (root / ".guard").exists()
    # per-run and total byte budgets delete the oldest files of finished runs only
    now = time.time()
    run3 = [put(f"run-3/P1/a1/art-{i}.log", 100, now - 100 + i) for i in range(3)]
    big_active = put("run-1/P1/a1/big.log", 500, now - 1000)
    cleanup(root, protected_runs={1}, policy=RetentionPolicy(max_age_days=0, max_run_bytes=150, max_total_bytes=0))
    assert [p.exists() for p in run3] == [False, False, True] and big_active.exists()
    cleanup(root, protected_runs={1}, policy=RetentionPolicy(max_age_days=0, max_run_bytes=0, max_total_bytes=1))
    assert big_active.exists() and active.exists(), "an active run's artifacts survive even the total budget"
    assert not recent.exists() and not run3[2].exists()
    again = cleanup(root, protected_runs={1}, policy=RetentionPolicy())
    assert again.deleted == [], "idempotent: a crash midway just leaves less to do"


def test_engine_cleanup_protects_resumable_runs(tmp_path):
    from app.models.run import Run
    from app.orchestration.orchestrator import Engine
    e = Engine(make_settings(), lambda k, p: None, str(tmp_path / "ret.db"))
    done = Run(id=None, goal="g", repo_path=str(tmp_path), state="COMPLETED")
    paused = Run(id=None, goal="g", repo_path=str(tmp_path), state="PAUSED")
    e.db.create_run(done)
    e.db.create_run(paused)
    old = time.time() - 40 * 86400
    files = {}
    for r in (done, paused):
        p = Path(e.artifact_root) / f"run-{r.id}" / "P1" / "a1" / "art.log"
        p.parent.mkdir(parents=True)
        p.write_text("x")
        os.utime(p, (old, old))
        files[r.state] = p
    rep = asyncio.run(e.cleanup_artifacts())
    assert not files["COMPLETED"].exists() and files["PAUSED"].exists() and len(rep.deleted) == 1
    e.db.close()


# ============================================================ G. NON-GIT MUTATION SAFETY

def plain_repo(tmp_path):
    from tests.test_evidence import PROBE, SERVICE
    r = tmp_path / "plain"
    (r / "checks").mkdir(parents=True)
    (r / ".aistudio").mkdir()
    (r / "README.md").write_text("readme\n")
    (r / "state.txt").write_text("ok\n")
    (r / "slow.txt").write_text("30\n")
    (r / "checks" / "probe.py").write_text(PROBE)
    (r / "checks" / "build.py").write_text('print("build ok")\n')
    (r / "checks" / "service.py").write_text(SERVICE)
    (r / ".aistudio" / "evidence.json").write_text(json.dumps(MANIFEST))
    return r


def _mutate_plan(ph, snap, reg):
    return validate_plan({"summary": "s", "requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": cap_id(snap, "runtime probe"), "args": ["mutate"],
                    "expect_output": "RUNTIME OK"}]}]}, ph, snap, reg, 12)


def test_non_git_mutation_is_detected_and_restored_from_a_bounded_snapshot(tmp_path):
    repo = plain_repo(tmp_path)
    ph = make_phase([req("E1")])
    reg, snap = asyncio.run(discover(repo, ph))
    fp = asyncio.run(repo_fingerprint(str(repo)))
    assert fp[1] == "manifest" and fp[0].startswith("mf:")
    guard = ManifestMutationGuard(str(repo), str(tmp_path / "guards"), max_files=100, max_snapshot_bytes=10_000_000)
    ex = EvidenceExecutor(reg, EvidenceLimits(wall_seconds=60, step_timeout=30), ArtifactStore(
        str(tmp_path / "art"), 1, "P1", 1, max_artifacts=5, max_bytes=100000), repo_path=str(repo),
        persist=lambda: None, log=lambda s: None, guard=guard, repo_fp=fp[0], repo_fp_kind=fp[1])
    b = asyncio.run(ex.run(EvidenceState("P1", attempt=1), _mutate_plan(ph, snap, reg), snap))
    r = b.results[0]
    assert r.status == INCONCLUSIVE and r.cause == "mutation" and "(restored)" in r.failure_reason
    assert (repo / "README.md").read_text() == "readme\n" and not (repo / "created_by_evidence.txt").exists()
    assert b.guard["kind"] == "manifest" and b.guard["exact"] and "restorable" in b.guard["detail"]
    assert any("restored to the pre-evidence state" in n and "README.md" in n for n in b.notes)
    assert asyncio.run(repo_fingerprint(str(repo)))[0] == fp[0], "the directory is exactly as before"
    assert not list((tmp_path / "guards").glob("guard-*")), "the bounded snapshot is removed afterwards"


def test_non_git_protection_that_cannot_restore_says_so(tmp_path):
    repo = plain_repo(tmp_path)
    ph = make_phase([req("E1")])
    reg, snap = asyncio.run(discover(repo, ph))
    guard = ManifestMutationGuard(str(repo), str(tmp_path / "guards"), max_files=100, max_snapshot_bytes=1)
    ex = EvidenceExecutor(reg, EvidenceLimits(wall_seconds=60, step_timeout=30), ArtifactStore(
        str(tmp_path / "art"), 1, "P1", 1, max_artifacts=5, max_bytes=100000), repo_path=str(repo),
        persist=lambda: None, log=lambda s: None, guard=guard)
    b = asyncio.run(ex.run(EvidenceState("P1", attempt=1), _mutate_plan(ph, snap, reg), snap))
    r = b.results[0]
    assert r.status == INCONCLUSIVE and "NOT restored" in r.failure_reason
    assert (repo / "README.md").read_text() == "mutated by an evidence step\n", "nothing is claimed that did not happen"
    assert not b.guard["exact"] and "restoration cannot be guaranteed" in b.guard["detail"]
    assert any("restoration cannot be guaranteed" in n for n in b.notes)
    partial = ManifestMutationGuard(str(repo), "", max_files=2)
    token = asyncio.run(partial.before())
    assert partial.describe(token)["partial"] and "first 2 files only" in partial.describe(token)["detail"]
    assert manifest_scan(str(repo), max_files=2).truncated
    assert asyncio.run(repo_fingerprint(str(repo), max_files=2)) == ("", ""), "beyond the budget: unknown, not faked"


def test_a_non_git_phase_reports_the_protection_it_actually_had(tmp_path):
    repo = plain_repo(tmp_path)
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo, with_p2([req("E1")])))
    assert e.sm.state == S.COMPLETED
    ev = e.db.load_attempts(e.run.id)["P2"][-1].evidence
    assert ev["results"][0]["status"] == SATISFIED and ev["guard"]["kind"] == "manifest" and ev["guard"]["exact"]
    assert ev["repo_fp_kind"] == "manifest"


# ============================================================ M. PROVENANCE

def test_capability_and_step_provenance_answer_who_what_when_and_which_step(erepo, tmp_path):
    ph = make_phase([req("E1")])
    reg = default_registry()
    ctx = DiscoveryContext(str(erepo), ph, [ph], {}, make_settings().orchestration,
                           changed_files=[".aistudio/evidence.json"])
    snap = asyncio.run(reg.discover(ctx))
    probe = next(c for c in snap.capabilities if c.name == "runtime probe")
    p = probe.provenance
    assert p["discovered_by"] == "command" and p["provider_version"] == "2" and p["discovered_at"] > 0
    assert p["declared_in"] == ".aistudio/evidence.json" and len(p["declared_sha1"]) == 16
    assert p["changed_in_phase"] is True and os.path.basename(p["executable"]).lower().startswith("python")
    assert "DECLARED IN A FILE THE WORKER CHANGED IN THIS PHASE" in capability_line(probe, snap)
    files = next(c for c in snap.capabilities if c.name == "repository file inspection")
    assert files.provenance["discovered_by"] == "repository_files"
    allowed = {"discovered_by", "provider_version", "discovered_at", "declared_in", "declared_sha1", "changed_in_phase",
               "executable", "source_detail", "validated_at", "used_by", "confirmed_by", "runtime", "engine"}
    assert all(set(c.provenance) <= allowed for c in snap.capabilities), "no environment dumps, no secrets"
    plan = validate_plan({"summary": "s", "requirements": [{"requirement_id": "E1", "strategy": "execute", "goal": "g",
                                                            "coverage": "direct", "steps": [
                                                                {"capability_id": probe.id, "args": ["ok"]}]}]},
                         ph, snap, reg, 12)
    state = EvidenceState("P1", attempt=1)
    discovered_at = p["discovered_at"]
    bundle = asyncio.run(executor_for(reg, erepo, tmp_path).run(state, plan, snap))
    assert probe.provenance["used_by"] == ["E1.s1"] and probe.provenance["confirmed_by"] == "executed as E1.s1"
    assert probe.provenance["validated_at"] >= discovered_at
    sp = state.steps["E1.s1"]["provenance"]
    assert (sp["capability_key"], sp["provider"], sp["provider_version"]) == (probe.key, "command", "2")
    assert sp["signature"] and sp["capability_fp"] == efresh.capability_fp(probe)
    used = bundle.capabilities[0]
    assert used["provider"] == "command" and used["provenance"]["declared_in"] == ".aistudio/evidence.json"
    again = evm.CapabilitySnapshot.from_dict(json.loads(json.dumps(snap.to_dict())))
    assert again.get(probe.id).provenance == probe.provenance, "provenance is persisted"


# ============================================================ N. HUMAN GATE QUALITY

def test_human_questions_are_deduplicated_bounded_and_skip_what_automation_proved():
    results = [{"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": SATISFIED,
                "freshness": "fresh", "summary": "the probe passed"},
               {"requirement_id": "E2", "kind": "visual_evidence", "required": True, "status": UNAVAILABLE,
                "goal": "the rear segments follow the jump"}]
    qs, ts, dropped = hm.select_questions(
        ["While moving forward, do the rear body segments visibly follow the jump?",
         "while moving forward do the rear body segments visibly follow the jump",
         "Does the probe pass?", "Is the head rising first?", "Anything else?"],
        ["E2", "E2", "E1", "E2", "E9"], results, {"E1", "E2"}, max_questions=2)
    assert qs == ["While moving forward, do the rear body segments visibly follow the jump?",
                  "Is the head rising first?"] and ts == ["E2", "E2"]
    joined = " | ".join(dropped)
    assert "duplicate" in joined and "already established by automation (E1)" in joined and "over the limit" in joined
    only, targets, _ = hm.select_questions(["Does the probe pass?"], ["E1"], results, {"E1", "E2"})
    assert only == ["Does the probe pass?"] and targets == ["E1"], "never an empty request"
    _q, unknown, _ = hm.select_questions(["q?"], ["E9"], results, {"E1", "E2"})
    assert unknown == [], "targets are validated against the phase's requirements"
    ph = make_phase([req("E1"), req("E2", kind="visual_evidence")], gate=True)
    review = Review(ReviewDecision.ASK_USER, "s", human_request={
        "reason": "r", "questions": ["Do the wheels rotate?", "Does the probe pass?"], "question_targets": ["E2", "E1"],
        "observations": []})
    rq = hm.build_request(ph, 1, review, False, [], {"results": results}, max_questions=4)
    assert rq.questions == ["Do the wheels rotate?"] and rq.targets == ["E2"]
    assert rq.established == ["E1 [runtime_behavior]: the probe passed"]
    assert rq.missing == ["E2 [visual_evidence]: the rear segments follow the jump"]
    again = hm.HumanRequest.from_dict(json.loads(json.dumps(rq.to_dict())))
    assert again == rq


def test_human_panel_shows_established_missing_and_questions_compactly():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from app.gui.human_gate import HumanGatePanel
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    panel = HumanGatePanel()
    panel.set_request({"status": "awaiting", "request": {
        "phase_id": "P2", "attempt": 1, "reason": "only a person can see it", "questions": ["Do the wheels rotate?"],
        "targets": ["E2"], "established": ["E1 [runtime_behavior]: the probe passed"],
        "missing": ["E2 [visual_evidence]: the wheels rotate"], "evidence": ["E1 ..."]}}, "AWAITING_HUMAN")
    text = panel.view.toPlainText()
    for part in ["PLEASE CHECK", "[E2] Do the wheels rotate?", "AUTOMATION COULD NOT ESTABLISH",
                 "E2 [visual_evidence]: the wheels rotate", "AUTOMATION ESTABLISHED", "the probe passed"]:
        assert part in text, part
    assert text.index("PLEASE CHECK") < text.index("AUTOMATION COULD NOT ESTABLISH") < \
        text.index("AUTOMATION ESTABLISHED")


# ============================================================ Q. PROMPT / TOKEN CONTROL

BIG_BUILD = "\n".join([
    "import sys",
    "print('Building MyGameEditor...')",
    "ERR = 'D:' + chr(92) + 'Proj' + chr(92) + 'Source' + chr(92) + 'Game' + chr(92) + 'Car.cpp(42,7): error C2065: "
    "' + chr(39) + 'wheelSpeed' + chr(39) + ': undeclared identifier'",
    "print(ERR)",
    "for i in range(30000):",
    "    print('noise line %05d compiling module chunk' % i)",
    "print(ERR)",
    "print('Result: Failed (OtherCompilationError)')",
    "sys.exit(6)",
    ""])


def test_a_huge_build_log_reaches_the_checkpoint_only_as_compact_diagnostics(erepo, tmp_path):
    from app.models.attempt import Attempt
    from app.orchestration.checkpoint import CheckpointPackage
    (erepo / "checks" / "bigbuild.py").write_text(BIG_BUILD)
    m = dict(MANIFEST, commands=MANIFEST["commands"] + [{"name": "game build", "command": f'"{PY}" checks/bigbuild.py',
                                                         "kind": "build", "timeout_seconds": 60}])
    (erepo / ".aistudio" / "evidence.json").write_text(json.dumps(m))
    ph = make_phase([req("E1", kind="build")])
    reg, snap = asyncio.run(discover(erepo, ph))
    plan = validate_plan({"summary": "s", "requirements": [
        {"requirement_id": "E1", "strategy": "execute", "goal": "g", "coverage": "direct",
         "steps": [{"capability_id": cap_id(snap, "game build")}]}]}, ph, snap, reg, 12)
    state = EvidenceState("P1", attempt=1)
    bundle = asyncio.run(executor_for(reg, erepo, tmp_path, max_output_bytes=8000, max_artifact_bytes=200_000)
                         .run(state, plan, snap))
    r = bundle.results[0]
    assert r.status == FAILED and r.cause == "product"
    assert r.diagnostics[0].startswith("build failed (MyGameEditor): 2 error(s), 1 distinct")
    assert r.diagnostics[1] == "compiler error C2065 Source/Game/Car.cpp(42): 'wheelSpeed': undeclared identifier (x2)"
    assert "C2065" in r.summary, "the step summary names the root error, not the last noise line"
    art = Path(bundle.artifacts[0].path_or_ref)
    assert art.is_file() and art.stat().st_size <= 200_000 + 200, "the full (bounded) log is an artifact"
    log = art.read_text()
    assert log.index("error C2065") < log.index("bytes of output omitted") < log.index("Result: Failed"), \
        "a build log keeps its head (root errors) and its end"
    assert len(json.dumps(state.to_dict())) < 30000, "the run state never holds the log"
    pk = CheckpointPackage(phase=ph, attempt=Attempt(task_id="P1", number=1, evidence=bundle.to_dict()), max_attempts=3,
                           evidence=dict(bundle.to_dict(), discovered=[]))
    prompt = build_checkpoint_context(pk, CheckpointContextLimits()).prompt
    assert "| compiler error C2065 Source/Game/Car.cpp(42)" in prompt and "noise line" not in prompt
    assert f"{bundle.artifacts[0].id} [log]" in prompt and len(prompt.encode()) <= 60000


# ============================================================ R. UI

def test_phase_details_show_freshness_replans_protection_and_the_consistency_review():
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from app.gui.task_details import TaskDetails
    QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    ph = make_phase([req("E1"), req("E2", kind="visual_evidence")], gate=True)
    ph.human_criteria = [hm.HumanCriterion("HF-P1-01", "the wheels rotate", "ISSUE_FOUND", "static", 1, 1.0,
                                           kind="visual_evidence", status="verified", reports=2,
                                           evidence={"status": SATISFIED, "attempt": 2})]
    ev = {"results": [
        {"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": INCONCLUSIVE,
         "freshness": STALE, "stale_status": SATISFIED, "stale_reason": "the repository changed after it ran",
         "cause": "stale", "capability_ids": ["cap-001"]},
        {"requirement_id": "HF-P1-01", "kind": "visual_evidence", "required": True, "status": SATISFIED,
         "freshness": "reused", "provenance": {"reused_from": 1}, "human_derived": True, "capability_ids": ["cap-002"],
         "summary": "analysis passed", "diagnostics": ["build failed (X): 1 error(s)"]}],
        "replans": [{"generation": 1, "requirements": ["E1"], "reasons": ["E1 (cap-004): never ready"]}],
        "guard": {"kind": "manifest", "exact": True, "detail": "bounded manifest of 12 files with an exact snapshot"},
        "capabilities": [{"id": "cap-001", "kind": "runtime_behavior", "name": "probe", "confidence": "CONFIRMED",
                          "provenance": {"discovered_by": "command", "changed_in_phase": True}}],
        "artifacts": [{"id": "art-1", "kind": "image", "media_type": "image/png", "description": "shot",
                       "path_or_ref": "C:/x/a.png"}], "capability_summary": "5 capabilities found"}
    review = {"decision": "PASS", "kind": "CONSISTENCY_REVIEW", "evidence_sufficient": False,
              "consistency": {"trigger": {"reason": "PASS while evidence_sufficient=false"},
                              "justification": "the unit test covers E1",
                              "dispositions": [{"requirement_id": "E1", "disposition": "substitute_evidence",
                                                "reason": "unit test"}]}}
    w = TaskDetails()
    w.show_task(ph.to_dict(), [{"number": 2, "kind": "repair", "started_at": 0, "evidence": ev, "review": review}])
    text = w.toPlainText()
    for part in ["STALE (was SATISFIED)", "the repository changed after it ran", "reused (attempt 1)", "HF-P1-01",
                 "Replans", "replan 1: E1", "repository protection: bounded manifest", "Consistency review",
                 "the unit test covers E1", "E1: substitute_evidence", "via command",
                 "declared in a file changed in this phase", "image/png", "Claude consistency review: PASS",
                 "HF-P1-01: the wheels rotate — verified, evidence SATISFIED (attempt 2), reported 2×"]:
        assert part in text, part


# ============================================================ S. PERSISTENCE / BACKWARD COMPATIBILITY

def test_pass2_era_evidence_state_loads_and_resumes(tmp_path, erepo):
    """A run persisted by V4 pass 2: an evidence state and bundle WITHOUT provenance, freshness, reuse pool or
    replans, stopped at the checkpoint."""
    from app.models.phase import parse_phase_plan
    from app.models.run import Run
    from app.orchestration.providers.mock import MOCK_PHASE_PLAN
    db_path = tmp_path / "p2.db"
    db = Database(db_path)
    plan = parse_phase_plan(dict(MOCK_PHASE_PLAN, goal="g"))
    plan.tasks[1].verification_contract = VerificationContract.from_any(contract([req("E1")]))
    bundle = {"phase_id": "P2", "attempt": 1, "results": [
        {"requirement_id": "E1", "kind": "runtime_behavior", "required": True, "status": "SATISFIED",
         "summary": "probe ok", "strategy": "execute", "capability_ids": ["cap-001"], "steps": ["E1.s1"],
         "confidence": "observed", "source": "execution"}], "artifacts": [], "plan_summary": "s",
        "plan_source": "claude", "capability_summary": "3 capabilities found", "notes": [], "created_at": 1.0}
    es = {"phase_id": "P2", "attempt": 1, "status": "complete",
          "discovery": {"capabilities": [{"id": "cap-001", "kind": "runtime_behavior", "name": "runtime probe",
                                          "confidence": "CONFIRMED", "provider": "command",
                                          "key": f"command:{PROBE_CMD}", "details": {"command": PROBE_CMD}}],
                        "providers": {"command": "ok (1)"}},
          "plan": {"requirements": [{"requirement_id": "E1", "kind": "runtime_behavior", "required": True,
                                     "strategy": "execute", "steps": [{"key": "E1.s1", "capability_id": "cap-001"}]}],
                   "fingerprint": "x"},
          "steps": {"E1.s1": {"key": "E1.s1", "requirement_id": "E1", "capability_id": "cap-001", "status": "PASSED"}},
          "bundle": bundle, "history": []}
    run = Run(id=None, goal="g", repo_path=str(erepo), state="STOPPED", resume_state="REVIEWING",
              pipeline_version="v2", pipeline_state={"replans": 0, "human": {}, "evidence": {"P2": es}},
              current_task_id="P2")
    run.plan = plan
    db.create_run(run)
    db.update_run(run)
    db.close()
    conn = sqlite3.connect(db_path)
    for p in plan.tasks:
        d = p.to_dict()
        if p.id == "P1":
            d.update(status="PASSED", attempts=1, result={"task_id": "P1", "result": "done"})
        if p.id == "P2":
            d.update(status="PAUSED", attempts=1)
        conn.execute("INSERT INTO tasks (run_id, task_id, position, status, attempts, data_json) VALUES (?,?,?,?,?,?)",
                     (run.id, p.id, p.position, d["status"], d.get("attempts", 0), json.dumps(d)))
    now = time.time()
    for pid in ("P1", "P2"):
        data = {"task_id": pid, "number": 1, "finished_at": now}
        if pid == "P2":
            data["evidence"] = bundle
        conn.execute("INSERT INTO task_attempts (run_id, task_id, number, kind, started_at, finished_at, data_json) "
                     "VALUES (?,?,?,?,?,?,?)", (run.id, pid, 1, "execute", now, now, json.dumps(data)))
    conn.commit()
    conn.close()
    orch = MockOrchestrator(delay=0.0, evidence_aware=True)
    h = V2Harness(tmp_path, make_settings(), orch=orch, db="p2.db")

    async def main():
        e = h.engine
        await e.load_run(run.id)
        st = evm.EvidenceState.from_dict(e.run.pipeline_state["evidence"]["P2"])
        assert st.reusable == {} and st.replans == 0 and st.replan_log == []
        await e.resume()
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and orch.checkpoints[0].phase.id == "P2"
    r = orch.checkpoints[0].evidence["results"][0]
    assert r["status"] == SATISFIED and "freshness" not in r, "old results without provenance are left as recorded"
    assert not orch.consistency_packets and [t[0] for t in h.turns()] == ["P3"]
