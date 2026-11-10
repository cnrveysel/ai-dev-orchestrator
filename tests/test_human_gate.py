"""V4 pass 1: verification contracts, the ASK_USER checkpoint verdict, AWAITING_HUMAN, structured
human feedback, Claude HUMAN_FOLLOWUP and human-derived acceptance criteria.

Deterministic: mock orchestrator / mock workers only (the conftest guard refuses any real Claude
or Codex process); throwaway git repositories.
"""
import asyncio
import json
import sqlite3
import time

import pytest

from app.models import human as hm
from app.models.phase import Phase, parse_phases
from app.models.review import (
    CHECKPOINT_JSON_SCHEMA, FOLLOWUP_DECISIONS, HUMAN_FOLLOWUP_JSON_SCHEMA, REVIEW_JSON_SCHEMA, Review, ReviewDecision,
    checkpoint_schema,
)
from app.models.task import Task, TaskStatus
from app.models.verification import VerificationContract, contract_lines
from app.orchestration.checkpoint import CheckpointContextLimits, build_checkpoint_context
from app.orchestration.human_gate import HumanFollowupLimits, HumanFollowupPacket, build_human_followup_prompt
from app.orchestration.orchestrator import EngineError
from app.orchestration.providers.base import OrchestratorError
from app.orchestration.providers.mock import MockOrchestrator
from app.orchestration.scheduler import Scheduler, run_payload
from app.orchestration.state_machine import RunState
from app.persistence.database import Database
from app.workers.mock import MockWorker
from tests.test_pipeline_v2 import PY, V2Harness, git, make_settings, phase

S = RunState
NOTE = "The panel opens and saves correctly, but its list never scrolls past the first page."
CONTRACT = {
    "success_dimensions": ["functional", "visual"],
    "evidence_requirements": [
        {"id": "E1", "goal": "The primary interaction works in the real runtime", "kind": "runtime_behavior",
         "required": True, "reason": "Static tests are insufficient"},
        {"id": "E2", "goal": "The result has intentional visual hierarchy", "kind": "visual_evidence",
         "required": True, "reason": "This is a user-visible phase"}],
    "human_gate": {"allowed": True, "reason": "Some criteria may require human perception",
                   "checklist": ["Scroll through the whole list"]},
}


@pytest.fixture
def repo(tmp_path):
    r = tmp_path / "proj"
    r.mkdir()
    git(r, "init", "-q", "-b", "main")
    (r / "README.md").write_text("x\n")
    git(r, "add", "-A")
    git(r, "commit", "-qm", "i")
    return r


class RecordingOrchestrator(MockOrchestrator):
    """Records the exact checkpoint prompt Claude would receive, at the time of each call."""

    def __init__(self, **kw):
        super().__init__(delay=0.0, **kw)
        self.prompts: list[tuple[str, str, bool]] = []  # phase id, prompt, ASK_USER offered

    async def checkpoint(self, package):
        ctx = build_checkpoint_context(package, CheckpointContextLimits())
        self.prompts.append((package.phase.id, ctx.prompt, package.human_verification_allowed))
        return await super().checkpoint(package)


def p2_contract(e):
    phase(e, "P2").verification_contract = VerificationContract.from_any(CONTRACT)


def logs(h, source=None):
    return [p["text"] for k, p in h.events if k == "log" and (source is None or p["source"] == source)]


def states(h):
    return [p["state"] for k, p in h.events if k == "state"]


def human_rec(e, pid="P2"):
    return e.run.pipeline_state.get("human", {}).get(pid)


async def ask(h, repo, before=p2_contract):
    e = await h.plan_and_start(repo, before)
    assert e.sm.state == S.AWAITING_HUMAN, (e.sm.state, e.run.status_message)
    return e


async def answer(h, verdict, note=""):
    await h.engine.submit_human_feedback(verdict, note)
    await h.wait_idle()
    return h.engine


# ------------------------------------------------------------ models (unit)

def test_verification_contract_is_semantic_tolerant_and_round_trips():
    c = VerificationContract.from_any({
        "success_dimensions": ["Functional", "visual", "visual", ""],
        "evidence_requirements": ["the build succeeds", {"goal": "works at runtime", "kind": "Runtime Behavior",
                                                         "required": "false"},
                                  {"id": "E1", "goal": "dup id", "kind": "x"}, {"kind": "no goal"}, 7],
        "human_gate": {"allowed": "yes", "checklist": ["look at it", "look at it"]}})
    assert c.success_dimensions == ["functional", "visual"]
    assert [(r.id, r.kind, r.required) for r in c.evidence_requirements] == \
        [("E1", "unspecified", True), ("E2", "runtime_behavior", False), ("E3", "x", True)], "ids stay unique"
    assert c.human_gate.allowed and c.human_gate.checklist == ["look at it"]
    assert VerificationContract.from_any(None) is None and VerificationContract.from_any({}) is None
    assert VerificationContract.from_any({"human_gate": {"reason": "r"}}).human_gate.allowed is False, \
        "the human gate is opt-in"
    assert "human gate: allowed" in contract_lines(c) and any("[runtime_behavior, optional]" in x
                                                              for x in contract_lines(c))
    # persisted with the phase; a new plan keeps the contract but never inherits derived human criteria
    raw = {"id": "P1", "title": "T", "description": "d", "subtasks": ["a"], "acceptance_criteria": ["c"],
           "verification_contract": CONTRACT,
           "human_criteria": [{"id": "HF-P1-01", "text": "stale"}]}
    ph = parse_phases([raw])[0]
    assert ph.verification_contract == VerificationContract.from_any(CONTRACT) and ph.human_criteria == []
    ph.human_criteria = [hm.HumanCriterion("HF-P1-01", "the list scrolls to the end", "ISSUE_FOUND", "n", 1, 1.0)]
    again = Phase.from_dict(json.loads(json.dumps(ph.to_dict())))
    assert again.verification_contract == ph.verification_contract and again.human_criteria == ph.human_criteria


def test_old_phase_and_review_rows_load_with_additive_defaults():
    old_phase = {"id": "P1", "title": "T", "description": "d", "subtasks": [{"id": "S1", "title": "a"}],
                 "acceptance_criteria": ["c"], "status": "PASSED", "kind": "phase"}
    ph = Phase.from_dict(old_phase)
    assert ph.verification_contract is None and ph.human_criteria == []
    r = Review.from_dict({"decision": "RETRY", "summary": "s", "problems": ["p"], "kind": "CHECKPOINT"})
    assert (r.issues, r.human_request, r.human_feedback, r.derived_criteria, r.do_not_redo) == ([], {}, {}, [], [])


def test_issue_metadata_is_normalized_and_never_decides():
    r = Review.from_dict({"decision": "RETRY", "summary": "s", "issues": [
        {"description": "slightly off-center", "severity": "Micro Deviation", "disposition": "accept"},
        {"description": "x", "severity": "catastrophic", "disposition": "maybe"}, "plain string issue", {"no": 1}]})
    assert r.decision == ReviewDecision.RETRY, "metadata never overrides Claude's decision"
    assert r.issues == [
        {"description": "slightly off-center", "severity": "micro_deviation", "disposition": "accept", "evidence_id": ""},
        {"description": "x", "severity": "", "disposition": "", "evidence_id": ""},
        {"description": "plain string issue", "severity": "", "disposition": "", "evidence_id": ""}]
    sev = CHECKPOINT_JSON_SCHEMA["properties"]["issues"]["items"]["properties"]["severity"]["enum"]
    assert sev == ["critical", "functional", "regression", "polish", "subjective", "micro_deviation"]


def test_checkpoint_issue_metadata_does_not_drive_the_run(tmp_path, repo):
    passing = Review(ReviewDecision.PASS, "accepted despite a critical-labelled note",
                     issues=[{"description": "d", "severity": "critical", "disposition": "repair"}],
                     task_result={"result": "ok"})
    retry = Review(ReviewDecision.RETRY, "a micro deviation still needs a repair", problems=["off by one pixel"],
                   corrective_instructions=["align it"],
                   issues=[{"description": "off by one pixel", "severity": "micro_deviation", "disposition": "accept"}])
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P1": [passing], "P2": [retry, "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo))
    assert e.sm.state == S.COMPLETED
    assert phase(e, "P1").status == TaskStatus.PASSED and "REPAIRING" in states(h), "Claude's decision is what counts"
    assert "  issue: [micro_deviation/accept] off by one pixel" in logs(h, "CLAUDE")


def test_derived_criteria_get_phase_scoped_ids_and_are_deduplicated():
    req = hm.HumanRequest("P2", "why", ["q?"], attempt=1)
    fb = hm.HumanFeedback.create(req, "issue_found", NOTE)
    existing: list = []
    added = hm.add_derived_criteria(existing, "P2", [{"text": "The list scrolls to its end"}, "The list scrolls to "
                                                     "its END", {"text": ""}, {"text": "Saving keeps working"}], fb, 1)
    assert [(c.id, c.text) for c in added] == [("HF-P2-01", "The list scrolls to its end"),
                                              ("HF-P2-02", "Saving keeps working")]
    assert added[0].verdict == "ISSUE_FOUND" and added[0].note == NOTE
    more = hm.add_derived_criteria(existing, "P2", ["saving keeps working!", "A third"], fb, 2)
    assert [c.id for c in more] == ["HF-P2-03"] and len(existing) == 3
    for bad in ("MAYBE", ""):
        with pytest.raises(ValueError, match="invalid human verdict"):
            hm.HumanFeedback.create(req, bad)
    with pytest.raises(ValueError, match="note is required"):
        hm.HumanFeedback.create(req, "ISSUE_FOUND", "   ")
    ok = hm.HumanFeedback.create(req, "CANNOT_VERIFY")
    assert ok.questions == ["q?"] and ok.request_id == "P2-a1" and ok.note == ""


def test_human_gate_state_round_trips_with_history():
    st = hm.HumanGateState("P2")
    st.begin(hm.HumanRequest("P2", "r1", ["q1?"], attempt=1, created_at=1.0))
    st.record_feedback(hm.HumanFeedback("P2", "ISSUE_FOUND", "n1", ["q1?"], 2.0, "P2-a1"))
    st.resolve("RETRY", "fix it", ["HF-P2-01"])
    st.begin(hm.HumanRequest("P2", "r2", ["q2?"], attempt=2, created_at=3.0))
    again = hm.HumanGateState.from_dict(json.loads(json.dumps(st.to_dict())))
    assert again == st and again.status == hm.AWAITING and again.pending
    assert [r["status"] for r in again.rounds()] == [hm.RESOLVED, hm.AWAITING]
    assert again.rounds()[0]["feedback"]["note"] == "n1" and again.rounds()[0]["followup"]["decision"] == "RETRY"
    assert again.supersede_pending() and not again.pending and again.history[-1]["status"] == hm.SUPERSEDED


# ------------------------------------------------------------ unchanged verdicts with an open human gate

def test_pass_retry_blocked_are_unchanged_when_the_human_gate_is_open(tmp_path, repo):
    orch = RecordingOrchestrator(checkpoint_script={"P2": ["RETRY", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(h.plan_and_start(repo, p2_contract))
    assert e.sm.state == S.COMPLETED
    assert "AWAITING_HUMAN" not in states(h) and "REPAIRING" in states(h)
    assert not orch.followups and not e.run.pipeline_state.get("human")
    assert [(pid, offered) for pid, _, offered in orch.prompts] == \
        [("P1", False), ("P2", True), ("P2", True), ("P3", False)], "only P2's contract opens the human gate"
    p2_prompt = orch.prompts[1][1]
    for part in ["VERIFICATION CONTRACT", "E2 [visual_evidence, required]", "human gate: allowed",
                 "PASS / RETRY / BLOCKED / REPLAN / ASK_USER.", "ASK_USER (human verification is enabled",
                 "do not ask by default", "evidence_assessment", "micro_deviation"]:
        assert part in p2_prompt, part
    p1_prompt = orch.prompts[0][1]
    assert "ASK_USER" not in p1_prompt and "VERIFICATION CONTRACT" not in p1_prompt, "no contract: no human gate"

    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["BLOCKED"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch, db="blocked.db")
    e = asyncio.run(h.plan_and_start(repo, p2_contract))
    assert e.sm.state == S.BLOCKED and e.run.resume_state == "RETRY_PENDING"
    assert "AWAITING_HUMAN" not in states(h) and phase(e, "P2").status == TaskStatus.BLOCKED


# ------------------------------------------------------------ ASK_USER -> AWAITING_HUMAN

def test_ask_user_moves_to_awaiting_human_and_persists_the_exact_request(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)
    e = asyncio.run(ask(h, repo))
    p2 = phase(e, "P2")
    assert e.run.resume_state == "AWAITING_HUMAN" and "waiting for human verification of P2" in e.run.status_message
    assert p2.status == TaskStatus.AWAITING_HUMAN and p2.last_checkpoint == "ASK_USER"
    assert [t[0] for t in h.turns()] == ["P1", "P2"], "nothing runs while waiting (P3 not started)"
    rec = human_rec(e)
    assert rec["status"] == "awaiting" and rec["feedback"] is None
    req = rec["request"]
    assert req["questions"] == ["In the running result, can you confirm: Scroll through the whole list?"], \
        "questions generated from the phase's contract"
    assert req["phase_title"] == "Feature implementation" and req["attempt"] == 1
    assert req["checklist"] == ["Scroll through the whole list"] and req["observations"]
    assert any("no verification commands ran" in x for x in req["evidence"])
    a = e.db.load_attempts(e.run.id)["P2"][-1]
    assert a.review.decision == ReviewDecision.ASK_USER and a.review.human_request["questions"] == req["questions"]
    assert e.db.load_run(e.run.id).pipeline_state["human"]["P2"] == rec, "persisted"
    text = logs(h)
    assert "CHECKPOINT P2 attempt 1: ASK_USER - [MOCK] P2: human observation needed" in text
    assert "state REVIEWING -> AWAITING_HUMAN (REVIEWING done)" in text
    assert "  question: In the running result, can you confirm: Scroll through the whole list?" in text
    assert "  issue: [subjective/ask_user] [MOCK] perceptual quality cannot be judged from the evidence" in text
    payload = [p for k, p in h.events if k == "run"][-1]
    assert payload["state"] == "AWAITING_HUMAN" and payload["human"]["request"]["questions"] == req["questions"]


def test_restart_while_awaiting_human_keeps_state_and_questions_and_starts_no_worker(tmp_path, repo):
    s = make_settings()
    h = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}),
                  db="restart.db")

    async def first():
        e = await ask(h, repo)
        run_id, rec = e.run.id, human_rec(e)
        await e.shutdown()
        return run_id, rec
    run_id, rec = asyncio.run(first())

    worker2 = MockWorker(delay=0.0, steps=1)
    orch2 = MockOrchestrator(delay=0.0)
    h2 = V2Harness(tmp_path, s, orch=orch2, worker=worker2, db="restart.db")  # a new app process
    e = h2.engine
    assert not e._startup_notes, "AWAITING_HUMAN is not an interrupted run"
    row = sqlite3.connect(tmp_path / "restart.db").execute("SELECT state, interrupted FROM runs WHERE id=?",
                                                           (run_id,)).fetchone()
    assert row == ("AWAITING_HUMAN", 0)

    async def second():
        await e.load_run(run_id)
        assert e.sm.state == S.AWAITING_HUMAN and e.run.resume_state == "AWAITING_HUMAN"
        assert human_rec(e) == rec, "the exact questions, observations and evidence survive"
        assert phase(e, "P2").status == TaskStatus.AWAITING_HUMAN
        loaded = [p for k, p in h2.events if k == "run"][-1]
        assert loaded["human"]["request"]["questions"] == rec["request"]["questions"]
        await asyncio.sleep(0.05)
        assert not worker2.turns and not orch2.checkpoints, "no worker or Claude call starts by itself"
        with pytest.raises(EngineError, match="waiting for human verification"):
            await e.resume()
        await answer(h2, "LOOKS_GOOD")
        return e
    e = asyncio.run(second())
    assert e.sm.state == S.COMPLETED
    assert [(t["phase"], t["purpose"]) for t in worker2.turns] == [("P3", "start")], "P2 is never re-run"
    assert orch2.followups[0].request.questions == rec["request"]["questions"]


def test_stop_while_awaiting_then_resume_waits_again(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings(), orch=MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}))

    async def main():
        e = await ask(h, repo)
        await e.stop()
        assert e.sm.state == S.STOPPED and e.run.resume_state == "AWAITING_HUMAN"
        await e.resume()
        await h.wait_idle()
        assert e.sm.state == S.AWAITING_HUMAN and human_rec(e)["status"] == "awaiting"
        await answer(h, "LOOKS_GOOD")
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and [t[0] for t in h.turns()] == ["P1", "P2", "P3"]


# ------------------------------------------------------------ setting disabled / no contract

def test_disabled_setting_prevents_ask_user(tmp_path, repo):
    orch = RecordingOrchestrator(checkpoint_script={"P2": ["ASK_USER"]})
    h = V2Harness(tmp_path, make_settings(human_verification_enabled=False), orch=orch)
    e = asyncio.run(h.plan_and_start(repo, p2_contract))
    assert e.sm.state == S.COMPLETED and "AWAITING_HUMAN" not in states(h)
    assert not e.run.pipeline_state.get("human") and not orch.followups
    pid, prompt, offered = orch.prompts[1]
    assert pid == "P2" and not offered
    assert "Human verification is not available for this phase" in prompt and "ASK_USER" not in prompt
    assert "human verification is currently disabled by the user" in prompt
    assert "ASK_USER" not in checkpoint_schema(False)["properties"]["decision"]["enum"]
    assert checkpoint_schema(False) is CHECKPOINT_JSON_SCHEMA and "human_request" not in CHECKPOINT_JSON_SCHEMA["properties"]
    open_ = checkpoint_schema(True)
    assert open_["properties"]["decision"]["enum"][-1] == "ASK_USER" and "human_request" in open_["properties"]
    assert CHECKPOINT_JSON_SCHEMA["properties"]["decision"]["enum"] == ["PASS", "RETRY", "BLOCKED", "REPLAN"], \
        "the base schema is not mutated"


def test_a_provider_answering_ask_user_without_an_open_gate_never_waits(tmp_path, repo):
    class Misbehaving(MockOrchestrator):
        async def checkpoint(self, package):
            self.checkpoints.append(package)
            if package.phase.id == "P2":
                return Review(ReviewDecision.ASK_USER, "asks anyway",
                              human_request={"reason": "r", "questions": ["q?"], "observations": []})
            return Review(ReviewDecision.PASS, "ok", task_result={"result": "ok"})
    h = V2Harness(tmp_path, make_settings(human_verification_enabled=False), orch=Misbehaving(delay=0.0))
    e = asyncio.run(h.plan_and_start(repo, p2_contract))
    assert e.sm.state == S.PAUSED and e.run.resume_state == "REVIEWING"
    assert "human verification is not available for this phase" in e.run.status_message
    assert "AWAITING_HUMAN" not in states(h) and not e.run.pipeline_state.get("human")
    # same for a phase without a verification contract, even with the setting on
    h = V2Harness(tmp_path, make_settings(), orch=Misbehaving(delay=0.0), db="nocontract.db")
    e = asyncio.run(h.plan_and_start(repo, lambda e: setattr(phase(e, "P2"), "verification_contract", None)))
    assert e.sm.state == S.PAUSED and "AWAITING_HUMAN" not in states(h)


# ------------------------------------------------------------ HUMAN_FOLLOWUP outcomes

def test_looks_good_followup_passes_the_phase(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        await ask(h, repo)
        return await answer(h, "LOOKS_GOOD", "Everything scrolls and looks tidy.")
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and phase(e, "P2").status == TaskStatus.PASSED
    assert [t[0] for t in h.turns()] == ["P1", "P2", "P3"], "no repair, nothing redone"
    text = logs(h)
    assert 'P2: LOOKS_GOOD - "Everything scrolls and looks tidy."' in logs(h, "HUMAN")
    assert any(x.startswith("HUMAN_FOLLOWUP P2: PASS") for x in logs(h, "CLAUDE"))
    assert "state AWAITING_HUMAN -> ADVANCING (AWAITING_HUMAN done)" in text
    a = e.db.load_attempts(e.run.id)["P2"][-1]
    assert a.review.kind == "HUMAN_FOLLOWUP" and a.review.decision == ReviewDecision.PASS
    assert a.review.human_feedback["verdict"] == "LOOKS_GOOD" and a.review.human_request["questions"]
    assert phase(e, "P2").result.result.endswith("(human-verified)")
    rec = e.db.load_run(e.run.id).pipeline_state["human"]["P2"]
    assert rec["status"] == "resolved" and rec["feedback"]["note"] == "Everything scrolls and looks tidy."
    assert rec["followup"]["decision"] == "PASS"
    reviews = [r[0] for r in e.db.conn.execute("SELECT decision FROM reviews WHERE task_id='P2' ORDER BY id")]
    assert reviews == ["ASK_USER", "PASS"], "both the request and its adjudication are on record"


def test_issue_found_followup_retry_uses_the_existing_repair_with_the_human_feedback(tmp_path, repo):
    orch = RecordingOrchestrator(checkpoint_script={"P2": ["ASK_USER", "PASS"]})
    worker = MockWorker(delay=0.0, steps=1)

    def s1_done(req):
        if req.phase_id == "P2" and req.purpose == "start":
            worker.on_todo([{"text": "S1: Implement logic", "completed": True}])
    worker.file_writer = s1_done
    h = V2Harness(tmp_path, make_settings(), orch=orch, worker=worker)

    async def main():
        await ask(h, repo)
        return await answer(h, "ISSUE_FOUND", NOTE)
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    seq = states(h)
    i = seq.index("AWAITING_HUMAN")
    assert seq[i:i + 4] == ["AWAITING_HUMAN", "RETRY_PENDING", "REPAIRING", "VERIFYING"], seq
    p2_turns = [t for t in worker.turns if t["phase"] == "P2"]
    assert [t["purpose"] for t in p2_turns] == ["start", "repair"]
    impl, rep = p2_turns
    assert not rep["resumed"] and rep["session"] != impl["session"], "a fresh worker session"
    for part in [f'the user\'s note: "{NOTE}"', "HUMAN VERIFICATION (authoritative", "HF-P2-01",
                 "HUMAN-DERIVED ACCEPTANCE CRITERIA", "DO NOT REDO (confirmed working", "- S1 Implement logic",
                 "COMPLETED SUBTASKS (done - do NOT redo): S1 Implement logic",
                 "FINDINGS FROM THE HUMAN VERIFICATION FOLLOW-UP", "Fix what the user observed",
                 "the user was asked: In the running result, can you confirm"]:
        assert part in rep["prompt"], part
    rl = logs(h, "REPAIR")
    assert any(x.startswith("P2: entering repair mode after human follow-up RETRY (repair attempt 1); completed "
                            "subtasks stay done: S1") for x in rl)
    assert any(x.startswith('P2: human feedback attached (ISSUE_FOUND: "The panel opens') for x in rl)
    assert "state AWAITING_HUMAN -> RETRY_PENDING (AWAITING_HUMAN done)" in logs(h)
    assert any(x.startswith("  derived criterion HF-P2-01: [MOCK] The user-observed problem is resolved")
               for x in logs(h, "CLAUDE"))
    # completed subtasks were never reset; the phase was never restarted
    p2 = e.db.load_run(e.run.id).plan.task("P2")
    assert p2.subtasks[0].done and p2.subtasks[0].completed_via == "todo"
    atts = e.db.load_attempts(e.run.id)["P2"]
    assert [a.kind for a in atts] == ["execute", "repair"] and atts[0].review.kind == "HUMAN_FOLLOWUP"
    assert atts[0].review.human_feedback["note"] == NOTE, "the human note persists with the attempt"
    # the derived criterion persists with the phase and reaches the later checkpoint of the phase
    assert [c.id for c in p2.human_criteria] == ["HF-P2-01"] and p2.human_criteria[0].note == NOTE
    later = [p for pid, p, _ in orch.prompts if pid == "P2"][1]
    for part in ["HUMAN-DERIVED ACCEPTANCE CRITERIA", "HF-P2-01", "HUMAN VERIFICATION OF THIS PHASE SO FAR",
                 f"user answered ISSUE_FOUND: {NOTE}", "RETRY (human follow-up)",
                 "this repair addressed the user's verification (ISSUE_FOUND)"]:
        assert part in later, part
    rec = e.db.load_run(e.run.id).pipeline_state["human"]["P2"]
    assert rec["feedback"]["verdict"] == "ISSUE_FOUND" and rec["followup"]["derived_criteria"] == ["HF-P2-01"]
    packet = orch.followups[0]
    assert packet.feedback.note == NOTE and packet.checkpoint["decision"] == "ASK_USER"
    for part in ["THE USER'S ANSWER", "verdict: ISSUE_FOUND", NOTE, "ASK_USER is not available in this follow-up",
                 "VERIFICATION CONTRACT", "COMPLETED SUBTASKS: S1 Implement logic", "derived_criteria", "do_not_redo"]:
        assert part in orch.followup_prompts[0], part


def test_cannot_verify_followup_blocked_then_resume_continues_the_phase(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER", "PASS"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        await ask(h, repo)
        e = await answer(h, "CANNOT_VERIFY", "I have no display attached to the test machine.")
        assert e.sm.state == S.BLOCKED and e.run.resume_state == "RETRY_PENDING"
        assert "Claude HUMAN_FOLLOWUP marked phase P2 BLOCKED" in e.run.status_message
        assert phase(e, "P2").status == TaskStatus.BLOCKED and phase(e, "P2").last_checkpoint == "BLOCKED"
        await e.resume("Verify it through the automated check instead")
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED
    assert [(t[0], t[1]) for t in h.turns()] == [("P1", "start"), ("P2", "start"), ("P2", "retry"), ("P3", "start")]
    assert "Verify it through the automated check instead" in h.worker.turns[2]["prompt"]


def test_cannot_verify_followup_retry_repairs(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER", "PASS"]}, followup_script={"P2": ["RETRY"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        await ask(h, repo)
        return await answer(h, "CANNOT_VERIFY")
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and "REPAIRING" in states(h)
    rep = next(t for t in h.worker.turns if t["purpose"] == "repair")
    assert "could not verify the result" in rep["prompt"]


def test_followup_can_never_recurse_into_ask_user(tmp_path, repo):
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}, followup_script={"P2": ["ASK_USER"]})
    h = V2Harness(tmp_path, make_settings(), orch=orch)

    async def main():
        e = await ask(h, repo)
        await answer(h, "LOOKS_GOOD")
        assert e.sm.state == S.PAUSED and e.run.resume_state == "AWAITING_HUMAN"
        assert "only PASS, RETRY, BLOCKED are allowed" in e.run.status_message
        rec = human_rec(e)
        assert rec["status"] == "feedback_received" and not rec["history"], "no second request was created"
        assert e.db.load_attempts(e.run.id)["P2"][-1].review.decision == ReviewDecision.ASK_USER
        await e.resume()  # re-runs the follow-up for the recorded answer (the script is used up: PASS)
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and len(orch.followups) == 2
    assert HUMAN_FOLLOWUP_JSON_SCHEMA["properties"]["decision"]["enum"] == list(FOLLOWUP_DECISIONS) == \
        ["PASS", "RETRY", "BLOCKED"]


def test_feedback_is_validated_and_recorded_once(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings(), orch=MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}))

    async def main():
        e = h.engine
        with pytest.raises(EngineError):
            await e.submit_human_feedback("LOOKS_GOOD")  # no run
        await ask(h, repo)
        with pytest.raises(EngineError, match="note is required"):
            await e.submit_human_feedback("ISSUE_FOUND", " ")
        with pytest.raises(EngineError, match="invalid human verdict"):
            await e.submit_human_feedback("PROBABLY_FINE")
        assert human_rec(e)["status"] == "awaiting" and not logs(h, "HUMAN"), "nothing recorded"
        await answer(h, "LOOKS_GOOD")
        with pytest.raises(EngineError, match="not waiting for human verification"):
            await e.submit_human_feedback("LOOKS_GOOD")
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and logs(h, "HUMAN") == ["P2: LOOKS_GOOD"]


def test_answer_recorded_but_followup_interrupted_resumes_the_followup(tmp_path, repo):
    s = make_settings()
    h = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}), db="fu.db")

    async def first():
        e = await ask(h, repo)
        hs = hm.HumanGateState.from_dict(human_rec(e))  # the app dies right after the answer was saved
        hs.record_feedback(hm.HumanFeedback.create(hs.request, "ISSUE_FOUND", NOTE))
        e.run.pipeline_state["human"]["P2"] = hs.to_dict()
        e._persist_run()
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(first())
    orch2 = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["PASS"]})
    h2 = V2Harness(tmp_path, s, orch=orch2, db="fu.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.AWAITING_HUMAN and [p for k, p in h2.events if k == "run"][-1]["human"]["status"] == \
            "feedback_received"
        with pytest.raises(EngineError, match="already recorded"):
            await e.submit_human_feedback("LOOKS_GOOD")
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(second())
    assert e.sm.state == S.COMPLETED and orch2.followups[0].feedback.note == NOTE
    assert [(t["phase"], t["purpose"]) for t in h2.worker.turns] == [("P2", "repair"), ("P3", "start")]


def test_followup_recorded_before_a_crash_is_not_adjudicated_twice(tmp_path, repo):
    s = make_settings()
    h = V2Harness(tmp_path, s, orch=MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER"]}), db="twice.db")

    async def first():
        await ask(h, repo)
        e = await answer(h, "CANNOT_VERIFY", "no display here")
        assert e.sm.state == S.BLOCKED
        run_id = e.run.id
        await e.shutdown()
        return run_id
    run_id = asyncio.run(first())
    # The app died after the follow-up review was written to the attempt, before the gate state and the run
    # state followed: the run is still AWAITING_HUMAN with an unadjudicated-looking answer.
    conn = sqlite3.connect(tmp_path / "twice.db")
    st = json.loads(conn.execute("SELECT pipeline_state_json FROM runs WHERE id=?", (run_id,)).fetchone()[0])
    st["human"]["P2"]["status"], st["human"]["P2"]["followup"] = "feedback_received", {}
    conn.execute("UPDATE runs SET state='AWAITING_HUMAN', resume_state='AWAITING_HUMAN', pipeline_state_json=? "
                 "WHERE id=?", (json.dumps(st), run_id))
    conn.commit()
    conn.close()
    orch2 = MockOrchestrator(delay=0.0)
    h2 = V2Harness(tmp_path, s, orch=orch2, db="twice.db")

    async def second():
        e = h2.engine
        await e.load_run(run_id)
        assert e.sm.state == S.AWAITING_HUMAN
        await e.resume()
        await h2.wait_idle()
        return e
    e = asyncio.run(second())
    assert not orch2.followups, "the recorded decision is reused; Claude is not asked again"
    assert e.sm.state == S.BLOCKED and "HUMAN_FOLLOWUP marked phase P2 BLOCKED" in e.run.status_message
    assert human_rec(e)["status"] == "resolved" and human_rec(e)["followup"]["decision"] == "BLOCKED"
    assert not h2.worker.turns


# ------------------------------------------------------------ bounded packet / Claude adapter

def _followup_packet(note=NOTE, diff_kb=0):
    ph = parse_phases([{"id": "P2", "title": "Panel", "description": "Build the panel", "subtasks": ["a", "b"],
                        "acceptance_criteria": ["the panel works"], "verification_contract": CONTRACT}])[0]
    ph.subtasks[0].done = True
    req = hm.HumanRequest("P2", "perception needed", ["Does the list scroll to the end?"], ["looks ok"],
                          ["Scroll through the whole list"], ["verification `npm test`: passed"], "Panel", 1, "ask")
    fb = hm.HumanFeedback.create(req, "ISSUE_FOUND", note)
    return HumanFollowupPacket(
        phase=ph, request=req, feedback=fb, goal="g", checkpoint={"decision": "ASK_USER", "summary": "ask",
                                                                  "problems": ["p"], "evidence_assessment": [
                                                                      {"id": "E1", "status": "satisfied"}]},
        diff="+x\n" * (diff_kb * 350), changed_files=["src/panel.ts"], diff_stat=" src/panel.ts | 3 +",
        prior_diagnoses=[{"status": "REPAIR", "diagnosis": "d", "recommended_actions": ["a"]}])


def test_human_followup_packet_is_bounded_and_keeps_the_answer():
    prompt, stats = build_human_followup_prompt(_followup_packet(diff_kb=200), HumanFollowupLimits(40000, 16000))
    assert stats["prompt_bytes"] <= 40000 and stats["diff_original_bytes"] > 60000
    for part in ["HUMAN_FOLLOWUP", "THE USER'S ANSWER", NOTE, "Does the list scroll to the end?", "E1: satisfied",
                 "RELEVANT DIFF", "truncated", "PRIOR REPAIR DIAGNOSES", "DECIDE (structured answer"]:
        assert part in prompt, part
    tiny, st = build_human_followup_prompt(_followup_packet(note="n" * 3000, diff_kb=50), HumanFollowupLimits(6000, 0))
    assert st["prompt_bytes"] <= 6000 and "DECIDE (structured answer" in tiny and "verdict: ISSUE_FOUND" in tiny


FAKE_CLAUDE = r'''
import json, sys
sys.stdin.read()
payload = json.loads(open(sys.argv[1], encoding="utf-8").read())
print(json.dumps({"type": "result", "subtype": "success", "is_error": False, "num_turns": 1, "total_cost_usd": 0.01,
                  "usage": {}, "structured_output": payload}), flush=True)
'''


def _claude(tmp_path, payload):
    from app.config.settings import ClaudeSettings
    from app.orchestration.providers.claude_code_cli import ClaudeCodeCLIProvider
    (tmp_path / "fake_claude_v4.py").write_text(FAKE_CLAUDE)
    (tmp_path / "payload_v4.json").write_text(json.dumps(payload))
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable="unused"))
    seen = []

    async def argv(limits, schema):
        seen.append((limits, schema))
        return [PY, str(tmp_path / "fake_claude_v4.py"), str(tmp_path / "payload_v4.json")]
    prov._build_argv = argv
    return prov, seen


def _ck_package(allowed, repo_path):
    from tests.test_pipeline_v2 import _package
    pk = _package(diff_size=2, repo_path=repo_path)
    pk.phase.verification_contract = VerificationContract.from_any(CONTRACT)
    pk.human_verification_allowed = allowed
    return pk


def test_claude_checkpoint_offers_ask_user_only_with_an_open_gate(tmp_path):
    ask_payload = {"decision": "ASK_USER", "summary": "s", "problems": [], "corrective_instructions": [],
                   "verification_notes": [], "plan_changes_required": False,
                   "human_request": {"reason": "r", "questions": ["Does it scroll?"], "observations": ["tests pass"]},
                   "evidence_assessment": [{"id": "E2", "status": "insufficient", "note": "needs eyes"}]}
    prov, seen = _claude(tmp_path, ask_payload)
    r = asyncio.run(prov.checkpoint(_ck_package(True, str(tmp_path))))
    assert r.decision == ReviewDecision.ASK_USER and r.human_request["questions"] == ["Does it scroll?"]
    assert r.evidence_assessment == [{"id": "E2", "status": "insufficient", "note": "needs eyes"}]
    limits, schema = seen[-1]
    assert limits.purpose == "CHECKPOINT" and "ASK_USER" in schema["properties"]["decision"]["enum"]
    prov, seen = _claude(tmp_path, ask_payload)
    with pytest.raises(OrchestratorError, match="not allowed here"):
        asyncio.run(prov.checkpoint(_ck_package(False, str(tmp_path))))
    assert seen[-1][1] is CHECKPOINT_JSON_SCHEMA
    prov, _ = _claude(tmp_path, dict(ask_payload, human_request={"reason": "r", "questions": [], "observations": []}))
    with pytest.raises(OrchestratorError, match="without any question"):
        asyncio.run(prov.checkpoint(_ck_package(True, str(tmp_path))))


def test_claude_human_followup_uses_its_own_bounded_call_and_schema(tmp_path):
    prov, seen = _claude(tmp_path, {"decision": "RETRY", "summary": "the list does not scroll",
                                    "problems": ["no scrolling"], "corrective_instructions": ["enable scrolling"],
                                    "derived_criteria": [{"text": "The list scrolls to its last entry",
                                                          "rationale": "user"}],
                                    "do_not_redo": ["saving"]})
    r = asyncio.run(prov.human_followup(_followup_packet()))
    assert r.decision == ReviewDecision.RETRY and r.kind == "HUMAN_FOLLOWUP"
    assert r.derived_criteria == [{"text": "The list scrolls to its last entry", "rationale": "user"}]
    assert r.do_not_redo == ["saving"]
    limits, schema = seen[-1]
    assert schema is HUMAN_FOLLOWUP_JSON_SCHEMA and "ASK_USER" not in schema["properties"]["decision"]["enum"]
    assert limits.purpose == "HUMAN_FOLLOWUP" and limits.tools == "" and limits.max_tool_calls == 0
    assert limits.timeout == 180 and limits.max_budget_usd == 0.40
    assert prov.calls[-1].kind == "HUMAN_FOLLOWUP"
    prov, _ = _claude(tmp_path, {"decision": "ASK_USER", "summary": "again", "problems": [],
                                 "corrective_instructions": [], "derived_criteria": [], "do_not_redo": [],
                                 "human_request": {"reason": "r", "questions": ["q?"]}})
    with pytest.raises(OrchestratorError, match="invalid human follow-up"):
        asyncio.run(prov.human_followup(_followup_packet()))


def test_claude_role_and_planner_schema_carry_the_contract(tmp_path):
    from app.config.settings import ClaudeSettings
    from app.models.phase import PHASE_PLAN_JSON_SCHEMA
    from app.orchestration.providers.claude_code_cli import (
        OPTIONAL_FLAGS, REQUIRED_FLAGS, SYSTEM_ROLE_V2, ClaudeCodeCLIProvider, _PHASE_RULES, limits_for,
    )
    item = PHASE_PLAN_JSON_SCHEMA["properties"]["phases"]["items"]
    assert "verification_contract" in item["properties"] and "verification_contract" not in item["required"]
    assert "human_gate" in _PHASE_RULES and "not how a tool obtains it" in _PHASE_RULES
    exe = tmp_path / "claude.exe"
    exe.write_bytes(b"MZ")
    prov = ClaudeCodeCLIProvider(ClaudeSettings(executable=str(exe)))
    prov._flags = set(REQUIRED_FLAGS + OPTIONAL_FLAGS)
    argv = asyncio.run(prov._build_argv(limits_for("HUMAN_FOLLOWUP", prov.settings), {}))
    assert argv[argv.index("--append-system-prompt") + 1] == SYSTEM_ROLE_V2 and argv[argv.index("--tools") + 1] == ""


# ------------------------------------------------------------ compatibility

def test_v3_era_database_without_v4_fields_loads_resumes_and_finishes(tmp_path, repo):
    """A V2/V3 phase run persisted before V4 existed: no contracts, no derived criteria, no human state."""
    db_path = tmp_path / "v3.db"
    db = Database(db_path)
    from app.models.run import Run
    from app.orchestration.providers.mock import MOCK_PHASE_PLAN
    from app.models.phase import parse_phase_plan
    plan = parse_phase_plan(dict(MOCK_PHASE_PLAN, goal="old goal"))
    run = Run(id=None, goal="old goal", repo_path=str(repo), state="STOPPED", resume_state="STARTING_TASK",
              pipeline_version="v2", pipeline_state={"replans": 0})
    run.plan = plan
    db.create_run(run)
    db.update_run(run)
    rows = []
    for p in plan.tasks:
        d = p.to_dict()
        d.pop("verification_contract")
        d.pop("human_criteria")
        if p.id == "P1":
            d.update(status="PASSED", attempts=1, result={"task_id": "P1", "result": "core done"})
        rows.append((p.id, p.position, d))
    db.close()
    conn = sqlite3.connect(db_path)
    for pid, pos, d in rows:
        conn.execute("INSERT INTO tasks (run_id, task_id, position, status, attempts, data_json) VALUES (?,?,?,?,?,?)",
                     (run.id, pid, pos, d["status"], d.get("attempts", 0), json.dumps(d)))
    old_review = {"decision": "PASS", "summary": "ok", "problems": [], "corrective_instructions": [],
                  "verification_notes": [], "plan_changes_required": False, "task_result": {"result": "core done"},
                  "kind": "CHECKPOINT"}
    conn.execute("INSERT INTO task_attempts (run_id, task_id, number, kind, started_at, finished_at, data_json) "
                 "VALUES (?,?,?,?,?,?,?)", (run.id, "P1", 1, "execute", time.time(), time.time(),
                                           json.dumps({"task_id": "P1", "number": 1, "review": old_review})))
    conn.commit()
    conn.close()

    h = V2Harness(tmp_path, make_settings(), db="v3.db")

    async def main():
        e = h.engine
        await e.load_run(run.id)
        assert all(p.verification_contract is None and p.human_criteria == [] for p in e.run.plan.tasks)
        assert "human" not in e.run.pipeline_state and run_payload(e.run)["human"] is None
        assert e.db.load_attempts(run.id)["P1"][0].review.issues == []
        await e.resume()
        await h.wait_idle()
        return e
    e = asyncio.run(main())
    assert e.sm.state == S.COMPLETED and [t[0] for t in h.turns()] == ["P2", "P3"]
    assert not h.orch.followups and "AWAITING_HUMAN" not in states(h)
    assert all(not pk.human_verification_allowed for pk in h.orch.checkpoints), "no contract: no human gate"


def test_v1_pipeline_is_unaffected(tmp_path, repo):
    h = V2Harness(tmp_path, make_settings(), worker=MockWorker(delay=0.0, steps=1))

    async def main():
        await h.engine.plan("g", str(repo))
        await h.wait_idle()
        await h.engine.start()
        await h.wait_idle()
    asyncio.run(main())
    e = h.engine
    assert e.run.pipeline_version == "v1" and type(e.scheduler) is Scheduler and e.sm.state == S.COMPLETED
    assert S.AWAITING_HUMAN not in e.scheduler.dispatch_states and S.AWAITING_HUMAN not in e.scheduler._handlers
    assert all(type(t) is Task for t in e.run.plan.tasks) and "human" not in e.run.pipeline_state
    assert REVIEW_JSON_SCHEMA["properties"]["decision"]["enum"] == ["PASS", "RETRY", "BLOCKED"]
    assert all(p.get("human") is None for k, p in h.events if k == "run")


# ------------------------------------------------------------ GUI (headless)

def test_gui_human_verification_panel_collects_a_structured_answer(tmp_path, monkeypatch):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    QtWidgets = pytest.importorskip("PySide6.QtWidgets")
    from app.config.settings import SettingsStore
    from app.gui.bridge import EventBridge
    from app.gui.main_window import MainWindow
    from app.orchestration.orchestrator import EngineHost
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])

    def pump(cond, timeout=60):
        t0 = time.time()
        while not cond():
            app.processEvents()
            time.sleep(0.01)
            assert time.time() - t0 < timeout, "condition not reached"

    repo = tmp_path / "proj"
    repo.mkdir()
    git(repo, "init", "-q", "-b", "main")
    s = make_settings(pipeline_version="v2")
    s.last_repo_path = str(repo)
    store = SettingsStore(tmp_path / "settings.json")
    bridge = EventBridge()
    host = EngineHost(s, bridge.emit, str(tmp_path / "gui_v4.db"))
    win = MainWindow(s, store, bridge, host)
    host.start()
    orch = MockOrchestrator(delay=0.0, checkpoint_script={"P2": ["ASK_USER", "PASS"]})
    host.engine._make_orchestrator = lambda mode: orch
    win.engine_started()
    infos = []
    monkeypatch.setattr(QtWidgets.QMessageBox, "information", lambda *a, **k: infos.append(a))
    monkeypatch.setattr(QtWidgets.QMessageBox, "warning", lambda *a, **k: None)
    panel = win.human_panel
    try:
        assert not panel.isVisibleTo(win)
        win.task_input.setPlainText("Build the panel")
        win.btn_plan.click()
        pump(lambda: win.state == "PLAN_READY" and win.tasks.topLevelItemCount() == 3)
        win.btn_start.click()
        pump(lambda: win.state == "AWAITING_HUMAN" and panel.isVisibleTo(win))
        assert "P2" in panel.title() and "Feature implementation" in panel.title()
        text = panel.view.toPlainText()
        assert "In the running result, can you confirm: The primary interaction works in the real runtime?" in text
        assert "human observation needed" in text and "Automated evidence" in text
        assert all(b.isEnabled() for b in panel.buttons.values())
        assert win.btn_stop.isEnabled() and not win.btn_resume.isEnabled() and not win.btn_pause.isEnabled()
        assert win.tasks.topLevelItem(1).text(2) == "AWAITING_HUMAN"
        panel.buttons["ISSUE_FOUND"].click()  # no note: refused locally, nothing sent
        assert infos and win.state == "AWAITING_HUMAN" and all(b.isEnabled() for b in panel.buttons.values())
        panel.note.setPlainText(NOTE)
        panel.buttons["ISSUE_FOUND"].click()
        pump(lambda: win.state == "COMPLETED")
        pump(lambda: not panel.isVisibleTo(win))
        assert orch.followups[0].feedback.verdict == "ISSUE_FOUND" and orch.followups[0].feedback.note == NOTE
        win.tasks.setCurrentItem(win.tasks.topLevelItem(1))
        app.processEvents()
        details = win.details.toPlainText()
        for part in ["Verification contract", "Human-derived acceptance criteria", "HF-P2-01",
                     "Claude human follow-up: RETRY", "User's answer: ISSUE_FOUND"]:
            assert part in details, part
    finally:
        win.close()
