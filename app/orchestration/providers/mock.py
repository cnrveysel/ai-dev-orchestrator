"""Deterministic mock orchestrator used to exercise the scheduler without Claude.
Clearly labelled MOCK in every log line."""
from __future__ import annotations

import asyncio
import copy
from typing import Any

from app.models.phase import TERMINAL_PHASE_STATUSES, Phase, parse_phase_plan, parse_phases
from app.models.plan import Plan
from app.models.repair import Diagnosis
from app.models.review import Review, ReviewDecision
from app.orchestration.providers.base import (
    LogFn, OrchestratorProvider, PhaseReplanRequest, PlanRequest, ReplanRequest,
)
from app.orchestration.reviewer import ReviewPackage

MOCK_PHASE_PLAN: dict[str, Any] = {
    "summary": "[MOCK] Example phase plan used to exercise the V2 phase pipeline.",
    "assumptions": ["Mock mode: no real model is called and no files are edited."],
    "architecture_notes": ["Mock architecture note."],
    "phases": [
        {"id": "P1", "title": "Core model and interfaces", "description": "Establish the core data model.",
         "instructions": "Use codebase-memory get_architecture first.",
         "subtasks": [{"id": "S1", "title": "Survey architecture"}, {"id": "S2", "title": "Add data model"},
                      {"id": "S3", "title": "Add interfaces"}],
         "acceptance_criteria": ["Model and interfaces compile"], "estimated_minutes": 30},
        {"id": "P2", "title": "Feature implementation", "description": "Implement the feature on the core.",
         "subtasks": [{"id": "S1", "title": "Implement logic"}, {"id": "S2", "title": "Wire into the app"},
                      {"id": "S3", "title": "Handle edge cases"}],
         "acceptance_criteria": ["Feature works end to end"],
         "integration_expectations": ["Uses the P1 interfaces"], "estimated_minutes": 45,
         # A human gate but no evidence requirements: default mock runs never collect evidence (V4 pass 2 tests
         # declare their own requirements).
         "verification_contract": {
             "success_dimensions": ["functional", "runtime"],
             "evidence_requirements": [],
             "human_gate": {"allowed": True, "reason": "The user-visible result may need human perception",
                            "checklist": ["The primary interaction works in the real runtime"]}}},
        {"id": "P3", "title": "Tests and polish", "description": "Tests, cleanup and polish.",
         "subtasks": [{"id": "S1", "title": "Add tests"}, {"id": "S2", "title": "Polish UX"},
                      {"id": "S3", "title": "Remove temporary artifacts"}],
         "acceptance_criteria": ["Tests pass"], "estimated_minutes": 30},
    ],
}


class MockOrchestrator(OrchestratorProvider):
    name = "mock-orchestrator"

    def __init__(self, log: LogFn | None = None, delay: float = 0.3,
                 retry_task_ids: list[str] | None = None,
                 blocked_task_ids: list[str] | None = None,
                 checkpoint_script: dict[str, list[str]] | None = None,
                 replan_result: list[dict[str, Any]] | None = None,
                 diagnose_script: list[Any] | None = None,
                 followup_script: dict[str, list[Any]] | None = None,
                 evidence_plan_script: dict[str, list[Any]] | None = None,
                 evidence_aware: bool = False,
                 consistency_script: dict[str, list[Any]] | None = None,
                 mid_phase_script: list[Any] | None = None):
        super().__init__(log)
        self.delay = delay
        self.retry_task_ids = set(retry_task_ids or [])
        self.blocked_task_ids = set(blocked_task_ids or [])
        # V2 test knobs: explicit checkpoint decisions per phase id, and the phases a replan returns.
        self.checkpoint_script: dict[str, list[str]] = checkpoint_script or {}
        self.replan_result = replan_result
        self.checkpoints: list[Any] = []  # CheckpointPackages received
        self.replan_requests: list[PhaseReplanRequest] = []
        self.diagnose_script: list[Any] = list(diagnose_script or [])  # REPAIR (default) | BLOCKED | Diagnosis
        self.diagnoses: list[Any] = []  # DiagnosisPackets received
        self.diagnosis_prompts: list[str] = []
        # V4: HUMAN_FOLLOWUP decisions per phase id (PASS | RETRY | BLOCKED | ASK_USER | Review); by default
        # LOOKS_GOOD -> PASS, ISSUE_FOUND -> RETRY, CANNOT_VERIFY -> BLOCKED.
        self.followup_script: dict[str, list[Any]] = followup_script or {}
        self.followups: list[Any] = []  # HumanFollowupPackets received
        self.followup_prompts: list[str] = []
        # V4 pass 2: EVIDENCE_PLAN answers per phase id (a raw plan dict, or a callable(packet) -> dict); by
        # default every contract requirement uses the usable capabilities of the SAME kind, else human/unavailable.
        self.evidence_plan_script: dict[str, list[Any]] = evidence_plan_script or {}
        self.evidence_plans: list[Any] = []  # EvidencePlanPackets received
        self.evidence_plan_prompts: list[str] = []
        # Test knob: judge checkpoints from the evidence bundle like the prompt asks Claude to (off by default,
        # so runs that do not test evidence judgment behave exactly as before).
        self.evidence_aware = evidence_aware
        # V4 pass 3: CONSISTENCY_REVIEW answers per phase id (PASS | RETRY | BLOCKED | ASK_USER | Review | raw dict);
        # by default a PASS is never justified by the mock: FAILED evidence -> RETRY, a gap a person can observe
        # with an open gate -> ASK_USER, anything else -> BLOCKED.
        self.consistency_script: dict[str, list[Any]] = consistency_script or {}
        self.consistency_packets: list[Any] = []
        self.consistency_prompts: list[str] = []
        # V4 pass 3: MID_PHASE_DIAGNOSE answers in order (CONTINUE (default) | BLOCKED | MidPhaseDiagnosis)
        self.mid_phase_script: list[Any] = list(mid_phase_script or [])
        self.mid_phase_packets: list[Any] = []
        self.mid_phase_prompts: list[str] = []
        self.phase_plan: dict[str, Any] = copy.deepcopy(MOCK_PHASE_PLAN)

    async def plan(self, request: PlanRequest) -> Plan:
        self.log("CLAUDE", "[MOCK] planning (no Claude call is made in mock mode)")
        await asyncio.sleep(self.delay)
        return Plan.from_dict({
            "goal": request.goal,
            "summary": "[MOCK] Example plan used to exercise the sequential scheduler.",
            "assumptions": ["Mock mode: no real model is called and no files are edited."],
            "architecture_notes": ["Mock architecture note."],
            "tasks": [
                {"id": "T001", "title": "Project architecture survey",
                 "description": "Survey the architecture relevant to the request.",
                 "instructions": "Use codebase-memory-mcp get_architecture.",
                 "acceptance_criteria": ["Architecture summary produced"],
                 "files_likely_involved": [], "verification_commands": []},
                {"id": "T002", "title": "Core interface", "dependencies": ["T001"],
                 "description": "Add the core interface.",
                 "acceptance_criteria": ["Interface compiles"],
                 "files_likely_involved": ["Source/Core/Interface.h"]},
                {"id": "T003", "title": "Implementation", "dependencies": ["T002"],
                 "description": "Implement the feature on top of the interface.",
                 "acceptance_criteria": ["Feature works"]},
                {"id": "T004", "title": "Tests", "dependencies": ["T003"],
                 "description": "Add tests.", "acceptance_criteria": ["Tests pass"]},
            ],
        })

    async def review(self, package: ReviewPackage) -> Review:
        t, a = package.task, package.attempt
        self.log("CLAUDE", f"[MOCK] reviewing {t.id} attempt {a.number}")
        await asyncio.sleep(self.delay)
        failed_verification = [v.command for v in a.verification if not v.passed and not v.skipped_reason]
        if t.id in self.blocked_task_ids and not t.user_guidance:
            return Review(ReviewDecision.BLOCKED, "[MOCK] needs a user decision",
                          problems=["Mock blocker: choose an approach"])
        if (t.id in self.retry_task_ids and a.number == 1) or failed_verification:
            return Review(ReviewDecision.RETRY, "[MOCK] first attempt rejected to exercise the fix loop",
                          problems=["Mock problem: acceptance criterion not demonstrated"]
                          + [f"verification failed: {c}" for c in failed_verification],
                          corrective_instructions=["Mock fix: address the problem and re-verify"])
        return Review(ReviewDecision.PASS, f"[MOCK] {t.id} accepted",
                      verification_notes=["mock review"],
                      task_result={"result": f"[MOCK] {t.title} done",
                                   "important_decisions": [], "remaining_risks": []})

    async def replan(self, request: ReplanRequest) -> Plan:
        self.log("CLAUDE", "[MOCK] replan requested; keeping plan unchanged")
        return request.plan

    # ------------------------------------------------------------ V2

    async def plan_phases(self, request: PlanRequest) -> Plan:
        self.log("CLAUDE", "[MOCK] phase planning (no Claude call is made in mock mode)")
        await asyncio.sleep(self.delay)
        plan = parse_phase_plan({"goal": request.goal, **copy.deepcopy(self.phase_plan)})
        return plan

    async def checkpoint(self, package: Any) -> Review:
        p, a = package.phase, package.attempt
        self.checkpoints.append(package)
        label = "final audit" if package.final else "checkpoint"
        self.log("CLAUDE", f"[MOCK] {label} {p.id} attempt {a.number}")
        await asyncio.sleep(self.delay)
        failed = [v.command for v in a.verification if not v.passed and not v.skipped_reason]
        script = self.checkpoint_script.get(p.id)
        if script:
            decision = script.pop(0)
        elif self.evidence_aware and getattr(package, "evidence", None):
            return self._evidence_judgment(package, failed)
        elif p.id in self.blocked_task_ids and not p.user_guidance:
            decision = "BLOCKED"
        elif failed or (p.id in self.retry_task_ids and a.number == 1):
            decision = "RETRY"
        else:
            decision = "PASS"
        if isinstance(decision, Review):  # test knob: an explicit answer
            return decision
        if decision == "EVIDENCE":  # test knob: judge this call from the evidence bundle
            return self._evidence_judgment(package, failed)
        if decision == "ASK_USER":
            if getattr(package, "human_verification_allowed", False):
                return Review(ReviewDecision.ASK_USER, f"[MOCK] {p.id}: human observation needed",
                              verification_notes=["mock checkpoint"], human_request=mock_human_request(p, failed),
                              issues=[{"description": "[MOCK] perceptual quality cannot be judged from the evidence",
                                       "severity": "subjective", "disposition": "ask_user", "evidence_id": ""}])
            # Like Claude with a schema that does not offer ASK_USER: decide from the available evidence.
            self.log("CLAUDE", f"[MOCK] ASK_USER is not available for {p.id}; deciding from the evidence")
            decision = "RETRY" if failed else "PASS"
        if decision in ("PASS", "PASS_CHANGES"):  # PASS_CHANGES: accepted, but later phases must change
            return Review(ReviewDecision.PASS, f"[MOCK] {p.id} accepted", verification_notes=["mock checkpoint"],
                          plan_changes_required=decision == "PASS_CHANGES",
                          replan_reason="[MOCK] later phases need restructuring" if decision == "PASS_CHANGES" else "",
                          task_result={"result": f"[MOCK] {p.title} done", "important_decisions": [],
                                       "remaining_risks": []})
        if decision == "RETRY":
            return Review(ReviewDecision.RETRY, "[MOCK] phase rejected to exercise the retry loop",
                          problems=["Mock problem: integration not demonstrated"]
                          + [f"verification failed: {c}" for c in failed],
                          corrective_instructions=["Mock fix: wire the feature into the entry point"])
        if decision == "REPLAN":
            return Review(ReviewDecision.REPLAN, "[MOCK] remaining phases no longer fit",
                          problems=["Mock: later phases must be restructured"],
                          replan_reason="[MOCK] the implementation diverged from the plan")
        return Review(ReviewDecision.BLOCKED, "[MOCK] needs a user decision",
                      problems=["Mock blocker: choose an approach"])

    async def replan_phases(self, request: PhaseReplanRequest) -> list[Phase]:
        self.replan_requests.append(request)
        self.log("CLAUDE", f"[MOCK] phase replan requested: {request.reason}")
        await asyncio.sleep(self.delay)
        if self.replan_result is not None:
            return parse_phases(copy.deepcopy(self.replan_result))
        remaining = [p for p in request.plan.tasks if p.status not in TERMINAL_PHASE_STATUSES]
        return parse_phases([p.to_dict() for p in remaining] or [
            {"id": "PX", "title": "[MOCK] follow-up", "description": "Follow-up work.",
             "subtasks": ["Finish"], "acceptance_criteria": ["Done"]}])

    async def diagnose_blocker(self, packet: Any) -> Diagnosis:
        from app.orchestration.repair import DiagnosisLimits, build_diagnosis_prompt
        self.diagnoses.append(packet)
        self.diagnosis_prompts.append(build_diagnosis_prompt(packet, DiagnosisLimits())[0])
        self.log("CLAUDE", f"[MOCK] DIAGNOSE_BLOCKER {packet.phase.id}: {packet.reason}")
        await asyncio.sleep(self.delay)
        status = self.diagnose_script.pop(0) if self.diagnose_script else "REPAIR"
        if isinstance(status, Diagnosis):  # test knob: an explicit answer
            return status
        if status == "BLOCKED":
            return Diagnosis("BLOCKED", "[MOCK] the blocker needs a user decision",
                             blocked_reason="[MOCK] choose the intended traffic rule")
        return Diagnosis("REPAIR", "[MOCK] the intersection lock is never released",
                         recommended_actions=["[MOCK] release the lock when a car leaves the junction"],
                         avoid=["[MOCK] adding more probe scripts"], target_files=["src/junction.ts"],
                         target_tests=["node smoke.mjs"])

    def _evidence_judgment(self, package: Any, failed: list[str]) -> Review:
        """Mock Claude judging the evidence bundle (deterministic stand-in for the prompt's guidance)."""
        p = package.phase
        results = package.evidence.get("results") or []
        ok = [r for r in results if r.get("status") == "SATISFIED"]
        gaps = [r for r in results if r.get("required") and r.get("status") != "SATISFIED"]
        missing = [f"{r['requirement_id']} ({r['kind']}): {r['status']}" for r in gaps]
        observed = [f"{r['requirement_id']} {r['kind']}: {r.get('summary', '')}"[:200] for r in ok] or \
                   ["no required property was demonstrated automatically"]
        broken = [r for r in gaps if r.get("status") == "FAILED"]
        self.log("CLAUDE", f"[MOCK] judging evidence of {p.id}: {len(ok)} satisfied, {len(gaps)} required gap(s)")
        if broken or failed:
            return Review(ReviewDecision.RETRY, f"[MOCK] evidence shows a defect in {p.id}",
                          problems=[f"evidence {r['requirement_id']} FAILED: {r.get('summary', '')}"[:300] for r in broken]
                          + [f"verification failed: {c}" for c in failed],
                          corrective_instructions=[f"[MOCK] Make {r['requirement_id']} pass: {r.get('goal', '')}"
                                                   for r in broken] or ["[MOCK] fix the failing verification"],
                          evidence_sufficient=False, missing_evidence=missing)
        human = [r for r in gaps if r.get("strategy") == "human"]
        if gaps and human and package.human_verification_allowed:
            return Review(ReviewDecision.ASK_USER, f"[MOCK] {p.id}: {len(human)} observation(s) need a person",
                          evidence_sufficient=False, missing_evidence=missing,
                          human_request={"reason": "[MOCK] no automated capability can observe this",
                                         "questions": [f"In the running result, can you confirm: {r.get('goal')}?"
                                                       for r in human][:6],
                                         "observations": observed})
        if gaps:
            return Review(ReviewDecision.BLOCKED, f"[MOCK] required evidence for {p.id} cannot be obtained",
                          problems=[f"missing evidence: {m}" for m in missing], evidence_sufficient=False,
                          missing_evidence=missing)
        return Review(ReviewDecision.PASS, f"[MOCK] {p.id} accepted on its evidence", evidence_sufficient=True,
                      verification_notes=observed, task_result={"result": f"[MOCK] {p.title} done (evidence)",
                                                                "important_decisions": [], "remaining_risks": []})

    async def evidence_plan(self, packet: Any) -> dict[str, Any]:
        from app.evidence.planning import EvidencePlanLimits, build_evidence_plan_prompt
        p = packet.phase
        self.evidence_plans.append(packet)
        self.evidence_plan_prompts.append(build_evidence_plan_prompt(packet, EvidencePlanLimits())[0])
        self.log("CLAUDE", f"[MOCK] EVIDENCE_PLAN {p.id}: {len(packet.snapshot.usable())} usable capabilit(ies)")
        await asyncio.sleep(self.delay)
        script = self.evidence_plan_script.get(p.id)
        if script:
            item = script.pop(0)
            return item(packet) if callable(item) else copy.deepcopy(item)
        reqs = []
        only = set((getattr(packet, "replan", None) or {}).get("requirements") or [])
        for r in p.evidence_requirements():
            if only and r.id not in only:
                continue
            caps = [c for c in packet.snapshot.usable() if c.kind == r.kind
                    and c.constraints.get("lifecycle") != "service" and not c.constraints.get("requires_resource")]
            if caps:
                reqs.append({"requirement_id": r.id, "strategy": "execute", "goal": r.goal, "coverage": "direct",
                             "steps": [{"capability_id": c.id} for c in caps[:2]],
                             "reason": "[MOCK] a capability of the same kind exists"})
            elif packet.human_gate_open and r.kind in ("visual_evidence", "human_judgment", "interaction"):
                reqs.append({"requirement_id": r.id, "strategy": "human", "goal": r.goal,
                             "reason": "[MOCK] only a person can observe this"})
            else:
                reqs.append({"requirement_id": r.id, "strategy": "unavailable", "goal": r.goal,
                             "reason": f"[MOCK] no capability of kind {r.kind}"})
        return {"summary": f"[MOCK] evidence plan for {p.id}", "requirements": reqs, "limitations": []}

    async def consistency_review(self, packet: Any) -> Review:
        from app.orchestration.consistency import (
            ConsistencyLimits, build_consistency_prompt, check_justified, parse_consistency,
        )
        p = packet.phase
        self.consistency_packets.append(packet)
        self.consistency_prompts.append(build_consistency_prompt(packet, ConsistencyLimits())[0])
        self.log("CLAUDE", f"[MOCK] CONSISTENCY_REVIEW {p.id}: {packet.trigger.get('reason', '')}")
        await asyncio.sleep(self.delay)
        unresolved = list(packet.trigger.get("unresolved") or [])
        script = self.consistency_script.get(p.id)
        item = script.pop(0) if script else None
        if isinstance(item, Review):
            return item
        if isinstance(item, dict):  # a raw answer, validated like Claude's
            return parse_consistency(item, unresolved_ids=unresolved, allow_ask_user=packet.human_gate_open)
        results = {r.get("requirement_id"): r for r in (packet.evidence or {}).get("results") or []}
        failed = [rid for rid in unresolved if (results.get(rid) or {}).get("status") == "FAILED"]
        human = [rid for rid in unresolved if (results.get(rid) or {}).get("strategy") == "human"]
        decision = item or ("RETRY" if failed else ("ASK_USER" if human and packet.human_gate_open else "BLOCKED"))
        raw: dict[str, Any] = {"decision": decision, "summary": f"[MOCK] consistency review of {p.id}: {decision}",
                               "justification": "", "evidence_dispositions": [], "problems": [],
                               "corrective_instructions": []}
        if decision == "PASS":
            raw["justification"] = "[MOCK] the unresolved evidence is covered by substitute evidence"
            raw["evidence_dispositions"] = [{"requirement_id": rid, "disposition": "substitute_evidence",
                                             "reason": "[MOCK] covered by the passing phase verification"}
                                            for rid in unresolved]
            raw["phase_result"] = {"result": f"[MOCK] {p.title} done (consistency review)", "important_decisions": [],
                                   "remaining_risks": []}
        elif decision == "PASS_UNJUSTIFIED":  # test knob: a PASS that leaves the evidence unexplained
            raw["decision"] = "PASS"
        elif decision == "RETRY":
            raw["problems"] = [f"[MOCK] required evidence unresolved: {', '.join(unresolved)}"]
            raw["corrective_instructions"] = [f"[MOCK] make {rid} pass: {(results.get(rid) or {}).get('goal', '')}"
                                              for rid in unresolved]
            raw["evidence_dispositions"] = [{"requirement_id": rid, "disposition": "still_required",
                                             "reason": "[MOCK] a repair can produce it"} for rid in unresolved]
        elif decision == "ASK_USER":
            raw["human_request"] = {"reason": "[MOCK] only a person can close the evidence gap",
                                    "questions": [f"In the running result, can you confirm: "
                                                  f"{(results.get(rid) or {}).get('goal', rid)}?" for rid in unresolved],
                                    "question_targets": list(unresolved), "observations": []}
        try:
            return parse_consistency(raw, unresolved_ids=unresolved, allow_ask_user=packet.human_gate_open)
        except ValueError:
            if decision != "PASS_UNJUSTIFIED":
                raise
            # like a misbehaving provider: hand back the unjustified PASS; the scheduler must reject it
            r = Review(ReviewDecision.PASS, raw["summary"], kind="CONSISTENCY_REVIEW",
                       consistency={"justification": "", "dispositions": []})
            assert check_justified(r, unresolved)
            return r

    async def mid_phase_diagnose(self, packet: Any) -> Any:
        from app.models.guard import MidPhaseDiagnosis
        from app.orchestration.impl_guard import MidPhaseLimits, build_mid_phase_prompt
        self.mid_phase_packets.append(packet)
        self.mid_phase_prompts.append(build_mid_phase_prompt(packet, MidPhaseLimits())[0])
        self.log("CLAUDE", f"[MOCK] MID_PHASE_DIAGNOSE {packet.phase.id}: {packet.reason}")
        await asyncio.sleep(self.delay)
        item = self.mid_phase_script.pop(0) if self.mid_phase_script else "CONTINUE"
        if isinstance(item, MidPhaseDiagnosis):
            return item
        if item == "BLOCKED":
            return MidPhaseDiagnosis("BLOCKED", "[MOCK] the worker needs a decision", "[MOCK] two valid designs",
                                     blocked_reason="[MOCK] choose the persistence format")
        return MidPhaseDiagnosis("CONTINUE", "[MOCK] the worker keeps re-reading the same module",
                                 "[MOCK] the parser is already implemented; only the wiring is missing",
                                 next_actions=["[MOCK] wire the parser into the entry point", "[MOCK] run the tests"],
                                 avoid=["[MOCK] re-exploring the repository"], target_files=["src/main.ts"],
                                 target_checks=["npm test"])

    async def human_followup(self, packet: Any) -> Review:
        from app.orchestration.human_gate import HumanFollowupLimits, build_human_followup_prompt
        p, fb = packet.phase, packet.feedback
        self.followups.append(packet)
        self.followup_prompts.append(build_human_followup_prompt(packet, HumanFollowupLimits())[0])
        self.log("CLAUDE", f"[MOCK] HUMAN_FOLLOWUP {p.id}: adjudicating {fb.verdict}")
        await asyncio.sleep(self.delay)
        script = self.followup_script.get(p.id)
        default = {"LOOKS_GOOD": "PASS", "ISSUE_FOUND": "RETRY"}.get(fb.verdict, "BLOCKED")
        decision = script.pop(0) if script else default
        if isinstance(decision, Review):
            return decision
        if decision == "PASS":
            return Review(ReviewDecision.PASS, f"[MOCK] the user's verification of {p.id} is accepted",
                          verification_notes=[f"user answered {fb.verdict}"],
                          task_result={"result": f"[MOCK] {p.title} done (human-verified)", "important_decisions": [],
                                       "remaining_risks": []})
        if decision == "RETRY":
            observed = fb.note or "the user could not confirm the behaviour"
            return Review(ReviewDecision.RETRY, f"[MOCK] the user's observation requires a repair of {p.id}",
                          problems=[f"User observed: {observed}"],
                          corrective_instructions=[f"[MOCK] Fix what the user observed: {observed}"],
                          derived_criteria=[{"text": f"[MOCK] The user-observed problem is resolved: {observed}"}],
                          do_not_redo=[f"{s.id} {s.title}" for s in p.subtasks if s.done],
                          issues=[{"description": observed, "severity": "functional", "disposition": "repair"}])
        if decision == "ASK_USER":  # misbehaving provider (the real schema never offers it here)
            return Review(ReviewDecision.ASK_USER, "[MOCK] asking again",
                          human_request={"reason": "again", "questions": ["Are you sure?"], "observations": []})
        return Review(ReviewDecision.BLOCKED, f"[MOCK] {p.id} cannot be accepted without the user's observation",
                      problems=[f"user answered {fb.verdict}" + (f": {fb.note}" if fb.note else "")])


def mock_human_request(p: Any, failed: list[str]) -> dict[str, Any]:
    """Questions generated from THIS phase (its contract's human checklist / evidence goals, else its criteria)."""
    c = p.verification_contract
    goals = (list(c.human_gate.checklist) or [r.goal for r in c.evidence_requirements if r.required]) if c else []
    goals = goals or list(p.acceptance_criteria)
    return {"reason": f"[MOCK] automated evidence cannot confirm what the user sees in {p.id}",
            "questions": [f"In the running result, can you confirm: {g}?" for g in goals[:3]],
            "observations": ([f"verification failed: {x}" for x in failed] or ["automated verification passed"]),
            "checklist": []}
