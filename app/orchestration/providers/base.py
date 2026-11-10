"""Orchestrator provider abstraction. The only real implementation is
ClaudeCodeCLIProvider (Claude via the installed Claude Code CLI)."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.models.plan import Plan
from app.models.review import Review
from app.models.task import TaskResult

LogFn = Callable[[str, str], None]  # (source, text)


class OrchestratorError(RuntimeError):
    """Claude could not produce a usable answer. The scheduler pauses on this;
    the worker never takes over planning/review."""


@dataclass
class PlanRequest:
    goal: str
    repo_path: str
    repo_summary: str = ""  # git/branch/dirty/Unreal/codebase-memory facts


@dataclass
class ReplanRequest:
    plan: Plan
    completed_results: list[TaskResult]
    reason: str
    repo_path: str


@dataclass
class PhaseReplanRequest:
    """V2: revise the FUTURE phases. Completed phases are history and are never changed."""
    goal: str
    plan: Plan  # current plan; tasks are Phase objects with their statuses
    reason: str
    repo_path: str
    superseded_phase_id: str = ""  # set when the checkpoint decision was REPLAN
    checkpoint_summary: str = ""
    checkpoint_problems: list[str] = field(default_factory=list)
    user_guidance: str = ""


@dataclass
class OrchestratorCall:
    """Telemetry about one orchestrator invocation (persisted)."""
    kind: str
    executable: str = ""
    model: str = ""
    argv: list[str] = field(default_factory=list)
    started_at: float = 0.0
    duration: float = 0.0
    exit_code: int | None = None
    cost_usd: float | None = None  # only when reported by the CLI
    usage: dict[str, Any] = field(default_factory=dict)
    session_id: str = ""
    error: str = ""
    num_turns: int | None = None  # as reported by the CLI's result event
    assistant_turns: int = 0  # model responses observed in the stream
    tool_calls: int = 0  # tool uses observed in the stream (excluding the structured answer)
    attempt: int = 1


@dataclass
class SmokeResult:
    ok: bool
    executable: str
    detail: str
    model: str = ""
    extra: dict[str, Any] = field(default_factory=dict)


class OrchestratorProvider(ABC):
    name = "orchestrator"

    def __init__(self, log: LogFn | None = None):
        self.log: LogFn = log or (lambda s, t: None)
        self.calls: list[OrchestratorCall] = []

    @abstractmethod
    async def plan(self, request: PlanRequest) -> Plan: ...

    @abstractmethod
    async def review(self, package: "ReviewPackage") -> Review: ...  # noqa: F821

    @abstractmethod
    async def replan(self, request: ReplanRequest) -> Plan: ...

    # --- Pipeline V2 (phase-based). Providers without V2 support refuse clearly. ---

    async def plan_phases(self, request: PlanRequest) -> Plan:
        raise OrchestratorError(f"{self.name} does not support phase-based (V2) planning")

    async def checkpoint(self, package: "CheckpointPackage") -> Review:  # noqa: F821
        raise OrchestratorError(f"{self.name} does not support phase checkpoints (V2)")

    async def replan_phases(self, request: PhaseReplanRequest) -> list:
        """Returns the revised list of remaining phases (list[Phase])."""
        raise OrchestratorError(f"{self.name} does not support phase replanning (V2)")

    async def diagnose_blocker(self, packet: "DiagnosisPacket") -> "Diagnosis":  # noqa: F821
        """V2 repair escalation: structured guidance for a stuck repair (never edits files)."""
        raise OrchestratorError(f"{self.name} does not support DIAGNOSE_BLOCKER (V2 repair)")

    async def human_followup(self, packet: "HumanFollowupPacket") -> Review:  # noqa: F821
        """V4: adjudicate the user's answer to an ASK_USER checkpoint. PASS / RETRY / BLOCKED only."""
        raise OrchestratorError(f"{self.name} does not support HUMAN_FOLLOWUP (V4 human verification)")

    async def evidence_plan(self, packet: "EvidencePlanPacket") -> dict[str, Any]:  # noqa: F821
        """V4 pass 2: the RAW evidence plan (the orchestrator validates it against the discovered capabilities
        before anything runs)."""
        raise OrchestratorError(f"{self.name} does not support EVIDENCE_PLAN (V4 evidence)")

    async def consistency_review(self, packet: "ConsistencyPacket") -> Review:  # noqa: F821
        """V4 pass 3: adjudicate a checkpoint whose answer contradicts its own evidence judgment. PASS (justified
        for every unresolved required requirement) / RETRY / BLOCKED / ASK_USER (human gate open only)."""
        raise OrchestratorError(f"{self.name} does not support CONSISTENCY_REVIEW (V4 evidence)")

    async def mid_phase_diagnose(self, packet: "MidPhasePacket") -> "MidPhaseDiagnosis":  # noqa: F821
        """V4 pass 3: diagnose an implementation session the guard stopped (CONTINUE with guidance / BLOCKED)."""
        raise OrchestratorError(f"{self.name} does not support MID_PHASE_DIAGNOSE (implementation guard)")

    async def cancel(self) -> None:
        return None
