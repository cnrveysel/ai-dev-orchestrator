"""The capability provider contract.

A provider knows ONE family of capabilities. It can:
  discover  report what it can genuinely do here (bounded, read-only, never launches the project)
  validate  accept or reject a planned step for one of its capabilities (no free-form execution)
  execute   run a validated step, bounded by the executor's limits, and report what happened
and, for capabilities that need something running while evidence is observed, the lifecycle
  start -> ready -> capture -> stop

A step capability that observes a running resource declares `requires_resource` (the key of the
lifecycle capability, or "*" for any resource of the requirement); the plan must list that resource and
the step receives its handle in ExecContext.resources. That is how one provider (e.g. a capture or a
UI driver) observes an application another provider started - without the scheduler knowing either.

The orchestration layer never looks inside a provider. New providers (a browser, a game engine, a
device, an emulator, ...) plug in through the registry without touching the scheduler.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable

from app.evidence.freshness import SCOPE_REPOSITORY
from app.evidence.model import Capability, PlanStep


@dataclass
class DiscoveryContext:
    repo_path: str
    phase: Any  # the phase whose evidence is being planned
    phases: list[Any] = field(default_factory=list)  # every phase of the run (their commands are known)
    ran_commands: dict[str, Any] = field(default_factory=dict)  # command -> last VerificationResult in this run
    settings: Any = None  # OrchestrationSettings (evidence_provider_settings holds provider-specific keys)
    human_gate_open: bool = False
    # V4 pass 3: repository files the worker changed in this phase (a capability declared in one of them is
    # marked in its provenance: the worker defined the check it is judged by)
    changed_files: list[str] = field(default_factory=list)

    def provider_setting(self, key: str, default: str = "") -> str:
        return str((getattr(self.settings, "evidence_provider_settings", None) or {}).get(key, default) or default)


@dataclass
class ExecContext:
    repo_path: str
    timeout: float  # this step's time budget (seconds)
    max_capture_bytes: int  # what is kept in memory for the artifact
    log: Callable[[str], None] = lambda s: None
    ready_timeout: float = 60.0
    # --- V4 pass 3 ---
    resources: dict[str, Any] = field(default_factory=dict)  # capability key -> handle of a READY resource
    prior_artifacts: dict[str, list[dict[str, Any]]] = field(default_factory=dict)  # step key -> artifact metadata
    phase_id: str = ""
    requirement_id: str = ""
    max_artifact_bytes: int = 1_000_000  # a full log (e.g. a build's) may be kept up to this size (head + tail)
    # when this attempt's evidence collection started: a generated file (log, report) last written before it was
    # not produced by an evidence step - it may come from an earlier run or from the worker - and proves nothing
    collection_started: float = 0.0


@dataclass
class StepOutcome:
    status: str  # PASSED | FAILED | TIMEOUT | ERROR | UNAVAILABLE
    exit_code: int | None = None
    output: str = ""  # bounded by ExecContext.max_capture_bytes
    summary: str = ""
    reason: str = ""
    truncated: bool = False
    duration: float = 0.0
    artifact_kind: str = "log"
    # --- V4 pass 3 ---
    details: dict[str, Any] = field(default_factory=dict)  # structured, compact (e.g. reduced build diagnostics)
    diagnostics: list[str] = field(default_factory=list)  # compact lines for prompts (never a whole log)
    # binary artifacts the executor stores through the artifact store: {data, media_type, ext, label,
    # description, metadata}; a provider never writes artifact files itself
    files: list[dict[str, Any]] = field(default_factory=list)
    cause: str = ""  # optional hint: strategy (the environment/capability failed) | product


class CapabilityProvider(ABC):
    name = "provider"
    version = "1"  # recorded in capability/step provenance; bump when execution semantics change

    @abstractmethod
    async def discover(self, ctx: DiscoveryContext) -> list[Capability]:
        """What this provider can do here. Must be bounded and must not modify the repository."""

    def validate(self, cap: Capability, step: PlanStep) -> str:
        """"" if the step may be executed, otherwise the reason it is rejected."""
        return "" if not step.args else f"{cap.id} does not accept arguments"

    @abstractmethod
    async def execute(self, cap: Capability, step: PlanStep, ctx: ExecContext) -> StepOutcome:
        """Run ONE validated step. Verification only: never edit, repair or commit anything."""

    # --- freshness (V4 pass 3): what a step's result depends on
    def input_scope(self, cap: Capability, step: PlanStep, repo_path: str = "") -> str:
        """repository (default: any authored file may matter) | files (exactly input_files())."""
        return SCOPE_REPOSITORY

    def input_files(self, cap: Capability, step: PlanStep, repo_path: str = "") -> list[str]:
        return []

    # --- lifecycle (START -> READY_PROBE -> OBSERVE -> CAPTURE -> STOP); optional
    def supports_lifecycle(self, cap: Capability) -> bool:
        return False

    async def start(self, cap: Capability, ctx: ExecContext) -> Any:
        raise NotImplementedError(f"{self.name} has no lifecycle capabilities")

    async def ready(self, handle: Any, cap: Capability, ctx: ExecContext) -> tuple[str, str]:
        return "ready", ""

    def capture(self, handle: Any) -> str:
        return ""

    async def stop(self, handle: Any) -> None:
        return None
