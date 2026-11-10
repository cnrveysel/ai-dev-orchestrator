"""Visual evidence: CAPTURE and ANALYSIS are separate capabilities.

    visual_capture  records an image of a RUNNING resource the plan started (requires_resource "*"). A
                    captured image proves only that something was captured: visual_capture is a
                    non-observational kind, so on its own it can never satisfy a visual requirement.
    image_analysis  judges a captured image (its argument is the number of the capture step of the same
                    requirement). Only an analysis - or a person - can say what the image shows.

Both are backed by pluggable backends. Without a backend the capability is reported UNAVAILABLE with
the reason; nothing is ever fabricated (no placeholder images, no assumed analysis). Image artifacts
are stored through the artifact store with their metadata (dimensions, capture time, source
capability, phase and requirement).
"""
from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

from app.evidence.model import ERROR, FAILED, INFERRED, PASSED, UNAVAILABLE, Capability, PlanStep
from app.evidence.providers.base import CapabilityProvider, DiscoveryContext, ExecContext, StepOutcome


@dataclass
class CapturedImage:
    data: bytes  # encoded image (PNG)
    width: int
    height: int
    media_type: str = "image/png"
    description: str = ""  # what was captured (e.g. "window 'Game' of the started runtime")
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImageAnalysis:
    passed: bool | None  # None = the analyzer could not decide
    summary: str
    findings: list[str] = field(default_factory=list)


class CaptureBackend(ABC):
    name = "capture"

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        """(usable here, why / how)."""

    @abstractmethod
    async def capture(self, handle: Any, *, timeout: float) -> CapturedImage | None:
        """An image of what the running resource shows, or None if there is nothing to capture."""


class ImageAnalyzer(ABC):
    name = "analyzer"

    @abstractmethod
    def available(self) -> tuple[bool, str]:
        ...

    @abstractmethod
    async def analyze(self, image_path: str, *, question: str, timeout: float) -> ImageAnalysis:
        ...


class VisualProvider(CapabilityProvider):
    name = "visual"
    version = "1"

    def __init__(self, capture: CaptureBackend | None = None, analyzer: ImageAnalyzer | None = None):
        self.capture_backend, self.analyzer = capture, analyzer

    async def discover(self, ctx: DiscoveryContext) -> list[Capability]:
        ok, why = self.capture_backend.available() if self.capture_backend else (
            False, "no screenshot/capture backend is available in this environment")
        a_ok, a_why = self.analyzer.available() if self.analyzer else (
            False, "no image analysis backend: a captured image alone is never visual evidence")
        return [
            Capability(id="", kind="visual_capture", name="visual capture of a running resource",
                       description="captures an image of what a running resource shows (list the resource under "
                                   "`resources`); proves only that it was captured",
                       source="environment", confidence=INFERRED if ok else UNAVAILABLE, provider=self.name,
                       key="visual:capture", constraints={"requires_resource": "*", "timeout_seconds": 30,
                                                          "accepts_args": False, "produces": "image"},
                       details={"backend": self.capture_backend.name if self.capture_backend else ""},
                       reason=why, provenance={"source_detail": "capture backend"}),
            Capability(id="", kind="image_analysis", name="image analysis",
                       description="judges an image captured by an earlier step of the same requirement against "
                                   "the question in expect_output",
                       source="environment", confidence=INFERRED if a_ok else UNAVAILABLE, provider=self.name,
                       key="visual:analysis", constraints={"input_step": "visual_capture", "accepts_args": True,
                                                           "max_args": 1, "timeout_seconds": 120,
                                                           "args_help": "the number of the capture step"},
                       details={"analyzer": self.analyzer.name if self.analyzer else ""},
                       reason=a_why, provenance={"source_detail": "image analyzer"}),
        ]

    def validate(self, cap: Capability, step: PlanStep) -> str:
        if cap.kind == "visual_capture":
            return "" if not step.args else f"{cap.id} does not accept arguments"
        if cap.kind == "image_analysis":
            if len(step.args) != 1 or not step.args[0]:
                return f"{cap.id} takes the number of the capture step it analyses"
            return ""
        return f"unknown visual capability {cap.kind}"

    async def execute(self, cap: Capability, step: PlanStep, ctx: ExecContext) -> StepOutcome:
        t0 = time.monotonic()
        if cap.kind == "visual_capture":
            if self.capture_backend is None:
                return StepOutcome(UNAVAILABLE, reason="no capture backend", cause="strategy")
            if not ctx.resources:
                return StepOutcome(UNAVAILABLE, reason="no running resource to capture", cause="strategy")
            rkey, handle = next(iter(ctx.resources.items()))
            img = await self.capture_backend.capture(handle, timeout=ctx.timeout)
            if img is None or not img.data:
                return StepOutcome(UNAVAILABLE, summary=f"nothing to capture for {rkey} (no visible output found)",
                                   reason="nothing to capture", cause="strategy", duration=time.monotonic() - t0)
            meta = {"width": img.width, "height": img.height, "captured_at": time.time(), "resource": rkey,
                    "backend": self.capture_backend.name, **img.metadata}
            return StepOutcome(
                PASSED, summary=f"captured a {img.width}x{img.height} image of {rkey}"
                                + (f" ({img.description})" if img.description else "")
                                + " - captured only, not analysed", duration=time.monotonic() - t0,
                artifact_kind="image", details={"image": {"width": img.width, "height": img.height}},
                files=[{"data": img.data, "media_type": img.media_type, "kind": "image", "label": f"{step.key}-capture",
                        "description": f"image captured by {step.key}" + (f": {img.description}"
                                                                           if img.description else ""),
                        "metadata": meta}])
        if cap.kind == "image_analysis":
            if self.analyzer is None:
                return StepOutcome(UNAVAILABLE, reason="no image analyzer", cause="strategy")
            ref = step.args[0] if step.args else ""
            images = [a for a in ctx.prior_artifacts.get(ref, []) if str(a.get("media_type", "")).startswith("image/")]
            if not images:
                return StepOutcome(UNAVAILABLE, reason=f"step {ref} produced no image to analyse", cause="strategy")
            question = step.expect_output or step.purpose or "Does the image show the expected result?"
            res = await self.analyzer.analyze(images[0]["path_or_ref"], question=question, timeout=ctx.timeout)
            status = {True: PASSED, False: FAILED}.get(res.passed, ERROR)
            return StepOutcome(status, summary=f"image analysis of {ref}: {res.summary}"[:400],
                               reason="" if res.passed else ("the analysis found a problem" if res.passed is False
                                                             else "the analysis could not decide"),
                               output="\n".join([res.summary, *res.findings]), duration=time.monotonic() - t0,
                               details={"analysis": {"image": images[0].get("id"), "findings": res.findings[:6]}})
        return StepOutcome(UNAVAILABLE, reason=f"unknown visual capability {cap.kind}", cause="strategy")
