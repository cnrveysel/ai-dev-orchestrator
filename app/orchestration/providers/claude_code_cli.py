"""Claude orchestrator via the locally installed Claude Code CLI.

Uses `claude --print --output-format stream-json --verbose --json-schema <schema>`
with the prompt written to stdin, after which stdin is CLOSED (EOF). Relies
entirely on the user's existing Claude Code authentication/config: no SDK, no
HTTP API, no API key.

Usage guards (per call): hard wall-clock timeout, turn and tool-call limits
enforced from the event stream (the installed CLI has no --max-turns flag),
--max-budget-usd when the CLI supports it, and at most one automatic retry
(default: none).
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from app.config.discovery import validate_claude_executable
from app.config.settings import ClaudeSettings
from app.models.phase import (
    PHASE_PLAN_JSON_SCHEMA, PHASE_REPLAN_JSON_SCHEMA, TERMINAL_PHASE_STATUSES, Phase, parse_phase_plan, parse_phases,
)
from app.models.plan import PLAN_JSON_SCHEMA, Plan, PlanValidationError
from app.evidence.planning import (
    EVIDENCE_PLAN_JSON_SCHEMA, EvidencePlanLimits, EvidencePlanPacket, build_evidence_plan_prompt,
)
from app.models.guard import MID_PHASE_JSON_SCHEMA, MidPhaseDiagnosis
from app.models.human import criteria_lines
from app.models.repair import DIAGNOSIS_JSON_SCHEMA, Diagnosis
from app.models.review import (
    CHECKPOINT_DECISIONS, FOLLOWUP_DECISIONS, HUMAN_FOLLOWUP_JSON_SCHEMA, REVIEW_JSON_SCHEMA, Review, checkpoint_schema,
)
from app.orchestration.checkpoint import CheckpointContextLimits, CheckpointPackage, build_checkpoint_context
from app.orchestration.consistency import (
    ConsistencyLimits, ConsistencyPacket, build_consistency_prompt, consistency_schema, parse_consistency,
)
from app.orchestration.human_gate import HumanFollowupLimits, HumanFollowupPacket, build_human_followup_prompt
from app.orchestration.impl_guard import MidPhaseLimits, MidPhasePacket, build_mid_phase_prompt
from app.orchestration.providers.base import (
    LogFn, OrchestratorCall, OrchestratorError, OrchestratorProvider, PhaseReplanRequest, PlanRequest,
    ReplanRequest, SmokeResult,
)
from app.orchestration.repair import DiagnosisLimits, DiagnosisPacket, build_diagnosis_prompt
from app.orchestration.reviewer import (
    ReviewContextLimits, ReviewContextTooLarge, ReviewPackage, build_review_context, utf8_len,
)
from app.workers.process_runner import ProcessLaunchError, ProcessRunner, tail

# Flags this adapter needs; verified against `claude --help` before first use.
REQUIRED_FLAGS = ["--print", "--output-format", "--json-schema", "--verbose"]
# Only passed when `claude --help` lists them (never invented).
OPTIONAL_FLAGS = ["--model", "--effort", "--no-session-persistence", "--tools", "--allowedTools",
                  "--append-system-prompt", "--permission-mode", "--permission-prompts",
                  "--max-budget-usd", "--max-turns"]

PLANNER_TOOLS = "Read,Glob,Grep"
# After Claude emits its final result event it must exit; otherwise it is terminated.
RESULT_EXIT_GRACE_SECONDS = 10.0

SYSTEM_ROLE = """You are the TECHNICAL LEAD / ORCHESTRATOR of the "AI Dev Orchestrator" system.
There is exactly ONE implementation worker: a local Qwen coding model running through Codex NInfer.
The worker executes ONE focused task at a time, strictly sequentially; there are never parallel workers.
You own the global plan and all review decisions. The worker owns only its current task.
You are in a read-only advisory role in this session: never modify files and never run commands.
Always answer through the structured output schema you were given."""

# Pipeline V2 (phase-based execution): same role, but the worker owns whole phases.
SYSTEM_ROLE_V2 = """You are the TECHNICAL LEAD / ORCHESTRATOR of the "AI Dev Orchestrator" system (phase-based pipeline).
There is exactly ONE implementation worker: a local Qwen coding model running through Codex NInfer.
The worker executes ONE phase at a time, strictly sequentially, in one long-lived session per phase, and manages
the phase's internal subtasks itself. There are never parallel workers.
You own the global phase plan and review each phase only at its boundary (checkpoint).
You are in a read-only advisory role in this session: never modify files and never run commands.
Always answer through the structured output schema you were given."""

V2_PURPOSES = frozenset({"PLAN_V2", "REPLAN_V2", "CHECKPOINT", "DIAGNOSE_BLOCKER", "HUMAN_FOLLOWUP",
                         "EVIDENCE_PLAN", "CONSISTENCY_REVIEW", "MID_PHASE_DIAGNOSE"})


@dataclass(frozen=True)
class CallLimits:
    # PLAN | REPLAN | REVIEW | SMOKE | PLAN_V2 | REPLAN_V2 | CHECKPOINT | DIAGNOSE_BLOCKER | HUMAN_FOLLOWUP |
    # EVIDENCE_PLAN | CONSISTENCY_REVIEW | MID_PHASE_DIAGNOSE
    purpose: str
    timeout: float
    max_turns: int
    max_tool_calls: int
    max_budget_usd: float  # 0 = not passed
    tools: str  # "" = no tools
    retries: int  # 0 or 1
    model: str = ""  # "" = CLI default


def limits_for(purpose: str, s: ClaudeSettings) -> CallLimits:
    retries = max(0, min(1, int(s.auto_retries)))
    read = s.allow_repo_read_tools
    if purpose in ("PLAN", "REPLAN", "PLAN_V2", "REPLAN_V2"):
        return CallLimits(purpose, s.plan_timeout_seconds, s.plan_max_turns,
                          s.plan_max_tool_calls if read else 0, s.plan_max_budget_usd,
                          PLANNER_TOOLS if read else "", retries, s.model)
    if purpose == "CHECKPOINT":
        # Like REVIEW: no tools, all evidence is provided. One call per phase, so a larger budget.
        return CallLimits(purpose, s.checkpoint_timeout_seconds, s.review_max_turns, 0, s.checkpoint_max_budget_usd,
                          "", retries, s.review_model or s.model)
    if purpose == "DIAGNOSE_BLOCKER":
        # A cheap, bounded diagnosis of a stuck repair: no tools (no repository crawl), the packet
        # carries the evidence; structured guidance only.
        return CallLimits(purpose, s.diagnose_timeout_seconds, s.review_max_turns, 0, s.diagnose_max_budget_usd,
                          "", retries, s.review_model or s.model)
    if purpose == "HUMAN_FOLLOWUP":
        # V4: adjudicate the user's answer to ASK_USER. No tools; the packet carries the evidence.
        return CallLimits(purpose, s.human_followup_timeout_seconds, s.review_max_turns, 0,
                          s.human_followup_max_budget_usd, "", retries, s.review_model or s.model)
    if purpose == "EVIDENCE_PLAN":
        # V4 pass 2: plan evidence with the DISCOVERED capabilities only. No tools: nothing else can be found or run.
        return CallLimits(purpose, s.evidence_plan_timeout_seconds, s.review_max_turns, 0,
                          s.evidence_plan_max_budget_usd, "", retries, s.review_model or s.model)
    if purpose == "CONSISTENCY_REVIEW":
        # V4 pass 3: resolve a self-contradictory checkpoint. No tools; its own turn limit and budget.
        return CallLimits(purpose, s.consistency_timeout_seconds, max(1, int(s.consistency_max_turns)), 0,
                          s.consistency_max_budget_usd, "", retries, s.review_model or s.model)
    if purpose == "MID_PHASE_DIAGNOSE":
        # V4 pass 3: the implementation guard stopped a worker session. No tools; short bounded packet.
        return CallLimits(purpose, s.mid_phase_timeout_seconds, s.review_max_turns, 0, s.mid_phase_max_budget_usd,
                          "", retries, s.review_model or s.model)
    if purpose == "REVIEW":
        # The reviewer never gets tools: the orchestrator provides all evidence.
        return CallLimits(purpose, s.review_timeout_seconds, s.review_max_turns, 0, s.review_max_budget_usd,
                          "", retries, s.review_model or s.model)
    if purpose == "SMOKE":
        return CallLimits(purpose, s.smoke_timeout_seconds, 2, 0, s.review_max_budget_usd, "", 0, s.model)
    raise ValueError(f"unknown Claude call purpose {purpose!r}")


class ClaudeCodeCLIProvider(OrchestratorProvider):
    name = "claude-code-cli"

    def __init__(self, settings: ClaudeSettings, log: LogFn | None = None):
        super().__init__(log)
        self.settings = settings
        self._runner: ProcessRunner | None = None
        self._flags: set[str] | None = None
        self._cancel_requested = False
        self.last_call: OrchestratorCall | None = None

    # ------------------------------------------------------------ capabilities

    @property
    def executable(self) -> str:
        return validate_claude_executable(self.settings.executable)

    async def detect_capabilities(self) -> dict[str, Any]:
        exe = self.executable
        help_res = await ProcessRunner().run([exe, "--help"], timeout=60)
        ver = await ProcessRunner().run([exe, "--version"], timeout=60)
        text = help_res.stdout + "\n" + help_res.stderr
        flags = {f for f in REQUIRED_FLAGS + OPTIONAL_FLAGS if f in text}
        self._flags = flags
        missing = [f for f in REQUIRED_FLAGS if f not in flags]
        return {"version": ver.stdout.strip(), "flags": sorted(flags), "missing": missing}

    async def _ensure_capabilities(self) -> set[str]:
        if self._flags is None:
            caps = await self.detect_capabilities()
            if caps["missing"]:
                raise OrchestratorError(
                    "installed Claude Code CLI lacks required flags: " + ", ".join(caps["missing"]))
        assert self._flags is not None
        return self._flags

    # ------------------------------------------------------------ invocation

    async def _build_argv(self, limits: CallLimits, schema: dict[str, Any]) -> list[str]:
        flags = await self._ensure_capabilities()
        argv = [self.executable, "--print", "--output-format", "stream-json", "--verbose",
                "--json-schema", json.dumps(schema, separators=(",", ":"))]
        if "--no-session-persistence" in flags:
            argv.append("--no-session-persistence")
        if limits.model and "--model" in flags:
            argv += ["--model", limits.model]
        if self.settings.effort and "--effort" in flags:
            argv += ["--effort", self.settings.effort]
        if "--append-system-prompt" in flags:
            argv += ["--append-system-prompt", SYSTEM_ROLE_V2 if limits.purpose in V2_PURPOSES else SYSTEM_ROLE]
        if "--tools" in flags:
            argv += ["--tools", limits.tools]
        if limits.tools and "--allowedTools" in flags:
            argv += ["--allowedTools", limits.tools]
        if "--permission-mode" in flags:
            argv += ["--permission-mode", "dontAsk"]  # anything not pre-approved is denied
        if "--permission-prompts" in flags:
            argv += ["--permission-prompts", "none"]  # never wait for a host to answer a prompt
        if limits.max_budget_usd > 0 and "--max-budget-usd" in flags:
            argv += ["--max-budget-usd", f"{limits.max_budget_usd:.2f}"]
        if limits.max_turns > 0 and "--max-turns" in flags:
            argv += ["--max-turns", str(limits.max_turns)]
        argv += list(self.settings.extra_args)
        return argv

    async def _invoke(self, purpose: str, prompt: str, schema: dict[str, Any], cwd: str | None) -> dict[str, Any]:
        limits = limits_for(purpose, self.settings)
        attempts = 1 + limits.retries
        for attempt in range(1, attempts + 1):
            try:
                return await self._invoke_once(limits, prompt, schema, cwd, attempt)
            except OrchestratorError as exc:
                if attempt >= attempts or self._cancel_requested:
                    raise
                self.log("SYSTEM", f"Claude {purpose} failed ({exc}); retrying ONCE (auto_retries=1)")
        raise AssertionError("unreachable")

    async def _invoke_once(self, limits: CallLimits, prompt: str, schema: dict[str, Any],
                           cwd: str | None, attempt: int) -> dict[str, Any]:
        purpose = limits.purpose
        if purpose in ("REVIEW", "CHECKPOINT"):
            # Final hard gate, before anything is spawned: an oversized review prompt never reaches Claude.
            limits_cls = ReviewContextLimits if purpose == "REVIEW" else CheckpointContextLimits
            size, cap = utf8_len(prompt), limits_cls.from_settings(self.settings).max_prompt_bytes
            if size > cap:
                raise ReviewContextTooLarge(size, cap)
        argv = await self._build_argv(limits, schema)
        flags = self._flags or set()
        call = OrchestratorCall(kind=purpose, executable=argv[0], model=limits.model or "CLI default",
                                argv=_redact_argv(argv), started_at=time.time(), attempt=attempt)
        self.last_call = call
        self.calls.append(call)
        budget = (f"${limits.max_budget_usd:.2f} via --max-budget-usd"
                  if limits.max_budget_usd > 0 and "--max-budget-usd" in flags else "not applied")
        turns_mode = "--max-turns" if "--max-turns" in flags else "enforced from stream"
        self.log("SYSTEM", f"Claude {purpose}: launching {argv[0]} (model={call.model}, timeout={limits.timeout:.0f}s, "
                           f"max turns={limits.max_turns} [{turns_mode}], max tool calls={limits.max_tool_calls}, "
                           f"tools={limits.tools or 'none'}, budget={budget})")
        result_event: dict[str, Any] = {}
        text_parts: list[str] = []
        message_ids: set[str] = set()
        limit_hit: list[str] = []
        runner = ProcessRunner(grace_seconds=3)
        loop = asyncio.get_running_loop()
        exit_timer: list[asyncio.TimerHandle] = []

        def stop(reason: str) -> None:
            if not limit_hit:
                limit_hit.append(reason)
                self.log("SYSTEM", f"Claude {purpose}: {reason} - terminating Claude")
                loop.create_task(runner.terminate(reason))

        def on_stdout(line: str) -> None:
            nonlocal result_event
            line = line.strip()
            if not line:
                return
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                self.log("CLAUDE", line)
                return
            etype = ev.get("type")
            if etype == "system" and ev.get("subtype") == "init":
                call.model = ev.get("model") or call.model
                self.log("CLAUDE", f"session init: model={ev.get('model')} tools={ev.get('tools')}")
            elif etype == "assistant":
                msg = ev.get("message") or {}
                mid = msg.get("id")
                if mid and mid not in message_ids:
                    message_ids.add(mid)
                    call.assistant_turns = len(message_ids)
                    if call.assistant_turns > limits.max_turns:
                        stop(f"turn limit exceeded ({call.assistant_turns} > {limits.max_turns})")
                for block in msg.get("content", []) or []:
                    btype = block.get("type")
                    if btype == "text" and block.get("text"):
                        text_parts.append(block["text"])
                        self.log("CLAUDE", block["text"])
                    elif btype == "tool_use":
                        name = block.get("name")
                        if name == "StructuredOutput":
                            self.log("CLAUDE", "structured output received")
                            continue
                        call.tool_calls += 1
                        self.log("CLAUDE", f"tool #{call.tool_calls} {name}: {_short(block.get('input'))}")
                        if call.tool_calls > limits.max_tool_calls:
                            stop(f"tool-call limit exceeded ({call.tool_calls} > {limits.max_tool_calls})")
            elif etype == "result":
                result_event = ev
                exit_timer.append(loop.call_later(
                    RESULT_EXIT_GRACE_SECONDS,
                    lambda: loop.create_task(runner.terminate("did not exit after its result"))))

        def on_stderr(line: str) -> None:
            if line.strip():
                self.log("CLAUDE", f"[stderr] {line}")

        def on_stdin_closed(n: int) -> None:
            self.log("SYSTEM", f"Claude {purpose}: prompt sent ({n} bytes), stdin closed (EOF)")

        self._runner = runner
        try:
            res = await runner.run(argv, cwd=cwd, stdin_data=prompt, timeout=limits.timeout,
                                   on_stdout=on_stdout, on_stderr=on_stderr, on_stdin_closed=on_stdin_closed)
        except ProcessLaunchError as exc:
            call.error = str(exc)
            raise OrchestratorError(str(exc)) from exc
        finally:
            self._runner = None
            for h in exit_timer:
                h.cancel()
        call.duration = res.duration
        call.exit_code = res.exit_code
        call.session_id = str(result_event.get("session_id", ""))
        if isinstance(result_event.get("total_cost_usd"), (int, float)):
            call.cost_usd = float(result_event["total_cost_usd"])
        if isinstance(result_event.get("usage"), dict):
            call.usage = result_event["usage"]
        if isinstance(result_event.get("num_turns"), int):
            call.num_turns = result_event["num_turns"]
        try:
            data = self._check_result(call, limits, res, result_event, text_parts, limit_hit)
        finally:
            self._log_usage(call)
        return data

    def _check_result(self, call: OrchestratorCall, limits: CallLimits, res, result_event: dict[str, Any],
                      text_parts: list[str], limit_hit: list[str]) -> dict[str, Any]:
        purpose = limits.purpose
        if res.cancelled:
            call.error = "cancelled"
            raise OrchestratorError(f"Claude {purpose} cancelled")
        if limit_hit:
            call.error = limit_hit[0]
            raise OrchestratorError(f"Claude {purpose} stopped: {limit_hit[0]}")
        if res.timed_out:
            call.error = f"hard timeout after {limits.timeout:.0f}s (terminated)"
            raise OrchestratorError(f"Claude {purpose} {call.error}")
        if res.terminated_reason and result_event:
            self.log("SYSTEM", f"Claude {purpose}: did not exit within {RESULT_EXIT_GRACE_SECONDS:.0f}s "
                               "after its result; terminated it (result kept)")
        if res.pipes_held_open:
            self.log("SYSTEM", f"Claude {purpose}: exited, but a child process still held its output pipes; "
                               "they were closed")
        if not result_event:
            call.error = f"no result event (exit {res.exit_code}): {tail(res.stderr, 800)}"
            raise OrchestratorError(f"Claude {purpose} failed: {call.error}")
        if result_event.get("is_error"):
            subtype = str(result_event.get("subtype") or "")
            detail = str(result_event.get("result") or "")
            if "budget" in subtype.lower():
                detail = f"budget limit ${limits.max_budget_usd:.2f} reached. {detail}".strip()
            call.error = f"{subtype}: {detail}" if subtype else detail
            raise OrchestratorError(f"Claude {purpose} returned an error: {call.error}")
        data = result_event.get("structured_output")
        if not isinstance(data, dict):
            data = _parse_json_text(str(result_event.get("result", "")) or "\n".join(text_parts))
        if not isinstance(data, dict):
            call.error = "no structured output in result"
            raise OrchestratorError(f"Claude {purpose}: response did not contain structured JSON")
        return data

    def _log_usage(self, c: OrchestratorCall) -> None:
        u = c.usage or {}
        if u:
            tokens = (f"input={u.get('input_tokens', '?')} cache_write={u.get('cache_creation_input_tokens', '?')} "
                      f"cache_read={u.get('cache_read_input_tokens', '?')} output={u.get('output_tokens', '?')}")
        else:
            tokens = "not reported"
        cost = f"${c.cost_usd:.4f} (reported by Claude CLI)" if c.cost_usd is not None else "not reported"
        cli_turns = f" (CLI num_turns={c.num_turns})" if c.num_turns is not None else ""
        self.log("SYSTEM", f"Claude usage [{c.kind}] model={c.model} elapsed={c.duration:.1f}s "
                           f"turns={c.assistant_turns}{cli_turns} tool_calls={c.tool_calls} tokens: {tokens} "
                           f"cost: {cost}" + (f" attempt={c.attempt}" if c.attempt > 1 else "")
                           + (f" ERROR: {c.error}" if c.error else ""))

    async def cancel(self) -> None:
        self._cancel_requested = True
        if self._runner is not None:
            await self._runner.cancel()

    # ------------------------------------------------------------ API

    async def plan(self, request: PlanRequest) -> Plan:
        limits = limits_for("PLAN", self.settings)
        data = await self._invoke("PLAN", build_plan_prompt(request, limits), PLAN_JSON_SCHEMA, request.repo_path)
        try:
            plan = Plan.from_dict(data)
        except PlanValidationError as exc:
            raise OrchestratorError(f"Claude returned an invalid plan: {exc}") from exc
        if not plan.goal:
            plan.goal = request.goal
        return plan

    async def review(self, package: ReviewPackage) -> Review:
        # Raises ReviewContextTooLarge (Claude not launched) if even the compacted context is too big.
        ctx = build_review_context(package, ReviewContextLimits.from_settings(self.settings))
        for line in ctx.log_lines():
            self.log("SYSTEM", line)
        data = await self._invoke("REVIEW", ctx.prompt, REVIEW_JSON_SCHEMA, package.repo_path or None)
        try:
            return Review.from_dict(data)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid review: {exc}") from exc

    async def replan(self, request: ReplanRequest) -> Plan:
        limits = limits_for("REPLAN", self.settings)
        data = await self._invoke("REPLAN", build_replan_prompt(request, limits), PLAN_JSON_SCHEMA, request.repo_path)
        try:
            return Plan.from_dict(data)
        except PlanValidationError as exc:
            raise OrchestratorError(f"Claude returned an invalid revised plan: {exc}") from exc

    # ------------------------------------------------------------ Pipeline V2

    async def plan_phases(self, request: PlanRequest) -> Plan:
        limits = limits_for("PLAN_V2", self.settings)
        data = await self._invoke("PLAN_V2", build_phase_plan_prompt(request, limits), PHASE_PLAN_JSON_SCHEMA,
                                  request.repo_path)
        try:
            plan = parse_phase_plan(data)
        except PlanValidationError as exc:
            raise OrchestratorError(f"Claude returned an invalid phase plan: {exc}") from exc
        if not plan.goal:
            plan.goal = request.goal
        return plan

    async def checkpoint(self, package: CheckpointPackage) -> Review:
        ctx = build_checkpoint_context(package, CheckpointContextLimits.from_settings(self.settings))
        for line in ctx.log_lines():
            self.log("SYSTEM", line)
        ask = package.human_verification_allowed  # V4: ASK_USER is in the schema only when the human gate is open
        data = await self._invoke("CHECKPOINT", ctx.prompt, checkpoint_schema(ask), package.repo_path or None)
        try:
            return Review.from_dict(data, allowed=(*CHECKPOINT_DECISIONS, "ASK_USER") if ask else CHECKPOINT_DECISIONS)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid checkpoint: {exc}") from exc

    async def human_followup(self, packet: HumanFollowupPacket) -> Review:
        prompt, stats = build_human_followup_prompt(packet, HumanFollowupLimits.from_settings(self.settings))
        self.log("SYSTEM", f"Claude HUMAN_FOLLOWUP context: prompt {stats['prompt_bytes']}/{stats['limit_bytes']} bytes "
                           f"(phase diff {stats['diff_original_bytes']} bytes before truncation)")
        data = await self._invoke("HUMAN_FOLLOWUP", prompt, HUMAN_FOLLOWUP_JSON_SCHEMA, packet.repo_path or None)
        try:
            review = Review.from_dict(data, allowed=FOLLOWUP_DECISIONS)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid human follow-up: {exc}") from exc
        review.kind = "HUMAN_FOLLOWUP"
        return review

    async def evidence_plan(self, packet: EvidencePlanPacket) -> dict[str, Any]:
        prompt, stats = build_evidence_plan_prompt(packet, EvidencePlanLimits.from_settings(self.settings))
        self.log("SYSTEM", f"Claude EVIDENCE_PLAN context: prompt {stats['prompt_bytes']}/{stats['limit_bytes']} bytes, "
                           f"{len(packet.snapshot.usable())} usable capabilit(ies)"
                           + (f" (compacted, level {stats['compacted']})" if stats.get("compacted") else ""))
        data = await self._invoke("EVIDENCE_PLAN", prompt, EVIDENCE_PLAN_JSON_SCHEMA, packet.repo_path or None)
        if not isinstance(data.get("requirements"), list):
            raise OrchestratorError("Claude returned an invalid evidence plan: no requirement list")
        return data  # validated by the orchestrator against the discovered capabilities

    async def consistency_review(self, packet: ConsistencyPacket) -> Review:
        prompt, stats = build_consistency_prompt(packet, ConsistencyLimits.from_settings(self.settings))
        self.log("SYSTEM", f"Claude CONSISTENCY_REVIEW context: prompt {stats['prompt_bytes']}/{stats['limit_bytes']} "
                           "bytes" + (" (optional sections compacted)" if stats.get("truncated") else ""))
        ask = packet.human_gate_open
        data = await self._invoke("CONSISTENCY_REVIEW", prompt, consistency_schema(ask), packet.repo_path or None)
        try:
            return parse_consistency(data, unresolved_ids=list(packet.trigger.get("unresolved") or []),
                                     allow_ask_user=ask)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid consistency review: {exc}") from exc

    async def mid_phase_diagnose(self, packet: MidPhasePacket) -> MidPhaseDiagnosis:
        prompt, stats = build_mid_phase_prompt(packet, MidPhaseLimits.from_settings(self.settings))
        self.log("SYSTEM", f"Claude MID_PHASE_DIAGNOSE context: prompt {stats['prompt_bytes']}/{stats['limit_bytes']} "
                           "bytes")
        data = await self._invoke("MID_PHASE_DIAGNOSE", prompt, MID_PHASE_JSON_SCHEMA, packet.repo_path or None)
        try:
            return MidPhaseDiagnosis.from_dict(data)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid mid-phase diagnosis: {exc}") from exc

    async def replan_phases(self, request: PhaseReplanRequest) -> list[Phase]:
        limits = limits_for("REPLAN_V2", self.settings)
        data = await self._invoke("REPLAN_V2", build_phase_replan_prompt(request, limits), PHASE_REPLAN_JSON_SCHEMA,
                                  request.repo_path)
        try:
            return parse_phases(data.get("phases") if isinstance(data, dict) else None)
        except PlanValidationError as exc:
            raise OrchestratorError(f"Claude returned an invalid revised phase plan: {exc}") from exc

    async def diagnose_blocker(self, packet: DiagnosisPacket) -> Diagnosis:
        prompt, stats = build_diagnosis_prompt(packet, DiagnosisLimits.from_settings(self.settings))
        self.log("SYSTEM", f"Claude DIAGNOSE_BLOCKER context: prompt {stats['prompt_bytes']}/{stats['limit_bytes']} bytes "
                           f"(repair diff {stats['diff_original_bytes']} bytes before truncation)")
        data = await self._invoke("DIAGNOSE_BLOCKER", prompt, DIAGNOSIS_JSON_SCHEMA, packet.repo_path or None)
        try:
            return Diagnosis.from_dict(data)
        except ValueError as exc:
            raise OrchestratorError(f"Claude returned an invalid diagnosis: {exc}") from exc

    async def smoke_test(self) -> SmokeResult:
        """Tiny structured JSON round trip through the Claude Code CLI (no tools, no retry)."""
        schema = {"type": "object",
                  "properties": {"ok": {"type": "boolean"}, "role": {"type": "string"}},
                  "required": ["ok", "role"]}
        try:
            caps = await self.detect_capabilities()
            if caps["missing"]:
                return SmokeResult(False, self.settings.executable,
                                   "missing flags: " + ", ".join(caps["missing"]))
            data = await self._invoke(
                "SMOKE", 'Smoke test. Do not use any tools. Respond with ok=true and role="orchestrator".',
                schema, None)
        except Exception as exc:
            return SmokeResult(False, self.settings.executable, str(exc))
        ok = data.get("ok") is True and data.get("role") == "orchestrator"
        call = self.last_call
        return SmokeResult(ok, self.executable,
                           f"structured response {json.dumps(data)}" if ok else f"unexpected response {data}",
                           model=call.model if call else self.settings.model,
                           extra={"version": caps["version"], "cost_usd": call.cost_usd if call else None,
                                  "session_id": call.session_id if call else "",
                                  "elapsed": round(call.duration, 1) if call else None,
                                  "usage": call.usage if call else {},
                                  "num_turns": call.num_turns if call else None})


# ------------------------------------------------------------ prompts

def _tool_budget_text(limits: CallLimits) -> str:
    if not limits.tools:
        return "You have no tools in this session; answer from the information given."
    return (f"Tool budget: at most {limits.max_tool_calls} tool calls and {limits.max_turns} turns in total. "
            "The session is terminated if you exceed either limit, so answer before reaching them.")


def build_plan_prompt(req: PlanRequest, limits: CallLimits) -> str:
    return f"""Create the implementation plan for this development request.

REPOSITORY: {req.repo_path}
{req.repo_summary}

USER REQUEST:
{req.goal}

Repository inspection (planning is architectural, not implementation):
- Inspect only the minimum repository context needed to create the plan.
- Do not recursively enumerate the entire repository.
- Prefer README, top-level project structure and directly relevant files.
- Stop exploring as soon as enough information exists to produce the plan.
- {_tool_budget_text(limits)}
- Do not modify anything.

Rules for the plan:
- Split the work into small, focused, sequential tasks. Each task must be completable by ONE local
  coding model in one focused session, touching a limited set of files.
- Tasks are executed strictly one at a time; `dependencies` only define ordering.
- Task ids: T001, T002, ... in execution order.
- Each task needs a precise description, concrete `instructions` for the worker (what to change and
  how, relevant conventions), and objectively checkable `acceptance_criteria`.
- `verification_commands`: optional non-destructive commands run from the repository root on
  Windows (cmd.exe) after the task, e.g. a build or test command. Never destructive commands
  (no deletes, git reset/clean/push). Leave empty when there is no fast meaningful check.
- Put cross-cutting facts future tasks need into `architecture_notes`.
- State assumptions explicitly in `assumptions`."""


def build_replan_prompt(req: ReplanRequest, limits: CallLimits) -> str:
    plan = req.plan.to_dict()
    for t in plan["tasks"]:
        t.pop("result", None)
        for k in ("started_at", "finished_at", "position", "extra_attempts"):
            t.pop(k, None)
    return f"""The current plan needs revision.

REASON: {req.reason}

CURRENT PLAN (with statuses):
{json.dumps(plan, indent=1, ensure_ascii=False)}

COMPLETED TASK RESULTS:
{json.dumps([r.to_dict() for r in req.completed_results], indent=1, ensure_ascii=False)}

Inspect the repository only if the reason above cannot be addressed without it.
{_tool_budget_text(limits)}

Return the full revised plan. Tasks with status PASSED must be kept unchanged with the same ids.
You may modify, add or remove tasks that have not passed. Keep tasks small and sequential."""


_PHASE_RULES = """Rules for the plan:
- Plan 3-6 coherent PHASES (fewer only for a genuinely small request; never more than 8). Each phase is ONE
  coherent engineering objective representing roughly 20-60 minutes of local-model work.
- Avoid both extremes: no single project-long phase, and no microscopic phases. Keep closely related changes
  in the same phase so the worker keeps its context; put unrelated subsystems into different phases.
- Each phase needs:
  - `description`: the objective; `instructions`: concrete implementation guidance and relevant conventions
  - `subtasks`: 3-8 concrete implementation steps (ids S1, S2, ...). The worker tracks them itself; they are
    NOT reviewed individually, so do not make them tiny.
  - `acceptance_criteria`: objectively checkable criteria for the whole phase, reviewed at the checkpoint
  - `files_likely_involved`, `integration_expectations` (how the phase must fit earlier phases and the
    existing code) and `estimated_minutes`
  - `verification_commands`: optional non-destructive commands run from the repository root on Windows
    (cmd.exe) after the phase, e.g. a build or test command. Never destructive commands (no deletes, git
    reset/clean/push). Leave empty when there is no fast meaningful check.
- Order phases so each builds on the previous ones; the last phase must leave the product integrated and
  verifiable end to end (it is also the final audit).
- Where subjective quality matters for this request (UX, visuals, feel, wording, docs), include explicit
  integration/polish acceptance criteria in the relevant phases.
- `verification_contract` (recommended for every phase): WHAT evidence would convincingly show that the phase
  succeeded - not how a tool obtains it. `success_dimensions` (semantic, e.g. functional, runtime, visual,
  interaction, performance, integration); `evidence_requirements` ({id E1.., goal, kind, required, reason}; the kind
  is semantic, e.g. build, automated_test, runtime_behavior, visual_evidence, interaction, performance,
  data_integrity, human_judgment - say why static tests are insufficient where that matters); and `human_gate`
  ({allowed, reason, checklist}). Open the human gate (allowed=true) only where a person's observation of the running
  result can materially decide acceptance (appearance, motion, feel, wording, real-world behaviour) and automated
  evidence cannot reliably settle it; purely technical phases keep allowed=false.
- Phase ids: P1, P2, ... in execution order.
- Put cross-cutting facts later phases need into `architecture_notes`; state assumptions in `assumptions`."""


def build_phase_plan_prompt(req: PlanRequest, limits: CallLimits) -> str:
    return f"""Create a PHASE-BASED implementation plan for this development request.

REPOSITORY: {req.repo_path}
{req.repo_summary}

USER REQUEST:
{req.goal}

How the plan is executed:
- ONE local coding model (Qwen via Codex) implements each phase in a single long-lived session and manages
  the phase's subtasks with its own todo list.
- You review only at phase boundaries (checkpoints): authored changes, acceptance criteria, integration,
  verification and regressions. Phases run strictly in order.

Repository inspection (planning is architectural, not implementation):
- Inspect only the minimum repository context needed to create the plan.
- Do not recursively enumerate the entire repository.
- Prefer README, top-level project structure and directly relevant files.
- Stop exploring as soon as enough information exists to produce the plan.
- {_tool_budget_text(limits)}
- Do not modify anything.

{_PHASE_RULES}"""


def _phase_brief(p: Phase) -> dict[str, Any]:
    d: dict[str, Any] = {"id": p.id, "title": p.title, "status": p.status.value, "description": p.description,
                         "subtasks": [f"{s.id}: {s.title}" for s in p.subtasks],
                         "acceptance_criteria": p.acceptance_criteria}
    if p.result is not None:
        d["result"] = p.result.result
        d["files_changed"] = p.result.files_changed[:25]
        d["important_decisions"] = p.result.important_decisions[:6]
    if p.human_criteria:  # V4: requirements the user's verification revealed (carry them into a re-scoped phase)
        d["human_derived_criteria"] = criteria_lines(p.human_criteria)
    if p.verification_contract is not None and p.status not in TERMINAL_PHASE_STATUSES:
        d["verification_contract"] = p.verification_contract.to_dict()
    return d


def build_phase_replan_prompt(req: PhaseReplanRequest, limits: CallLimits) -> str:
    phases = [p for p in req.plan.tasks if isinstance(p, Phase)]
    history = [_phase_brief(p) for p in phases if p.status in TERMINAL_PHASE_STATUSES]
    remaining = [_phase_brief(p) for p in phases if p.status not in TERMINAL_PHASE_STATUSES]
    used = ", ".join(p.id for p in phases) or "(none)"
    superseded = ""
    if req.superseded_phase_id:
        superseded = (f"\nThe checkpoint of phase {req.superseded_phase_id} decided REPLAN. That phase is now closed as "
                      "SUPERSEDED; its changes stay in the repository.\nCheckpoint summary: "
                      f"{req.checkpoint_summary}\nCheckpoint problems: {json.dumps(req.checkpoint_problems, ensure_ascii=False)}\n")
    guidance = f"\nUSER GUIDANCE FOR THE REPLAN:\n{req.user_guidance}\n" if req.user_guidance else ""
    return f"""The remaining phase plan needs revision, based on the ACTUAL state of the implementation.

OVERALL GOAL:
{req.goal}

REASON: {req.reason}
{superseded}{guidance}
COMPLETED / CLOSED PHASES (history - never changed, never repeated):
{json.dumps(history, indent=1, ensure_ascii=False)}

REMAINING PHASES AS CURRENTLY PLANNED:
{json.dumps(remaining, indent=1, ensure_ascii=False)}

Inspect the repository only as far as needed to understand the current implementation.
{_tool_budget_text(limits)}

Return ONLY the revised REMAINING phases (at least one), in execution order. Do not include completed or
closed phases. Ids already used in this run: {used}. Reuse an id only for a remaining phase you keep; give new
phases new ids (e.g. P4, P5).

{_PHASE_RULES}"""


# ------------------------------------------------------------ helpers

def _short(value: Any, n: int = 160) -> str:
    s = json.dumps(value, ensure_ascii=False) if not isinstance(value, str) else value
    return s if len(s) <= n else s[:n] + "..."


def _redact_argv(argv: list[str]) -> list[str]:
    """Keep argv readable in logs/DB: the schema JSON is large but not secret."""
    out = []
    for a in argv:
        out.append(a if len(a) < 300 else a[:120] + f"...[{len(a)} chars]")
    return out


def _parse_json_text(text: str) -> Any:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        if text.lower().startswith("json"):
            text = text[4:]
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        return json.loads(text[start:end + 1])
    except json.JSONDecodeError:
        return None
