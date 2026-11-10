"""V4 pass 3: the NORMAL implementation guard (WORKER_RUNNING) and Claude's MID_PHASE_DIAGNOSE answer.

A normal implementation session (before any checkpoint) is bounded like a repair session:
    wall-clock time, worker tool calls, and time WITHOUT planned-subtask progress
Progress means structured progress only: a planned subtask (S1, S2, ...) newly reported done through the
worker's todo list. Helper todo entries, prose and verbose output never count.

When a limit is hit the turn is stopped, Claude diagnoses the blocker (MID_PHASE_DIAGNOSE: no tools,
short bounded prompt) and a FRESH worker session continues with that guidance and a fresh budget. After
the maximum number of diagnoses, another trip BLOCKS the run instead of looping.

The state lives in run.pipeline_state["impl_guard"][<phase id>] (plain JSON; in-flight usage is persisted
at every tool call, so a crash never refunds a budget).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

MAX_DIAGNOSES_KEPT = 6
MAX_TRIPS_KEPT = 10
MID_PHASE_STATUSES = ("CONTINUE", "BLOCKED")

_STR_LIST = {"type": "array", "items": {"type": "string"}}

MID_PHASE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "status": {"type": "string", "enum": list(MID_PHASE_STATUSES)},
        "blocker": {"type": "string", "description": "What the worker is stuck on, grounded in the evidence."},
        "diagnosis": {"type": "string"},
        "next_actions": {**_STR_LIST, "description": "Precise, ordered next actions for a fresh worker session."},
        "avoid": {**_STR_LIST, "description": "Approaches that already failed or waste the budget."},
        "target_files": _STR_LIST,
        "target_checks": {**_STR_LIST, "description": "Fast checks to run after each change."},
        "blocked_reason": {"type": "string", "description": "Only with BLOCKED: what the user must decide or fix."},
    },
    "required": ["status", "blocker", "diagnosis", "next_actions", "avoid"],
}


def _strs(v: Any, limit: int = 12) -> list[str]:
    return [str(x)[:500] for x in v][:limit] if isinstance(v, list) else []


@dataclass
class MidPhaseDiagnosis:
    status: str  # CONTINUE | BLOCKED
    blocker: str = ""
    diagnosis: str = ""
    next_actions: list[str] = field(default_factory=list)
    avoid: list[str] = field(default_factory=list)
    target_files: list[str] = field(default_factory=list)
    target_checks: list[str] = field(default_factory=list)
    blocked_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "MidPhaseDiagnosis":
        if not isinstance(d, dict):
            raise ValueError("diagnosis must be a JSON object")
        status = str(d.get("status") or "").strip().upper()
        if status not in MID_PHASE_STATUSES:
            raise ValueError(f"invalid MID_PHASE_DIAGNOSE status {status!r}")
        return cls(status, str(d.get("blocker") or "")[:1000], str(d.get("diagnosis") or "")[:2000],
                   _strs(d.get("next_actions")), _strs(d.get("avoid")), _strs(d.get("target_files"), 30),
                   _strs(d.get("target_checks")), str(d.get("blocked_reason") or "")[:1000])

    def guidance_lines(self) -> list[str]:
        lines = [f"- blocker: {self.blocker}"] if self.blocker else []
        lines += [f"- diagnosis: {self.diagnosis}"] if self.diagnosis else []
        lines += [f"- do: {x}" for x in self.next_actions]
        lines += [f"- avoid: {x}" for x in self.avoid]
        if self.target_files:
            lines.append("- target files: " + ", ".join(self.target_files[:15]))
        if self.target_checks:
            lines.append("- checks after each change: " + "; ".join(self.target_checks[:6]))
        return lines


@dataclass
class ImplGuardState:
    phase_id: str
    window: int = 1  # budget window: a new one after a diagnosis, a user resume or a new checkpoint cycle
    wall_used: float = 0.0  # seconds of implementation turns in this window
    tool_calls: int = 0
    progress_wall: float = 0.0  # wall_used when planned-subtask progress was last observed (window start = 0)
    progress_subtasks: list[str] = field(default_factory=list)  # planned subtasks done at that moment
    inflight_tool_calls: int = 0  # usage of the turn in flight (persisted at every tool call)
    inflight_wall: float = 0.0
    trip: str = ""  # why the guard stopped the current turn
    pending: str = ""  # a trip waiting for MID_PHASE_DIAGNOSE (survives a restart)
    diagnoses_used: int = 0  # MID_PHASE_DIAGNOSE calls in this work cycle
    diagnoses: list[dict[str, Any]] = field(default_factory=list)
    guidance: dict[str, Any] = field(default_factory=dict)  # the diagnosis the next fresh session receives
    status: str = "working"  # working | blocked
    trips: list[dict[str, Any]] = field(default_factory=list)  # bounded history
    cycle_attempt: int = 0  # the reviewed attempt that started this work cycle (0 = the first cycle)

    def fold_inflight(self) -> None:
        """A turn killed by a crash still counts (its usage was persisted at every tool call)."""
        self.tool_calls += self.inflight_tool_calls
        self.wall_used += self.inflight_wall
        self.inflight_tool_calls, self.inflight_wall = 0, 0.0

    def new_window(self) -> None:
        self.window += 1
        self.wall_used, self.tool_calls, self.progress_wall, self.trip = 0.0, 0, 0.0, ""
        self.inflight_tool_calls, self.inflight_wall = 0, 0.0

    def add_trip(self, rec: dict[str, Any]) -> None:
        self.trips = (self.trips + [rec])[-MAX_TRIPS_KEPT:]

    def add_diagnosis(self, rec: dict[str, Any]) -> None:
        self.diagnoses = (self.diagnoses + [rec])[-MAX_DIAGNOSES_KEPT:]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: Any) -> "ImplGuardState | None":
        if not isinstance(d, dict) or not d.get("phase_id"):
            return None
        return cls(phase_id=str(d["phase_id"]), window=int(d.get("window") or 1),
                   wall_used=float(d.get("wall_used") or 0.0), tool_calls=int(d.get("tool_calls") or 0),
                   progress_wall=float(d.get("progress_wall") or 0.0),
                   progress_subtasks=[str(x) for x in d.get("progress_subtasks") or []],
                   inflight_tool_calls=int(d.get("inflight_tool_calls") or 0),
                   inflight_wall=float(d.get("inflight_wall") or 0.0), trip=str(d.get("trip") or ""),
                   pending=str(d.get("pending") or ""), diagnoses_used=int(d.get("diagnoses_used") or 0),
                   diagnoses=[x for x in d.get("diagnoses") or [] if isinstance(x, dict)][-MAX_DIAGNOSES_KEPT:],
                   guidance=dict(d.get("guidance") or {}) if isinstance(d.get("guidance"), dict) else {},
                   status=str(d.get("status") or "working"),
                   trips=[x for x in d.get("trips") or [] if isinstance(x, dict)][-MAX_TRIPS_KEPT:],
                   cycle_attempt=int(d.get("cycle_attempt") or 0))


def _limit(o: Any, name: str, default: float) -> float:
    try:
        return float(getattr(o, name, default))
    except (TypeError, ValueError):
        return float(default)


def guard_limits(o: Any) -> tuple[float, int, float, int]:
    """(wall seconds, tool calls, no-progress seconds, diagnoses); a limit <= 0 is disabled."""
    return (_limit(o, "implementation_max_wall_seconds", 2400), int(_limit(o, "implementation_max_tool_calls", 50)),
            _limit(o, "implementation_max_no_subtask_progress_seconds", 1200),
            int(_limit(o, "implementation_max_diagnoses", 2)))


def impl_guard_trip(g: ImplGuardState, o: Any, extra_wall: float = 0.0, extra_tools: int = 0,
                    caps: Any = None) -> str:
    """Why the implementation session must stop ("" = within all limits). extra_* = the turn in flight.
    V5.1: `caps` = the worker's WorkerCapabilities. A limit whose signal the worker runtime cannot report is not
    applied (never faked): no exact tool calls -> no tool budget; no structured subtask progress -> no no-progress
    limit. None (and the Codex worker) = every limit, exactly as before."""
    wall, tools, nop, _ = guard_limits(o)
    count_tools = getattr(caps, "tool_call_count", True)
    subtask_progress = getattr(caps, "subtask_progress", True)
    used = g.wall_used + extra_wall
    if wall > 0 and used >= wall:
        return f"wall-clock budget reached ({used:.0f}s/{wall:.0f}s)"
    if count_tools and tools > 0 and g.tool_calls + extra_tools >= tools:
        return f"tool budget reached {g.tool_calls + extra_tools}/{tools}"
    if subtask_progress and nop > 0 and used - g.progress_wall >= nop:
        return f"no planned subtask progress for {used - g.progress_wall:.0f}s (max {nop:.0f}s)"
    return ""
