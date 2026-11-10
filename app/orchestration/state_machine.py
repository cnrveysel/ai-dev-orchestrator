"""Explicit run-level state machine. Every transition is validated and logged."""
from __future__ import annotations

from enum import Enum
from typing import Callable


class RunState(str, Enum):
    IDLE = "IDLE"
    PLANNING = "PLANNING"
    PLAN_READY = "PLAN_READY"
    STARTING_TASK = "STARTING_TASK"
    WORKER_RUNNING = "WORKER_RUNNING"
    VERIFYING = "VERIFYING"
    # V4 pass 2 (phase pipeline) only: discovery, Claude EVIDENCE_PLAN and bounded evidence execution for a
    # phase whose verification contract lists evidence requirements. Verification work, never implementation.
    COLLECTING_EVIDENCE = "COLLECTING_EVIDENCE"
    REVIEWING = "REVIEWING"
    # V4 pass 3 (phase pipeline) only: the checkpoint's structured answer contradicted itself (e.g. PASS with
    # evidence judged insufficient while required evidence is unresolved): one bounded, tool-free Claude
    # CONSISTENCY_REVIEW decides PASS / RETRY / BLOCKED / ASK_USER. Never recursive.
    CONSISTENCY_REVIEW = "CONSISTENCY_REVIEW"
    RETRY_PENDING = "RETRY_PENDING"
    ADVANCING = "ADVANCING"
    REPLANNING = "REPLANNING"  # V2 only: Claude revises the FUTURE phases
    REPAIRING = "REPAIRING"  # V2 only: bounded repair after a checkpoint RETRY
    # V4 (phase pipeline) only: the checkpoint asked the user to verify the phase. The run waits here -
    # across restarts - without running any worker; the user's answer is adjudicated by Claude
    # HUMAN_FOLLOWUP in this state, which then continues to ADVANCING / RETRY_PENDING / BLOCKED.
    AWAITING_HUMAN = "AWAITING_HUMAN"
    BLOCKED = "BLOCKED"
    PAUSED = "PAUSED"
    COMPLETED = "COMPLETED"
    STOPPING = "STOPPING"
    STOPPED = "STOPPED"
    ERROR = "ERROR"


S = RunState

# States the execution loop dispatches on; also the valid resume targets.
EXECUTION_STATES = frozenset({
    S.STARTING_TASK, S.WORKER_RUNNING, S.VERIFYING, S.REVIEWING, S.RETRY_PENDING, S.ADVANCING,
    S.REPLANNING, S.REPAIRING, S.COLLECTING_EVIDENCE, S.CONSISTENCY_REVIEW,
})
# States in which the run is halted and waiting for the user.
HALTED_STATES = frozenset({S.PAUSED, S.BLOCKED, S.STOPPED, S.ERROR})
# V4: waiting for a human answer. Neither executing (another run may be planned/loaded, and a
# restart leaves it untouched) nor halted (RESUME alone does not continue it; an answer does).
WAITING_STATES = frozenset({S.AWAITING_HUMAN})
# Interrupt targets reachable from any execution state.
_INTERRUPTS = {S.PAUSED, S.STOPPING, S.ERROR, S.BLOCKED}
_RESUMABLE = EXECUTION_STATES | WAITING_STATES

TRANSITIONS: dict[RunState, frozenset[RunState]] = {
    S.IDLE: frozenset({S.PLANNING}),
    S.PLANNING: frozenset({S.PLAN_READY, S.ERROR, S.STOPPED}),
    S.PLAN_READY: frozenset({S.PLANNING, S.STARTING_TASK, S.STOPPED}),
    S.STARTING_TASK: frozenset({S.WORKER_RUNNING, S.COMPLETED} | _INTERRUPTS),
    S.WORKER_RUNNING: frozenset({S.VERIFYING} | _INTERRUPTS),
    S.VERIFYING: frozenset({S.REVIEWING, S.COLLECTING_EVIDENCE} | _INTERRUPTS),
    S.COLLECTING_EVIDENCE: frozenset({S.REVIEWING} | _INTERRUPTS),
    S.REVIEWING: frozenset({S.ADVANCING, S.RETRY_PENDING, S.REPLANNING, S.AWAITING_HUMAN, S.CONSISTENCY_REVIEW}
                           | _INTERRUPTS),
    S.CONSISTENCY_REVIEW: frozenset({S.ADVANCING, S.RETRY_PENDING, S.AWAITING_HUMAN} | _INTERRUPTS),
    S.RETRY_PENDING: frozenset({S.WORKER_RUNNING, S.REPAIRING} | _INTERRUPTS),
    S.REPAIRING: frozenset({S.VERIFYING} | _INTERRUPTS),
    S.ADVANCING: frozenset({S.STARTING_TASK, S.REPLANNING} | _INTERRUPTS),
    S.REPLANNING: frozenset({S.STARTING_TASK} | _INTERRUPTS),
    # HUMAN_FOLLOWUP outcome: PASS / RETRY (existing repair pipeline) / BLOCKED; STOPPED = abandoned by the user.
    S.AWAITING_HUMAN: frozenset({S.ADVANCING, S.RETRY_PENDING, S.STOPPED} | _INTERRUPTS),
    S.STOPPING: frozenset({S.STOPPED}),
    S.PAUSED: _RESUMABLE | {S.STOPPED, S.PLANNING},
    S.BLOCKED: _RESUMABLE | {S.STOPPED, S.PLANNING},
    S.STOPPED: _RESUMABLE | {S.PLANNING},
    S.ERROR: _RESUMABLE | {S.PLANNING, S.STOPPED},
    S.COMPLETED: frozenset({S.PLANNING}),
}


class IllegalTransition(RuntimeError):
    pass


class StateMachine:
    def __init__(self, initial: RunState = S.IDLE,
                 on_transition: Callable[[RunState, RunState, str], None] | None = None):
        self._state = initial
        self._on_transition = on_transition
        self.history: list[tuple[RunState, RunState, str]] = []

    @property
    def state(self) -> RunState:
        return self._state

    def can(self, new: RunState) -> bool:
        return new in TRANSITIONS.get(self._state, frozenset())

    def transition(self, new: RunState, reason: str = "") -> None:
        old = self._state
        if not self.can(new):
            raise IllegalTransition(f"{old.value} -> {new.value} is not allowed ({reason})")
        self._state = new
        self.history.append((old, new, reason))
        if self._on_transition:
            self._on_transition(old, new, reason)

    def force(self, new: RunState, reason: str = "") -> None:
        """Used only when restoring persisted state at startup."""
        old = self._state
        self._state = new
        self.history.append((old, new, "restore: " + reason))
