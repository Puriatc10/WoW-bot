"""Feedback intake layer and state transition decision logic.

This module provides typed feedback representation, classification from action
results and reflex signals, and pure decision logic mapping feedback to proposed
FSM state transitions.

Callers MUST NOT mutate the `payload` dict of a `Feedback` instance after construction.
"""

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from wow_bot.actuation.mapper import ActionResult, ActionStatus
from wow_bot.executor.states import FSMState, can_transition


class FeedbackKind(str, Enum):
    """Enumeration of all supported feedback signal sources and events."""

    ACTION_SUCCESS = "action_success"
    ACTION_FAILED = "action_failed"
    ACTION_TIMEOUT = "action_timeout"
    FOCUS_LOST = "focus_lost"
    POSITION_STUCK = "position_stuck"
    POSITION_CLEAR = "position_clear"
    TARGET_LOST = "target_lost"
    SESSION_TIMEOUT = "session_timeout"
    SAFETY_ABORT = "safety_abort"
    KILL_SWITCH = "kill_switch"


@dataclass(frozen=True)
class Feedback:
    """Typed feedback snapshot emitted by action execution or reflex signals."""

    kind: FeedbackKind
    ts: float
    reason: str = ""
    payload: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str):
            raise ValueError(f"reason must be a str, got {type(self.reason).__name__}")  # noqa: TRY004
        if not isinstance(self.payload, dict):
            raise ValueError(f"payload must be a dict, got {type(self.payload).__name__}")  # noqa: TRY004
        if not math.isfinite(self.ts) or self.ts < 0.0:
            raise ValueError(f"ts must be finite and >= 0.0, got {self.ts}")


def classify_action_result(
    result: ActionResult,
    *,
    ts: float,
    reason: str = "",
) -> Feedback:
    """Classify an ActionResult into a typed Feedback object."""
    if result.status == ActionStatus.SUCCESS:
        kind = FeedbackKind.ACTION_SUCCESS
    elif result.status == ActionStatus.FAILED:
        kind = FeedbackKind.ACTION_FAILED
    elif result.status == ActionStatus.TIMEOUT:
        kind = FeedbackKind.ACTION_TIMEOUT
    else:
        raise ValueError(f"Unknown ActionResult status: {result.status}")

    payload = {
        "latency_ms": result.latency_ms,
        "notes": result.notes,
    }
    return Feedback(
        kind=kind,
        ts=ts,
        reason=reason,
        payload=payload,
    )


_REFLEX_SIGNAL_MAP: dict[str, FeedbackKind] = {
    "focus_lost": FeedbackKind.FOCUS_LOST,
    "position_stuck": FeedbackKind.POSITION_STUCK,
    "position_clear": FeedbackKind.POSITION_CLEAR,
    "session_timeout": FeedbackKind.SESSION_TIMEOUT,
    "safety_abort": FeedbackKind.SAFETY_ABORT,
    "kill_switch": FeedbackKind.KILL_SWITCH,
}


def classify_reflex_signal(
    name: str,
    *,
    ts: float,
    payload: dict[str, Any] | None = None,
) -> Feedback:
    """Classify a reflex signal by name into a typed Feedback object."""
    kind = _REFLEX_SIGNAL_MAP.get(name)
    if kind is None:
        raise ValueError(f"Unknown reflex signal name: {name!r}")

    payload_copy = dict(payload) if payload is not None else {}
    reason_val = payload_copy.get("reason")
    reason = reason_val if isinstance(reason_val, str) else ""

    return Feedback(
        kind=kind,
        ts=ts,
        reason=reason,
        payload=payload_copy,
    )


class FeedbackOutcome(str, Enum):
    """Possible outcomes of evaluating feedback against current state."""

    STAY = "stay"
    TO_IDLE = "to_idle"
    TO_SCANNING = "to_scanning"
    TO_RECOVERING = "to_recovering"
    TO_STUCK_RECOVERY = "to_stuck_recovery"
    TO_PAUSED = "to_paused"


@dataclass(frozen=True)
class FeedbackDecision:
    """Transition decision returned by decide()."""

    outcome: FeedbackOutcome
    next_state: FSMState | None
    reason: str = ""

    def __post_init__(self) -> None:
        if self.outcome == FeedbackOutcome.STAY and self.next_state is not None:
            raise ValueError("outcome == STAY implies next_state must be None")
        if self.outcome != FeedbackOutcome.STAY and self.next_state is None:
            raise ValueError("outcome != STAY implies next_state must not be None")


def decide(
    current: FSMState,
    feedback: Feedback,
) -> FeedbackDecision:
    """Pure decision logic mapping current FSMState and Feedback to FeedbackDecision."""
    candidate_outcome: FeedbackOutcome
    candidate_next: FSMState | None
    candidate_reason: str

    # Priority 1 — KILL_SWITCH or SAFETY_ABORT
    if feedback.kind in (FeedbackKind.KILL_SWITCH, FeedbackKind.SAFETY_ABORT):
        candidate_outcome = FeedbackOutcome.TO_PAUSED
        candidate_next = FSMState.PAUSED
        candidate_reason = feedback.kind.value

    # Priority 2 — SESSION_TIMEOUT
    elif feedback.kind == FeedbackKind.SESSION_TIMEOUT:
        candidate_outcome = FeedbackOutcome.TO_PAUSED
        candidate_next = FSMState.PAUSED
        candidate_reason = "session_timeout"

    # Priority 3 — FOCUS_LOST
    elif feedback.kind == FeedbackKind.FOCUS_LOST:
        candidate_outcome = FeedbackOutcome.TO_PAUSED
        candidate_next = FSMState.PAUSED
        candidate_reason = "focus_lost"

    # Priority 4 — POSITION_STUCK
    elif feedback.kind == FeedbackKind.POSITION_STUCK:
        if current == FSMState.STUCK_RECOVERY:
            return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")
        candidate_outcome = FeedbackOutcome.TO_STUCK_RECOVERY
        candidate_next = FSMState.STUCK_RECOVERY
        candidate_reason = "position_stuck"

    # Priority 5 — POSITION_CLEAR
    elif feedback.kind == FeedbackKind.POSITION_CLEAR:
        if current == FSMState.STUCK_RECOVERY:
            candidate_outcome = FeedbackOutcome.TO_RECOVERING
            candidate_next = FSMState.RECOVERING
            candidate_reason = "position_clear"
        else:
            return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")

    # Priority 6 — TARGET_LOST
    elif feedback.kind == FeedbackKind.TARGET_LOST:
        if current in (FSMState.TARGETING, FSMState.COMBAT):
            candidate_outcome = FeedbackOutcome.TO_RECOVERING
            candidate_next = FSMState.RECOVERING
            candidate_reason = "target_lost"
        else:
            return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")

    # Priority 7 — ACTION_TIMEOUT
    elif feedback.kind == FeedbackKind.ACTION_TIMEOUT:
        if current == FSMState.STUCK_RECOVERY:
            return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")
        candidate_outcome = FeedbackOutcome.TO_STUCK_RECOVERY
        candidate_next = FSMState.STUCK_RECOVERY
        candidate_reason = "action_timeout"

    # Priority 8 — ACTION_FAILED
    elif feedback.kind == FeedbackKind.ACTION_FAILED:
        if current == FSMState.RECOVERING:
            candidate_outcome = FeedbackOutcome.TO_IDLE
            candidate_next = FSMState.IDLE
            candidate_reason = "repeated_failure"
        else:
            candidate_outcome = FeedbackOutcome.TO_RECOVERING
            candidate_next = FSMState.RECOVERING
            candidate_reason = "action_failed"

    # Priority 9 — ACTION_SUCCESS
    elif feedback.kind == FeedbackKind.ACTION_SUCCESS:
        return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")

    else:
        # Fallback for unexpected enum value if any
        return FeedbackDecision(outcome=FeedbackOutcome.STAY, next_state=None, reason="")

    # Cross-check after choosing an outcome (except STAY which returned early)
    if candidate_next is not None and not can_transition(current, candidate_next):
        return FeedbackDecision(
            outcome=FeedbackOutcome.STAY,
            next_state=None,
            reason=f"blocked:{candidate_reason}",
        )

    return FeedbackDecision(
        outcome=candidate_outcome,
        next_state=candidate_next,
        reason=candidate_reason,
    )
