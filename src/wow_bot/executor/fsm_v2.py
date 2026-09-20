"""FSM v2 engine implementation.

Provides state storage, tick loop execution, feedback processing, manual controls
(pause/resume/enter_recovery), and structured session event logging.
"""

import random
import threading
from dataclasses import dataclass
from typing import Any, Protocol

from wow_bot.actuation.mapper import Intent, MoveTo, Turn
from wow_bot.executor.feedback import Feedback, FeedbackDecision, FeedbackOutcome, decide
from wow_bot.executor.states import (
    FSMState,
    assert_transition,
    can_transition,
    timeout_for,
)
from wow_bot.session import Session


class FSMError(Exception):
    """Raised when an invalid state transition or operation occurs in the FSM engine."""


class GameStateLike(Protocol):
    """Marker protocol. The FSM does not read fields directly in
    this phase. Concrete behaviors in Phase 5+ will extend this
    protocol with specific fields."""


class MetaStateLike(Protocol):
    """Marker protocol. The FSM does not read fields directly in
    this phase. Concrete behaviors in Phase 5+ will extend this
    protocol with specific fields."""


@dataclass(frozen=False)
class FSMConfig:
    """Configuration options for the FSM v2 engine."""

    seed: int = 0
    stuck_recovery_max_attempts: int = 3
    hard_stuck_pause: bool = True

    def __post_init__(self) -> None:
        if self.stuck_recovery_max_attempts < 1:
            raise ValueError(
                f"stuck_recovery_max_attempts must be >= 1, got {self.stuck_recovery_max_attempts}"
            )


class Behavior(Protocol):
    """Protocol for FSM state behavior decision handlers."""

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        """Decide an intent given current state, game state, meta state, timestamp, and per-tick RNG."""
        ...


class NullBehavior:
    """Behavior implementation that always returns None."""

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        """Always return None."""
        return None


class ConstantTargetBehavior:
    """Behavior implementation emitting target movement intents or recovery turns."""

    def __init__(self, target: tuple[float, float]) -> None:
        self.target = target

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        """Return MoveTo for movement states, Turn for STUCK_RECOVERY, or None."""
        if state in (FSMState.MOVING_TO_TARGET, FSMState.PATHING, FSMState.FLEEING):
            return MoveTo(x=self.target[0], y=self.target[1])
        if state == FSMState.STUCK_RECOVERY and rng.random() < 0.5:
            return Turn(angle_rad=0.0)
        return None


class FSM:
    """Finite State Machine execution loop engine v2."""

    def __init__(
        self,
        *,
        config: FSMConfig | None = None,
        behavior: Behavior | None = None,
        session: Session | None = None,
    ) -> None:
        self._config = config if config is not None else FSMConfig()
        self._behavior = behavior if behavior is not None else NullBehavior()
        self._session = session

        self._state: FSMState = FSMState.IDLE
        self._previous_state: FSMState | None = None
        self._state_entered_at: float | None = None
        self._last_decision: FeedbackDecision | None = None
        self._stuck_attempts: int = 0

        self._lock = threading.Lock()
        self._rng = random.Random(self._config.seed)

    @property
    def current_state(self) -> FSMState:
        """Return current FSMState."""
        with self._lock:
            return self._state

    @property
    def previous_state(self) -> FSMState | None:
        """Return previous FSMState if any."""
        with self._lock:
            return self._previous_state

    @property
    def is_paused(self) -> bool:
        """Return True if current state is PAUSED."""
        with self._lock:
            return self._state == FSMState.PAUSED

    @property
    def last_decision(self) -> FeedbackDecision | None:
        """Return the most recent FeedbackDecision."""
        with self._lock:
            return self._last_decision

    @property
    def stuck_attempts(self) -> int:
        """Return the current consecutive stuck recovery attempt counter."""
        with self._lock:
            return self._stuck_attempts

    def _emit_event(self, event_type: str, payload: dict[str, Any]) -> None:
        """Emit a structured event to session logs if a session is attached."""
        if self._session is not None:
            event_data: dict[str, Any] = {"event": event_type}
            event_data.update(payload)
            self._session.write_event(event_data)

    def tick(
        self,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        *,
        now: float,
    ) -> Intent | None:
        """Execute one tick step of the FSM loop."""
        timeout_event: tuple[str, str] | None = None

        with self._lock:
            if self._state_entered_at is None:
                self._state_entered_at = now

            if self._state == FSMState.PAUSED:
                return None

            current_state = self._state
            timeout_s = timeout_for(current_state)

            if timeout_s is not None and (now - self._state_entered_at > timeout_s):
                old_state = current_state
                self._previous_state = old_state
                self._state = FSMState.IDLE
                self._state_entered_at = now
                self._stuck_attempts = 0
                current_state = FSMState.IDLE
                timeout_event = (old_state.value, f"state_timeout:{old_state.value}")

            seed_bits = self._rng.getrandbits(64)

        if timeout_event is not None:
            from_state_val, reason_str = timeout_event
            self._emit_event(
                "fsm_transition",
                {
                    "from": from_state_val,
                    "to": FSMState.IDLE.value,
                    "reason": reason_str,
                },
            )

        tick_rng = random.Random(seed_bits)
        intent = self._behavior.decide(
            current_state,
            game_state,
            meta_state,
            now,
            tick_rng,
        )

        if intent is not None:
            self._emit_event(
                "fsm_intent",
                {
                    "state": current_state.value,
                    "intent": repr(intent),
                },
            )

        return intent

    def submit_feedback(
        self,
        feedback: Feedback,
        *,
        now: float,
    ) -> FeedbackDecision:
        """Submit feedback to evaluate FSM state transitions."""
        transition_event: tuple[str, str, str] | None = None
        hard_stuck_event_attempts: int | None = None

        with self._lock:
            if self._state_entered_at is None:
                self._state_entered_at = now

            current_state = self._state
            decision = decide(current_state, feedback)
            self._last_decision = decision

            if decision.outcome != FeedbackOutcome.STAY:
                next_st = decision.next_state
                assert next_st is not None
                assert_transition(current_state, next_st)

                old_state = current_state
                self._previous_state = current_state
                self._state = next_st
                self._state_entered_at = now

                if next_st == FSMState.STUCK_RECOVERY:
                    self._stuck_attempts += 1
                else:
                    self._stuck_attempts = 0

                transition_event = (old_state.value, next_st.value, decision.reason)

                if (
                    self._stuck_attempts >= self._config.stuck_recovery_max_attempts
                    and self._config.hard_stuck_pause
                ):
                    attempts_count = self._stuck_attempts
                    self._previous_state = self._state
                    self._state = FSMState.PAUSED
                    self._stuck_attempts = 0
                    hard_stuck_event_attempts = attempts_count

        self._emit_event(
            "fsm_feedback",
            {
                "kind": feedback.kind.value,
                "outcome": decision.outcome.value,
                "next_state": decision.next_state.value if decision.next_state else None,
                "reason": decision.reason,
            },
        )

        if transition_event is not None:
            from_val, to_val, reason_val = transition_event
            self._emit_event(
                "fsm_transition",
                {
                    "from": from_val,
                    "to": to_val,
                    "reason": reason_val,
                },
            )

        if hard_stuck_event_attempts is not None:
            self._emit_event(
                "fsm_hard_stuck",
                {
                    "attempts": hard_stuck_event_attempts,
                },
            )

        return decision

    def pause(self, reason: str, *, now: float) -> None:
        """Pause FSM execution."""
        prev_state: FSMState | None = None

        with self._lock:
            if self._state_entered_at is None:
                self._state_entered_at = now

            if self._state == FSMState.PAUSED:
                return

            prev_state = self._state
            self._previous_state = prev_state
            self._state = FSMState.PAUSED
            self._state_entered_at = now

        self._emit_event(
            "fsm_paused",
            {
                "reason": reason,
                "from": prev_state.value,
            },
        )

    def resume(self, reason: str, *, now: float) -> None:
        """Resume FSM execution from PAUSED state."""
        target_state: FSMState | None = None

        with self._lock:
            if self._state_entered_at is None:
                self._state_entered_at = now

            if self._state != FSMState.PAUSED:
                return

            target = FSMState.IDLE if self._previous_state is None else self._previous_state
            assert_transition(FSMState.PAUSED, target)

            self._state = target
            self._previous_state = None
            self._state_entered_at = now
            target_state = target

        self._emit_event(
            "fsm_resumed",
            {
                "reason": reason,
                "to": target_state.value,
            },
        )

    def enter_recovery(self, reason: str, *, now: float) -> None:
        """Force entry into STUCK_RECOVERY state if transition is legal."""
        prev_state: FSMState | None = None

        with self._lock:
            if self._state_entered_at is None:
                self._state_entered_at = now

            if self._state == FSMState.STUCK_RECOVERY:
                return

            if not can_transition(self._state, FSMState.STUCK_RECOVERY):
                raise FSMError(
                    f"illegal transition: {self._state.value} -> {FSMState.STUCK_RECOVERY.value}"
                )

            prev_state = self._state
            self._previous_state = prev_state
            self._state = FSMState.STUCK_RECOVERY
            self._state_entered_at = now
            self._stuck_attempts += 1

        self._emit_event(
            "fsm_recovery_entered",
            {
                "reason": reason,
                "from": prev_state.value,
            },
        )
