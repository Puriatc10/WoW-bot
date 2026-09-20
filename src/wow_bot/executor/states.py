"""FSM v2 state schema definition, transition table, and validation utilities.

This module is the single source of truth for legal FSM states and transitions.
It contains pure definitions and schema validation logic only.
"""

from dataclasses import dataclass
from enum import Enum


class FSMState(str, Enum):
    """FSM states supported in execution loop v2."""

    IDLE = "IDLE"
    SCANNING = "SCANNING"
    MOVING_TO_TARGET = "MOVING_TO_TARGET"
    TARGETING = "TARGETING"
    PATHING = "PATHING"
    COMBAT = "COMBAT"
    LOOTING = "LOOTING"
    FLEEING = "FLEEING"
    STUCK_RECOVERY = "STUCK_RECOVERY"
    WAITING_GCD = "WAITING_GCD"
    RECOVERING = "RECOVERING"
    PAUSED = "PAUSED"


class TransitionError(Exception):
    """Raised when an illegal FSM state transition or invalid table specification is encountered."""


@dataclass(frozen=True)
class StateSpec:
    """Specification for a single FSM state, including transition allowed set and timeouts."""

    state: FSMState
    allowed_transitions: frozenset[FSMState]
    entry_hook_name: str | None = None
    exit_hook_name: str | None = None
    timeout_s: float | None = None

    def __post_init__(self) -> None:
        if self.timeout_s is not None and self.timeout_s <= 0:
            raise ValueError(f"timeout_s must be positive, got {self.timeout_s}")

        for hook_field, hook_val in [
            ("entry_hook_name", self.entry_hook_name),
            ("exit_hook_name", self.exit_hook_name),
        ]:
            if hook_val is not None and (not hook_val or any(c.isspace() for c in hook_val)):
                raise ValueError(
                    f"{hook_field} must be a non-empty string without whitespace, got {hook_val!r}"
                )

        if self.state in self.allowed_transitions and self.state != FSMState.PAUSED:
            raise ValueError(
                f"Self-loop transition is forbidden for non-PAUSED state {self.state.value}"
            )


TRANSITION_TABLE: dict[FSMState, StateSpec] = {
    FSMState.IDLE: StateSpec(
        state=FSMState.IDLE,
        allowed_transitions=frozenset({FSMState.SCANNING, FSMState.PAUSED}),
        timeout_s=None,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.SCANNING: StateSpec(
        state=FSMState.SCANNING,
        allowed_transitions=frozenset(
            {
                FSMState.MOVING_TO_TARGET,
                FSMState.TARGETING,
                FSMState.PATHING,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=30.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.MOVING_TO_TARGET: StateSpec(
        state=FSMState.MOVING_TO_TARGET,
        allowed_transitions=frozenset(
            {
                FSMState.TARGETING,
                FSMState.PATHING,
                FSMState.STUCK_RECOVERY,
                FSMState.RECOVERING,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=60.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.TARGETING: StateSpec(
        state=FSMState.TARGETING,
        allowed_transitions=frozenset(
            {
                FSMState.COMBAT,
                FSMState.MOVING_TO_TARGET,
                FSMState.PATHING,
                FSMState.RECOVERING,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=5.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.PATHING: StateSpec(
        state=FSMState.PATHING,
        allowed_transitions=frozenset(
            {
                FSMState.MOVING_TO_TARGET,
                FSMState.STUCK_RECOVERY,
                FSMState.RECOVERING,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=120.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.COMBAT: StateSpec(
        state=FSMState.COMBAT,
        allowed_transitions=frozenset(
            {
                FSMState.LOOTING,
                FSMState.FLEEING,
                FSMState.RECOVERING,
                FSMState.STUCK_RECOVERY,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=180.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.LOOTING: StateSpec(
        state=FSMState.LOOTING,
        allowed_transitions=frozenset(
            {
                FSMState.SCANNING,
                FSMState.IDLE,
                FSMState.RECOVERING,
                FSMState.PAUSED,
            }
        ),
        timeout_s=10.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.FLEEING: StateSpec(
        state=FSMState.FLEEING,
        allowed_transitions=frozenset(
            {
                FSMState.IDLE,
                FSMState.SCANNING,
                FSMState.RECOVERING,
                FSMState.PAUSED,
            }
        ),
        timeout_s=30.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.STUCK_RECOVERY: StateSpec(
        state=FSMState.STUCK_RECOVERY,
        allowed_transitions=frozenset(
            {
                FSMState.IDLE,
                FSMState.SCANNING,
                FSMState.RECOVERING,
                FSMState.PAUSED,
            }
        ),
        timeout_s=15.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.WAITING_GCD: StateSpec(
        state=FSMState.WAITING_GCD,
        allowed_transitions=frozenset(
            {
                FSMState.COMBAT,
                FSMState.TARGETING,
                FSMState.IDLE,
                FSMState.PAUSED,
            }
        ),
        timeout_s=3.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.RECOVERING: StateSpec(
        state=FSMState.RECOVERING,
        allowed_transitions=frozenset(
            {
                FSMState.IDLE,
                FSMState.SCANNING,
                FSMState.STUCK_RECOVERY,
                FSMState.PAUSED,
            }
        ),
        timeout_s=10.0,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
    FSMState.PAUSED: StateSpec(
        state=FSMState.PAUSED,
        allowed_transitions=frozenset(FSMState),
        timeout_s=None,
        entry_hook_name=None,
        exit_hook_name=None,
    ),
}


def validate_table(table: dict[FSMState, StateSpec]) -> None:
    """Validate that an FSM transition table covers all states correctly."""
    missing_states = set(FSMState) - set(table.keys())
    if missing_states:
        raise TransitionError(
            f"Transition table missing required FSMState members: {sorted(s.value for s in missing_states)}"
        )

    for key_state, spec in table.items():
        if spec.state != key_state:
            raise TransitionError(
                f"Table key {key_state.value!r} does not match StateSpec state {spec.state.value!r}"
            )

        for allowed in spec.allowed_transitions:
            if not isinstance(allowed, FSMState) or allowed not in FSMState:
                raise TransitionError(
                    f"State {key_state.value!r} contains invalid allowed transition target: {allowed!r}"
                )

        if key_state in spec.allowed_transitions and key_state != FSMState.PAUSED:
            raise TransitionError(
                f"State {key_state.value!r} contains illegal self-loop in allowed_transitions"
            )


def can_transition(from_state: FSMState, to_state: FSMState) -> bool:
    """Check whether a transition from from_state to to_state is legal."""
    spec = TRANSITION_TABLE.get(from_state)
    if spec is None:
        return False
    return to_state in spec.allowed_transitions


def assert_transition(from_state: FSMState, to_state: FSMState) -> None:
    """Assert that a transition from from_state to to_state is legal or raise TransitionError."""
    if not can_transition(from_state, to_state):
        raise TransitionError(f"illegal transition: {from_state.value} -> {to_state.value}")


def all_states() -> tuple[FSMState, ...]:
    """Return all FSMState members in definition order."""
    return tuple(FSMState)


def timeout_for(state: FSMState) -> float | None:
    """Return the configured state timeout in seconds for state, or None if no timeout."""
    return TRANSITION_TABLE[state].timeout_s


# Module-level validation call at import time
validate_table(TRANSITION_TABLE)
