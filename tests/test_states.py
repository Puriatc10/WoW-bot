"""Unit tests for wow_bot.executor.states schema and validation utilities."""

import ast
from pathlib import Path

import pytest

from wow_bot.executor.states import (
    TRANSITION_TABLE,
    FSMState,
    StateSpec,
    TransitionError,
    all_states,
    assert_transition,
    can_transition,
    timeout_for,
    validate_table,
)


def test_fsm_state_members() -> None:
    """FSMState has exactly the 12 documented members."""
    expected_members = [
        "IDLE",
        "SCANNING",
        "MOVING_TO_TARGET",
        "TARGETING",
        "PATHING",
        "COMBAT",
        "LOOTING",
        "FLEEING",
        "STUCK_RECOVERY",
        "WAITING_GCD",
        "RECOVERING",
        "PAUSED",
    ]
    assert [state.name for state in FSMState] == expected_members
    assert len(FSMState) == 12


def test_transition_table_covers_all_states() -> None:
    """TRANSITION_TABLE covers every FSMState member."""
    assert set(TRANSITION_TABLE.keys()) == set(FSMState)


def test_state_spec_matches_dict_key() -> None:
    """Every StateSpec.state matches its dict key."""
    for key, spec in TRANSITION_TABLE.items():
        assert spec.state == key


def test_validate_table_missing_state() -> None:
    """validate_table raises TransitionError when a state is missing."""
    incomplete_table = dict(TRANSITION_TABLE)
    del incomplete_table[FSMState.PAUSED]

    with pytest.raises(TransitionError, match="missing required FSMState members"):
        validate_table(incomplete_table)


def test_validate_table_key_mismatch() -> None:
    """validate_table raises TransitionError when StateSpec.state does not match its key."""
    mismatched_table = dict(TRANSITION_TABLE)
    mismatched_table[FSMState.IDLE] = StateSpec(
        state=FSMState.COMBAT,
        allowed_transitions=frozenset({FSMState.SCANNING, FSMState.PAUSED}),
    )

    with pytest.raises(TransitionError, match="does not match StateSpec state"):
        validate_table(mismatched_table)


def test_validate_table_non_paused_self_loop() -> None:
    """validate_table raises TransitionError when a non-PAUSED state has itself in allowed_transitions."""
    spec = StateSpec(
        state=FSMState.IDLE,
        allowed_transitions=frozenset({FSMState.SCANNING, FSMState.PAUSED}),
    )
    object.__setattr__(
        spec, "allowed_transitions", frozenset({FSMState.SCANNING, FSMState.PAUSED, FSMState.IDLE})
    )

    bad_table = dict(TRANSITION_TABLE)
    bad_table[FSMState.IDLE] = spec

    with pytest.raises(TransitionError, match="contains illegal self-loop"):
        validate_table(bad_table)


def test_no_non_paused_self_loops_in_real_table() -> None:
    """No non-PAUSED state has itself in allowed_transitions (verify by scanning the real table)."""
    for state, spec in TRANSITION_TABLE.items():
        if state != FSMState.PAUSED:
            assert state not in spec.allowed_transitions


def test_paused_self_loop_and_full_reachability() -> None:
    """PAUSED allows a self-loop and allows every other state."""
    paused_spec = TRANSITION_TABLE[FSMState.PAUSED]
    assert FSMState.PAUSED in paused_spec.allowed_transitions
    assert paused_spec.allowed_transitions == frozenset(FSMState)


def test_can_transition_legal_pairs() -> None:
    """can_transition returns True for every documented pair in TRANSITION_TABLE."""
    for from_state, spec in TRANSITION_TABLE.items():
        for to_state in spec.allowed_transitions:
            assert can_transition(from_state, to_state) is True


def test_can_transition_illegal_pairs() -> None:
    """can_transition returns False for explicitly illegal pairs."""
    assert can_transition(FSMState.IDLE, FSMState.FLEEING) is False
    assert can_transition(FSMState.COMBAT, FSMState.MOVING_TO_TARGET) is False
    assert can_transition(FSMState.LOOTING, FSMState.COMBAT) is False


def test_can_transition_non_paused_self_loops() -> None:
    """can_transition returns False when to_state == from_state for all non-PAUSED states."""
    for state in FSMState:
        if state != FSMState.PAUSED:
            assert can_transition(state, state) is False


def test_assert_transition_illegal() -> None:
    """assert_transition raises TransitionError with the exact message format on illegal pairs."""
    with pytest.raises(TransitionError, match=r"^illegal transition: IDLE -> FLEEING$"):
        assert_transition(FSMState.IDLE, FSMState.FLEEING)

    with pytest.raises(TransitionError, match=r"^illegal transition: COMBAT -> MOVING_TO_TARGET$"):
        assert_transition(FSMState.COMBAT, FSMState.MOVING_TO_TARGET)


def test_assert_transition_legal() -> None:
    """assert_transition returns None on legal pairs."""
    assert_transition(FSMState.IDLE, FSMState.SCANNING)
    assert_transition(FSMState.COMBAT, FSMState.LOOTING)
    assert_transition(FSMState.PAUSED, FSMState.PAUSED)


def test_all_states_order() -> None:
    """all_states returns a tuple in the same order as FSMState definition."""
    assert all_states() == tuple(FSMState)


def test_timeout_for_values() -> None:
    """timeout_for returns the documented value for every state, and None for IDLE and PAUSED."""
    expected_timeouts = {
        FSMState.IDLE: None,
        FSMState.SCANNING: 30.0,
        FSMState.MOVING_TO_TARGET: 60.0,
        FSMState.TARGETING: 5.0,
        FSMState.PATHING: 120.0,
        FSMState.COMBAT: 180.0,
        FSMState.LOOTING: 10.0,
        FSMState.FLEEING: 30.0,
        FSMState.STUCK_RECOVERY: 15.0,
        FSMState.WAITING_GCD: 3.0,
        FSMState.RECOVERING: 10.0,
        FSMState.PAUSED: None,
    }

    for state, expected in expected_timeouts.items():
        assert timeout_for(state) == expected


def test_statespec_invalid_timeout() -> None:
    """StateSpec with timeout_s <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="timeout_s must be positive"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.SCANNING}),
            timeout_s=0.0,
        )

    with pytest.raises(ValueError, match="timeout_s must be positive"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.SCANNING}),
            timeout_s=-5.0,
        )


def test_statespec_invalid_entry_hook() -> None:
    """StateSpec with empty entry_hook_name raises ValueError."""
    with pytest.raises(ValueError, match="entry_hook_name must be a non-empty string"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.SCANNING}),
            entry_hook_name="",
        )

    with pytest.raises(ValueError, match="entry_hook_name must be a non-empty string"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.SCANNING}),
            entry_hook_name="  ",
        )


def test_statespec_invalid_exit_hook() -> None:
    """StateSpec with whitespace-only exit_hook_name raises ValueError."""
    with pytest.raises(ValueError, match="exit_hook_name must be a non-empty string"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.SCANNING}),
            exit_hook_name=" \t\n",
        )


def test_statespec_non_paused_self_loop_validation() -> None:
    """StateSpec with a non-PAUSED self-loop raises ValueError."""
    with pytest.raises(ValueError, match="Self-loop transition is forbidden"):
        StateSpec(
            state=FSMState.IDLE,
            allowed_transitions=frozenset({FSMState.IDLE}),
        )


def test_statespec_paused_self_loop_allowed() -> None:
    """PAUSED StateSpec with a self-loop does NOT raise."""
    spec = StateSpec(
        state=FSMState.PAUSED,
        allowed_transitions=frozenset({FSMState.PAUSED}),
    )
    assert spec.state == FSMState.PAUSED


def test_ast_forbidden_imports() -> None:
    """Static AST check: states.py does not import forbidden modules or LLM packages."""
    states_file = Path(__file__).parent.parent / "src" / "wow_bot" / "executor" / "states.py"
    source = states_file.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(states_file))

    forbidden_modules = [
        "wow_bot.strategist",
        "wow_bot.reflex",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
        "wow_bot.actuation",
    ]
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    imported_modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

    for mod in imported_modules:
        for forbidden in forbidden_modules:
            assert not mod.startswith(
                forbidden
            ), f"Forbidden import found in states.py: {mod}"
        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM-related import found in states.py: {mod}"
