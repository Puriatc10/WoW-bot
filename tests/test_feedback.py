"""Unit tests for feedback intake layer and state transition decision logic."""

import ast
from pathlib import Path

import pytest

from wow_bot.actuation.mapper import ActionResult, ActionStatus
from wow_bot.executor.feedback import (
    Feedback,
    FeedbackDecision,
    FeedbackKind,
    FeedbackOutcome,
    classify_action_result,
    classify_reflex_signal,
    decide,
)
from wow_bot.executor.states import FSMState, can_transition


# Item 1: Feedback with non-dict payload raises ValueError
def test_feedback_non_dict_payload_raises() -> None:
    with pytest.raises(ValueError, match="payload must be a dict"):
        Feedback(
            kind=FeedbackKind.ACTION_SUCCESS,
            ts=1.0,
            payload="not_a_dict",  # type: ignore[arg-type]
        )


# Item 2: Feedback with negative ts raises ValueError
def test_feedback_negative_ts_raises() -> None:
    with pytest.raises(ValueError, match="ts must be finite and >= 0.0"):
        Feedback(
            kind=FeedbackKind.ACTION_SUCCESS,
            ts=-0.1,
        )


# Item 3: Feedback with non-finite ts raises ValueError
@pytest.mark.parametrize("bad_ts", [float("nan"), float("inf"), float("-inf")])
def test_feedback_non_finite_ts_raises(bad_ts: float) -> None:
    with pytest.raises(ValueError, match="ts must be finite and >= 0.0"):
        Feedback(
            kind=FeedbackKind.ACTION_SUCCESS,
            ts=bad_ts,
        )


# Item 4 & 5: classify_action_result maps SUCCESS/FAILED/TIMEOUT correctly with payload latency_ms and notes
@pytest.mark.parametrize(
    ("status", "expected_kind"),
    [
        (ActionStatus.SUCCESS, FeedbackKind.ACTION_SUCCESS),
        (ActionStatus.FAILED, FeedbackKind.ACTION_FAILED),
        (ActionStatus.TIMEOUT, FeedbackKind.ACTION_TIMEOUT),
    ],
)
def test_classify_action_result_mapping(status: ActionStatus, expected_kind: FeedbackKind) -> None:
    result = ActionResult(status=status, latency_ms=123.45, notes="test notes")
    fb = classify_action_result(result, ts=10.0, reason="action_reason")

    assert fb.kind == expected_kind
    assert fb.ts == 10.0
    assert fb.reason == "action_reason"
    assert fb.payload == {"latency_ms": 123.45, "notes": "test notes"}


# Item 6: classify_reflex_signal maps each of the 6 known names
@pytest.mark.parametrize(
    ("name", "expected_kind"),
    [
        ("focus_lost", FeedbackKind.FOCUS_LOST),
        ("position_stuck", FeedbackKind.POSITION_STUCK),
        ("position_clear", FeedbackKind.POSITION_CLEAR),
        ("session_timeout", FeedbackKind.SESSION_TIMEOUT),
        ("safety_abort", FeedbackKind.SAFETY_ABORT),
        ("kill_switch", FeedbackKind.KILL_SWITCH),
    ],
)
def test_classify_reflex_signal_known_names(name: str, expected_kind: FeedbackKind) -> None:
    fb = classify_reflex_signal(name, ts=5.0)
    assert fb.kind == expected_kind
    assert fb.ts == 5.0


# Item 7: classify_reflex_signal with unknown name raises ValueError
def test_classify_reflex_signal_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown reflex signal name: 'unknown_signal'"):
        classify_reflex_signal("unknown_signal", ts=1.0)


# Item 8: classify_reflex_signal copies payload shallowly
def test_classify_reflex_signal_shallow_copy_payload() -> None:
    original_payload = {"key": "value", "count": 1}
    fb = classify_reflex_signal("position_stuck", ts=1.0, payload=original_payload)

    # Mutate original dictionary
    original_payload["key"] = "mutated"
    original_payload["new_key"] = "new"

    assert fb.payload == {"key": "value", "count": 1}


# Item 9: classify_reflex_signal populates reason from payload["reason"] when present and str
def test_classify_reflex_signal_reason_from_str_payload() -> None:
    fb = classify_reflex_signal(
        "safety_abort",
        ts=1.0,
        payload={"reason": "safety_violation_detected"},
    )
    assert fb.reason == "safety_violation_detected"


# Item 10: classify_reflex_signal leaves reason empty when payload has no "reason" or a non-str value
@pytest.mark.parametrize("payload", [None, {}, {"reason": 12345}, {"reason": None}, {"reason": ["a"]}])
def test_classify_reflex_signal_reason_empty_when_non_str(payload: dict[str, object] | None) -> None:
    fb = classify_reflex_signal("focus_lost", ts=1.0, payload=payload)
    assert fb.reason == ""


# Item 11: decide with KILL_SWITCH returns TO_PAUSED, next_state=PAUSED, reason="kill_switch"
def test_decide_kill_switch_returns_to_paused() -> None:
    fb = Feedback(kind=FeedbackKind.KILL_SWITCH, ts=1.0)
    decision = decide(FSMState.IDLE, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_PAUSED,
        next_state=FSMState.PAUSED,
        reason="kill_switch",
    )


# Item 12: decide with SAFETY_ABORT returns TO_PAUSED, reason="safety_abort"
def test_decide_safety_abort_returns_to_paused() -> None:
    fb = Feedback(kind=FeedbackKind.SAFETY_ABORT, ts=1.0)
    decision = decide(FSMState.COMBAT, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_PAUSED,
        next_state=FSMState.PAUSED,
        reason="safety_abort",
    )


# Item 13: decide with SESSION_TIMEOUT returns TO_PAUSED
def test_decide_session_timeout_returns_to_paused() -> None:
    fb = Feedback(kind=FeedbackKind.SESSION_TIMEOUT, ts=1.0)
    decision = decide(FSMState.SCANNING, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_PAUSED,
        next_state=FSMState.PAUSED,
        reason="session_timeout",
    )


# Item 14: decide with FOCUS_LOST returns TO_PAUSED
def test_decide_focus_lost_returns_to_paused() -> None:
    fb = Feedback(kind=FeedbackKind.FOCUS_LOST, ts=1.0)
    decision = decide(FSMState.MOVING_TO_TARGET, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_PAUSED,
        next_state=FSMState.PAUSED,
        reason="focus_lost",
    )


# Item 15: decide with POSITION_STUCK from IDLE returns TO_STUCK_RECOVERY
def test_decide_position_stuck_from_idle_blocked() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=1.0)
    # IDLE -> STUCK_RECOVERY is blocked by schema
    decision = decide(FSMState.IDLE, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="blocked:position_stuck",
    )


def test_decide_position_stuck_from_moving_returns_to_stuck_recovery() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=1.0)
    decision = decide(FSMState.MOVING_TO_TARGET, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_STUCK_RECOVERY,
        next_state=FSMState.STUCK_RECOVERY,
        reason="position_stuck",
    )


# Item 16: decide with POSITION_STUCK from STUCK_RECOVERY returns STAY
def test_decide_position_stuck_from_stuck_recovery_returns_stay() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=1.0)
    decision = decide(FSMState.STUCK_RECOVERY, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="",
    )


# Item 17: decide with POSITION_STUCK from state that cannot transition to STUCK_RECOVERY returns STAY with reason "blocked:"
def test_decide_position_stuck_blocked_transition() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_STUCK, ts=1.0)
    # IDLE cannot transition to STUCK_RECOVERY
    decision = decide(FSMState.IDLE, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="blocked:position_stuck",
    )


# Item 18: decide with POSITION_CLEAR from STUCK_RECOVERY returns TO_RECOVERING
def test_decide_position_clear_from_stuck_recovery() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_CLEAR, ts=1.0)
    decision = decide(FSMState.STUCK_RECOVERY, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_RECOVERING,
        next_state=FSMState.RECOVERING,
        reason="position_clear",
    )


# Item 19: decide with POSITION_CLEAR from IDLE returns STAY
def test_decide_position_clear_from_idle_returns_stay() -> None:
    fb = Feedback(kind=FeedbackKind.POSITION_CLEAR, ts=1.0)
    decision = decide(FSMState.IDLE, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="",
    )


# Item 20: decide with TARGET_LOST from TARGETING returns TO_RECOVERING
def test_decide_target_lost_from_targeting() -> None:
    fb = Feedback(kind=FeedbackKind.TARGET_LOST, ts=1.0)
    decision = decide(FSMState.TARGETING, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_RECOVERING,
        next_state=FSMState.RECOVERING,
        reason="target_lost",
    )


# Item 21: decide with TARGET_LOST from COMBAT returns TO_RECOVERING
def test_decide_target_lost_from_combat() -> None:
    fb = Feedback(kind=FeedbackKind.TARGET_LOST, ts=1.0)
    decision = decide(FSMState.COMBAT, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_RECOVERING,
        next_state=FSMState.RECOVERING,
        reason="target_lost",
    )


# Item 22: decide with TARGET_LOST from IDLE returns STAY
def test_decide_target_lost_from_idle_returns_stay() -> None:
    fb = Feedback(kind=FeedbackKind.TARGET_LOST, ts=1.0)
    decision = decide(FSMState.IDLE, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="",
    )


# Item 23: decide with ACTION_TIMEOUT returns TO_STUCK_RECOVERY from allowed states, STAY from STUCK_RECOVERY
def test_decide_action_timeout_from_moving() -> None:
    fb = Feedback(kind=FeedbackKind.ACTION_TIMEOUT, ts=1.0)
    decision = decide(FSMState.MOVING_TO_TARGET, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_STUCK_RECOVERY,
        next_state=FSMState.STUCK_RECOVERY,
        reason="action_timeout",
    )


def test_decide_action_timeout_from_stuck_recovery() -> None:
    fb = Feedback(kind=FeedbackKind.ACTION_TIMEOUT, ts=1.0)
    decision = decide(FSMState.STUCK_RECOVERY, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="",
    )


# Item 24: decide with ACTION_FAILED from RECOVERING returns TO_IDLE with reason "repeated_failure"
def test_decide_action_failed_from_recovering() -> None:
    fb = Feedback(kind=FeedbackKind.ACTION_FAILED, ts=1.0)
    decision = decide(FSMState.RECOVERING, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_IDLE,
        next_state=FSMState.IDLE,
        reason="repeated_failure",
    )


# Item 25: decide with ACTION_FAILED from COMBAT returns TO_RECOVERING
def test_decide_action_failed_from_combat() -> None:
    fb = Feedback(kind=FeedbackKind.ACTION_FAILED, ts=1.0)
    decision = decide(FSMState.COMBAT, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.TO_RECOVERING,
        next_state=FSMState.RECOVERING,
        reason="action_failed",
    )


# Item 26: decide with ACTION_SUCCESS returns STAY with empty reason
def test_decide_action_success_returns_stay() -> None:
    fb = Feedback(kind=FeedbackKind.ACTION_SUCCESS, ts=1.0)
    decision = decide(FSMState.COMBAT, fb)
    assert decision == FeedbackDecision(
        outcome=FeedbackOutcome.STAY,
        next_state=None,
        reason="",
    )


# Item 27: FeedbackDecision with STAY and non-None next_state raises ValueError
def test_feedback_decision_stay_non_none_next_state_raises() -> None:
    with pytest.raises(ValueError, match="outcome == STAY implies next_state must be None"):
        FeedbackDecision(
            outcome=FeedbackOutcome.STAY,
            next_state=FSMState.IDLE,
        )


# Item 28: FeedbackDecision with non-STAY and None next_state raises ValueError
def test_feedback_decision_non_stay_none_next_state_raises() -> None:
    with pytest.raises(ValueError, match="outcome != STAY implies next_state must not be None"):
        FeedbackDecision(
            outcome=FeedbackOutcome.TO_IDLE,
            next_state=None,
        )


# Item 29: FeedbackDecision with non-STAY is always a legal transition
def test_decide_non_stay_is_always_legal_transition() -> None:
    all_states = list(FSMState)
    all_kinds = list(FeedbackKind)

    for state in all_states:
        for kind in all_kinds:
            fb = Feedback(kind=kind, ts=1.0)
            decision = decide(state, fb)
            if decision.outcome != FeedbackOutcome.STAY:
                assert decision.next_state is not None
                assert can_transition(state, decision.next_state), (
                    f"Illegal decision produced for state={state.value}, kind={kind.value}: "
                    f"proposed target {decision.next_state.value}"
                )


# Item 30: determinism: decide called twice with same inputs returns equal FeedbackDecision objects
def test_decide_determinism() -> None:
    all_states = list(FSMState)
    all_kinds = list(FeedbackKind)

    for state in all_states:
        for kind in all_kinds:
            fb = Feedback(kind=kind, ts=1.0, reason="det_test", payload={"a": 1})
            d1 = decide(state, fb)
            d2 = decide(state, fb)
            assert d1 == d2


# Item 31: Static AST check: feedback.py does not import forbidden modules
def test_static_ast_import_check() -> None:
    target_path = Path("src/wow_bot/executor/feedback.py")
    assert target_path.exists(), "feedback.py must exist"

    source = target_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(target_path))

    forbidden_modules = {
        "wow_bot.reflex",
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
        "wow_bot.actuation.actuator",
    }
    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import: {mod}"
                for sub in forbidden_substrings:
                    assert sub not in mod.lower(), f"Forbidden module name substring: {mod}"

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod.startswith(forbidden), f"Forbidden import from: {mod}"
            for sub in forbidden_substrings:
                assert sub not in mod.lower(), f"Forbidden module name substring in ImportFrom: {mod}"
