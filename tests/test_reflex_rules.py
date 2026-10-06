"""Unit tests for reflex rules."""

import ast
import random
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.reflex.rules import (
    RULES_PRIORITY_ORDER,
    SIGNAL_DISPOSITIONS,
    SignalDisposition,
    UnknownSignalError,
    default_rules,
)
from wow_bot.reflex.signals import Signal
from wow_bot.reflex.sinks import ActuatorAbortSink, FSMSink, NullFSMController, RecoverySink
from wow_bot.reflex.sources import (
    SafeSignalSource,
    SourceError,
    enumerate_registered_signal_types,
)


def test_rules_priority_order_constant() -> None:
    expected = (
        "kill_switch",
        "safety_abort",
        "session_timeout",
        "focus_lost",
        "focus_gained",
    )
    assert RULES_PRIORITY_ORDER == expected


def test_empty_signal_list_produces_no_control_signals() -> None:
    rng = random.Random(0)
    signals: list[Signal] = []
    result = default_rules(signals, tick_index=0, rng=rng)
    assert result == []


def test_kill_switch_produces_abort_actuation() -> None:
    rng = random.Random(0)
    signals = [Signal(name="kill_switch", payload={}, ts=10.5)]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="kill_switch", ts=10.5)
    ]


def test_kill_switch_takes_priority_over_all_other_signals() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_gained", payload={}, ts=1.0),
        Signal(name="focus_lost", payload={}, ts=2.0),
        Signal(name="session_timeout", payload={}, ts=3.0),
        Signal(name="safety_abort", payload={}, ts=4.0),
        Signal(name="kill_switch", payload={}, ts=5.0),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="kill_switch", ts=5.0)
    ]


def test_safety_abort_produces_abort_actuation_when_kill_switch_absent() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_gained", payload={}, ts=1.0),
        Signal(name="focus_lost", payload={}, ts=2.0),
        Signal(name="session_timeout", payload={}, ts=3.0),
        Signal(name="safety_abort", payload={}, ts=4.0),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="safety_abort", ts=4.0)
    ]


def test_session_timeout_produces_pause_fsm_without_abort_actuation() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_gained", payload={}, ts=1.0),
        Signal(name="focus_lost", payload={}, ts=2.0),
        Signal(name="session_timeout", payload={}, ts=3.0),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="session_timeout", ts=3.0)
    ]


def test_focus_lost_produces_abort_then_pause_in_order() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_gained", payload={}, ts=1.0),
        Signal(name="focus_lost", payload={}, ts=2.5),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="focus_lost", ts=2.5),
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="focus_lost", ts=2.5),
    ]


def test_focus_gained_produces_resume_fsm_when_focus_lost_absent() -> None:
    rng = random.Random(0)
    signals = [Signal(name="focus_gained", payload={}, ts=7.2)]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.RESUME_FSM, reason="focus_gained", ts=7.2)
    ]


def test_focus_gained_ignored_when_focus_lost_present_same_tick() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_gained", payload={}, ts=8.0),
        Signal(name="focus_lost", payload={}, ts=8.5),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert len(result) == 2
    assert result[0] == ControlSignal(
        kind=ControlKind.ABORT_ACTUATION, reason="focus_lost", ts=8.5
    )
    assert result[1] == ControlSignal(
        kind=ControlKind.PAUSE_FSM, reason="focus_lost", ts=8.5
    )


def test_multiple_signals_same_name_do_not_produce_duplicates() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="kill_switch", payload={"source": "A"}, ts=1.0),
        Signal(name="kill_switch", payload={"source": "B"}, ts=2.0),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert result == [
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="kill_switch", ts=2.0)
    ]


def test_emitted_signal_ts_equals_max_triggering_ts() -> None:
    rng = random.Random(0)
    signals = [
        Signal(name="focus_lost", payload={}, ts=12.0),
        Signal(name="focus_lost", payload={}, ts=18.5),
        Signal(name="focus_lost", payload={}, ts=15.0),
    ]
    result = default_rules(signals, tick_index=1, rng=rng)

    assert len(result) == 2
    assert result[0].ts == 18.5
    assert result[1].ts == 18.5


def test_rules_determinism_and_input_immutability() -> None:
    rng1 = random.Random(42)
    rng2 = random.Random(99)
    signals = [
        Signal(name="focus_lost", payload={}, ts=10.0),
        Signal(name="focus_gained", payload={}, ts=11.0),
    ]
    signals_copy = list(signals)

    res1 = default_rules(signals, tick_index=5, rng=rng1)
    res2 = default_rules(signals, tick_index=5, rng=rng2)

    assert res1 == res2
    assert signals == signals_copy


def test_static_ast_default_rules_unused_parameters() -> None:
    rules_file = Path("src/wow_bot/reflex/rules.py")
    tree = ast.parse(rules_file.read_text("utf-8"))

    func_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name == "default_rules":
            func_node = node
            break

    assert func_node is not None, "default_rules function not found in rules.py"

    # Check function body (excluding signature arguments) for Name nodes matching "rng" or "tick_index"
    used_names: list[str] = []
    for stmt in func_node.body:
        for node in ast.walk(stmt):
            if isinstance(node, ast.Name):
                used_names.append(node.id)

    assert "rng" not in used_names, "'rng' was referenced inside default_rules body"
    assert "tick_index" not in used_names, "'tick_index' was referenced inside default_rules body"


def test_static_ast_rules_imports_isolation() -> None:
    rules_file = Path("src/wow_bot/reflex/rules.py")
    tree = ast.parse(rules_file.read_text("utf-8"))

    allowed_wow_bot_modules = {
        "wow_bot.reflex.signals",
        "wow_bot.reflex.controls",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                if mod_name.startswith("wow_bot."):
                    assert mod_name in allowed_wow_bot_modules, f"Disallowed import: {mod_name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if mod_name.startswith("wow_bot"):
                assert mod_name in allowed_wow_bot_modules, f"Disallowed import from: {mod_name}"


def test_all_registered_source_signals_have_disposition() -> None:
    """Acceptance 1: Every signal type produced by registered sources has a rule disposition."""
    registered_signals = enumerate_registered_signal_types()
    assert len(registered_signals) > 0

    for sig_name in registered_signals:
        assert (
            sig_name in SIGNAL_DISPOSITIONS
        ), f"Signal {sig_name!r} has no declared disposition in SIGNAL_DISPOSITIONS"
        assert isinstance(
            SIGNAL_DISPOSITIONS[sig_name], SignalDisposition
        ), f"Signal {sig_name!r} disposition is not a SignalDisposition enum"

    # Specific check that ignored and handled are explicitly classified
    assert SIGNAL_DISPOSITIONS["position_stuck"] == SignalDisposition.HANDLED
    assert SIGNAL_DISPOSITIONS["position_clear"] == SignalDisposition.IGNORED
    assert SIGNAL_DISPOSITIONS["combat_interrupt"] == SignalDisposition.IGNORED
    assert SIGNAL_DISPOSITIONS["combat_defensive"] == SignalDisposition.IGNORED
    assert SIGNAL_DISPOSITIONS["combat_retreat"] == SignalDisposition.IGNORED
    assert SIGNAL_DISPOSITIONS["source_error"] == SignalDisposition.HANDLED


def test_position_stuck_signal_reaches_recovery_sink() -> None:
    """Acceptance 2: A position_stuck signal produces ENTER_RECOVERY and reaches RecoverySink and FSMSink."""
    rng = random.Random(0)
    stuck_signal = Signal(
        name="position_stuck",
        payload={"spread": 0.2, "window_s": 3.0, "samples": 5},
        ts=42.5,
    )

    control_signals = default_rules([stuck_signal], tick_index=1, rng=rng)
    assert len(control_signals) == 1
    assert control_signals[0] == ControlSignal(
        kind=ControlKind.ENTER_RECOVERY,
        reason="position_stuck",
        ts=42.5,
    )

    # Dispatch to RecoverySink with session
    fsm = NullFSMController()
    session_mock = MagicMock()
    recovery_sink = RecoverySink(fsm=fsm, session=session_mock)
    recovery_sink.handle(control_signals[0])

    assert fsm.calls() == [("enter_recovery", "position_stuck")]
    session_mock.write_event.assert_called_once_with({
        "event": "recovery_entered",
        "reason": "position_stuck",
    })

    # Dispatch to FSMSink
    fsm2 = NullFSMController()
    fsm_sink = FSMSink(fsm=fsm2)
    fsm_sink.handle(control_signals[0])
    assert fsm2.calls() == [("enter_recovery", "position_stuck")]


def test_source_failure_fail_closed_and_observable() -> None:
    """Acceptance 3: A source failure is fail-closed, observable, and aborts actuation."""
    failing_source = MagicMock()
    failing_source.poll.side_effect = RuntimeError("hardware glitch")

    # Safe wrapping with re_raise=False -> produces source_error signal
    safe_src = SafeSignalSource(failing_source, re_raise=False, source_name="GpsSensor")
    signals = safe_src.poll(100.0)

    assert safe_src.error_count == 1
    assert isinstance(safe_src.last_error, RuntimeError)
    assert len(signals) == 1
    assert signals[0].name == "source_error"
    assert signals[0].payload["source"] == "GpsSensor"
    assert "hardware glitch" in signals[0].payload["error"]
    assert signals[0].payload["error_count"] == 1
    assert signals[0].ts == 100.0

    # Rules map source_error to ABORT_ACTUATION
    rng = random.Random(0)
    controls = default_rules(signals, tick_index=1, rng=rng)
    assert controls == [
        ControlSignal(
            kind=ControlKind.ABORT_ACTUATION,
            reason="source_error",
            ts=100.0,
        )
    ]

    # ActuatorAbortSink handles ABORT_ACTUATION by calling actuator.abort()
    actuator_mock = MagicMock()
    abort_sink = ActuatorAbortSink(actuator=actuator_mock)
    abort_sink.handle(controls[0])
    actuator_mock.abort.assert_called_once_with("source_error")

    # Safe wrapping with re_raise=True -> raises SourceError
    re_raise_src = SafeSignalSource(failing_source, re_raise=True, source_name="BadSensor")
    with pytest.raises(SourceError, match="Signal source BadSensor failed"):
        re_raise_src.poll(101.0)
    assert re_raise_src.error_count == 1
    assert isinstance(re_raise_src.last_error, RuntimeError)


def test_unknown_signal_type_raises_or_routes_to_explicit_catch_all() -> None:
    """Acceptance 4: An unknown signal type raises UnknownSignalError or routes to catch-all."""
    rng = random.Random(0)
    alien_signal = Signal(name="unknown_external_event", payload={}, ts=15.0)

    # 1. Without catch_all, raises UnknownSignalError
    with pytest.raises(UnknownSignalError, match="unknown_external_event"):
        default_rules([alien_signal], tick_index=1, rng=rng)

    # 2. With explicit catch_all returning a control signal
    catch_all_called: list[Signal] = []

    def custom_catch_all(sig: Signal) -> ControlSignal | None:
        catch_all_called.append(sig)
        return ControlSignal(
            kind=ControlKind.ABORT_ACTUATION,
            reason=f"unhandled:{sig.name}",
            ts=sig.ts,
        )

    controls = default_rules(
        [alien_signal], tick_index=1, rng=rng, catch_all=custom_catch_all
    )
    assert len(catch_all_called) == 1
    assert catch_all_called[0].name == "unknown_external_event"
    assert controls == [
        ControlSignal(
            kind=ControlKind.ABORT_ACTUATION,
            reason="unhandled:unknown_external_event",
            ts=15.0,
        )
    ]

    # 3. With catch_all returning None, signal is absorbed without raise
    controls_none = default_rules(
        [alien_signal], tick_index=1, rng=rng, catch_all=lambda s: None
    )
    assert controls_none == []

