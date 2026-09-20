"""Unit tests for reflex rules."""

import ast
import random
from pathlib import Path

from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.reflex.rules import RULES_PRIORITY_ORDER, default_rules
from wow_bot.reflex.signals import Signal


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
