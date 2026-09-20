"""Tests for concrete reflex signal sources."""

import ast
from pathlib import Path

import pytest

from wow_bot.reflex.signals import SignalSource
from wow_bot.reflex.sources import (
    FocusSignalSource,
    KillSwitchSignalSource,
    SafetySignalSource,
    SessionTimeoutSignalSource,
)


class FakeFocusManager:
    def __init__(self, focused: bool = True) -> None:
        self.focused = focused

    def is_focused(self) -> bool:
        return self.focused


class FakeSafetyLayer:
    def __init__(self, aborted: bool = False, reason: str | None = None) -> None:
        self.aborted = aborted
        self.reason = reason

    def is_aborted(self) -> bool:
        return self.aborted

    def abort_reason(self) -> str | None:
        return self.reason


class FakeKillSwitch:
    def __init__(self, triggered: bool = False, err: BaseException | None = None) -> None:
        self.triggered = triggered
        self.err = err

    def is_triggered(self) -> bool:
        return self.triggered

    def last_error(self) -> BaseException | None:
        return self.err


def test_focus_source_emit_initial_false_unfocused() -> None:
    """FocusSignalSource with emit_initial=False emits nothing on first poll when unfocused."""
    focus = FakeFocusManager(focused=False)
    source = FocusSignalSource(focus, emit_initial=False)  # type: ignore[arg-type]
    signals = source.poll(now=10.0)
    assert signals == []


def test_focus_source_emit_initial_true_unfocused() -> None:
    """FocusSignalSource with emit_initial=True emits one 'focus_lost' on first poll when unfocused."""
    focus = FakeFocusManager(focused=False)
    source = FocusSignalSource(focus, emit_initial=True)  # type: ignore[arg-type]
    signals = source.poll(now=10.0)
    assert len(signals) == 1
    assert signals[0].name == FocusSignalSource.NAME_LOST
    assert signals[0].name == "focus_lost"
    assert signals[0].payload == {}
    assert signals[0].ts == 10.0


def test_focus_source_transitions() -> None:
    """FocusSignalSource emits 'focus_lost' on transition to unfocused and 'focus_gained' on reverse transition."""
    focus = FakeFocusManager(focused=True)
    source = FocusSignalSource(focus, emit_initial=False)  # type: ignore[arg-type]

    # First poll initializes state to focused, emits nothing
    assert source.poll(now=1.0) == []

    # State unchanged
    assert source.poll(now=2.0) == []

    # Transition to unfocused -> emits focus_lost
    focus.focused = False
    signals = source.poll(now=3.0)
    assert len(signals) == 1
    assert signals[0].name == FocusSignalSource.NAME_LOST
    assert signals[0].name == "focus_lost"
    assert signals[0].payload == {}
    assert signals[0].ts == 3.0

    # Subsequent poll while still unfocused -> emits nothing
    assert source.poll(now=4.0) == []

    # Transition to focused -> emits focus_gained
    focus.focused = True
    signals_gained = source.poll(now=5.0)
    assert len(signals_gained) == 1
    assert signals_gained[0].name == FocusSignalSource.NAME_GAINED
    assert signals_gained[0].name == "focus_gained"
    assert signals_gained[0].payload == {}
    assert signals_gained[0].ts == 5.0

    # Subsequent poll while still focused -> emits nothing
    assert source.poll(now=6.0) == []


def test_session_timeout_before_deadline() -> None:
    """SessionTimeoutSignalSource emits nothing before deadline."""
    source = SessionTimeoutSignalSource(deadline_s=100.0, start_time_s=10.0)
    assert source.poll(now=50.0) == []
    assert source.poll(now=99.9) == []


def test_session_timeout_at_and_after_deadline() -> None:
    """SessionTimeoutSignalSource emits 'session_timeout' at/after deadline with correct payload, no second emission."""
    source = SessionTimeoutSignalSource(deadline_s=100.0, start_time_s=10.0)

    # Poll at deadline
    signals = source.poll(now=100.0)
    assert len(signals) == 1
    assert signals[0].name == SessionTimeoutSignalSource.NAME
    assert signals[0].name == "session_timeout"
    assert signals[0].payload == {"elapsed_s": 90.0, "max_s": 90.0}
    assert signals[0].ts == 100.0

    # Subsequent polls long after deadline emit nothing
    assert source.poll(now=101.0) == []
    assert source.poll(now=1000.0) == []


def test_session_timeout_validation() -> None:
    """SessionTimeoutSignalSource raises ValueError if deadline_s <= start_time_s."""
    with pytest.raises(ValueError, match="deadline_s"):
        SessionTimeoutSignalSource(deadline_s=10.0, start_time_s=10.0)

    with pytest.raises(ValueError, match="deadline_s"):
        SessionTimeoutSignalSource(deadline_s=5.0, start_time_s=10.0)


def test_safety_source_first_poll_not_aborted() -> None:
    """SafetySignalSource emits nothing on first poll when not aborted."""
    safety = FakeSafetyLayer(aborted=False)
    source = SafetySignalSource(safety)  # type: ignore[arg-type]
    assert source.poll(now=1.0) == []


def test_safety_source_transition_to_aborted() -> None:
    """SafetySignalSource emits 'safety_abort' on transition to aborted, never emits second time."""
    safety = FakeSafetyLayer(aborted=False)
    source = SafetySignalSource(safety)  # type: ignore[arg-type]

    # First poll: not aborted
    assert source.poll(now=1.0) == []

    # Transition to aborted
    safety.aborted = True
    safety.reason = "user_requested"
    signals = source.poll(now=2.0)
    assert len(signals) == 1
    assert signals[0].name == SafetySignalSource.NAME
    assert signals[0].name == "safety_abort"
    assert signals[0].payload == {"reason": "user_requested"}
    assert signals[0].ts == 2.0

    # Subsequent poll when still aborted: emits nothing
    assert source.poll(now=3.0) == []


def test_safety_source_already_aborted_at_construction() -> None:
    """SafetySignalSource emits 'safety_abort' on FIRST poll if already aborted at construction time."""
    safety = FakeSafetyLayer(aborted=True, reason="pre_existing_abort")
    source = SafetySignalSource(safety)  # type: ignore[arg-type]

    signals = source.poll(now=1.0)
    assert len(signals) == 1
    assert signals[0].name == SafetySignalSource.NAME
    assert signals[0].name == "safety_abort"
    assert signals[0].payload == {"reason": "pre_existing_abort"}
    assert signals[0].ts == 1.0

    # Never emits again
    assert source.poll(now=2.0) == []


def test_kill_switch_source_first_poll_not_triggered() -> None:
    """KillSwitchSignalSource emits nothing on first poll when not triggered."""
    ks = FakeKillSwitch(triggered=False)
    source = KillSwitchSignalSource(ks)  # type: ignore[arg-type]
    assert source.poll(now=1.0) == []


def test_kill_switch_source_transition_to_triggered() -> None:
    """KillSwitchSignalSource emits 'kill_switch' on transition with error payload."""
    err = RuntimeError("key_press_error")
    ks = FakeKillSwitch(triggered=False, err=err)
    source = KillSwitchSignalSource(ks)  # type: ignore[arg-type]

    assert source.poll(now=1.0) == []

    ks.triggered = True
    signals = source.poll(now=2.0)
    assert len(signals) == 1
    assert signals[0].name == KillSwitchSignalSource.NAME
    assert signals[0].name == "kill_switch"
    assert signals[0].payload == {"error": repr(err)}
    assert signals[0].ts == 2.0

    # Never emits again
    assert source.poll(now=3.0) == []


def test_kill_switch_source_already_triggered_at_construction() -> None:
    """KillSwitchSignalSource emits 'kill_switch' on FIRST poll if already triggered at construction time."""
    ks = FakeKillSwitch(triggered=True, err=None)
    source = KillSwitchSignalSource(ks)  # type: ignore[arg-type]

    signals = source.poll(now=1.0)
    assert len(signals) == 1
    assert signals[0].name == KillSwitchSignalSource.NAME
    assert signals[0].name == "kill_switch"
    assert signals[0].payload == {"error": None}
    assert signals[0].ts == 1.0

    # Never emits again
    assert source.poll(now=2.0) == []


def test_deterministic_ordering_when_polling_multiple_sources() -> None:
    """When multiple sources are polled in a list, signals order matches registration order."""
    focus = FakeFocusManager(focused=False)
    focus_source = FocusSignalSource(focus, emit_initial=True)  # type: ignore[arg-type]

    timeout_source = SessionTimeoutSignalSource(deadline_s=10.0, start_time_s=0.0)

    sources: list[SignalSource] = [focus_source, timeout_source]
    now = 10.0

    signals = []
    for src in sources:
        signals.extend(src.poll(now))

    assert len(signals) == 2
    assert signals[0].name == "focus_lost"
    assert signals[1].name == "session_timeout"
    assert signals[0].ts == now
    assert signals[1].ts == now


def test_static_ast_check_sources_imports() -> None:
    """Static AST check: sources.py does not import forbidden layer or LLM modules."""
    sources_file = Path("src/wow_bot/reflex/sources.py")
    tree = ast.parse(sources_file.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.executor",
        "wow_bot.reflex.stuck",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }

    forbidden_llm_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import found: {mod}"
                for llm_sub in forbidden_llm_substrings:
                    assert llm_sub not in mod.lower(), f"Forbidden LLM import found: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod.startswith(forbidden), f"Forbidden import found: {mod}"
            for llm_sub in forbidden_llm_substrings:
                assert llm_sub not in mod.lower(), f"Forbidden LLM import found: {mod}"
            for alias in node.names:
                name = alias.name
                for llm_sub in forbidden_llm_substrings:
                    assert llm_sub not in name.lower(), f"Forbidden LLM import found: {name}"
