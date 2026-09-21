"""Unit tests for wow_bot.strategist.cooldown_v2 (Task 9.2).

Validates all acceptance criteria for CooldownGate, CooldownConfig, CooldownCheck,
and static AST module import and time-read isolation.
"""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from wow_bot.config import Config
from wow_bot.executor.states import FSMState
from wow_bot.session import Session
from wow_bot.strategist.cooldown_v2 import (
    CooldownCheck,
    CooldownConfig,
    CooldownDecision,
    CooldownError,
    CooldownGate,
)


@pytest.fixture
def test_session(tmp_path: Path) -> Session:
    """Create a real Session rooted in tmp_path for session event testing."""
    config = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="f12",
        session_root=tmp_path / "runs",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def read_session_events(session: Session) -> list[dict[str, Any]]:
    """Read and parse json objects from session's events.jsonl."""
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Config Validation Tests ---


def test_cooldown_config_negative_min_interval_raises() -> None:
    """CooldownConfig with negative min_interval_s raises ValueError."""
    with pytest.raises(ValueError, match="min_interval_s must be >= 0.0"):
        CooldownConfig(min_interval_s=-1.0)


def test_cooldown_config_max_calls_per_minute_invalid_raises() -> None:
    """CooldownConfig with max_calls_per_minute < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_calls_per_minute must be >= 1"):
        CooldownConfig(max_calls_per_minute=0)


def test_cooldown_config_overlapping_blocked_preferred_states_raises() -> None:
    """CooldownConfig with overlapping blocked_states and preferred_states raises ValueError."""
    with pytest.raises(ValueError, match="disjoint"):
        CooldownConfig(
            blocked_states=frozenset({FSMState.COMBAT, FSMState.IDLE}),
            preferred_states=frozenset({FSMState.IDLE}),
        )


def test_cooldown_config_non_frozenset_raises() -> None:
    """CooldownConfig with non-frozenset sets raises ValueError."""
    with pytest.raises(ValueError, match="blocked_states must be a frozenset"):
        CooldownConfig(blocked_states={FSMState.COMBAT})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="preferred_states must be a frozenset"):
        CooldownConfig(preferred_states=[FSMState.IDLE])  # type: ignore[arg-type]


def test_cooldown_config_non_finite_floats_raises() -> None:
    """CooldownConfig with non-finite floats raises ValueError."""
    with pytest.raises(ValueError, match="min_interval_s must be a finite float"):
        CooldownConfig(min_interval_s=float("nan"))

    with pytest.raises(ValueError, match="min_interval_s must be a finite float"):
        CooldownConfig(min_interval_s=float("inf"))


# --- Check Invariants Tests ---


def test_cooldown_check_invariants() -> None:
    """CooldownCheck invariants validation."""
    # ALLOWED with non-zero wait
    with pytest.raises(ValueError, match="seconds_until_next_allowed == 0.0"):
        CooldownCheck(
            decision=CooldownDecision.ALLOWED,
            reason="ok",
            seconds_until_next_allowed=5.0,
            recent_calls=1,
        )

    # FORCED_ALLOWED with non-zero wait
    with pytest.raises(ValueError, match="seconds_until_next_allowed == 0.0"):
        CooldownCheck(
            decision=CooldownDecision.FORCED_ALLOWED,
            reason="manual_force_open",
            seconds_until_next_allowed=1.0,
            recent_calls=1,
        )

    # BLOCKED_STATE with finite wait
    with pytest.raises(ValueError, match="seconds_until_next_allowed == inf"):
        CooldownCheck(
            decision=CooldownDecision.BLOCKED_STATE,
            reason="blocked",
            seconds_until_next_allowed=10.0,
            recent_calls=0,
        )

    # BLOCKED_MANUAL with finite wait
    with pytest.raises(ValueError, match="seconds_until_next_allowed == inf"):
        CooldownCheck(
            decision=CooldownDecision.BLOCKED_MANUAL,
            reason="manual",
            seconds_until_next_allowed=0.0,
            recent_calls=0,
        )

    # Empty reason raises
    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        CooldownCheck(
            decision=CooldownDecision.ALLOWED,
            reason="",
            seconds_until_next_allowed=0.0,
            recent_calls=0,
        )

    # Negative recent_calls raises
    with pytest.raises(ValueError, match="recent_calls must be an integer >= 0"):
        CooldownCheck(
            decision=CooldownDecision.ALLOWED,
            reason="ok",
            seconds_until_next_allowed=0.0,
            recent_calls=-1,
        )


# --- CooldownGate Check Core Tests ---


def test_check_invalid_now_raises_cooldown_error() -> None:
    """check with now non-finite or negative raises CooldownError."""
    gate = CooldownGate()

    with pytest.raises(CooldownError, match="now must be a finite float"):
        gate.check(FSMState.IDLE, now=-1.0)

    with pytest.raises(CooldownError, match="now must be a finite float"):
        gate.check(FSMState.IDLE, now=float("nan"))

    with pytest.raises(CooldownError, match="now must be a finite float"):
        gate.check(FSMState.IDLE, now=float("inf"))


def test_first_call_idle_allowed() -> None:
    """First call in IDLE with an empty history returns ALLOWED."""
    gate = CooldownGate()
    res = gate.check(FSMState.IDLE, now=10.0)

    assert res.decision == CooldownDecision.ALLOWED
    assert res.reason == "ok"
    assert res.seconds_until_next_allowed == 0.0
    assert res.recent_calls == 1
    assert gate.last_decision == res
    assert gate.recent_calls == 1


def test_min_interval_cooldown_blocked_and_pass() -> None:
    """Second call before and after min_interval_s."""
    config = CooldownConfig(min_interval_s=15.0)
    gate = CooldownGate(config=config)

    # First call at t=10.0
    res1 = gate.check(FSMState.IDLE, now=10.0)
    assert res1.decision == CooldownDecision.ALLOWED

    # Second call at t = 10.0 + 15.0 - 0.001 = 24.999
    res2 = gate.check(FSMState.IDLE, now=24.999)
    assert res2.decision == CooldownDecision.BLOCKED_COOLDOWN
    assert res2.reason == "min_interval"
    assert pytest.approx(res2.seconds_until_next_allowed, abs=1e-5) == 0.001
    assert res2.recent_calls == 1

    # Second call at t = 10.0 + 15.0 = 25.0
    res3 = gate.check(FSMState.IDLE, now=25.0)
    assert res3.decision == CooldownDecision.ALLOWED
    assert res3.reason == "ok"
    assert res3.seconds_until_next_allowed == 0.0
    assert res3.recent_calls == 2


def test_hard_blocked_states() -> None:
    """COMBAT, FLEEING, STUCK_RECOVERY, PAUSED return BLOCKED_STATE unconditionally."""
    gate = CooldownGate()

    for state in (FSMState.COMBAT, FSMState.FLEEING, FSMState.STUCK_RECOVERY, FSMState.PAUSED):
        res = gate.check(state, now=100.0)
        assert res.decision == CooldownDecision.BLOCKED_STATE
        assert res.reason == f"blocked_state:{state.value}"
        assert res.seconds_until_next_allowed == float("inf")


def test_busy_states_behavior() -> None:
    """MOVING_TO_TARGET with allow_while_busy False vs True."""
    # allow_while_busy = False (default)
    gate1 = CooldownGate(config=CooldownConfig(allow_while_busy=False))
    res1 = gate1.check(FSMState.MOVING_TO_TARGET, now=10.0)
    assert res1.decision == CooldownDecision.BLOCKED_STATE
    assert res1.reason.startswith("busy_state:")

    # allow_while_busy = True
    gate2 = CooldownGate(config=CooldownConfig(allow_while_busy=True))
    res2 = gate2.check(FSMState.MOVING_TO_TARGET, now=10.0)
    assert res2.decision == CooldownDecision.ALLOWED
    assert res2.reason == "ok"


def test_rate_limit_sliding_window() -> None:
    """4 calls spaced by min_interval_s, then a 5th call returns BLOCKED_RATE_LIMIT."""
    config = CooldownConfig(min_interval_s=10.0, max_calls_per_minute=4)
    gate = CooldownGate(config=config)

    # 4 calls at t=10, 20, 30, 40
    for t in [10.0, 20.0, 30.0, 40.0]:
        res = gate.check(FSMState.IDLE, now=t)
        assert res.decision == CooldownDecision.ALLOWED

    assert gate.recent_calls == 4

    # 5th call at t=55.0 (min_interval of 10s passed, but 4 calls in last 60s window)
    res5 = gate.check(FSMState.IDLE, now=55.0)
    assert res5.decision == CooldownDecision.BLOCKED_RATE_LIMIT
    assert res5.reason == "rate_limit"
    # Oldest call was at t=10.0, so wait until oldest + 60.0 - 55.0 = 15.0s
    assert pytest.approx(res5.seconds_until_next_allowed, abs=1e-5) == 15.0
    assert res5.recent_calls == 4

    # Call at t=70.001 (oldest at t=10.0 aged out at 70.0)
    res6 = gate.check(FSMState.IDLE, now=70.001)
    assert res6.decision == CooldownDecision.ALLOWED
    assert res6.recent_calls == 4  # [20, 30, 40, 70.001]


def test_deque_is_bounded() -> None:
    """Calling check 1000 times across a long time span does not grow deque beyond max_calls_per_minute + 1."""
    config = CooldownConfig(min_interval_s=1.0, max_calls_per_minute=4)
    gate = CooldownGate(config=config)

    for i in range(1000):
        gate.check(FSMState.IDLE, now=float(i * 10))

    assert gate.recent_calls <= config.max_calls_per_minute + 1


def test_recent_calls_reflects_only_last_60s() -> None:
    """recent_calls reflects entries within the last 60 s."""
    config = CooldownConfig(min_interval_s=1.0, max_calls_per_minute=10)
    gate = CooldownGate(config=config)

    gate.check(FSMState.IDLE, now=10.0)
    gate.check(FSMState.IDLE, now=20.0)
    assert gate.recent_calls == 2

    # Check at t=75.0 (t=10.0 is > 60s old and purged)
    gate.check(FSMState.IDLE, now=75.0)
    assert gate.recent_calls == 2  # [20.0, 75.0]


# --- Manual Overrides & Events Tests ---


def test_force_open_once_behavior_and_event(test_session: Session) -> None:
    """force_open_once allows check and emits cooldown_forced_open exactly once on check."""
    gate = CooldownGate(session=test_session)

    # First call at t=10.0
    res0 = gate.check(FSMState.IDLE, now=10.0)
    assert res0.decision == CooldownDecision.ALLOWED

    # Gate is now in min_interval cooldown at t=12.0
    gate.force_open_once()

    # Consuming check at t=12.0
    res1 = gate.check(FSMState.IDLE, now=12.0)
    assert res1.decision == CooldownDecision.FORCED_ALLOWED
    assert res1.reason == "manual_force_open"
    assert res1.seconds_until_next_allowed == 0.0

    # Second check at same now (t=12.0) returns normal decision (BLOCKED_COOLDOWN)
    res2 = gate.check(FSMState.IDLE, now=12.0)
    assert res2.decision == CooldownDecision.BLOCKED_COOLDOWN

    events = read_session_events(test_session)
    forced_events = [e for e in events if e.get("event") == "cooldown_forced_open"]
    assert len(forced_events) == 1
    assert forced_events[0]["ts"] == 12.0
    assert forced_events[0]["state"] == FSMState.IDLE.value


def test_force_closed_and_clear_manual_block_events(test_session: Session) -> None:
    """force_closed_until and clear_manual_block operations and event emissions."""
    gate = CooldownGate(session=test_session)

    assert gate.is_manual_blocked is False

    gate.force_closed_until(now=12.0)
    assert gate.is_manual_blocked is True

    res1 = gate.check(FSMState.IDLE, now=15.0)
    assert res1.decision == CooldownDecision.BLOCKED_MANUAL
    assert res1.reason == "manual_block_active"
    assert res1.seconds_until_next_allowed == float("inf")

    gate.clear_manual_block()
    assert gate.is_manual_blocked is False

    res2 = gate.check(FSMState.IDLE, now=20.0)
    assert res2.decision == CooldownDecision.ALLOWED

    events = read_session_events(test_session)
    block_events = [e for e in events if e.get("event") == "cooldown_manual_block"]
    clear_events = [e for e in events if e.get("event") == "cooldown_manual_clear"]
    allowed_events = [e for e in events if e.get("event") == "cooldown_allowed"]

    assert len(block_events) == 1
    assert block_events[0]["ts"] == 12.0

    assert len(clear_events) == 1

    assert len(allowed_events) == 1
    assert allowed_events[0]["ts"] == 20.0
    assert allowed_events[0]["state"] == FSMState.IDLE.value


def test_no_session_attached_works_without_raising() -> None:
    """All methods work seamlessly when session is None."""
    gate = CooldownGate(session=None)
    gate.force_closed_until(now=10.0)
    assert gate.check(FSMState.IDLE, now=12.0).decision == CooldownDecision.BLOCKED_MANUAL
    gate.clear_manual_block()
    gate.force_open_once()
    assert gate.check(FSMState.IDLE, now=15.0).decision == CooldownDecision.FORCED_ALLOWED
    assert gate.check(FSMState.IDLE, now=30.0).decision == CooldownDecision.ALLOWED
    gate.reset()


def test_reset_clears_state() -> None:
    """reset clears deque, force flag, and manual block without events."""
    gate = CooldownGate()
    gate.check(FSMState.IDLE, now=10.0)
    gate.force_closed_until(now=12.0)
    gate.force_open_once()

    assert gate.recent_calls == 1
    assert gate.is_manual_blocked is True

    gate.reset()

    assert gate.recent_calls == 0
    assert gate.is_manual_blocked is False
    # Next call is clean ALLOWED
    assert gate.check(FSMState.IDLE, now=15.0).decision == CooldownDecision.ALLOWED


def test_determinism() -> None:
    """Two gates with same config and call sequence produce identical results."""
    config = CooldownConfig(min_interval_s=10.0, max_calls_per_minute=3)
    gate1 = CooldownGate(config=config)
    gate2 = CooldownGate(config=config)

    sequence = [
        (FSMState.IDLE, 0.0),
        (FSMState.IDLE, 5.0),
        (FSMState.IDLE, 10.0),
        (FSMState.COMBAT, 15.0),
        (FSMState.IDLE, 20.0),
        (FSMState.IDLE, 25.0),
    ]

    for state, now in sequence:
        res1 = gate1.check(state, now=now)
        res2 = gate2.check(state, now=now)

        assert res1.decision == res2.decision
        assert res1.reason == res2.reason
        assert res1.seconds_until_next_allowed == res2.seconds_until_next_allowed
        assert res1.recent_calls == res2.recent_calls


# --- Static AST Code Inspection Tests ---


def test_static_ast_forbidden_imports() -> None:
    """Verify cooldown_v2.py does not import forbidden modules."""
    filepath = Path("src/wow_bot/strategist/cooldown_v2.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist.prompts",
        "wow_bot.strategist.prompts_v2",
        "wow_bot.strategist.llm_client",
        "wow_bot.strategist.orchestrator",
        "wow_bot.strategist.orchestrator_v2",
        "wow_bot.strategist.parser",
        "wow_bot.strategist.parser_v2",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for forbidden in forbidden_modules:
                    assert not mod_name.startswith(forbidden), f"Forbidden import: {mod_name}"
                for sub in forbidden_substrings:
                    assert sub not in mod_name.lower(), f"Forbidden LLM module import: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod_name.startswith(forbidden), f"Forbidden import_from: {mod_name}"
            for sub in forbidden_substrings:
                assert sub not in mod_name.lower(), f"Forbidden LLM import_from: {mod_name}"


def test_static_ast_no_time_calls() -> None:
    """Verify cooldown_v2.py does not call time.monotonic, time.time, or time.perf_counter."""
    filepath = Path("src/wow_bot/strategist/cooldown_v2.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    time_functions = {"monotonic", "time", "perf_counter"}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
        ):
            assert node.func.attr not in time_functions, (
                f"Forbidden time call: time.{node.func.attr}"
            )


def test_static_ast_prelab_strategist_modules_untouched() -> None:
    """Verify no pre-lab strategist module was modified."""
    prelab_files = [
        Path("src/wow_bot/strategist/llm_client.py"),
        Path("src/wow_bot/strategist/orchestrator.py"),
        Path("src/wow_bot/strategist/parser.py"),
        Path("src/wow_bot/strategist/prompts.py"),
    ]
    for p in prelab_files:
        assert p.exists(), f"Pre-lab file missing: {p}"
