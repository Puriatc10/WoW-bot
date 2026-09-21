"""Unit tests for the health state machine (Task 8.2).

Verifies all 35 acceptance items covering threshold validations, state transitions,
hold windows, observation thresholds, hysteresis, rate reliability skipping,
worst-metric ranking, force_state, session events, determinism, snapshot immutability,
and static AST isolation checks.
"""

import ast
import json
import math
from pathlib import Path
from typing import Any, cast

import pytest

from wow_bot.config import Config
from wow_bot.session import Session
from wow_bot.watchdog.health import (
    HealthConfig,
    HealthError,
    HealthState,
    HealthStateMachine,
    HealthTransition,
    MetricName,
    MetricThresholds,
)
from wow_bot.watchdog.metrics import ProgressSnapshot


def make_test_config(session_root: Path) -> Config:
    """Helper to construct a valid Config instance for session testing."""
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=session_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def make_snapshot(
    position_delta: float = 10.0,
    inventory_delta: int = 5,
    level_or_xp_delta: float = 100.0,
    successful_actions_per_minute: float = 10.0,
    reflex_tick_rate_hz: float = 10.0,
    is_rate_reliable: bool = True,
) -> ProgressSnapshot:
    """Helper creating a baseline healthy ProgressSnapshot."""
    return ProgressSnapshot(
        window_s=60.0,
        sample_count=10,
        window_start_ts=0.0,
        window_end_ts=60.0,
        position_delta=position_delta,
        inventory_delta=inventory_delta,
        level_or_xp_delta=level_or_xp_delta,
        successful_actions_per_minute=successful_actions_per_minute,
        reflex_tick_rate_hz=reflex_tick_rate_hz,
        is_rate_reliable=is_rate_reliable,
    )


def assert_state(sm: HealthStateMachine, expected: HealthState) -> None:
    """Helper asserting machine current state without mypy scope narrowing."""
    assert sm.current_state == expected


# 1. MetricThresholds with critical_below > degraded_below raises ValueError.
def test_metric_thresholds_critical_greater_than_degraded_raises() -> None:
    with pytest.raises(ValueError, match="critical_below"):
        MetricThresholds(degraded_below=1.0, critical_below=2.0)


# 2. MetricThresholds with negative recovery_margin raises ValueError.
def test_metric_thresholds_negative_recovery_margin_raises() -> None:
    with pytest.raises(ValueError, match="recovery_margin"):
        MetricThresholds(degraded_below=2.0, critical_below=1.0, recovery_margin=-0.1)


# 3. MetricThresholds with non-finite floats raises ValueError.
def test_metric_thresholds_non_finite_floats_raises() -> None:
    with pytest.raises(ValueError, match="degraded_below"):
        MetricThresholds(degraded_below=math.nan, critical_below=1.0)
    with pytest.raises(ValueError, match="critical_below"):
        MetricThresholds(degraded_below=2.0, critical_below=math.inf)
    with pytest.raises(ValueError, match="recovery_margin"):
        MetricThresholds(degraded_below=2.0, critical_below=1.0, recovery_margin=math.nan)


# 4. HealthConfig with a missing metric raises ValueError.
def test_health_config_missing_metric_raises() -> None:
    partial_thresholds = {
        MetricName.POSITION_DELTA: MetricThresholds(degraded_below=0.5, critical_below=0.1),
    }
    with pytest.raises(ValueError, match="MUST cover every MetricName"):
        HealthConfig(thresholds=cast(Any, partial_thresholds))


# 5. HealthConfig with hold_s < 0 raises ValueError.
def test_health_config_negative_hold_s_raises() -> None:
    default_cfg = HealthConfig.default()
    with pytest.raises(ValueError, match="hold_s"):
        HealthConfig(thresholds=default_cfg.thresholds, hold_s=-1.0)


# 6. HealthConfig with min_observations_for_transition < 1 raises ValueError.
def test_health_config_invalid_min_observations_raises() -> None:
    default_cfg = HealthConfig.default()
    with pytest.raises(ValueError, match="min_observations_for_transition"):
        HealthConfig(thresholds=default_cfg.thresholds, min_observations_for_transition=0)


# 7. HealthConfig with rate_metrics containing an unknown name raises ValueError.
def test_health_config_invalid_rate_metrics_raises() -> None:
    default_cfg = HealthConfig.default()
    with pytest.raises(ValueError, match="rate_metrics"):
        HealthConfig(
            thresholds=default_cfg.thresholds,
            rate_metrics=frozenset({cast(MetricName, "invalid_metric")}),
        )


# 8. HealthConfig.default() returns a config covering all metrics.
def test_health_config_default_covers_all_metrics() -> None:
    cfg = HealthConfig.default()
    assert set(cfg.thresholds.keys()) == set(MetricName)
    assert cfg.hold_s == 5.0
    assert cfg.min_observations_for_transition == 3
    assert cfg.rate_metrics == frozenset({
        MetricName.SUCCESSFUL_ACTIONS_PER_MINUTE,
        MetricName.REFLEX_TICK_RATE_HZ,
    })


# 9. HealthTransition invariants checks.
def test_health_transition_invariants() -> None:
    # from == to
    with pytest.raises(ValueError, match="cannot be equal"):
        HealthTransition(
            from_state=HealthState.HEALTHY,
            to_state=HealthState.HEALTHY,
            ts=1.0,
            trigger_metrics=(MetricName.POSITION_DELTA,),
            reason="test",
        )
    # empty trigger_metrics
    with pytest.raises(ValueError, match="trigger_metrics"):
        HealthTransition(
            from_state=HealthState.HEALTHY,
            to_state=HealthState.DEGRADED,
            ts=1.0,
            trigger_metrics=(),
            reason="test",
        )
    # empty reason
    with pytest.raises(ValueError, match="reason"):
        HealthTransition(
            from_state=HealthState.HEALTHY,
            to_state=HealthState.DEGRADED,
            ts=1.0,
            trigger_metrics=(MetricName.POSITION_DELTA,),
            reason="   ",
        )
    # negative ts
    with pytest.raises(ValueError, match="ts"):
        HealthTransition(
            from_state=HealthState.HEALTHY,
            to_state=HealthState.DEGRADED,
            ts=-1.0,
            trigger_metrics=(MetricName.POSITION_DELTA,),
            reason="test",
        )


# 10. observe with a healthy snapshot returns None and keeps state HEALTHY.
def test_observe_healthy_snapshot_returns_none() -> None:
    sm = HealthStateMachine()
    snap = make_snapshot()
    result = sm.observe(snap, now=1.0)
    assert result is None
    assert_state(sm, HealthState.HEALTHY)


# 11. observe transitions HEALTHY -> DEGRADED when a metric falls below degraded_below and hold_s elapses with enough observations.
def test_observe_healthy_to_degraded_transition() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)  # degraded_below is 0.5

    # Obs 1 at t=0.0
    assert sm.observe(deg_snap, now=0.0) is None
    assert sm.pending_state == HealthState.DEGRADED
    assert sm.pending_count == 1

    # Obs 2 at t=2.0
    assert sm.observe(deg_snap, now=2.0) is None
    assert sm.pending_count == 2

    # Obs 3 at t=5.0 (hold_s=5.0 elapsed, count=3 >= 3)
    tr = sm.observe(deg_snap, now=5.0)
    assert tr is not None
    assert tr.from_state == HealthState.HEALTHY
    assert tr.to_state == HealthState.DEGRADED
    assert tr.ts == 5.0
    assert tr.trigger_metrics == (MetricName.POSITION_DELTA,)
    assert "position_delta=0.3" in tr.reason
    assert_state(sm, HealthState.DEGRADED)


# 12. observe transitions DEGRADED -> CRITICAL when a metric falls below critical_below.
def test_observe_degraded_to_critical_transition() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)
    sm.observe(deg_snap, now=0.0)
    sm.observe(deg_snap, now=2.0)
    sm.observe(deg_snap, now=5.0)
    assert_state(sm, HealthState.DEGRADED)

    crit_snap = make_snapshot(position_delta=0.05)  # critical_below is 0.1
    assert sm.observe(crit_snap, now=6.0) is None
    assert sm.observe(crit_snap, now=8.0) is None
    tr = sm.observe(crit_snap, now=11.0)
    assert tr is not None
    assert tr.from_state == HealthState.DEGRADED
    assert tr.to_state == HealthState.CRITICAL
    assert_state(sm, HealthState.CRITICAL)


# 13. observe does NOT transition before hold_s elapses, even if candidate has enough observations.
def test_observe_hold_s_window_enforced() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)

    # 5 observations in 3 seconds (hold_s = 5.0)
    assert sm.observe(deg_snap, now=0.0) is None
    assert sm.observe(deg_snap, now=0.5) is None
    assert sm.observe(deg_snap, now=1.0) is None
    assert sm.observe(deg_snap, now=2.0) is None
    assert sm.observe(deg_snap, now=3.0) is None
    assert_state(sm, HealthState.HEALTHY)

    # At t=5.0 hold_s elapses
    tr = sm.observe(deg_snap, now=5.0)
    assert tr is not None
    assert_state(sm, HealthState.DEGRADED)


# 14. observe does NOT transition before min_observations_for_transition is reached, even if hold_s has elapsed.
def test_observe_min_observations_enforced() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)

    # First observation at t=0.0
    assert sm.observe(deg_snap, now=0.0) is None
    # Second observation at t=10.0 (hold_s=5.0 elapsed, but pending_count=2 < 3)
    assert sm.observe(deg_snap, now=10.0) is None
    assert_state(sm, HealthState.HEALTHY)

    # Third observation at t=11.0 (pending_count=3 >= 3)
    tr = sm.observe(deg_snap, now=11.0)
    assert tr is not None
    assert_state(sm, HealthState.DEGRADED)


# 15. Hysteresis: after DEGRADED, a metric must rise above degraded_below + recovery_margin to recover to HEALTHY.
def test_hysteresis_degraded_recovery() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)
    sm.observe(deg_snap, now=0.0)
    sm.observe(deg_snap, now=2.0)
    sm.observe(deg_snap, now=5.0)
    assert_state(sm, HealthState.DEGRADED)

    # degraded_below=0.5, recovery_margin=0.2 => recovery threshold = 0.7
    # Value 0.6 is > degraded_below but <= degraded_below + recovery_margin
    snap_06 = make_snapshot(position_delta=0.6)
    sm.observe(snap_06, now=6.0)
    sm.observe(snap_06, now=8.0)
    sm.observe(snap_06, now=11.0)
    assert_state(sm, HealthState.DEGRADED)  # Does NOT recover

    # Value 0.75 >= 0.7 recovery threshold
    snap_075 = make_snapshot(position_delta=0.75)
    sm.observe(snap_075, now=12.0)
    sm.observe(snap_075, now=14.0)
    tr = sm.observe(snap_075, now=17.0)
    assert tr is not None
    assert tr.from_state == HealthState.DEGRADED
    assert tr.to_state == HealthState.HEALTHY
    assert_state(sm, HealthState.HEALTHY)


# 16. Hysteresis: after CRITICAL, a metric must rise above critical_below + recovery_margin to leave CRITICAL.
def test_hysteresis_critical_recovery() -> None:
    sm = HealthStateMachine()
    crit_snap = make_snapshot(position_delta=0.05)  # critical_below = 0.1, degraded_below = 0.5, margin = 0.2
    sm.observe(crit_snap, now=0.0)
    sm.observe(crit_snap, now=2.0)
    sm.observe(crit_snap, now=5.0)
    assert_state(sm, HealthState.CRITICAL)

    # critical_below + recovery_margin = 0.1 + 0.2 = 0.3
    # Value 0.2 < 0.3 => remains CRITICAL
    snap_02 = make_snapshot(position_delta=0.2)
    sm.observe(snap_02, now=6.0)
    sm.observe(snap_02, now=8.0)
    sm.observe(snap_02, now=11.0)
    assert_state(sm, HealthState.CRITICAL)

    # Value 0.4 >= 0.3 (critical recovery) but < 0.7 (degraded recovery) => recovers to DEGRADED
    snap_04 = make_snapshot(position_delta=0.4)
    sm.observe(snap_04, now=12.0)
    sm.observe(snap_04, now=14.0)
    tr = sm.observe(snap_04, now=17.0)
    assert tr is not None
    assert tr.from_state == HealthState.CRITICAL
    assert tr.to_state == HealthState.DEGRADED
    assert_state(sm, HealthState.DEGRADED)


# 17. Hysteresis: a metric hovering exactly at degraded_below does NOT oscillate the state across repeated observations.
def test_hysteresis_hovering_at_boundary_no_oscillation() -> None:
    sm = HealthStateMachine()
    snap_boundary = make_snapshot(position_delta=0.5)  # degraded_below is 0.5

    # First 3 observations transition HEALTHY -> DEGRADED
    sm.observe(snap_boundary, now=0.0)
    sm.observe(snap_boundary, now=2.0)
    tr = sm.observe(snap_boundary, now=5.0)
    assert tr is not None
    assert_state(sm, HealthState.DEGRADED)

    # Repeated observations at 0.5 remain DEGRADED without state change
    for t in range(6, 20):
        assert sm.observe(snap_boundary, now=float(t)) is None
        assert_state(sm, HealthState.DEGRADED)


# 18. Rate metrics with is_rate_reliable=False are skipped entirely.
def test_unreliable_rate_metrics_skipped() -> None:
    sm = HealthStateMachine()
    # successful_actions_per_minute = 0.0 (below degraded 2.0), but is_rate_reliable=False
    unreliable_snap = make_snapshot(successful_actions_per_minute=0.0, is_rate_reliable=False)

    for t in [0.0, 2.0, 5.0, 10.0]:
        assert sm.observe(unreliable_snap, now=t) is None
        assert_state(sm, HealthState.HEALTHY)


# 19. Rate metrics with is_rate_reliable=True DO trigger.
def test_reliable_rate_metrics_trigger() -> None:
    sm = HealthStateMachine()
    reliable_snap = make_snapshot(successful_actions_per_minute=0.0, is_rate_reliable=True)

    sm.observe(reliable_snap, now=0.0)
    sm.observe(reliable_snap, now=2.0)
    tr = sm.observe(reliable_snap, now=5.0)
    assert tr is not None
    assert tr.to_state == HealthState.CRITICAL  # 0.0 <= critical_below 0.5
    assert_state(sm, HealthState.CRITICAL)


# 20. Delta metrics are evaluated regardless of is_rate_reliable.
def test_delta_metrics_evaluated_regardless_of_rate_reliability() -> None:
    sm = HealthStateMachine()
    # position_delta = 0.3 (degraded), is_rate_reliable = False
    snap = make_snapshot(position_delta=0.3, is_rate_reliable=False)

    sm.observe(snap, now=0.0)
    sm.observe(snap, now=2.0)
    tr = sm.observe(snap, now=5.0)
    assert tr is not None
    assert tr.to_state == HealthState.DEGRADED
    assert_state(sm, HealthState.DEGRADED)


# 21. Overall state is the worst per-metric state: DEGRADED on one metric yields DEGRADED overall.
def test_overall_state_degraded_on_one_metric() -> None:
    sm = HealthStateMachine()
    snap = make_snapshot(position_delta=0.3)  # position_delta degraded, all others healthy

    sm.observe(snap, now=0.0)
    sm.observe(snap, now=2.0)
    tr = sm.observe(snap, now=5.0)
    assert tr is not None
    assert tr.to_state == HealthState.DEGRADED
    assert_state(sm, HealthState.DEGRADED)


# 22. Overall state is CRITICAL when any metric is CRITICAL even if others are DEGRADED or HEALTHY.
def test_overall_state_critical_takes_precedence() -> None:
    sm = HealthStateMachine()
    # position_delta = 0.3 (degraded), reflex_tick_rate_hz = 0.1 (critical)
    snap = make_snapshot(position_delta=0.3, reflex_tick_rate_hz=0.1)

    sm.observe(snap, now=0.0)
    sm.observe(snap, now=2.0)
    tr = sm.observe(snap, now=5.0)
    assert tr is not None
    assert tr.to_state == HealthState.CRITICAL
    assert_state(sm, HealthState.CRITICAL)


# 23. trigger_metrics lists exactly the non-HEALTHY metrics, sorted by MetricName.value.
def test_trigger_metrics_lists_non_healthy_sorted_by_value() -> None:
    sm = HealthStateMachine()
    # position_delta (degraded), reflex_tick_rate_hz (degraded)
    snap = make_snapshot(position_delta=0.3, reflex_tick_rate_hz=1.0)

    sm.observe(snap, now=0.0)
    sm.observe(snap, now=2.0)
    tr = sm.observe(snap, now=5.0)
    assert tr is not None
    expected_sorted = (
        MetricName.POSITION_DELTA,
        MetricName.REFLEX_TICK_RATE_HZ,
    )
    assert tr.trigger_metrics == expected_sorted
    assert tr.trigger_metrics[0].value < tr.trigger_metrics[1].value


# 24. reason mentions the worst metric name and its value.
def test_reason_mentions_worst_metric_name_and_value() -> None:
    sm = HealthStateMachine()
    snap = make_snapshot(position_delta=0.3)

    sm.observe(snap, now=0.0)
    sm.observe(snap, now=2.0)
    tr = sm.observe(snap, now=5.0)
    assert tr is not None
    assert MetricName.POSITION_DELTA.value in tr.reason
    assert "0.3" in tr.reason


# 25. force_state transitions immediately, bypassing hysteresis.
def test_force_state_immediate_transition() -> None:
    sm = HealthStateMachine()
    assert_state(sm, HealthState.HEALTHY)

    tr = sm.force_state(HealthState.CRITICAL, now=1.0, reason="manual override")
    assert tr.from_state == HealthState.HEALTHY
    assert tr.to_state == HealthState.CRITICAL
    assert tr.ts == 1.0
    assert tr.reason == "manual override"
    assert_state(sm, HealthState.CRITICAL)


# 26. force_state with the current state raises HealthError.
def test_force_state_same_state_raises() -> None:
    sm = HealthStateMachine()
    with pytest.raises(HealthError, match="Cannot force state to current state"):
        sm.force_state(HealthState.HEALTHY, now=1.0, reason="no change")


# 27. force_state emits "watchdog_transition" exactly once with a session attached.
def test_force_state_emits_session_event(tmp_path: Path) -> None:
    cfg = make_test_config(tmp_path)
    session = Session.start(cfg)

    sm = HealthStateMachine(session=session)
    sm.force_state(HealthState.DEGRADED, now=2.5, reason="test force")

    session.close("test_complete")

    events_file = session.path / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    transition_events = [e for e in events if e.get("event") == "watchdog_transition"]

    assert len(transition_events) == 1
    evt = transition_events[0]
    assert evt["from_state"] == "healthy"
    assert evt["to_state"] == "degraded"
    assert evt["ts"] == 2.5
    assert evt["reason"] == "test force"


# 28. force_state with empty reason raises HealthError.
def test_force_state_empty_reason_raises() -> None:
    sm = HealthStateMachine()
    with pytest.raises(HealthError, match="reason"):
        sm.force_state(HealthState.CRITICAL, now=1.0, reason="   ")


# 29. Session events: a normal transition emits exactly one "watchdog_transition" with documented payload keys.
def test_observe_emits_session_event(tmp_path: Path) -> None:
    cfg = make_test_config(tmp_path)
    session = Session.start(cfg)

    sm = HealthStateMachine(session=session)
    deg_snap = make_snapshot(position_delta=0.3)

    sm.observe(deg_snap, now=0.0)
    sm.observe(deg_snap, now=2.0)
    sm.observe(deg_snap, now=5.0)

    session.close("test_complete")

    events_file = session.path / "events.jsonl"
    events = [json.loads(line) for line in events_file.read_text().splitlines() if line.strip()]
    transition_events = [e for e in events if e.get("event") == "watchdog_transition"]

    assert len(transition_events) == 1
    evt = transition_events[0]
    assert evt["event"] == "watchdog_transition"
    assert evt["from_state"] == "healthy"
    assert evt["to_state"] == "degraded"
    assert evt["ts"] == 5.0
    assert "position_delta" in evt["trigger_metrics"]
    assert "reason" in evt


# 30. No session attached: observe and force_state work without raising.
def test_no_session_works_without_raising() -> None:
    sm = HealthStateMachine(session=None)
    deg_snap = make_snapshot(position_delta=0.3)

    sm.observe(deg_snap, now=0.0)
    sm.observe(deg_snap, now=2.0)
    tr1 = sm.observe(deg_snap, now=5.0)
    assert tr1 is not None

    tr2 = sm.force_state(HealthState.CRITICAL, now=10.0, reason="force")
    assert tr2 is not None


# 31. reset returns to HEALTHY and clears pending state.
def test_reset_clears_pending_and_returns_to_healthy() -> None:
    sm = HealthStateMachine()
    deg_snap = make_snapshot(position_delta=0.3)
    sm.observe(deg_snap, now=0.0)
    assert sm.pending_state == HealthState.DEGRADED
    assert sm.pending_count == 1

    sm.reset()
    assert_state(sm, HealthState.HEALTHY)
    assert sm.pending_state is None
    assert sm.pending_count == 0


# 32. Determinism: two machines with same config, snapshot sequence, and now sequence produce identical transitions.
def test_determinism() -> None:
    sm1 = HealthStateMachine()
    sm2 = HealthStateMachine()

    snapshots = [
        make_snapshot(position_delta=10.0),
        make_snapshot(position_delta=0.3),
        make_snapshot(position_delta=0.3),
        make_snapshot(position_delta=0.3),
        make_snapshot(position_delta=0.05),
        make_snapshot(position_delta=0.05),
        make_snapshot(position_delta=0.05),
    ]
    timestamps = [0.0, 1.0, 3.0, 6.0, 7.0, 9.0, 12.0]

    tr1_list = []
    tr2_list = []

    for snap, now in zip(snapshots, timestamps, strict=True):
        tr1_list.append(sm1.observe(snap, now=now))
        tr2_list.append(sm2.observe(snap, now=now))

    assert tr1_list == tr2_list
    assert sm1.current_state == sm2.current_state


# 33. observe does not mutate the snapshot (snapshot comparison).
def test_observe_does_not_mutate_snapshot() -> None:
    sm = HealthStateMachine()
    snap = make_snapshot(position_delta=0.3)
    snap_dict_before = snap.to_json()

    sm.observe(snap, now=1.0)
    assert snap.to_json() == snap_dict_before


# 34. Static AST check: health.py does not import forbidden modules.
def test_ast_no_forbidden_imports() -> None:
    health_py_path = Path("src/wow_bot/watchdog/health.py")
    tree = ast.parse(health_py_path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in forbidden_modules, f"Forbidden import: {mod}"
                assert not any(
                    kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
                ), f"Forbidden LLM import: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_modules, f"Forbidden import from: {mod}"
            assert not any(
                kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
            ), f"Forbidden LLM import from: {mod}"


# 35. Static AST check: health.py does not call time.monotonic, time.time, or time.perf_counter.
def test_ast_no_time_calls() -> None:
    health_py_path = Path("src/wow_bot/watchdog/health.py")
    tree = ast.parse(health_py_path.read_text(encoding="utf-8"))

    forbidden_time_calls = {"monotonic", "time", "perf_counter"}

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
        ):
            assert (
                node.func.attr not in forbidden_time_calls
            ), f"Forbidden time call: time.{node.func.attr}"
