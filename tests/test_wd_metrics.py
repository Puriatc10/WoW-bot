"""Tests for progress metrics collector and sliding window snapshot engine (T8.1)."""

import ast
import json
import math
from pathlib import Path

import pytest

from wow_bot.watchdog.metrics import (
    MetricsConfig,
    MetricsError,
    ProgressSample,
    ProgressSnapshot,
    ProgressTracker,
)


def test_metrics_config_window_s_le_zero() -> None:
    with pytest.raises(ValueError, match="window_s must be > 0.0"):
        MetricsConfig(window_s=0.0)
    with pytest.raises(ValueError, match="window_s must be > 0.0"):
        MetricsConfig(window_s=-10.0)


def test_metrics_config_sample_interval_s_le_zero() -> None:
    with pytest.raises(ValueError, match="sample_interval_s must be > 0.0"):
        MetricsConfig(sample_interval_s=0.0)
    with pytest.raises(ValueError, match="sample_interval_s must be > 0.0"):
        MetricsConfig(sample_interval_s=-1.0)


def test_metrics_config_window_lt_sample_interval() -> None:
    with pytest.raises(ValueError, match="must be >= sample_interval_s"):
        MetricsConfig(window_s=5.0, sample_interval_s=10.0)


def test_metrics_config_min_samples_for_rate_lt_2() -> None:
    with pytest.raises(
        ValueError, match="min_samples_for_rate must be an integer >= 2"
    ):
        MetricsConfig(min_samples_for_rate=1)
    with pytest.raises(
        ValueError, match="min_samples_for_rate must be an integer >= 2"
    ):
        MetricsConfig(min_samples_for_rate=0)


def test_metrics_config_non_finite_float() -> None:
    with pytest.raises(ValueError, match="window_s must be a finite float"):
        MetricsConfig(window_s=float("inf"))
    with pytest.raises(ValueError, match="sample_interval_s must be a finite float"):
        MetricsConfig(sample_interval_s=float("nan"))


def test_progress_sample_negative_ts() -> None:
    with pytest.raises(ValueError, match="ts must be finite and >= 0.0"):
        ProgressSample(
            ts=-1.0,
            position=(0.0, 0.0),
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )


def test_progress_sample_malformed_position() -> None:
    with pytest.raises(
        ValueError, match="position must be a tuple of two finite floats"
    ):
        ProgressSample(
            ts=1.0,
            position=(0.0, float("nan")),
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )
    with pytest.raises(
        ValueError, match="position must be a tuple of two finite floats"
    ):
        ProgressSample(
            ts=1.0,
            position=(0.0,),  # type: ignore[arg-type]
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )


def test_progress_sample_negative_inventory_count() -> None:
    with pytest.raises(ValueError, match="inventory_count must be an integer >= 0"):
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=-1,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )


def test_progress_sample_negative_level_or_xp() -> None:
    with pytest.raises(ValueError, match="level_or_xp must be finite and >= 0.0"):
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=10,
            level_or_xp=-0.5,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )


def test_progress_sample_negative_successful_actions_total() -> None:
    with pytest.raises(
        ValueError, match="successful_actions_total must be an integer >= 0"
    ):
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=-1,
            reflex_ticks_total=10,
        )


def test_progress_sample_negative_reflex_ticks_total() -> None:
    with pytest.raises(
        ValueError, match="reflex_ticks_total must be an integer >= 0"
    ):
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=-5,
        )


def test_snapshot_empty_tracker() -> None:
    tracker = ProgressTracker()
    assert tracker.sample_count == 0
    assert tracker.window_start_ts is None
    assert tracker.window_end_ts is None

    snap = tracker.snapshot()
    assert snap.sample_count == 0
    assert snap.window_start_ts is None
    assert snap.window_end_ts is None
    assert snap.position_delta == 0.0
    assert snap.inventory_delta == 0
    assert snap.level_or_xp_delta == 0.0
    assert snap.successful_actions_per_minute == 0.0
    assert snap.reflex_tick_rate_hz == 0.0
    assert snap.is_rate_reliable is False


def test_snapshot_single_sample() -> None:
    tracker = ProgressTracker()
    sample = ProgressSample(
        ts=10.0,
        position=(100.0, 200.0),
        inventory_count=5,
        level_or_xp=10.0,
        successful_actions_total=2,
        reflex_ticks_total=20,
    )
    tracker.update(sample)
    assert tracker.sample_count == 1
    assert tracker.window_start_ts == 10.0
    assert tracker.window_end_ts == 10.0

    snap = tracker.snapshot()
    assert snap.sample_count == 1
    assert snap.window_start_ts == 10.0
    assert snap.window_end_ts == 10.0
    assert snap.position_delta == 0.0
    assert snap.inventory_delta == 0
    assert snap.level_or_xp_delta == 0.0
    assert snap.successful_actions_per_minute == 0.0
    assert snap.reflex_tick_rate_hz == 0.0
    assert snap.is_rate_reliable is False


def test_update_rejects_non_monotone_ts() -> None:
    tracker = ProgressTracker()
    tracker.update(
        ProgressSample(
            ts=10.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    with pytest.raises(MetricsError, match="strictly greater"):
        tracker.update(
            ProgressSample(
                ts=10.0,
                position=(1.0, 1.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )
    with pytest.raises(MetricsError, match="strictly greater"):
        tracker.update(
            ProgressSample(
                ts=9.0,
                position=(1.0, 1.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )


def test_update_rejects_decreasing_successful_actions() -> None:
    tracker = ProgressTracker()
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=10,
            reflex_ticks_total=0,
        )
    )
    with pytest.raises(MetricsError, match="successful_actions_total decreased"):
        tracker.update(
            ProgressSample(
                ts=2.0,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=9,
                reflex_ticks_total=0,
            )
        )


def test_update_rejects_decreasing_reflex_ticks() -> None:
    tracker = ProgressTracker()
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=20,
        )
    )
    with pytest.raises(MetricsError, match="reflex_ticks_total decreased"):
        tracker.update(
            ProgressSample(
                ts=2.0,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=19,
            )
        )


def test_update_rejects_decreasing_level_or_xp() -> None:
    tracker = ProgressTracker()
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=100.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    with pytest.raises(MetricsError, match="level_or_xp decreased"):
        tracker.update(
            ProgressSample(
                ts=2.0,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=99.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )


def test_sliding_window_eviction() -> None:
    """Inserting 100 samples spanning 100 s with window_s=60.

    Sample times: ts = 1.0, 2.0, ..., 100.0.
    When ts = 100.0, cutoff is 100.0 - 60.0 = 40.0.
    Samples with ts < 40.0 (ts = 1.0 ... 39.0, i.e., 39 samples) are pruned.
    Samples remaining: ts = 40.0 ... 100.0 (exact count = 61).
    Note: The ring buffer maxlen is ceil(60/1) + 1 = 61, perfectly matching this expectation.
    """
    config = MetricsConfig(window_s=60.0, sample_interval_s=1.0)
    tracker = ProgressTracker(config=config)

    for i in range(1, 101):
        ts = float(i)
        tracker.update(
            ProgressSample(
                ts=ts,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )

    assert tracker.sample_count == 61
    assert tracker.window_start_ts == 40.0
    assert tracker.window_end_ts == 100.0


def test_position_delta_accumulates_distance_not_displacement() -> None:
    """Agent moves in a triangle (0,0) -> (3,0) -> (3,4) -> (0,0).

    Leg 1: (0,0) to (3,0) = 3.0
    Leg 2: (3,0) to (3,4) = 4.0
    Leg 3: (3,4) to (0,0) = 5.0
    Total distance = 12.0. Displacement would be 0.0.
    """
    tracker = ProgressTracker(config=MetricsConfig(min_samples_for_rate=2))
    positions = [(0.0, 0.0), (3.0, 0.0), (3.0, 4.0), (0.0, 0.0)]
    for idx, pos in enumerate(positions):
        tracker.update(
            ProgressSample(
                ts=float(idx + 1),
                position=pos,
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )

    snap = tracker.snapshot()
    assert math.isclose(snap.position_delta, 12.0)


def test_position_delta_stationary() -> None:
    tracker = ProgressTracker(config=MetricsConfig(min_samples_for_rate=2))
    for i in range(1, 6):
        tracker.update(
            ProgressSample(
                ts=float(i),
                position=(50.0, 75.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )

    snap = tracker.snapshot()
    assert snap.position_delta == 0.0


def test_inventory_delta() -> None:
    tracker = ProgressTracker(config=MetricsConfig(min_samples_for_rate=2))
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=5,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    tracker.update(
        ProgressSample(
            ts=2.0,
            position=(0.0, 0.0),
            inventory_count=12,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )

    snap = tracker.snapshot()
    assert snap.inventory_delta == 7


def test_level_or_xp_delta() -> None:
    tracker = ProgressTracker(config=MetricsConfig(min_samples_for_rate=2))
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=100.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    tracker.update(
        ProgressSample(
            ts=2.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=250.5,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )

    snap = tracker.snapshot()
    assert math.isclose(snap.level_or_xp_delta, 150.5)


def test_successful_actions_per_minute_below_threshold() -> None:
    config = MetricsConfig(min_samples_for_rate=5)
    tracker = ProgressTracker(config=config)

    for i in range(1, 5):
        tracker.update(
            ProgressSample(
                ts=float(i),
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=i * 10,
                reflex_ticks_total=0,
            )
        )

    snap = tracker.snapshot()
    assert snap.sample_count == 4
    assert snap.is_rate_reliable is False
    assert snap.successful_actions_per_minute == 0.0


def test_successful_actions_per_minute_known_series() -> None:
    """5 samples over dt = 60s. Actions total increases by 10.

    10 actions / 60 s * 60 = 10.0 actions per minute.
    """
    config = MetricsConfig(min_samples_for_rate=5)
    tracker = ProgressTracker(config=config)

    for i in range(5):
        ts = float(i * 15)  # 0, 15, 30, 45, 60
        tracker.update(
            ProgressSample(
                ts=ts,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=int(i * 2.5),
                reflex_ticks_total=0,
            )
        )

    snap = tracker.snapshot()
    assert snap.sample_count == 5
    assert snap.is_rate_reliable is True
    assert math.isclose(snap.successful_actions_per_minute, 10.0)


def test_reflex_tick_rate_hz_known_series() -> None:
    """5 samples over dt = 10s. Reflex ticks total increases from 0 to 100.

    Rate = 100 / 10 = 10.0 Hz.
    """
    config = MetricsConfig(min_samples_for_rate=5)
    tracker = ProgressTracker(config=config)

    for i in range(5):
        ts = float(i * 2.5)  # 0.0, 2.5, 5.0, 7.5, 10.0
        tracker.update(
            ProgressSample(
                ts=ts,
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=i * 25,
            )
        )

    snap = tracker.snapshot()
    assert snap.sample_count == 5
    assert snap.is_rate_reliable is True
    assert math.isclose(snap.reflex_tick_rate_hz, 10.0)


def test_is_rate_reliable_conditions() -> None:
    config = MetricsConfig(min_samples_for_rate=3)
    tracker = ProgressTracker(config=config)

    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    tracker.update(
        ProgressSample(
            ts=2.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    assert tracker.snapshot().is_rate_reliable is False

    tracker.update(
        ProgressSample(
            ts=3.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
    )
    assert tracker.snapshot().is_rate_reliable is True


def test_snapshot_non_mutating() -> None:
    tracker = ProgressTracker()
    for i in range(1, 4):
        tracker.update(
            ProgressSample(
                ts=float(i),
                position=(float(i), float(i)),
                inventory_count=i,
                level_or_xp=float(i * 10),
                successful_actions_total=i,
                reflex_ticks_total=i * 5,
            )
        )

    s1 = tracker.snapshot()
    s2 = tracker.snapshot()
    assert s1 == s2


def test_snapshot_determinism() -> None:
    def run_sequence() -> ProgressSnapshot:
        t = ProgressTracker(config=MetricsConfig(min_samples_for_rate=3))
        for i in range(1, 6):
            t.update(
                ProgressSample(
                    ts=float(i * 2),
                    position=(float(i), float(i * 2)),
                    inventory_count=i * 2,
                    level_or_xp=float(i * 50),
                    successful_actions_total=i * 3,
                    reflex_ticks_total=i * 10,
                )
            )
        return t.snapshot()

    snap1 = run_sequence()
    snap2 = run_sequence()
    assert snap1 == snap2
    assert snap1.to_json() == snap2.to_json()


def test_reset_clears_buffer() -> None:
    tracker = ProgressTracker()
    for i in range(1, 5):
        tracker.update(
            ProgressSample(
                ts=float(i),
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )
    assert tracker.sample_count == 4

    tracker.reset()
    assert tracker.sample_count == 0
    assert tracker.window_start_ts is None
    assert tracker.window_end_ts is None

    empty_snap = tracker.snapshot()
    assert empty_snap.sample_count == 0
    assert empty_snap.is_rate_reliable is False

    tracker.reset()
    assert tracker.sample_count == 0


def test_progress_snapshot_to_json() -> None:
    tracker = ProgressTracker(config=MetricsConfig(min_samples_for_rate=2))
    tracker.update(
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=2,
            level_or_xp=10.0,
            successful_actions_total=1,
            reflex_ticks_total=10,
        )
    )
    tracker.update(
        ProgressSample(
            ts=3.0,
            position=(3.0, 4.0),
            inventory_count=5,
            level_or_xp=20.0,
            successful_actions_total=3,
            reflex_ticks_total=30,
        )
    )

    snap = tracker.snapshot()
    d = snap.to_json()

    serialized = json.dumps(d)
    deserialized = json.loads(serialized)

    assert deserialized == {
        "window_s": 60.0,
        "sample_count": 2,
        "window_start_ts": 1.0,
        "window_end_ts": 3.0,
        "position_delta": 5.0,
        "inventory_delta": 3,
        "level_or_xp_delta": 10.0,
        "successful_actions_per_minute": 60.0,
        "reflex_tick_rate_hz": 10.0,
        "is_rate_reliable": True,
    }


def test_ring_buffer_bounded_capacity() -> None:
    config = MetricsConfig(window_s=10.0, sample_interval_s=1.0)
    tracker = ProgressTracker(config=config)

    for i in range(1, 10_001):
        tracker.update(
            ProgressSample(
                ts=float(i),
                position=(0.0, 0.0),
                inventory_count=0,
                level_or_xp=0.0,
                successful_actions_total=0,
                reflex_ticks_total=0,
            )
        )
        assert tracker.sample_count <= 11

    assert tracker.sample_count == 11
    assert tracker.window_start_ts == 9990.0
    assert tracker.window_end_ts == 10000.0


def test_static_ast_isolation() -> None:
    metrics_path = Path("src/wow_bot/watchdog/metrics.py")
    assert metrics_path.exists(), "src/wow_bot/watchdog/metrics.py does not exist"

    tree = ast.parse(metrics_path.read_text(encoding="utf-8"))

    prohibited_modules = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
    }
    prohibited_keywords = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert (
                    mod not in prohibited_modules
                ), f"Prohibited module imported: {mod}"
                assert not any(
                    kw in mod.lower() for kw in prohibited_keywords
                ), f"Prohibited LLM keyword in import: {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert (
                mod not in prohibited_modules
            ), f"Prohibited module imported: {mod}"
            assert not any(
                kw in mod.lower() for kw in prohibited_keywords
            ), f"Prohibited LLM keyword in import: {mod}"

        if (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            assert node.attr not in (
                "monotonic",
                "time",
                "perf_counter",
            ), f"Forbidden time function called: time.{node.attr}"
