"""Unit tests for position-based stuck detection signal source."""

import ast
from pathlib import Path

import pytest

from wow_bot.reflex.stuck import PositionStuckSignalSource, StuckConfig


class FakePositionSource:
    """Helper fake position source returning a sequence of positions."""

    def __init__(self, positions: list[tuple[float, float]]) -> None:
        self._positions = list(positions)
        self.call_count = 0

    def __call__(self) -> tuple[float, float]:
        self.call_count += 1
        if self._positions:
            if len(self._positions) == 1:
                return self._positions[0]
            return self._positions.pop(0)
        return (0.0, 0.0)


def test_invalid_config_parameters_raise_value_error() -> None:
    """Constructing StuckConfig with invalid field values raises ValueError."""
    with pytest.raises(ValueError, match="window_s must be > 0"):
        StuckConfig(window_s=0.0)

    with pytest.raises(ValueError, match="window_s must be > 0"):
        StuckConfig(window_s=-1.0)

    with pytest.raises(ValueError, match="threshold_units must be > 0"):
        StuckConfig(threshold_units=0.0)

    with pytest.raises(ValueError, match="threshold_units must be > 0"):
        StuckConfig(threshold_units=-0.5)

    with pytest.raises(ValueError, match="sample_interval_s must be > 0"):
        StuckConfig(sample_interval_s=0.0)

    with pytest.raises(ValueError, match="sample_interval_s must be > 0"):
        StuckConfig(sample_interval_s=-0.1)

    with pytest.raises(ValueError, match="min_samples must be >= 2"):
        StuckConfig(min_samples=1)

    with pytest.raises(ValueError, match="min_samples must be >= 2"):
        StuckConfig(min_samples=0)


def test_sample_interval_exceeds_window_raises_value_error() -> None:
    """Constructing StuckConfig with sample_interval_s > window_s raises ValueError."""
    with pytest.raises(ValueError, match="cannot exceed window_s"):
        StuckConfig(window_s=2.0, sample_interval_s=2.5)


def test_first_poll_returns_empty_and_calls_position_source_once() -> None:
    """First poll returns [] and calls position_source exactly once."""
    fake = FakePositionSource([(10.0, 10.0)])
    source = PositionStuckSignalSource(fake)

    signals = source.poll(100.0)
    assert signals == []
    assert fake.call_count == 1


def test_poll_within_sample_interval_does_not_call_source() -> None:
    """Polling within sample_interval_s of the last sample does not call position_source again."""
    fake = FakePositionSource([(10.0, 10.0), (10.0, 10.0)])
    config = StuckConfig(sample_interval_s=0.5)
    source = PositionStuckSignalSource(fake, config=config)

    source.poll(100.0)
    assert fake.call_count == 1

    signals = source.poll(100.2)
    assert signals == []
    assert fake.call_count == 1


def test_poll_after_sample_interval_calls_source_again() -> None:
    """Polling after sample_interval_s calls position_source again."""
    fake = FakePositionSource([(10.0, 10.0), (10.0, 10.0)])
    config = StuckConfig(sample_interval_s=0.5)
    source = PositionStuckSignalSource(fake, config=config)

    source.poll(100.0)
    assert fake.call_count == 1

    source.poll(100.5)
    assert fake.call_count == 2


def test_fewer_than_min_samples_returns_empty() -> None:
    """With fewer than min_samples samples, poll returns []."""
    fake = FakePositionSource([(10.0, 10.0)] * 10)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    for i in range(3):
        signals = source.poll(now + i * 0.5)
        assert signals == []
        assert source.sample_count == i + 1


def test_stationary_positions_emit_position_stuck_once() -> None:
    """Stationary positions emit exactly one 'position_stuck' signal once min_samples are reached."""
    fake = FakePositionSource([(10.0, 10.0)] * 10)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5, window_s=3.0, threshold_units=1.0)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    assert source.poll(now) == []
    assert source.poll(now + 0.5) == []
    assert source.poll(now + 1.0) == []

    stuck_ts = now + 1.5
    signals = source.poll(stuck_ts)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.name == PositionStuckSignalSource.NAME_STUCK
    assert sig.ts == stuck_ts
    assert sig.payload == {"spread": 0.0, "window_s": 3.0, "samples": 4}
    assert source.is_stuck is True
    assert source.last_spread == 0.0


def test_subsequent_stuck_polls_do_not_re_emit() -> None:
    """Subsequent polls while remaining stuck return [] (no duplicate emission)."""
    fake = FakePositionSource([(10.0, 10.0)] * 10)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    for i in range(4):
        source.poll(now + i * 0.5)

    assert source.is_stuck is True

    for i in range(4, 8):
        signals = source.poll(now + i * 0.5)
        assert signals == []
        assert source.is_stuck is True


def test_moving_positions_emit_position_clear() -> None:
    """When positions start moving again (spread > threshold), poll emits exactly one 'position_clear' signal."""
    positions = [(10.0, 10.0)] * 4 + [(15.0, 10.0)]
    fake = FakePositionSource(positions)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5, threshold_units=1.0)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    for i in range(4):
        source.poll(now + i * 0.5)

    assert source.is_stuck is True

    signals = source.poll(now + 2.0)
    assert len(signals) == 1
    sig = signals[0]
    assert sig.name == PositionStuckSignalSource.NAME_CLEAR
    assert sig.ts == now + 2.0
    assert sig.payload == {"spread": 5.0}
    assert source.is_stuck is False
    assert source.last_spread == 5.0

    # Subsequent moving poll emits []
    assert source.poll(now + 2.5) == []


def test_new_stuck_episode_after_position_clear() -> None:
    """After 'position_clear', a new stuck episode emits 'position_stuck' again."""
    positions = [(10.0, 10.0)] * 4 + [(50.0, 50.0)] * 5
    fake = FakePositionSource(positions)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5, window_s=2.0, threshold_units=1.0)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    # First episode: 4 stationary samples at (10, 10) at t=100.0, 100.5, 101.0, 101.5
    for i in range(4):
        source.poll(now + i * 0.5)
    assert source.is_stuck is True

    # Move to (50, 50) at t=102.0 -> position_clear
    clear_signals = source.poll(now + 2.0)
    assert len(clear_signals) == 1
    assert clear_signals[0].name == PositionStuckSignalSource.NAME_CLEAR
    assert source.is_stuck is False

    # Additional samples at (50, 50) at t=102.5, 103.0, 103.5, 104.0
    # At t=104.0, cutoff = 104.0 - 2.0 = 102.0. All (10, 10) samples (<102.0) are pruned.
    source.poll(now + 2.5)
    source.poll(now + 3.0)
    source.poll(now + 3.5)

    stuck_signals = source.poll(now + 4.0)
    assert len(stuck_signals) == 1
    assert stuck_signals[0].name == PositionStuckSignalSource.NAME_STUCK
    assert source.is_stuck is True


def test_position_stuck_payload_and_timestamp() -> None:
    """Payload of 'position_stuck' contains 'spread', 'window_s', and 'samples' with correct values, and 'ts' equals poll now."""
    positions = [(10.0, 10.0), (10.2, 10.1), (10.1, 10.2), (10.3, 10.0)]
    fake = FakePositionSource(positions)
    config = StuckConfig(window_s=2.0, threshold_units=1.0, sample_interval_s=0.5, min_samples=4)
    source = PositionStuckSignalSource(fake, config=config)

    now = 50.0
    source.poll(now)
    source.poll(now + 0.5)
    source.poll(now + 1.0)

    poll_ts = now + 1.5
    signals = source.poll(poll_ts)

    assert len(signals) == 1
    sig = signals[0]
    assert sig.ts == poll_ts
    assert "spread" in sig.payload
    assert "window_s" in sig.payload
    assert "samples" in sig.payload
    assert sig.payload["window_s"] == 2.0
    assert sig.payload["samples"] == 4
    assert sig.payload["spread"] <= 1.0


def test_samples_older_than_window_s_are_dropped() -> None:
    """Samples older than window_s are dropped: moving samples age out when stationary."""
    # window_s = 2.0, sample_interval_s = 0.5, min_samples = 4
    # Positions: moving initially at t=0.0, 0.5, 1.0, 1.5
    # Then staying at (100, 100) at t=2.0, 2.5, 3.0, 3.5, 4.0, 4.5
    positions = [
        (0.0, 0.0),
        (10.0, 0.0),
        (20.0, 0.0),
        (30.0, 0.0),
        (100.0, 100.0),
        (100.0, 100.0),
        (100.0, 100.0),
        (100.0, 100.0),
        (100.0, 100.0),
    ]
    fake = FakePositionSource(positions)
    config = StuckConfig(window_s=2.0, threshold_units=1.0, sample_interval_s=0.5, min_samples=4)
    source = PositionStuckSignalSource(fake, config=config)

    # Initial moving phase: window [0.0..1.5] has high spread
    for i in range(4):
        assert source.poll(i * 0.5) == []
        assert source.is_stuck is False

    # At t=2.0, position is (100, 100). Old sample t=0.0 is dropped (cutoff = 2.0 - 2.0 = 0.0).
    # Spread is still large between t=0.5 (10,0) and t=2.0 (100,100).
    assert source.poll(2.0) == []
    assert source.is_stuck is False

    # More samples at (100, 100)
    assert source.poll(2.5) == []
    assert source.poll(3.0) == []
    assert source.poll(3.5) == []

    # At t=4.0 (cutoff = 2.0), all remaining samples in deque are from t>=2.0 at (100, 100).
    signals = source.poll(4.0)
    assert len(signals) == 1
    assert signals[0].name == PositionStuckSignalSource.NAME_STUCK
    assert source.is_stuck is True


def test_position_source_exception_propagates() -> None:
    """If position_source raises, poll propagates the exception."""

    def raising_source() -> tuple[float, float]:
        raise RuntimeError("Position sensor failure")

    source = PositionStuckSignalSource(raising_source)
    with pytest.raises(RuntimeError, match="Position sensor failure"):
        source.poll(100.0)


def test_reset_clears_state_and_starts_new_episode() -> None:
    """reset() clears sample_count, is_stuck, and last_spread, and subsequent polls start a new episode."""
    fake = FakePositionSource([(10.0, 10.0)] * 10)
    config = StuckConfig(min_samples=4, sample_interval_s=0.5)
    source = PositionStuckSignalSource(fake, config=config)

    now = 100.0
    for i in range(4):
        source.poll(now + i * 0.5)

    assert source.is_stuck is True
    assert source.sample_count == 4
    assert source.last_spread == 0.0

    source.reset()

    assert source.is_stuck is False
    assert source.sample_count == 0
    assert source.last_spread is None

    # Subsequent poll immediately after reset acts like first poll
    signals = source.poll(now + 2.5)
    assert signals == []
    assert source.sample_count == 1


def test_determinism_across_instances() -> None:
    """Two PositionStuckSignalSource instances with the same inputs produce identical signal sequences."""
    positions = [
        (0.0, 0.0),
        (0.1, 0.0),
        (0.0, 0.1),
        (0.1, 0.1),  # Stuck
        (0.1, 0.1),  # Still stuck
        (10.0, 10.0),  # Clear
        (10.0, 10.0),
        (10.0, 10.0),
        (10.0, 10.0),
        (10.0, 10.0),  # Stuck again
    ]

    fake1 = FakePositionSource(positions)
    fake2 = FakePositionSource(positions)

    config = StuckConfig(window_s=2.0, threshold_units=0.5, sample_interval_s=0.5, min_samples=4)

    src1 = PositionStuckSignalSource(fake1, config=config)
    src2 = PositionStuckSignalSource(fake2, config=config)

    timestamps = [10.0 + i * 0.5 for i in range(len(positions))]

    seq1 = [src1.poll(ts) for ts in timestamps]
    seq2 = [src2.poll(ts) for ts in timestamps]

    assert seq1 == seq2


def test_static_ast_import_isolation() -> None:
    """Static AST check: stuck.py does not import wow_bot.strategist, executor, navigation, combat, world, perception, or LLM modules."""
    stuck_path = Path(__file__).parent.parent / "src" / "wow_bot" / "reflex" / "stuck.py"
    content = stuck_path.read_text(encoding="utf-8")
    tree = ast.parse(content, filename=str(stuck_path))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.executor",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }
    forbidden_substrings = {"ollama", "openai", "anthropic", "llm"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for forbidden in forbidden_modules:
                    assert not mod_name.startswith(
                        forbidden
                    ), f"Forbidden import found in stuck.py: {mod_name}"
                for sub in forbidden_substrings:
                    assert (
                        sub not in mod_name.lower()
                    ), f"Forbidden LLM substring found in stuck.py import: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod_name.startswith(
                    forbidden
                ), f"Forbidden from-import found in stuck.py: {mod_name}"
            for sub in forbidden_substrings:
                assert (
                    sub not in mod_name.lower()
                ), f"Forbidden LLM substring found in stuck.py import: {mod_name}"
