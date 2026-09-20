"""Unit tests for stochastic timing and error simulation in humanize.py (Task 5.2).

Tests satisfy all requirements A through Z from the Task 5.2 prompt:
  - Default delay generation
  - Deterministic seeded sequence reproducibility and seed divergence
  - 1000-sample mean acceptance
  - Kolmogorov-Smirnov goodness-of-fit statistical test vs log-normal distribution
  - Delay clipping bounds [50, 2000]
  - Validation of base_ms, fatigue, chaos_component, drive vector, radius, and coordinates
  - Behavioral monotonicity for fatigue and curiosity
  - Exact error probability bounds [0.02, 0.10]
  - Coordinate jitter bounds, radius zero determinism, and symmetry
  - Module purity checks (no sleeping, no OS/controller interaction imports)
"""

from __future__ import annotations

import sys

import numpy as np
import pytest
from scipy.stats import kstest  # type: ignore[import-untyped]

from wow_bot.executor.humanize import (
    MAX_DELAY_MS,
    MIN_DELAY_MS,
    human_delay,
    human_error_probability,
    jitter_coordinates,
)

# ============================================================================
# Requirement A & F: Default Delay & Clipping Bounds
# ============================================================================


def test_human_delay_default_call() -> None:
    """Test A: Call human_delay with defaults returns a finite float within [50, 2000]."""
    delay = human_delay()
    assert isinstance(delay, float)
    assert MIN_DELAY_MS <= delay <= MAX_DELAY_MS


def test_human_delay_clipping_bounds() -> None:
    """Test F: Extreme parameters yield values strictly clipped to [50.0, 2000.0]."""
    rng = np.random.default_rng(123)
    # Generate large sample under extreme parameters to exercise lower and upper clip
    low_delays = [
        human_delay(base_ms=10, fatigue=0.0, chaos_component=-1.0, rng=rng)
        for _ in range(500)
    ]
    high_delays = [
        human_delay(base_ms=5000, fatigue=1.0, chaos_component=1.0, rng=rng)
        for _ in range(500)
    ]

    for d in low_delays + high_delays:
        assert isinstance(d, float)
        assert MIN_DELAY_MS <= d <= MAX_DELAY_MS

    # Verify that clipping at boundaries actually took place
    assert any(d == MIN_DELAY_MS for d in low_delays)
    assert any(d == MAX_DELAY_MS for d in high_delays)


# ============================================================================
# Requirement B & C: Seeded Reproducibility and Divergence
# ============================================================================


def test_human_delay_deterministic_seeded_sequence() -> None:
    """Test B: Identical RNG seeds produce identical delay sequences."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(42)

    seq_a = [human_delay(rng=rng_a) for _ in range(100)]
    seq_b = [human_delay(rng=rng_b) for _ in range(100)]

    assert seq_a == seq_b


def test_human_delay_different_seeds_diverge() -> None:
    """Test C: Different RNG seeds produce different delay sequences."""
    rng_a = np.random.default_rng(42)
    rng_b = np.random.default_rng(999)

    seq_a = [human_delay(rng=rng_a) for _ in range(10)]
    seq_b = [human_delay(rng=rng_b) for _ in range(10)]

    assert seq_a != seq_b


# ============================================================================
# Requirement D & E: 1000-Sample Baseline Mean & Statistical KS Distribution
# ============================================================================


def test_human_delay_sample_mean_baseline() -> None:
    """Test D: 1000-sample mean for baseline config is within [150, 300] ms."""
    fixed_seed = 42
    rng = np.random.default_rng(fixed_seed)
    samples = [
        human_delay(base_ms=200, fatigue=0.5, chaos_component=0.0, rng=rng)
        for _ in range(1000)
    ]

    mean_val = float(np.mean(samples))
    assert 150.0 <= mean_val <= 300.0


def test_human_delay_ks_distribution_check() -> None:
    """Check the clipped model using all samples and the fixed randomized PIT seed."""
    from wow_bot.analysis.timing import compute_randomized_pit

    rng = np.random.default_rng(42)
    detailed = [
        {
            "delay_ms": human_delay(200, 0.5, 0.0, rng=rng),
            "base_ms": 200, "fatigue": 0.5, "chaos_component": 0.0,
        }
        for _ in range(1000)
    ]
    pit = compute_randomized_pit(detailed)
    assert len(pit) == 1000
    assert kstest(pit, "uniform").pvalue > 0.05


@pytest.mark.parametrize("invalid_base", [0, -1, -200, 1.5, "200", True, False, None])
def test_human_delay_invalid_base_ms(invalid_base: object) -> None:
    """Test G: Reject non-integer, non-positive, or boolean base_ms."""
    with pytest.raises(ValueError, match="base_ms"):
        human_delay(base_ms=invalid_base)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    "invalid_fatigue", [-0.01, 1.01, 2.0, -1.0, float("nan"), float("inf"), float("-inf"), True, False, "0.5"]
)
def test_human_delay_invalid_fatigue(invalid_fatigue: object) -> None:
    """Test H: Reject out-of-range, non-finite, boolean, or string fatigue."""
    with pytest.raises(ValueError, match="fatigue"):
        human_delay(fatigue=invalid_fatigue)  # type: ignore[arg-type]


def test_human_delay_fatigue_boundaries() -> None:
    """Test H: Accept fatigue boundaries 0.0 and 1.0."""
    d0 = human_delay(fatigue=0.0)
    d1 = human_delay(fatigue=1.0)
    assert MIN_DELAY_MS <= d0 <= MAX_DELAY_MS
    assert MIN_DELAY_MS <= d1 <= MAX_DELAY_MS


@pytest.mark.parametrize(
    "invalid_chaos", [-1.01, 1.01, 2.0, float("nan"), float("inf"), float("-inf"), True, False, "0.0"]
)
def test_human_delay_invalid_chaos(invalid_chaos: object) -> None:
    """Test I: Reject out-of-range, non-finite, boolean, or string chaos_component."""
    with pytest.raises(ValueError, match="chaos_component"):
        human_delay(chaos_component=invalid_chaos)  # type: ignore[arg-type]


def test_human_delay_chaos_boundaries() -> None:
    """Test I: Accept chaos_component boundaries -1.0, 0.0, and 1.0."""
    d_min = human_delay(chaos_component=-1.0)
    d_mid = human_delay(chaos_component=0.0)
    d_max = human_delay(chaos_component=1.0)
    assert MIN_DELAY_MS <= d_min <= MAX_DELAY_MS
    assert MIN_DELAY_MS <= d_mid <= MAX_DELAY_MS
    assert MIN_DELAY_MS <= d_max <= MAX_DELAY_MS


# ============================================================================
# Requirement J & K: Fatigue & Chaos Sensitivity / Distribution Spread
# ============================================================================


def test_human_delay_fatigue_changes_spread() -> None:
    """Test J: Higher fatigue produces greater standard deviation / spread."""
    rng_a = np.random.default_rng(100)
    rng_b = np.random.default_rng(100)

    samples_low_fatigue = [
        human_delay(base_ms=200, fatigue=0.0, chaos_component=0.0, rng=rng_a)
        for _ in range(2000)
    ]
    samples_high_fatigue = [
        human_delay(base_ms=200, fatigue=1.0, chaos_component=0.0, rng=rng_b)
        for _ in range(2000)
    ]

    std_low = float(np.std(samples_low_fatigue))
    std_high = float(np.std(samples_high_fatigue))

    assert std_high > std_low


def test_human_delay_chaos_changes_spread() -> None:
    """Test K: Positive chaos produces greater standard deviation / spread than negative chaos."""
    rng_a = np.random.default_rng(200)
    rng_b = np.random.default_rng(200)

    samples_neg_chaos = [
        human_delay(base_ms=200, fatigue=0.5, chaos_component=-1.0, rng=rng_a)
        for _ in range(2000)
    ]
    samples_pos_chaos = [
        human_delay(base_ms=200, fatigue=0.5, chaos_component=1.0, rng=rng_b)
        for _ in range(2000)
    ]

    std_neg = float(np.std(samples_neg_chaos))
    std_pos = float(np.std(samples_pos_chaos))

    assert std_pos > std_neg


# ============================================================================
# Requirement L through S: Error Probability Formula & Validation
# ============================================================================


def test_human_error_probability_canonical_formula() -> None:
    """Test L: Exact output for mid-range vector [0.5, 0.5, 0.5, 0.5, 0.5]."""
    vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5], dtype=np.float64)
    expected = 0.02 + 0.05 * 0.5 + 0.03 * (1.0 - 0.5)  # 0.06
    prob = human_error_probability(vec)
    assert prob == pytest.approx(expected)


def test_human_error_probability_minimum() -> None:
    """Test M: Minimum error probability when fatigue=0 and curiosity=1."""
    vec = np.array([0.5, 0.0, 1.0, 0.5, 0.5], dtype=np.float64)
    expected = 0.02 + 0.05 * 0.0 + 0.03 * (1.0 - 1.0)  # 0.02
    assert human_error_probability(vec) == pytest.approx(expected)


def test_human_error_probability_maximum() -> None:
    """Test N: Maximum error probability when fatigue=1 and curiosity=0."""
    vec = np.array([0.5, 1.0, 0.0, 0.5, 0.5], dtype=np.float64)
    expected = 0.02 + 0.05 * 1.0 + 0.03 * (1.0 - 0.0)  # 0.10
    prob = human_error_probability(vec)
    assert prob == pytest.approx(expected)
    assert prob <= 0.15  # Satisfies roadmap upper bound


def test_human_error_probability_global_range() -> None:
    """Test O: All drive boundary combinations lie within [0.02, 0.15]."""
    for fatigue in [0.0, 0.5, 1.0]:
        for curiosity in [0.0, 0.5, 1.0]:
            vec = np.array([0.5, fatigue, curiosity, 0.5, 0.5], dtype=np.float64)
            p = human_error_probability(vec)
            assert 0.02 <= p <= 0.15


def test_human_error_probability_fatigue_monotonicity() -> None:
    """Test P: Increasing fatigue increases error probability monotonic ally."""
    curiosity = 0.5
    p_low = human_error_probability(np.array([0.0, 0.1, curiosity, 0.0, 0.0]))
    p_mid = human_error_probability(np.array([0.0, 0.5, curiosity, 0.0, 0.0]))
    p_high = human_error_probability(np.array([0.0, 0.9, curiosity, 0.0, 0.0]))

    assert p_low < p_mid < p_high


def test_human_error_probability_curiosity_monotonicity() -> None:
    """Test Q: Decreasing curiosity increases error probability monotonically."""
    fatigue = 0.5
    p_curiosity_high = human_error_probability(np.array([0.0, fatigue, 0.9, 0.0, 0.0]))
    p_curiosity_mid = human_error_probability(np.array([0.0, fatigue, 0.5, 0.0, 0.0]))
    p_curiosity_low = human_error_probability(np.array([0.0, fatigue, 0.1, 0.0, 0.0]))

    assert p_curiosity_high < p_curiosity_mid < p_curiosity_low


@pytest.mark.parametrize(
    "bad_shape",
    [
        np.array([0.5, 0.5, 0.5, 0.5]),
        np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5]),
        np.array([[0.5, 0.5, 0.5, 0.5, 0.5]]),
        np.array([]),
    ],
)
def test_human_error_probability_malformed_shape(bad_shape: np.ndarray) -> None:
    """Test R: Reject drive vectors with shape != (5,)."""
    with pytest.raises(ValueError, match="shape"):
        human_error_probability(bad_shape)


@pytest.mark.parametrize(
    "invalid_vec",
    [
        np.array([-0.1, 0.5, 0.5, 0.5, 0.5]),
        np.array([0.5, 1.1, 0.5, 0.5, 0.5]),
        np.array([0.5, np.nan, 0.5, 0.5, 0.5]),
        np.array([0.5, float("inf"), 0.5, 0.5, 0.5]),
        "not an array",
    ],
)
def test_human_error_probability_invalid_values(invalid_vec: object) -> None:
    """Test S: Reject out-of-bounds, NaN, Inf, or non-ndarray drive inputs."""
    with pytest.raises(ValueError):
        human_error_probability(invalid_vec)  # type: ignore[arg-type]


# ============================================================================
# Requirement T through X: Coordinate Jitter
# ============================================================================


def test_jitter_coordinates_radius_zero() -> None:
    """Test T & 27: Radius zero returns exact original input coordinates deterministically."""
    res = jitter_coordinates(100, 200, radius=0)
    assert res == (100, 200)


def test_jitter_coordinates_bounds_and_symmetry() -> None:
    """Test U & 30: All perturbed coordinates remain strictly within abs(diff) <= radius."""
    x_orig, y_orig = 500, 300
    radius = 5
    rng = np.random.default_rng(42)

    x_offsets: list[int] = []
    y_offsets: list[int] = []

    for _ in range(500):
        jx, jy = jitter_coordinates(x_orig, y_orig, radius=radius, rng=rng)
        dx = jx - x_orig
        dy = jy - y_orig

        assert abs(dx) <= radius
        assert abs(dy) <= radius

        x_offsets.append(dx)
        y_offsets.append(dy)

    # Symmetry check over 500 samples: positive, negative, and zero offsets occur
    assert any(dx > 0 for dx in x_offsets)
    assert any(dx < 0 for dx in x_offsets)
    assert any(dx == 0 for dx in x_offsets)


def test_jitter_coordinates_seeded_reproducibility() -> None:
    """Test V: Identical seed produces same sequence; different seed diverges."""
    rng_a = np.random.default_rng(123)
    rng_b = np.random.default_rng(123)
    rng_c = np.random.default_rng(999)

    seq_a = [jitter_coordinates(10, 20, radius=10, rng=rng_a) for _ in range(50)]
    seq_b = [jitter_coordinates(10, 20, radius=10, rng=rng_b) for _ in range(50)]
    seq_c = [jitter_coordinates(10, 20, radius=10, rng=rng_c) for _ in range(50)]

    assert seq_a == seq_b
    assert seq_a != seq_c


@pytest.mark.parametrize("invalid_radius", [-1, -5, 2.5, "5", True, False, None])
def test_jitter_coordinates_invalid_radius(invalid_radius: object) -> None:
    """Test W: Reject negative, float, string, or boolean radius."""
    with pytest.raises(ValueError, match="radius"):
        jitter_coordinates(10, 20, radius=invalid_radius)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("bad_x", "bad_y"),
    [
        (1.5, 10),
        (10, 2.5),
        ("100", 200),
        (100, "200"),
        (True, 100),
        (100, False),
    ],
)
def test_jitter_coordinates_invalid_coords(bad_x: object, bad_y: object) -> None:
    """Test X: Reject non-integer and boolean coordinates; accept negative integers."""
    with pytest.raises(ValueError):
        jitter_coordinates(bad_x, bad_y)  # type: ignore[arg-type]


def test_jitter_coordinates_negative_integers_allowed() -> None:
    """Test X: Negative integer coordinates are valid."""
    res = jitter_coordinates(-100, -200, radius=5)
    assert isinstance(res[0], int)
    assert isinstance(res[1], int)
    assert abs(res[0] - (-100)) <= 5
    assert abs(res[1] - (-200)) <= 5


# ============================================================================
# Requirement Y & Z: Module Purity and Independence
# ============================================================================


def test_humanize_no_sleeping_or_time_waits() -> None:
    """Test Y: Ensure humanize.py does not call asyncio.sleep or time.sleep."""
    import wow_bot.executor.humanize as hum_mod

    code = hum_mod.__file__
    assert code is not None
    with open(code, "r", encoding="utf-8") as f:
        content = f.read()

    assert "asyncio.sleep" not in content
    assert "time.sleep" not in content
    assert "import time" not in content


def test_humanize_no_input_control_coupling() -> None:
    """Test Z: Importing humanize module imports zero OS/input control backends."""
    import wow_bot.executor.humanize  # noqa: F401

    forbidden_modules = {"pynput", "pyautogui", "keyboard", "mouse"}
    loaded_modules = set(sys.modules.keys())

    assert forbidden_modules.isdisjoint(loaded_modules)
