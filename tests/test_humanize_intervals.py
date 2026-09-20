import ast
import dataclasses
import math
import random
import statistics
from pathlib import Path

import pytest

from wow_bot.humanize.intervals import (
    IntervalConfig,
    IntervalError,
    cdf,
    is_in_support,
    logpdf,
    sample_interval,
    theoretical_cv,
    theoretical_mean,
    theoretical_median,
)


# Minimal Kolmogorov-Smirnov test against Uniform(0, 1) using standard library
# and standard asymptotic / precomputed critical values.
def ks_test_uniform(samples: list[float]) -> tuple[float, float]:
    """Calculates KS statistic D and asymptotic p-value for testing if samples ~ Uniform(0, 1).

    Uses stdlib math/statistics to avoid third-party dependencies (e.g. scipy).
    Formula for asymptotic p-value uses Kolmogorov distribution approximation:
        P(D > d) ~ 2 * sum_{k=1}^inf (-1)^{k-1} exp(-2 k^2 x^2) where x = sqrt(n) * d.
    """
    n = len(samples)
    sorted_samples = sorted(samples)

    d_max = 0.0
    for i, x in enumerate(sorted_samples):
        # Sample CDF step at x is (i+1)/n, and before x is i/n.
        # Theoretical CDF for Uniform(0,1) at x is x.
        d_plus = (i + 1) / n - x
        d_minus = x - i / n
        d_max = max(d_max, d_plus)
        d_max = max(d_max, d_minus)

    # Asymptotics for Kolmogorov distribution
    x = (math.sqrt(n) + 0.12 + 0.11 / math.sqrt(n)) * d_max
    # Approximating series for P(K > x)
    p_val = 0.0
    for k in range(1, 101):
        term = 2.0 * ((-1) ** (k - 1)) * math.exp(-2.0 * (k**2) * (x**2))
        p_val += term
        if abs(term) < 1e-10:
            break

    p_val = max(0.0, min(1.0, p_val))
    return d_max, p_val


# 1. IntervalConfig validation tests
def test_config_clip_low_invalid() -> None:
    with pytest.raises(ValueError, match="clip_low must be > 0.0"):
        IntervalConfig(clip_low=0.0)
    with pytest.raises(ValueError, match="clip_low must be > 0.0"):
        IntervalConfig(clip_low=-0.01)


def test_config_clip_high_invalid() -> None:
    with pytest.raises(ValueError, match="clip_high must be > clip_low"):
        IntervalConfig(clip_low=0.5, clip_high=0.5)
    with pytest.raises(ValueError, match="clip_high must be > clip_low"):
        IntervalConfig(clip_low=0.5, clip_high=0.4)


def test_config_sigma_invalid() -> None:
    with pytest.raises(ValueError, match="sigma must be > 0.0"):
        IntervalConfig(sigma=0.0)
    with pytest.raises(ValueError, match="sigma must be > 0.0"):
        IntervalConfig(sigma=-0.5)


def test_config_max_rejection_attempts_invalid() -> None:
    with pytest.raises(ValueError, match="max_rejection_attempts must be >= 1"):
        IntervalConfig(max_rejection_attempts=0)


def test_config_mu_invalid() -> None:
    with pytest.raises(ValueError, match="mu must be finite"):
        IntervalConfig(mu=float("inf"))
    with pytest.raises(ValueError, match="mu must be finite"):
        IntervalConfig(mu=float("-inf"))
    with pytest.raises(ValueError, match="mu must be finite"):
        IntervalConfig(mu=float("nan"))


# 2. sample_interval tests
def test_sample_interval_bounds() -> None:
    rng = random.Random(42)
    config = IntervalConfig()
    for _ in range(10_000):
        val = sample_interval(rng, config)
        assert config.clip_low <= val <= config.clip_high


def test_sample_interval_determinism() -> None:
    rng1 = random.Random(12345)
    rng2 = random.Random(12345)
    config = IntervalConfig()

    seq1 = [sample_interval(rng1, config) for _ in range(100)]
    seq2 = [sample_interval(rng2, config) for _ in range(100)]
    assert seq1 == seq2


def test_sample_interval_seed_sensitivity() -> None:
    rng1 = random.Random(100)
    rng2 = random.Random(200)
    config = IntervalConfig()

    seq1 = [sample_interval(rng1, config) for _ in range(100)]
    seq2 = [sample_interval(rng2, config) for _ in range(100)]
    assert seq1 != seq2


def test_sample_interval_rejection_exceeded() -> None:
    rng = random.Random(0)
    # Impossible range given distribution mu=0.0, sigma=0.001
    config = IntervalConfig(
        mu=0.0,
        sigma=0.001,
        clip_low=100.0,
        clip_high=200.0,
        max_rejection_attempts=50,
    )
    with pytest.raises(IntervalError, match="Exhausted max_rejection_attempts"):
        sample_interval(rng, config)


# 3. CDF tests
def test_cdf_below_clip_low() -> None:
    config = IntervalConfig()
    assert cdf(0.0, config) == 0.0
    assert cdf(0.01, config) == 0.0
    assert cdf(-1.0, config) == 0.0


def test_cdf_above_clip_high() -> None:
    config = IntervalConfig()
    assert cdf(1.5, config) == 1.0
    assert cdf(2.0, config) == 1.0
    assert cdf(100.0, config) == 1.0


def test_cdf_monotonicity() -> None:
    config = IntervalConfig()
    xs = [config.clip_low + i * 0.01 for i in range(140)]
    cdf_vals = [cdf(x, config) for x in xs]
    for i in range(len(cdf_vals) - 1):
        assert cdf_vals[i] < cdf_vals[i + 1]


def test_cdf_exact_endpoints() -> None:
    config = IntervalConfig()
    assert cdf(config.clip_low, config) == 0.0
    assert cdf(config.clip_high, config) == 1.0


def test_cdf_at_empirical_median() -> None:
    rng = random.Random(999)
    config = IntervalConfig()
    samples = [sample_interval(rng, config) for _ in range(20_000)]
    emp_median = statistics.median(samples)

    cdf_at_median = cdf(emp_median, config)
    assert abs(cdf_at_median - 0.5) < 0.05


# 4. PIT & KS Test
def test_pit_ks_test() -> None:
    rng = random.Random(777)
    config = IntervalConfig()
    n_samples = 10_000
    samples = [sample_interval(rng, config) for _ in range(n_samples)]
    pit_transformed = [cdf(x, config) for x in samples]

    d_stat, p_value = ks_test_uniform(pit_transformed)
    assert p_value > 0.05, f"PIT KS test failed with p-value={p_value}, D={d_stat}"


def test_pit_ks_determinism() -> None:
    # Verify two runs with same seed and config produce identical p-values
    def run_test(seed: int) -> float:
        r = random.Random(seed)
        cfg = IntervalConfig()
        s = [sample_interval(r, cfg) for _ in range(1000)]
        pit = [cdf(x, cfg) for x in s]
        _, p = ks_test_uniform(pit)
        return p

    p1 = run_test(4242)
    p2 = run_test(4242)
    assert p1 == p2


# 5. Theoretical vs Empirical statistics tests
def test_empirical_cv_default_config() -> None:
    rng = random.Random(31415)
    config = IntervalConfig()
    samples = [sample_interval(rng, config) for _ in range(10_000)]
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples)
    emp_cv = stdev / mean
    assert emp_cv > 0.3


def test_theoretical_cv_matches_empirical() -> None:
    rng = random.Random(27182)
    config = IntervalConfig()
    samples = [sample_interval(rng, config) for _ in range(50_000)]
    mean = statistics.mean(samples)
    stdev = statistics.stdev(samples)
    emp_cv = stdev / mean

    theo_cv = theoretical_cv(config)
    # Truncation reduces CV slightly; check within 25%
    assert abs(emp_cv - theo_cv) / theo_cv < 0.25


def test_theoretical_mean_greater_than_median() -> None:
    config = IntervalConfig()
    assert theoretical_mean(config) > theoretical_median(config)


def test_theoretical_median_equals_exp_mu() -> None:
    config = IntervalConfig(mu=-1.5)
    assert theoretical_median(config) == math.exp(-1.5)


# 6. logpdf tests
def test_logpdf_outside_support() -> None:
    config = IntervalConfig()
    assert logpdf(0.0, config) == float("-inf")
    assert logpdf(0.01, config) == float("-inf")
    assert logpdf(1.6, config) == float("-inf")


def test_logpdf_inside_support() -> None:
    config = IntervalConfig()
    val = logpdf(0.2, config)
    assert math.isfinite(val)


def test_logpdf_maximal_near_mode() -> None:
    rng = random.Random(888)
    config = IntervalConfig()
    samples = [sample_interval(rng, config) for _ in range(10_000)]
    emp_median = statistics.median(samples)

    logpdf_med = logpdf(emp_median, config)
    logpdf_low = logpdf(config.clip_low + 1e-6, config)
    logpdf_high = logpdf(config.clip_high - 1e-6, config)

    assert logpdf_med > logpdf_low
    assert logpdf_med > logpdf_high


# 7. is_in_support tests
def test_is_in_support() -> None:
    config = IntervalConfig(clip_low=0.1, clip_high=2.0)
    assert is_in_support(0.1, config) is True
    assert is_in_support(2.0, config) is True
    assert is_in_support(1.0, config) is True
    assert is_in_support(0.099, config) is False
    assert is_in_support(2.001, config) is False


# 8. Dataclass frozen test
def test_dataclass_frozen() -> None:
    config = IntervalConfig()
    assert dataclasses.is_dataclass(config)
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.mu = 0.0  # type: ignore[misc]


# 9. Static AST checks
def test_static_ast_no_forbidden_imports() -> None:
    intervals_path = Path("src/wow_bot/humanize/intervals.py")
    tree = ast.parse(intervals_path.read_text(), filename=str(intervals_path))

    forbidden_exact = {
        "aiosqlite",
        "asyncio",
        "threading",
        "numpy",
    }
    forbidden_prefixes = (
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.session",
    )
    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in forbidden_exact, f"Forbidden import: {mod}"
                assert not any(
                    mod.startswith(p) for p in forbidden_prefixes
                ), f"Forbidden import prefix: {mod}"
                assert not any(
                    s in mod.lower() for s in forbidden_substrings
                ), f"Forbidden import substring: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_exact, f"Forbidden import: {mod}"
            assert not any(
                mod.startswith(p) for p in forbidden_prefixes
            ), f"Forbidden import prefix: {mod}"
            assert not any(
                s in mod.lower() for s in forbidden_substrings
            ), f"Forbidden import substring: {mod}"


def test_static_ast_no_time_reads() -> None:
    intervals_path = Path("src/wow_bot/humanize/intervals.py")
    tree = ast.parse(intervals_path.read_text(), filename=str(intervals_path))

    forbidden_attrs = {"monotonic", "time", "perf_counter"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            assert (
                node.attr not in forbidden_attrs
            ), f"Forbidden time function access: {node.attr}"
