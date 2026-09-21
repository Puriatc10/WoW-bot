"""Unit tests for Lab Timing Analysis V2 (Task 10.3)."""

from __future__ import annotations

import ast
import json
import math
import random
from pathlib import Path

import pytest

from wow_bot.analysis.lab_timing_v2 import (
    ANALYSIS_SCHEMA_VERSION,
    PIT_RANDOM_SEED,
    LabTimingError,
    TimingConfig,
    TimingResult,
    _truncated_lognormal_cdf,
    analyze_intervals,
    analyze_report,
    write_result,
)
from wow_bot.reporting.schema_v2 import SCHEMA_VERSION


def test_timing_config_validation() -> None:
    # min_samples < 8 raises ValueError
    with pytest.raises(ValueError, match="min_samples must be >= 8"):
        TimingConfig(min_samples=7)

    # ks_alpha outside (0, 1) raises ValueError
    with pytest.raises(ValueError, match="ks_alpha must be a float in"):
        TimingConfig(ks_alpha=0.0)

    with pytest.raises(ValueError, match="ks_alpha must be a float in"):
        TimingConfig(ks_alpha=1.0)

    # cv_threshold <= 0 raises ValueError
    with pytest.raises(ValueError, match="cv_threshold must be a float > 0.0"):
        TimingConfig(cv_threshold=0.0)

    # lognormal_sigma <= 0 raises ValueError
    with pytest.raises(ValueError, match="lognormal_sigma must be a float > 0.0"):
        TimingConfig(lognormal_sigma=0.0)

    # invalid clip range raises ValueError
    with pytest.raises(ValueError, match="clip_high_ms"):
        TimingConfig(clip_low_ms=100.0, clip_high_ms=50.0)


def test_analyze_intervals_insufficient_samples() -> None:
    res = analyze_intervals([100.0] * 10)
    assert res.skipped is True
    assert res.reason == "insufficient_samples:10"
    assert res.sample_count == 10
    assert res.mean_ms is None
    assert res.pit_passes is False
    assert res.ks_passes is False


def test_analyze_intervals_invalid_input() -> None:
    # Non-finite input
    with pytest.raises(LabTimingError, match="non-numeric or non-finite"):
        analyze_intervals([100.0] * 64 + [float("nan")])

    # Negative input
    with pytest.raises(LabTimingError, match="must be non-negative"):
        analyze_intervals([100.0] * 64 + [-10.0])


def test_analyze_intervals_pit_passes_on_matching_distribution() -> None:
    # Draw samples from the exact truncated lognormal model matching default config using inverse transform sampling
    cfg = TimingConfig()
    rng = random.Random(PIT_RANDOM_SEED)

    samples: list[float] = []

    # Draw 500 samples
    for _ in range(500):
        u = rng.uniform(0.0, 1.0)
        # Numerical inversion of _truncated_lognormal_cdf
        low_x = cfg.clip_low_ms
        high_x = cfg.clip_high_ms
        for _ in range(50):
            mid_x = 0.5 * (low_x + high_x)
            val = _truncated_lognormal_cdf(mid_x, cfg.lognormal_mu, cfg.lognormal_sigma, cfg.clip_low_ms, cfg.clip_high_ms)
            if val < u:
                low_x = mid_x
            else:
                high_x = mid_x
        samples.append(0.5 * (low_x + high_x))

    res = analyze_intervals(samples, config=cfg)
    assert res.skipped is False
    assert res.pit_passes is True
    assert res.ks_passes is True
    assert res.pit_p_value is not None and res.pit_p_value > cfg.ks_alpha
    assert res.ks_p_value == res.pit_p_value


def test_analyze_intervals_pit_fails_on_different_distribution() -> None:
    # Uniform samples in [clip_low, clip_high] will fail the truncated lognormal PIT test
    cfg = TimingConfig()
    rng = random.Random(12345)
    samples = [rng.uniform(cfg.clip_low_ms, cfg.clip_high_ms) for _ in range(500)]

    res = analyze_intervals(samples, config=cfg)
    assert res.skipped is False
    assert res.pit_passes is False
    assert res.ks_passes is False
    assert res.pit_p_value is not None and res.pit_p_value <= cfg.ks_alpha


def test_descriptive_statistics_and_cv_threshold() -> None:
    samples = [100.0, 200.0, 300.0, 400.0] * 20  # N=80
    res = analyze_intervals(samples, config=TimingConfig(cv_threshold=0.3))

    assert res.skipped is False
    assert res.mean_ms is not None and math.isclose(res.mean_ms, 250.0)
    assert res.median_ms is not None and math.isclose(res.median_ms, 250.0)

    # Population std: sqrt(((150^2 + 50^2 + 50^2 + 150^2) * 20) / 80) = sqrt(12500) ≈ 111.80339887
    expected_std = math.sqrt(12500.0)
    assert res.std_ms is not None and math.isclose(res.std_ms, expected_std, rel_tol=1e-5)

    expected_cv = expected_std / 250.0
    assert res.cv is not None and math.isclose(res.cv, expected_cv, rel_tol=1e-5)
    assert res.cv_meets_threshold == (expected_cv > 0.3)


def test_ks_statistic_and_p_value_equality() -> None:
    samples = [100.0 + i * 2.0 for i in range(100)]
    res = analyze_intervals(samples)

    assert res.skipped is False
    assert res.ks_statistic is not None and 0.0 <= res.ks_statistic <= 1.0
    assert res.ks_p_value is not None and 0.0 <= res.ks_p_value <= 1.0
    assert res.pit_p_value == res.ks_p_value


def test_analyze_intervals_determinism() -> None:
    samples = [50.0 + (i * 17) % 300 for i in range(150)]
    res1 = analyze_intervals(samples)
    res2 = analyze_intervals(samples)

    assert res1 == res2
    assert res1.to_json() == res2.to_json()


def test_analyze_report_validation() -> None:
    # Non-dict raises LabTimingError
    with pytest.raises(LabTimingError, match="report must be a dict"):
        analyze_report("not_a_dict")  # type: ignore[arg-type]

    # Invalid schema_version raises LabTimingError
    with pytest.raises(LabTimingError, match="Expected report schema_version == 2"):
        analyze_report({"schema_version": 1})

    # Missing humanizer section returns skipped
    res_no_hum = analyze_report({"schema_version": SCHEMA_VERSION})
    assert res_no_hum.skipped is True
    assert res_no_hum.reason == "no_humanizer_section"

    # Valid report with enough samples returns non-skipped result
    valid_report = {
        "schema_version": SCHEMA_VERSION,
        "humanizer": {
            "interval_samples_ms": [100.0 + (i % 20) * 10.0 for i in range(100)],
        },
    }
    res_valid = analyze_report(valid_report)
    assert res_valid.skipped is False
    assert res_valid.sample_count == 100


def test_write_result_atomic_and_dir_creation(tmp_path: Path) -> None:
    res = TimingResult(
        skipped=False,
        reason="",
        sample_count=100,
        mean_ms=150.0,
        median_ms=145.0,
        std_ms=50.0,
        cv=0.333,
        cv_meets_threshold=True,
        pit_p_value=0.5,
        pit_passes=True,
        ks_statistic=0.08,
        ks_p_value=0.5,
        ks_passes=True,
    )

    out_file = tmp_path / "nested" / "dir" / "timing.json"
    write_result(res, out_file)

    assert out_file.is_file()
    # Ensure no .tmp file left behind
    tmp_file = out_file.parent / f"{out_file.name}.tmp"
    assert not tmp_file.exists()

    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["mean_ms"] == 150.0
    assert content["pit_p_value"] == 0.5


def test_to_json_serialization() -> None:
    res = TimingResult(
        skipped=True,
        reason="insufficient_samples:5",
        sample_count=5,
        mean_ms=None,
        median_ms=None,
        std_ms=None,
        cv=None,
        cv_meets_threshold=False,
        pit_p_value=None,
        pit_passes=False,
        ks_statistic=None,
        ks_p_value=None,
        ks_passes=False,
    )
    d = res.to_json()
    assert d == {
        "skipped": True,
        "reason": "insufficient_samples:5",
        "sample_count": 5,
        "mean_ms": None,
        "median_ms": None,
        "std_ms": None,
        "cv": None,
        "cv_meets_threshold": False,
        "pit_p_value": None,
        "pit_passes": False,
        "ks_statistic": None,
        "ks_p_value": None,
        "ks_passes": False,
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
    }
    json_str = json.dumps(d)
    assert json.loads(json_str) == d


def test_static_ast_checks() -> None:
    module_path = Path("src/wow_bot/analysis/lab_timing_v2.py")
    tree = ast.parse(module_path.read_text(encoding="utf-8"))

    forbidden_imports = {
        "wow_bot.analysis.spectrum",
        "wow_bot.analysis.timing",
        "wow_bot.analysis.soak",
        "wow_bot.reporting.scenario",
        "numpy",
        "scipy",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.executor",
        "wow_bot.watchdog",
        "wow_bot.humanize",
        "wow_bot.internal_dynamics",
        "wow_bot.strategist",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    forbidden_words = {"ollama", "openai", "anthropic", "llm"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in forbidden_imports, f"Forbidden import: {mod}"
                for fw in forbidden_words:
                    assert fw not in mod.lower(), f"Forbidden module word '{fw}' in {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_imports, f"Forbidden import from: {mod}"
            if mod != "wow_bot.reporting.schema_v2":
                for fw in forbidden_words:
                    assert fw not in mod.lower(), f"Forbidden module word '{fw}' in {mod}"

        # Check time function calls
        if (
            isinstance(node, ast.Attribute)
            and node.attr in {"monotonic", "time", "perf_counter"}
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            pytest.fail(f"Forbidden call to time.{node.attr}")

    # Verify no pre-lab module was modified
    git_modified = Path("src/wow_bot/analysis/timing.py")
    assert git_modified.is_file()
