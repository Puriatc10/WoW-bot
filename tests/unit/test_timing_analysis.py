"""Unit tests for Task 8.2 Timing Distribution Analysis (tests/unit/test_timing_analysis.py).

Tests requirements A through R for timing distribution analysis, randomized PIT,
KS testing, CV calculation, legacy/enhanced report handling, and strict JSON output.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.analysis.timing import (
    CV_THRESHOLD,
    LOWER_DELAY_MS,
    PIT_RANDOM_SEED,
    UPPER_DELAY_MS,
    build_timing_analysis,
    compute_descriptive_statistics,
    compute_randomized_pit,
    load_timing_report,
    plot_distribution,
    run_conditional_ks,
    validate_timing_samples,
    write_analysis_output,
)
from wow_bot.executor.humanize import delay_distribution_parameters
from wow_bot.reporting.scenario import SCENARIO_REPORT_SCHEMA_VERSION


def _make_valid_report_dict(
    sample_count: int = 150,
    with_metadata: bool = True,
    base_ms: int = 200,
    fatigue: float = 0.5,
    chaos_component: float = 0.0,
    seed: int = 42,
) -> dict[str, Any]:
    """Helper to generate a valid synthetic scenario report dict for testing."""
    rng = np.random.default_rng(seed)
    mu, sigma = delay_distribution_parameters(base_ms, fatigue, chaos_component)

    raw_delays = rng.lognormal(mean=mu, sigma=sigma, size=sample_count)
    clipped_delays = np.clip(raw_delays, LOWER_DELAY_MS, UPPER_DELAY_MS)

    samples_ms = [float(x) for x in clipped_delays]
    timing_dict: dict[str, Any] = {
        "unit": "ms",
        "model": "lognormal_simulation",
        "samples_ms": samples_ms,
    }

    if with_metadata:
        detailed = []
        for idx, d in enumerate(samples_ms):
            detailed.append({
                "simulation_timestamp": float(100.0 + idx * 0.1),
                "delay_ms": d,
                "base_ms": base_ms,
                "fatigue": fatigue,
                "chaos_component": chaos_component,
            })
        timing_dict["samples"] = detailed

    return {
        "schema_version": SCENARIO_REPORT_SCHEMA_VERSION,
        "run": {
            "scenario": "combat_light",
            "seed": seed,
            "requested_duration_seconds": 600.0,
            "completed_duration_seconds": 600.0,
            "completed_normally": True,
            "error_type": None,
        },
        "meta_state": {
            "dimensions": ["hunger", "fatigue", "curiosity", "aggression", "social"],
            "samples": [
                {"simulation_timestamp": float(i), "vector": [0.5] * 5}
                for i in range(sample_count)
            ],
        },
        "timing": timing_dict,
    }


# Required Test A — report loading
def test_load_report_success(tmp_path: Path) -> None:
    """Test A: Valid Task 7.2 schema report loads successfully."""
    report_data = _make_valid_report_dict()
    report_file = tmp_path / "valid_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report_data, f)

    loaded = load_timing_report(report_file)
    assert loaded["schema_version"] == SCENARIO_REPORT_SCHEMA_VERSION
    assert loaded["run"]["scenario"] == "combat_light"


# Required Test B — wrong schema
def test_load_report_wrong_schema(tmp_path: Path) -> None:
    """Test B: Incompatible or wrong schema version is rejected clearly."""
    report_data = _make_valid_report_dict()
    report_data["schema_version"] = 999
    report_file = tmp_path / "bad_schema.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report_data, f)

    with pytest.raises(ValueError, match="Unsupported schema_version"):
        load_timing_report(report_file)


# Required Test C — malformed delays
@pytest.mark.parametrize(
    "invalid_sample",
    [
        float("nan"),
        float("inf"),
        -10.0,
        0.0,
        49.9,   # Below LOWER_DELAY_MS 50
        2000.1, # Above UPPER_DELAY_MS 2000
    ],
)
def test_validate_malformed_delays(invalid_sample: float) -> None:
    """Test C: Reject NaN, Infinity, <=0, or outside clipping bounds."""
    timing_dict = {
        "samples_ms": [100.0, invalid_sample, 200.0],
    }
    with pytest.raises(ValueError):
        validate_timing_samples(timing_dict)


# Required Test D — malformed metadata
@pytest.mark.parametrize(
    ("key", "invalid_value"),
    [
        ("base_ms", 0),
        ("base_ms", -100),
        ("base_ms", 100.5),
        ("fatigue", -0.1),
        ("fatigue", 1.1),
        ("chaos_component", -1.1),
        ("chaos_component", 1.1),
        ("simulation_timestamp", float("nan")),
    ],
)
def test_validate_malformed_metadata(key: str, invalid_value: Any) -> None:
    """Test D: Reject invalid base_ms, fatigue, chaos_component, simulation_timestamp."""
    sample_item: dict[str, Any] = {
        "simulation_timestamp": 100.0,
        "delay_ms": 200.0,
        "base_ms": 200,
        "fatigue": 0.5,
        "chaos_component": 0.0,
    }
    sample_item[key] = invalid_value
    timing_dict = {
        "samples_ms": [200.0],
        "samples": [sample_item],
    }
    with pytest.raises(ValueError):
        validate_timing_samples(timing_dict)


# Required Test E — CV formula
def test_cv_formula_exact() -> None:
    """Test E: Verify std(ddof=1) / mean on known deterministic vector."""
    delays = [100.0, 200.0, 300.0, 400.0, 500.0]
    stats = compute_descriptive_statistics(delays)

    arr = np.array(delays, dtype=np.float64)
    expected_mean = float(np.mean(arr))
    expected_std = float(np.std(arr, ddof=1))
    expected_cv = expected_std / expected_mean

    assert stats["mean_ms"] == pytest.approx(expected_mean)
    assert stats["std_ms"] == pytest.approx(expected_std)
    assert stats["cv"] == pytest.approx(expected_cv)


# Required Test F — CV threshold semantics
def test_cv_threshold_semantics() -> None:
    """Test F: Verify CV threshold semantics (exclusive > 0.3)."""
    # Helper to check acceptance
    def check_cv(cv: float) -> bool:
        return cv > CV_THRESHOLD

    assert check_cv(0.3000001) is True
    assert check_cv(0.3) is False
    assert check_cv(0.2999999) is False


# Required Test G — interior PIT
def test_interior_pit_known_cdf() -> None:
    """Test G: Interior sample u_i = F(delay) for known lognormal parameters."""
    base_ms = 200
    fatigue = 0.5
    chaos = 0.0
    delay_ms = 250.0  # Interior (50 < 250 < 2000)

    detailed_samples = [
        {
            "simulation_timestamp": 100.0,
            "delay_ms": delay_ms,
            "base_ms": base_ms,
            "fatigue": fatigue,
            "chaos_component": chaos,
        }
    ]

    pit_vals = compute_randomized_pit(detailed_samples, pit_seed=8202)
    assert len(pit_vals) == 1

    mu, sigma = delay_distribution_parameters(base_ms, fatigue, chaos)
    from scipy.stats import lognorm  # type: ignore[import-untyped]
    expected_u = float(lognorm.cdf(delay_ms, s=sigma, scale=math.exp(mu), loc=0.0))

    assert pit_vals[0] == pytest.approx(expected_u)


# Required Test H — lower-bound PIT
def test_lower_bound_pit() -> None:
    """Test H: Lower clipped sample (50.0) mapping lies in [0, F(LOWER)]."""
    base_ms = 200
    fatigue = 0.5
    chaos = 0.0
    delay_ms = 50.0

    detailed_samples = [
        {
            "simulation_timestamp": 100.0,
            "delay_ms": delay_ms,
            "base_ms": base_ms,
            "fatigue": fatigue,
            "chaos_component": chaos,
        }
    ]

    pit_vals = compute_randomized_pit(detailed_samples, pit_seed=8202)
    assert len(pit_vals) == 1

    mu, sigma = delay_distribution_parameters(base_ms, fatigue, chaos)
    from scipy.stats import lognorm
    f_lower = float(lognorm.cdf(LOWER_DELAY_MS, s=sigma, scale=math.exp(mu), loc=0.0))

    assert 0.0 <= pit_vals[0] <= f_lower


# Required Test I — upper-bound PIT
def test_upper_bound_pit() -> None:
    """Test I: Upper clipped sample (2000.0) mapping lies in [F(UPPER), 1]."""
    base_ms = 200
    fatigue = 0.5
    chaos = 0.0
    delay_ms = 2000.0

    detailed_samples = [
        {
            "simulation_timestamp": 100.0,
            "delay_ms": delay_ms,
            "base_ms": base_ms,
            "fatigue": fatigue,
            "chaos_component": chaos,
        }
    ]

    pit_vals = compute_randomized_pit(detailed_samples, pit_seed=8202)
    assert len(pit_vals) == 1

    mu, sigma = delay_distribution_parameters(base_ms, fatigue, chaos)
    from scipy.stats import lognorm
    f_upper = float(lognorm.cdf(UPPER_DELAY_MS, s=sigma, scale=math.exp(mu), loc=0.0))

    assert f_upper <= pit_vals[0] <= 1.0


# Required Test J — deterministic PIT
def test_deterministic_pit_reproducibility() -> None:
    """Test J: Same input + same PIT seed -> identical PIT sequence and KS result."""
    report = _make_valid_report_dict(sample_count=150, seed=123)
    detailed_samples = report["timing"]["samples"]

    pit_1 = compute_randomized_pit(detailed_samples, pit_seed=PIT_RANDOM_SEED)
    ks_stat_1, p_val_1 = run_conditional_ks(pit_1)

    pit_2 = compute_randomized_pit(detailed_samples, pit_seed=PIT_RANDOM_SEED)
    ks_stat_2, p_val_2 = run_conditional_ks(pit_2)

    assert pit_1 == pit_2
    assert ks_stat_1 == ks_stat_2
    assert p_val_1 == p_val_2


# Required Test K — no seed searching
def test_no_seed_searching_constant() -> None:
    """Test K: Verify fixed seed constant PIT_RANDOM_SEED = 8202 used by default."""
    assert PIT_RANDOM_SEED == 8202


# Required Test L — KS helper
def test_ks_helper_uniform() -> None:
    """Test L: Verify run_conditional_ks on approximately uniform PIT values."""
    rng = np.random.default_rng(42)
    uniform_pit = list(rng.uniform(0.0, 1.0, size=200))

    ks_stat, p_val = run_conditional_ks(uniform_pit)
    assert math.isfinite(ks_stat)
    assert math.isfinite(p_val)
    assert p_val > 0.05


# Required Test M — clipping statistics
def test_clipping_statistics() -> None:
    """Test M: Verify lower/upper clip counts and total clipped fraction."""
    delays = [50.0, 50.0, 100.0, 200.0, 2000.0]
    stats = compute_descriptive_statistics(delays)

    assert stats["lower_clip_count"] == 2
    assert stats["upper_clip_count"] == 1
    assert stats["clipped_fraction"] == pytest.approx(3 / 5)


# Required Test N — legacy report
def test_legacy_report_handling() -> None:
    """Test N: Legacy report with samples_ms only produces descriptive stats, marks KS unavailable, doesn't crash."""
    report = _make_valid_report_dict(sample_count=150, with_metadata=False)

    analysis = build_timing_analysis(report)
    assert analysis["sample_summary"]["count"] == 150
    assert analysis["conditional_model_test"]["status"] == "unavailable"
    assert "per-sample model metadata missing" in analysis["conditional_model_test"]["reason"]
    assert analysis["roadmap_acceptance"]["overall_within_target"] is None


# Required Test O — enhanced report
def test_enhanced_report_handling() -> None:
    """Test O: Enhanced report with full metadata produces conditional PIT/KS output."""
    report = _make_valid_report_dict(sample_count=150, with_metadata=True)

    analysis = build_timing_analysis(report)
    assert analysis["sample_summary"]["count"] == 150
    assert analysis["conditional_model_test"]["status"] == "available"
    assert analysis["conditional_model_test"]["method"] == "randomized_pit_ks_uniform"
    assert math.isfinite(analysis["conditional_model_test"]["p_value"])
    assert isinstance(analysis["roadmap_acceptance"]["overall_within_target"], bool)


# Required Test P — strict JSON
def test_strict_json_no_nan_inf(tmp_path: Path) -> None:
    """Test P: Generated analysis JSON contains no NaN or Infinity."""
    report = _make_valid_report_dict(sample_count=150, with_metadata=True)
    analysis = build_timing_analysis(report)

    output_path = tmp_path / "timing_analysis.json"
    write_analysis_output(analysis, output_path)

    # Re-read raw text and parse strictly
    with output_path.open("r", encoding="utf-8") as f:
        raw_text = f.read()

    assert "NaN" not in raw_text
    assert "Infinity" not in raw_text

    loaded = json.loads(raw_text)
    assert loaded["analysis_schema_version"] == 1


# Required Test Q — input report unchanged
def test_input_report_unchanged(tmp_path: Path) -> None:
    """Test Q: Analyzer must not mutate/rewrite source report file."""
    report = _make_valid_report_dict(sample_count=150, with_metadata=True)
    report_file = tmp_path / "original_report.json"
    with report_file.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    original_bytes = report_file.read_bytes()

    # Perform analysis & write output to output_dir
    output_dir = tmp_path / "analysis_out"
    loaded_report = load_timing_report(report_file)
    delays, detailed = validate_timing_samples(loaded_report["timing"])
    analysis = build_timing_analysis(loaded_report)

    write_analysis_output(analysis, output_dir / "timing_analysis.json")
    plot_distribution(analysis, delays, detailed, output_dir)

    # Check input file remains byte-for-byte unchanged
    assert report_file.read_bytes() == original_bytes


# Required Test R — minimum sample handling
def test_minimum_sample_handling() -> None:
    """Test R: Below MIN_TIMING_SAMPLES (100), statistical acceptance is set to insufficient_samples."""
    report = _make_valid_report_dict(sample_count=50, with_metadata=True)

    analysis = build_timing_analysis(report, min_samples=100)
    cond = analysis["conditional_model_test"]
    assert cond["status"] == "insufficient_samples"
    assert "below minimum threshold of 100" in cond["reason"]
    assert analysis["roadmap_acceptance"]["overall_within_target"] is None
