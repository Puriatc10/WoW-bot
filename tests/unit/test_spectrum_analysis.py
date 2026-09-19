"""Unit tests for Scenario Report Spectrum Analysis (Task 8.1)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.analysis.spectrum import (
    ANALYSIS_SCHEMA_VERSION,
    MIN_SPECTRUM_SAMPLES,
    SpectrumAnalysisError,
    analyze_spectrum,
    compute_dimension_spectrum,
    load_and_validate_scenario_report,
    plot_psd,
    resample_uniform,
    write_analysis_json,
)

# Ensure root scripts directory is importable
_scripts_dir = str(Path(__file__).resolve().parent.parent.parent / "scripts")
if _scripts_dir not in sys.path:
    sys.path.insert(0, _scripts_dir)

from analyze_spectrum import main as cli_main  # type: ignore[import-not-found]


def _make_valid_report_dict(
    sample_count: int = 200,
    dt: float = 0.1,
    scenario: str = "combat_light",
    seed: int = 42,
    vector_fn: Any = None,
) -> dict[str, Any]:
    """Helper to generate a valid Task 7.2 scenario report dictionary."""
    samples = []
    for i in range(sample_count):
        ts = i * dt
        if vector_fn is not None:
            vec = vector_fn(i, ts)
        else:
            # Default mild variation
            v_val = 0.5 + 0.1 * np.sin(2 * np.pi * 0.05 * ts)
            vec = [float(v_val)] * 5

        samples.append({
            "simulation_timestamp": float(ts),
            "vector": vec,
        })

    return {
        "schema_version": 1,
        "run": {
            "scenario": scenario,
            "seed": seed,
            "requested_duration_seconds": sample_count * dt,
            "completed_duration_seconds": sample_count * dt,
            "completed_normally": True,
            "error_type": None,
        },
        "meta_state": {
            "dimensions": ["hunger", "fatigue", "curiosity", "aggression", "social"],
            "samples": samples,
        },
    }


def test_load_and_validate_report_missing_file(tmp_path: Path) -> None:
    """Requirement: Missing file raises SpectrumAnalysisError."""
    missing_file = tmp_path / "non_existent.json"
    with pytest.raises(SpectrumAnalysisError, match="does not exist"):
        load_and_validate_scenario_report(missing_file)


def test_load_and_validate_report_malformed_json(tmp_path: Path) -> None:
    """Requirement: Malformed JSON raises SpectrumAnalysisError."""
    bad_json = tmp_path / "bad.json"
    bad_json.write_text("{ incomplete json ...", encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="Failed to parse"):
        load_and_validate_scenario_report(bad_json)


def test_load_and_validate_report_invalid_schema_version(tmp_path: Path) -> None:
    """Requirement: Schema version != 1 is rejected."""
    report = _make_valid_report_dict()
    report["schema_version"] = 999
    p = tmp_path / "bad_schema.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="Unsupported report schema_version"):
        load_and_validate_scenario_report(p)


def test_load_and_validate_report_missing_meta_state(tmp_path: Path) -> None:
    """Requirement: Missing meta_state section is rejected."""
    report = _make_valid_report_dict()
    del report["meta_state"]
    p = tmp_path / "no_meta.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="missing required 'meta_state'"):
        load_and_validate_scenario_report(p)


def test_load_and_validate_report_wrong_dimensions(tmp_path: Path) -> None:
    """Requirement: Wrong MetaState dimension names/order are rejected."""
    report = _make_valid_report_dict()
    report["meta_state"]["dimensions"] = ["hunger", "fatigue", "curiosity"]
    p = tmp_path / "wrong_dims.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="Invalid MetaState dimensions"):
        load_and_validate_scenario_report(p)


def test_load_and_validate_report_insufficient_samples(tmp_path: Path) -> None:
    """Requirement: Reports with fewer than MIN_SPECTRUM_SAMPLES are rejected."""
    report = _make_valid_report_dict(sample_count=50)  # 50 < 128
    p = tmp_path / "short.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="Insufficient MetaState samples"):
        load_and_validate_scenario_report(p)


def test_load_and_validate_report_non_increasing_timestamps(tmp_path: Path) -> None:
    """Requirement: Non-strictly increasing timestamps are rejected."""
    report = _make_valid_report_dict(sample_count=200)
    report["meta_state"]["samples"][10]["simulation_timestamp"] = 0.5
    report["meta_state"]["samples"][11]["simulation_timestamp"] = 0.5  # duplicate

    p = tmp_path / "dup_ts.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="strictly increasing"):
        load_and_validate_scenario_report(p)


def test_load_and_validate_report_invalid_vector(tmp_path: Path) -> None:
    """Requirement: Vector length != 5 or non-finite elements are rejected."""
    report = _make_valid_report_dict(sample_count=200)
    report["meta_state"]["samples"][5]["vector"] = [0.1, 0.2, 0.3]  # len = 3

    p = tmp_path / "bad_vec.json"
    p.write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(SpectrumAnalysisError, match="vector must be a list of 5 floats"):
        load_and_validate_scenario_report(p)


def test_resample_uniform_diagnostics_and_interpolation() -> None:
    """Requirement: resample_uniform correctly calculates diagnostics and interpolates grid."""
    # Create slightly jittered timestamps
    rng = np.random.default_rng(123)
    n = 200
    base_ts = np.linspace(0.0, 20.0, n)
    jitter = rng.uniform(-0.01, 0.01, size=n)
    jitter[0] = 0.0
    jitter[-1] = 0.0
    ts = base_ts + jitter
    ts = np.sort(ts)  # Ensure strictly increasing

    vecs = np.zeros((n, 5), dtype=np.float64)
    for d in range(5):
        vecs[:, d] = np.sin((d + 1) * ts)

    diag, grid_ts, resampled_vecs = resample_uniform(ts, vecs)

    assert "median_dt" in diag
    assert "sampling_rate_hz" in diag
    assert "dt_cv" in diag
    assert diag["sampling_rate_hz"] > 0
    assert len(grid_ts) == len(resampled_vecs)
    assert len(grid_ts) >= MIN_SPECTRUM_SAMPLES


def test_compute_dimension_spectrum_zero_variance_signal() -> None:
    """Requirement: Constant/zero-variance signal raises SpectrumAnalysisError."""
    const_signal = np.full(500, 0.5, dtype=np.float64)

    with pytest.raises(SpectrumAnalysisError, match="near-zero variance"):
        compute_dimension_spectrum(const_signal, sampling_rate_hz=10.0)


def test_compute_dimension_spectrum_reference_signals() -> None:
    """Requirement: White noise and Brownian noise reference signals yield expected slopes."""
    rng = np.random.default_rng(42)
    fs = 10.0
    n = 2000

    # 1. White noise (theoretical slope ≈ 0)
    white_noise = rng.normal(0, 1, size=n)
    dim_meta_w, _ = compute_dimension_spectrum(white_noise, sampling_rate_hz=fs)
    slope_w = dim_meta_w["slope"]
    assert abs(slope_w) < 0.6, f"Expected white noise slope near 0, got {slope_w}"

    # 2. Brownian noise (theoretical slope ≈ -2)
    brownian_noise = np.cumsum(rng.normal(0, 1, size=n))
    dim_meta_b, _ = compute_dimension_spectrum(brownian_noise, sampling_rate_hz=fs)
    slope_b = dim_meta_b["slope"]
    assert -2.6 < slope_b < -1.4, f"Expected Brownian noise slope near -2.0, got {slope_b}"


def test_analyze_spectrum_full_pipeline(tmp_path: Path) -> None:
    """Requirement: analyze_spectrum completes, writes JSON, and generates plot."""
    # Construct synthetic Brownian-like drive series (1000 samples)
    n = 1000
    dt = 0.1

    def gen_vector(step_i: int, sim_ts: float) -> list[float]:
        return [float(v) for v in np.sin(np.array([1, 2, 3, 4, 5]) * 0.1 * sim_ts) + 0.5]

    report = _make_valid_report_dict(
        sample_count=n,
        dt=dt,
        scenario="peaceful_farm",
        seed=123,
        vector_fn=gen_vector,
    )

    report_file = tmp_path / "test_report.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")

    out_dir = tmp_path / "analysis_output"

    analysis_result, per_dim_arrays = analyze_spectrum(report_file)

    assert analysis_result["analysis_schema_version"] == ANALYSIS_SCHEMA_VERSION
    assert analysis_result["source_provenance"]["scenario"] == "peaceful_farm"
    assert analysis_result["source_provenance"]["seed"] == 123
    assert analysis_result["sampling"]["original_sample_count"] == n

    # Verify JSON output
    json_path = write_analysis_json(analysis_result, out_dir)
    assert json_path.is_file()

    # Confirm strict JSON (allow_nan=False) by loading it
    loaded_json = json.loads(json_path.read_text(encoding="utf-8"))
    assert loaded_json["analysis_schema_version"] == ANALYSIS_SCHEMA_VERSION

    # Verify Plot output
    plot_path = plot_psd(analysis_result, per_dim_arrays, out_dir)
    assert plot_path.is_file()
    assert plot_path.stat().st_size > 0


def test_spectrum_analyzer_cli_invocation(tmp_path: Path) -> None:
    """Requirement: CLI script executes cleanly on a valid report fixture."""
    report = _make_valid_report_dict(sample_count=200, dt=0.1)
    report_file = tmp_path / "cli_report.json"
    report_file.write_text(json.dumps(report), encoding="utf-8")

    out_dir = tmp_path / "cli_out"

    cli_args = ["--input", str(report_file), "--output-dir", str(out_dir)]

    # Run CLI main
    cli_main(cli_args)

    assert (out_dir / "spectrum_analysis.json").is_file()
    assert (out_dir / "spectrum_psd.png").is_file()


def test_spectrum_analyzer_cli_missing_input_raises_exit(tmp_path: Path) -> None:
    """Requirement: CLI exits with code 1 when given a non-existent input report."""
    missing = tmp_path / "missing.json"
    cli_args = ["--input", str(missing)]

    with pytest.raises(SystemExit) as exc_info:
        cli_main(cli_args)

    assert exc_info.value.code == 1
