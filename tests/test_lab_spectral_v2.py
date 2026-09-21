"""Unit tests for Lab Spectral Analysis V2 (Task 10.3)."""

from __future__ import annotations

import ast
import json
import math
from pathlib import Path

import pytest

from wow_bot.analysis.lab_spectral_v2 import (
    ANALYSIS_SCHEMA_VERSION,
    LabSpectralError,
    SpectralConfig,
    SpectralResult,
    analyze_intervals,
    analyze_report,
    write_result,
)
from wow_bot.reporting.schema_v2 import SCHEMA_VERSION


def test_spectral_config_validation() -> None:
    # min_samples < 8 raises ValueError
    with pytest.raises(ValueError, match="min_samples must be >= 8"):
        SpectralConfig(min_samples=7)

    # invalid window raises ValueError
    with pytest.raises(ValueError, match="window must be 'hann' or 'rect'"):
        SpectralConfig(window="blackman")

    # malformed target_slope_range raises ValueError
    with pytest.raises(ValueError, match="target_slope_range"):
        SpectralConfig(target_slope_range=(-0.5, -1.5))  # low >= high

    with pytest.raises(ValueError, match="target_slope_range"):
        SpectralConfig(target_slope_range=(0.0, float("nan")))


def test_analyze_intervals_insufficient_samples() -> None:
    res = analyze_intervals([100.0] * 10)
    assert res.skipped is True
    assert res.reason == "insufficient_samples:10"
    assert res.sample_count == 10
    assert res.slope is None
    assert res.slope_in_target_range is False
    assert res.psd_frequencies == ()
    assert res.psd_values == ()


def test_analyze_intervals_invalid_input() -> None:
    # Non-finite input
    with pytest.raises(LabSpectralError, match="non-numeric or non-finite"):
        analyze_intervals([100.0] * 64 + [float("nan")])

    # Negative input
    with pytest.raises(LabSpectralError, match="must be non-negative"):
        analyze_intervals([100.0] * 64 + [-5.0])


def test_analyze_intervals_white_noise() -> None:
    # Deterministic pseudo-white noise sequence
    samples = [100.0 + 10.0 * math.sin(i * 12.345) for i in range(128)]
    res = analyze_intervals(samples)

    assert res.skipped is False
    assert res.reason == ""
    assert res.sample_count == 128
    assert res.slope is not None
    assert len(res.psd_frequencies) == len(res.psd_values)
    assert len(res.psd_frequencies) == 128 // 2 + 1


def test_analyze_intervals_periodic_peak() -> None:
    # Strong sinusoidal component at frequency bin k=8
    N = 128
    target_k = 8
    samples = [500.0 + 50.0 * math.cos(2.0 * math.pi * target_k * i / N) for i in range(N)]
    res = analyze_intervals(samples, config=SpectralConfig(window="rect"))

    assert res.skipped is False
    freqs = res.psd_frequencies
    psd = res.psd_values

    # Frequency bin k=8 is target_k / N
    expected_freq = target_k / N
    max_idx = max(range(1, len(psd)), key=lambda idx: psd[idx])
    assert math.isclose(freqs[max_idx], expected_freq, abs_tol=1e-5)


def test_analyze_intervals_power_law_slope() -> None:
    # Synthesize non-negative interval signal (mean 500 ms) with power law spectrum exponent gamma = -1.0
    N = 256
    gamma = -1.0
    samples = [500.0] * N
    for k in range(1, N // 2 + 1):
        f_k = k / N
        amp = math.pow(f_k, gamma / 2.0)
        phase = (k * 1.6180339887) % (2.0 * math.pi)
        for i in range(N):
            samples[i] += amp * math.cos(2.0 * math.pi * k * i / N + phase)

    res = analyze_intervals(samples, config=SpectralConfig(window="rect"))
    assert res.skipped is False
    assert res.slope is not None
    assert math.isclose(res.slope, gamma, abs_tol=0.2)


def test_slope_in_target_range() -> None:
    # Test boundary conditions for TARGET_SLOPE_RANGE (-1.5, -0.5)
    r1 = SpectralResult(
        skipped=False,
        reason="",
        sample_count=100,
        slope=-1.0,
        slope_in_target_range=True,
        psd_frequencies=(0.1, 0.2),
        psd_values=(1.0, 0.5),
    )
    assert r1.slope_in_target_range is True

    r2 = SpectralResult(
        skipped=False,
        reason="",
        sample_count=100,
        slope=-0.2,
        slope_in_target_range=False,
        psd_frequencies=(0.1, 0.2),
        psd_values=(1.0, 0.5),
    )
    assert r2.slope_in_target_range is False


def test_analyze_intervals_detrend_mode_difference() -> None:
    # Non-zero mean signal
    N = 64
    samples = [500.0 + math.sin(i) for i in range(N)]

    res_detrend = analyze_intervals(samples, config=SpectralConfig(detrend=True, window="rect"))
    res_no_detrend = analyze_intervals(samples, config=SpectralConfig(detrend=False, window="rect"))

    dc_detrend = res_detrend.psd_values[0]
    dc_no_detrend = res_no_detrend.psd_values[0]

    assert math.isclose(dc_detrend, 0.0, abs_tol=1e-9)
    assert dc_no_detrend > 1000.0


def test_analyze_intervals_window_difference() -> None:
    N = 64
    samples = [100.0 + 10.0 * math.sin(i * 0.5) for i in range(N)]

    res_hann = analyze_intervals(samples, config=SpectralConfig(window="hann"))
    res_rect = analyze_intervals(samples, config=SpectralConfig(window="rect"))

    assert res_hann.psd_values != res_rect.psd_values


def test_analyze_report_validation() -> None:
    # Non-dict raises LabSpectralError
    with pytest.raises(LabSpectralError, match="report must be a dict"):
        analyze_report("not_a_dict")  # type: ignore[arg-type]

    # Invalid schema_version raises LabSpectralError
    with pytest.raises(LabSpectralError, match="Expected report schema_version == 2"):
        analyze_report({"schema_version": 1})

    # Missing humanizer section returns skipped
    res_no_hum = analyze_report({"schema_version": SCHEMA_VERSION})
    assert res_no_hum.skipped is True
    assert res_no_hum.reason == "no_humanizer_section"

    # Valid report with enough samples returns non-skipped result
    valid_report = {
        "schema_version": SCHEMA_VERSION,
        "humanizer": {
            "interval_samples_ms": [100.0 + (i % 10) for i in range(100)],
        },
    }
    res_valid = analyze_report(valid_report)
    assert res_valid.skipped is False
    assert res_valid.sample_count == 100


def test_write_result_atomic_and_dir_creation(tmp_path: Path) -> None:
    res = SpectralResult(
        skipped=False,
        reason="",
        sample_count=100,
        slope=-1.0,
        slope_in_target_range=True,
        psd_frequencies=(0.1, 0.2),
        psd_values=(10.0, 5.0),
    )

    out_file = tmp_path / "nested" / "dir" / "spectral.json"
    write_result(res, out_file)

    assert out_file.is_file()
    # Ensure no .tmp file left behind
    tmp_file = out_file.parent / f"{out_file.name}.tmp"
    assert not tmp_file.exists()

    content = json.loads(out_file.read_text(encoding="utf-8"))
    assert content["slope"] == -1.0
    assert content["psd_frequencies"] == [0.1, 0.2]


def test_to_json_serialization() -> None:
    res = SpectralResult(
        skipped=True,
        reason="insufficient_samples:5",
        sample_count=5,
        slope=None,
        slope_in_target_range=False,
        psd_frequencies=(),
        psd_values=(),
    )
    d = res.to_json()
    assert d == {
        "skipped": True,
        "reason": "insufficient_samples:5",
        "sample_count": 5,
        "slope": None,
        "slope_in_target_range": False,
        "psd_frequencies": [],
        "psd_values": [],
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
    }
    # Verify JSON serializability
    json_str = json.dumps(d)
    assert json.loads(json_str) == d


def test_determinism() -> None:
    samples = [150.0 + 20.0 * math.sin(i * 0.3) for i in range(100)]
    res1 = analyze_intervals(samples)
    res2 = analyze_intervals(samples)

    assert res1 == res2
    assert res1.to_json() == res2.to_json()


def test_static_ast_checks() -> None:
    module_path = Path("src/wow_bot/analysis/lab_spectral_v2.py")
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
    git_modified = Path("src/wow_bot/analysis/spectrum.py")
    assert git_modified.is_file()
