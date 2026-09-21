"""# Pre-lab (MOCK_MODE)
Scenario Report Spectrum Analysis Module (Task 8.1).

Provides FFT and Welch Power Spectral Density (PSD) analysis on MetaState time series
data extracted from Task 7.2 Scenario Reports.

Public API:
    - :class:`SpectrumAnalysisError`
    - :data:`ANALYSIS_SCHEMA_VERSION`
    - :data:`MIN_SPECTRUM_SAMPLES`
    - :data:`MIN_FIT_POINTS`
    - :data:`TARGET_SLOPE_RANGE`
    - :func:`load_and_validate_scenario_report` (DEPRECATED: see REPORTING_RECONCILIATION.md)
    - :func:`resample_uniform`
    - :func:`compute_dimension_spectrum`
    - :func:`analyze_spectrum` (DEPRECATED: see REPORTING_RECONCILIATION.md)
    - :func:`write_analysis_json`
    - :func:`plot_psd`
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # Non-interactive headless backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal  # type: ignore[import-untyped]

from wow_bot.reporting.scenario import META_STATE_DIMENSIONS, SCENARIO_REPORT_SCHEMA_VERSION

ANALYSIS_SCHEMA_VERSION: int = 1
MIN_SPECTRUM_SAMPLES: int = 128
MIN_FIT_POINTS: int = 8
TARGET_SLOPE_RANGE: tuple[float, float] = (-1.5, -0.5)


class SpectrumAnalysisError(ValueError):
    """Domain exception raised when spectrum analysis cannot be performed on report input."""


def load_and_validate_scenario_report(report_path: Path | str) -> dict[str, Any]:
    """Load and strictly validate a Task 7.2 scenario report JSON file.

    DEPRECATED: Pre-lab scenario report loader. See REPORTING_RECONCILIATION.md.

    Args:
        report_path: File path to scenario report JSON.

    Returns:
        Validated report dictionary.

    Raises:
        SpectrumAnalysisError: If report path is missing, malformed, or violates schema contracts.
    """
    path = Path(report_path)
    if not path.is_file():
        raise SpectrumAnalysisError(f"Scenario report file does not exist: '{report_path}'")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as exc:
        raise SpectrumAnalysisError(f"Failed to parse scenario report JSON: {exc}") from exc

    if not isinstance(data, dict):
        raise SpectrumAnalysisError("Scenario report root must be a JSON object")

    schema_ver = data.get("schema_version")
    if schema_ver != SCENARIO_REPORT_SCHEMA_VERSION:
        raise SpectrumAnalysisError(
            f"Unsupported report schema_version {schema_ver!r} (expected {SCENARIO_REPORT_SCHEMA_VERSION})"
        )

    meta_state = data.get("meta_state")
    if not isinstance(meta_state, dict):
        raise SpectrumAnalysisError("Scenario report missing required 'meta_state' dictionary")

    dims = meta_state.get("dimensions")
    if not isinstance(dims, list) or tuple(dims) != META_STATE_DIMENSIONS:
        raise SpectrumAnalysisError(
            f"Invalid MetaState dimensions {dims!r} (expected {list(META_STATE_DIMENSIONS)})"
        )

    samples = meta_state.get("samples")
    if not isinstance(samples, list):
        raise SpectrumAnalysisError("Scenario report missing required 'meta_state.samples' list")

    if len(samples) < MIN_SPECTRUM_SAMPLES:
        raise SpectrumAnalysisError(
            f"Insufficient MetaState samples for spectrum analysis ({len(samples)} < {MIN_SPECTRUM_SAMPLES})"
        )

    last_ts: float | None = None
    for idx, sample in enumerate(samples):
        if not isinstance(sample, dict):
            raise SpectrumAnalysisError(f"Sample #{idx} is not a JSON object")

        ts = sample.get("simulation_timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise SpectrumAnalysisError(
                f"Sample #{idx} contains non-finite or invalid timestamp: {ts!r}"
            )

        ts_float = float(ts)
        if last_ts is not None and ts_float <= last_ts:
            raise SpectrumAnalysisError(
                f"Sample #{idx} timestamp must be strictly increasing, got {ts_float} <= {last_ts}"
            )
        last_ts = ts_float

        vec = sample.get("vector")
        if not isinstance(vec, list) or len(vec) != 5:
            raise SpectrumAnalysisError(
                f"Sample #{idx} vector must be a list of 5 floats, got {vec!r}"
            )

        for v_idx, elem in enumerate(vec):
            if isinstance(elem, bool) or not isinstance(elem, (int, float)) or not math.isfinite(elem):
                raise SpectrumAnalysisError(
                    f"Sample #{idx} vector[{v_idx}] is non-finite or invalid: {elem!r}"
                )

    return data


def resample_uniform(
    timestamps: np.ndarray,
    vectors: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Calculate sampling diagnostics and resample drive vectors onto a uniform grid.

    Args:
        timestamps: 1D NumPy float array of simulation timestamps.
        vectors: 2D NumPy float array of shape (N, 5) representing drive time series.

    Returns:
        Tuple of (sampling_diagnostics_dict, resampled_timestamps, resampled_vectors).

    Raises:
        SpectrumAnalysisError: If timestamps are invalid or resulting grid is insufficient.
    """
    ts = np.asarray(timestamps, dtype=np.float64)
    vecs = np.asarray(vectors, dtype=np.float64)

    if ts.ndim != 1 or vecs.ndim != 2 or vecs.shape[1] != 5 or len(ts) != len(vecs):
        raise SpectrumAnalysisError("Input arrays shape mismatch for uniform resampling")

    delta_t = ts[1:] - ts[:-1]
    if np.any(delta_t <= 0) or not np.all(np.isfinite(delta_t)):
        raise SpectrumAnalysisError("Timestamps must be strictly increasing and finite")

    median_dt = float(np.median(delta_t))
    mean_dt = float(np.mean(delta_t))
    std_dt = float(np.std(delta_t))
    dt_cv = float(std_dt / mean_dt) if mean_dt > 0 else 0.0
    sampling_rate_hz = 1.0 / median_dt if median_dt > 0 else 0.0

    if median_dt <= 0 or not math.isfinite(sampling_rate_hz):
        raise SpectrumAnalysisError(f"Invalid median_dt ({median_dt}) calculated from timestamps")

    t_start = float(ts[0])
    t_end = float(ts[-1])
    duration = t_end - t_start

    num_grid_points = math.floor(duration / median_dt) + 1
    if num_grid_points < MIN_SPECTRUM_SAMPLES:
        raise SpectrumAnalysisError(
            f"Resampled grid points ({num_grid_points}) fewer than required minimum ({MIN_SPECTRUM_SAMPLES})"
        )

    grid_ts = t_start + np.arange(num_grid_points, dtype=np.float64) * median_dt
    grid_ts = grid_ts[grid_ts <= t_end]

    resampled_vecs = np.zeros((len(grid_ts), 5), dtype=np.float64)
    for dim_idx in range(5):
        resampled_vecs[:, dim_idx] = np.interp(grid_ts, ts, vecs[:, dim_idx])

    diagnostics = {
        "median_dt": median_dt,
        "mean_dt": mean_dt,
        "std_dt": std_dt,
        "dt_cv": dt_cv,
        "sampling_rate_hz": sampling_rate_hz,
    }

    return diagnostics, grid_ts, resampled_vecs


def compute_dimension_spectrum(
    signal: np.ndarray,
    sampling_rate_hz: float,
    nperseg: int | None = None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Compute FFT diagnostics, Welch PSD, log-log linear fit, and slope target evaluation for 1D drive signal.

    Args:
        signal: 1D NumPy array of uniformly sampled drive values.
        sampling_rate_hz: Sampling frequency in Hz.
        nperseg: Welch segment length (defaults to min(256, len(signal))).

    Returns:
        Tuple of (dimension_metadata_dict, resampled_arrays_dict).

    Raises:
        SpectrumAnalysisError: If signal has zero variance or insufficient non-zero PSD fit points.
    """
    sig = np.asarray(signal, dtype=np.float64)
    if sig.ndim != 1:
        raise SpectrumAnalysisError(f"Signal must be 1D, got shape {sig.shape}")

    std_val = float(np.std(sig))
    if std_val < 1e-12 or not math.isfinite(std_val):
        raise SpectrumAnalysisError(
            f"Insufficient non-zero PSD points or near-zero variance signal (std = {std_val:.2e})"
        )

    centered = sig - float(np.mean(sig))

    # FFT diagnostics
    fft_vals = np.abs(np.fft.rfft(centered))
    fft_freqs = np.fft.rfftfreq(len(centered), d=1.0 / sampling_rate_hz)

    # Welch PSD calculation
    actual_nperseg = nperseg if nperseg is not None else min(256, len(centered))
    actual_nperseg = max(16, min(actual_nperseg, len(centered)))

    freqs, psd = scipy.signal.welch(
        centered,
        fs=float(sampling_rate_hz),
        detrend="constant",
        nperseg=actual_nperseg,
    )

    valid = (freqs > 0) & (psd > 0) & np.isfinite(freqs) & np.isfinite(psd)
    freqs_fit = freqs[valid]
    psd_fit = psd[valid]

    if len(freqs_fit) < MIN_FIT_POINTS:
        raise SpectrumAnalysisError(
            f"Insufficient non-zero PSD points ({len(freqs_fit)} < {MIN_FIT_POINTS}) for regression fit"
        )

    log_f = np.log10(freqs_fit)
    log_p = np.log10(psd_fit)

    slope_val, intercept_val = np.polyfit(log_f, log_p, 1)
    slope = float(slope_val)
    intercept = float(intercept_val)

    if not (math.isfinite(slope) and math.isfinite(intercept)):
        raise SpectrumAnalysisError("Non-finite slope or intercept calculated in log-log fit")

    # Diagnostic R² calculation
    r_matrix = np.corrcoef(log_f, log_p)
    r_val = float(r_matrix[0, 1]) if r_matrix.shape == (2, 2) else 0.0
    r_squared = float(r_val**2) if math.isfinite(r_val) else 0.0

    target_min, target_max = TARGET_SLOPE_RANGE
    within_target_range = bool(target_min <= slope <= target_max)

    dim_metadata = {
        "welch_nperseg": int(actual_nperseg),
        "fit_policy": "positive_welch_bins",
        "fit_points": len(freqs_fit),
        "slope": slope,
        "intercept": intercept,
        "r_squared": r_squared,
        "within_target_range": within_target_range,
        "pass_target": within_target_range,
    }

    arrays_data = {
        "welch_freqs": freqs.tolist(),
        "welch_psd": psd.tolist(),
        "fft_freqs": fft_freqs.tolist(),
        "fft_magnitude": fft_vals.tolist(),
    }

    return dim_metadata, arrays_data


def analyze_spectrum(report_path: Path | str) -> tuple[dict[str, Any], dict[str, Any]]:
    """Perform complete spectrum analysis on a scenario report file.

    DEPRECATED: Pre-lab scenario spectrum analysis. See REPORTING_RECONCILIATION.md.

    Args:
        report_path: Path to Task 7.2 scenario report JSON.

    Returns:
        Tuple of (analysis_result_dict, per_dimension_arrays_dict).
    """
    report = load_and_validate_scenario_report(report_path)

    samples = report["meta_state"]["samples"]
    timestamps = np.array([float(s["simulation_timestamp"]) for s in samples], dtype=np.float64)
    vectors = np.array([[float(v) for v in s["vector"]] for s in samples], dtype=np.float64)

    sampling_diag, grid_ts, resampled_vecs = resample_uniform(timestamps, vectors)

    dimensions_summary: dict[str, Any] = {}
    per_dimension_arrays: dict[str, Any] = {}
    pass_count = 0

    for idx, name in enumerate(META_STATE_DIMENSIONS):
        dim_meta, dim_arrays = compute_dimension_spectrum(
            signal=resampled_vecs[:, idx],
            sampling_rate_hz=sampling_diag["sampling_rate_hz"],
        )
        dimensions_summary[name] = dim_meta
        per_dimension_arrays[name] = dim_arrays
        if dim_meta["within_target_range"]:
            pass_count += 1

    run_info = report.get("run", {})
    scenario_name = run_info.get("scenario")
    seed_val = run_info.get("seed")

    analysis_result = {
        "analysis_schema_version": ANALYSIS_SCHEMA_VERSION,
        "source_provenance": {
            "input_report": str(report_path),
            "schema_version": report.get("schema_version"),
            "scenario": scenario_name,
            "seed": seed_val,
        },
        "sampling": {
            "original_sample_count": len(timestamps),
            "resampled_sample_count": len(grid_ts),
            "start_simulation_timestamp": float(timestamps[0]),
            "end_simulation_timestamp": float(timestamps[-1]),
            "median_dt": sampling_diag["median_dt"],
            "mean_dt": sampling_diag["mean_dt"],
            "std_dt": sampling_diag["std_dt"],
            "sampling_rate_hz": sampling_diag["sampling_rate_hz"],
            "dt_cv": sampling_diag["dt_cv"],
        },
        "summary": {
            "total_dimensions": len(META_STATE_DIMENSIONS),
            "dimensions_within_target_range": pass_count,
            "target_range": list(TARGET_SLOPE_RANGE),
        },
        "dimensions": dimensions_summary,
    }

    return analysis_result, per_dimension_arrays


def write_analysis_json(
    analysis_result: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    """Write structured analysis JSON to `<output_dir>/spectrum_analysis.json`.

    Args:
        analysis_result: Analysis dictionary.
        output_dir: Target output directory.

    Returns:
        Path to written spectrum_analysis.json file.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    target_path = out_dir / "spectrum_analysis.json"
    json_text = json.dumps(analysis_result, indent=2, sort_keys=True, allow_nan=False)

    with target_path.open("w", encoding="utf-8") as f:
        f.write(json_text)

    return target_path


def plot_psd(
    analysis_result: dict[str, Any],
    per_dimension_arrays: dict[str, Any],
    output_dir: Path | str,
) -> Path:
    """Generate and save per-dimension Welch PSD log-log plot to `<output_dir>/spectrum_psd.png`.

    Args:
        analysis_result: Analysis metadata dictionary.
        per_dimension_arrays: Numeric freqs/psd dictionary per dimension.
        output_dir: Target output directory.

    Returns:
        Path to generated PNG plot.
    """
    out_dir = Path(output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    target_path = out_dir / "spectrum_psd.png"

    fig, ax = plt.subplots(figsize=(10, 6), dpi=150)

    for name in META_STATE_DIMENSIONS:
        arrays = per_dimension_arrays.get(name, {})
        freqs = np.array(arrays.get("welch_freqs", []), dtype=np.float64)
        psd = np.array(arrays.get("welch_psd", []), dtype=np.float64)

        valid = (freqs > 0) & (psd > 0)
        if not np.any(valid):
            continue

        dim_info = analysis_result.get("dimensions", {}).get(name, {})
        slope = dim_info.get("slope", 0.0)
        status = "PASS" if dim_info.get("within_target_range") else "OUTSIDE TARGET"
        label = f"{name} (slope={slope:.2f}, {status})"

        ax.loglog(freqs[valid], psd[valid], label=label, linewidth=1.5)

    scenario_name = analysis_result.get("source_provenance", {}).get("scenario") or "Scenario"
    ax.set_title(f"MetaState Power Spectral Density (PSD) — {scenario_name}")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Power Spectral Density")
    ax.grid(True, which="both", linestyle="--", alpha=0.5)
    ax.legend(loc="best", fontsize=9)

    fig.tight_layout()
    fig.savefig(target_path)
    plt.close(fig)

    return target_path
