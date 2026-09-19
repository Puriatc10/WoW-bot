"""Timing Distribution Analysis module (Task 8.2).

Provides conditional log-normal model validation using randomized Probability Integral
Transform (PIT), Kolmogorov-Smirnov (KS) testing against Uniform(0,1), coefficient of
variation (CV) calculation, and plot generation on Task 7.2 scenario reports.

Public API:
    - :data:`PIT_RANDOM_SEED`
    - :data:`MIN_TIMING_SAMPLES`
    - :data:`KS_ALPHA`
    - :data:`CV_THRESHOLD`
    - :func:`load_timing_report`
    - :func:`validate_timing_samples`
    - :func:`compute_descriptive_statistics`
    - :func:`compute_randomized_pit`
    - :func:`run_conditional_ks`
    - :func:`build_timing_analysis`
    - :func:`plot_distribution`
    - :func:`write_analysis_output`
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from pathlib import Path
from typing import Any, Final

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import scipy.stats  # type: ignore[import-untyped]

from wow_bot.executor.humanize import (
    MAX_DELAY_MS,
    MIN_DELAY_MS,
    delay_distribution_parameters,
)
from wow_bot.reporting.scenario import SCENARIO_REPORT_SCHEMA_VERSION

PIT_RANDOM_SEED: Final[int] = 8202
MIN_TIMING_SAMPLES: Final[int] = 100
KS_ALPHA: Final[float] = 0.05
CV_THRESHOLD: Final[float] = 0.3
LOWER_DELAY_MS: Final[float] = MIN_DELAY_MS
UPPER_DELAY_MS: Final[float] = MAX_DELAY_MS


def load_timing_report(report_path: str | Path) -> dict[str, Any]:
    """Load and perform basic validation on a Task 7.2 scenario report.

    Args:
        report_path: Path to the JSON report file.

    Returns:
        Loaded report dictionary.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If JSON is invalid or schema_version is incompatible.
    """
    path = Path(report_path)
    if not path.is_file():
        raise FileNotFoundError(f"Scenario report file not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON in scenario report {path}: {exc}") from exc

    if not isinstance(data, dict):
        raise ValueError(f"Scenario report must be a JSON object, got {type(data).__name__}")

    schema_ver = data.get("schema_version")
    if schema_ver != SCENARIO_REPORT_SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported schema_version {schema_ver!r} in report (expected {SCENARIO_REPORT_SCHEMA_VERSION})"
        )

    return data


def validate_timing_samples(
    timing_dict: dict[str, Any],
) -> tuple[list[float], list[dict[str, Any]] | None]:
    """Validate and extract raw delays and detailed per-sample metadata.

    Args:
        timing_dict: "timing" section from scenario report dictionary.

    Returns:
        Tuple of (delays_ms, detailed_samples or None).

    Raises:
        ValueError: If delays or metadata contain NaN, Inf, non-positive,
            or out-of-range parameter values.
    """
    if not isinstance(timing_dict, dict):
        raise ValueError(f"'timing' section must be a dictionary, got {type(timing_dict).__name__}")

    samples_ms = timing_dict.get("samples_ms")
    if not isinstance(samples_ms, list):
        raise ValueError(f"'timing.samples_ms' must be a list, got {type(samples_ms).__name__}")

    delays: list[float] = []
    for idx, raw in enumerate(samples_ms):
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            raise ValueError(f"Sample #{idx} in 'samples_ms' is not numeric: {raw!r}")
        val = float(raw)
        if not math.isfinite(val):
            raise ValueError(f"Sample #{idx} in 'samples_ms' is non-finite: {val}")
        if val <= 0.0:
            raise ValueError(f"Sample #{idx} in 'samples_ms' must be positive (> 0), got {val}")
        if val < LOWER_DELAY_MS or val > UPPER_DELAY_MS:
            raise ValueError(
                f"Sample #{idx} in 'samples_ms' outside model bounds [{LOWER_DELAY_MS}, {UPPER_DELAY_MS}]: {val}"
            )
        delays.append(val)

    detailed_samples_raw = timing_dict.get("samples")
    if detailed_samples_raw is None:
        return delays, None

    if not isinstance(detailed_samples_raw, list):
        raise ValueError(f"'timing.samples' must be a list, got {type(detailed_samples_raw).__name__}")

    if len(detailed_samples_raw) != len(delays):
        raise ValueError(
            f"Length mismatch: 'samples_ms' ({len(delays)}) vs 'samples' ({len(detailed_samples_raw)})"
        )

    detailed_samples: list[dict[str, Any]] = []
    for idx, s in enumerate(detailed_samples_raw):
        if not isinstance(s, dict):
            raise ValueError(f"Detailed sample #{idx} in 'samples' must be dict, got {type(s).__name__}")

        ts = s.get("simulation_timestamp")
        if isinstance(ts, bool) or not isinstance(ts, (int, float)) or not math.isfinite(ts):
            raise ValueError(f"Detailed sample #{idx} invalid simulation_timestamp: {ts!r}")

        d_ms = s.get("delay_ms")
        if isinstance(d_ms, bool) or not isinstance(d_ms, (int, float)) or not math.isfinite(d_ms):
            raise ValueError(f"Detailed sample #{idx} invalid delay_ms: {d_ms!r}")
        d_val = float(d_ms)
        if d_val <= 0.0 or d_val < LOWER_DELAY_MS or d_val > UPPER_DELAY_MS:
            raise ValueError(f"Detailed sample #{idx} delay_ms out of bounds: {d_val}")

        b_ms = s.get("base_ms")
        if isinstance(b_ms, bool) or not isinstance(b_ms, int) or b_ms <= 0:
            raise ValueError(f"Detailed sample #{idx} invalid base_ms: {b_ms!r}")

        fat = s.get("fatigue")
        if isinstance(fat, bool) or not isinstance(fat, (int, float)) or not math.isfinite(fat):
            raise ValueError(f"Detailed sample #{idx} invalid fatigue: {fat!r}")
        fat_val = float(fat)
        if fat_val < 0.0 or fat_val > 1.0:
            raise ValueError(f"Detailed sample #{idx} fatigue out of range [0, 1]: {fat_val}")

        chaos = s.get("chaos_component")
        if isinstance(chaos, bool) or not isinstance(chaos, (int, float)) or not math.isfinite(chaos):
            raise ValueError(f"Detailed sample #{idx} invalid chaos_component: {chaos!r}")
        chaos_val = float(chaos)
        if chaos_val < -1.0 or chaos_val > 1.0:
            raise ValueError(f"Detailed sample #{idx} chaos_component out of range [-1, 1]: {chaos_val}")

        detailed_samples.append({
            "simulation_timestamp": float(ts),
            "delay_ms": d_val,
            "base_ms": int(b_ms),
            "fatigue": fat_val,
            "chaos_component": chaos_val,
        })

    return delays, detailed_samples


def compute_descriptive_statistics(delays: list[float]) -> dict[str, Any]:
    """Compute descriptive statistics on raw clipped delay_ms values.

    CV is calculated as sample standard deviation (ddof=1) divided by sample mean.

    Args:
        delays: Validated list of delay milliseconds floats.

    Returns:
        Dictionary of descriptive statistics.
    """
    count = len(delays)
    if count == 0:
        return {
            "count": 0,
            "mean_ms": 0.0,
            "std_ms": 0.0,
            "cv": 0.0,
            "min_ms": 0.0,
            "max_ms": 0.0,
            "median_ms": 0.0,
            "p05_ms": 0.0,
            "p95_ms": 0.0,
            "lower_clip_count": 0,
            "upper_clip_count": 0,
            "clipped_fraction": 0.0,
        }

    arr = np.array(delays, dtype=np.float64)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr, ddof=1)) if count > 1 else 0.0
    cv_val = float(std_val / mean_val) if mean_val > 0.0 and count > 1 else 0.0

    lower_clip_count = int(np.sum(arr == LOWER_DELAY_MS))
    upper_clip_count = int(np.sum(arr == UPPER_DELAY_MS))
    clipped_fraction = float((lower_clip_count + upper_clip_count) / count)

    return {
        "count": count,
        "mean_ms": mean_val,
        "std_ms": std_val,
        "cv": cv_val,
        "min_ms": float(np.min(arr)),
        "max_ms": float(np.max(arr)),
        "median_ms": float(np.median(arr)),
        "p05_ms": float(np.percentile(arr, 5)),
        "p95_ms": float(np.percentile(arr, 95)),
        "lower_clip_count": lower_clip_count,
        "upper_clip_count": upper_clip_count,
        "clipped_fraction": clipped_fraction,
    }


def compute_randomized_pit(
    detailed_samples: list[dict[str, Any]],
    pit_seed: int = PIT_RANDOM_SEED,
) -> list[float]:
    """Compute randomized Probability Integral Transform (PIT) values for conditional delays.

    For interior samples (50 < y < 2000): u_i = F_i(y_i)
    For lower clipped samples (y <= 50): u_i = v_i * F_i(50), v_i ~ Uniform(0,1)
    For upper clipped samples (y >= 2000): u_i = F_i(2000) + v_i * (1 - F_i(2000)), v_i ~ Uniform(0,1)

    Args:
        detailed_samples: List of per-sample metadata dicts.
        pit_seed: Deterministic RNG seed.

    Returns:
        List of PIT values in [0.0, 1.0].
    """
    rng = np.random.default_rng(pit_seed)
    pit_values: list[float] = []

    for s in detailed_samples:
        delay_ms = float(s["delay_ms"])
        base_ms = int(s["base_ms"])
        fatigue = float(s["fatigue"])
        chaos = float(s["chaos_component"])

        mu, sigma = delay_distribution_parameters(base_ms, fatigue, chaos)

        scale = math.exp(mu)
        dist = scipy.stats.lognorm(s=sigma, scale=scale, loc=0.0)

        if delay_ms <= LOWER_DELAY_MS:
            f_lower = float(dist.cdf(LOWER_DELAY_MS))
            v_i = float(rng.uniform(0.0, 1.0))
            u_i = v_i * f_lower
        elif delay_ms >= UPPER_DELAY_MS:
            f_upper = float(dist.cdf(UPPER_DELAY_MS))
            v_i = float(rng.uniform(0.0, 1.0))
            u_i = f_upper + v_i * (1.0 - f_upper)
        else:
            u_i = float(dist.cdf(delay_ms))

        u_clamped = float(np.clip(u_i, 0.0, 1.0))
        pit_values.append(u_clamped)

    return pit_values


def run_conditional_ks(pit_values: list[float]) -> tuple[float, float]:
    """Run one-sample Kolmogorov-Smirnov test against Uniform(0,1).

    Args:
        pit_values: PIT transformed observations in [0, 1].

    Returns:
        Tuple of (ks_statistic, p_value).
    """
    res = scipy.stats.kstest(pit_values, "uniform")
    return float(res.statistic), float(res.pvalue)


def build_timing_analysis(
    report_dict: dict[str, Any],
    pit_seed: int = PIT_RANDOM_SEED,
    min_samples: int = MIN_TIMING_SAMPLES,
) -> dict[str, Any]:
    """Perform timing distribution analysis on scenario report dict.

    Args:
        report_dict: Task 7.2 Scenario Report dictionary.
        pit_seed: Deterministic PIT seed.
        min_samples: Minimum required sample count threshold.

    Returns:
        Structured timing analysis dictionary.
    """
    run_info = report_dict.get("run", {})
    timing_dict = report_dict.get("timing", {})

    delays, detailed_samples = validate_timing_samples(timing_dict)
    summary = compute_descriptive_statistics(delays)
    sample_count = summary["count"]

    source_provenance = {
        "scenario": run_info.get("scenario"),
        "seed": run_info.get("seed"),
        "report_schema_version": report_dict.get("schema_version"),
        "timing_unit": timing_dict.get("unit", "ms"),
        "timing_model": timing_dict.get("model", "lognormal_simulation"),
        "sample_count": sample_count,
    }

    aggregate_fit: dict[str, Any] = {}
    if sample_count > 0:
        shape, _loc, scale = scipy.stats.lognorm.fit(delays, floc=0)
        aggregate_fit = {
            "type": "aggregate_heuristic_fit",
            "caveat": "Fitted across heterogeneous samples for plotting only; not generative truth.",
            "fitted_mu": float(np.log(scale)),
            "fitted_sigma": float(shape),
        }

    cv_val = float(summary["cv"])
    cv_within_target = cv_val > CV_THRESHOLD

    cv_target_dict = {
        "metric": "coefficient_of_variation",
        "value": cv_val,
        "target": f"> {CV_THRESHOLD}",
        "threshold_exclusive": CV_THRESHOLD,
        "within_target": cv_within_target,
    }

    conditional_test_dict: dict[str, Any] = {}
    ks_within_target: bool | None = None

    if sample_count < min_samples:
        conditional_test_dict = {
            "status": "insufficient_samples",
            "reason": f"Sample count {sample_count} is below minimum threshold of {min_samples}",
            "sample_count": sample_count,
            "min_samples_required": min_samples,
        }
        overall_within_target: bool | None = None
    elif detailed_samples is None:
        conditional_test_dict = {
            "status": "unavailable",
            "reason": "per-sample model metadata missing from timing section",
            "sample_count": sample_count,
        }
        overall_within_target = None
    else:
        pit_values = compute_randomized_pit(detailed_samples, pit_seed=pit_seed)
        ks_stat, p_val = run_conditional_ks(pit_values)
        ks_within_target = p_val > KS_ALPHA

        conditional_test_dict = {
            "status": "available",
            "method": "randomized_pit_ks_uniform",
            "pit_seed": pit_seed,
            "sample_count": sample_count,
            "ks_statistic": ks_stat,
            "p_value": p_val,
            "target_p_value_exclusive": KS_ALPHA,
            "within_target": ks_within_target,
            "methodological_caveat": (
                "The KS p-value serves as a roadmap model-consistency diagnostic under the "
                "conditional clipped-lognormal assumption. Temporal dependence across "
                "scenario time series may affect standard independence assumptions."
            ),
        }
        overall_within_target = bool(ks_within_target and cv_within_target)

    roadmap_acceptance = {
        "ks_target": {
            "condition": f"p_value > {KS_ALPHA}",
            "within_target": ks_within_target,
        },
        "cv_target": {
            "condition": f"cv > {CV_THRESHOLD}",
            "within_target": cv_within_target,
        },
        "overall_within_target": overall_within_target,
    }

    clipping_dict = {
        "lower_bound_ms": LOWER_DELAY_MS,
        "upper_bound_ms": UPPER_DELAY_MS,
        "lower_clip_count": summary["lower_clip_count"],
        "upper_clip_count": summary["upper_clip_count"],
        "clipped_fraction": summary["clipped_fraction"],
    }

    return {
        "analysis_schema_version": 1,
        "source_report": source_provenance,
        "sample_summary": summary,
        "clipping": clipping_dict,
        "conditional_model_test": conditional_test_dict,
        "aggregate_diagnostic": aggregate_fit,
        "cv_target": cv_target_dict,
        "roadmap_acceptance": roadmap_acceptance,
    }


def plot_distribution(
    analysis_dict: dict[str, Any],
    delays: list[float],
    detailed_samples: list[dict[str, Any]] | None,
    output_dir: Path,
) -> tuple[Path, Path | None]:
    """Generate distribution histogram plot and optional PIT ECDF plot.

    Args:
        analysis_dict: Built analysis result dictionary.
        delays: Raw delay milliseconds.
        detailed_samples: Per-sample metadata dicts or None.
        output_dir: Output directory path.

    Returns:
        Tuple of (path_to_distribution_png, path_to_pit_ecdf_png_or_None).
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    dist_png_path = output_dir / "timing_distribution.png"

    fig, ax = plt.subplots(figsize=(8, 5))
    _counts, _bin_edges, _patches = ax.hist(
        delays,
        bins=30,
        density=True,
        alpha=0.6,
        color="skyblue",
        edgecolor="navy",
        label="Observed Delays",
    )

    agg = analysis_dict.get("aggregate_diagnostic", {})
    if "fitted_mu" in agg and "fitted_sigma" in agg:
        mu = agg["fitted_mu"]
        sigma = agg["fitted_sigma"]
        x = np.linspace(min(delays), max(delays), 200)
        pdf = scipy.stats.lognorm.pdf(x, s=sigma, scale=math.exp(mu), loc=0)
        ax.plot(
            x,
            pdf,
            "r--",
            linewidth=2,
            label=f"Aggregate Fit (μ={mu:.2f}, σ={sigma:.2f}) [Diagnostic Only]",
        )

    ax.set_title("Action Timing Distribution (delay_ms)")
    ax.set_xlabel("Delay (ms)")
    ax.set_ylabel("Density")
    ax.legend(loc="upper right")
    ax.grid(True, linestyle=":", alpha=0.5)

    plt.tight_layout()
    plt.savefig(dist_png_path, dpi=150)
    plt.close(fig)

    pit_png_path: Path | None = None
    if detailed_samples is not None and len(detailed_samples) >= MIN_TIMING_SAMPLES:
        pit_seed = analysis_dict.get("conditional_model_test", {}).get("pit_seed", PIT_RANDOM_SEED)
        pit_values = compute_randomized_pit(detailed_samples, pit_seed=pit_seed)

        pit_png_path = output_dir / "timing_pit_ecdf.png"
        fig_pit, ax_pit = plt.subplots(figsize=(6, 5))

        sorted_pit = np.sort(pit_values)
        y_ecdf = np.arange(1, len(sorted_pit) + 1) / len(sorted_pit)

        ax_pit.plot(sorted_pit, y_ecdf, "b-", label="Randomized PIT ECDF")
        ax_pit.plot([0, 1], [0, 1], "r--", label="Uniform(0,1) Theoretical")

        p_val = analysis_dict.get("conditional_model_test", {}).get("p_value")
        p_val_str = f"{p_val:.4f}" if isinstance(p_val, float) else "N/A"
        ax_pit.set_title(f"PIT Empirical CDF (KS p={p_val_str})")
        ax_pit.set_xlabel("u_i (PIT Transform)")
        ax_pit.set_ylabel("Cumulative Probability")
        ax_pit.legend(loc="lower right")
        ax_pit.grid(True, linestyle=":", alpha=0.5)

        plt.tight_layout()
        plt.savefig(pit_png_path, dpi=150)
        plt.close(fig_pit)

    return dist_png_path, pit_png_path


def write_analysis_output(
    analysis_result: dict[str, Any],
    output_path: str | Path,
) -> None:
    """Atomically write analysis dictionary as JSON using strict serialization (allow_nan=False).

    Args:
        analysis_result: Structured analysis dictionary.
        output_path: Output target path for timing_analysis.json.

    Raises:
        ValueError: If JSON contains NaN or Infinity.
    """
    target_path = Path(output_path).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)

    json_text = json.dumps(
        analysis_result,
        indent=2,
        sort_keys=True,
        allow_nan=False,
    )

    temp_fd, temp_path_str = tempfile.mkstemp(
        dir=target_path.parent,
        prefix=f".tmp_{target_path.name}_",
    )
    temp_path = Path(temp_path_str)

    try:
        with os.fdopen(temp_fd, "w", encoding="utf-8") as f:
            f.write(json_text)
            f.flush()
            os.fsync(f.fileno())

        os.replace(temp_path, target_path)
    except Exception:
        if temp_path.exists():
            try:
                temp_path.unlink()
            except OSError:
                pass
        raise
