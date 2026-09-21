"""Lab Timing Analysis V2 for WoW-bot humanizer interval sample series.

Performs descriptive statistical analysis, truncated lognormal Probability Integral Transform (PIT),
and Kolmogorov-Smirnov (KS) goodness-of-fit testing on humanizer interval series (in milliseconds).
Isolated from pre-lab analysis modules, third-party libraries (numpy/scipy), time reads,
LLMs, and database I/O.
"""

from __future__ import annotations

import json
import math
import os
import random
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from wow_bot.reporting.schema_v2 import SCHEMA_VERSION


class LabTimingError(Exception):
    """Exception raised for lab timing analysis errors."""


ANALYSIS_SCHEMA_VERSION: int = 1
PIT_RANDOM_SEED: int = 8202
KS_ALPHA: float = 0.05
CV_THRESHOLD: float = 0.3
MIN_SAMPLES: int = 64


@dataclass(frozen=True)
class TimingConfig:
    min_samples: int = MIN_SAMPLES
    pit_random_seed: int = PIT_RANDOM_SEED
    ks_alpha: float = KS_ALPHA
    cv_threshold: float = CV_THRESHOLD
    lognormal_mu: float = -2.0
    lognormal_sigma: float = 0.6
    clip_low_ms: float = 20.0
    clip_high_ms: float = 1500.0

    def __post_init__(self) -> None:
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise ValueError(f"min_samples must be an int, got {self.min_samples!r}")  # noqa: TRY004
        if self.min_samples < 8:
            raise ValueError(f"min_samples must be >= 8, got {self.min_samples}")

        if isinstance(self.pit_random_seed, bool) or not isinstance(self.pit_random_seed, int):
            raise ValueError(f"pit_random_seed must be an int, got {self.pit_random_seed!r}")  # noqa: TRY004

        if (
            isinstance(self.ks_alpha, bool)
            or not isinstance(self.ks_alpha, (int, float))
            or not math.isfinite(self.ks_alpha)
            or not (0.0 < float(self.ks_alpha) < 1.0)
        ):
            raise ValueError(f"ks_alpha must be a float in (0.0, 1.0), got {self.ks_alpha!r}")

        if (
            isinstance(self.cv_threshold, bool)
            or not isinstance(self.cv_threshold, (int, float))
            or not math.isfinite(self.cv_threshold)
            or float(self.cv_threshold) <= 0.0
        ):
            raise ValueError(f"cv_threshold must be a float > 0.0, got {self.cv_threshold!r}")

        if (
            isinstance(self.lognormal_mu, bool)
            or not isinstance(self.lognormal_mu, (int, float))
            or not math.isfinite(self.lognormal_mu)
        ):
            raise ValueError(f"lognormal_mu must be a finite float, got {self.lognormal_mu!r}")

        if (
            isinstance(self.lognormal_sigma, bool)
            or not isinstance(self.lognormal_sigma, (int, float))
            or not math.isfinite(self.lognormal_sigma)
            or float(self.lognormal_sigma) <= 0.0
        ):
            raise ValueError(f"lognormal_sigma must be a float > 0.0, got {self.lognormal_sigma!r}")

        if (
            isinstance(self.clip_low_ms, bool)
            or not isinstance(self.clip_low_ms, (int, float))
            or not math.isfinite(self.clip_low_ms)
            or float(self.clip_low_ms) <= 0.0
        ):
            raise ValueError(f"clip_low_ms must be a float > 0.0, got {self.clip_low_ms!r}")

        if (
            isinstance(self.clip_high_ms, bool)
            or not isinstance(self.clip_high_ms, (int, float))
            or not math.isfinite(self.clip_high_ms)
            or float(self.clip_high_ms) <= float(self.clip_low_ms)
        ):
            raise ValueError(
                f"clip_high_ms ({self.clip_high_ms}) must be a float > clip_low_ms ({self.clip_low_ms})"
            )


@dataclass(frozen=True)
class TimingResult:
    skipped: bool
    reason: str
    sample_count: int
    mean_ms: float | None
    median_ms: float | None
    std_ms: float | None
    cv: float | None
    cv_meets_threshold: bool
    pit_p_value: float | None
    pit_passes: bool
    ks_statistic: float | None
    ks_p_value: float | None
    ks_passes: bool
    analysis_schema_version: int = ANALYSIS_SCHEMA_VERSION

    def __post_init__(self) -> None:
        if not isinstance(self.skipped, bool):
            raise ValueError(f"skipped must be a bool, got {type(self.skipped).__name__}")  # noqa: TRY004
        if not isinstance(self.reason, str):
            raise ValueError(f"reason must be a str, got {type(self.reason).__name__}")  # noqa: TRY004
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int):
            raise ValueError(f"sample_count must be an int, got {self.sample_count!r}")  # noqa: TRY004
        if self.sample_count < 0:
            raise ValueError(f"sample_count must be >= 0, got {self.sample_count}")

        if self.analysis_schema_version != ANALYSIS_SCHEMA_VERSION:
            raise ValueError(
                f"analysis_schema_version must be {ANALYSIS_SCHEMA_VERSION}, "
                f"got {self.analysis_schema_version}"
            )

        if self.skipped:
            if (
                self.mean_ms is not None
                or self.median_ms is not None
                or self.std_ms is not None
                or self.cv is not None
                or self.pit_p_value is not None
                or self.ks_statistic is not None
                or self.ks_p_value is not None
            ):
                raise ValueError("All metric fields must be None when skipped is True")
            if self.cv_meets_threshold is not False or self.pit_passes is not False or self.ks_passes is not False:
                raise ValueError("All boolean pass flags must be False when skipped is True")
            if self.reason == "":
                raise ValueError("reason must be non-empty when skipped is True")
        else:
            if (
                self.mean_ms is None
                or self.median_ms is None
                or self.std_ms is None
                or self.cv is None
                or self.pit_p_value is None
                or self.ks_statistic is None
                or self.ks_p_value is None
            ):
                raise ValueError("All metric fields must be non-None when skipped is False")
            if self.reason != "":
                raise ValueError("reason must be empty string when skipped is False")

            if not (0.0 <= self.pit_p_value <= 1.0):
                raise ValueError(f"pit_p_value must be in [0.0, 1.0], got {self.pit_p_value}")
            if not (0.0 <= self.ks_p_value <= 1.0):
                raise ValueError(f"ks_p_value must be in [0.0, 1.0], got {self.ks_p_value}")
            if not (0.0 <= self.ks_statistic <= 1.0):
                raise ValueError(f"ks_statistic must be in [0.0, 1.0], got {self.ks_statistic}")
            if self.cv < 0.0:
                raise ValueError(f"cv must be >= 0.0, got {self.cv}")

    def to_json(self) -> dict[str, Any]:
        return {
            "skipped": self.skipped,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "mean_ms": self.mean_ms,
            "median_ms": self.median_ms,
            "std_ms": self.std_ms,
            "cv": self.cv,
            "cv_meets_threshold": self.cv_meets_threshold,
            "pit_p_value": self.pit_p_value,
            "pit_passes": self.pit_passes,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "ks_passes": self.ks_passes,
            "analysis_schema_version": self.analysis_schema_version,
        }


def _truncated_lognormal_cdf(
    x_ms: float,
    mu: float,
    sigma: float,
    clip_low_ms: float,
    clip_high_ms: float,
) -> float:
    """Compute truncated lognormal CDF value F_trunc(x_ms) in [0.0, 1.0].

    Note: x_ms, clip_low_ms, clip_high_ms are in milliseconds; mu is in log(seconds).
    Converting to seconds before evaluating ln(x) ensures alignment with T7.1 parameters.
    """
    if x_ms <= clip_low_ms:
        return 0.0
    if x_ms >= clip_high_ms:
        return 1.0

    x_sec = x_ms / 1000.0
    clip_low_sec = clip_low_ms / 1000.0
    clip_high_sec = clip_high_ms / 1000.0

    sqrt2 = math.sqrt(2.0)
    f_low = 0.5 * (1.0 + math.erf((math.log(clip_low_sec) - mu) / (sigma * sqrt2)))
    f_high = 0.5 * (1.0 + math.erf((math.log(clip_high_sec) - mu) / (sigma * sqrt2)))
    f_x = 0.5 * (1.0 + math.erf((math.log(x_sec) - mu) / (sigma * sqrt2)))

    denom = f_high - f_low
    if denom <= 0.0:
        return 0.0

    res = (f_x - f_low) / denom
    if res <= 0.0:
        return 0.0
    if res >= 1.0:
        return 1.0
    return res


def _kolmogorov_p_value(d: float, n: int) -> float:
    """Compute asymptotic Kolmogorov distribution p-value for KS statistic D and sample size N."""
    if d <= 0.0 or n <= 0:
        return 1.0

    x = math.sqrt(n) * d
    if x <= 0.0:
        return 1.0

    sum_val = 0.0
    for k in range(1, 101):
        term = ((-1) ** (k - 1)) * math.exp(-2.0 * (k * k) * (x * x))
        sum_val += term
        if abs(term) < 1e-15:
            break

    p = 2.0 * sum_val
    if p < 0.0:
        return 0.0
    if p > 1.0:
        return 1.0
    return p


def analyze_intervals(
    intervals_ms: Sequence[float],
    *,
    config: TimingConfig | None = None,
) -> TimingResult:
    """Analyze interval timing statistics and goodness-of-fit against truncated lognormal distribution.

    Args:
        intervals_ms: Sequence of interval sample durations in milliseconds.
        config: Optional TimingConfig override.

    Returns:
        TimingResult instance.

    Raises:
        LabTimingError: If interval values are non-numeric, non-finite, or negative.
    """
    if config is None:
        config = TimingConfig()

    n = len(intervals_ms)
    for idx, val in enumerate(intervals_ms):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise LabTimingError(
                f"Sample #{idx} in intervals_ms is non-numeric or non-finite: {val!r}"
            )
        if float(val) < 0.0:
            raise LabTimingError(
                f"Sample #{idx} in intervals_ms must be non-negative (>= 0), got {val}"
            )

    if n < config.min_samples:
        return TimingResult(
            skipped=True,
            reason=f"insufficient_samples:{n}",
            sample_count=n,
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

    samples = [float(v) for v in intervals_ms]

    # Descriptive statistics (population std)
    mean_val = sum(samples) / n
    med_val = float(median(samples))
    variance = sum((x - mean_val) ** 2 for x in samples) / n
    std_val = math.sqrt(variance)
    cv_val = std_val / mean_val if mean_val > 0.0 else 0.0
    cv_meets = cv_val > config.cv_threshold

    # PIT transformation with randomized tie breaking
    rng = random.Random(config.pit_random_seed)

    # Sort samples for tie-group identification
    sorted_samples = sorted(samples)

    # Group equal values
    unique_vals: list[float] = []
    val_counts: list[int] = []
    for x in sorted_samples:
        if not unique_vals or x != unique_vals[-1]:
            unique_vals.append(x)
            val_counts.append(1)
        else:
            val_counts[-1] += 1

    # Map each distinct x to PIT u_i values
    val_to_u: dict[float, list[float]] = {}
    for idx_u, val in enumerate(unique_vals):
        count = val_counts[idx_u]
        if count == 1:
            u_val = _truncated_lognormal_cdf(
                val,
                config.lognormal_mu,
                config.lognormal_sigma,
                config.clip_low_ms,
                config.clip_high_ms,
            )
            val_to_u[val] = [u_val]
        else:
            # Tie group: uniform tie-break between F_lower and F_upper
            if idx_u == 0:
                f_low = 0.0
            else:
                prev_val = unique_vals[idx_u - 1]
                f_low = _truncated_lognormal_cdf(
                    prev_val,
                    config.lognormal_mu,
                    config.lognormal_sigma,
                    config.clip_low_ms,
                    config.clip_high_ms,
                )
            f_high = _truncated_lognormal_cdf(
                val,
                config.lognormal_mu,
                config.lognormal_sigma,
                config.clip_low_ms,
                config.clip_high_ms,
            )

            if f_low >= f_high:
                val_to_u[val] = [f_high] * count
            else:
                val_to_u[val] = [rng.uniform(f_low, f_high) for _ in range(count)]

    # Collect u values in original order (consuming tied entries)
    val_indices: dict[float, int] = {v: 0 for v in unique_vals}
    u_vals: list[float] = []
    for x in samples:
        idx_entry = val_indices[x]
        u_vals.append(val_to_u[x][idx_entry])
        val_indices[x] = idx_entry + 1

    # Kolmogorov-Smirnov test against Uniform(0,1)
    sorted_u = sorted(u_vals)
    d_plus = max((i + 1) / n - u for i, u in enumerate(sorted_u))
    d_minus = max(u - i / n for i, u in enumerate(sorted_u))
    ks_stat = max(0.0, min(1.0, max(d_plus, d_minus)))

    p_val = _kolmogorov_p_value(ks_stat, n)

    pit_passes = p_val > config.ks_alpha
    ks_passes = p_val > config.ks_alpha

    return TimingResult(
        skipped=False,
        reason="",
        sample_count=n,
        mean_ms=mean_val,
        median_ms=med_val,
        std_ms=std_val,
        cv=cv_val,
        cv_meets_threshold=cv_meets,
        pit_p_value=p_val,
        pit_passes=pit_passes,
        ks_statistic=ks_stat,
        ks_p_value=p_val,
        ks_passes=ks_passes,
    )


def analyze_report(
    report: dict[str, Any],
    *,
    config: TimingConfig | None = None,
) -> TimingResult:
    """Analyze humanizer interval samples contained within a schema v2 report dictionary.

    Args:
        report: Schema v2 report dictionary.
        config: Optional TimingConfig override.

    Returns:
        TimingResult instance.

    Raises:
        LabTimingError: If report is not a dict or schema_version != 2.
    """
    if not isinstance(report, dict):
        raise LabTimingError(f"report must be a dict, got {type(report).__name__}")

    ver = report.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise LabTimingError(
            f"Expected report schema_version == {SCHEMA_VERSION}, got {ver!r}"
        )

    humanizer_sec = report.get("humanizer")
    if humanizer_sec is None or not isinstance(humanizer_sec, dict):
        return TimingResult(
            skipped=True,
            reason="no_humanizer_section",
            sample_count=0,
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

    samples = humanizer_sec.get("interval_samples_ms")
    if samples is None or not isinstance(samples, (list, tuple)):
        return TimingResult(
            skipped=True,
            reason="no_humanizer_section",
            sample_count=0,
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

    return analyze_intervals(samples, config=config)


def write_result(result: TimingResult, path: Path) -> None:
    """Atomically write TimingResult to path as UTF-8 JSON.

    Args:
        result: TimingResult instance.
        path: Path to target output JSON file.

    Raises:
        LabTimingError: On failure to write file due to OSError.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.parent / f"{p.name}.tmp"
        content = json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, p)
    except Exception as exc:
        raise LabTimingError(f"Failed to write result to {path}: {exc}") from exc


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "CV_THRESHOLD",
    "KS_ALPHA",
    "MIN_SAMPLES",
    "PIT_RANDOM_SEED",
    "LabTimingError",
    "TimingConfig",
    "TimingResult",
    "analyze_intervals",
    "analyze_report",
    "write_result",
]
