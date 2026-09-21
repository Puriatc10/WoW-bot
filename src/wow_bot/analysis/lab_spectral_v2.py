"""Lab Spectral Analysis V2 for WoW-bot humanizer interval sample series.

Computes power spectral density (PSD) using a direct DFT periodogram on humanizer interval series
(in milliseconds) and evaluates the log-log PSD slope against target 1/f noise bounds.
Isolated from pre-lab analysis modules, third-party libraries (numpy/scipy), time reads,
RNG, LLMs, and database I/O.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wow_bot.reporting.schema_v2 import SCHEMA_VERSION


class LabSpectralError(Exception):
    """Exception raised for lab spectral analysis errors."""


ANALYSIS_SCHEMA_VERSION: int = 1
TARGET_SLOPE_RANGE: tuple[float, float] = (-1.5, -0.5)
MIN_SAMPLES: int = 64


@dataclass(frozen=True)
class SpectralConfig:
    min_samples: int = MIN_SAMPLES
    detrend: bool = True
    window: str = "hann"  # "hann" or "rect"
    target_slope_range: tuple[float, float] = TARGET_SLOPE_RANGE

    def __post_init__(self) -> None:
        if isinstance(self.min_samples, bool) or not isinstance(self.min_samples, int):
            raise ValueError(f"min_samples must be an int, got {self.min_samples!r}")  # noqa: TRY004
        if self.min_samples < 8:
            raise ValueError(f"min_samples must be >= 8, got {self.min_samples}")

        if self.window not in {"hann", "rect"}:
            raise ValueError(f"window must be 'hann' or 'rect', got {self.window!r}")

        if (
            not isinstance(self.target_slope_range, tuple)
            or len(self.target_slope_range) != 2
            or not isinstance(self.target_slope_range[0], (int, float))
            or not isinstance(self.target_slope_range[1], (int, float))
            or isinstance(self.target_slope_range[0], bool)
            or isinstance(self.target_slope_range[1], bool)
            or not math.isfinite(self.target_slope_range[0])
            or not math.isfinite(self.target_slope_range[1])
        ):
            raise ValueError(
                f"target_slope_range must be a tuple of two finite floats, got {self.target_slope_range!r}"
            )

        low, high = float(self.target_slope_range[0]), float(self.target_slope_range[1])
        if low >= high:
            raise ValueError(
                f"target_slope_range low ({low}) must be strictly less than high ({high})"
            )


@dataclass(frozen=True)
class SpectralResult:
    skipped: bool
    reason: str
    sample_count: int
    slope: float | None
    slope_in_target_range: bool
    psd_frequencies: tuple[float, ...]
    psd_values: tuple[float, ...]
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
            if self.slope is not None:
                raise ValueError("slope must be None when skipped is True")
            if self.slope_in_target_range is not False:
                raise ValueError("slope_in_target_range must be False when skipped is True")
            if self.psd_frequencies != ():
                raise ValueError("psd_frequencies must be empty tuple when skipped is True")
            if self.psd_values != ():
                raise ValueError("psd_values must be empty tuple when skipped is True")
            if self.reason == "":
                raise ValueError("reason must be non-empty when skipped is True")
        else:
            if self.slope is None or not math.isfinite(self.slope):
                raise ValueError("slope must be a finite float when skipped is False")
            if not isinstance(self.psd_frequencies, tuple) or not isinstance(self.psd_values, tuple):
                raise ValueError("psd_frequencies and psd_values must be tuples when skipped is False")
            if len(self.psd_frequencies) != len(self.psd_values):
                raise ValueError(
                    f"psd_frequencies length ({len(self.psd_frequencies)}) must equal "
                    f"psd_values length ({len(self.psd_values)})"
                )
            if self.reason != "":
                raise ValueError("reason must be empty string when skipped is False")

    def to_json(self) -> dict[str, Any]:
        return {
            "skipped": self.skipped,
            "reason": self.reason,
            "sample_count": self.sample_count,
            "slope": self.slope,
            "slope_in_target_range": self.slope_in_target_range,
            "psd_frequencies": list(self.psd_frequencies),
            "psd_values": list(self.psd_values),
            "analysis_schema_version": self.analysis_schema_version,
        }


def analyze_intervals(
    intervals_ms: Sequence[float],
    *,
    config: SpectralConfig | None = None,
) -> SpectralResult:
    """Compute periodogram PSD and log-log slope for a sequence of interval samples in ms.

    O(N^2) direct DFT periodogram calculation suitable for bounded sample sizes (N <= 5000).

    Args:
        intervals_ms: Sequence of interval sample durations in milliseconds.
        config: Optional SpectralConfig override.

    Returns:
        SpectralResult instance.

    Raises:
        LabSpectralError: If interval values are non-numeric, non-finite, or negative.
    """
    if config is None:
        config = SpectralConfig()

    n = len(intervals_ms)
    for idx, val in enumerate(intervals_ms):
        if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
            raise LabSpectralError(
                f"Sample #{idx} in intervals_ms is non-numeric or non-finite: {val!r}"
            )
        if float(val) < 0.0:
            raise LabSpectralError(
                f"Sample #{idx} in intervals_ms must be non-negative (>= 0), got {val}"
            )

    if n < config.min_samples:
        return SpectralResult(
            skipped=True,
            reason=f"insufficient_samples:{n}",
            sample_count=n,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    samples = [float(v) for v in intervals_ms]

    # Detrending (subtract arithmetic mean)
    if config.detrend:
        mean_val = sum(samples) / n
        samples = [s - mean_val for s in samples]

    # Windowing
    if config.window == "hann":
        denom = n - 1
        samples = [
            s * 0.5 * (1.0 - math.cos(2.0 * math.pi * i / denom))
            for i, s in enumerate(samples)
        ]

    # Direct O(N^2) DFT periodogram computation
    # For k in [0, N//2]:
    #   X_k = sum_{n=0}^{N-1} x_n * exp(-2j*pi*k*n/N)
    #   P_k = (|X_k|^2) / N
    num_bins = n // 2 + 1
    freqs: list[float] = []
    psd_vals: list[float] = []

    two_pi_over_n = 2.0 * math.pi / n

    for k in range(num_bins):
        freqs.append(k / n)
        real_acc = 0.0
        imag_acc = 0.0
        angle_k = k * two_pi_over_n
        for i, val in enumerate(samples):
            angle = angle_k * i
            real_acc += val * math.cos(angle)
            imag_acc -= val * math.sin(angle)
        mag_sq = real_acc * real_acc + imag_acc * imag_acc
        psd_vals.append(mag_sq / n)

    # Compute log-log slope
    # Drop DC bin (k=0) and non-positive P_k
    fit_x: list[float] = []
    fit_y: list[float] = []

    for k in range(1, num_bins):
        f = freqs[k]
        p = psd_vals[k]
        if f > 0.0 and p > 0.0:
            fit_x.append(math.log10(f))
            fit_y.append(math.log10(p))

    if len(fit_x) < 2:
        return SpectralResult(
            skipped=True,
            reason="insufficient_psd_points",
            sample_count=n,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    m_x = sum(fit_x) / len(fit_x)
    m_y = sum(fit_y) / len(fit_y)

    num = sum((x - m_x) * (y - m_y) for x, y in zip(fit_x, fit_y, strict=True))
    den = sum((x - m_x) ** 2 for x in fit_x)

    if den <= 0.0 or not math.isfinite(den):
        return SpectralResult(
            skipped=True,
            reason="insufficient_psd_points",
            sample_count=n,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    slope = num / den

    if not math.isfinite(slope):
        return SpectralResult(
            skipped=True,
            reason="insufficient_psd_points",
            sample_count=n,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    low_bound, high_bound = config.target_slope_range
    slope_in_target = low_bound <= slope <= high_bound

    return SpectralResult(
        skipped=False,
        reason="",
        sample_count=n,
        slope=slope,
        slope_in_target_range=slope_in_target,
        psd_frequencies=tuple(freqs),
        psd_values=tuple(psd_vals),
    )


def analyze_report(
    report: dict[str, Any],
    *,
    config: SpectralConfig | None = None,
) -> SpectralResult:
    """Analyze humanizer interval samples contained within a schema v2 report dictionary.

    Args:
        report: Schema v2 report dictionary.
        config: Optional SpectralConfig override.

    Returns:
        SpectralResult instance.

    Raises:
        LabSpectralError: If report is not a dict or schema_version != 2.
    """
    if not isinstance(report, dict):
        raise LabSpectralError(f"report must be a dict, got {type(report).__name__}")

    ver = report.get("schema_version")
    if ver != SCHEMA_VERSION:
        raise LabSpectralError(
            f"Expected report schema_version == {SCHEMA_VERSION}, got {ver!r}"
        )

    humanizer_sec = report.get("humanizer")
    if humanizer_sec is None or not isinstance(humanizer_sec, dict):
        return SpectralResult(
            skipped=True,
            reason="no_humanizer_section",
            sample_count=0,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    samples = humanizer_sec.get("interval_samples_ms")
    if samples is None or not isinstance(samples, (list, tuple)):
        return SpectralResult(
            skipped=True,
            reason="no_humanizer_section",
            sample_count=0,
            slope=None,
            slope_in_target_range=False,
            psd_frequencies=(),
            psd_values=(),
        )

    return analyze_intervals(samples, config=config)


def write_result(result: SpectralResult, path: Path) -> None:
    """Atomically write SpectralResult to path as UTF-8 JSON.

    Args:
        result: SpectralResult instance.
        path: Path to target output JSON file.

    Raises:
        LabSpectralError: On failure to write file due to OSError.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.parent / f"{p.name}.tmp"
        content = json.dumps(result.to_json(), ensure_ascii=False, indent=2) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, p)
    except Exception as exc:
        raise LabSpectralError(f"Failed to write result to {path}: {exc}") from exc


__all__ = [
    "ANALYSIS_SCHEMA_VERSION",
    "MIN_SAMPLES",
    "TARGET_SLOPE_RANGE",
    "LabSpectralError",
    "SpectralConfig",
    "SpectralResult",
    "analyze_intervals",
    "analyze_report",
    "write_result",
]
