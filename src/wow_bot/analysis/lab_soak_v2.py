"""Lab Soak Analysis Engine V2 for WoW-bot long-running telemetry aggregation.

Aggregates operational resource snapshots into deterministic, self-contained
soak test reports with schema version 2. Isolated from pre-lab modules, I/O,
wall-clock time reads, randomness, third-party libraries, LLMs, and database access.
"""

from __future__ import annotations

import json
import math
import os
import statistics
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Protocol

SOAK_SCHEMA_VERSION: int = 2


class LabSoakError(Exception):
    """Exception raised for lab soak analysis errors."""


@dataclass(frozen=True)
class ResourceSnapshot:
    """Immutable operational snapshot captured at a given monotonic timestamp."""

    ts: float
    cpu_percent: float
    rss_bytes: int
    log_size_bytes: int

    def __post_init__(self) -> None:
        if isinstance(self.ts, bool) or not isinstance(self.ts, (int, float)) or not math.isfinite(self.ts) or self.ts < 0.0:
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts!r}")

        if (
            isinstance(self.cpu_percent, bool)
            or not isinstance(self.cpu_percent, (int, float))
            or not math.isfinite(self.cpu_percent)
            or self.cpu_percent < 0.0
        ):
            raise ValueError(f"cpu_percent must be a finite float >= 0.0, got {self.cpu_percent!r}")

        if isinstance(self.rss_bytes, bool) or not isinstance(self.rss_bytes, int) or self.rss_bytes < 0:
            raise ValueError(f"rss_bytes must be an int >= 0, got {self.rss_bytes!r}")

        if isinstance(self.log_size_bytes, bool) or not isinstance(self.log_size_bytes, int) or self.log_size_bytes < 0:
            raise ValueError(f"log_size_bytes must be an int >= 0, got {self.log_size_bytes!r}")


class ResourceSampler(Protocol):
    """Protocol abstraction for sampling operational resource usage."""

    def sample(self, now: float) -> ResourceSnapshot:
        ...


class ProcessHandle(Protocol):
    """Protocol abstraction for monitored process handle liveness and exit status."""

    def is_alive(self) -> bool:
        ...

    def exit_code(self) -> int | None:
        ...


@dataclass(frozen=True)
class SoakConfig:
    """Configuration options for soak telemetry aggregation and trend analysis."""

    sample_interval_s: float = 1.0
    max_samples: int = 500_000
    crash_exit_code: int = 2
    memory_slope_window: int = 300
    memory_growth_threshold_bytes_per_hour: float = 10_485_760.0
    log_growth_threshold_bytes_per_hour: float = 104_857_600.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.sample_interval_s, bool)
            or not isinstance(self.sample_interval_s, (int, float))
            or not math.isfinite(self.sample_interval_s)
            or float(self.sample_interval_s) <= 0.0
        ):
            raise ValueError(f"sample_interval_s must be a finite float > 0.0, got {self.sample_interval_s!r}")

        if isinstance(self.max_samples, bool) or not isinstance(self.max_samples, int) or self.max_samples < 1:
            raise ValueError(f"max_samples must be an int >= 1, got {self.max_samples!r}")

        if isinstance(self.crash_exit_code, bool) or not isinstance(self.crash_exit_code, int) or self.crash_exit_code < 1:
            raise ValueError(f"crash_exit_code must be an int >= 1, got {self.crash_exit_code!r}")

        if isinstance(self.memory_slope_window, bool) or not isinstance(self.memory_slope_window, int) or self.memory_slope_window < 10:
            raise ValueError(f"memory_slope_window must be an int >= 10, got {self.memory_slope_window!r}")

        if (
            isinstance(self.memory_growth_threshold_bytes_per_hour, bool)
            or not isinstance(self.memory_growth_threshold_bytes_per_hour, (int, float))
            or not math.isfinite(self.memory_growth_threshold_bytes_per_hour)
            or float(self.memory_growth_threshold_bytes_per_hour) < 0.0
        ):
            raise ValueError(
                f"memory_growth_threshold_bytes_per_hour must be a finite float >= 0.0, "
                f"got {self.memory_growth_threshold_bytes_per_hour!r}"
            )

        if (
            isinstance(self.log_growth_threshold_bytes_per_hour, bool)
            or not isinstance(self.log_growth_threshold_bytes_per_hour, (int, float))
            or not math.isfinite(self.log_growth_threshold_bytes_per_hour)
            or float(self.log_growth_threshold_bytes_per_hour) < 0.0
        ):
            raise ValueError(
                f"log_growth_threshold_bytes_per_hour must be a finite float >= 0.0, "
                f"got {self.log_growth_threshold_bytes_per_hour!r}"
            )


@dataclass(frozen=True)
class SoakSample:
    """Immutable telemetry sample recorded during a soak run."""

    ts: float
    cpu_percent: float
    rss_bytes: int
    log_size_bytes: int
    position_delta: float
    inventory_delta: int
    successful_actions_total: int
    reflex_ticks_total: int

    def __post_init__(self) -> None:
        if isinstance(self.ts, bool) or not isinstance(self.ts, (int, float)) or not math.isfinite(self.ts) or self.ts < 0.0:
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts!r}")

        if (
            isinstance(self.cpu_percent, bool)
            or not isinstance(self.cpu_percent, (int, float))
            or not math.isfinite(self.cpu_percent)
            or self.cpu_percent < 0.0
        ):
            raise ValueError(f"cpu_percent must be a finite float >= 0.0, got {self.cpu_percent!r}")

        if isinstance(self.rss_bytes, bool) or not isinstance(self.rss_bytes, int) or self.rss_bytes < 0:
            raise ValueError(f"rss_bytes must be an int >= 0, got {self.rss_bytes!r}")

        if isinstance(self.log_size_bytes, bool) or not isinstance(self.log_size_bytes, int) or self.log_size_bytes < 0:
            raise ValueError(f"log_size_bytes must be an int >= 0, got {self.log_size_bytes!r}")

        if (
            isinstance(self.position_delta, bool)
            or not isinstance(self.position_delta, (int, float))
            or not math.isfinite(self.position_delta)
            or float(self.position_delta) < 0.0
        ):
            raise ValueError(f"position_delta must be a finite float >= 0.0, got {self.position_delta!r}")

        if isinstance(self.inventory_delta, bool) or not isinstance(self.inventory_delta, int) or self.inventory_delta < 0:
            raise ValueError(f"inventory_delta must be an int >= 0, got {self.inventory_delta!r}")

        if (
            isinstance(self.successful_actions_total, bool)
            or not isinstance(self.successful_actions_total, int)
            or self.successful_actions_total < 0
        ):
            raise ValueError(
                f"successful_actions_total must be an int >= 0, got {self.successful_actions_total!r}"
            )

        if (
            isinstance(self.reflex_ticks_total, bool)
            or not isinstance(self.reflex_ticks_total, int)
            or self.reflex_ticks_total < 0
        ):
            raise ValueError(
                f"reflex_ticks_total must be an int >= 0, got {self.reflex_ticks_total!r}"
            )


@dataclass(frozen=True)
class SoakSummary:
    """Aggregate summary statistics over a sequence of soak samples."""

    sample_count: int
    duration_s: float
    cpu_mean_percent: float
    cpu_max_percent: float
    rss_start_bytes: int
    rss_end_bytes: int
    rss_peak_bytes: int
    rss_slope_bytes_per_hour: float
    rss_growth_suspect: bool
    log_start_bytes: int
    log_end_bytes: int
    log_slope_bytes_per_hour: float
    log_growth_suspect: bool
    position_delta_total: float
    inventory_delta_total: int
    successful_actions_total: int
    reflex_ticks_total: int
    reflex_tick_rate_hz: float

    def __post_init__(self) -> None:
        if isinstance(self.sample_count, bool) or not isinstance(self.sample_count, int) or self.sample_count < 0:
            raise ValueError(f"sample_count must be an int >= 0, got {self.sample_count!r}")

        if (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, (int, float))
            or not math.isfinite(self.duration_s)
            or float(self.duration_s) < 0.0
        ):
            raise ValueError(f"duration_s must be a finite float >= 0.0, got {self.duration_s!r}")

        for field_name in ("cpu_mean_percent", "cpu_max_percent"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val) or float(val) < 0.0:
                raise ValueError(f"{field_name} must be a finite float >= 0.0, got {val!r}")

        for field_name in ("rss_start_bytes", "rss_end_bytes", "rss_peak_bytes", "log_start_bytes", "log_end_bytes"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{field_name} must be an int >= 0, got {val!r}")

        for field_name in ("rss_slope_bytes_per_hour", "log_slope_bytes_per_hour", "reflex_tick_rate_hz"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, (int, float)) or not math.isfinite(val):
                raise ValueError(f"{field_name} must be a finite float, got {val!r}")

        if not isinstance(self.rss_growth_suspect, bool):
            raise ValueError(f"rss_growth_suspect must be a bool, got {type(self.rss_growth_suspect).__name__}")  # noqa: TRY004

        if not isinstance(self.log_growth_suspect, bool):
            raise ValueError(f"log_growth_suspect must be a bool, got {type(self.log_growth_suspect).__name__}")  # noqa: TRY004

        if (
            isinstance(self.position_delta_total, bool)
            or not isinstance(self.position_delta_total, (int, float))
            or not math.isfinite(self.position_delta_total)
            or float(self.position_delta_total) < 0.0
        ):
            raise ValueError(f"position_delta_total must be a finite float >= 0.0, got {self.position_delta_total!r}")

        for field_name in ("inventory_delta_total", "successful_actions_total", "reflex_ticks_total"):
            val = getattr(self, field_name)
            if isinstance(val, bool) or not isinstance(val, int) or val < 0:
                raise ValueError(f"{field_name} must be an int >= 0, got {val!r}")

    def to_json(self) -> dict[str, Any]:
        """Convert SoakSummary to a JSON-serializable dictionary."""
        return asdict(self)


@dataclass(frozen=True)
class SoakReport:
    """Self-contained soak report object."""

    schema_version: int
    session_id: str
    started_at: float
    finished_at: float
    summary: SoakSummary
    samples: tuple[SoakSample, ...]
    crash: bool
    crash_reason: str | None

    def __post_init__(self) -> None:
        if self.schema_version != SOAK_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SOAK_SCHEMA_VERSION}, got {self.schema_version!r}"
            )

        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError(f"session_id must be a non-empty string, got {self.session_id!r}")

        if (
            isinstance(self.started_at, bool)
            or not isinstance(self.started_at, (int, float))
            or not math.isfinite(self.started_at)
            or float(self.started_at) < 0.0
        ):
            raise ValueError(f"started_at must be a finite float >= 0.0, got {self.started_at!r}")

        if (
            isinstance(self.finished_at, bool)
            or not isinstance(self.finished_at, (int, float))
            or not math.isfinite(self.finished_at)
            or float(self.finished_at) < float(self.started_at)
        ):
            raise ValueError(
                f"finished_at ({self.finished_at!r}) must be a finite float >= started_at ({self.started_at!r})"
            )

        if not isinstance(self.summary, SoakSummary):
            raise ValueError(f"summary must be a SoakSummary instance, got {type(self.summary).__name__}")  # noqa: TRY004

        if not isinstance(self.samples, tuple):
            raise ValueError(f"samples must be a tuple of SoakSample objects, got {type(self.samples).__name__}")  # noqa: TRY004

        if not isinstance(self.crash, bool):
            raise ValueError(f"crash must be a bool, got {type(self.crash).__name__}")  # noqa: TRY004

        if self.crash:
            if self.crash_reason is None or not isinstance(self.crash_reason, str) or self.crash_reason == "":
                raise ValueError("crash_reason must be a non-empty string when crash is True")
        else:
            if self.crash_reason is not None:
                raise ValueError(f"crash_reason must be None when crash is False, got {self.crash_reason!r}")

    def to_json(self) -> dict[str, Any]:
        """Convert SoakReport to a JSON-serializable dictionary."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "session_id": self.session_id,
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
            "summary": self.summary.to_json(),
            "crash": self.crash,
            "crash_reason": self.crash_reason,
        }

        if len(self.samples) > 10_000:
            out["samples_omitted"] = True
        else:
            out["samples"] = [asdict(s) for s in self.samples]

        return out


def _calculate_slope(ts_values: Sequence[float], y_values: Sequence[float]) -> float:
    """Compute linear regression slope in y-units per hour against relative ts seconds."""
    n = len(ts_values)
    if n < 2:
        return 0.0

    t0 = ts_values[0]
    x = [t - t0 for t in ts_values]
    mean_x = sum(x) / n
    mean_y = sum(y_values) / n

    denom = sum((xi - mean_x) ** 2 for xi in x)
    if denom <= 0.0 or not math.isfinite(denom):
        return 0.0

    num = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y_values, strict=True))
    slope_per_s = num / denom
    if not math.isfinite(slope_per_s):
        return 0.0

    return float(slope_per_s * 3600.0)


def build_soak_summary(
    samples: Sequence[SoakSample],
    *,
    config: SoakConfig | None = None,
) -> SoakSummary:
    """Compute SoakSummary statistics from a sequence of SoakSample objects.

    Args:
        samples: Sequence of SoakSample.
        config: Optional SoakConfig override.

    Returns:
        SoakSummary dataclass instance.

    Raises:
        LabSoakError: If samples is empty or timestamps are not strictly increasing.
    """
    if config is None:
        config = SoakConfig()

    if not samples:
        raise LabSoakError("Cannot build summary from an empty sample sequence.")

    first = samples[0]
    last = samples[-1]

    # Validate strictly increasing ts
    for i in range(1, len(samples)):
        if samples[i].ts <= samples[i - 1].ts:
            raise LabSoakError(
                f"Sample timestamps must be strictly increasing: sample[{i-1}].ts={samples[i-1].ts} >= sample[{i}].ts={samples[i].ts}"
            )

    sample_count = len(samples)
    duration_s = float(last.ts - first.ts)

    cpu_percents = [s.cpu_percent for s in samples]
    cpu_mean_percent = float(statistics.mean(cpu_percents))
    cpu_max_percent = float(max(cpu_percents))

    rss_start_bytes = first.rss_bytes
    rss_end_bytes = last.rss_bytes
    rss_peak_bytes = max(s.rss_bytes for s in samples)

    # Trailing window for memory slope
    mem_window = samples[-config.memory_slope_window :]
    if len(mem_window) < 2:
        rss_slope = 0.0
    else:
        rss_slope = _calculate_slope(
            [s.ts for s in mem_window],
            [float(s.rss_bytes) for s in mem_window],
        )

    rss_growth_suspect = rss_slope > config.memory_growth_threshold_bytes_per_hour

    log_start_bytes = first.log_size_bytes
    log_end_bytes = last.log_size_bytes

    log_window = samples[-config.memory_slope_window :]
    if len(log_window) < 2:
        log_slope = 0.0
    else:
        log_slope = _calculate_slope(
            [s.ts for s in log_window],
            [float(s.log_size_bytes) for s in log_window],
        )

    log_growth_suspect = log_slope > config.log_growth_threshold_bytes_per_hour

    position_delta_total = float(sum(s.position_delta for s in samples))
    inventory_delta_total = int(last.inventory_delta - first.inventory_delta)
    successful_actions_total = int(last.successful_actions_total - first.successful_actions_total)
    reflex_ticks_total = int(last.reflex_ticks_total - first.reflex_ticks_total)

    if duration_s > 0.0:
        reflex_tick_rate_hz = float(reflex_ticks_total / duration_s)
    else:
        reflex_tick_rate_hz = 0.0

    return SoakSummary(
        sample_count=sample_count,
        duration_s=duration_s,
        cpu_mean_percent=cpu_mean_percent,
        cpu_max_percent=cpu_max_percent,
        rss_start_bytes=rss_start_bytes,
        rss_end_bytes=rss_end_bytes,
        rss_peak_bytes=rss_peak_bytes,
        rss_slope_bytes_per_hour=rss_slope,
        rss_growth_suspect=rss_growth_suspect,
        log_start_bytes=log_start_bytes,
        log_end_bytes=log_end_bytes,
        log_slope_bytes_per_hour=log_slope,
        log_growth_suspect=log_growth_suspect,
        position_delta_total=position_delta_total,
        inventory_delta_total=inventory_delta_total,
        successful_actions_total=successful_actions_total,
        reflex_ticks_total=reflex_ticks_total,
        reflex_tick_rate_hz=reflex_tick_rate_hz,
    )


def _empty_summary() -> SoakSummary:
    """Factory creating a zeroed SoakSummary for empty crash reports."""
    return SoakSummary(
        sample_count=0,
        duration_s=0.0,
        cpu_mean_percent=0.0,
        cpu_max_percent=0.0,
        rss_start_bytes=0,
        rss_end_bytes=0,
        rss_peak_bytes=0,
        rss_slope_bytes_per_hour=0.0,
        rss_growth_suspect=False,
        log_start_bytes=0,
        log_end_bytes=0,
        log_slope_bytes_per_hour=0.0,
        log_growth_suspect=False,
        position_delta_total=0.0,
        inventory_delta_total=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
        reflex_tick_rate_hz=0.0,
    )


def build_soak_report(
    samples: Sequence[SoakSample],
    *,
    session_id: str,
    crash: bool = False,
    crash_reason: str | None = None,
    config: SoakConfig | None = None,
) -> SoakReport:
    """Build a complete SoakReport object from telemetry samples and session metadata.

    Args:
        samples: Sequence of SoakSample.
        session_id: Non-empty session identifier.
        crash: Whether the run crashed.
        crash_reason: Description of crash if crashed.
        config: Optional SoakConfig override.

    Returns:
        SoakReport dataclass instance.

    Raises:
        LabSoakError: On invalid input state or empty non-crash sample sequence.
    """
    if config is None:
        config = SoakConfig()

    if not isinstance(session_id, str) or not session_id:
        raise LabSoakError(f"session_id must be a non-empty string, got {session_id!r}")

    if not samples:
        if crash:
            summary = _empty_summary()
            return SoakReport(
                schema_version=SOAK_SCHEMA_VERSION,
                session_id=session_id,
                started_at=0.0,
                finished_at=0.0,
                summary=summary,
                samples=(),
                crash=True,
                crash_reason=crash_reason,
            )
        raise LabSoakError("Cannot build non-crash soak report with empty sample sequence.")

    summary = build_soak_summary(samples, config=config)
    started_at = samples[0].ts
    finished_at = samples[-1].ts

    return SoakReport(
        schema_version=SOAK_SCHEMA_VERSION,
        session_id=session_id,
        started_at=started_at,
        finished_at=finished_at,
        summary=summary,
        samples=tuple(samples),
        crash=crash,
        crash_reason=crash_reason,
    )


def validate_soak_report_dict(data: dict[str, Any]) -> None:
    """Validate that a dictionary conforms strictly to the SoakReport schema v2 shape.

    Args:
        data: Dictionary to validate.

    Raises:
        LabSoakError: If data violates schema requirements, unknown keys, or type invariants.
    """
    if not isinstance(data, dict):
        raise LabSoakError(f"data must be a dict, got {type(data).__name__}")

    # Check top-level keys
    has_samples = "samples" in data
    has_samples_omitted = "samples_omitted" in data

    if has_samples_omitted:
        allowed_keys = {
            "schema_version",
            "session_id",
            "started_at",
            "finished_at",
            "summary",
            "samples_omitted",
            "crash",
            "crash_reason",
        }
        if data["samples_omitted"] is not True:
            raise LabSoakError("samples_omitted must be True if present.")
    else:
        allowed_keys = {
            "schema_version",
            "session_id",
            "started_at",
            "finished_at",
            "summary",
            "samples",
            "crash",
            "crash_reason",
        }

    data_keys = set(data.keys())
    missing_keys = allowed_keys - data_keys
    if missing_keys:
        raise LabSoakError(f"Missing required top-level keys in soak report dict: {sorted(missing_keys)}")

    extra_keys = data_keys - allowed_keys
    if extra_keys:
        raise LabSoakError(f"Unknown top-level keys in soak report dict: {sorted(extra_keys)}")

    # Check schema_version
    ver = data["schema_version"]
    if ver != SOAK_SCHEMA_VERSION:
        raise LabSoakError(f"Expected schema_version {SOAK_SCHEMA_VERSION}, got {ver!r}")

    # Check session_id
    sid = data["session_id"]
    if not isinstance(sid, str) or not sid:
        raise LabSoakError(f"session_id must be a non-empty str, got {sid!r}")

    # Check timestamps
    st = data["started_at"]
    ft = data["finished_at"]
    if isinstance(st, bool) or not isinstance(st, (int, float)) or not math.isfinite(st) or float(st) < 0.0:
        raise LabSoakError(f"started_at must be a finite float >= 0.0, got {st!r}")
    if isinstance(ft, bool) or not isinstance(ft, (int, float)) or not math.isfinite(ft) or float(ft) < float(st):
        raise LabSoakError(f"finished_at must be a finite float >= started_at, got {ft!r}")

    # Check crash/crash_reason
    crash = data["crash"]
    crash_reason = data["crash_reason"]
    if not isinstance(crash, bool):
        raise LabSoakError(f"crash must be a bool, got {type(crash).__name__}")

    if crash:
        if crash_reason is None or not isinstance(crash_reason, str) or crash_reason == "":
            raise LabSoakError("crash_reason must be non-empty string when crash is True")
    else:
        if crash_reason is not None:
            raise LabSoakError(f"crash_reason must be None when crash is False, got {crash_reason!r}")

    # Check summary
    summ = data["summary"]
    if not isinstance(summ, dict):
        raise LabSoakError(f"summary must be a dict, got {type(summ).__name__}")

    try:
        SoakSummary(**summ)
    except Exception as exc:
        raise LabSoakError(f"Invalid summary dictionary: {exc}") from exc

    # Check samples if present
    if has_samples:
        s_list = data["samples"]
        if not isinstance(s_list, list):
            raise LabSoakError(f"samples must be a list, got {type(s_list).__name__}")
        for idx, s_item in enumerate(s_list):
            if not isinstance(s_item, dict):
                raise LabSoakError(f"sample[{idx}] must be a dict, got {type(s_item).__name__}")
            try:
                SoakSample(**s_item)
            except Exception as exc:
                raise LabSoakError(f"Invalid sample[{idx}] dictionary: {exc}") from exc


def write_soak_report(report: SoakReport, path: Path) -> None:
    """Write SoakReport to path as UTF-8 formatted JSON atomically via os.replace.

    Args:
        report: SoakReport dataclass instance.
        path: Target file path.

    Raises:
        LabSoakError: On file write or directory creation failure.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = p.parent / f"{p.name}.tmp"
        content = json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, p)
    except OSError as exc:
        raise LabSoakError(f"Failed to write soak report to {path}: {exc}") from exc


__all__ = [
    "SOAK_SCHEMA_VERSION",
    "LabSoakError",
    "ProcessHandle",
    "ResourceSampler",
    "ResourceSnapshot",
    "SoakConfig",
    "SoakReport",
    "SoakSample",
    "SoakSummary",
    "build_soak_report",
    "build_soak_summary",
    "validate_soak_report_dict",
    "write_soak_report",
]
