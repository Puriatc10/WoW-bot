"""Cross-session aggregate analysis for WoW-bot laboratory execution reports.

Aggregates individual session reports (ReportV2 and SoakReport) across multiple
sessions into a unified, versioned aggregate report (AGGREGATE_SCHEMA_VERSION = 1).

Seperates perception-agnostic execution metrics from outcome-dependent metrics.
Pure library with respect to OS and I/O beyond reading/writing JSON files.
Isolated from pre-lab analysis modules, spectral/timing modules, async runtime,
LLMs, and database access.
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from wow_bot.analysis.lab_soak_v2 import SoakReport, validate_soak_report_dict
from wow_bot.analysis.lab_timing_v2 import TimingConfig, analyze_intervals
from wow_bot.reporting.schema_v2 import validate_report_dict

AGGREGATE_SCHEMA_VERSION: int = 1

NON_CLAIMS: tuple[str, ...] = (
    (
        "We do NOT claim the agent farmed successfully in a real"
        " game. Perception was mocked; outcomes were not observed."
    ),
    "We do NOT claim anti-cheat evasion.",
    "We do NOT claim humanizer timing would evade detection.",
    (
        "We do NOT claim 24-hour stability. Only 1 hour was"
        " measured in Phase 12."
    ),
)


class AggregateError(Exception):
    """Exception raised for aggregate analysis errors."""


def _utc_now_iso() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _check_non_negative_int(val: Any, name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"{name} must be an int, got {val!r}")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")


def _check_optional_non_negative_float(val: Any, name: str) -> None:
    if val is not None:
        if isinstance(val, bool) or not isinstance(val, (int, float)):
            raise TypeError(f"{name} must be a float or None, got {val!r}")
        if not math.isfinite(float(val)) or float(val) < 0.0:
            raise ValueError(f"{name} must be a finite float >= 0.0, got {val}")


@dataclass(frozen=True)
class AggregateConfig:
    """Configuration options for cross-session report aggregation."""

    aggregate_filename: str = "aggregate_v1.json"
    require_report_v2: bool = False
    require_soak_report: bool = False
    include_non_claims: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.aggregate_filename, str) or not self.aggregate_filename:
            raise ValueError("aggregate_filename must be a non-empty string")
        if (
            "/" in self.aggregate_filename
            or "\\" in self.aggregate_filename
            or ".." in self.aggregate_filename
        ):
            raise ValueError(
                f"aggregate_filename must not contain '/', '\\', or '..', got {self.aggregate_filename!r}"
            )

        if not isinstance(self.require_report_v2, bool):
            raise TypeError(f"require_report_v2 must be a bool, got {type(self.require_report_v2).__name__}")

        if not isinstance(self.require_soak_report, bool):
            raise TypeError(
                f"require_soak_report must be a bool, got {type(self.require_soak_report).__name__}"
            )

        if not isinstance(self.include_non_claims, bool):
            raise TypeError(f"include_non_claims must be a bool, got {type(self.include_non_claims).__name__}")


@dataclass(frozen=True)
class PerceptionAgnosticMetrics:
    """Perception-agnostic execution metrics aggregated across sessions.

    Note on percentile fields: Cross-session percentiles cannot be exactly
    recomputed from per-session percentiles without raw samples.
    - reflex_tick_period_ms_mean: weighted mean by tick_count across sessions.
    - reflex_tick_period_ms_p95: max p99_tick_period_ms across sessions.
    - action_latency_ms_p50: weighted mean by actuator_result_count across sessions.
    - action_latency_ms_p95: max latency_ms_p95 across sessions.
    - strategist_latency_ms_p50: weighted mean by call_count across sessions.
    - strategist_latency_ms_p95: max latency_ms_p95 across sessions.
    """

    reflex_tick_count_total: int
    reflex_tick_period_ms_mean: float | None
    reflex_tick_period_ms_p95: float | None
    reflex_overrun_count_total: int
    reflex_sink_error_count_total: int
    action_intent_count_total: int
    action_result_count_total: int
    action_latency_ms_p50: float | None
    action_latency_ms_p95: float | None
    humanizer_action_count_total: int
    humanizer_interval_samples_ms_total: int
    humanizer_pause_count_total: int
    humanizer_miss_click_count_total: int
    strategist_call_count_total: int
    strategist_success_count_total: int
    strategist_invalid_json_count_total: int
    strategist_llm_error_count_total: int
    strategist_latency_ms_p50: float | None
    strategist_latency_ms_p95: float | None
    watchdog_transition_count_total: int
    watchdog_loop_detected_count_total: int
    watchdog_shutdown_count_total: int
    navigation_trip_count_total: int
    navigation_replan_count_total: int
    world_sync_count_total: int
    world_nodes_discovered_total: int
    world_entities_seen_total: int
    report_count: int
    soak_report_count: int

    def __post_init__(self) -> None:
        int_fields = (
            "reflex_tick_count_total",
            "reflex_overrun_count_total",
            "reflex_sink_error_count_total",
            "action_intent_count_total",
            "action_result_count_total",
            "humanizer_action_count_total",
            "humanizer_interval_samples_ms_total",
            "humanizer_pause_count_total",
            "humanizer_miss_click_count_total",
            "strategist_call_count_total",
            "strategist_success_count_total",
            "strategist_invalid_json_count_total",
            "strategist_llm_error_count_total",
            "watchdog_transition_count_total",
            "watchdog_loop_detected_count_total",
            "watchdog_shutdown_count_total",
            "navigation_trip_count_total",
            "navigation_replan_count_total",
            "world_sync_count_total",
            "world_nodes_discovered_total",
            "world_entities_seen_total",
            "report_count",
            "soak_report_count",
        )
        for field_name in int_fields:
            _check_non_negative_int(getattr(self, field_name), field_name)

        float_fields = (
            "reflex_tick_period_ms_mean",
            "reflex_tick_period_ms_p95",
            "action_latency_ms_p50",
            "action_latency_ms_p95",
            "strategist_latency_ms_p50",
            "strategist_latency_ms_p95",
        )
        for field_name in float_fields:
            _check_optional_non_negative_float(getattr(self, field_name), field_name)


@dataclass(frozen=True)
class InternalCounters:
    """Outcome-dependent or ambiguous internal counters, marked as non-research-findings.

    In Phase 12, schema v2 reports do not expose these fields (perception is mocked).
    All fields are None in Phase 12.
    """

    cycle_success_rate: float | None
    cycle_failure_rate: float | None
    mean_distance_to_target: float | None
    farm_outcome_count: int | None

    def __post_init__(self) -> None:
        if self.farm_outcome_count is not None:
            _check_non_negative_int(self.farm_outcome_count, "farm_outcome_count")

        for name in ("cycle_success_rate", "cycle_failure_rate"):
            val = getattr(self, name)
            if val is not None:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be a float or None, got {val!r}")
                f_val = float(val)
                if not (0.0 <= f_val <= 1.0):
                    raise ValueError(f"{name} must be in [0.0, 1.0], got {f_val}")

        if self.mean_distance_to_target is not None:
            _check_optional_non_negative_float(
                self.mean_distance_to_target, "mean_distance_to_target"
            )


@dataclass(frozen=True)
class SessionSource:
    """Metadata regarding a session directory included in the aggregate."""

    session_id: str
    session_dir: str
    has_report_v2: bool
    has_soak_report: bool
    report_schema_version: int | None
    soak_schema_version: int | None

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or not self.session_id:
            raise ValueError(f"session_id must be a non-empty string, got {self.session_id!r}")
        if not isinstance(self.session_dir, str) or not self.session_dir:
            raise ValueError(f"session_dir must be a non-empty string, got {self.session_dir!r}")

        if not isinstance(self.has_report_v2, bool):
            raise TypeError(f"has_report_v2 must be a bool, got {type(self.has_report_v2).__name__}")
        if not isinstance(self.has_soak_report, bool):
            raise TypeError(f"has_soak_report must be a bool, got {type(self.has_soak_report).__name__}")

        if self.report_schema_version is not None:
            if isinstance(self.report_schema_version, bool) or not isinstance(
                self.report_schema_version, int
            ):
                raise TypeError(
                    f"report_schema_version must be an int >= 1 or None, got {self.report_schema_version!r}"
                )
            if self.report_schema_version < 1:
                raise ValueError(f"report_schema_version must be >= 1, got {self.report_schema_version}")

        if self.soak_schema_version is not None:
            if isinstance(self.soak_schema_version, bool) or not isinstance(
                self.soak_schema_version, int
            ):
                raise TypeError(
                    f"soak_schema_version must be an int >= 1 or None, got {self.soak_schema_version!r}"
                )
            if self.soak_schema_version < 1:
                raise ValueError(f"soak_schema_version must be >= 1, got {self.soak_schema_version}")


@dataclass(frozen=True)
class CrashTrendSummary:
    """Summary of crashes across soak reports.

    When input_report_count == 0, total_crashes and crash_rate are None,
    distinguishing no soak reports from 0 crashes observed in >= 1 reports.
    """

    input_report_count: int
    total_crashes: int | None
    crash_rate: float | None
    crash_reasons: tuple[str, ...]

    def __post_init__(self) -> None:
        _check_non_negative_int(self.input_report_count, "input_report_count")
        if self.total_crashes is not None:
            _check_non_negative_int(self.total_crashes, "total_crashes")
        if self.crash_rate is not None:
            if isinstance(self.crash_rate, bool) or not isinstance(self.crash_rate, (int, float)):
                raise TypeError(f"crash_rate must be a float or None, got {self.crash_rate!r}")
            if not (0.0 <= float(self.crash_rate) <= 1.0):
                raise ValueError(f"crash_rate must be in [0.0, 1.0], got {self.crash_rate}")
        if not isinstance(self.crash_reasons, tuple):
            raise TypeError("crash_reasons must be a tuple")
        for r in self.crash_reasons:
            if not isinstance(r, str):
                raise TypeError("crash_reasons elements must be strings")

    def to_json(self) -> dict[str, Any]:
        return {
            "input_report_count": self.input_report_count,
            "total_crashes": self.total_crashes,
            "crash_rate": self.crash_rate,
            "crash_reasons": list(self.crash_reasons),
        }


@dataclass(frozen=True)
class RSSTrendSummary:
    """Summary of resident memory trends across soak reports.

    When input_report_count == 0, metrics are None.
    """

    input_report_count: int
    peak_bytes_max: int | None
    mean_slope_bytes_per_hour: float | None
    max_slope_bytes_per_hour: float | None
    any_growth_suspect: bool | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.input_report_count, "input_report_count")
        if self.peak_bytes_max is not None:
            _check_non_negative_int(self.peak_bytes_max, "peak_bytes_max")
        for name in ("mean_slope_bytes_per_hour", "max_slope_bytes_per_hour"):
            val = getattr(self, name)
            if val is not None:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be a float or None, got {val!r}")
                if not math.isfinite(float(val)):
                    raise ValueError(f"{name} must be finite, got {val}")
        if self.any_growth_suspect is not None and not isinstance(self.any_growth_suspect, bool):
            raise TypeError("any_growth_suspect must be a bool or None")

    def to_json(self) -> dict[str, Any]:
        return {
            "input_report_count": self.input_report_count,
            "peak_bytes_max": self.peak_bytes_max,
            "mean_slope_bytes_per_hour": self.mean_slope_bytes_per_hour,
            "max_slope_bytes_per_hour": self.max_slope_bytes_per_hour,
            "any_growth_suspect": self.any_growth_suspect,
        }


@dataclass(frozen=True)
class LogTrendSummary:
    """Summary of log growth trends across soak reports.

    When input_report_count == 0, metrics are None.
    """

    input_report_count: int
    mean_slope_bytes_per_hour: float | None
    max_slope_bytes_per_hour: float | None
    any_growth_suspect: bool | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.input_report_count, "input_report_count")
        for name in ("mean_slope_bytes_per_hour", "max_slope_bytes_per_hour"):
            val = getattr(self, name)
            if val is not None:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be a float or None, got {val!r}")
                if not math.isfinite(float(val)):
                    raise ValueError(f"{name} must be finite, got {val}")
        if self.any_growth_suspect is not None and not isinstance(self.any_growth_suspect, bool):
            raise TypeError("any_growth_suspect must be a bool or None")

    def to_json(self) -> dict[str, Any]:
        return {
            "input_report_count": self.input_report_count,
            "mean_slope_bytes_per_hour": self.mean_slope_bytes_per_hour,
            "max_slope_bytes_per_hour": self.max_slope_bytes_per_hour,
            "any_growth_suspect": self.any_growth_suspect,
        }


@dataclass(frozen=True)
class HumanizerFitSummary:
    """Summary of humanizer timing distribution fits across reports.

    Status:
        "no_input_reports": input_report_count == 0
        "no_interval_samples": input_report_count > 0 but no interval samples
        "evaluated": PIT/KS goodness-of-fit evaluated
        "insufficient_samples:<n>": sample count < min_samples threshold
    """

    input_report_count: int
    total_samples: int
    pit_p_value: float | None
    pit_passes: bool | None
    ks_statistic: float | None
    ks_p_value: float | None
    ks_passes: bool | None
    status: str

    def __post_init__(self) -> None:
        _check_non_negative_int(self.input_report_count, "input_report_count")
        _check_non_negative_int(self.total_samples, "total_samples")
        if not isinstance(self.status, str) or not self.status:
            raise ValueError("status must be a non-empty string")
        for name in ("pit_p_value", "ks_statistic", "ks_p_value"):
            val = getattr(self, name)
            if val is not None:
                if isinstance(val, bool) or not isinstance(val, (int, float)):
                    raise TypeError(f"{name} must be a float or None, got {val!r}")
                if not (0.0 <= float(val) <= 1.0):
                    raise ValueError(f"{name} must be in [0.0, 1.0], got {val}")
        for name in ("pit_passes", "ks_passes"):
            val = getattr(self, name)
            if val is not None and not isinstance(val, bool):
                raise TypeError(f"{name} must be a bool or None")

    def to_json(self) -> dict[str, Any]:
        return {
            "input_report_count": self.input_report_count,
            "total_samples": self.total_samples,
            "pit_p_value": self.pit_p_value,
            "pit_passes": self.pit_passes,
            "ks_statistic": self.ks_statistic,
            "ks_p_value": self.ks_p_value,
            "ks_passes": self.ks_passes,
            "status": self.status,
        }


@dataclass(frozen=True)
class StabilitySummaries:
    """Cross-session stability summaries over soak reports and humanizer series."""

    soak_report_count: int
    report_v2_count: int
    crash_trend: CrashTrendSummary
    rss_trend: RSSTrendSummary
    log_trend: LogTrendSummary
    humanizer_fit: HumanizerFitSummary

    def __post_init__(self) -> None:
        _check_non_negative_int(self.soak_report_count, "soak_report_count")
        _check_non_negative_int(self.report_v2_count, "report_v2_count")
        if not isinstance(self.crash_trend, CrashTrendSummary):
            raise TypeError("crash_trend must be CrashTrendSummary")
        if not isinstance(self.rss_trend, RSSTrendSummary):
            raise TypeError("rss_trend must be RSSTrendSummary")
        if not isinstance(self.log_trend, LogTrendSummary):
            raise TypeError("log_trend must be LogTrendSummary")
        if not isinstance(self.humanizer_fit, HumanizerFitSummary):
            raise TypeError("humanizer_fit must be HumanizerFitSummary")

    def to_json(self) -> dict[str, Any]:
        return {
            "soak_report_count": self.soak_report_count,
            "report_v2_count": self.report_v2_count,
            "crash_trend": self.crash_trend.to_json(),
            "rss_trend": self.rss_trend.to_json(),
            "log_trend": self.log_trend.to_json(),
            "humanizer_fit": self.humanizer_fit.to_json(),
        }


@dataclass(frozen=True)
class AggregateReport:
    """Versioned cross-session aggregate report artifact."""

    schema_version: int
    generated_at: str
    session_ids: tuple[str, ...]
    session_count: int
    perception_agnostic: PerceptionAgnosticMetrics
    internal_counters_not_research_findings: InternalCounters
    session_sources: tuple[SessionSource, ...]
    non_claims: tuple[str, ...]
    stability_summaries: StabilitySummaries | None = None

    def __post_init__(self) -> None:
        if self.schema_version != AGGREGATE_SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {AGGREGATE_SCHEMA_VERSION}, got {self.schema_version!r}"
            )

        if not isinstance(self.generated_at, str) or not self.generated_at:
            raise ValueError(f"generated_at must be a non-empty string, got {self.generated_at!r}")

        if not isinstance(self.session_ids, tuple):
            raise TypeError(f"session_ids must be a tuple, got {type(self.session_ids).__name__}")
        for s in self.session_ids:
            if not isinstance(s, str) or not s:
                raise ValueError("session_ids elements must be non-empty strings")
        if list(self.session_ids) != sorted(set(self.session_ids)):
            raise ValueError("session_ids must be sorted and unique")

        _check_non_negative_int(self.session_count, "session_count")
        if self.session_count != len(self.session_ids):
            raise ValueError(
                f"session_count ({self.session_count}) must equal len(session_ids) ({len(self.session_ids)})"
            )

        if not isinstance(self.perception_agnostic, PerceptionAgnosticMetrics):
            raise TypeError(
                f"perception_agnostic must be a PerceptionAgnosticMetrics instance, got {type(self.perception_agnostic).__name__}"
            )

        if not isinstance(self.internal_counters_not_research_findings, InternalCounters):
            raise TypeError(
                f"internal_counters_not_research_findings must be an InternalCounters instance, got {type(self.internal_counters_not_research_findings).__name__}"
            )

        if not isinstance(self.session_sources, tuple):
            raise TypeError(
                f"session_sources must be a tuple, got {type(self.session_sources).__name__}"
            )
        for src in self.session_sources:
            if not isinstance(src, SessionSource):
                raise TypeError("session_sources elements must be SessionSource instances")

        if not isinstance(self.non_claims, tuple):
            raise TypeError(f"non_claims must be a tuple, got {type(self.non_claims).__name__}")
        for claim in self.non_claims:
            if not isinstance(claim, str):
                raise TypeError("non_claims elements must be strings")

        if self.stability_summaries is not None and not isinstance(
            self.stability_summaries, StabilitySummaries
        ):
            raise TypeError(
                f"stability_summaries must be a StabilitySummaries instance or None, got {type(self.stability_summaries).__name__}"
            )

    def to_json(self) -> dict[str, Any]:
        """Convert AggregateReport to a JSON-serializable dictionary."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "generated_at": self.generated_at,
            "session_ids": list(self.session_ids),
            "session_count": self.session_count,
            "perception_agnostic": asdict(self.perception_agnostic),
            "internal_counters_not_research_findings": asdict(
                self.internal_counters_not_research_findings
            ),
            "session_sources": [asdict(s) for s in self.session_sources],
            "non_claims": list(self.non_claims),
        }
        if self.stability_summaries is not None:
            out["stability_summaries"] = self.stability_summaries.to_json()
        return out


def load_soak_report(path: Path) -> SoakReport:
    """Read path as UTF-8 JSON and parse into a validated SoakReport instance.

    Raises:
        AggregateError: On missing file, invalid JSON, or schema validation failure.
    """
    p = Path(path)
    if not p.exists():
        raise AggregateError(f"Soak report file does not exist: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregateError(f"Failed to parse JSON from soak report {p}: {exc}") from exc

    try:
        validate_soak_report_dict(data)
    except Exception as exc:
        raise AggregateError(f"Soak report validation failed for {p}: {exc}") from exc

    try:
        from wow_bot.analysis.lab_soak_v2 import SoakSample, SoakSummary

        summary = SoakSummary(**data["summary"])
        samples = (
            tuple(SoakSample(**s) for s in data["samples"])
            if "samples" in data and isinstance(data["samples"], list)
            else ()
        )
        return SoakReport(
            schema_version=data["schema_version"],
            session_id=data["session_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            summary=summary,
            samples=samples,
            crash=data["crash"],
            crash_reason=data["crash_reason"],
        )
    except Exception as exc:
        raise AggregateError(f"Failed to construct SoakReport from dict for {p}: {exc}") from exc


def load_report_v2(path: Path) -> dict[str, Any]:
    """Read path as UTF-8 JSON and validate against schema_v2 rules.

    Returns the raw report dictionary.

    Raises:
        AggregateError: On missing file, invalid JSON, or schema validation failure.
    """
    p = Path(path)
    if not p.exists():
        raise AggregateError(f"Report v2 file does not exist: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregateError(f"Failed to parse JSON from report v2 {p}: {exc}") from exc

    try:
        validate_report_dict(data)
    except Exception as exc:
        raise AggregateError(f"Report v2 validation failed for {p}: {exc}") from exc

    return cast(dict[str, Any], data)


def aggregate_sessions(
    session_dirs: Sequence[Path],
    *,
    config: AggregateConfig | None = None,
    now: str | None = None,
) -> AggregateReport:
    """Aggregate execution metrics across multiple session directories into an AggregateReport.

    Args:
        session_dirs: Sequence of session directory paths.
        config: Optional AggregateConfig override.
        now: Optional ISO 8601 UTC timestamp string override.

    Returns:
        AggregateReport dataclass instance.

    Raises:
        AggregateError: On missing session directory, missing/malformed session.json,
            or unsatisfied required report config options.
    """
    if config is None:
        config = AggregateConfig()

    generated_at = _utc_now_iso() if now is None else now

    session_sources: list[SessionSource] = []
    loaded_reports: list[dict[str, Any]] = []
    loaded_soak_reports: list[SoakReport] = []

    for s_dir_raw in session_dirs:
        s_dir = Path(s_dir_raw)
        if not s_dir.exists() or not s_dir.is_dir():
            raise AggregateError(f"Session directory does not exist or is not a directory: {s_dir}")

        session_json_path = s_dir / "session.json"
        if not session_json_path.exists():
            raise AggregateError(f"Session directory missing session.json: {s_dir}")

        try:
            session_data = json.loads(session_json_path.read_text(encoding="utf-8"))
        except Exception as exc:
            raise AggregateError(
                f"Failed to parse JSON from {session_json_path}: {exc}"
            ) from exc

        if not isinstance(session_data, dict) or "session_id" not in session_data:
            raise AggregateError(
                f"Invalid session.json structure in {session_json_path}: missing session_id"
            )

        session_id = session_data["session_id"]
        if not isinstance(session_id, str) or not session_id:
            raise AggregateError(
                f"Invalid session_id in {session_json_path}: must be a non-empty string"
            )

        report_v2_path = s_dir / "report_v2.json"
        soak_report_path = s_dir / "soak_report.json"

        has_report_v2 = report_v2_path.exists()
        has_soak_report = soak_report_path.exists()

        if config.require_report_v2 and not has_report_v2:
            raise AggregateError(f"Session {s_dir} missing required report_v2.json")

        if config.require_soak_report and not has_soak_report:
            raise AggregateError(f"Session {s_dir} missing required soak_report.json")

        report_schema_version: int | None = None
        if has_report_v2:
            r_dict = load_report_v2(report_v2_path)
            report_schema_version = r_dict["schema_version"]
            loaded_reports.append(r_dict)

        soak_schema_version: int | None = None
        if has_soak_report:
            s_report = load_soak_report(soak_report_path)
            soak_schema_version = s_report.schema_version
            loaded_soak_reports.append(s_report)

        session_sources.append(
            SessionSource(
                session_id=session_id,
                session_dir=str(s_dir),
                has_report_v2=has_report_v2,
                has_soak_report=has_soak_report,
                report_schema_version=report_schema_version,
                soak_schema_version=soak_schema_version,
            )
        )

    # Accumulate perception-agnostic metrics
    reflex_tick_count_total = 0
    reflex_overrun_count_total = 0
    reflex_sink_error_count_total = 0
    reflex_tick_period_ms_weighted_sum = 0.0
    reflex_tick_period_ms_weight_total = 0
    reflex_p95_values: list[float] = []

    action_intent_count_total = 0
    action_result_count_total = 0
    action_latency_p50_weighted_sum = 0.0
    action_latency_p50_weight_total = 0
    action_p95_values: list[float] = []

    humanizer_action_count_total = 0
    humanizer_interval_samples_ms_total = 0
    humanizer_pause_count_total = 0
    humanizer_miss_click_count_total = 0

    strategist_call_count_total = 0
    strategist_success_count_total = 0
    strategist_invalid_json_count_total = 0
    strategist_llm_error_count_total = 0
    strategist_latency_p50_weighted_sum = 0.0
    strategist_latency_p50_weight_total = 0
    strategist_p95_values: list[float] = []

    watchdog_transition_count_total = 0
    watchdog_loop_detected_count_total = 0
    watchdog_shutdown_count_total = 0

    navigation_trip_count_total = 0
    navigation_replan_count_total = 0

    world_sync_count_total = 0
    world_nodes_discovered_total = 0
    world_entities_seen_total = 0

    for report in loaded_reports:
        # Reflex
        if "reflex" in report and report["reflex"] is not None:
            ref = report["reflex"]
            t_count = ref.get("tick_count", 0)
            reflex_tick_count_total += t_count
            reflex_overrun_count_total += ref.get("overrun_count", 0)
            reflex_sink_error_count_total += ref.get("sink_error_count", 0)

            mean_tp = ref.get("mean_tick_period_ms")
            if mean_tp is not None and t_count > 0:
                reflex_tick_period_ms_weighted_sum += mean_tp * t_count
                reflex_tick_period_ms_weight_total += t_count

            p99_tp = ref.get("p99_tick_period_ms")
            if p99_tp is not None:
                reflex_p95_values.append(p99_tp)

        # Action
        if "action" in report and report["action"] is not None:
            act = report["action"]
            action_intent_count_total += act.get("actuator_intent_count", 0)
            res_count = act.get("actuator_result_count", 0)
            action_result_count_total += res_count

            p50_lat = act.get("latency_ms_p50")
            if p50_lat is not None and res_count > 0:
                action_latency_p50_weighted_sum += p50_lat * res_count
                action_latency_p50_weight_total += res_count

            p95_lat = act.get("latency_ms_p95")
            if p95_lat is not None:
                action_p95_values.append(p95_lat)

        # Humanizer
        if "humanizer" in report and report["humanizer"] is not None:
            hum = report["humanizer"]
            humanizer_action_count_total += hum.get("action_count", 0)
            samples = hum.get("interval_samples_ms", ())
            humanizer_interval_samples_ms_total += len(samples)
            humanizer_pause_count_total += hum.get("pause_count", 0)
            humanizer_miss_click_count_total += hum.get("miss_click_count", 0)

        # Strategist
        if "strategist" in report and report["strategist"] is not None:
            strat = report["strategist"]
            c_count = strat.get("call_count", 0)
            strategist_call_count_total += c_count
            strategist_success_count_total += strat.get("success_count", 0)
            strategist_invalid_json_count_total += strat.get("invalid_json_count", 0)
            strategist_llm_error_count_total += strat.get("llm_error_count", 0)

            p50_lat = strat.get("latency_ms_p50")
            if p50_lat is not None and c_count > 0:
                strategist_latency_p50_weighted_sum += p50_lat * c_count
                strategist_latency_p50_weight_total += c_count

            p95_lat = strat.get("latency_ms_p95")
            if p95_lat is not None:
                strategist_p95_values.append(p95_lat)

        # Watchdog
        if "watchdog" in report and report["watchdog"] is not None:
            wd = report["watchdog"]
            watchdog_transition_count_total += wd.get("transition_count", 0)
            watchdog_loop_detected_count_total += wd.get("loop_detected_count", 0)
            if wd.get("shutdown_requested") is True:
                watchdog_shutdown_count_total += 1

        # Navigation
        if "navigation" in report and report["navigation"] is not None:
            nav = report["navigation"]
            navigation_trip_count_total += nav.get("trip_count", 0)
            navigation_replan_count_total += nav.get("replan_count", 0)

        # World
        if "world" in report and report["world"] is not None:
            wld = report["world"]
            world_sync_count_total += wld.get("sync_count", 0)
            world_nodes_discovered_total += wld.get("nodes_discovered", 0)
            world_entities_seen_total += wld.get("entities_seen", 0)

    # Compute percentiles / weighted averages
    reflex_tick_period_ms_mean = (
        reflex_tick_period_ms_weighted_sum / reflex_tick_period_ms_weight_total
        if reflex_tick_period_ms_weight_total > 0
        else None
    )
    reflex_tick_period_ms_p95 = max(reflex_p95_values) if reflex_p95_values else None

    action_latency_ms_p50 = (
        action_latency_p50_weighted_sum / action_latency_p50_weight_total
        if action_latency_p50_weight_total > 0
        else None
    )
    action_latency_ms_p95 = max(action_p95_values) if action_p95_values else None

    strategist_latency_ms_p50 = (
        strategist_latency_p50_weighted_sum / strategist_latency_p50_weight_total
        if strategist_latency_p50_weight_total > 0
        else None
    )
    strategist_latency_ms_p95 = max(strategist_p95_values) if strategist_p95_values else None

    perception_agnostic = PerceptionAgnosticMetrics(
        reflex_tick_count_total=reflex_tick_count_total,
        reflex_tick_period_ms_mean=reflex_tick_period_ms_mean,
        reflex_tick_period_ms_p95=reflex_tick_period_ms_p95,
        reflex_overrun_count_total=reflex_overrun_count_total,
        reflex_sink_error_count_total=reflex_sink_error_count_total,
        action_intent_count_total=action_intent_count_total,
        action_result_count_total=action_result_count_total,
        action_latency_ms_p50=action_latency_ms_p50,
        action_latency_ms_p95=action_latency_ms_p95,
        humanizer_action_count_total=humanizer_action_count_total,
        humanizer_interval_samples_ms_total=humanizer_interval_samples_ms_total,
        humanizer_pause_count_total=humanizer_pause_count_total,
        humanizer_miss_click_count_total=humanizer_miss_click_count_total,
        strategist_call_count_total=strategist_call_count_total,
        strategist_success_count_total=strategist_success_count_total,
        strategist_invalid_json_count_total=strategist_invalid_json_count_total,
        strategist_llm_error_count_total=strategist_llm_error_count_total,
        strategist_latency_ms_p50=strategist_latency_ms_p50,
        strategist_latency_ms_p95=strategist_latency_ms_p95,
        watchdog_transition_count_total=watchdog_transition_count_total,
        watchdog_loop_detected_count_total=watchdog_loop_detected_count_total,
        watchdog_shutdown_count_total=watchdog_shutdown_count_total,
        navigation_trip_count_total=navigation_trip_count_total,
        navigation_replan_count_total=navigation_replan_count_total,
        world_sync_count_total=world_sync_count_total,
        world_nodes_discovered_total=world_nodes_discovered_total,
        world_entities_seen_total=world_entities_seen_total,
        report_count=len(loaded_reports),
        soak_report_count=len(loaded_soak_reports),
    )

    internal_counters = InternalCounters(
        cycle_success_rate=None,
        cycle_failure_rate=None,
        mean_distance_to_target=None,
        farm_outcome_count=None,
    )

    non_claims = NON_CLAIMS if config.include_non_claims else ()

    sorted_session_ids = tuple(sorted({src.session_id for src in session_sources}))

    stability_summaries = aggregate_stability_summaries(
        loaded_soak_reports, loaded_reports
    )

    return AggregateReport(
        schema_version=AGGREGATE_SCHEMA_VERSION,
        generated_at=generated_at,
        session_ids=sorted_session_ids,
        session_count=len(sorted_session_ids),
        perception_agnostic=perception_agnostic,
        internal_counters_not_research_findings=internal_counters,
        session_sources=tuple(session_sources),
        non_claims=non_claims,
        stability_summaries=stability_summaries,
    )


def aggregate_stability_summaries(
    soak_reports: Sequence[SoakReport],
    reports: Sequence[dict[str, Any]],
    *,
    timing_config: TimingConfig | None = None,
) -> StabilitySummaries:
    """Aggregate crash trends, RSS trends, log trends, and humanizer PIT/KS fits across reports.

    When no soak reports are supplied (input_report_count == 0), crash, RSS, and log trend metrics
    are None to distinguish absence of soak reports from 0 observed crashes/growth.
    When no report_v2 reports are supplied, humanizer fit status is 'no_input_reports'.
    """
    soak_count = len(soak_reports)
    if soak_count == 0:
        crash_trend = CrashTrendSummary(
            input_report_count=0,
            total_crashes=None,
            crash_rate=None,
            crash_reasons=(),
        )
        rss_trend = RSSTrendSummary(
            input_report_count=0,
            peak_bytes_max=None,
            mean_slope_bytes_per_hour=None,
            max_slope_bytes_per_hour=None,
            any_growth_suspect=None,
        )
        log_trend = LogTrendSummary(
            input_report_count=0,
            mean_slope_bytes_per_hour=None,
            max_slope_bytes_per_hour=None,
            any_growth_suspect=None,
        )
    else:
        # Sort soak reports by session_id for determinism
        sorted_soaks = sorted(soak_reports, key=lambda s: s.session_id)
        crashes = [s for s in sorted_soaks if s.crash]
        total_crashes = len(crashes)
        crash_rate = float(total_crashes / soak_count)
        crash_reasons = tuple(sorted({s.crash_reason for s in crashes if s.crash_reason}))
        crash_trend = CrashTrendSummary(
            input_report_count=soak_count,
            total_crashes=total_crashes,
            crash_rate=crash_rate,
            crash_reasons=crash_reasons,
        )

        rss_peaks = [s.summary.rss_peak_bytes for s in sorted_soaks]
        rss_slopes = [s.summary.rss_slope_bytes_per_hour for s in sorted_soaks]
        rss_suspects = [s.summary.rss_growth_suspect for s in sorted_soaks]
        rss_trend = RSSTrendSummary(
            input_report_count=soak_count,
            peak_bytes_max=max(rss_peaks),
            mean_slope_bytes_per_hour=float(sum(rss_slopes) / soak_count),
            max_slope_bytes_per_hour=max(rss_slopes),
            any_growth_suspect=any(rss_suspects),
        )

        log_slopes = [s.summary.log_slope_bytes_per_hour for s in sorted_soaks]
        log_suspects = [s.summary.log_growth_suspect for s in sorted_soaks]
        log_trend = LogTrendSummary(
            input_report_count=soak_count,
            mean_slope_bytes_per_hour=float(sum(log_slopes) / soak_count),
            max_slope_bytes_per_hour=max(log_slopes),
            any_growth_suspect=any(log_suspects),
        )

    report_count = len(reports)
    all_intervals: list[float] = []
    # Sort reports by session_id if available for determinism
    sorted_reports = sorted(
        reports,
        key=lambda r: str(r.get("meta", {}).get("session_id", ""))
        if isinstance(r.get("meta"), dict)
        else "",
    )
    for rep in sorted_reports:
        hum = rep.get("humanizer")
        if isinstance(hum, dict):
            samples = hum.get("interval_samples_ms")
            if isinstance(samples, (list, tuple)):
                for s in samples:
                    if isinstance(s, (int, float)) and not isinstance(s, bool):
                        all_intervals.append(float(s))

    if report_count == 0:
        humanizer_fit = HumanizerFitSummary(
            input_report_count=0,
            total_samples=0,
            pit_p_value=None,
            pit_passes=None,
            ks_statistic=None,
            ks_p_value=None,
            ks_passes=None,
            status="no_input_reports",
        )
    elif len(all_intervals) == 0:
        humanizer_fit = HumanizerFitSummary(
            input_report_count=report_count,
            total_samples=0,
            pit_p_value=None,
            pit_passes=None,
            ks_statistic=None,
            ks_p_value=None,
            ks_passes=None,
            status="no_interval_samples",
        )
    else:
        timing_res = analyze_intervals(all_intervals, config=timing_config)
        if timing_res.skipped:
            humanizer_fit = HumanizerFitSummary(
                input_report_count=report_count,
                total_samples=len(all_intervals),
                pit_p_value=None,
                pit_passes=False,
                ks_statistic=None,
                ks_p_value=None,
                ks_passes=False,
                status=timing_res.reason or "insufficient_samples",
            )
        else:
            humanizer_fit = HumanizerFitSummary(
                input_report_count=report_count,
                total_samples=len(all_intervals),
                pit_p_value=timing_res.pit_p_value,
                pit_passes=timing_res.pit_passes,
                ks_statistic=timing_res.ks_statistic,
                ks_p_value=timing_res.ks_p_value,
                ks_passes=timing_res.ks_passes,
                status="evaluated",
            )

    return StabilitySummaries(
        soak_report_count=soak_count,
        report_v2_count=report_count,
        crash_trend=crash_trend,
        rss_trend=rss_trend,
        log_trend=log_trend,
        humanizer_fit=humanizer_fit,
    )


def load_aggregate(path: Path) -> AggregateReport:
    """Read path as UTF-8 JSON and parse into a validated AggregateReport instance.

    Raises:
        AggregateError: On missing file, invalid JSON, or schema validation failure.
    """
    p = Path(path)
    if not p.exists():
        raise AggregateError(f"Aggregate report file does not exist: {p}")

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except Exception as exc:
        raise AggregateError(f"Failed to parse JSON from aggregate report {p}: {exc}") from exc

    if not isinstance(data, dict):
        raise AggregateError(f"Invalid aggregate structure in {p}: expected JSON object")

    if data.get("schema_version") != AGGREGATE_SCHEMA_VERSION:
        raise AggregateError(
            f"Unsupported aggregate schema_version in {p}: {data.get('schema_version')}"
        )

    try:
        pa_dict = data["perception_agnostic"]
        perception_agnostic = PerceptionAgnosticMetrics(**pa_dict)

        ic_dict = data["internal_counters_not_research_findings"]
        internal_counters = InternalCounters(**ic_dict)

        session_sources = tuple(
            SessionSource(**src) for src in data.get("session_sources", ())
        )

        stability_summaries: StabilitySummaries | None = None
        if "stability_summaries" in data and data["stability_summaries"] is not None:
            ss_data = data["stability_summaries"]
            crash_data = ss_data["crash_trend"]
            crash_trend = CrashTrendSummary(
                input_report_count=crash_data["input_report_count"],
                total_crashes=crash_data.get("total_crashes"),
                crash_rate=crash_data.get("crash_rate"),
                crash_reasons=tuple(crash_data.get("crash_reasons", ())),
            )
            rss_data = ss_data["rss_trend"]
            rss_trend = RSSTrendSummary(
                input_report_count=rss_data["input_report_count"],
                peak_bytes_max=rss_data.get("peak_bytes_max"),
                mean_slope_bytes_per_hour=rss_data.get("mean_slope_bytes_per_hour"),
                max_slope_bytes_per_hour=rss_data.get("max_slope_bytes_per_hour"),
                any_growth_suspect=rss_data.get("any_growth_suspect"),
            )
            log_data = ss_data["log_trend"]
            log_trend = LogTrendSummary(
                input_report_count=log_data["input_report_count"],
                mean_slope_bytes_per_hour=log_data.get("mean_slope_bytes_per_hour"),
                max_slope_bytes_per_hour=log_data.get("max_slope_bytes_per_hour"),
                any_growth_suspect=log_data.get("any_growth_suspect"),
            )
            hum_data = ss_data["humanizer_fit"]
            humanizer_fit = HumanizerFitSummary(
                input_report_count=hum_data["input_report_count"],
                total_samples=hum_data["total_samples"],
                pit_p_value=hum_data.get("pit_p_value"),
                pit_passes=hum_data.get("pit_passes"),
                ks_statistic=hum_data.get("ks_statistic"),
                ks_p_value=hum_data.get("ks_p_value"),
                ks_passes=hum_data.get("ks_passes"),
                status=hum_data.get("status", "evaluated"),
            )
            stability_summaries = StabilitySummaries(
                soak_report_count=ss_data.get("soak_report_count", 0),
                report_v2_count=ss_data.get("report_v2_count", 0),
                crash_trend=crash_trend,
                rss_trend=rss_trend,
                log_trend=log_trend,
                humanizer_fit=humanizer_fit,
            )

        return AggregateReport(
            schema_version=data["schema_version"],
            generated_at=data["generated_at"],
            session_ids=tuple(data.get("session_ids", ())),
            session_count=data["session_count"],
            perception_agnostic=perception_agnostic,
            internal_counters_not_research_findings=internal_counters,
            session_sources=session_sources,
            non_claims=tuple(data.get("non_claims", ())),
            stability_summaries=stability_summaries,
        )
    except Exception as exc:
        raise AggregateError(f"Failed to construct AggregateReport from dict for {p}: {exc}") from exc


def write_aggregate(report: AggregateReport, path: Path) -> None:
    """Write AggregateReport to path as UTF-8 formatted JSON atomically via os.replace.

    Raises:
        AggregateError: On file write or directory creation failure.
    """
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        tmp_p = p.parent / f"{p.name}.tmp"
        content = json.dumps(report.to_json(), ensure_ascii=False, indent=2, sort_keys=True) + "\n"
        tmp_p.write_text(content, encoding="utf-8")
        os.replace(tmp_p, p)
    except OSError as exc:
        raise AggregateError(f"Failed to write aggregate report to {path}: {exc}") from exc


def run_aggregate(
    session_dirs: Sequence[Path],
    *,
    output_dir: Path,
    config: AggregateConfig | None = None,
    now: str | None = None,
) -> Path:
    """Convenience function running aggregation and writing output artifact to disk.

    Returns the path to the written aggregate JSON file.
    """
    if config is None:
        config = AggregateConfig()

    report = aggregate_sessions(session_dirs, config=config, now=now)
    output_path = Path(output_dir) / config.aggregate_filename
    write_aggregate(report, output_path)
    return output_path


__all__ = [
    "AGGREGATE_SCHEMA_VERSION",
    "NON_CLAIMS",
    "AggregateConfig",
    "AggregateError",
    "AggregateReport",
    "CrashTrendSummary",
    "HumanizerFitSummary",
    "InternalCounters",
    "LogTrendSummary",
    "PerceptionAgnosticMetrics",
    "RSSTrendSummary",
    "SessionSource",
    "StabilitySummaries",
    "aggregate_sessions",
    "aggregate_stability_summaries",
    "load_aggregate",
    "load_report_v2",
    "load_soak_report",
    "run_aggregate",
    "write_aggregate",
]
