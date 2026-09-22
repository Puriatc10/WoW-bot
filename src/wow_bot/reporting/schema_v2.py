"""Report Schema V2 definitions and validator for WoW-bot lab reporting.

Defines pure, frozen dataclasses representing the aggregated laboratory execution report,
along with the single validation function `validate_report_dict` and conversion helpers.
Isolated from I/O, clock reads, randomness, database access, and domain modules
(except strategist prompts/vocab constants).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, fields
from datetime import datetime
from typing import Any

from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS
from wow_bot.strategist.vocab_v2 import RejectionReason


class SchemaV2Error(Exception):
    """Exception raised for schema version 2 validation errors."""


SCHEMA_VERSION: int = 2

REQUIRED_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"schema_version", "meta"})

OPTIONAL_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "perception",
        "action",
        "reflex",
        "navigation",
        "combat",
        "humanizer",
        "strategist",
        "watchdog",
        "world",
    }
)

ALL_TOP_LEVEL_KEYS: frozenset[str] = REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS

REPORT_V2_JSON_SCHEMA_PATH: str = "schemas/report_v2.json"

_CPU_COUNT = os.cpu_count()
CPU_PERCENT_CEILING: float = float("inf") if _CPU_COUNT is None else float(100.0 * _CPU_COUNT)

_VALID_REJECTION_REASONS: frozenset[str] = frozenset(r.value for r in RejectionReason)


def _check_non_negative_int(val: Any, name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(f"{name} must be an int, got {val!r}")
    if val < 0:
        raise ValueError(f"{name} must be >= 0, got {val}")


def _check_non_negative_float(val: Any, name: str) -> None:
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise TypeError(f"{name} must be a float or int, got {val!r}")
    if float(val) < 0.0:
        raise ValueError(f"{name} must be >= 0.0, got {val}")


def _check_optional_non_negative_float(val: Any, name: str) -> None:
    if val is not None:
        _check_non_negative_float(val, name)


def _parse_iso_8601(ts: str, name: str) -> datetime:
    if not isinstance(ts, str):
        raise TypeError(f"{name} must be an ISO 8601 string, got {ts!r}")
    try:
        if ts.endswith("Z"):
            ts_iso = ts[:-1] + "+00:00"
        else:
            ts_iso = ts
        return datetime.fromisoformat(ts_iso)
    except (ValueError, TypeError) as e:
        raise ValueError(f"Invalid ISO 8601 string for {name}: {ts!r}") from e


@dataclass(frozen=True)
class MetaSection:
    session_id: str
    mode: str
    started_at: str
    stopped_at: str | None
    stop_reason: str | None
    config_snapshot: dict[str, Any]
    event_count: int

    def __post_init__(self) -> None:
        if not isinstance(self.session_id, str) or len(self.session_id) == 0:
            raise ValueError("session_id must be a non-empty string")
        if self.mode not in {"MOCK", "LAB"}:
            raise ValueError(f"mode must be 'MOCK' or 'LAB', got {self.mode!r}")

        parsed_started = _parse_iso_8601(self.started_at, "started_at")

        if self.stopped_at is not None:
            parsed_stopped = _parse_iso_8601(self.stopped_at, "stopped_at")
            if parsed_stopped < parsed_started:
                raise ValueError(
                    f"stopped_at ({self.stopped_at}) must be >= started_at ({self.started_at})"
                )

        if self.stop_reason is not None and not isinstance(self.stop_reason, str):
            raise ValueError("stop_reason must be a string or None")

        if not isinstance(self.config_snapshot, dict):
            raise TypeError("config_snapshot must be a dict")

        _check_non_negative_int(self.event_count, "event_count")


@dataclass(frozen=True)
class PerceptionSection:
    source: str
    game_state_count: int
    confidence_mean: float | None
    known_fields: tuple[str, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.source, str) or len(self.source) == 0:
            raise ValueError("source must be a non-empty string")

        _check_non_negative_int(self.game_state_count, "game_state_count")

        if self.confidence_mean is not None:
            if isinstance(self.confidence_mean, bool) or not isinstance(
                self.confidence_mean, (int, float)
            ):
                raise ValueError("confidence_mean must be a float or None")
            val = float(self.confidence_mean)
            if not (0.0 <= val <= 1.0):
                raise ValueError(f"confidence_mean must be in [0.0, 1.0], got {val}")

        if not isinstance(self.known_fields, tuple):
            raise TypeError("known_fields must be a tuple")
        for f in self.known_fields:
            if not isinstance(f, str):
                raise TypeError("known_fields elements must be strings")
        if list(self.known_fields) != sorted(set(self.known_fields)):
            raise ValueError("known_fields must be sorted and unique")


@dataclass(frozen=True)
class ActionSection:
    actuator_intent_count: int
    actuator_result_count: int
    rejected_count: int
    failed_count: int
    success_count: int
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    latency_ms_max: float | None
    status_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        _check_non_negative_int(self.actuator_intent_count, "actuator_intent_count")
        _check_non_negative_int(self.actuator_result_count, "actuator_result_count")
        _check_non_negative_int(self.rejected_count, "rejected_count")
        _check_non_negative_int(self.failed_count, "failed_count")
        _check_non_negative_int(self.success_count, "success_count")

        _check_optional_non_negative_float(self.latency_ms_p50, "latency_ms_p50")
        _check_optional_non_negative_float(self.latency_ms_p95, "latency_ms_p95")
        _check_optional_non_negative_float(self.latency_ms_max, "latency_ms_max")

        if not isinstance(self.status_counts, Mapping):
            raise TypeError("status_counts must be a Mapping")
        for k, v in self.status_counts.items():
            if k not in {"success", "failed", "timeout"}:
                raise ValueError(
                    f"status_counts key must be in {{'success', 'failed', 'timeout'}}, got {k!r}"
                )
            _check_non_negative_int(v, f"status_counts[{k!r}]")


@dataclass(frozen=True)
class ReflexSection:
    tick_count: int
    mean_tick_period_ms: float | None
    p99_tick_period_ms: float | None
    overrun_count: int
    signal_counts: Mapping[str, int]
    sink_error_count: int
    source_error_count: int

    def __post_init__(self) -> None:
        _check_non_negative_int(self.tick_count, "tick_count")
        _check_optional_non_negative_float(self.mean_tick_period_ms, "mean_tick_period_ms")
        _check_optional_non_negative_float(self.p99_tick_period_ms, "p99_tick_period_ms")
        _check_non_negative_int(self.overrun_count, "overrun_count")
        _check_non_negative_int(self.sink_error_count, "sink_error_count")
        _check_non_negative_int(self.source_error_count, "source_error_count")

        if not isinstance(self.signal_counts, Mapping):
            raise TypeError("signal_counts must be a Mapping")
        for k, v in self.signal_counts.items():
            if not isinstance(k, str):
                raise TypeError(f"signal_counts key must be a string, got {k!r}")
            _check_non_negative_int(v, f"signal_counts[{k!r}]")


@dataclass(frozen=True)
class NavigationSection:
    trip_count: int
    success_count: int
    failure_count: int
    replan_count: int
    segment_count: int
    total_duration_s: float
    mean_trip_duration_s: float | None
    hard_failure_count: int
    telemetry_sample_count: int
    cpu_percent_p95: float | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.trip_count, "trip_count")
        _check_non_negative_int(self.success_count, "success_count")
        _check_non_negative_int(self.failure_count, "failure_count")
        _check_non_negative_int(self.replan_count, "replan_count")
        _check_non_negative_int(self.segment_count, "segment_count")
        _check_non_negative_float(self.total_duration_s, "total_duration_s")
        _check_optional_non_negative_float(self.mean_trip_duration_s, "mean_trip_duration_s")
        _check_non_negative_int(self.hard_failure_count, "hard_failure_count")
        _check_non_negative_int(self.telemetry_sample_count, "telemetry_sample_count")

        if self.cpu_percent_p95 is not None:
            if isinstance(self.cpu_percent_p95, bool) or not isinstance(
                self.cpu_percent_p95, (int, float)
            ):
                raise ValueError("cpu_percent_p95 must be a float or None")
            val = float(self.cpu_percent_p95)
            if not (0.0 <= val <= CPU_PERCENT_CEILING):
                raise ValueError(
                    f"cpu_percent_p95 must be in [0.0, {CPU_PERCENT_CEILING}], got {val}"
                )


@dataclass(frozen=True)
class CombatSection:
    encounter_count: int
    win_count: int
    loss_count: int
    flee_count: int
    timeout_count: int
    interrupt_attempt_count: int
    interrupt_success_count: int
    defensive_cast_count: int
    mean_encounter_duration_s: float | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.encounter_count, "encounter_count")
        _check_non_negative_int(self.win_count, "win_count")
        _check_non_negative_int(self.loss_count, "loss_count")
        _check_non_negative_int(self.flee_count, "flee_count")
        _check_non_negative_int(self.timeout_count, "timeout_count")
        _check_non_negative_int(self.interrupt_attempt_count, "interrupt_attempt_count")
        _check_non_negative_int(self.interrupt_success_count, "interrupt_success_count")
        _check_non_negative_int(self.defensive_cast_count, "defensive_cast_count")
        _check_optional_non_negative_float(
            self.mean_encounter_duration_s, "mean_encounter_duration_s"
        )

        outcome_sum = self.win_count + self.loss_count + self.flee_count + self.timeout_count
        if outcome_sum != self.encounter_count:
            raise ValueError(
                f"win ({self.win_count}) + loss ({self.loss_count}) + flee ({self.flee_count}) "
                f"+ timeout ({self.timeout_count}) must equal encounter_count "
                f"({self.encounter_count}), got {outcome_sum}"
            )


@dataclass(frozen=True)
class HumanizerSection:
    action_count: int
    interval_samples_ms: tuple[float, ...]
    pause_count: int
    pause_duration_ms_mean: float | None
    miss_click_count: int
    config_snapshot: dict[str, Any]

    def __post_init__(self) -> None:
        _check_non_negative_int(self.action_count, "action_count")

        if not isinstance(self.interval_samples_ms, tuple):
            raise TypeError("interval_samples_ms must be a tuple")
        for sample in self.interval_samples_ms:
            _check_non_negative_float(sample, "interval_samples_ms element")

        _check_non_negative_int(self.pause_count, "pause_count")
        _check_optional_non_negative_float(self.pause_duration_ms_mean, "pause_duration_ms_mean")
        _check_non_negative_int(self.miss_click_count, "miss_click_count")

        if not isinstance(self.config_snapshot, dict):
            raise TypeError("config_snapshot must be a dict")


@dataclass(frozen=True)
class StrategistSection:
    call_count: int
    success_count: int
    blocked_by_cooldown_count: int
    invalid_json_count: int
    llm_error_count: int
    vocab_rejected_count: int
    prompt_hash_count: int
    unique_prompt_hashes: tuple[str, ...]
    latency_ms_p50: float | None
    latency_ms_p95: float | None
    goal_counts: Mapping[str, int]
    rejection_reason_counts: Mapping[str, int]

    def __post_init__(self) -> None:
        _check_non_negative_int(self.call_count, "call_count")
        _check_non_negative_int(self.success_count, "success_count")
        _check_non_negative_int(self.blocked_by_cooldown_count, "blocked_by_cooldown_count")
        _check_non_negative_int(self.invalid_json_count, "invalid_json_count")
        _check_non_negative_int(self.llm_error_count, "llm_error_count")
        _check_non_negative_int(self.vocab_rejected_count, "vocab_rejected_count")
        _check_non_negative_int(self.prompt_hash_count, "prompt_hash_count")

        if not isinstance(self.unique_prompt_hashes, tuple):
            raise TypeError("unique_prompt_hashes must be a tuple")
        for h in self.unique_prompt_hashes:
            if not isinstance(h, str):
                raise TypeError("unique_prompt_hashes elements must be strings")
        if list(self.unique_prompt_hashes) != sorted(set(self.unique_prompt_hashes)):
            raise ValueError("unique_prompt_hashes must be sorted and unique")

        _check_optional_non_negative_float(self.latency_ms_p50, "latency_ms_p50")
        _check_optional_non_negative_float(self.latency_ms_p95, "latency_ms_p95")

        if not isinstance(self.goal_counts, Mapping):
            raise TypeError("goal_counts must be a Mapping")
        allowed_goals_set = set(ALLOWED_GOALS)
        for k, v in self.goal_counts.items():
            if k not in allowed_goals_set:
                raise ValueError(f"goal_counts key {k!r} is not in ALLOWED_GOALS")
            _check_non_negative_int(v, f"goal_counts[{k!r}]")

        if not isinstance(self.rejection_reason_counts, Mapping):
            raise TypeError("rejection_reason_counts must be a Mapping")
        for k, v in self.rejection_reason_counts.items():
            if k not in _VALID_REJECTION_REASONS:
                raise ValueError(
                    f"rejection_reason_counts key {k!r} is not a valid RejectionReason"
                )
            _check_non_negative_int(v, f"rejection_reason_counts[{k!r}]")


@dataclass(frozen=True)
class WatchdogSection:
    transition_count: int
    final_health_state: str
    transition_timeline: tuple[Mapping[str, str], ...]
    loop_detected_count: int
    shutdown_requested: bool
    shutdown_exit_code: int | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.transition_count, "transition_count")

        if self.final_health_state not in {"healthy", "degraded", "critical"}:
            raise ValueError(
                f"final_health_state must be in {{'healthy', 'degraded', 'critical'}}, "
                f"got {self.final_health_state!r}"
            )

        if not isinstance(self.transition_timeline, tuple):
            raise TypeError("transition_timeline must be a tuple")
        for entry in self.transition_timeline:
            if not isinstance(entry, Mapping):
                raise TypeError("transition_timeline entry must be a Mapping")
            if set(entry.keys()) != {"ts", "from", "to", "reason"}:
                raise ValueError(
                    f"transition_timeline entry keys must be exactly "
                    f"{{'ts', 'from', 'to', 'reason'}}, got {sorted(entry.keys())}"
                )
            for k, v in entry.items():
                if not isinstance(v, str):
                    raise TypeError(
                        f"transition_timeline entry[{k!r}] must be a string, got {v!r}"
                    )

        _check_non_negative_int(self.loop_detected_count, "loop_detected_count")

        if not isinstance(self.shutdown_requested, bool):
            raise TypeError("shutdown_requested must be a bool")

        if self.shutdown_exit_code is not None:
            if isinstance(self.shutdown_exit_code, bool) or not isinstance(
                self.shutdown_exit_code, int
            ):
                raise ValueError("shutdown_exit_code must be an int or None")
            if self.shutdown_exit_code not in {0, 1}:
                raise ValueError(
                    f"shutdown_exit_code must be in {{0, 1, None}}, got {self.shutdown_exit_code}"
                )


@dataclass(frozen=True)
class WorldSection:
    sync_count: int
    nodes_discovered: int
    entities_seen: int
    graph_built_count: int
    node_count: int
    edge_count: int
    load_duration_ms: float | None

    def __post_init__(self) -> None:
        _check_non_negative_int(self.sync_count, "sync_count")
        _check_non_negative_int(self.nodes_discovered, "nodes_discovered")
        _check_non_negative_int(self.entities_seen, "entities_seen")
        _check_non_negative_int(self.graph_built_count, "graph_built_count")
        _check_non_negative_int(self.node_count, "node_count")
        _check_non_negative_int(self.edge_count, "edge_count")
        _check_optional_non_negative_float(self.load_duration_ms, "load_duration_ms")


_SECTION_CLASS_MAP: dict[str, type] = {
    "perception": PerceptionSection,
    "action": ActionSection,
    "reflex": ReflexSection,
    "navigation": NavigationSection,
    "combat": CombatSection,
    "humanizer": HumanizerSection,
    "strategist": StrategistSection,
    "watchdog": WatchdogSection,
    "world": WorldSection,
}


@dataclass(frozen=True)
class ReportV2:
    schema_version: int
    meta: MetaSection
    perception: PerceptionSection | None = None
    action: ActionSection | None = None
    reflex: ReflexSection | None = None
    navigation: NavigationSection | None = None
    combat: CombatSection | None = None
    humanizer: HumanizerSection | None = None
    strategist: StrategistSection | None = None
    watchdog: WatchdogSection | None = None
    world: WorldSection | None = None

    def __post_init__(self) -> None:
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(
                f"schema_version must equal {SCHEMA_VERSION}, got {self.schema_version}"
            )
        if not isinstance(self.meta, MetaSection):
            raise TypeError("meta must be a MetaSection")

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dict representation of the report."""
        out: dict[str, Any] = {
            "schema_version": self.schema_version,
            "meta": _serialize_value(self.meta),
        }
        for name in section_names():
            sec_val = getattr(self, name)
            if sec_val is not None:
                out[name] = _serialize_value(sec_val)
        return out

    @classmethod
    def from_json(cls, data: dict[str, Any]) -> ReportV2:
        """Parse a dictionary into a ReportV2 instance after validating."""
        validate_report_dict(data)

        try:
            meta = MetaSection(**data["meta"])
            kwargs: dict[str, Any] = {
                "schema_version": data["schema_version"],
                "meta": meta,
            }

            for sec_name in section_names():
                if sec_name in data:
                    sec_cls = _SECTION_CLASS_MAP[sec_name]
                    sec_dict = dict(data[sec_name])
                    kwargs[sec_name] = _instantiate_section(sec_cls, sec_dict)
                else:
                    kwargs[sec_name] = None

            return cls(**kwargs)
        except Exception as e:
            if isinstance(e, SchemaV2Error):
                raise
            raise SchemaV2Error(f"Failed to parse ReportV2 from dict: {e}") from e


def _serialize_value(val: Any) -> Any:
    if hasattr(val, "__dataclass_fields__"):
        res: dict[str, Any] = {}
        for f in fields(val):
            field_val = getattr(val, f.name)
            res[f.name] = _serialize_value(field_val)
        return res
    if isinstance(val, tuple):
        return [_serialize_value(x) for x in val]
    if isinstance(val, Mapping):
        return {k: _serialize_value(v) for k, v in val.items()}
    return val


def _instantiate_section(cls: type, sec_dict: dict[str, Any]) -> Any:
    kw = dict(sec_dict)
    for f in fields(cls):
        fname = f.name
        if (
            fname in kw
            and kw[fname] is not None
            and (
                f.type is tuple
                or getattr(f.type, "__origin__", None) is tuple
                or (isinstance(f.type, str) and f.type.startswith("tuple"))
            )
            and isinstance(kw[fname], list)
        ):
            kw[fname] = tuple(kw[fname])
    if (
        cls is WatchdogSection
        and "transition_timeline" in kw
        and isinstance(kw["transition_timeline"], list)
    ):
        kw["transition_timeline"] = tuple(dict(x) for x in kw["transition_timeline"])
    return cls(**kw)


def validate_report_dict(data: dict[str, Any]) -> None:
    """Validate a raw report dictionary against Schema V2 rules."""
    if not isinstance(data, dict):
        raise SchemaV2Error(f"Report data must be a dict, got {type(data).__name__}")

    missing_req = REQUIRED_TOP_LEVEL_KEYS - data.keys()
    if missing_req:
        raise SchemaV2Error(f"Missing required top-level keys: {sorted(missing_req)}")

    unknown_top = data.keys() - ALL_TOP_LEVEL_KEYS
    if unknown_top:
        raise SchemaV2Error(f"Unknown top-level keys in report: {sorted(unknown_top)}")

    version = data.get("schema_version")
    if version != SCHEMA_VERSION:
        raise SchemaV2Error(f"schema_version must equal {SCHEMA_VERSION}, got {version!r}")

    meta_dict = data.get("meta")
    if not isinstance(meta_dict, dict):
        raise SchemaV2Error(f"meta must be a dict, got {type(meta_dict).__name__}")

    meta_fields = {f.name for f in fields(MetaSection)}
    meta_unknown = meta_dict.keys() - meta_fields
    if meta_unknown:
        raise SchemaV2Error(f"Unknown key inside section 'meta': {sorted(meta_unknown)}")

    try:
        MetaSection(**meta_dict)
    except (TypeError, ValueError) as e:
        raise SchemaV2Error(f"Validation failed for section 'meta': {e}") from e

    for sec_name in OPTIONAL_TOP_LEVEL_KEYS:
        if sec_name in data:
            sec_dict = data[sec_name]
            if sec_dict is None:
                raise SchemaV2Error(f"Section '{sec_name}' cannot be null when key is present")
            if not isinstance(sec_dict, dict):
                raise SchemaV2Error(
                    f"Section '{sec_name}' must be a dict, got {type(sec_dict).__name__}"
                )

            sec_cls = _SECTION_CLASS_MAP[sec_name]
            sec_fields = {f.name for f in fields(sec_cls)}
            sec_unknown = sec_dict.keys() - sec_fields
            if sec_unknown:
                raise SchemaV2Error(
                    f"Unknown key inside section '{sec_name}': {sorted(sec_unknown)}"
                )

            try:
                _instantiate_section(sec_cls, sec_dict)
            except (TypeError, ValueError) as e:
                raise SchemaV2Error(f"Validation failed for section '{sec_name}': {e}") from e


def empty_report(
    *,
    session_id: str,
    mode: str,
    started_at: str,
    config_snapshot: dict[str, Any],
) -> ReportV2:
    """Construct an empty ReportV2 with only schema_version and meta populated."""
    meta = MetaSection(
        session_id=session_id,
        mode=mode,
        started_at=started_at,
        stopped_at=None,
        stop_reason=None,
        config_snapshot=config_snapshot,
        event_count=0,
    )
    return ReportV2(
        schema_version=SCHEMA_VERSION,
        meta=meta,
    )


def section_names() -> tuple[str, ...]:
    """Return the names of optional report sections in declaration order."""
    return (
        "perception",
        "action",
        "reflex",
        "navigation",
        "combat",
        "humanizer",
        "strategist",
        "watchdog",
        "world",
    )


__all__ = [
    "ALL_TOP_LEVEL_KEYS",
    "CPU_PERCENT_CEILING",
    "OPTIONAL_TOP_LEVEL_KEYS",
    "REPORT_V2_JSON_SCHEMA_PATH",
    "REQUIRED_TOP_LEVEL_KEYS",
    "SCHEMA_VERSION",
    "ActionSection",
    "CombatSection",
    "HumanizerSection",
    "MetaSection",
    "NavigationSection",
    "PerceptionSection",
    "ReflexSection",
    "ReportV2",
    "SchemaV2Error",
    "StrategistSection",
    "WatchdogSection",
    "WorldSection",
    "empty_report",
    "section_names",
    "validate_report_dict",
]
