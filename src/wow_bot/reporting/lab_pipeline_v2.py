"""Lab Reporting Pipeline V2 for WoW-bot.

Processes session execution directories (session.json, events.jsonl) and compiles
schema v2 lab reports (report_v2.json) with deterministic aggregation.
"""

from __future__ import annotations

import json
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from statistics import fmean, quantiles
from typing import Any

from wow_bot.reporting.schema_v2 import (
    SCHEMA_VERSION,
    ActionSection,
    CombatSection,
    HumanizerSection,
    MetaSection,
    NavigationSection,
    PerceptionSection,
    ReflexSection,
    ReportV2,
    SchemaV2Error,
    StrategistSection,
    WatchdogSection,
    WorldSection,
    section_names,
    validate_report_dict,
)
from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS
from wow_bot.strategist.vocab_v2 import RejectionReason


class PipelineError(Exception):
    """Exception raised for errors during lab pipeline post-processing."""


@dataclass(frozen=True)
class PipelineConfig:
    output_filename: str = "report_v2.json"
    include_empty_sections: bool = False
    humanizer_interval_samples_max: int = 5000
    strategist_prompt_hashes_max: int = 100
    strategist_unique_hashes_max: int = 100
    watchdog_timeline_max: int = 1000
    reflex_signal_counts_max: int = 64
    strict_unknown_events: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.output_filename, str) or not self.output_filename:
            raise ValueError("output_filename must be a non-empty string")
        if (
            "/" in self.output_filename
            or "\\" in self.output_filename
            or ".." in self.output_filename
        ):
            raise ValueError(
                f"output_filename must not contain '/', '\\\\', or '..', got {self.output_filename!r}"
            )

        if not isinstance(self.include_empty_sections, bool):
            raise ValueError(  # noqa: TRY004
                f"include_empty_sections must be a bool, got {type(self.include_empty_sections).__name__}"
            )

        if not isinstance(self.strict_unknown_events, bool):
            raise ValueError(  # noqa: TRY004
                f"strict_unknown_events must be a bool, got {type(self.strict_unknown_events).__name__}"
            )

        max_fields = (
            ("humanizer_interval_samples_max", self.humanizer_interval_samples_max),
            ("strategist_prompt_hashes_max", self.strategist_prompt_hashes_max),
            ("strategist_unique_hashes_max", self.strategist_unique_hashes_max),
            ("watchdog_timeline_max", self.watchdog_timeline_max),
            ("reflex_signal_counts_max", self.reflex_signal_counts_max),
        )

        for name, val in max_fields:
            if isinstance(val, bool) or not isinstance(val, int):
                raise ValueError(f"{name} must be an int, got {val!r}")  # noqa: TRY004
            if val < 1:
                raise ValueError(f"{name} must be >= 1, got {val}")


@dataclass(frozen=True)
class PipelineStats:
    events_total: int
    events_used: int
    events_unknown: int
    events_malformed: int
    sections_present: tuple[str, ...]
    truncated_sections: tuple[str, ...]

    def to_json(self) -> dict[str, Any]:
        return {
            "events_total": self.events_total,
            "events_used": self.events_used,
            "events_unknown": self.events_unknown,
            "events_malformed": self.events_malformed,
            "sections_present": list(self.sections_present),
            "truncated_sections": list(self.truncated_sections),
        }


ACTION_EVENTS: frozenset[str] = frozenset(
    {"actuator_intent", "actuator_result", "actuator_rejected", "actuator_abort"}
)
REFLEX_EVENTS: frozenset[str] = frozenset(
    {"reflex_tick", "reflex_watchdog_loop_detected"}
)
NAVIGATION_EVENTS: frozenset[str] = frozenset(
    {
        "nav_started",
        "nav_segment",
        "nav_replan",
        "nav_completed",
        "nav_graph_built",
        "nav_telemetry",
        "flee_started",
        "flee_completed",
        "flee_failed",
    }
)
COMBAT_EVENTS: frozenset[str] = frozenset()
HUMANIZER_EVENTS: frozenset[str] = frozenset({"humanizer_action", "humanizer_abort"})
STRATEGIST_EVENTS: frozenset[str] = frozenset(
    {
        "strategist_success",
        "strategist_retry",
        "strategist_invalid_json",
        "strategist_llm_error",
        "strategist_prompt_error",
        "cooldown_allowed",
        "cooldown_forced_open",
        "cooldown_manual_block",
        "cooldown_manual_clear",
        "vocab_accepted",
        "vocab_rejected",
    }
)
WATCHDOG_EVENTS: frozenset[str] = frozenset(
    {"watchdog_transition", "watchdog_loop_detected", "shutdown_complete"}
)
WORLD_EVENTS: frozenset[str] = frozenset(
    {"world_sync", "world_loaded", "nav_graph_built"}
)
PERCEPTION_EVENTS: frozenset[str] = frozenset()

META_EVENTS: frozenset[str] = frozenset(
    {
        "mode_entered",
        "session_closed",
        "crash_written",
        "safety_abort",
        "safety_callback_failed",
    }
)
FSM_EVENTS: frozenset[str] = frozenset(
    {
        "fsm_transition",
        "fsm_feedback",
        "fsm_intent",
        "fsm_paused",
        "fsm_resumed",
        "fsm_recovery_entered",
        "fsm_hard_stuck",
    }
)

ALL_RECOGNIZED_EVENTS: frozenset[str] = (
    ACTION_EVENTS
    | REFLEX_EVENTS
    | NAVIGATION_EVENTS
    | COMBAT_EVENTS
    | HUMANIZER_EVENTS
    | STRATEGIST_EVENTS
    | WATCHDOG_EVENTS
    | WORLD_EVENTS
    | PERCEPTION_EVENTS
    | META_EVENTS
    | FSM_EVENTS
)

_VALID_REJECTION_REASONS: frozenset[str] = frozenset(r.value for r in RejectionReason)


def load_session_meta(session_dir: Path) -> dict[str, Any]:
    session_file = session_dir / "session.json"
    if not session_file.is_file():
        raise PipelineError(f"Session meta file not found at {session_file}")
    try:
        content = session_file.read_text(encoding="utf-8")
        data = json.loads(content)
    except Exception as e:
        raise PipelineError(f"Failed to parse session meta at {session_file}: {e}") from e

    if not isinstance(data, dict):
        raise PipelineError(f"Session meta root must be a dict, got {type(data).__name__}")
    return data


def load_events(session_dir: Path) -> list[dict[str, Any]]:
    events_file = session_dir / "events.jsonl"
    if not events_file.is_file():
        raise PipelineError(f"Events file not found at {events_file}")

    events: list[dict[str, Any]] = []
    try:
        with events_file.open("r", encoding="utf-8") as f:
            for line_no, raw_line in enumerate(f, start=1):
                stripped = raw_line.strip()
                if not stripped:
                    continue
                try:
                    obj = json.loads(stripped)
                except Exception as e:
                    raise PipelineError(
                        f"Line {line_no}: invalid JSON in events log: {e}"
                    ) from e

                if not isinstance(obj, dict):
                    raise PipelineError(
                        f"Line {line_no}: event must be a dict, got {type(obj).__name__}"
                    )

                expected_keys = {"ts", "event", "payload"}
                if set(obj.keys()) != expected_keys:
                    raise PipelineError(
                        f"Line {line_no}: event dict keys must be exactly ('ts', 'event', 'payload'), got {sorted(obj.keys())}"
                    )

                events.append(obj)
    except PipelineError:
        raise
    except Exception as e:
        raise PipelineError(f"Failed to read events file at {events_file}: {e}") from e

    return events


def _calc_quantiles_p50_p95(vals: list[float]) -> tuple[float | None, float | None]:
    if not vals:
        return None, None
    if len(vals) == 1:
        return vals[0], vals[0]
    q = quantiles(vals, n=100, method="inclusive")
    return q[49], q[94]


def _calc_quantiles_p99(vals: list[float]) -> float | None:
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    q = quantiles(vals, n=100, method="inclusive")
    return q[98]


def build_report_v2(
    events: Sequence[dict[str, Any]],
    session_meta: dict[str, Any],
    *,
    config: PipelineConfig | None = None,
) -> tuple[ReportV2, PipelineStats]:
    if config is None:
        config = PipelineConfig()

    session_id = session_meta.get("session_id")
    if not isinstance(session_id, str) or not session_id:
        raise PipelineError(
            f"session_meta['session_id'] must be a non-empty string, got {session_id!r}"
        )

    mode = session_meta.get("mode")
    if mode not in {"MOCK", "LAB"}:
        raise PipelineError(
            f"session_meta['mode'] must be 'MOCK' or 'LAB', got {mode!r}"
        )

    started_at = session_meta.get("started_at")
    if not isinstance(started_at, str):
        raise PipelineError(
            f"session_meta['started_at'] must be an ISO 8601 string, got {started_at!r}"
        )

    stopped_at = session_meta.get("stopped_at")
    if stopped_at is not None and not isinstance(stopped_at, str):
        raise PipelineError(
            f"session_meta['stopped_at'] must be an ISO 8601 string or None, got {stopped_at!r}"
        )

    stop_reason = session_meta.get("stop_reason")
    if stop_reason is not None and not isinstance(stop_reason, str):
        raise PipelineError(
            f"session_meta['stop_reason'] must be a string or None, got {stop_reason!r}"
        )

    config_snapshot = session_meta.get("config_snapshot")
    if not isinstance(config_snapshot, dict):
        raise PipelineError(
            f"session_meta['config_snapshot'] must be a dict, got {type(config_snapshot).__name__}"
        )

    events_used = 0
    events_unknown = 0

    section_event_counts: dict[str, int] = {
        "perception": 0,
        "action": 0,
        "reflex": 0,
        "navigation": 0,
        "combat": 0,
        "humanizer": 0,
        "strategist": 0,
        "watchdog": 0,
        "world": 0,
    }

    for idx, e in enumerate(events):
        name = e["event"]
        if name not in ALL_RECOGNIZED_EVENTS:
            if config.strict_unknown_events:
                raise PipelineError(f"Unknown event name at index {idx}: {name!r}")
            events_unknown += 1
            continue

        events_used += 1

        if name in ACTION_EVENTS:
            section_event_counts["action"] += 1
        if name in REFLEX_EVENTS:
            section_event_counts["reflex"] += 1
        if name in NAVIGATION_EVENTS:
            section_event_counts["navigation"] += 1
        if name in COMBAT_EVENTS:
            section_event_counts["combat"] += 1
        if name in HUMANIZER_EVENTS:
            section_event_counts["humanizer"] += 1
        if name in STRATEGIST_EVENTS:
            section_event_counts["strategist"] += 1
        if name in WATCHDOG_EVENTS:
            section_event_counts["watchdog"] += 1
        if name in WORLD_EVENTS:
            section_event_counts["world"] += 1
        if name in PERCEPTION_EVENTS:
            section_event_counts["perception"] += 1

    truncated_sections_set: set[str] = set()

    # Build ActionSection
    action_section: ActionSection | None = None
    if section_event_counts["action"] > 0 or config.include_empty_sections:
        intent_count = sum(1 for e in events if e["event"] == "actuator_intent")
        result_count = sum(1 for e in events if e["event"] == "actuator_result")
        rejected_count = sum(1 for e in events if e["event"] == "actuator_rejected")

        status_counts = {"success": 0, "failed": 0, "timeout": 0}
        latencies: list[float] = []

        for e in events:
            if e["event"] == "actuator_result":
                p = e["payload"]
                st = p.get("status")
                if st in status_counts:
                    status_counts[st] += 1
                lat = p.get("latency_ms")
                if lat is not None:
                    latencies.append(float(lat))

        succ_count = status_counts["success"]
        fail_count = status_counts["failed"] + status_counts["timeout"]

        p50, p95 = _calc_quantiles_p50_p95(latencies)
        max_lat = max(latencies) if latencies else None

        action_section = ActionSection(
            actuator_intent_count=intent_count,
            actuator_result_count=result_count,
            rejected_count=rejected_count,
            failed_count=fail_count,
            success_count=succ_count,
            latency_ms_p50=p50,
            latency_ms_p95=p95,
            latency_ms_max=max_lat,
            status_counts=status_counts,
        )

    # Build ReflexSection
    reflex_section: ReflexSection | None = None
    if section_event_counts["reflex"] > 0 or config.include_empty_sections:
        tick_count = sum(1 for e in events if e["event"] == "reflex_tick")
        tick_periods: list[float] = []
        overrun_count = 0
        sink_error_count = 0
        sig_counts: dict[str, int] = {}

        for e in events:
            if e["event"] == "reflex_tick":
                p = e["payload"]
                if "ended_at" in p and "started_at" in p:
                    tick_periods.append((float(p["ended_at"]) - float(p["started_at"])) * 1000.0)
                if p.get("overrun") is True:
                    overrun_count += 1
                sink_error_count += int(p.get("sink_errors", 0))

                sc = p.get("signal_counts")
                if isinstance(sc, dict):
                    for k, v in sc.items():
                        sig_counts[k] = sig_counts.get(k, 0) + int(v)

        mean_period = fmean(tick_periods) if tick_periods else None
        p99_period = _calc_quantiles_p99(tick_periods)

        if len(sig_counts) > config.reflex_signal_counts_max:
            truncated_sections_set.add("reflex")
            kept_keys = sorted(sig_counts.keys())[: config.reflex_signal_counts_max]
            sig_counts = {k: sig_counts[k] for k in kept_keys}

        reflex_section = ReflexSection(
            tick_count=tick_count,
            mean_tick_period_ms=mean_period,
            p99_tick_period_ms=p99_period,
            overrun_count=overrun_count,
            signal_counts=sig_counts,
            sink_error_count=sink_error_count,
            source_error_count=0,
        )

    # Build NavigationSection
    nav_section: NavigationSection | None = None
    if section_event_counts["navigation"] > 0 or config.include_empty_sections:
        trip_count = sum(1 for e in events if e["event"] == "nav_started")
        succ_count = 0
        fail_count = 0
        hard_fail_count = 0
        replan_count = sum(1 for e in events if e["event"] == "nav_replan")
        segment_count = sum(1 for e in events if e["event"] == "nav_segment")
        total_duration = 0.0

        for e in events:
            if e["event"] == "nav_completed":
                p = e["payload"]
                st = p.get("status")
                if st == "success":
                    succ_count += 1
                elif st in {"failed", "timeout"}:
                    fail_count += 1
                elif st == "hard_failure":
                    hard_fail_count += 1
                total_duration += float(p.get("duration_s", 0.0))

        mean_trip_dur = total_duration / trip_count if trip_count > 0 else None

        cpu_vals: list[float] = []
        sample_count = 0
        for e in events:
            if e["event"] == "nav_telemetry":
                samples = e["payload"].get("cpu_samples", [])
                sample_count += len(samples)
                for s in samples:
                    if isinstance(s, dict) and "cpu_percent" in s:
                        cpu_vals.append(float(s["cpu_percent"]))
                    elif isinstance(s, (int, float)):
                        cpu_vals.append(float(s))

        _, cpu_p95 = _calc_quantiles_p50_p95(cpu_vals)

        nav_section = NavigationSection(
            trip_count=trip_count,
            success_count=succ_count,
            failure_count=fail_count,
            replan_count=replan_count,
            segment_count=segment_count,
            total_duration_s=total_duration,
            mean_trip_duration_s=mean_trip_dur,
            hard_failure_count=hard_fail_count,
            telemetry_sample_count=sample_count,
            cpu_percent_p95=cpu_p95,
        )

    # Build CombatSection
    combat_section: CombatSection | None = None
    if section_event_counts["combat"] > 0 or config.include_empty_sections:
        combat_section = CombatSection(
            encounter_count=0,
            win_count=0,
            loss_count=0,
            flee_count=0,
            timeout_count=0,
            interrupt_attempt_count=0,
            interrupt_success_count=0,
            defensive_cast_count=0,
            mean_encounter_duration_s=None,
        )

    # Build HumanizerSection
    humanizer_section: HumanizerSection | None = None
    if section_event_counts["humanizer"] > 0 or config.include_empty_sections:
        action_count = sum(1 for e in events if e["event"] == "humanizer_action")
        raw_intervals = [
            float(e["payload"]["interval_s"]) * 1000.0
            for e in events
            if e["event"] == "humanizer_action" and "interval_s" in e["payload"]
        ]

        if len(raw_intervals) > config.humanizer_interval_samples_max:
            truncated_sections_set.add("humanizer")
            raw_intervals = raw_intervals[-config.humanizer_interval_samples_max :]

        pause_count = sum(
            1
            for e in events
            if e["event"] == "humanizer_action" and e["payload"].get("pause_occurred") is True
        )
        pause_durs = [
            float(e["payload"]["pause_duration_s"]) * 1000.0
            for e in events
            if e["event"] == "humanizer_action"
            and e["payload"].get("pause_occurred") is True
            and "pause_duration_s" in e["payload"]
        ]
        mean_pause_dur = fmean(pause_durs) if pause_durs else None

        hum_config: dict[str, Any] = {}
        for e in events:
            if (
                e["event"] == "humanizer_action"
                and "config" in e["payload"]
                and isinstance(e["payload"]["config"], dict)
            ):
                hum_config = dict(e["payload"]["config"])
                break

        humanizer_section = HumanizerSection(
            action_count=action_count,
            interval_samples_ms=tuple(raw_intervals),
            pause_count=pause_count,
            pause_duration_ms_mean=mean_pause_dur,
            miss_click_count=0,
            config_snapshot=hum_config,
        )

    # Build StrategistSection
    strategist_section: StrategistSection | None = None
    if section_event_counts["strategist"] > 0 or config.include_empty_sections:
        succ_count = sum(1 for e in events if e["event"] == "strategist_success")
        retry_count = sum(1 for e in events if e["event"] == "strategist_retry")
        inv_json_count = sum(1 for e in events if e["event"] == "strategist_invalid_json")
        llm_err_count = sum(1 for e in events if e["event"] == "strategist_llm_error")
        prompt_err_count = sum(1 for e in events if e["event"] == "strategist_prompt_error")
        vocab_rej_count = sum(1 for e in events if e["event"] == "vocab_rejected")

        call_count = succ_count + retry_count + inv_json_count + llm_err_count + prompt_err_count

        prompt_hashes: list[str] = []
        strat_latencies: list[float] = []
        goal_counts: dict[str, int] = {}

        for e in events:
            if e["event"] == "strategist_success":
                p = e["payload"]
                ph = p.get("prompt_hash")
                if isinstance(ph, str) and ph:
                    prompt_hashes.append(ph)
                lat = p.get("latency_ms")
                if lat is not None:
                    strat_latencies.append(float(lat))
                g = p.get("goal")
                if isinstance(g, str):
                    if g not in ALLOWED_GOALS:
                        raise PipelineError(f"Goal '{g}' in strategist_success is not in ALLOWED_GOALS")
                    goal_counts[g] = goal_counts.get(g, 0) + 1

        rejection_reason_counts: dict[str, int] = {}
        for e in events:
            if e["event"] == "vocab_rejected":
                p = e["payload"]
                r = p.get("reason")
                if isinstance(r, str):
                    if r not in _VALID_REJECTION_REASONS:
                        raise PipelineError(f"Rejection reason '{r}' in vocab_rejected is not a valid RejectionReason")
                    rejection_reason_counts[r] = rejection_reason_counts.get(r, 0) + 1

        prompt_hash_count = len(prompt_hashes)
        sorted_unique_hashes = sorted(set(prompt_hashes))
        if len(sorted_unique_hashes) > config.strategist_unique_hashes_max:
            truncated_sections_set.add("strategist")
            sorted_unique_hashes = sorted_unique_hashes[: config.strategist_unique_hashes_max]

        p50_lat, p95_lat = _calc_quantiles_p50_p95(strat_latencies)

        strategist_section = StrategistSection(
            call_count=call_count,
            success_count=succ_count,
            blocked_by_cooldown_count=0,
            invalid_json_count=inv_json_count,
            llm_error_count=llm_err_count,
            vocab_rejected_count=vocab_rej_count,
            prompt_hash_count=prompt_hash_count,
            unique_prompt_hashes=tuple(sorted_unique_hashes),
            latency_ms_p50=p50_lat,
            latency_ms_p95=p95_lat,
            goal_counts=goal_counts,
            rejection_reason_counts=rejection_reason_counts,
        )

    # Build WatchdogSection
    watchdog_section: WatchdogSection | None = None
    if section_event_counts["watchdog"] > 0 or config.include_empty_sections:
        trans_count = sum(1 for e in events if e["event"] == "watchdog_transition")
        timeline: list[dict[str, str]] = []

        max_trans_ts = ""
        max_trans_idx = -1
        final_health = "healthy"

        for idx, e in enumerate(events):
            if e["event"] == "watchdog_transition":
                p = e["payload"]
                ts_val = str(p.get("ts", e["ts"]))
                entry = {
                    "ts": ts_val,
                    "from": str(p["from"]),
                    "to": str(p["to"]),
                    "reason": str(p["reason"]),
                }
                timeline.append(entry)

                if idx >= max_trans_idx and ts_val >= max_trans_ts:
                    max_trans_ts = ts_val
                    max_trans_idx = idx
                    final_health = str(p["to"])

        if len(timeline) > config.watchdog_timeline_max:
            truncated_sections_set.add("watchdog")
            timeline = timeline[-config.watchdog_timeline_max :]

        loop_count = sum(1 for e in events if e["event"] == "watchdog_loop_detected")
        shutdown_req = any(e["event"] == "shutdown_complete" for e in events)

        shutdown_code: int | None = None
        for e in events:
            if e["event"] == "shutdown_complete":
                sc = e["payload"].get("exit_code")
                if sc is not None:
                    shutdown_code = int(sc)

        watchdog_section = WatchdogSection(
            transition_count=trans_count,
            final_health_state=final_health,
            transition_timeline=tuple(timeline),
            loop_detected_count=loop_count,
            shutdown_requested=shutdown_req,
            shutdown_exit_code=shutdown_code,
        )

    # Build WorldSection
    world_section: WorldSection | None = None
    if section_event_counts["world"] > 0 or config.include_empty_sections:
        sync_count = sum(1 for e in events if e["event"] == "world_sync")
        nodes_disc = sum(
            1 for e in events if e["event"] == "world_sync" and e["payload"].get("player_node_created") is True
        )
        entities_seen = sum(
            int(e["payload"].get("entities_upserted", 0)) for e in events if e["event"] == "world_sync"
        )
        graph_built_count = sum(1 for e in events if e["event"] == "nav_graph_built")

        last_node_count = 0
        last_edge_count = 0
        for e in events:
            if e["event"] == "nav_graph_built":
                last_node_count = int(e["payload"].get("node_count", 0))
                last_edge_count = int(e["payload"].get("edge_count", 0))

        load_dur: float | None = None
        for e in events:
            if e["event"] == "world_loaded" and "duration_ms" in e["payload"]:
                load_dur = float(e["payload"]["duration_ms"])

        world_section = WorldSection(
            sync_count=sync_count,
            nodes_discovered=nodes_disc,
            entities_seen=entities_seen,
            graph_built_count=graph_built_count,
            node_count=last_node_count,
            edge_count=last_edge_count,
            load_duration_ms=load_dur,
        )

    # Build PerceptionSection
    perception_section: PerceptionSection | None = None
    if section_event_counts["perception"] > 0 or config.include_empty_sections:
        perception_section = PerceptionSection(
            source="mock",
            game_state_count=0,
            confidence_mean=None,
            known_fields=(),
        )

    meta_section = MetaSection(
        session_id=session_id,
        mode=mode,
        started_at=started_at,
        stopped_at=stopped_at,
        stop_reason=stop_reason,
        config_snapshot=dict(config_snapshot),
        event_count=len(events),
    )

    report = ReportV2(
        schema_version=SCHEMA_VERSION,
        meta=meta_section,
        perception=perception_section,
        action=action_section,
        reflex=reflex_section,
        navigation=nav_section,
        combat=combat_section,
        humanizer=humanizer_section,
        strategist=strategist_section,
        watchdog=watchdog_section,
        world=world_section,
    )

    try:
        validate_report_dict(report.to_json())
    except (SchemaV2Error, ValueError) as err:
        raise PipelineError(f"Report validation failed: {err}") from err

    present_names = tuple(
        name for name in section_names() if getattr(report, name) is not None
    )

    truncated_names = tuple(
        name for name in section_names() if name in truncated_sections_set
    )

    stats = PipelineStats(
        events_total=len(events),
        events_used=events_used,
        events_unknown=events_unknown,
        events_malformed=0,
        sections_present=present_names,
        truncated_sections=truncated_names,
    )

    return report, stats


def write_report_v2(report: ReportV2, path: Path) -> None:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f"{path.name}.tmp"
        content = json.dumps(report.to_json(), ensure_ascii=False, indent=2) + "\n"
        tmp_path.write_text(content, encoding="utf-8")
        os.replace(tmp_path, path)
    except Exception as e:
        raise PipelineError(f"Failed to write report to {path}: {e}") from e


def run_lab_pipeline(
    session_dir: Path,
    *,
    config: PipelineConfig | None = None,
) -> tuple[Path, PipelineStats]:
    if config is None:
        config = PipelineConfig()
    session_meta = load_session_meta(session_dir)
    events = load_events(session_dir)
    report, stats = build_report_v2(events, session_meta, config=config)
    output_path = session_dir / config.output_filename
    write_report_v2(report, output_path)
    return output_path, stats


__all__ = [
    "PipelineConfig",
    "PipelineError",
    "PipelineStats",
    "build_report_v2",
    "load_events",
    "load_session_meta",
    "run_lab_pipeline",
    "write_report_v2",
]
