"""Tests for WoW-bot Lab Reporting Pipeline V2 (src/wow_bot/reporting/lab_pipeline_v2.py)."""

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from wow_bot.reporting.lab_pipeline_v2 import (
    PipelineConfig,
    PipelineError,
    PipelineStats,
    build_report_v2,
    load_events,
    load_session_meta,
    run_lab_pipeline,
    write_report_v2,
)
from wow_bot.reporting.schema_v2 import validate_report_dict
from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS
from wow_bot.strategist.vocab_v2 import RejectionReason


@pytest.fixture
def valid_session_meta() -> dict[str, Any]:
    return {
        "session_id": "20260330-120000-abcd1234",
        "mode": "LAB",
        "started_at": "2026-03-30T12:00:00Z",
        "stopped_at": "2026-03-30T12:05:00Z",
        "stop_reason": "completed",
        "config_snapshot": {"lab_mode": True, "dry_run": False},
    }


# Config validation tests
def test_pipeline_config_empty_output_filename() -> None:
    with pytest.raises(ValueError, match="output_filename must be a non-empty string"):
        PipelineConfig(output_filename="")


def test_pipeline_config_output_filename_slash() -> None:
    with pytest.raises(ValueError, match="output_filename must not contain"):
        PipelineConfig(output_filename="sub/report.json")


def test_pipeline_config_output_filename_dotdot() -> None:
    with pytest.raises(ValueError, match="output_filename must not contain"):
        PipelineConfig(output_filename="../report.json")


def test_pipeline_config_max_field_less_than_one() -> None:
    with pytest.raises(ValueError, match="humanizer_interval_samples_max must be >= 1"):
        PipelineConfig(humanizer_interval_samples_max=0)


# load_session_meta tests
def test_load_session_meta_missing_file(tmp_path: Path) -> None:
    with pytest.raises(PipelineError, match="Session meta file not found"):
        load_session_meta(tmp_path)


def test_load_session_meta_invalid_json(tmp_path: Path) -> None:
    (tmp_path / "session.json").write_text("invalid json", encoding="utf-8")
    with pytest.raises(PipelineError, match="Failed to parse session meta"):
        load_session_meta(tmp_path)


def test_load_session_meta_non_dict_root(tmp_path: Path) -> None:
    (tmp_path / "session.json").write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(PipelineError, match="Session meta root must be a dict"):
        load_session_meta(tmp_path)


def test_load_session_meta_valid(tmp_path: Path, valid_session_meta: dict[str, Any]) -> None:
    (tmp_path / "session.json").write_text(json.dumps(valid_session_meta), encoding="utf-8")
    res = load_session_meta(tmp_path)
    assert res == valid_session_meta


# load_events tests
def test_load_events_skips_empty_lines(tmp_path: Path) -> None:
    lines = [
        "",
        '{"ts": "2026-03-30T12:00:01Z", "event": "mode_entered", "payload": {}}',
        "   ",
        "\n",
        '{"ts": "2026-03-30T12:00:02Z", "event": "session_closed", "payload": {}}',
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    events = load_events(tmp_path)
    assert len(events) == 2
    assert events[0]["event"] == "mode_entered"
    assert events[1]["event"] == "session_closed"


def test_load_events_malformed_json_line(tmp_path: Path) -> None:
    lines = [
        '{"ts": "2026-03-30T12:00:01Z", "event": "mode_entered", "payload": {}}',
        "{bad json",
    ]
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(PipelineError, match="Line 2: invalid JSON"):
        load_events(tmp_path)


def test_load_events_non_dict_line(tmp_path: Path) -> None:
    lines = ["123"]
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(PipelineError, match="Line 1: event must be a dict"):
        load_events(tmp_path)


def test_load_events_missing_keys(tmp_path: Path) -> None:
    lines = ['{"ts": "2026-03-30T12:00:01Z", "event": "mode_entered"}']
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")
    with pytest.raises(PipelineError, match="Line 1: event dict keys must be exactly"):
        load_events(tmp_path)


# build_report_v2 tests
def test_build_report_v2_empty_events(valid_session_meta: dict[str, Any]) -> None:
    report, stats = build_report_v2([], valid_session_meta)
    assert report.schema_version == 2
    assert report.meta.session_id == valid_session_meta["session_id"]
    assert report.meta.event_count == 0
    assert report.action is None
    assert report.reflex is None
    assert report.navigation is None
    assert report.combat is None
    assert report.humanizer is None
    assert report.strategist is None
    assert report.watchdog is None
    assert report.world is None
    assert report.perception is None
    assert stats.events_total == 0
    assert stats.events_used == 0
    assert stats.events_unknown == 0


def test_build_report_v2_include_empty_sections(valid_session_meta: dict[str, Any]) -> None:
    cfg = PipelineConfig(include_empty_sections=True)
    report, stats = build_report_v2([], valid_session_meta, config=cfg)
    assert report.action is not None
    assert report.reflex is not None
    assert report.navigation is not None
    assert report.combat is not None
    assert report.humanizer is not None
    assert report.strategist is not None
    assert report.watchdog is not None
    assert report.world is not None
    assert report.perception is not None
    assert len(stats.sections_present) == 9


def test_build_report_v2_meta_reflection(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "mode_entered", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "session_closed", "payload": {}},
    ]
    report, stats = build_report_v2(events, valid_session_meta)
    assert report.meta.event_count == 2
    assert stats.events_total == 2
    assert stats.events_used == 2


def test_build_report_v2_unknown_events(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "unknown_future_event", "payload": {}},
    ]
    report, stats = build_report_v2(events, valid_session_meta)
    assert stats.events_total == 1
    assert stats.events_used == 0
    assert stats.events_unknown == 1
    assert report.action is None


def test_build_report_v2_strict_unknown_events(valid_session_meta: dict[str, Any]) -> None:
    cfg = PipelineConfig(strict_unknown_events=True)
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "unknown_future_event", "payload": {}},
    ]
    with pytest.raises(PipelineError, match="Unknown event name"):
        build_report_v2(events, valid_session_meta, config=cfg)


def test_build_report_v2_action_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "actuator_intent", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "actuator_result", "payload": {"status": "success", "latency_ms": 10.0}},
        {"ts": "2026-03-30T12:00:03Z", "event": "actuator_result", "payload": {"status": "failed", "latency_ms": 20.0}},
        {"ts": "2026-03-30T12:00:04Z", "event": "actuator_rejected", "payload": {}},
    ]
    report, _ = build_report_v2(events, valid_session_meta)
    assert report.action is not None
    assert report.action.actuator_intent_count == 1
    assert report.action.actuator_result_count == 2
    assert report.action.rejected_count == 1
    assert report.action.success_count == 1
    assert report.action.failed_count == 1
    assert report.action.status_counts == {"success": 1, "failed": 1, "timeout": 0}
    assert report.action.latency_ms_p50 == 15.0
    assert report.action.latency_ms_max == 20.0


def test_build_report_v2_reflex_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {
            "ts": "2026-03-30T12:00:01Z",
            "event": "reflex_tick",
            "payload": {
                "started_at": 100.0,
                "ended_at": 100.05,
                "overrun": False,
                "sink_errors": 0,
                "signal_counts": {"sig_b": 2, "sig_a": 1, "sig_c": 3},
            },
        },
        {
            "ts": "2026-03-30T12:00:02Z",
            "event": "reflex_tick",
            "payload": {
                "started_at": 100.1,
                "ended_at": 100.15,
                "overrun": True,
                "sink_errors": 1,
                "signal_counts": {"sig_a": 4},
            },
        },
    ]
    cfg = PipelineConfig(reflex_signal_counts_max=2)
    report, stats = build_report_v2(events, valid_session_meta, config=cfg)
    assert report.reflex is not None
    assert report.reflex.tick_count == 2
    assert report.reflex.overrun_count == 1
    assert report.reflex.sink_error_count == 1
    assert report.reflex.source_error_count == 0
    assert report.reflex.mean_tick_period_ms == pytest.approx(50.0)
    # signal_counts capped at 2, sorted keys sig_a, sig_b kept
    assert report.reflex.signal_counts == {"sig_a": 5, "sig_b": 2}
    assert "reflex" in stats.truncated_sections


def test_build_report_v2_navigation_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "nav_started", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "nav_segment", "payload": {}},
        {"ts": "2026-03-30T12:00:03Z", "event": "nav_replan", "payload": {}},
        {"ts": "2026-03-30T12:00:04Z", "event": "nav_completed", "payload": {"status": "success", "duration_s": 10.0}},
        {"ts": "2026-03-30T12:00:05Z", "event": "nav_completed", "payload": {"status": "failed", "duration_s": 5.0}},
        {"ts": "2026-03-30T12:00:06Z", "event": "nav_completed", "payload": {"status": "hard_failure", "duration_s": 1.0}},
        {"ts": "2026-03-30T12:00:07Z", "event": "nav_telemetry", "payload": {"cpu_samples": [{"cpu_percent": 10.0}, {"cpu_percent": 20.0}]}},
    ]
    report, _ = build_report_v2(events, valid_session_meta)
    assert report.navigation is not None
    assert report.navigation.trip_count == 1
    assert report.navigation.segment_count == 1
    assert report.navigation.replan_count == 1
    assert report.navigation.success_count == 1
    assert report.navigation.failure_count == 1
    assert report.navigation.hard_failure_count == 1
    assert report.navigation.total_duration_s == 16.0
    assert report.navigation.mean_trip_duration_s == 16.0
    assert report.navigation.telemetry_sample_count == 2
    assert report.navigation.cpu_percent_p95 is not None


def test_build_report_v2_humanizer_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "humanizer_action", "payload": {"interval_s": 0.1, "pause_occurred": True, "pause_duration_s": 0.5, "config": {"clip_low": 0.05}}},
        {"ts": "2026-03-30T12:00:02Z", "event": "humanizer_action", "payload": {"interval_s": 0.2, "pause_occurred": False}},
        {"ts": "2026-03-30T12:00:03Z", "event": "humanizer_action", "payload": {"interval_s": 0.3, "pause_occurred": True, "pause_duration_s": 1.5}},
    ]
    cfg = PipelineConfig(humanizer_interval_samples_max=2)
    report, stats = build_report_v2(events, valid_session_meta, config=cfg)
    assert report.humanizer is not None
    assert report.humanizer.action_count == 3
    # Bounded to last 2 samples: 200.0, 300.0
    assert report.humanizer.interval_samples_ms == (200.0, 300.0)
    assert "humanizer" in stats.truncated_sections
    assert report.humanizer.pause_count == 2
    assert report.humanizer.pause_duration_ms_mean == 1000.0
    assert report.humanizer.miss_click_count == 0
    assert report.humanizer.config_snapshot == {"clip_low": 0.05}


def test_build_report_v2_strategist_section(valid_session_meta: dict[str, Any]) -> None:
    allowed_goal = ALLOWED_GOALS[0]
    valid_reason = next(iter(RejectionReason)).value

    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "strategist_success", "payload": {"prompt_hash": "hash_b", "latency_ms": 100.0, "goal": allowed_goal}},
        {"ts": "2026-03-30T12:00:02Z", "event": "strategist_success", "payload": {"prompt_hash": "hash_a", "latency_ms": 200.0, "goal": allowed_goal}},
        {"ts": "2026-03-30T12:00:03Z", "event": "strategist_invalid_json", "payload": {}},
        {"ts": "2026-03-30T12:00:04Z", "event": "vocab_rejected", "payload": {"reason": valid_reason}},
    ]
    cfg = PipelineConfig(strategist_unique_hashes_max=1)
    report, stats = build_report_v2(events, valid_session_meta, config=cfg)
    assert report.strategist is not None
    assert report.strategist.call_count == 3
    assert report.strategist.success_count == 2
    assert report.strategist.invalid_json_count == 1
    assert report.strategist.vocab_rejected_count == 1
    assert report.strategist.blocked_by_cooldown_count == 0
    assert report.strategist.prompt_hash_count == 2
    assert report.strategist.unique_prompt_hashes == ("hash_a",)
    assert "strategist" in stats.truncated_sections
    assert report.strategist.goal_counts == {allowed_goal: 2}
    assert report.strategist.rejection_reason_counts == {valid_reason: 1}


def test_build_report_v2_watchdog_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "watchdog_transition", "payload": {"from": "healthy", "to": "degraded", "reason": "rate_drop"}},
        {"ts": "2026-03-30T12:00:02Z", "event": "watchdog_transition", "payload": {"from": "degraded", "to": "critical", "reason": "no_progress"}},
        {"ts": "2026-03-30T12:00:03Z", "event": "watchdog_loop_detected", "payload": {}},
        {"ts": "2026-03-30T12:00:04Z", "event": "shutdown_complete", "payload": {"exit_code": 0}},
    ]
    cfg = PipelineConfig(watchdog_timeline_max=1)
    report, stats = build_report_v2(events, valid_session_meta, config=cfg)
    assert report.watchdog is not None
    assert report.watchdog.transition_count == 2
    assert report.watchdog.final_health_state == "critical"
    assert len(report.watchdog.transition_timeline) == 1
    assert report.watchdog.transition_timeline[0]["to"] == "critical"
    assert "watchdog" in stats.truncated_sections
    assert report.watchdog.loop_detected_count == 1
    assert report.watchdog.shutdown_requested is True
    assert report.watchdog.shutdown_exit_code == 0


def test_build_report_v2_world_section(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "world_sync", "payload": {"player_node_created": True, "entities_upserted": 3}},
        {"ts": "2026-03-30T12:00:02Z", "event": "world_sync", "payload": {"player_node_created": False, "entities_upserted": 2}},
        {"ts": "2026-03-30T12:00:03Z", "event": "nav_graph_built", "payload": {"node_count": 10, "edge_count": 20}},
        {"ts": "2026-03-30T12:00:04Z", "event": "world_loaded", "payload": {"duration_ms": 150.0}},
    ]
    report, _ = build_report_v2(events, valid_session_meta)
    assert report.world is not None
    assert report.world.sync_count == 2
    assert report.world.nodes_discovered == 1
    assert report.world.entities_seen == 5
    assert report.world.graph_built_count == 1
    assert report.world.node_count == 10
    assert report.world.edge_count == 20
    assert report.world.load_duration_ms == 150.0


def test_build_report_v2_perception_section_always_none(valid_session_meta: dict[str, Any]) -> None:
    report, _ = build_report_v2([], valid_session_meta)
    assert report.perception is None


def test_build_report_v2_does_not_mutate_inputs(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "mode_entered", "payload": {"a": 1}},
    ]
    events_snapshot = json.dumps(events)
    meta_snapshot = json.dumps(valid_session_meta)

    build_report_v2(events, valid_session_meta)

    assert json.dumps(events) == events_snapshot
    assert json.dumps(valid_session_meta) == meta_snapshot


def test_pipeline_stats_to_json() -> None:
    stats = PipelineStats(
        events_total=10,
        events_used=8,
        events_unknown=2,
        events_malformed=0,
        sections_present=("action", "reflex"),
        truncated_sections=("reflex",),
    )
    s_json = stats.to_json()
    assert s_json == {
        "events_total": 10,
        "events_used": 8,
        "events_unknown": 2,
        "events_malformed": 0,
        "sections_present": ["action", "reflex"],
        "truncated_sections": ["reflex"],
    }


# write_report_v2 tests
def test_write_report_v2_atomic_and_mkdir(tmp_path: Path, valid_session_meta: dict[str, Any]) -> None:
    report, _ = build_report_v2([], valid_session_meta)
    out_path = tmp_path / "sub_dir" / "report_v2.json"

    write_report_v2(report, out_path)

    assert out_path.is_file()
    assert not (tmp_path / "sub_dir" / "report_v2.json.tmp").exists()
    content = out_path.read_text(encoding="utf-8")
    parsed = json.loads(content)
    validate_report_dict(parsed)


# run_lab_pipeline tests & idempotency
def test_run_lab_pipeline_end_to_end_and_idempotent(tmp_path: Path, valid_session_meta: dict[str, Any]) -> None:
    (tmp_path / "session.json").write_text(json.dumps(valid_session_meta), encoding="utf-8")

    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "mode_entered", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "actuator_intent", "payload": {}},
        {"ts": "2026-03-30T12:00:03Z", "event": "actuator_result", "payload": {"status": "success", "latency_ms": 12.5}},
        {"ts": "2026-03-30T12:00:04Z", "event": "session_closed", "payload": {}},
    ]
    lines = [json.dumps(e) for e in events]
    (tmp_path / "events.jsonl").write_text("\n".join(lines), encoding="utf-8")

    out_path1, stats1 = run_lab_pipeline(tmp_path)
    content1 = out_path1.read_bytes()

    out_path2, stats2 = run_lab_pipeline(tmp_path)
    content2 = out_path2.read_bytes()

    assert out_path1 == tmp_path / "report_v2.json"
    assert content1 == content2
    assert stats1 == stats2


def test_determinism_build_report_v2(valid_session_meta: dict[str, Any]) -> None:
    events = [
        {"ts": "2026-03-30T12:00:01Z", "event": "actuator_intent", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "actuator_result", "payload": {"status": "success", "latency_ms": 15.0}},
    ]
    rep1, stats1 = build_report_v2(events, valid_session_meta)
    rep2, stats2 = build_report_v2(events, valid_session_meta)

    assert rep1 == rep2
    assert stats1 == stats2
    assert json.dumps(rep1.to_json()) == json.dumps(rep2.to_json())


def test_full_synthetic_events_report(tmp_path: Path, valid_session_meta: dict[str, Any]) -> None:
    allowed_goal = ALLOWED_GOALS[0]
    valid_reason = next(iter(RejectionReason)).value

    events = [
        {"ts": "2026-03-30T12:00:00Z", "event": "mode_entered", "payload": {}},
        {"ts": "2026-03-30T12:00:01Z", "event": "actuator_intent", "payload": {}},
        {"ts": "2026-03-30T12:00:02Z", "event": "actuator_result", "payload": {"status": "success", "latency_ms": 10.0}},
        {"ts": "2026-03-30T12:00:03Z", "event": "actuator_rejected", "payload": {}},
        {"ts": "2026-03-30T12:00:04Z", "event": "actuator_abort", "payload": {}},
        {"ts": "2026-03-30T12:00:05Z", "event": "reflex_tick", "payload": {"started_at": 1.0, "ended_at": 1.05, "overrun": False, "sink_errors": 0, "signal_counts": {"sig1": 1}}},
        {"ts": "2026-03-30T12:00:06Z", "event": "reflex_watchdog_loop_detected", "payload": {}},
        {"ts": "2026-03-30T12:00:07Z", "event": "nav_started", "payload": {}},
        {"ts": "2026-03-30T12:00:08Z", "event": "nav_segment", "payload": {}},
        {"ts": "2026-03-30T12:00:09Z", "event": "nav_replan", "payload": {}},
        {"ts": "2026-03-30T12:00:10Z", "event": "nav_completed", "payload": {"status": "success", "duration_s": 5.0}},
        {"ts": "2026-03-30T12:00:11Z", "event": "nav_telemetry", "payload": {"cpu_samples": [12.0]}},
        {"ts": "2026-03-30T12:00:12Z", "event": "flee_started", "payload": {}},
        {"ts": "2026-03-30T12:00:13Z", "event": "flee_completed", "payload": {}},
        {"ts": "2026-03-30T12:00:14Z", "event": "flee_failed", "payload": {}},
        {"ts": "2026-03-30T12:00:15Z", "event": "humanizer_action", "payload": {"interval_s": 0.1, "pause_occurred": False}},
        {"ts": "2026-03-30T12:00:16Z", "event": "humanizer_abort", "payload": {}},
        {"ts": "2026-03-30T12:00:17Z", "event": "strategist_success", "payload": {"prompt_hash": "abc", "latency_ms": 500.0, "goal": allowed_goal}},
        {"ts": "2026-03-30T12:00:18Z", "event": "strategist_retry", "payload": {}},
        {"ts": "2026-03-30T12:00:19Z", "event": "strategist_invalid_json", "payload": {}},
        {"ts": "2026-03-30T12:00:20Z", "event": "strategist_llm_error", "payload": {}},
        {"ts": "2026-03-30T12:00:21Z", "event": "strategist_prompt_error", "payload": {}},
        {"ts": "2026-03-30T12:00:22Z", "event": "cooldown_allowed", "payload": {}},
        {"ts": "2026-03-30T12:00:23Z", "event": "cooldown_forced_open", "payload": {}},
        {"ts": "2026-03-30T12:00:24Z", "event": "cooldown_manual_block", "payload": {}},
        {"ts": "2026-03-30T12:00:25Z", "event": "cooldown_manual_clear", "payload": {}},
        {"ts": "2026-03-30T12:00:26Z", "event": "vocab_accepted", "payload": {}},
        {"ts": "2026-03-30T12:00:27Z", "event": "vocab_rejected", "payload": {"reason": valid_reason}},
        {"ts": "2026-03-30T12:00:28Z", "event": "watchdog_transition", "payload": {"from": "healthy", "to": "degraded", "reason": "warn"}},
        {"ts": "2026-03-30T12:00:29Z", "event": "watchdog_loop_detected", "payload": {}},
        {"ts": "2026-03-30T12:00:30Z", "event": "shutdown_complete", "payload": {"exit_code": 0}},
        {"ts": "2026-03-30T12:00:31Z", "event": "world_sync", "payload": {"player_node_created": True, "entities_upserted": 1}},
        {"ts": "2026-03-30T12:00:32Z", "event": "world_loaded", "payload": {"duration_ms": 10.0}},
        {"ts": "2026-03-30T12:00:33Z", "event": "nav_graph_built", "payload": {"node_count": 5, "edge_count": 8}},
        {"ts": "2026-03-30T12:00:34Z", "event": "session_closed", "payload": {}},
        {"ts": "2026-03-30T12:00:35Z", "event": "crash_written", "payload": {}},
        {"ts": "2026-03-30T12:00:36Z", "event": "safety_abort", "payload": {}},
        {"ts": "2026-03-30T12:00:37Z", "event": "safety_callback_failed", "payload": {}},
        {"ts": "2026-03-30T12:00:38Z", "event": "fsm_transition", "payload": {}},
        {"ts": "2026-03-30T12:00:39Z", "event": "fsm_feedback", "payload": {}},
        {"ts": "2026-03-30T12:00:40Z", "event": "fsm_intent", "payload": {}},
        {"ts": "2026-03-30T12:00:41Z", "event": "fsm_paused", "payload": {}},
        {"ts": "2026-03-30T12:00:42Z", "event": "fsm_resumed", "payload": {}},
        {"ts": "2026-03-30T12:00:43Z", "event": "fsm_recovery_entered", "payload": {}},
        {"ts": "2026-03-30T12:00:44Z", "event": "fsm_hard_stuck", "payload": {}},
    ]

    report, stats = build_report_v2(events, valid_session_meta)
    validate_report_dict(report.to_json())

    assert stats.events_total == len(events)
    assert stats.events_used == len(events)
    assert stats.events_unknown == 0


# Static AST Checks
def test_static_ast_prohibited_imports_and_calls() -> None:
    target_path = Path("src/wow_bot/reporting/lab_pipeline_v2.py")
    tree = ast.parse(target_path.read_text(encoding="utf-8"))

    prohibited_modules = {
        "wow_bot.reporting.scenario",
        "aiosqlite",
        "asyncio",
        "threading",
    }
    prohibited_prefixes = (
        "wow_bot.analysis",
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
    )
    prohibited_llm_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                assert mod_name not in prohibited_modules, f"Prohibited import: {mod_name}"
                for prefix in prohibited_prefixes:
                    assert not mod_name.startswith(prefix), f"Prohibited import prefix: {mod_name}"
                for sub in prohibited_llm_substrings:
                    assert sub not in mod_name.lower(), f"Prohibited LLM import: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                mod_name = node.module
                allowed_exceptions = {
                    "wow_bot.strategist.prompts_v2",
                    "wow_bot.strategist.vocab_v2",
                    "wow_bot.reporting.schema_v2",
                }
                if mod_name in allowed_exceptions:
                    continue
                assert mod_name not in prohibited_modules, f"Prohibited import: {mod_name}"
                for prefix in prohibited_prefixes:
                    assert not mod_name.startswith(prefix), f"Prohibited import prefix: {mod_name}"
                for sub in prohibited_llm_substrings:
                    assert sub not in mod_name.lower(), f"Prohibited LLM import: {mod_name}"

        elif isinstance(node, ast.Call):
            # Check for time.time, time.monotonic, time.perf_counter
            func = node.func
            if (
                isinstance(func, ast.Attribute)
                and isinstance(func.value, ast.Name)
                and func.value.id == "time"
            ):
                assert func.attr not in {
                    "time",
                    "monotonic",
                    "perf_counter",
                }, f"Prohibited time call: time.{func.attr}"
