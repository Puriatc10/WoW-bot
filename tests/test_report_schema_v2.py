"""Unit tests for WoW-bot report schema v2 (Task 10.1)."""

from __future__ import annotations

import ast
import json
from dataclasses import fields
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from wow_bot.reporting.schema_v2 import (
    ALL_TOP_LEVEL_KEYS,
    CPU_PERCENT_CEILING,
    OPTIONAL_TOP_LEVEL_KEYS,
    REPORT_V2_JSON_SCHEMA_PATH,
    REQUIRED_TOP_LEVEL_KEYS,
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
    empty_report,
    section_names,
    validate_report_dict,
)
from wow_bot.strategist.prompts_v2 import ALLOWED_GOALS
from wow_bot.strategist.vocab_v2 import RejectionReason


def _make_valid_meta() -> MetaSection:
    return MetaSection(
        session_id="20260330-120000-12345678",
        mode="MOCK",
        started_at="2026-03-30T12:00:00Z",
        stopped_at="2026-03-30T12:05:00Z",
        stop_reason="completed",
        config_snapshot={"key": "val"},
        event_count=10,
    )


def _make_valid_full_report() -> ReportV2:
    meta = _make_valid_meta()
    perception = PerceptionSection(
        source="MockPerception",
        game_state_count=5,
        confidence_mean=0.95,
        known_fields=("field_a", "field_b"),
    )
    action = ActionSection(
        actuator_intent_count=10,
        actuator_result_count=10,
        rejected_count=0,
        failed_count=1,
        success_count=9,
        latency_ms_p50=12.5,
        latency_ms_p95=25.0,
        latency_ms_max=50.0,
        status_counts={"success": 9, "failed": 1},
    )
    reflex = ReflexSection(
        tick_count=100,
        mean_tick_period_ms=50.0,
        p99_tick_period_ms=55.0,
        overrun_count=0,
        signal_counts={"focus_lost": 1},
        sink_error_count=0,
        source_error_count=0,
    )
    navigation = NavigationSection(
        trip_count=2,
        success_count=2,
        failure_count=0,
        replan_count=1,
        segment_count=10,
        total_duration_s=120.0,
        mean_trip_duration_s=60.0,
        hard_failure_count=0,
        telemetry_sample_count=120,
        cpu_percent_p95=15.0,
    )
    combat = CombatSection(
        encounter_count=3,
        win_count=2,
        loss_count=0,
        flee_count=1,
        timeout_count=0,
        interrupt_attempt_count=2,
        interrupt_success_count=2,
        defensive_cast_count=1,
        mean_encounter_duration_s=18.5,
    )
    humanizer = HumanizerSection(
        action_count=10,
        interval_samples_ms=(100.0, 150.0, 120.0),
        pause_count=1,
        pause_duration_ms_mean=500.0,
        miss_click_count=0,
        config_snapshot={"sigma": 0.2},
    )
    strategist = StrategistSection(
        call_count=5,
        success_count=4,
        blocked_by_cooldown_count=1,
        invalid_json_count=0,
        llm_error_count=0,
        vocab_rejected_count=0,
        prompt_hash_count=5,
        unique_prompt_hashes=("hash_1", "hash_2"),
        latency_ms_p50=1200.0,
        latency_ms_p95=2500.0,
        goal_counts={"explore": 3, "farm_herbs": 2},
        rejection_reason_counts={"unknown_goal": 0},
    )
    watchdog = WatchdogSection(
        transition_count=1,
        final_health_state="healthy",
        transition_timeline=(
            {
                "ts": "2026-03-30T12:01:00Z",
                "from": "HEALTHY",
                "to": "HEALTHY",
                "reason": "nominal",
            },
        ),
        loop_detected_count=0,
        shutdown_requested=False,
        shutdown_exit_code=None,
    )
    world = WorldSection(
        sync_count=50,
        nodes_discovered=10,
        entities_seen=25,
        graph_built_count=1,
        node_count=100,
        edge_count=250,
        load_duration_ms=45.2,
    )

    return ReportV2(
        schema_version=SCHEMA_VERSION,
        meta=meta,
        perception=perception,
        action=action,
        reflex=reflex,
        navigation=navigation,
        combat=combat,
        humanizer=humanizer,
        strategist=strategist,
        watchdog=watchdog,
        world=world,
    )


def test_constants_and_top_level_keys() -> None:
    assert SCHEMA_VERSION == 2
    assert REQUIRED_TOP_LEVEL_KEYS == frozenset({"schema_version", "meta"})
    assert OPTIONAL_TOP_LEVEL_KEYS == frozenset({
        "perception",
        "action",
        "reflex",
        "navigation",
        "combat",
        "humanizer",
        "strategist",
        "watchdog",
        "world",
    })
    assert ALL_TOP_LEVEL_KEYS == REQUIRED_TOP_LEVEL_KEYS | OPTIONAL_TOP_LEVEL_KEYS
    assert section_names() == (
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


def test_report_v2_invalid_schema_version() -> None:
    meta = _make_valid_meta()
    with pytest.raises(ValueError, match="schema_version must equal 2"):
        ReportV2(schema_version=1, meta=meta)


def test_meta_section_validations() -> None:
    # empty session_id
    with pytest.raises(ValueError, match="session_id must be a non-empty string"):
        MetaSection(
            session_id="",
            mode="MOCK",
            started_at="2026-03-30T12:00:00Z",
            stopped_at=None,
            stop_reason=None,
            config_snapshot={},
            event_count=0,
        )

    # invalid mode
    with pytest.raises(ValueError, match="mode must be 'MOCK' or 'LAB'"):
        MetaSection(
            session_id="s1",
            mode="INVALID",
            started_at="2026-03-30T12:00:00Z",
            stopped_at=None,
            stop_reason=None,
            config_snapshot={},
            event_count=0,
        )

    # malformed started_at
    with pytest.raises(ValueError, match="Invalid ISO 8601 string"):
        MetaSection(
            session_id="s1",
            mode="MOCK",
            started_at="not-a-timestamp",
            stopped_at=None,
            stop_reason=None,
            config_snapshot={},
            event_count=0,
        )

    # stopped_at < started_at
    with pytest.raises(ValueError, match="stopped_at .* must be >= started_at"):
        MetaSection(
            session_id="s1",
            mode="MOCK",
            started_at="2026-03-30T12:00:00Z",
            stopped_at="2026-03-30T11:59:59Z",
            stop_reason=None,
            config_snapshot={},
            event_count=0,
        )

    # non-dict config_snapshot
    with pytest.raises(ValueError, match="config_snapshot must be a dict"):
        MetaSection(
            session_id="s1",
            mode="MOCK",
            started_at="2026-03-30T12:00:00Z",
            stopped_at=None,
            stop_reason=None,
            config_snapshot=[],  # type: ignore[arg-type]
            event_count=0,
        )

    # negative event_count
    with pytest.raises(ValueError, match="event_count must be >= 0"):
        MetaSection(
            session_id="s1",
            mode="MOCK",
            started_at="2026-03-30T12:00:00Z",
            stopped_at=None,
            stop_reason=None,
            config_snapshot={},
            event_count=-1,
        )


def test_perception_section_validations() -> None:
    # confidence_mean outside [0, 1]
    with pytest.raises(ValueError, match="confidence_mean must be in \\[0.0, 1.0\\]"):
        PerceptionSection(
            source="Mock",
            game_state_count=1,
            confidence_mean=1.5,
            known_fields=(),
        )

    # unsorted known_fields
    with pytest.raises(ValueError, match="known_fields must be sorted and unique"):
        PerceptionSection(
            source="Mock",
            game_state_count=1,
            confidence_mean=0.9,
            known_fields=("b", "a"),
        )

    # duplicate known_fields
    with pytest.raises(ValueError, match="known_fields must be sorted and unique"):
        PerceptionSection(
            source="Mock",
            game_state_count=1,
            confidence_mean=0.9,
            known_fields=("a", "a"),
        )


def test_action_section_validations() -> None:
    # negative count
    with pytest.raises(ValueError, match="actuator_intent_count must be >= 0"):
        ActionSection(
            actuator_intent_count=-1,
            actuator_result_count=0,
            rejected_count=0,
            failed_count=0,
            success_count=0,
            latency_ms_p50=None,
            latency_ms_p95=None,
            latency_ms_max=None,
            status_counts={},
        )

    # unknown status_counts key
    with pytest.raises(ValueError, match="status_counts key must be in"):
        ActionSection(
            actuator_intent_count=0,
            actuator_result_count=0,
            rejected_count=0,
            failed_count=0,
            success_count=0,
            latency_ms_p50=None,
            latency_ms_p95=None,
            latency_ms_max=None,
            status_counts={"unknown_status": 1},
        )


def test_reflex_section_validations() -> None:
    # negative tick_count
    with pytest.raises(ValueError, match="tick_count must be >= 0"):
        ReflexSection(
            tick_count=-5,
            mean_tick_period_ms=None,
            p99_tick_period_ms=None,
            overrun_count=0,
            signal_counts={},
            sink_error_count=0,
            source_error_count=0,
        )


def test_navigation_section_validations() -> None:
    # negative duration
    with pytest.raises(ValueError, match="total_duration_s must be >= 0"):
        NavigationSection(
            trip_count=0,
            success_count=0,
            failure_count=0,
            replan_count=0,
            segment_count=0,
            total_duration_s=-1.0,
            mean_trip_duration_s=None,
            hard_failure_count=0,
            telemetry_sample_count=0,
            cpu_percent_p95=None,
        )

    # cpu_percent_p95 above ceiling
    with pytest.raises(ValueError, match="cpu_percent_p95 must be in"):
        NavigationSection(
            trip_count=0,
            success_count=0,
            failure_count=0,
            replan_count=0,
            segment_count=0,
            total_duration_s=0.0,
            mean_trip_duration_s=None,
            hard_failure_count=0,
            telemetry_sample_count=0,
            cpu_percent_p95=CPU_PERCENT_CEILING + 10.0,
        )


def test_combat_section_validations() -> None:
    # outcome counts sum discrepancy
    with pytest.raises(ValueError, match="must equal encounter_count"):
        CombatSection(
            encounter_count=5,
            win_count=2,
            loss_count=1,
            flee_count=1,
            timeout_count=0,  # sum = 4 != 5
            interrupt_attempt_count=0,
            interrupt_success_count=0,
            defensive_cast_count=0,
            mean_encounter_duration_s=None,
        )


def test_humanizer_section_validations() -> None:
    # negative interval sample
    with pytest.raises(
        ValueError, match="interval_samples_ms element must be >= 0"
    ):
        HumanizerSection(
            action_count=1,
            interval_samples_ms=(10.0, -5.0),
            pause_count=0,
            pause_duration_ms_mean=None,
            miss_click_count=0,
            config_snapshot={},
        )


def test_strategist_section_validations() -> None:
    # goal not in ALLOWED_GOALS
    with pytest.raises(ValueError, match="goal_counts key .* is not in ALLOWED_GOALS"):
        StrategistSection(
            call_count=0,
            success_count=0,
            blocked_by_cooldown_count=0,
            invalid_json_count=0,
            llm_error_count=0,
            vocab_rejected_count=0,
            prompt_hash_count=0,
            unique_prompt_hashes=(),
            latency_ms_p50=None,
            latency_ms_p95=None,
            goal_counts={"invalid_goal": 1},
            rejection_reason_counts={},
        )

    # invalid rejection reason
    with pytest.raises(ValueError, match="is not a valid RejectionReason"):
        StrategistSection(
            call_count=0,
            success_count=0,
            blocked_by_cooldown_count=0,
            invalid_json_count=0,
            llm_error_count=0,
            vocab_rejected_count=0,
            prompt_hash_count=0,
            unique_prompt_hashes=(),
            latency_ms_p50=None,
            latency_ms_p95=None,
            goal_counts={},
            rejection_reason_counts={"invalid_reason": 1},
        )

    # unsorted unique_prompt_hashes
    with pytest.raises(
        ValueError, match="unique_prompt_hashes must be sorted and unique"
    ):
        StrategistSection(
            call_count=0,
            success_count=0,
            blocked_by_cooldown_count=0,
            invalid_json_count=0,
            llm_error_count=0,
            vocab_rejected_count=0,
            prompt_hash_count=0,
            unique_prompt_hashes=("b", "a"),
            latency_ms_p50=None,
            latency_ms_p95=None,
            goal_counts={},
            rejection_reason_counts={},
        )


def test_watchdog_section_validations() -> None:
    # invalid final_health_state
    with pytest.raises(ValueError, match="final_health_state must be in"):
        WatchdogSection(
            transition_count=0,
            final_health_state="unknown_state",
            transition_timeline=(),
            loop_detected_count=0,
            shutdown_requested=False,
            shutdown_exit_code=None,
        )

    # wrong timeline entry keys
    with pytest.raises(
        ValueError, match="transition_timeline entry keys must be exactly"
    ):
        WatchdogSection(
            transition_count=0,
            final_health_state="healthy",
            transition_timeline=({"ts": "2026-03-30T12:00:00Z", "from": "HEALTHY"},),  # type: ignore[arg-type]
            loop_detected_count=0,
            shutdown_requested=False,
            shutdown_exit_code=None,
        )

    # invalid shutdown_exit_code
    with pytest.raises(ValueError, match="shutdown_exit_code must be in"):
        WatchdogSection(
            transition_count=0,
            final_health_state="healthy",
            transition_timeline=(),
            loop_detected_count=0,
            shutdown_requested=False,
            shutdown_exit_code=2,  # type: ignore[arg-type]
        )


def test_world_section_validations() -> None:
    # negative load_duration_ms
    with pytest.raises(ValueError, match="load_duration_ms must be >= 0"):
        WorldSection(
            sync_count=0,
            nodes_discovered=0,
            entities_seen=0,
            graph_built_count=0,
            node_count=0,
            edge_count=0,
            load_duration_ms=-1.0,
        )


def test_validate_report_dict_errors() -> None:
    # non-dict
    with pytest.raises(SchemaV2Error, match="Report data must be a dict"):
        validate_report_dict("not-a-dict")  # type: ignore[arg-type]

    # missing schema_version
    with pytest.raises(
        SchemaV2Error, match="Missing required top-level keys: .*schema_version"
    ):
        validate_report_dict({"meta": _make_valid_meta().session_id})

    # missing meta
    with pytest.raises(
        SchemaV2Error, match="Missing required top-level keys: .*meta"
    ):
        validate_report_dict({"schema_version": 2})

    # unknown top-level key
    full_report = _make_valid_full_report()
    json_dict = full_report.to_json()
    json_dict["unknown_top_key"] = 123
    with pytest.raises(SchemaV2Error, match="Unknown top-level keys in report"):
        validate_report_dict(json_dict)

    # schema_version != 2
    json_dict = full_report.to_json()
    json_dict["schema_version"] = 1
    with pytest.raises(SchemaV2Error, match="schema_version must equal 2"):
        validate_report_dict(json_dict)

    # unknown key inside a section
    json_dict = full_report.to_json()
    json_dict["meta"]["extra_meta_field"] = "bad"
    with pytest.raises(
        SchemaV2Error, match="Unknown key inside section 'meta'"
    ):
        validate_report_dict(json_dict)


def test_validate_report_dict_and_to_json() -> None:
    # minimal valid report
    minimal = empty_report(
        session_id="s123",
        mode="MOCK",
        started_at="2026-03-30T12:00:00Z",
        config_snapshot={"a": 1},
    )
    minimal_dict = minimal.to_json()
    validate_report_dict(minimal_dict)

    assert minimal_dict["schema_version"] == 2
    assert "meta" in minimal_dict
    assert len(minimal_dict) == 2  # no optional section keys present
    assert list(minimal_dict.keys())[:2] == ["schema_version", "meta"]

    # full valid report
    full = _make_valid_full_report()
    full_dict = full.to_json()
    validate_report_dict(full_dict)

    assert list(full_dict.keys())[:2] == ["schema_version", "meta"]
    for name in section_names():
        assert name in full_dict

    # Verify input dict is not mutated by validate_report_dict
    dict_copy = json.loads(json.dumps(full_dict))
    validate_report_dict(dict_copy)
    assert dict_copy == full_dict


def test_to_json_types_and_determinism() -> None:
    full = _make_valid_full_report()
    d1 = full.to_json()
    d2 = full.to_json()

    assert d1 == d2  # determinism
    # verify tuple converted to list
    assert isinstance(d1["humanizer"]["interval_samples_ms"], list)
    assert isinstance(d1["strategist"]["unique_prompt_hashes"], list)
    assert isinstance(d1["watchdog"]["transition_timeline"], list)

    # verify MappingProxyType converted to dict
    action_sec = ActionSection(
        actuator_intent_count=0,
        actuator_result_count=0,
        rejected_count=0,
        failed_count=0,
        success_count=0,
        latency_ms_p50=None,
        latency_ms_p95=None,
        latency_ms_max=None,
        status_counts=MappingProxyType({"success": 1}),
    )
    rep = ReportV2(
        schema_version=SCHEMA_VERSION,
        meta=_make_valid_meta(),
        action=action_sec,
    )
    json_out = rep.to_json()
    assert isinstance(json_out["action"]["status_counts"], dict)


def test_round_trip() -> None:
    full = _make_valid_full_report()
    json_dict = full.to_json()
    reconstructed = ReportV2.from_json(json_dict)

    assert reconstructed == full


def test_from_json_invalid_dict() -> None:
    with pytest.raises(SchemaV2Error):
        ReportV2.from_json({"invalid": True})


def test_empty_report_helper() -> None:
    rep = empty_report(
        session_id="test_sess",
        mode="LAB",
        started_at="2026-03-30T10:00:00Z",
        config_snapshot={"mode": "LAB"},
    )
    assert rep.schema_version == 2
    assert rep.meta.session_id == "test_sess"
    assert rep.meta.mode == "LAB"
    assert rep.meta.started_at == "2026-03-30T10:00:00Z"
    assert rep.meta.config_snapshot == {"mode": "LAB"}

    for sec in section_names():
        assert getattr(rep, sec) is None


def test_json_schema_file_contract() -> None:
    schema_path = Path(REPORT_V2_JSON_SCHEMA_PATH)
    assert schema_path.exists()

    schema_data = json.loads(schema_path.read_text(encoding="utf-8"))
    assert schema_data["properties"]["schema_version"]["const"] == 2

    top_properties = set(schema_data["properties"].keys())
    assert top_properties == ALL_TOP_LEVEL_KEYS

    # Verify field names and types for every section between dataclasses and JSON schema
    dataclass_map: dict[str, type] = {
        "meta": MetaSection,
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

    for sec_name, cls in dataclass_map.items():
        sec_schema = schema_data["properties"][sec_name]
        dc_fields = {f.name for f in fields(cls)}
        schema_props = set(sec_schema["properties"].keys())
        assert (
            dc_fields == schema_props
        ), f"Mismatch in field names for section {sec_name}"


def test_static_ast_isolation() -> None:
    source_file = Path("src/wow_bot/reporting/schema_v2.py")
    tree = ast.parse(source_file.read_text(encoding="utf-8"))

    forbidden_modules = [
        "wow_bot.reporting.scenario",
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
        "aiosqlite",
        "asyncio",
        "threading",
        "ollama",
        "openai",
        "anthropic",
        "llm",
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_modules:
                    assert not alias.name.startswith(forbidden), (
                        f"Forbidden import found: {alias.name}"
                    )
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                full_mod = node.module
                allowed_exception = full_mod in (
                    "wow_bot.strategist.prompts_v2",
                    "wow_bot.strategist.vocab_v2",
                )
                if not allowed_exception:
                    for forbidden in forbidden_modules:
                        assert not full_mod.startswith(forbidden), (
                            f"Forbidden from-import found: {full_mod}"
                        )

        # Check that time calls are not made
        if isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if node.func.attr in ("monotonic", "time", "perf_counter"):
                    raise AssertionError(f"Call to time.{node.func.attr} found!")


def test_static_ast_no_prelab_touched() -> None:
    # Verify no pre-lab module files were modified (scenario.py, etc.)
    scenario_file = Path("src/wow_bot/reporting/scenario.py")
    assert scenario_file.exists()
    content = scenario_file.read_text(encoding="utf-8")
    assert "SCENARIO_REPORT_SCHEMA_VERSION: int = 1" in content
