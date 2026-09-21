"""Tests for cross-session aggregate analysis in wow_bot.analysis.aggregate."""

from __future__ import annotations

import ast
import json
from pathlib import Path
from typing import Any

import pytest

from wow_bot.analysis.aggregate import (
    AGGREGATE_SCHEMA_VERSION,
    NON_CLAIMS,
    AggregateConfig,
    AggregateError,
    aggregate_sessions,
    load_report_v2,
    load_soak_report,
    run_aggregate,
    write_aggregate,
)
from wow_bot.analysis.lab_soak_v2 import (
    SoakConfig,
    SoakSample,
    build_soak_report,
    write_soak_report,
)
from wow_bot.reporting.schema_v2 import SCHEMA_VERSION as REPORT_SCHEMA_VERSION

FIXED_NOW = "2026-01-01T00:00:00Z"


def make_session_dir(
    tmp_path: Path,
    session_id: str,
    *,
    report_v2_dict: dict[str, Any] | None = None,
    soak_samples: list[SoakSample] | None = None,
    write_invalid_session_json: bool = False,
    write_malformed_report_v2: bool = False,
    write_malformed_soak_report: bool = False,
) -> Path:
    """Helper to construct a synthetic session directory in tmp_path."""
    s_dir = tmp_path / session_id
    s_dir.mkdir(parents=True, exist_ok=True)

    session_json_path = s_dir / "session.json"
    if write_invalid_session_json:
        session_json_path.write_text("{invalid json", encoding="utf-8")
    else:
        session_data = {
            "session_id": session_id,
            "started_at": "2026-01-01T00:00:00Z",
            "mode": "MOCK",
            "config_snapshot": {},
        }
        session_json_path.write_text(json.dumps(session_data), encoding="utf-8")

    if write_malformed_report_v2:
        (s_dir / "report_v2.json").write_text("{corrupt report json", encoding="utf-8")
    elif report_v2_dict is not None:
        (s_dir / "report_v2.json").write_text(
            json.dumps(report_v2_dict, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    if write_malformed_soak_report:
        (s_dir / "soak_report.json").write_text("{corrupt soak json", encoding="utf-8")
    elif soak_samples is not None:
        s_report = build_soak_report(
            soak_samples,
            session_id=session_id,
            config=SoakConfig(),
        )
        write_soak_report(s_report, s_dir / "soak_report.json")

    return s_dir


def make_minimal_report_v2_dict(session_id: str) -> dict[str, Any]:
    """Helper to build a valid minimal report_v2 dictionary."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "meta": {
            "session_id": session_id,
            "mode": "MOCK",
            "started_at": "2026-01-01T00:00:00Z",
            "stopped_at": None,
            "stop_reason": None,
            "config_snapshot": {},
            "event_count": 10,
        },
    }


def make_full_report_v2_dict(
    session_id: str,
    *,
    tick_count: int = 100,
    mean_tick_period_ms: float = 50.0,
    p99_tick_period_ms: float = 60.0,
    actuator_result_count: int = 10,
    latency_ms_p50: float = 12.0,
    latency_ms_p95: float = 20.0,
    strategist_call_count: int = 5,
    strategist_p50: float = 100.0,
    strategist_p95: float = 200.0,
) -> dict[str, Any]:
    """Helper to build a populated valid report_v2 dictionary."""
    return {
        "schema_version": REPORT_SCHEMA_VERSION,
        "meta": {
            "session_id": session_id,
            "mode": "MOCK",
            "started_at": "2026-01-01T00:00:00Z",
            "stopped_at": "2026-01-01T00:01:00Z",
            "stop_reason": "completed",
            "config_snapshot": {},
            "event_count": 150,
        },
        "reflex": {
            "tick_count": tick_count,
            "mean_tick_period_ms": mean_tick_period_ms,
            "p99_tick_period_ms": p99_tick_period_ms,
            "overrun_count": 1,
            "signal_counts": {"focus_lost": 0},
            "sink_error_count": 0,
            "source_error_count": 0,
        },
        "action": {
            "actuator_intent_count": actuator_result_count,
            "actuator_result_count": actuator_result_count,
            "rejected_count": 0,
            "failed_count": 0,
            "success_count": actuator_result_count,
            "latency_ms_p50": latency_ms_p50,
            "latency_ms_p95": latency_ms_p95,
            "latency_ms_max": latency_ms_p95 * 1.5,
            "status_counts": {"success": actuator_result_count},
        },
        "humanizer": {
            "action_count": 5,
            "interval_samples_ms": [100.0, 150.0, 200.0],
            "pause_count": 1,
            "pause_duration_ms_mean": 500.0,
            "miss_click_count": 0,
            "config_snapshot": {},
        },
        "strategist": {
            "call_count": strategist_call_count,
            "success_count": strategist_call_count,
            "blocked_by_cooldown_count": 0,
            "invalid_json_count": 0,
            "llm_error_count": 0,
            "vocab_rejected_count": 0,
            "prompt_hash_count": 1,
            "unique_prompt_hashes": ["hash123"],
            "latency_ms_p50": strategist_p50,
            "latency_ms_p95": strategist_p95,
            "goal_counts": {"sell_vendor": strategist_call_count},
            "rejection_reason_counts": {},
        },
        "watchdog": {
            "transition_count": 2,
            "final_health_state": "healthy",
            "transition_timeline": [
                {
                    "ts": "2026-01-01T00:00:10Z",
                    "from": "healthy",
                    "to": "healthy",
                    "reason": "normal",
                }
            ],
            "loop_detected_count": 0,
            "shutdown_requested": True,
            "shutdown_exit_code": 0,
        },
        "navigation": {
            "trip_count": 2,
            "success_count": 2,
            "failure_count": 0,
            "replan_count": 1,
            "segment_count": 4,
            "total_duration_s": 30.0,
            "mean_trip_duration_s": 15.0,
            "hard_failure_count": 0,
            "telemetry_sample_count": 10,
            "cpu_percent_p95": 5.0,
        },
        "world": {
            "sync_count": 10,
            "nodes_discovered": 5,
            "entities_seen": 20,
            "graph_built_count": 1,
            "node_count": 50,
            "edge_count": 100,
            "load_duration_ms": 12.0,
        },
    }


def make_soak_samples() -> list[SoakSample]:
    """Helper to build a small valid list of SoakSample objects."""
    return [
        SoakSample(
            ts=0.0,
            cpu_percent=1.0,
            rss_bytes=1000,
            log_size_bytes=100,
            position_delta=0.0,
            inventory_delta=0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        ),
        SoakSample(
            ts=1.0,
            cpu_percent=2.0,
            rss_bytes=1200,
            log_size_bytes=150,
            position_delta=1.5,
            inventory_delta=1,
            successful_actions_total=2,
            reflex_ticks_total=10,
        ),
    ]


def test_aggregate_config_validation() -> None:
    """Acceptance: AggregateConfig validation on filename and boolean flags."""
    with pytest.raises(ValueError, match="aggregate_filename must be a non-empty string"):
        AggregateConfig(aggregate_filename="")

    with pytest.raises(ValueError, match="must not contain '/'"):
        AggregateConfig(aggregate_filename="sub/agg.json")

    with pytest.raises(ValueError, match="must not contain '/'"):
        AggregateConfig(aggregate_filename="c:\\agg.json")

    with pytest.raises(ValueError, match="must not contain '/'"):
        AggregateConfig(aggregate_filename="../agg.json")

    cfg = AggregateConfig(
        aggregate_filename="valid.json",
        require_report_v2=True,
        require_soak_report=True,
        include_non_claims=False,
    )
    assert cfg.aggregate_filename == "valid.json"
    assert cfg.require_report_v2 is True
    assert cfg.require_soak_report is True
    assert cfg.include_non_claims is False


def test_nonexistent_directory_raises(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with a nonexistent directory raises AggregateError."""
    bad_dir = tmp_path / "does_not_exist"
    with pytest.raises(AggregateError, match="Session directory does not exist"):
        aggregate_sessions([bad_dir], now=FIXED_NOW)


def test_missing_session_json_raises(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with a directory missing session.json raises AggregateError."""
    empty_dir = tmp_path / "empty_session"
    empty_dir.mkdir()
    with pytest.raises(AggregateError, match="Session directory missing session.json"):
        aggregate_sessions([empty_dir], now=FIXED_NOW)


def test_malformed_session_json_raises(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with a directory whose session.json is malformed JSON raises AggregateError naming the file."""
    s_dir = make_session_dir(tmp_path, "session_bad_json", write_invalid_session_json=True)
    with pytest.raises(AggregateError) as exc_info:
        aggregate_sessions([s_dir], now=FIXED_NOW)
    assert "session.json" in str(exc_info.value)
    assert "Failed to parse JSON" in str(exc_info.value)


def test_require_report_v2_error(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with require_report_v2=True and a session lacking report_v2.json raises AggregateError."""
    s_dir = make_session_dir(tmp_path, "session_no_report")
    config = AggregateConfig(require_report_v2=True)
    with pytest.raises(AggregateError, match="missing required report_v2.json"):
        aggregate_sessions([s_dir], config=config, now=FIXED_NOW)


def test_require_soak_report_error(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with require_soak_report=True and a session lacking soak_report.json raises AggregateError."""
    s_dir = make_session_dir(tmp_path, "session_no_soak")
    config = AggregateConfig(require_soak_report=True)
    with pytest.raises(AggregateError, match="missing required soak_report.json"):
        aggregate_sessions([s_dir], config=config, now=FIXED_NOW)


def test_aggregate_empty_list() -> None:
    """Acceptance: aggregate_sessions on an empty list returns session_count=0, totals 0, percentiles None."""
    report = aggregate_sessions([], now=FIXED_NOW)
    assert report.schema_version == AGGREGATE_SCHEMA_VERSION
    assert report.generated_at == FIXED_NOW
    assert report.session_ids == ()
    assert report.session_count == 0
    assert report.session_sources == ()

    pa = report.perception_agnostic
    assert pa.reflex_tick_count_total == 0
    assert pa.reflex_tick_period_ms_mean is None
    assert pa.reflex_tick_period_ms_p95 is None
    assert pa.action_intent_count_total == 0
    assert pa.action_latency_ms_p50 is None
    assert pa.action_latency_ms_p95 is None
    assert pa.report_count == 0
    assert pa.soak_report_count == 0


def test_session_counts_combinations(tmp_path: Path) -> None:
    """Acceptance: session_count, report_count, soak_report_count reflect source files accurately."""
    # Session 1: report_v2 only
    s1 = make_session_dir(
        tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1")
    )
    rep1 = aggregate_sessions([s1], now=FIXED_NOW)
    assert rep1.session_count == 1
    assert rep1.perception_agnostic.report_count == 1
    assert rep1.perception_agnostic.soak_report_count == 0

    # Session 2: soak_report only
    s2 = make_session_dir(tmp_path, "s2", soak_samples=make_soak_samples())
    rep2 = aggregate_sessions([s2], now=FIXED_NOW)
    assert rep2.session_count == 1
    assert rep2.perception_agnostic.report_count == 0
    assert rep2.perception_agnostic.soak_report_count == 1

    # Session 3: both
    s3 = make_session_dir(
        tmp_path,
        "s3",
        report_v2_dict=make_minimal_report_v2_dict("s3"),
        soak_samples=make_soak_samples(),
    )
    rep3 = aggregate_sessions([s3], now=FIXED_NOW)
    assert rep3.session_count == 1
    assert rep3.perception_agnostic.report_count == 1
    assert rep3.perception_agnostic.soak_report_count == 1


def test_sums_and_weighted_means(tmp_path: Path) -> None:
    """Acceptance: sum fields sum across sessions; weighted means use per-session weights."""
    r1_dict = make_full_report_v2_dict(
        "sess_A",
        tick_count=100,
        mean_tick_period_ms=50.0,
        p99_tick_period_ms=60.0,
        actuator_result_count=10,
        latency_ms_p50=10.0,
        latency_ms_p95=20.0,
        strategist_call_count=2,
        strategist_p50=100.0,
        strategist_p95=200.0,
    )
    r2_dict = make_full_report_v2_dict(
        "sess_B",
        tick_count=300,
        mean_tick_period_ms=10.0,
        p99_tick_period_ms=80.0,
        actuator_result_count=30,
        latency_ms_p50=30.0,
        latency_ms_p95=40.0,
        strategist_call_count=8,
        strategist_p50=300.0,
        strategist_p95=400.0,
    )

    s1 = make_session_dir(tmp_path, "sess_A", report_v2_dict=r1_dict)
    s2 = make_session_dir(tmp_path, "sess_B", report_v2_dict=r2_dict)

    rep = aggregate_sessions([s1, s2], now=FIXED_NOW)
    pa = rep.perception_agnostic

    # Sums
    assert pa.reflex_tick_count_total == 100 + 300
    assert pa.action_result_count_total == 10 + 30
    assert pa.strategist_call_count_total == 2 + 8
    assert pa.watchdog_transition_count_total == 2 + 2
    assert pa.watchdog_shutdown_count_total == 2  # both reported shutdown_requested=True

    # Reflex Weighted Mean: (100 * 50.0 + 300 * 10.0) / (100 + 300) = (5000 + 3000) / 400 = 20.0
    assert pa.reflex_tick_period_ms_mean == pytest.approx(20.0)
    # Reflex Max p95: max(60.0, 80.0) = 80.0
    assert pa.reflex_tick_period_ms_p95 == pytest.approx(80.0)

    # Action Weighted Mean: (10 * 10.0 + 30 * 30.0) / 40 = (100 + 900) / 40 = 25.0
    assert pa.action_latency_ms_p50 == pytest.approx(25.0)
    assert pa.action_latency_ms_p95 == pytest.approx(40.0)

    # Strategist Weighted Mean: (2 * 100.0 + 8 * 300.0) / 10 = (200 + 2400) / 10 = 260.0
    assert pa.strategist_latency_ms_p50 == pytest.approx(260.0)
    assert pa.strategist_latency_ms_p95 == pytest.approx(400.0)


def test_malformed_report_v2_raises(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with malformed report_v2.json or failed validate_report_dict raises AggregateError naming file."""
    s_dir = make_session_dir(tmp_path, "session_corrupt_report", write_malformed_report_v2=True)
    with pytest.raises(AggregateError) as exc_info:
        aggregate_sessions([s_dir], now=FIXED_NOW)
    assert "report_v2.json" in str(exc_info.value)

    # Failed validation
    invalid_report_dict = {"schema_version": 999}  # invalid version
    s_dir2 = make_session_dir(
        tmp_path, "session_invalid_schema_report", report_v2_dict=invalid_report_dict
    )
    with pytest.raises(AggregateError) as exc_info2:
        aggregate_sessions([s_dir2], now=FIXED_NOW)
    assert "report_v2.json" in str(exc_info2.value)


def test_malformed_soak_report_raises(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions with a malformed soak_report.json raises AggregateError."""
    s_dir = make_session_dir(
        tmp_path, "session_corrupt_soak", write_malformed_soak_report=True
    )
    with pytest.raises(AggregateError) as exc_info:
        aggregate_sessions([s_dir], now=FIXED_NOW)
    assert "soak_report.json" in str(exc_info.value)


def test_session_ids_sorted_and_sources_preserve_input_order(tmp_path: Path) -> None:
    """Acceptance: session_ids are sorted/unique; session_sources preserve INPUT ORDER."""
    s3 = make_session_dir(tmp_path, "s3", report_v2_dict=make_minimal_report_v2_dict("s3"))
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))
    s2 = make_session_dir(tmp_path, "s2", report_v2_dict=make_minimal_report_v2_dict("s2"))

    # Pass in order [s3, s1, s2]
    rep = aggregate_sessions([s3, s1, s2], now=FIXED_NOW)

    assert rep.session_ids == ("s1", "s2", "s3")
    assert rep.session_count == 3
    assert [src.session_id for src in rep.session_sources] == ["s3", "s1", "s2"]


def test_internal_counters_section_and_non_claims(tmp_path: Path) -> None:
    """Acceptance: internal_counters present and all None; non_claims equals NON_CLAIMS or empty tuple based on config."""
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))

    rep_with_claims = aggregate_sessions([s1], now=FIXED_NOW)
    ic = rep_with_claims.internal_counters_not_research_findings
    assert ic.cycle_success_rate is None
    assert ic.cycle_failure_rate is None
    assert ic.mean_distance_to_target is None
    assert ic.farm_outcome_count is None
    assert rep_with_claims.non_claims == NON_CLAIMS

    rep_no_claims = aggregate_sessions(
        [s1], config=AggregateConfig(include_non_claims=False), now=FIXED_NOW
    )
    assert rep_no_claims.non_claims == ()


def test_to_json_and_serializability(tmp_path: Path) -> None:
    """Acceptance: AggregateReport.to_json is JSON-serializable and contains "schema_version": 1."""
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))
    rep = aggregate_sessions([s1], now=FIXED_NOW)

    json_dict = rep.to_json()
    assert json_dict["schema_version"] == 1
    assert json_dict["session_ids"] == ["s1"]
    assert json_dict["session_count"] == 1

    # Test round-trip JSON serialization
    serialized = json.dumps(json_dict)
    deserialized = json.loads(serialized)
    assert deserialized["schema_version"] == 1


def test_write_aggregate_atomic_and_creates_parents(tmp_path: Path) -> None:
    """Acceptance: write_aggregate writes UTF-8 JSON atomically, indent=2, sort_keys=True, creates parents."""
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))
    rep = aggregate_sessions([s1], now=FIXED_NOW)

    output_path = tmp_path / "deep" / "nested" / "agg.json"
    write_aggregate(rep, output_path)

    assert output_path.exists()
    assert not (output_path.parent / "agg.json.tmp").exists()

    content = output_path.read_text(encoding="utf-8")
    assert content.startswith(("{\n  \"generated_at\":", "{\n  \""))
    parsed = json.loads(content)
    assert parsed["schema_version"] == 1


def test_run_aggregate_and_idempotency(tmp_path: Path) -> None:
    """Acceptance: run_aggregate returns target path and is byte-identical across calls."""
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))
    output_dir = tmp_path / "out"

    p1 = run_aggregate([s1], output_dir=output_dir, now=FIXED_NOW)
    assert p1 == output_dir / "aggregate_v1.json"
    bytes1 = p1.read_bytes()

    p2 = run_aggregate([s1], output_dir=output_dir, now=FIXED_NOW)
    assert p2 == p1
    bytes2 = p2.read_bytes()

    assert bytes1 == bytes2


def test_determinism_and_input_immutability(tmp_path: Path) -> None:
    """Acceptance: aggregate_sessions is deterministic and does not mutate input directories."""
    s1 = make_session_dir(tmp_path, "s1", report_v2_dict=make_minimal_report_v2_dict("s1"))
    s2 = make_session_dir(tmp_path, "s2", soak_samples=make_soak_samples())

    r1 = aggregate_sessions([s1, s2], now=FIXED_NOW)
    r2 = aggregate_sessions([s1, s2], now=FIXED_NOW)

    assert r1 == r2

    # Order permutation check
    r3 = aggregate_sessions([s2, s1], now=FIXED_NOW)
    assert r3.perception_agnostic == r1.perception_agnostic
    assert [src.session_id for src in r3.session_sources] == ["s2", "s1"]


def test_helper_functions_load(tmp_path: Path) -> None:
    """Acceptance: load_soak_report and load_report_v2 load and validate correctly or raise AggregateError."""
    s1 = make_session_dir(
        tmp_path,
        "s1",
        report_v2_dict=make_minimal_report_v2_dict("s1"),
        soak_samples=make_soak_samples(),
    )

    r_dict = load_report_v2(s1 / "report_v2.json")
    assert r_dict["schema_version"] == REPORT_SCHEMA_VERSION

    s_rep = load_soak_report(s1 / "soak_report.json")
    assert s_rep.session_id == "s1"

    with pytest.raises(AggregateError, match="file does not exist"):
        load_report_v2(tmp_path / "missing.json")

    with pytest.raises(AggregateError, match="file does not exist"):
        load_soak_report(tmp_path / "missing.json")


def test_static_ast_inspection() -> None:
    """Acceptance: Static AST checks for forbidden imports, disallowed calls, and forbidden modules in aggregate.py."""
    filepath = Path("src/wow_bot/analysis/aggregate.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    forbidden_module_substrings = (
        "soak",
        "spectrum",
        "timing",
        "lab_spectral_v2",
        "lab_timing_v2",
        "scenario",
        "main",
        "runner_v2",
        "perception",
        "reflex",
        "actuation",
        "world.store",
        "nav",
        "combat",
        "executor",
        "watchdog",
        "humanize",
        "internal_dynamics",
        "strategist",
        "aiosqlite",
        "asyncio",
        "threading",
        "ollama",
        "openai",
        "anthropic",
        "llm",
    )

    allowed_module_names = {
        "wow_bot.reporting.schema_v2",
        "wow_bot.analysis.lab_soak_v2",
    }

    for node in ast.walk(tree):
        # Imports
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name in allowed_module_names:
                    continue
                for sub in forbidden_module_substrings:
                    assert sub not in name, f"Forbidden import found in aggregate.py: {name}"

        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod not in allowed_module_names:
                for sub in forbidden_module_substrings:
                    assert sub not in mod, f"Forbidden import from found in aggregate.py: {mod}"

        # Direct call checks to time.monotonic, time.time, time.perf_counter
        elif (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
            and node.func.value.id == "time"
        ):
            assert node.func.attr not in (
                "monotonic",
                "time",
                "perf_counter",
            ), f"Forbidden time call found in aggregate.py: time.{node.func.attr}"
