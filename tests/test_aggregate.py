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
    CrashTrendSummary,
    HumanizerFitSummary,
    LogTrendSummary,
    RSSTrendSummary,
    StabilitySummaries,
    aggregate_sessions,
    aggregate_stability_summaries,
    load_aggregate,
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
        "wow_bot.analysis.lab_timing_v2",
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


# ---------------------------------------------------------------------------
# T-FIX-18: Stability Summaries Acceptance Tests
# ---------------------------------------------------------------------------


def test_aggregate_stability_summaries_with_real_soak_fixture() -> None:
    """Acceptance (T-FIX-18): Real soak report fixture (runs/lab/soak-1h) aggregates stability summaries correctly."""
    soak_sess_dir = Path("runs/lab/soak-1h")
    assert soak_sess_dir.exists(), f"Real fixture missing: {soak_sess_dir}"

    report = aggregate_sessions([soak_sess_dir], now=FIXED_NOW)

    # 1. Report v2 is absent in soak-1h -> report_count is 0 distinctly reported
    assert report.perception_agnostic.report_count == 0
    assert report.perception_agnostic.soak_report_count == 1
    assert [s.has_report_v2 for s in report.session_sources] == [False]
    assert [s.has_soak_report for s in report.session_sources] == [True]

    # 2. Stability summaries populated
    stab = report.stability_summaries
    assert stab is not None
    assert isinstance(stab, StabilitySummaries)
    assert isinstance(stab.crash_trend, CrashTrendSummary)
    assert isinstance(stab.rss_trend, RSSTrendSummary)
    assert isinstance(stab.log_trend, LogTrendSummary)
    assert isinstance(stab.humanizer_fit, HumanizerFitSummary)
    assert stab.soak_report_count == 1
    assert stab.report_v2_count == 0

    # 3. Crash trend: zero crashes in 1 report
    assert stab.crash_trend.input_report_count == 1
    assert stab.crash_trend.total_crashes == 0
    assert stab.crash_trend.crash_rate == 0.0
    assert stab.crash_trend.crash_reasons == ()

    # 4. RSS trend: values match real soak report fixture
    assert stab.rss_trend.input_report_count == 1
    assert stab.rss_trend.peak_bytes_max == 142811136
    assert stab.rss_trend.mean_slope_bytes_per_hour == pytest.approx(895226.59, rel=1e-3)
    assert stab.rss_trend.max_slope_bytes_per_hour == pytest.approx(895226.59, rel=1e-3)
    assert stab.rss_trend.any_growth_suspect is False

    # 5. Log trend: 0 slope in real soak fixture
    assert stab.log_trend.input_report_count == 1
    assert stab.log_trend.mean_slope_bytes_per_hour == 0.0
    assert stab.log_trend.any_growth_suspect is False

    # 6. Humanizer fit: no report_v2 input reports -> distinctly "no_input_reports"
    assert stab.humanizer_fit.input_report_count == 0
    assert stab.humanizer_fit.status == "no_input_reports"
    assert stab.humanizer_fit.total_samples == 0
    assert stab.humanizer_fit.pit_p_value is None
    assert stab.humanizer_fit.pit_passes is None


def test_aggregate_stability_summaries_empty_sessions_distinguishes_no_reports() -> None:
    """Acceptance (T-FIX-18): Zero counts are distinguishable from 'no input reports'."""
    stab = aggregate_stability_summaries([], [])
    assert stab.soak_report_count == 0
    assert stab.report_v2_count == 0

    # Crash trend: metrics are None when input_report_count == 0 (not 0.0 rate)
    assert stab.crash_trend.input_report_count == 0
    assert stab.crash_trend.total_crashes is None
    assert stab.crash_trend.crash_rate is None
    assert stab.crash_trend.crash_reasons == ()

    # RSS & Log trends: None when no input reports
    assert stab.rss_trend.input_report_count == 0
    assert stab.rss_trend.peak_bytes_max is None
    assert stab.rss_trend.mean_slope_bytes_per_hour is None
    assert stab.rss_trend.any_growth_suspect is None

    assert stab.log_trend.input_report_count == 0
    assert stab.log_trend.mean_slope_bytes_per_hour is None
    assert stab.log_trend.any_growth_suspect is None

    # Humanizer fit: status is 'no_input_reports', metrics None
    assert stab.humanizer_fit.input_report_count == 0
    assert stab.humanizer_fit.status == "no_input_reports"
    assert stab.humanizer_fit.total_samples == 0
    assert stab.humanizer_fit.pit_passes is None


def test_aggregate_stability_summaries_with_crashes_and_humanizer_samples(tmp_path: Path) -> None:
    """Acceptance (T-FIX-18): Crash trends, RSS/log trends, and humanizer PIT/KS fits are summarized across >= 1 reports."""
    # Build soak report 1: normal, no crash
    s1_samples = [
        SoakSample(
            ts=float(i),
            cpu_percent=10.0,
            rss_bytes=1000 + i * 10,
            log_size_bytes=100 + i * 5,
            position_delta=1.0,
            inventory_delta=1,
            successful_actions_total=i,
            reflex_ticks_total=i * 10,
        )
        for i in range(10)
    ]
    s1_rep = build_soak_report(s1_samples, session_id="s1_soak", config=SoakConfig())

    # Build soak report 2: crashed with reason
    s2_samples = [
        SoakSample(
            ts=float(i),
            cpu_percent=20.0,
            rss_bytes=2000 + i * 20,
            log_size_bytes=200 + i * 10,
            position_delta=1.0,
            inventory_delta=1,
            successful_actions_total=i,
            reflex_ticks_total=i * 10,
        )
        for i in range(10)
    ]
    s2_rep = build_soak_report(
        s2_samples,
        session_id="s2_soak",
        config=SoakConfig(),
        crash=True,
        crash_reason="watchdog_stagnation_timeout",
    )

    # Build report_v2 with humanizer samples
    humanizer_samples = [100.0, 150.0, 200.0, 250.0] * 20  # 80 samples
    r1 = {
        "schema_version": REPORT_SCHEMA_VERSION,
        "meta": {"session_id": "r1"},
        "humanizer": {
            "action_count": 80,
            "interval_samples_ms": humanizer_samples,
            "pause_count": 0,
            "pause_duration_ms_mean": None,
            "miss_click_count": 0,
            "config_snapshot": {},
        },
    }

    stab = aggregate_stability_summaries([s1_rep, s2_rep], [r1])

    # Check crash trend: 1 out of 2 crashed -> rate 0.5
    assert stab.soak_report_count == 2
    assert stab.crash_trend.input_report_count == 2
    assert stab.crash_trend.total_crashes == 1
    assert stab.crash_trend.crash_rate == pytest.approx(0.5)
    assert stab.crash_trend.crash_reasons == ("watchdog_stagnation_timeout",)

    # Check RSS trend
    assert stab.rss_trend.input_report_count == 2
    assert stab.rss_trend.peak_bytes_max == max(
        s1_rep.summary.rss_peak_bytes, s2_rep.summary.rss_peak_bytes
    )
    expected_mean_slope = (
        s1_rep.summary.rss_slope_bytes_per_hour + s2_rep.summary.rss_slope_bytes_per_hour
    ) / 2.0
    assert stab.rss_trend.mean_slope_bytes_per_hour == pytest.approx(expected_mean_slope)

    # Check log trend
    assert stab.log_trend.input_report_count == 2
    expected_log_slope = (
        s1_rep.summary.log_slope_bytes_per_hour + s2_rep.summary.log_slope_bytes_per_hour
    ) / 2.0
    assert stab.log_trend.mean_slope_bytes_per_hour == pytest.approx(expected_log_slope)

    # Check humanizer fit: 80 samples -> evaluated
    assert stab.humanizer_fit.input_report_count == 1
    assert stab.humanizer_fit.total_samples == 80
    assert stab.humanizer_fit.status == "evaluated"
    assert stab.humanizer_fit.pit_p_value is not None
    assert isinstance(stab.humanizer_fit.pit_passes, bool)
    assert stab.humanizer_fit.ks_statistic is not None
    assert stab.humanizer_fit.ks_p_value is not None
    assert isinstance(stab.humanizer_fit.ks_passes, bool)


def test_load_aggregate_frozen_and_round_trip(tmp_path: Path) -> None:
    """Acceptance (T-FIX-18): load_aggregate parses frozen aggregate_v1.json and supports round-trip."""
    frozen_path = Path("runs/lab/aggregate-1h/aggregate_v1.json")
    assert frozen_path.exists()

    # 1. Parse frozen artifact
    frozen_rep = load_aggregate(frozen_path)
    assert frozen_rep.schema_version == AGGREGATE_SCHEMA_VERSION
    assert frozen_rep.session_count == 1
    assert frozen_rep.session_ids == ("soak-1h-20260922",)
    assert frozen_rep.perception_agnostic.report_count == 0
    assert frozen_rep.perception_agnostic.soak_report_count == 1
    assert frozen_rep.stability_summaries is None
    assert len(frozen_rep.non_claims) == 4

    # 2. Round-trip serialization with stability_summaries populated
    s_dir = tmp_path / "sess_round_trip"
    s_dir.mkdir(parents=True)
    (s_dir / "session.json").write_text(
        json.dumps({"session_id": "sess_round_trip"}), encoding="utf-8"
    )
    s_samples = [
        SoakSample(
            ts=float(i),
            cpu_percent=5.0,
            rss_bytes=1000,
            log_size_bytes=100,
            position_delta=0.0,
            inventory_delta=0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
        for i in range(5)
    ]
    write_soak_report(
        build_soak_report(s_samples, session_id="sess_round_trip", config=SoakConfig()),
        s_dir / "soak_report.json",
    )

    out_file = tmp_path / "output_agg.json"
    rep_gen = aggregate_sessions([s_dir], now=FIXED_NOW)
    assert rep_gen.stability_summaries is not None
    write_aggregate(rep_gen, out_file)

    loaded_gen = load_aggregate(out_file)
    assert loaded_gen.schema_version == AGGREGATE_SCHEMA_VERSION
    assert loaded_gen.session_ids == ("sess_round_trip",)
    assert loaded_gen.stability_summaries is not None
    assert loaded_gen.stability_summaries.soak_report_count == 1
    assert loaded_gen.stability_summaries.crash_trend.total_crashes == 0
    assert loaded_gen.stability_summaries.rss_trend.peak_bytes_max == 1000


def test_documented_duplicate_session_dedup_behaviour_matches_code(tmp_path: Path) -> None:
    """Acceptance (T-FIX-19): The documented duplicate-session behavior matches code implementation."""
    r1_dict = make_full_report_v2_dict(
        "dup_sess",
        tick_count=100,
        mean_tick_period_ms=50.0,
        p99_tick_period_ms=60.0,
        actuator_result_count=10,
        latency_ms_p50=10.0,
        latency_ms_p95=20.0,
        strategist_call_count=5,
        strategist_p50=100.0,
        strategist_p95=200.0,
    )
    s1 = make_session_dir(tmp_path, "dup_sess", report_v2_dict=r1_dict)

    # Pass the same session directory twice
    rep = aggregate_sessions([s1, s1], now=FIXED_NOW)

    # 1. session_ids is deduplicated to unique set
    assert rep.session_ids == ("dup_sess",)
    assert rep.session_count == 1

    # 2. session_sources preserves every input directory (length 2)
    assert len(rep.session_sources) == 2
    assert [src.session_id for src in rep.session_sources] == ["dup_sess", "dup_sess"]

    # 3. Counters are accumulated twice across loaded reports
    assert rep.perception_agnostic.report_count == 2
    assert rep.perception_agnostic.reflex_tick_count_total == 200
    assert rep.perception_agnostic.action_result_count_total == 20
    assert rep.perception_agnostic.strategist_call_count_total == 10

    # 4. Weighted means remain mathematically equivalent
    assert rep.perception_agnostic.reflex_tick_period_ms_mean == pytest.approx(50.0)
    assert rep.perception_agnostic.action_latency_ms_p50 == pytest.approx(10.0)
    assert rep.perception_agnostic.strategist_latency_ms_p50 == pytest.approx(100.0)


