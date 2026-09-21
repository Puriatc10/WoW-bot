"""Unit and CLI integration tests for Lab Soak Harness V2 (Task T10.4)."""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import pytest

from scripts.lab.soak_v2 import (
    FakeClock,
    FakeProcessHandle,
    FakeProgressSource,
    FakeResourceSampler,
    SoakRunner,
    main,
)
from wow_bot.analysis.lab_soak_v2 import (
    SOAK_SCHEMA_VERSION,
    LabSoakError,
    ResourceSnapshot,
    SoakConfig,
    SoakReport,
    SoakSample,
    SoakSummary,
    build_soak_report,
    build_soak_summary,
    validate_soak_report_dict,
    write_soak_report,
)

# --- Library Dataclass Validation Tests ---


def test_soak_config_validation() -> None:
    # Valid default config
    cfg = SoakConfig()
    assert cfg.sample_interval_s == 1.0

    # Invalid sample_interval_s
    with pytest.raises(ValueError, match="sample_interval_s"):
        SoakConfig(sample_interval_s=0.0)
    with pytest.raises(ValueError, match="sample_interval_s"):
        SoakConfig(sample_interval_s=-1.0)

    # Invalid max_samples
    with pytest.raises(ValueError, match="max_samples"):
        SoakConfig(max_samples=0)

    # Invalid crash_exit_code
    with pytest.raises(ValueError, match="crash_exit_code"):
        SoakConfig(crash_exit_code=0)

    # Invalid memory_slope_window
    with pytest.raises(ValueError, match="memory_slope_window"):
        SoakConfig(memory_slope_window=9)

    # Invalid thresholds
    with pytest.raises(ValueError, match="memory_growth_threshold_bytes_per_hour"):
        SoakConfig(memory_growth_threshold_bytes_per_hour=-1.0)
    with pytest.raises(ValueError, match="log_growth_threshold_bytes_per_hour"):
        SoakConfig(log_growth_threshold_bytes_per_hour=-5.0)

    # Non-finite float
    with pytest.raises(ValueError, match="sample_interval_s"):
        SoakConfig(sample_interval_s=float("nan"))


def test_resource_snapshot_validation() -> None:
    # Valid
    snap = ResourceSnapshot(ts=1.0, cpu_percent=5.0, rss_bytes=1000, log_size_bytes=500)
    assert snap.ts == 1.0

    # Invalid cpu_percent
    with pytest.raises(ValueError, match="cpu_percent"):
        ResourceSnapshot(ts=1.0, cpu_percent=-0.1, rss_bytes=1000, log_size_bytes=500)

    # Invalid rss_bytes
    with pytest.raises(ValueError, match="rss_bytes"):
        ResourceSnapshot(ts=1.0, cpu_percent=5.0, rss_bytes=-1, log_size_bytes=500)


def test_soak_sample_validation() -> None:
    # Valid
    s = SoakSample(
        ts=0.0,
        cpu_percent=10.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.5,
        inventory_delta=2,
        successful_actions_total=5,
        reflex_ticks_total=10,
    )
    assert s.ts == 0.0

    # Non-finite ts
    with pytest.raises(ValueError, match="ts"):
        SoakSample(
            ts=float("nan"),
            cpu_percent=10.0,
            rss_bytes=100,
            log_size_bytes=50,
            position_delta=1.5,
            inventory_delta=2,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )

    # Negative field
    with pytest.raises(ValueError, match="position_delta"):
        SoakSample(
            ts=0.0,
            cpu_percent=10.0,
            rss_bytes=100,
            log_size_bytes=50,
            position_delta=-0.5,
            inventory_delta=2,
            successful_actions_total=5,
            reflex_ticks_total=10,
        )


def test_soak_summary_invariants() -> None:
    # Negative field in SoakSummary
    with pytest.raises(ValueError, match="sample_count"):
        SoakSummary(
            sample_count=-1,
            duration_s=10.0,
            cpu_mean_percent=5.0,
            cpu_max_percent=10.0,
            rss_start_bytes=100,
            rss_end_bytes=200,
            rss_peak_bytes=200,
            rss_slope_bytes_per_hour=0.0,
            rss_growth_suspect=False,
            log_start_bytes=50,
            log_end_bytes=60,
            log_slope_bytes_per_hour=0.0,
            log_growth_suspect=False,
            position_delta_total=1.0,
            inventory_delta_total=0,
            successful_actions_total=0,
            reflex_ticks_total=0,
            reflex_tick_rate_hz=0.0,
        )


def test_soak_report_validation() -> None:
    summary = SoakSummary(
        sample_count=1,
        duration_s=0.0,
        cpu_mean_percent=5.0,
        cpu_max_percent=5.0,
        rss_start_bytes=100,
        rss_end_bytes=100,
        rss_peak_bytes=100,
        rss_slope_bytes_per_hour=0.0,
        rss_growth_suspect=False,
        log_start_bytes=50,
        log_end_bytes=50,
        log_slope_bytes_per_hour=0.0,
        log_growth_suspect=False,
        position_delta_total=1.0,
        inventory_delta_total=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
        reflex_tick_rate_hz=0.0,
    )
    sample = SoakSample(
        ts=0.0,
        cpu_percent=5.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.0,
        inventory_delta=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )

    # Schema version mismatch
    with pytest.raises(ValueError, match="schema_version"):
        SoakReport(
            schema_version=99,
            session_id="test-session",
            started_at=0.0,
            finished_at=0.0,
            summary=summary,
            samples=(sample,),
            crash=False,
            crash_reason=None,
        )

    # finished_at < started_at
    with pytest.raises(ValueError, match="finished_at"):
        SoakReport(
            schema_version=SOAK_SCHEMA_VERSION,
            session_id="test-session",
            started_at=10.0,
            finished_at=5.0,
            summary=summary,
            samples=(sample,),
            crash=False,
            crash_reason=None,
        )

    # crash=True with crash_reason=None
    with pytest.raises(ValueError, match="crash_reason"):
        SoakReport(
            schema_version=SOAK_SCHEMA_VERSION,
            session_id="test-session",
            started_at=0.0,
            finished_at=0.0,
            summary=summary,
            samples=(sample,),
            crash=True,
            crash_reason=None,
        )

    # crash=False with crash_reason set
    with pytest.raises(ValueError, match="crash_reason"):
        SoakReport(
            schema_version=SOAK_SCHEMA_VERSION,
            session_id="test-session",
            started_at=0.0,
            finished_at=0.0,
            summary=summary,
            samples=(sample,),
            crash=False,
            crash_reason="some reason",
        )


# --- Library Function Tests ---


def test_build_soak_summary_validation_and_calculations() -> None:
    # Empty sequence
    with pytest.raises(LabSoakError, match="empty sample sequence"):
        build_soak_summary([])

    # Non-monotone ts
    s1 = SoakSample(
        ts=10.0,
        cpu_percent=5.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.0,
        inventory_delta=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )
    s2 = SoakSample(
        ts=5.0,
        cpu_percent=10.0,
        rss_bytes=200,
        log_size_bytes=60,
        position_delta=2.0,
        inventory_delta=1,
        successful_actions_total=2,
        reflex_ticks_total=10,
    )
    with pytest.raises(LabSoakError, match="strictly increasing"):
        build_soak_summary([s1, s2])

    # Valid sequence calculations
    s3 = SoakSample(
        ts=20.0,
        cpu_percent=15.0,
        rss_bytes=300,
        log_size_bytes=70,
        position_delta=3.0,
        inventory_delta=2,
        successful_actions_total=4,
        reflex_ticks_total=20,
    )
    s4 = SoakSample(
        ts=30.0,
        cpu_percent=20.0,
        rss_bytes=10_000_000,
        log_size_bytes=100_000_000,
        position_delta=4.0,
        inventory_delta=3,
        successful_actions_total=6,
        reflex_ticks_total=30,
    )

    summ = build_soak_summary([s1, s3, s4])
    assert summ.sample_count == 3
    assert summ.duration_s == 20.0  # 30.0 - 10.0
    assert summ.cpu_mean_percent == pytest.approx((5.0 + 15.0 + 20.0) / 3.0)
    assert summ.cpu_max_percent == 20.0
    assert summ.rss_start_bytes == 100
    assert summ.rss_end_bytes == 10_000_000
    assert summ.rss_peak_bytes == 10_000_000
    assert summ.position_delta_total == pytest.approx(1.0 + 3.0 + 4.0)
    assert summ.inventory_delta_total == 3  # 3 - 0
    assert summ.successful_actions_total == 6
    assert summ.reflex_ticks_total == 30
    assert summ.reflex_tick_rate_hz == pytest.approx(30 / 20.0)
    assert summ.rss_growth_suspect is True
    assert summ.log_growth_suspect is True


def test_build_soak_report_empty_and_non_empty() -> None:
    # Empty session_id
    with pytest.raises(LabSoakError, match="session_id"):
        build_soak_report([], session_id="", crash=False)

    # Empty samples non-crash -> raises
    with pytest.raises(LabSoakError, match="empty sample sequence"):
        build_soak_report([], session_id="s1", crash=False)

    # Empty samples crash=True -> returns report with empty summary
    rep_crash = build_soak_report([], session_id="s1", crash=True, crash_reason="Out of memory")
    assert rep_crash.crash is True
    assert rep_crash.crash_reason == "Out of memory"
    assert rep_crash.summary.sample_count == 0

    # Non-empty samples
    s = SoakSample(
        ts=1.0,
        cpu_percent=5.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.0,
        inventory_delta=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )
    rep = build_soak_report([s], session_id="s1", crash=False)
    assert rep.schema_version == SOAK_SCHEMA_VERSION
    assert rep.session_id == "s1"
    assert rep.crash is False
    assert len(rep.samples) == 1


def test_validate_soak_report_dict() -> None:
    s = SoakSample(
        ts=1.0,
        cpu_percent=5.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.0,
        inventory_delta=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )
    rep = build_soak_report([s], session_id="s1", crash=False)
    dict_data = rep.to_json()

    # Valid dict passes
    validate_soak_report_dict(dict_data)

    # Missing key
    bad_dict = dict(dict_data)
    del bad_dict["session_id"]
    with pytest.raises(LabSoakError, match="Missing required top-level keys"):
        validate_soak_report_dict(bad_dict)

    # Unknown key
    bad_dict2 = dict(dict_data)
    bad_dict2["unknown_key"] = 123
    with pytest.raises(LabSoakError, match="Unknown top-level keys"):
        validate_soak_report_dict(bad_dict2)

    # Mismatched schema version
    bad_dict3 = dict(dict_data)
    bad_dict3["schema_version"] = 1
    with pytest.raises(LabSoakError, match="Expected schema_version"):
        validate_soak_report_dict(bad_dict3)

    # Ensure input dict was not mutated
    assert "unknown_key" not in dict_data


def test_to_json_omits_samples_when_large() -> None:
    # Construct sequence with > 10,000 samples
    samples = tuple(
        SoakSample(
            ts=float(i),
            cpu_percent=1.0,
            rss_bytes=100,
            log_size_bytes=50,
            position_delta=0.0,
            inventory_delta=0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )
        for i in range(10_001)
    )
    rep = build_soak_report(samples, session_id="large-session", crash=False)
    dict_data = rep.to_json()

    assert "samples" not in dict_data
    assert dict_data.get("samples_omitted") is True
    validate_soak_report_dict(dict_data)


def test_write_soak_report_atomic(tmp_path: Path) -> None:
    s = SoakSample(
        ts=1.0,
        cpu_percent=5.0,
        rss_bytes=100,
        log_size_bytes=50,
        position_delta=1.0,
        inventory_delta=0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )
    rep = build_soak_report([s], session_id="s1", crash=False)
    out_file = tmp_path / "subdir" / "soak_report.json"

    write_soak_report(rep, out_file)

    assert out_file.exists()
    assert not (tmp_path / "subdir" / "soak_report.json.tmp").exists()

    content = out_file.read_text(encoding="utf-8")
    assert '"schema_version": 2' in content


# --- CLI and SoakRunner Tests ---


def test_soak_runner_fake_clock_sampling() -> None:
    fake_clock = FakeClock(start_ts=0.0)
    sampler = FakeResourceSampler()
    proc = FakeProcessHandle(alive=True)
    prog = FakeProgressSource()
    cfg = SoakConfig(sample_interval_s=1.0)

    runner = SoakRunner(
        clock=fake_clock.now,
        sleep_fn=fake_clock.sleep,
        sampler=sampler,
        process=proc,
        progress_source=prog,
        config=cfg,
    )

    duration = 5.0
    samples = runner.run(duration)

    # ceil(duration / interval) + 1 = 6 samples (ts 0.0, 1.0, 2.0, 3.0, 4.0, 5.0)
    assert len(samples) == 6
    assert samples[0].ts == 0.0
    assert samples[-1].ts == 5.0


def test_soak_runner_early_process_exit() -> None:
    fake_clock = FakeClock(start_ts=0.0)
    sampler = FakeResourceSampler()
    # Process dies on 3rd sample check
    proc = FakeProcessHandle(alive=True)

    def custom_sleep(d: float) -> None:
        fake_clock.sleep(d)
        if fake_clock.now() >= 2.0:
            proc._alive = False

    prog = FakeProgressSource()
    cfg = SoakConfig(sample_interval_s=1.0)

    runner = SoakRunner(
        clock=fake_clock.now,
        sleep_fn=custom_sleep,
        sampler=sampler,
        process=proc,
        progress_source=prog,
        config=cfg,
    )

    samples = runner.run(10.0)
    assert len(samples) == 3  # ts 0.0, 1.0, 2.0
    assert samples[-1].ts == 2.0


def test_cli_help() -> None:
    ret = main(["--help"])
    assert ret == 0


def test_cli_missing_required_args() -> None:
    ret = main([])
    assert ret == 2


def test_cli_dry_run_success(tmp_path: Path) -> None:
    session_dir = tmp_path / "session_1"
    args = [
        "--session-dir",
        str(session_dir),
        "--session-id",
        "dry-run-sess",
        "--duration-s",
        "10.0",
        "--sample-interval-s",
        "1.0",
        "--mode",
        "MOCK",
        "--dry-run",
    ]

    ret = main(args)
    assert ret == 0

    report_path = session_dir / "soak_report.json"
    assert report_path.exists()

    import json

    data = json.loads(report_path.read_text(encoding="utf-8"))
    validate_soak_report_dict(data)
    assert data["session_id"] == "dry-run-sess"
    assert data["crash"] is False
    assert len(data["samples"]) > 0


def test_cli_crash_with_and_without_write_crash_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Monkeypatch SoakRunner.run to raise unhandled exception
    def bad_run(self: Any, duration_s: float) -> Any:
        raise RuntimeError("Simulated harness crash")

    monkeypatch.setattr(SoakRunner, "run", bad_run)

    session_dir = tmp_path / "crash_session"
    args = [
        "--session-dir",
        str(session_dir),
        "--session-id",
        "crash-sess",
        "--duration-s",
        "5.0",
        "--dry-run",
        "--write-crash-on-error",
    ]

    ret = main(args)
    assert ret == 3

    crash_file = session_dir / "crash.json"
    assert crash_file.exists()
    assert "Simulated harness crash" in crash_file.read_text(encoding="utf-8")

    # Test without --write-crash-on-error flag
    session_dir2 = tmp_path / "crash_session2"
    args2 = [
        "--session-dir",
        str(session_dir2),
        "--session-id",
        "crash-sess2",
        "--duration-s",
        "5.0",
        "--dry-run",
    ]

    ret2 = main(args2)
    assert ret2 == 3
    assert not (session_dir2 / "crash.json").exists()


# --- Static AST Import & Purity Checks ---


def test_static_ast_checks_library() -> None:
    lib_path = Path("src/wow_bot/analysis/lab_soak_v2.py")
    tree = ast.parse(lib_path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.analysis.soak",
        "wow_bot.analysis.spectrum",
        "wow_bot.analysis.timing",
        "wow_bot.analysis.lab_spectral_v2",
        "wow_bot.analysis.lab_timing_v2",
        "wow_bot.reporting.scenario",
        "wow_bot.reporting.schema_v2",
        "wow_bot.reporting.lab_pipeline_v2",
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
        "wow_bot.strategist",
        "aiosqlite",
        "asyncio",
        "threading",
        "numpy",
        "scipy",
    }

    forbidden_time_calls = {"monotonic", "time", "perf_counter"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_modules:
                    assert not alias.name.startswith(forbidden), f"Forbidden import found: {alias.name}"
                assert "ollama" not in alias.name.lower()
                assert "openai" not in alias.name.lower()
                assert "anthropic" not in alias.name.lower()
                assert "llm" not in alias.name.lower()

        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forbidden in forbidden_modules:
                    assert not node.module.startswith(forbidden), f"Forbidden import from found: {node.module}"
                assert "ollama" not in node.module.lower()
                assert "openai" not in node.module.lower()
                assert "anthropic" not in node.module.lower()
                assert "llm" not in node.module.lower()

        elif (
            isinstance(node, ast.Attribute)
            and node.attr in forbidden_time_calls
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            pytest.fail(f"Forbidden time call found in library: time.{node.attr}")


def test_static_ast_checks_script() -> None:
    script_path = Path("scripts/lab/soak_v2.py")
    tree = ast.parse(script_path.read_text(encoding="utf-8"))

    forbidden_script_modules = {
        "wow_bot.analysis.soak",
        "numpy",
        "scipy",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_script_modules:
                    assert not alias.name.startswith(forbidden), f"Forbidden script import found: {alias.name}"

        elif isinstance(node, ast.ImportFrom) and node.module:
            for forbidden in forbidden_script_modules:
                assert not node.module.startswith(forbidden), f"Forbidden script import from found: {node.module}"
