"""Unit tests for Long-Running 24-Hour Soak Test Harness (Task 8.3).

Validates CLI argument parsing, resource snapshot data structures, log size
calculations, summary mathematics, linear memory growth slope regression,
strict JSON report serialization, crash/completion semantics, atomic file writing,
overwrite protection, missing metric handling, and process resource sampler safety.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

# Ensure repository root is in sys.path
_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.run_soak_test import parse_args, run_soak_test_async
from wow_bot.analysis.soak import (
    SOAK_REPORT_SCHEMA_VERSION,
    FakeResourceSampler,
    ProcessResourceSampler,
    ResourceSnapshot,
    SoakSample,
    build_soak_report,
    build_soak_summary,
    calculate_log_size,
    calculate_memory_slope,
    resolve_log_path,
    write_soak_report_atomically,
)


# Test A — Duration validation
def test_duration_validation() -> None:
    """Reject 0, negative, NaN, Inf, bool for --duration; accept positive finite."""
    assert parse_args(["--duration", "10"]).duration == 10.0
    assert parse_args(["--duration", "86400"]).duration == 86400.0

    invalid_values = ["0", "-5", "nan", "inf", "-inf"]
    for val in invalid_values:
        with pytest.raises(SystemExit):
            parse_args(["--duration", val])


# Test B — Sample interval validation
def test_sample_interval_validation() -> None:
    """Reject 0, negative, NaN, Inf, bool for --sample-interval; accept positive finite."""
    assert parse_args(["--sample-interval", "5"]).sample_interval == 5.0
    assert parse_args(["--sample-interval", "30"]).sample_interval == 30.0

    invalid_values = ["0", "-1", "nan", "inf", "-inf"]
    for val in invalid_values:
        with pytest.raises(SystemExit):
            parse_args(["--sample-interval", val])


# Test C — ResourceSnapshot representation
def test_resource_snapshot_immutability() -> None:
    """Ensure ResourceSnapshot is typed, immutable, and supports default process count."""
    snap = ResourceSnapshot(cpu_percent=12.5, memory_rss_mb=150.2, process_count=2)
    assert snap.cpu_percent == 12.5
    assert snap.memory_rss_mb == 150.2
    assert snap.process_count == 2

    with pytest.raises(AttributeError):
        snap.cpu_percent = 20.0  # type: ignore[misc]


# Test D — Summary calculation
def test_build_soak_summary_memory() -> None:
    """Verify initial, final, min, max, mean, and delta memory calculations."""
    samples = [
        SoakSample(
            elapsed_seconds=0.0,
            cpu_percent=10.0,
            memory_rss_mb=100.0,
            log_size_bytes=1000,
            progress_token=0,
            watchdog_alive=True,
            shutdown_requested=False,
        ),
        SoakSample(
            elapsed_seconds=1800.0,
            cpu_percent=15.0,
            memory_rss_mb=110.0,
            log_size_bytes=2000,
            progress_token=50,
            watchdog_alive=True,
            shutdown_requested=False,
        ),
        SoakSample(
            elapsed_seconds=3600.0,
            cpu_percent=20.0,
            memory_rss_mb=120.0,
            log_size_bytes=3000,
            progress_token=100,
            watchdog_alive=True,
            shutdown_requested=False,
        ),
    ]

    summary = build_soak_summary(samples)
    assert summary["initial_memory_mb"] == 100.0
    assert summary["final_memory_mb"] == 120.0
    assert summary["minimum_memory_mb"] == 100.0
    assert summary["maximum_memory_mb"] == 120.0
    assert summary["average_memory_mb"] == 110.0
    assert summary["memory_change_mb"] == 20.0
    assert summary["memory_growth_percent"] == 20.0


# Test E — Memory growth slope (positive linear growth)
def test_memory_growth_slope_positive() -> None:
    """Verify memory slope calculation with known positive linear growth in MB/hour."""
    # 10 MB growth per 1 hour = 10 MB/hour
    samples = [
        SoakSample(0.0, 10.0, 100.0, 0, 0, True, False),
        SoakSample(1800.0, 10.0, 105.0, 0, 1, True, False),
        SoakSample(3600.0, 10.0, 110.0, 0, 2, True, False),
    ]
    slope = calculate_memory_slope(samples)
    assert slope is not None
    assert abs(slope - 10.0) < 1e-3


# Test F — Stable memory slope (constant memory -> slope ~ 0.0)
def test_memory_growth_slope_constant() -> None:
    """Verify slope for constant memory samples is approximately 0."""
    samples = [
        SoakSample(0.0, 10.0, 150.0, 0, 0, True, False),
        SoakSample(1800.0, 10.0, 150.0, 0, 1, True, False),
        SoakSample(3600.0, 10.0, 150.0, 0, 2, True, False),
    ]
    slope = calculate_memory_slope(samples)
    assert slope is not None
    assert abs(slope) < 1e-3


# Test G — CPU summary statistics
def test_build_soak_summary_cpu() -> None:
    """Verify average and maximum CPU calculations."""
    samples = [
        SoakSample(0.0, 10.0, 100.0, 0, 0, True, False),
        SoakSample(10.0, 30.0, 100.0, 0, 1, True, False),
        SoakSample(20.0, 20.0, 100.0, 0, 2, True, False),
    ]
    summary = build_soak_summary(samples)
    assert summary["average_cpu_percent"] == 20.0
    assert summary["maximum_cpu_percent"] == 30.0


# Test H — Log size calculation
def test_calculate_log_size(tmp_path: Path) -> None:
    """Verify calculate_log_size for single file, nested directory, and missing path."""
    # Single file
    f = tmp_path / "test.log"
    f.write_text("1234567890")
    assert calculate_log_size(f) == 10

    # Nested directory
    sub = tmp_path / "logs"
    sub.mkdir()
    (sub / "a.log").write_text("hello")
    (sub / "b.log").write_text("world!")
    assert calculate_log_size(sub) == 11

    # Missing path
    assert calculate_log_size(tmp_path / "nonexistent.log") is None
    assert calculate_log_size(None) is None


# Test I — Log growth calculation
def test_log_growth_summary() -> None:
    """Verify initial/final log size and growth calculation."""
    samples = [
        SoakSample(0.0, 10.0, 100.0, 1024, 0, True, False),
        SoakSample(10.0, 10.0, 100.0, 2048, 1, True, False),
        SoakSample(20.0, 10.0, 100.0, 4096, 2, True, False),
    ]
    summary = build_soak_summary(samples)
    assert summary["initial_log_size_bytes"] == 1024
    assert summary["final_log_size_bytes"] == 4096
    assert summary["log_growth_bytes"] == 3072


# Test J — Report serialization
def test_report_serialization_strict_json(tmp_path: Path) -> None:
    """Verify report dictionary conforms to schema version 1 and produces strict valid JSON."""
    samples = [
        SoakSample(0.0, 10.0, 100.0, 500, 0, True, False),
        SoakSample(30.0, 15.0, 102.0, 600, 1, True, False),
    ]
    report = build_soak_report(
        scenario="peaceful_farm",
        seed=42,
        requested_duration_seconds=300.0,
        sample_interval_seconds=30.0,
        completed_duration_seconds=300.0,
        samples=samples,
        completed_normally=True,
        termination_reason="duration_completed",
    )

    assert report["schema_version"] == SOAK_REPORT_SCHEMA_VERSION
    assert report["run"]["scenario"] == "peaceful_farm"
    assert report["acceptance"]["zero_crash_target_met"] is True

    target_file = tmp_path / "report.json"
    write_soak_report_atomically(report, target_file)

    assert target_file.exists()
    loaded = json.loads(target_file.read_text(encoding="utf-8"))
    assert loaded["schema_version"] == SOAK_REPORT_SCHEMA_VERSION


# Test K — Normal completion semantics
def test_normal_completion_semantics() -> None:
    """Verify completed_normally=True and zero_crash_target_met=True on full duration."""
    samples = [SoakSample(0.0, 10.0, 100.0, 0, 0, True, False)]
    report = build_soak_report(
        scenario="peaceful_farm",
        seed=42,
        requested_duration_seconds=100.0,
        sample_interval_seconds=10.0,
        completed_duration_seconds=100.0,
        samples=samples,
        completed_normally=True,
        termination_reason="duration_completed",
    )
    assert report["termination"]["completed_normally"] is True
    assert report["acceptance"]["zero_crash_target_met"] is True


# Test L — Failure semantics
def test_failure_semantics() -> None:
    """Verify completed_normally=False and zero_crash_target_met=False on pipeline failure."""
    samples = [SoakSample(0.0, 10.0, 100.0, 0, 0, True, False)]
    report = build_soak_report(
        scenario="peaceful_farm",
        seed=42,
        requested_duration_seconds=100.0,
        sample_interval_seconds=10.0,
        completed_duration_seconds=20.0,
        samples=samples,
        completed_normally=False,
        termination_reason="pipeline_failure",
        failure_type="RuntimeError",
        failure_message="Pipeline error",
    )
    assert report["termination"]["completed_normally"] is False
    assert report["acceptance"]["zero_crash_target_met"] is False
    assert report["termination"]["failure_type"] == "RuntimeError"


# Test M — Memory acceptance status
def test_memory_acceptance_status() -> None:
    """Verify memory_stability_status is always 'manual_review_required'."""
    samples = [SoakSample(0.0, 10.0, 100.0, 0, 0, True, False)]
    report = build_soak_report(
        scenario="peaceful_farm",
        seed=42,
        requested_duration_seconds=10.0,
        sample_interval_seconds=5.0,
        completed_duration_seconds=10.0,
        samples=samples,
        completed_normally=True,
        termination_reason="duration_completed",
    )
    assert report["acceptance"]["memory_stability_status"] == "manual_review_required"


# Test N — Atomic report write
def test_atomic_report_write(tmp_path: Path) -> None:
    """Verify atomic write replaces target file and cleans up temp files."""
    dest = tmp_path / "soak.json"
    report = {"schema_version": 1, "run": {}}

    write_soak_report_atomically(report, dest)
    assert dest.exists()
    assert json.loads(dest.read_text())["schema_version"] == 1

    tmp_leftovers = list(tmp_path.glob("*.tmp"))
    assert len(tmp_leftovers) == 0


# Test O — Existing output protection
@pytest.mark.asyncio
async def test_existing_output_protection(tmp_path: Path) -> None:
    """Verify harness refuses accidental overwrite when output file already exists."""
    dest = tmp_path / "soak_24h.json"
    dest.write_text("{}")

    code = await run_soak_test_async(
        scenario="peaceful_farm",
        duration=10.0,
        sample_interval=1.0,
        seed=42,
        output_path=dest,
    )
    assert code == 1


# Test P — Missing optional log metric handling
def test_missing_log_metric_handling() -> None:
    """Verify missing log path does not crash summary or report generation."""
    samples = [
        SoakSample(0.0, 10.0, 100.0, None, 0, True, False),
        SoakSample(10.0, 10.0, 100.0, None, 1, True, False),
    ]
    summary = build_soak_summary(samples)
    assert summary["initial_log_size_bytes"] is None
    assert summary["final_log_size_bytes"] is None
    assert summary["log_growth_bytes"] is None

    _path, status = resolve_log_path("non_existent_dir_xyz/missing.log")
    assert status == "unavailable"


# Test Q — ResourceSampler exception safety
def test_process_resource_sampler_safety() -> None:
    """Verify ProcessResourceSampler safely handles sampling current process."""
    sampler = ProcessResourceSampler()
    snap = sampler.sample()
    assert snap.process_count is not None and snap.process_count >= 1
    assert snap.memory_rss_mb is None or snap.memory_rss_mb > 0.0

    fake_sampler = FakeResourceSampler([
        ResourceSnapshot(cpu_percent=12.0, memory_rss_mb=200.0, process_count=1)
    ])
    f_snap = fake_sampler.sample()
    assert f_snap.cpu_percent == 12.0
    assert f_snap.memory_rss_mb == 200.0
