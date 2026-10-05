"""Tests for cross-platform resource metric semantics and telemetry (T-FIX-14)."""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import pytest

from wow_bot.analysis.lab_soak_v2 import (
    ProcessResourceSampler,
    ResourceSnapshot,
    make_default_resource_sampler,
)

# ---------------------------------------------------------------------------
# Acceptance Criterion 1: Peak-vs-current mismatch resolution and typing/labelling
# ---------------------------------------------------------------------------


@pytest.mark.skipif(sys.platform != "win32", reason="Windows test only")
def test_windows_sampler_peak_vs_current_modes_and_labels() -> None:
    """Acceptance: WindowsResourceSampler default mode is 'peak' with label 'peak_rss',

    and opt-in 'current' mode has label 'current_working_set'.
    """
    from wow_bot.analysis.windows_sampler import WindowsResourceSampler

    # Default mode is peak
    peak_sampler = WindowsResourceSampler()
    assert peak_sampler.metric_mode == "peak"
    assert peak_sampler.metric_label == "peak_rss"

    snap_peak = peak_sampler.sample(now=10.0)
    assert isinstance(snap_peak, ResourceSnapshot)
    assert snap_peak.metric_label == "peak_rss"
    assert snap_peak.rss_bytes > 0

    # Explicit opt-in current mode
    curr_sampler = WindowsResourceSampler(metric_mode="current")
    assert curr_sampler.metric_mode == "current"
    assert curr_sampler.metric_label == "current_working_set"

    snap_curr = curr_sampler.sample(now=20.0)
    assert isinstance(snap_curr, ResourceSnapshot)
    assert snap_curr.metric_label == "current_working_set"
    assert snap_curr.rss_bytes > 0

    # Invalid mode raises ValueError
    with pytest.raises(ValueError, match="Invalid metric_mode 'invalid'"):
        WindowsResourceSampler(metric_mode="invalid")


@pytest.mark.skipif(sys.platform != "win32", reason="Windows test only")
def test_windows_sampler_peak_mode_is_monotonically_non_decreasing() -> None:
    """Acceptance: In 'peak' mode on Windows, PeakWorkingSetSize is monotonically non-decreasing."""
    from wow_bot.analysis.windows_sampler import WindowsResourceSampler

    sampler = WindowsResourceSampler(metric_mode="peak")
    snap1 = sampler.sample(now=1.0)

    # Allocate a large buffer to increase memory usage
    blob = bytearray(30 * 1024 * 1024)
    # Touch pages
    for i in range(0, len(blob), 4096):
        blob[i] = 1

    snap2 = sampler.sample(now=2.0)
    assert snap2.rss_bytes >= snap1.rss_bytes, "Peak RSS must be non-decreasing"

    # Delete reference to allow deallocation
    del blob
    snap3 = sampler.sample(now=3.0)
    assert snap3.rss_bytes >= snap2.rss_bytes, "Peak RSS must not decrease even after deallocation"


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX test only")
def test_posix_sampler_metric_label_and_mode() -> None:
    """Acceptance: On POSIX, ProcessResourceSampler exposes label 'peak_rss' and mode 'peak'."""
    sampler = ProcessResourceSampler()
    assert sampler.metric_mode == "peak"
    assert sampler.metric_label == "peak_rss"

    snap = sampler.sample(now=1.0)
    assert snap.metric_label == "peak_rss"
    assert snap.rss_bytes > 0


def test_make_default_resource_sampler_cross_platform_label() -> None:
    """Acceptance: make_default_resource_sampler defaults to peak_rss on all platforms."""
    sampler = make_default_resource_sampler()
    assert getattr(sampler, "metric_label", None) == "peak_rss"
    snap = sampler.sample(now=10.0)
    assert snap.metric_label == "peak_rss"
    assert snap.rss_bytes > 0


def test_resource_snapshot_metric_label_validation() -> None:
    """Acceptance: ResourceSnapshot validates metric_label as non-empty string."""
    snap = ResourceSnapshot(ts=1.0, cpu_percent=0.0, rss_bytes=1024, log_size_bytes=0)
    assert snap.metric_label == "peak_rss"

    snap_custom = ResourceSnapshot(
        ts=1.0,
        cpu_percent=0.0,
        rss_bytes=1024,
        log_size_bytes=0,
        metric_label="current_working_set",
    )
    assert snap_custom.metric_label == "current_working_set"

    with pytest.raises(ValueError, match="metric_label must be a non-empty string"):
        ResourceSnapshot(ts=1.0, cpu_percent=0.0, rss_bytes=1024, log_size_bytes=0, metric_label="")


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: Log size sample is non-zero once logger writes
# ---------------------------------------------------------------------------


def test_log_size_sample_is_nonzero_once_logger_writes(tmp_path: Path) -> None:
    """Acceptance: Monitored log file size is non-zero once the file logger writes to it,

    and zero when the file does not exist or is empty.
    """
    log_file = tmp_path / "app.log"
    sampler = make_default_resource_sampler(log_path=log_file)

    # 1. Before file exists -> 0
    snap0 = sampler.sample(now=1.0)
    assert snap0.log_size_bytes == 0

    # 2. File created but 0 bytes -> 0
    log_file.touch()
    snap1 = sampler.sample(now=2.0)
    assert snap1.log_size_bytes == 0

    # 3. Configure a real file logger writing to log_file
    logger = logging.getLogger("test_log_size_sampler")
    logger.setLevel(logging.INFO)
    handler = logging.FileHandler(log_file, mode="a", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logger.addHandler(handler)

    try:
        logger.info("First log record written by harness")
        handler.flush()

        # 4. Sample after logger wrote -> non-zero
        snap2 = sampler.sample(now=3.0)
        assert snap2.log_size_bytes > 0
        assert snap2.log_size_bytes == log_file.stat().st_size

        # Write more log data -> size increases
        logger.info("Second log record with payload data")
        handler.flush()
        snap3 = sampler.sample(now=4.0)
        assert snap3.log_size_bytes > snap2.log_size_bytes
    finally:
        logger.removeHandler(handler)
        handler.close()


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Metric definition asserted against documentation
# ---------------------------------------------------------------------------


def test_metric_definition_asserted_against_documentation() -> None:
    """Acceptance: The metric definition in code is asserted against docs/RESOURCE_METRICS.md."""
    doc_path = Path("docs/RESOURCE_METRICS.md")
    assert doc_path.exists(), "docs/RESOURCE_METRICS.md documentation file must exist"
    doc_text = doc_path.read_text(encoding="utf-8")

    # Verify key terminology and contract elements in the documentation
    assert "peak_rss" in doc_text
    assert "current_working_set" in doc_text
    assert "PeakWorkingSetSize" in doc_text
    assert "WorkingSetSize" in doc_text
    assert "ru_maxrss" in doc_text
    assert "Peak Resident Set Size" in doc_text
    assert "app.log" in doc_text
    assert "ratchet" in doc_text.lower() or "high-water mark" in doc_text.lower()

    # Assert code defaults match documentation
    default_sampler = make_default_resource_sampler()
    assert default_sampler.metric_label == "peak_rss"
    snap = default_sampler.sample(now=0.0)
    assert snap.metric_label == "peak_rss"


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: Frozen Phase 12 artifacts parse unchanged
# ---------------------------------------------------------------------------


def test_frozen_aggregate_v1_parses_unchanged() -> None:
    """Acceptance: The frozen runs/lab/aggregate-1h/aggregate_v1.json artifact parses unchanged."""
    aggregate_path = Path("runs/lab/aggregate-1h/aggregate_v1.json")
    assert aggregate_path.exists(), f"Frozen artifact not found: {aggregate_path}"

    content = aggregate_path.read_text(encoding="utf-8")
    data = json.loads(content)

    assert data["schema_version"] == 1
    assert data["session_count"] == 1
    assert "soak-1h-20260922" in data["session_ids"]
    assert "perception_agnostic" in data
    assert "internal_counters_not_research_findings" in data
    assert "non_claims" in data
    assert len(data["non_claims"]) == 4


def test_frozen_soak_report_parses_unchanged() -> None:
    """Acceptance: The frozen runs/lab/soak-1h/soak_report.json artifact parses unchanged."""
    soak_path = Path("runs/lab/soak-1h/soak_report.json")
    assert soak_path.exists(), f"Frozen artifact not found: {soak_path}"

    from wow_bot.analysis.aggregate import load_soak_report

    report = load_soak_report(soak_path)
    assert report.schema_version == 2
    assert report.session_id == "soak-1h-20260922"
    assert report.summary.sample_count == 3524
    assert report.summary.rss_peak_bytes > 0
    assert report.summary.duration_s > 0
