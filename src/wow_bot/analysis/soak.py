"""Long-Running 24-Hour Soak Test Analysis and Instrumentation Module (Task 8.3).

Provides operational resource sampling, log size measurement, linear memory trend
analysis, summary metric aggregation, and atomic JSON report generation for soak
testing without modifying application pipeline dynamics or executor behavior.

Public API:
    - :const:`SOAK_REPORT_SCHEMA_VERSION`
    - :class:`ResourceSnapshot`
    - :class:`SoakSample`
    - :class:`ResourceSampler`
    - :class:`ProcessResourceSampler`
    - :class:`FakeResourceSampler`
    - :class:`SoakObserver`
    - :func:`calculate_log_size`
    - :func:`resolve_log_path`
    - :func:`calculate_memory_slope`
    - :func:`build_soak_summary`
    - :func:`build_soak_report`
    - :func:`write_soak_report_atomically`
"""

from __future__ import annotations

import json
import math
import os
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final, Protocol

import psutil  # type: ignore[import-untyped]

from wow_bot.shared.interfaces import GameState, MetaState, Strategy
from wow_bot.shared.logger import get_logger

logger = get_logger("SOAK")

#: Soak Report Schema Version
SOAK_REPORT_SCHEMA_VERSION: Final[int] = 1


@dataclass(frozen=True)
class ResourceSnapshot:
    """Immutable resource measurement snapshot."""

    cpu_percent: float | None
    memory_rss_mb: float | None
    process_count: int | None = 1


@dataclass(frozen=True)
class SoakSample:
    """Immutable operational sample recorded during soak execution."""

    elapsed_seconds: float
    cpu_percent: float | None
    memory_rss_mb: float | None
    log_size_bytes: int | None
    progress_token: int | None
    watchdog_alive: bool | None
    shutdown_requested: bool
    fsm_state: str | None = None


class ResourceSampler(Protocol):
    """Protocol abstraction for operational resource sampling."""

    def sample(self) -> ResourceSnapshot:
        ...


class ProcessResourceSampler:
    """Cross-platform resource sampler measuring main process + child process tree using psutil."""

    def __init__(self, pid: int | None = None) -> None:
        target_pid = pid if pid is not None else os.getpid()
        try:
            self._proc = psutil.Process(target_pid)
            # Prime CPU measurement for main process
            self._proc.cpu_percent(interval=None)
            for child in self._proc.children(recursive=True):
                try:
                    child.cpu_percent(interval=None)
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except (psutil.NoSuchProcess, psutil.AccessDenied) as exc:
            logger.warning(f"Could not initialize ProcessResourceSampler for pid {target_pid}: {exc}")
            self._proc = None

    def sample(self) -> ResourceSnapshot:
        """Sample CPU percent and RSS memory MB for the process tree."""
        if self._proc is None:
            return ResourceSnapshot(cpu_percent=None, memory_rss_mb=None, process_count=0)

        try:
            if not self._proc.is_running():
                return ResourceSnapshot(cpu_percent=None, memory_rss_mb=None, process_count=0)

            total_rss_bytes = 0
            total_cpu_pct = 0.0
            proc_count = 0

            # Main process
            try:
                mem_info = self._proc.memory_info()
                total_rss_bytes += mem_info.rss
                cpu_val = self._proc.cpu_percent(interval=None)
                if math.isfinite(cpu_val):
                    total_cpu_pct += cpu_val
                proc_count += 1
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                return ResourceSnapshot(cpu_percent=None, memory_rss_mb=None, process_count=0)

            # Child processes
            try:
                children = self._proc.children(recursive=True)
                for child in children:
                    try:
                        c_mem = child.memory_info()
                        total_rss_bytes += c_mem.rss
                        c_cpu = child.cpu_percent(interval=None)
                        if math.isfinite(c_cpu):
                            total_cpu_pct += c_cpu
                        proc_count += 1
                    except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                        continue
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

            rss_mb = total_rss_bytes / (1024.0 * 1024.0)
            return ResourceSnapshot(
                cpu_percent=round(total_cpu_pct, 2),
                memory_rss_mb=round(rss_mb, 2),
                process_count=proc_count,
            )

        except Exception as exc:  # noqa: BLE001
            logger.error(f"Unexpected error during process resource sampling: {exc}")
            return ResourceSnapshot(cpu_percent=None, memory_rss_mb=None, process_count=None)


class FakeResourceSampler:
    """Testing resource sampler supplying deterministic or configured snapshots."""

    def __init__(self, snapshots: Sequence[ResourceSnapshot] | None = None) -> None:
        self._snapshots = list(snapshots) if snapshots is not None else []
        self._index = 0

    def sample(self) -> ResourceSnapshot:
        if not self._snapshots:
            return ResourceSnapshot(cpu_percent=0.0, memory_rss_mb=100.0, process_count=1)
        snapshot = self._snapshots[min(self._index, len(self._snapshots) - 1)]
        self._index += 1
        return snapshot


class SoakObserver:
    """Passive pipeline observer tracking progress tokens and FSM state for soak testing."""

    def __init__(self) -> None:
        self.progress_token: int = 0
        self.current_fsm_state: str = "IDLE"

    def on_game_state(self, game_state: GameState) -> None:
        pass

    def on_meta_state(self, meta_state: MetaState) -> None:
        pass

    def on_strategy_attempt(self, sim_ts: float) -> None:
        pass

    def on_strategy_accepted(self, strategy: Strategy, sim_ts: float) -> None:
        pass

    def on_strategy_fallback(self, sim_ts: float) -> None:
        pass

    def on_fsm_transition(self, sim_ts: float, from_state: str, to_state: str, reason: str) -> None:
        self.current_fsm_state = to_state

    def on_fsm_tick(self, sim_ts: float, fatigue: float) -> None:
        pass

    def on_idle_intent(self, sim_ts: float, behavior_name: str) -> None:
        pass

    def on_death_event(self, sim_ts: float) -> None:
        pass

    def on_progress_step(self) -> None:
        self.progress_token += 1


def calculate_log_size(path: Path | None) -> int | None:
    """Calculate recursive total size in bytes for a file or directory.

    Handles missing paths, disappeared files, and avoids following symlink loops.

    Args:
        path: Path to file or directory, or None.

    Returns:
        Total size in bytes, or None if unavailable.
    """
    if path is None:
        return None

    try:
        p = Path(path).resolve()
        if not p.exists():
            return None

        if p.is_file():
            return p.stat().st_size

        if p.is_dir():
            total_size = 0
            for root, _, files in os.walk(p, followlinks=False):
                for f in files:
                    fp = Path(root) / f
                    try:
                        if fp.is_file() and not fp.is_symlink():
                            total_size += fp.stat().st_size
                    except (FileNotFoundError, OSError):
                        continue
            return total_size

        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning(f"Error calculating log size for '{path}': {exc}")
        return None


def resolve_log_path(configured_path: str | Path | None = None) -> tuple[Path | None, str]:
    """Resolve a log file or directory path for measurement.

    Args:
        configured_path: Explicitly provided log path, or None.

    Returns:
        Tuple of (resolved Path or None, status string "resolved" or "unavailable").
    """
    if configured_path is not None:
        p = Path(configured_path)
        if p.exists() or p.parent.exists():
            return p, "resolved"
        return None, "unavailable"

    # Try default log path from config if available
    default_path = Path("logs/bot.log")
    if default_path.exists() or default_path.parent.exists():
        return default_path, "resolved"

    return None, "unavailable"


def calculate_memory_slope(samples: Sequence[SoakSample]) -> float | None:
    """Calculate least-squares linear growth slope of memory RSS MB per hour.

    Args:
        samples: Sequence of SoakSample.

    Returns:
        Slope in MB/hour, or None if fewer than 3 valid samples exist or variance is 0.
    """
    valid_pts: list[tuple[float, float]] = []
    for s in samples:
        if s.memory_rss_mb is not None and math.isfinite(s.memory_rss_mb) and s.elapsed_seconds >= 0:
            t_hours = s.elapsed_seconds / 3600.0
            valid_pts.append((t_hours, s.memory_rss_mb))

    if len(valid_pts) < 3:
        return None

    n = len(valid_pts)
    sum_t = sum(p[0] for p in valid_pts)
    sum_y = sum(p[1] for p in valid_pts)
    sum_tt = sum(p[0] ** 2 for p in valid_pts)
    sum_ty = sum(p[0] * p[1] for p in valid_pts)

    denominator = n * sum_tt - (sum_t ** 2)
    if abs(denominator) < 1e-9:
        return 0.0

    slope = (n * sum_ty - sum_t * sum_y) / denominator
    return float(slope)


def build_soak_summary(samples: Sequence[SoakSample]) -> dict[str, Any]:
    """Compute summary statistics for a sequence of soak operational samples.

    Args:
        samples: Sequence of SoakSample.

    Returns:
        Dictionary of summary statistics.
    """
    mem_vals = [(s.elapsed_seconds, s.memory_rss_mb) for s in samples if s.memory_rss_mb is not None]
    cpu_vals = [(s.elapsed_seconds, s.cpu_percent) for s in samples if s.cpu_percent is not None]
    log_vals = [s.log_size_bytes for s in samples if s.log_size_bytes is not None]
    prg_vals = [s.progress_token for s in samples if s.progress_token is not None]

    # Memory statistics
    initial_memory_mb: float | None = mem_vals[0][1] if mem_vals else None
    final_memory_mb: float | None = mem_vals[-1][1] if mem_vals else None
    min_memory_mb: float | None = min((v[1] for v in mem_vals), default=None)
    max_memory_mb: float | None = max((v[1] for v in mem_vals), default=None)
    avg_memory_mb: float | None = (
        sum(v[1] for v in mem_vals) / len(mem_vals) if mem_vals else None
    )

    memory_change_mb: float | None = (
        final_memory_mb - initial_memory_mb
        if final_memory_mb is not None and initial_memory_mb is not None
        else None
    )
    memory_growth_percent: float | None = (
        (memory_change_mb / initial_memory_mb * 100.0)
        if memory_change_mb is not None and initial_memory_mb is not None and initial_memory_mb > 0
        else None
    )

    memory_growth_slope = calculate_memory_slope(samples)

    # First and last quarter memory averages
    first_quarter_mean_memory_mb: float | None = None
    last_quarter_mean_memory_mb: float | None = None
    if len(mem_vals) >= 4:
        q_size = max(1, len(mem_vals) // 4)
        first_q = mem_vals[:q_size]
        last_q = mem_vals[-q_size:]
        first_quarter_mean_memory_mb = sum(v[1] for v in first_q) / len(first_q)
        last_quarter_mean_memory_mb = sum(v[1] for v in last_q) / len(last_q)

    # Peak memory diagnostic
    peak_memory_mb: float | None = None
    peak_memory_elapsed_seconds: float | None = None
    if mem_vals:
        peak_pair = max(mem_vals, key=lambda p: p[1])
        peak_memory_mb = peak_pair[1]
        peak_memory_elapsed_seconds = peak_pair[0]

    # CPU statistics
    avg_cpu_percent: float | None = (
        sum(v[1] for v in cpu_vals) / len(cpu_vals) if cpu_vals else None
    )
    max_cpu_percent: float | None = max((v[1] for v in cpu_vals), default=None)
    peak_cpu_elapsed_seconds: float | None = None
    if cpu_vals:
        peak_cpu_pair = max(cpu_vals, key=lambda p: p[1])
        peak_cpu_elapsed_seconds = peak_cpu_pair[0]

    # Log size statistics
    initial_log_size_bytes: int | None = log_vals[0] if log_vals else None
    final_log_size_bytes: int | None = log_vals[-1] if log_vals else None
    log_growth_bytes: int | None = (
        final_log_size_bytes - initial_log_size_bytes
        if final_log_size_bytes is not None and initial_log_size_bytes is not None
        else None
    )

    # Progress token statistics
    initial_progress_token: int | None = prg_vals[0] if prg_vals else None
    final_progress_token: int | None = prg_vals[-1] if prg_vals else None
    progress_delta: int | None = (
        final_progress_token - initial_progress_token
        if final_progress_token is not None and initial_progress_token is not None
        else None
    )

    return {
        "initial_memory_mb": round(initial_memory_mb, 2) if initial_memory_mb is not None else None,
        "final_memory_mb": round(final_memory_mb, 2) if final_memory_mb is not None else None,
        "minimum_memory_mb": round(min_memory_mb, 2) if min_memory_mb is not None else None,
        "maximum_memory_mb": round(max_memory_mb, 2) if max_memory_mb is not None else None,
        "average_memory_mb": round(avg_memory_mb, 2) if avg_memory_mb is not None else None,
        "memory_change_mb": round(memory_change_mb, 2) if memory_change_mb is not None else None,
        "memory_growth_percent": (
            round(memory_growth_percent, 2) if memory_growth_percent is not None else None
        ),
        "memory_growth_slope_mb_per_hour": (
            round(memory_growth_slope, 4) if memory_growth_slope is not None else None
        ),
        "first_quarter_mean_memory_mb": (
            round(first_quarter_mean_memory_mb, 2)
            if first_quarter_mean_memory_mb is not None
            else None
        ),
        "last_quarter_mean_memory_mb": (
            round(last_quarter_mean_memory_mb, 2)
            if last_quarter_mean_memory_mb is not None
            else None
        ),
        "peak_memory_mb": round(peak_memory_mb, 2) if peak_memory_mb is not None else None,
        "peak_memory_elapsed_seconds": (
            round(peak_memory_elapsed_seconds, 2)
            if peak_memory_elapsed_seconds is not None
            else None
        ),
        "average_cpu_percent": round(avg_cpu_percent, 2) if avg_cpu_percent is not None else None,
        "maximum_cpu_percent": round(max_cpu_percent, 2) if max_cpu_percent is not None else None,
        "peak_cpu_elapsed_seconds": (
            round(peak_cpu_elapsed_seconds, 2)
            if peak_cpu_elapsed_seconds is not None
            else None
        ),
        "initial_log_size_bytes": initial_log_size_bytes,
        "final_log_size_bytes": final_log_size_bytes,
        "log_growth_bytes": log_growth_bytes,
        "initial_progress_token": initial_progress_token,
        "final_progress_token": final_progress_token,
        "progress_delta": progress_delta,
    }


def build_soak_report(
    *,
    scenario: str,
    seed: int,
    requested_duration_seconds: float,
    sample_interval_seconds: float,
    completed_duration_seconds: float,
    samples: Sequence[SoakSample],
    completed_normally: bool,
    termination_reason: str,
    failure_type: str | None = None,
    failure_message: str | None = None,
    log_path: Path | None = None,
    log_size_status: str = "unavailable",
) -> dict[str, Any]:
    """Construct a complete soak test report dictionary conforming to schema version 1.

    Args:
        scenario: Scenario name.
        seed: Experiment seed integer.
        requested_duration_seconds: Target run duration in seconds.
        sample_interval_seconds: Metrics sampling interval in seconds.
        completed_duration_seconds: Actual elapsed duration in seconds.
        samples: Sequence of SoakSample objects.
        completed_normally: Whether the pipeline completed normally without unexpected error.
        termination_reason: Descriptive reason for termination.
        failure_type: Name of exception class if failed.
        failure_message: Short exception message if failed.
        log_path: Resolved log path or None.
        log_size_status: Status of log size measurement ("resolved" or "unavailable").

    Returns:
        Structured JSON-serializable report dictionary.
    """
    summary = build_soak_summary(samples)
    zero_crash = completed_normally and (
        completed_duration_seconds >= requested_duration_seconds - 1.0
    )

    sample_dicts = [asdict(s) for s in samples]

    return {
        "schema_version": SOAK_REPORT_SCHEMA_VERSION,
        "run": {
            "scenario": scenario,
            "seed": seed,
            "requested_duration_seconds": float(requested_duration_seconds),
            "sample_interval_seconds": float(sample_interval_seconds),
            "completed_duration_seconds": round(float(completed_duration_seconds), 2),
            "sample_count": len(samples),
        },
        "resource_scope": {
            "memory_metric": "rss_mb",
            "cpu_metric": "process_tree_cpu_percent",
            "process_scope": "main_plus_recursive_children",
            "log_path": str(log_path) if log_path is not None else None,
            "log_size_status": log_size_status,
        },
        "samples": sample_dicts,
        "summary": summary,
        "termination": {
            "completed_normally": completed_normally,
            "termination_reason": termination_reason,
            "failure_type": failure_type,
            "failure_message": failure_message,
        },
        "acceptance": {
            "zero_crash_target_met": zero_crash,
            "memory_stability_status": "manual_review_required",
        },
    }


def write_soak_report_atomically(report: dict[str, Any], destination: Path | str) -> None:
    """Write report dictionary atomically to destination JSON file using a temp file.

    Enforces allow_nan=False to ensure strict JSON compatibility.

    Args:
        report: Report dictionary.
        destination: Target destination file path.
    """
    dest = Path(destination).resolve()
    dest.parent.mkdir(parents=True, exist_ok=True)

    tmp_file = dest.with_suffix(".tmp")
    json_text = json.dumps(report, indent=2, allow_nan=False)

    try:
        tmp_file.write_text(json_text, encoding="utf-8")
        tmp_file.replace(dest)
    except Exception:
        if tmp_file.exists():
            try:
                tmp_file.unlink()
            except OSError:
                pass
        raise
