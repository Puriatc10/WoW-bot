"""Telemetry observation layer for navigation performance and CPU metrics."""

from __future__ import annotations

import json
import math
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

if TYPE_CHECKING:
    from wow_bot.nav.navigator import Navigator, NavResult


class TelemetryError(Exception):
    """Exception raised for telemetry configuration or runtime errors."""


@dataclass(frozen=True)
class TelemetryConfig:
    """Configuration options for navigation telemetry observation."""

    sample_interval_s: float = 1.0
    max_samples: int = 10_000
    write_artifact: bool = False
    artifact_filename: str = "nav_telemetry.json"

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.sample_interval_s <= 0:
            raise ValueError(
                f"sample_interval_s must be > 0, got {self.sample_interval_s}"
            )
        if self.max_samples < 1:
            raise ValueError(f"max_samples must be >= 1, got {self.max_samples}")
        if (
            not isinstance(self.artifact_filename, str)
            or not self.artifact_filename
            or "/" in self.artifact_filename
            or "\\" in self.artifact_filename
            or ".." in self.artifact_filename
        ):
            raise ValueError(
                f"artifact_filename must be a non-empty string without path separators or '..', got {self.artifact_filename!r}"
            )


@dataclass(frozen=True)
class CpuSample:
    """CPU usage snapshot taken at a point in simulation/monotonic time."""

    ts: float
    cpu_percent: float


@dataclass(frozen=True)
class NavTelemetryReport:
    """Aggregated telemetry report for a navigation run."""

    session_id: str
    target_xy: tuple[float, float]
    status: str
    iterations: int
    replans: int
    path_attempts: int
    duration_s: float
    reason: str
    cpu_samples: tuple[CpuSample, ...]
    cpu_min_percent: float
    cpu_max_percent: float
    cpu_mean_percent: float

    def __post_init__(self) -> None:
        """Validate report invariants."""
        if not self.cpu_samples:
            if (
                self.cpu_min_percent != 0.0
                or self.cpu_mean_percent != 0.0
                or self.cpu_max_percent != 0.0
            ):
                raise ValueError(
                    "cpu_min, cpu_mean, and cpu_max must be 0.0 when cpu_samples is empty"
                )
        else:
            if not (
                self.cpu_min_percent <= self.cpu_mean_percent <= self.cpu_max_percent
            ):
                raise ValueError(
                    f"Expected cpu_min_percent <= cpu_mean_percent <= cpu_max_percent, "
                    f"got {self.cpu_min_percent}, {self.cpu_mean_percent}, {self.cpu_max_percent}"
                )

        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.replans < 0:
            raise ValueError(f"replans must be >= 0, got {self.replans}")
        if self.path_attempts < 0:
            raise ValueError(f"path_attempts must be >= 0, got {self.path_attempts}")
        if self.duration_s < 0.0:
            raise ValueError(f"duration_s must be >= 0.0, got {self.duration_s}")

    def to_json(self) -> dict[str, Any]:
        """Convert report into JSON-serializable dictionary representation."""
        return {
            "session_id": self.session_id,
            "target_xy": [self.target_xy[0], self.target_xy[1]],
            "status": self.status,
            "iterations": self.iterations,
            "replans": self.replans,
            "path_attempts": self.path_attempts,
            "duration_s": self.duration_s,
            "reason": self.reason,
            "cpu_samples": [
                {"ts": s.ts, "cpu_percent": s.cpu_percent} for s in self.cpu_samples
            ],
            "cpu_min_percent": self.cpu_min_percent,
            "cpu_max_percent": self.cpu_max_percent,
            "cpu_mean_percent": self.cpu_mean_percent,
        }


@runtime_checkable
class Clock(Protocol):
    """Protocol for injected time sources."""

    def now(self) -> float: ...


@runtime_checkable
class CpuSource(Protocol):
    """Protocol for process CPU usage sampling."""

    def cpu_percent(self) -> float: ...


class RealCpuSource:
    """Estimates process CPU percentage using process_time and monotonic clock deltas."""

    def __init__(self) -> None:
        self._last_sample: tuple[float, float] | None = None

    def cpu_percent(self) -> float:
        """Return process CPU percent since last sample, or 0.0 on baseline measurement."""
        pt_now = time.process_time()
        mt_now = time.monotonic()

        if self._last_sample is None:
            self._last_sample = (pt_now, mt_now)
            return 0.0

        pt_prev, mt_prev = self._last_sample
        self._last_sample = (pt_now, mt_now)

        mt_delta = mt_now - mt_prev
        if mt_delta <= 0.0:
            return 0.0

        pt_delta = pt_now - pt_prev
        cpus = os.cpu_count() or 1
        max_cpu = 100.0 * cpus
        raw_pct = 100.0 * (pt_delta / mt_delta)
        return max(0.0, min(max_cpu, raw_pct))


class TelemetryCollector:
    """In-memory buffer for collecting CPU usage samples during navigation."""

    def __init__(
        self,
        *,
        session_id: str,
        target_xy: tuple[float, float],
        clock: Clock,
        cpu_source: CpuSource,
        config: TelemetryConfig | None = None,
    ) -> None:
        if not isinstance(session_id, str) or not session_id.strip():
            raise TelemetryError("session_id must be a non-empty string")

        if (
            not isinstance(target_xy, (tuple, list))
            or len(target_xy) != 2
            or not math.isfinite(target_xy[0])
            or not math.isfinite(target_xy[1])
        ):
            raise TelemetryError("target_xy must be a 2-tuple of finite floats")

        self._session_id = session_id
        self._target_xy = (float(target_xy[0]), float(target_xy[1]))
        self._clock = clock
        self._cpu_source = cpu_source
        self._config = config if config is not None else TelemetryConfig()

        self._samples: list[CpuSample] = []
        self._last_sample_ts: float | None = None

    @property
    def sample_count(self) -> int:
        """Return the number of collected CPU samples."""
        return len(self._samples)

    def should_sample(self, now: float) -> bool:
        """Return True if time since last sample >= sample_interval_s or no sample yet taken."""
        if self._last_sample_ts is None:
            return True
        return (now - self._last_sample_ts) >= self._config.sample_interval_s

    def sample(self) -> CpuSample | None:
        """Take a CPU sample using the clock and CPU source if buffer is not full."""
        ts = self._clock.now()
        raw_cpu = self._cpu_source.cpu_percent()
        cpus = os.cpu_count() or 1
        max_cpu = 100.0 * cpus
        clamped_cpu = max(0.0, min(max_cpu, float(raw_cpu)))

        if len(self._samples) >= self._config.max_samples:
            return None

        sample_obj = CpuSample(ts=ts, cpu_percent=clamped_cpu)
        self._samples.append(sample_obj)
        self._last_sample_ts = ts
        return sample_obj

    def reset(self) -> None:
        """Clear collected samples and reset sampling timestamp."""
        self._samples.clear()
        self._last_sample_ts = None

    def build_report(self, result: NavResult) -> NavTelemetryReport:
        """Aggregate collected CPU samples and NavResult into a NavTelemetryReport."""
        if not self._samples:
            cpu_min = 0.0
            cpu_max = 0.0
            cpu_mean = 0.0
        else:
            cpu_vals = [s.cpu_percent for s in self._samples]
            cpu_min = float(min(cpu_vals))
            cpu_max = float(max(cpu_vals))
            cpu_mean = float(sum(cpu_vals) / len(cpu_vals))

        status_str = result.status.value if hasattr(result.status, "value") else str(result.status)

        return NavTelemetryReport(
            session_id=self._session_id,
            target_xy=self._target_xy,
            status=status_str,
            iterations=result.iterations,
            replans=result.replans,
            path_attempts=result.path_attempts,
            duration_s=result.duration_s,
            reason=result.reason,
            cpu_samples=tuple(self._samples),
            cpu_min_percent=cpu_min,
            cpu_max_percent=cpu_max,
            cpu_mean_percent=cpu_mean,
        )


def make_sampling_sleep(
    base_sleep: Callable[[float], None],
    collector: TelemetryCollector,
    clock: Clock,
) -> Callable[[float], None]:
    """Return a sleep wrapper that triggers telemetry sampling at configured intervals."""

    def _sampling_sleep(seconds: float) -> None:
        base_sleep(seconds)
        now = clock.now()
        if collector.should_sample(now):
            collector.sample()

    return _sampling_sleep


def run_navigation_observed(
    navigator: Navigator,
    target_xy: tuple[float, float],
    *,
    session: Any,
    clock: Clock,
    cpu_source: CpuSource,
    config: TelemetryConfig | None = None,
    sleep: Callable[[float], None] | None = None,
) -> tuple[NavResult, NavTelemetryReport]:
    """Execute observed navigation run with passive CPU telemetry sampling and reporting."""
    if not hasattr(navigator, "_clock") or not hasattr(navigator, "_sleep"):
        raise TelemetryError(
            "Navigator must expose '_clock' and '_sleep' attributes to support observed navigation"
        )

    if hasattr(session, "session_id"):
        sess_id_attr = session.session_id
        session_id = sess_id_attr() if callable(sess_id_attr) else str(sess_id_attr)
    else:
        raise TelemetryError("session object must have a 'session_id' attribute or method")

    cfg = config if config is not None else TelemetryConfig()
    collector = TelemetryCollector(
        session_id=session_id,
        target_xy=target_xy,
        clock=clock,
        cpu_source=cpu_source,
        config=cfg,
    )

    collector.sample()

    result = navigator.go_to(target_xy)

    report = collector.build_report(result)

    if cfg.write_artifact:
        if hasattr(session, "path"):
            session_path = Path(session.path)
        else:
            raise TelemetryError("session object must have a 'path' attribute for artifact writing")

        artifact_path = session_path / cfg.artifact_filename
        artifact_path.parent.mkdir(parents=True, exist_ok=True)
        with open(artifact_path, "w", encoding="utf-8") as f:
            json.dump(report.to_json(), f, ensure_ascii=False, indent=2, sort_keys=True)
            f.write("\n")

    evt = {"event": "nav_telemetry"}
    evt.update(report.to_json())
    session.write_event(evt)

    return result, report
