"""Progress metrics collector and sliding window snapshot engine for the behavioral watchdog.

Pure, deterministic, and isolated from OS and I/O.
"""

import math
from collections import deque
from dataclasses import dataclass
from typing import Any


class MetricsError(Exception):
    """Raised when progress metrics or sample updates violate constraints."""


@dataclass(frozen=True)
class MetricsConfig:
    """Configuration parameters for progress window tracking.

    Invariants:
      * window_s > 0.0
      * sample_interval_s > 0.0
      * window_s >= sample_interval_s
      * min_samples_for_rate >= 2
      * All floats must be finite
    """

    window_s: float = 60.0
    sample_interval_s: float = 1.0
    min_samples_for_rate: int = 5

    def __post_init__(self) -> None:
        if not math.isfinite(self.window_s):
            raise ValueError("window_s must be a finite float")
        if not math.isfinite(self.sample_interval_s):
            raise ValueError("sample_interval_s must be a finite float")
        if self.window_s <= 0.0:
            raise ValueError(f"window_s must be > 0.0, got {self.window_s}")
        if self.sample_interval_s <= 0.0:
            raise ValueError(
                f"sample_interval_s must be > 0.0, got {self.sample_interval_s}"
            )
        if self.window_s < self.sample_interval_s:
            raise ValueError(
                f"window_s ({self.window_s}) must be >= sample_interval_s ({self.sample_interval_s})"
            )
        if (
            not isinstance(self.min_samples_for_rate, int)
            or isinstance(self.min_samples_for_rate, bool)
            or self.min_samples_for_rate < 2
        ):
            raise ValueError(
                f"min_samples_for_rate must be an integer >= 2, got {self.min_samples_for_rate}"
            )


@dataclass(frozen=True)
class ProgressSample:
    """A single progress measurement sample.

    Invariants:
      * ts is finite, >= 0.0.
      * position has exactly two finite floats.
      * inventory_count >= 0.
      * level_or_xp >= 0.0.
      * successful_actions_total >= 0.
      * reflex_ticks_total >= 0.
    """

    ts: float
    position: tuple[float, float]
    inventory_count: int
    level_or_xp: float
    successful_actions_total: int
    reflex_ticks_total: int

    def __post_init__(self) -> None:
        if not math.isfinite(self.ts) or self.ts < 0.0:
            raise ValueError(f"ts must be finite and >= 0.0, got {self.ts}")
        if not (
            isinstance(self.position, tuple)
            and len(self.position) == 2
            and math.isfinite(self.position[0])
            and math.isfinite(self.position[1])
        ):
            raise ValueError(
                f"position must be a tuple of two finite floats, got {self.position}"
            )
        if (
            not isinstance(self.inventory_count, int)
            or isinstance(self.inventory_count, bool)
            or self.inventory_count < 0
        ):
            raise ValueError(
                f"inventory_count must be an integer >= 0, got {self.inventory_count}"
            )
        if not math.isfinite(self.level_or_xp) or self.level_or_xp < 0.0:
            raise ValueError(
                f"level_or_xp must be finite and >= 0.0, got {self.level_or_xp}"
            )
        if (
            not isinstance(self.successful_actions_total, int)
            or isinstance(self.successful_actions_total, bool)
            or self.successful_actions_total < 0
        ):
            raise ValueError(
                f"successful_actions_total must be an integer >= 0, got {self.successful_actions_total}"
            )
        if (
            not isinstance(self.reflex_ticks_total, int)
            or isinstance(self.reflex_ticks_total, bool)
            or self.reflex_ticks_total < 0
        ):
            raise ValueError(
                f"reflex_ticks_total must be an integer >= 0, got {self.reflex_ticks_total}"
            )


@dataclass(frozen=True)
class ProgressSnapshot:
    """A snapshot summary of progress metrics over the sliding window."""

    window_s: float
    sample_count: int
    window_start_ts: float | None
    window_end_ts: float | None
    position_delta: float
    inventory_delta: int
    level_or_xp_delta: float
    successful_actions_per_minute: float
    reflex_tick_rate_hz: float
    is_rate_reliable: bool

    def __post_init__(self) -> None:
        if self.sample_count == 0:
            if self.window_start_ts is not None or self.window_end_ts is not None:
                raise ValueError(
                    "window_start_ts and window_end_ts must be None when sample_count == 0"
                )
        else:
            if self.window_start_ts is None or self.window_end_ts is None:
                raise ValueError(
                    "window_start_ts and window_end_ts must not be None when sample_count > 0"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "window_s": float(self.window_s),
            "sample_count": int(self.sample_count),
            "window_start_ts": (
                float(self.window_start_ts)
                if self.window_start_ts is not None
                else None
            ),
            "window_end_ts": (
                float(self.window_end_ts) if self.window_end_ts is not None else None
            ),
            "position_delta": float(self.position_delta),
            "inventory_delta": int(self.inventory_delta),
            "level_or_xp_delta": float(self.level_or_xp_delta),
            "successful_actions_per_minute": float(self.successful_actions_per_minute),
            "reflex_tick_rate_hz": float(self.reflex_tick_rate_hz),
            "is_rate_reliable": bool(self.is_rate_reliable),
        }


class ProgressTracker:
    """Sliding-window observation buffer for tracking progress metrics.

    Single-thread ownership note:
    This class is NOT thread-safe. It must be used from a single sampling thread.
    """

    def __init__(self, *, config: MetricsConfig | None = None) -> None:
        self._config: MetricsConfig = config if config is not None else MetricsConfig()
        capacity: int = (
            math.ceil(self._config.window_s / self._config.sample_interval_s) + 1
        )
        self._buffer: deque[ProgressSample] = deque(maxlen=capacity)

    def update(self, sample: ProgressSample) -> None:
        """Append a new progress sample to the sliding window.

        Enforces strict timestamp monotonicity and counter non-decreasing invariants.
        Prunes samples older than now - window_s.
        """
        if self._buffer:
            last = self._buffer[-1]
            if sample.ts <= last.ts:
                raise MetricsError(
                    f"Sample timestamp {sample.ts} is not strictly greater than previous timestamp {last.ts}"
                )
            if sample.successful_actions_total < last.successful_actions_total:
                raise MetricsError(
                    f"successful_actions_total decreased from {last.successful_actions_total} to {sample.successful_actions_total}"
                )
            if sample.reflex_ticks_total < last.reflex_ticks_total:
                raise MetricsError(
                    f"reflex_ticks_total decreased from {last.reflex_ticks_total} to {sample.reflex_ticks_total}"
                )
            if sample.level_or_xp < last.level_or_xp:
                raise MetricsError(
                    f"level_or_xp decreased from {last.level_or_xp} to {sample.level_or_xp}"
                )

        self._buffer.append(sample)

        now = sample.ts
        cutoff = now - self._config.window_s
        while self._buffer and self._buffer[0].ts < cutoff:
            self._buffer.popleft()

    def snapshot(self) -> ProgressSnapshot:
        """Computes all progress metrics over the current window without mutating state."""
        count = len(self._buffer)
        if count == 0:
            return ProgressSnapshot(
                window_s=self._config.window_s,
                sample_count=0,
                window_start_ts=None,
                window_end_ts=None,
                position_delta=0.0,
                inventory_delta=0,
                level_or_xp_delta=0.0,
                successful_actions_per_minute=0.0,
                reflex_tick_rate_hz=0.0,
                is_rate_reliable=False,
            )

        if count == 1:
            s = self._buffer[0]
            return ProgressSnapshot(
                window_s=self._config.window_s,
                sample_count=1,
                window_start_ts=s.ts,
                window_end_ts=s.ts,
                position_delta=0.0,
                inventory_delta=0,
                level_or_xp_delta=0.0,
                successful_actions_per_minute=0.0,
                reflex_tick_rate_hz=0.0,
                is_rate_reliable=False,
            )

        first = self._buffer[0]
        last = self._buffer[-1]

        pos_dist = 0.0
        for i in range(1, count):
            p1 = self._buffer[i - 1].position
            p2 = self._buffer[i].position
            pos_dist += math.hypot(p2[0] - p1[0], p2[1] - p1[1])

        raw_inv_delta = last.inventory_count - first.inventory_count
        inv_delta = max(0, raw_inv_delta)

        raw_xp_delta = last.level_or_xp - first.level_or_xp
        xp_delta = max(0.0, raw_xp_delta)

        dt = last.ts - first.ts
        is_reliable = count >= self._config.min_samples_for_rate and dt > 0.0

        if is_reliable and dt > 0.0:
            actions_delta = max(
                0, last.successful_actions_total - first.successful_actions_total
            )
            ticks_delta = max(0, last.reflex_ticks_total - first.reflex_ticks_total)
            actions_per_min = (actions_delta / dt) * 60.0
            ticks_hz = ticks_delta / dt
        else:
            actions_per_min = 0.0
            ticks_hz = 0.0

        return ProgressSnapshot(
            window_s=self._config.window_s,
            sample_count=count,
            window_start_ts=first.ts,
            window_end_ts=last.ts,
            position_delta=pos_dist,
            inventory_delta=inv_delta,
            level_or_xp_delta=xp_delta,
            successful_actions_per_minute=actions_per_min,
            reflex_tick_rate_hz=ticks_hz,
            is_rate_reliable=is_reliable,
        )

    def reset(self) -> None:
        """Clears the ring buffer. Idempotent."""
        self._buffer.clear()

    @property
    def sample_count(self) -> int:
        return len(self._buffer)

    @property
    def window_start_ts(self) -> float | None:
        return self._buffer[0].ts if self._buffer else None

    @property
    def window_end_ts(self) -> float | None:
        return self._buffer[-1].ts if self._buffer else None
