"""Position-based stuck detection signal source for the reflex loop."""

import collections
import math
from collections.abc import Callable
from dataclasses import dataclass

from wow_bot.reflex.signals import Signal


class StuckError(Exception):
    """Raised when stuck detection configuration or processing fails."""


@dataclass(frozen=True)
class StuckConfig:
    """Configuration parameters for position-based stuck detection.

    Attributes:
        window_s: Sliding time window in seconds over which positions are evaluated.
        threshold_units: Maximum spatial spread in distance units to consider stationary.
        sample_interval_s: Minimum interval in seconds between position samples.
        min_samples: Minimum number of samples required before evaluating stuck status.
    """

    window_s: float = 3.0
    threshold_units: float = 1.0
    sample_interval_s: float = 0.5
    min_samples: int = 4

    def __post_init__(self) -> None:
        if self.window_s <= 0:
            raise ValueError(f"window_s must be > 0, got {self.window_s}")
        if self.threshold_units <= 0:
            raise ValueError(f"threshold_units must be > 0, got {self.threshold_units}")
        if self.sample_interval_s <= 0:
            raise ValueError(
                f"sample_interval_s must be > 0, got {self.sample_interval_s}"
            )
        if self.sample_interval_s > self.window_s:
            raise ValueError(
                f"sample_interval_s ({self.sample_interval_s}) cannot exceed window_s ({self.window_s})"
            )
        if self.min_samples < 2:
            raise ValueError(f"min_samples must be >= 2, got {self.min_samples}")


class PositionStuckSignalSource:
    """Poll-based signal source detecting stationary player position condition."""

    NAME_STUCK = "position_stuck"
    NAME_CLEAR = "position_clear"

    def __init__(
        self,
        position_source: Callable[[], tuple[float, float]],
        *,
        config: StuckConfig | None = None,
        emit_initial: bool = False,
    ) -> None:
        self._position_source = position_source
        self._config = config if config is not None else StuckConfig()
        self._emit_initial = emit_initial

        self._deque: collections.deque[tuple[float, float, float]] = collections.deque()
        self._last_sample_time: float | None = None
        self._is_stuck = False
        self._last_spread: float | None = None

    def poll(self, now: float) -> list[Signal]:
        """Poll position source and detect edge-triggered stuck state transitions."""
        if (
            self._last_sample_time is not None
            and (now - self._last_sample_time) < self._config.sample_interval_s
        ):
            return []

        pos = self._position_source()
        self._last_sample_time = now
        x, y = float(pos[0]), float(pos[1])

        self._deque.append((now, x, y))

        cutoff = now - self._config.window_s
        while self._deque and self._deque[0][0] < cutoff:
            self._deque.popleft()

        if len(self._deque) < self._config.min_samples:
            return []

        _earliest_t, earliest_x, earliest_y = self._deque[0]
        max_dist = 0.0
        for _t, px, py in self._deque:
            dist = math.hypot(px - earliest_x, py - earliest_y)
            max_dist = max(max_dist, dist)

        spread = max_dist
        self._last_spread = spread

        stuck_now = spread <= self._config.threshold_units

        if stuck_now and not self._is_stuck:
            self._is_stuck = True
            return [
                Signal(
                    name=self.NAME_STUCK,
                    payload={
                        "spread": spread,
                        "window_s": self._config.window_s,
                        "samples": len(self._deque),
                    },
                    ts=now,
                )
            ]

        if not stuck_now and self._is_stuck:
            self._is_stuck = False
            return [
                Signal(
                    name=self.NAME_CLEAR,
                    payload={
                        "spread": spread,
                    },
                    ts=now,
                )
            ]

        return []

    def reset(self) -> None:
        """Clear all buffered position samples, stuck flag, and last sample time."""
        self._deque.clear()
        self._is_stuck = False
        self._last_sample_time = None
        self._last_spread = None

    @property
    def is_stuck(self) -> bool:
        """Return True if currently in a stuck condition."""
        return self._is_stuck

    @property
    def sample_count(self) -> int:
        """Return the current number of samples buffered in the window deque."""
        return len(self._deque)

    @property
    def last_spread(self) -> float | None:
        """Return the spatial spread calculated on the most recent evaluation, if any."""
        return self._last_spread
