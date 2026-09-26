"""Combat edge detector (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/combat.py`` (author
Amirm227) and reshaped: the red-hue ratio on the four screen edges is
unchanged, but ``time.time()`` is gone — :meth:`CombatDetector.detect`
takes the clock as a parameter (defaulting to :func:`time.monotonic`) so
the cooldown latch is deterministic in tests.

This is a cheap channel and scans every frame by default
(``[combat].sampling_hz = inf``).
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

from wow_bot.perception import deps
from wow_bot.perception.capture import PER_FRAME_HZ, FrameLike, Throttle, normalize_bgr

__all__ = ["CombatDetector", "CombatReading"]

#: HSV gates for the two red bands (hue wraps at 180 in OpenCV).
_RED_LOW = ((0, 100, 80), (10, 255, 255))
_RED_HIGH = ((170, 100, 80), (180, 255, 255))


@dataclass(frozen=True)
class CombatReading:
    """One frame's combat verdict plus the measured red ratio."""

    in_combat: bool
    red_ratio: float


class CombatDetector:
    """Declares combat from a red-pixel ratio on the four screen edges.

    The cooldown latch keeps ``in_combat`` true for ``cooldown_s`` after
    the last red edge, so a momentary occlusion does not clear combat.
    """

    def __init__(
        self,
        edge_size: int = 150,
        red_ratio_thresh: float = 0.02,
        cooldown_s: float = 1.0,
        *,
        sampling_hz: float = PER_FRAME_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if edge_size <= 0:
            raise ValueError("edge_size must be > 0")
        if not 0.0 <= red_ratio_thresh <= 1.0:
            raise ValueError("red_ratio_thresh must be within [0, 1]")
        if cooldown_s < 0.0:
            raise ValueError("cooldown_s must be >= 0")
        self.edge_size = int(edge_size)
        self.red_ratio_thresh = float(red_ratio_thresh)
        self.cooldown_s = float(cooldown_s)
        self.last_combat_time: float | None = None
        self._clock = clock
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = CombatReading(in_combat=False, red_ratio=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> CombatReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    @staticmethod
    def _red_mask(bgr: FrameLike) -> FrameLike:
        cv2 = deps.require_cv2()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        low = cv2.inRange(hsv, _RED_LOW[0], _RED_LOW[1])
        high = cv2.inRange(hsv, _RED_HIGH[0], _RED_HIGH[1])
        combined: FrameLike = low | high
        return combined

    def red_ratio(self, frame: FrameLike) -> float:
        """Return the red-pixel ratio across the four edge strips."""
        normalized = normalize_bgr(frame)
        height, width = normalized.shape[:2]
        edge = min(self.edge_size, height, width)
        if edge <= 0:
            return 0.0
        strips = (
            normalized[0:edge, :, :],
            normalized[height - edge : height, :, :],
            normalized[:, 0:edge, :],
            normalized[:, width - edge : width, :],
        )
        total_red = 0
        total_px = 0
        for strip in strips:
            mask = self._red_mask(strip)
            total_red += int((mask > 0).sum())
            total_px += int(mask.size)
        if total_px <= 0:
            return 0.0
        return total_red / total_px

    def detect(self, frame: FrameLike, now: float | None = None) -> tuple[bool, float]:
        """Return ``(in_combat, red_ratio)`` for ``frame``.

        ``now`` is the clock reading for the cooldown latch. It defaults
        to the injected clock; tests pass a fixed value to stay
        deterministic.
        """
        if not self._throttle.allow(now):
            last = self._last
            return last.in_combat, last.red_ratio

        current = self._clock() if now is None else now
        ratio = self.red_ratio(frame)
        instant_combat = ratio > self.red_ratio_thresh

        if instant_combat:
            self.last_combat_time = current

        latch_active = (
            self.last_combat_time is not None
            and (current - self.last_combat_time) < self.cooldown_s
        )
        in_combat = instant_combat or latch_active

        self._last = CombatReading(in_combat=in_combat, red_ratio=ratio)
        self._throttle.record(current)
        return in_combat, ratio
