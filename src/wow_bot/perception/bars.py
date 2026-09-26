"""Health and mana bar reader (T-FIX-27).

Ported from ``hamberger@9b968a4`` ``perception/bars.py`` (author
Amirm227) and reshaped: no module-level ``pytesseract`` assignment, no
import-time ``cv2``, ROIs and the sampling budget injected from
:class:`~wow_bot.perception.perception_config.PerceptionConfig`.

This is a cheap channel and runs on every frame by default
(``[bars].sampling_hz = inf``); a finite budget only caches the last
reading, it never fabricates a value.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception.capture import PER_FRAME_HZ, FrameLike, Throttle, normalize_bgr

__all__ = ["BarReader", "BarReading", "fill_ratio"]

#: A column counts as filled when at least this fraction of its pixels
#: are strongly saturated and bright (the hamberger heuristic).
_COLUMN_FILL_FRACTION = 0.3

#: HSV gates for "coloured and lit" — an empty bar segment is dark grey.
_SATURATION_MIN = 80
_VALUE_MIN = 80


def _column_filled_mask(crop: FrameLike) -> tuple[FrameLike, int]:
    """Return the per-column fill mask and the crop width."""
    cv2 = deps.require_cv2()
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    colored = (saturation > _SATURATION_MIN) & (value > _VALUE_MIN)
    _, width = saturation.shape
    return colored.mean(axis=0) > _COLUMN_FILL_FRACTION, int(width)


def fill_ratio(frame: FrameLike, roi: tuple[int, int, int, int]) -> float:
    """Return the fill fraction in ``[0, 1]`` of the bar inside ``roi``.

    The bar is measured as the span from the leftmost to the rightmost
    filled column, matching the hamberger ``_ratio`` heuristic, then
    clamped so a mis-calibrated ROI can never return more than 1.0.
    """
    x, y, width, height = roi
    normalized = normalize_bgr(frame)
    crop = normalized[y : y + height, x : x + width]
    if crop.size == 0 or width <= 0:
        return 0.0
    filled, crop_width = _column_filled_mask(crop)
    if crop_width <= 0:
        return 0.0

    indexes = np.flatnonzero(filled)
    if indexes.size == 0:
        return 0.0

    span = int(indexes.max()) - int(indexes.min()) + 1
    return min(span / crop_width, 1.0)


@dataclass(frozen=True)
class BarReading:
    """One frame's bar ratios, each in ``[0, 1]``."""

    hp: float
    mana: float


class BarReader:
    """Reads the HP and mana bar ratios from an injected ``(x, y, w, h)`` ROI.

    Returns ratios, never percentages: ``GameState.hp_pct`` and
    ``mana_pct`` are fractions in ``[0, 1]``, so no unit conversion
    happens here and none is needed downstream.
    """

    def __init__(
        self,
        hp_roi: tuple[int, int, int, int],
        mana_roi: tuple[int, int, int, int],
        *,
        sampling_hz: float = PER_FRAME_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.hp_roi = hp_roi
        self.mana_roi = mana_roi
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = BarReading(hp=0.0, mana=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> BarReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def read_all(self, frame: FrameLike) -> BarReading:
        """Return both ratios for ``frame``, reusing the cache when throttled."""
        if self._throttle.allow():
            self._last = BarReading(
                hp=fill_ratio(frame, self.hp_roi),
                mana=fill_ratio(frame, self.mana_roi),
            )
            self._throttle.record()
        return self._last

    def read_hp(self, frame: FrameLike) -> float:
        """Return the HP fill ratio in ``[0, 1]``."""
        return self.read_all(frame).hp

    def read_mana(self, frame: FrameLike) -> float:
        """Return the mana fill ratio in ``[0, 1]``."""
        return self.read_all(frame).mana
