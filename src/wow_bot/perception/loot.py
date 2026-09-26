"""Lootable-corpse channel: loot sparkle -> ``target_is_lootable`` (T-FIX-31).

Implements the **Lootable-corpse indicator** row of ``docs/PERCEPTION.md``
§2.1, described there as the weakest channel in the document.

**Source channel.** The loot sparkle on a corpse in the game world.

**Extraction.** Colour segmentation for the sparkle class inside
``[loot].sparkle_roi``: a pixel is *sparkle* when it is bright and either its
red channel leads its blue by a wide margin (gold, the lootable cue) or its
blue channel leads its red (white-blue, the same sparkle at a different
phase). :func:`classify_loot_sparkle` is the whole classifier and is a pure
function of a crop, so the colour rule is unit testable without a frame
corpus.

**The channel does not emit today, and that is the honest answer.** §2.1
requires a high threshold to assert ``True`` and states there is no confident
negative; §6 records the precision/recall target as *not measurable* because
the frozen frame corpus does not exist in the repository. Asserting the
field's value before that measurement exists would be guessing at exactly the
decision the loot consumer gates on. :class:`LootSparkleReader` therefore
performs the full measurement, reports it, and withholds
``target_is_lootable`` — the gate is the module constant
:data:`LOOT_CHANNEL_MEASURED`, which may only be set to ``True`` once a
corpus exists and the numbers are recorded in §6. This is a recorded
uncertainty, not an assumption.

**Never ``False``.** The loot consumer skips looting on a not-lootable
answer, so a fabricated ``False`` walks away from loot that was there. The
field is ``None`` on every path that is not a measured, confident positive.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = [
    "LOOT_CHANNEL_MEASURED",
    "LootSparkleReader",
    "LootSparkleReading",
    "classify_loot_sparkle",
]

#: Whether this channel may emit ``target_is_lootable`` at all.
#:
#: ``False`` until §6's precision/recall target is measured against the
#: frozen frame corpus. It is a module constant, not a config key, precisely
#: so that enabling it is a deliberate, reviewable code change rather than a
#: config toggle that silently turns an unvalidated guess into a bot action.
LOOT_CHANNEL_MEASURED: bool = False

#: Default minimum sparkle-coloured pixels before a positive is considered.
_DEFAULT_MIN_PIXELS = 40

#: Default minimum share of the bright pixels the sparkle class must hold.
_DEFAULT_DOMINANCE_THRESH = 0.6

#: Default minimum sparkle dominance for the field to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.9

#: Brightness floor: sparkle pixels are the bright, near-white pixels.
_BRIGHTNESS_MIN = 200

#: Warm sparkle (gold): red clearly dominant over blue. The floor is
#: deliberately high, because a saturated orange item icon is not a sparkle.
_WARM_RGB_DELTA = 20

#: Cool sparkle (white-blue): blue clearly ahead of red, which is what
#: separates the sparkle from a neutral white highlight or UI text.
_COOL_RGB_DELTA = 20


def classify_loot_sparkle(crop: FrameLike) -> tuple[bool, float, int]:
    """Classify a BGR crop into ``(positive, dominance, sparkle_pixels)``.

    ``dominance`` is the share of the *bright* pixels that belong to the
    sparkle class, so a ROI whose bright pixels are mostly ordinary UI text
    scores near zero; a ROI that is only sparkle scores near one. When there
    are no bright pixels at all the dominance is ``0.0`` — "not measured" —
    never a fabricated certainty.
    """
    pixels = np.asarray(crop)
    if pixels.ndim != 3 or pixels.size == 0:
        return False, 0.0, 0
    b = pixels[:, :, 0].astype(np.int16)
    g = pixels[:, :, 1].astype(np.int16)
    r = pixels[:, :, 2].astype(np.int16)

    bright = (r >= _BRIGHTNESS_MIN) & (g >= _BRIGHTNESS_MIN) & (b >= _BRIGHTNESS_MIN)
    # Red may *equal* green in a fully saturated gold, so the warm test is
    # ``r >= g`` rather than a strict dominance.
    warm = bright & ((r - b) >= _WARM_RGB_DELTA) & ((r - g) >= 0)
    cool = bright & ((b - r) >= _COOL_RGB_DELTA)
    sparkle = warm | cool
    bright_count = int(np.count_nonzero(bright))
    sparkle_count = int(np.count_nonzero(sparkle))
    if bright_count == 0:
        return False, 0.0, 0
    dominance = sparkle_count / bright_count
    return sparkle_count > 0, dominance, sparkle_count


@dataclass(frozen=True)
class LootSparkleReading:
    """One sample's loot-sparkle observation.

    ``target_is_lootable`` is ``None`` whenever the channel may not assert a
    value (see :data:`LOOT_CHANNEL_MEASURED`) or when the measurement was not
    confident. ``confidence`` and ``sparkle_pixels`` are the *measured*
    values and are reported even when the field is withheld, so the
    measurement is observable and testable while the assertion stays off.
    """

    target_is_lootable: bool | None
    confidence: float
    sparkle_pixels: int = 0


class LootSparkleReader:
    """Measures the loot sparkle; asserts ``target_is_lootable`` only when measured."""

    def __init__(
        self,
        sparkle_roi: tuple[int, int, int, int],
        *,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        dominance_thresh: float = _DEFAULT_DOMINANCE_THRESH,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        sampling_hz: float = 1.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_pixels <= 0:
            raise ValueError(f"min_pixels must be > 0, got {min_pixels}")
        if not 0.0 <= dominance_thresh <= 1.0:
            raise ValueError(
                f"dominance_thresh must be within [0, 1], got {dominance_thresh}"
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        self.roi = sparkle_roi
        self.min_pixels = int(min_pixels)
        self.dominance_thresh = float(dominance_thresh)
        self.min_confidence = float(min_confidence)
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = LootSparkleReading(target_is_lootable=None, confidence=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> LootSparkleReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def read(self, frame: FrameLike, *, now: float | None = None) -> LootSparkleReading:
        """Measure the sparkle, returning the cached sample while throttled.

        A measurement that clears every threshold still yields ``None`` while
        :data:`LOOT_CHANNEL_MEASURED` is ``False``; the measured confidence
        and pixel count are reported either way.
        """
        if not self._throttle.allow(now):
            return self._last

        x, y, width, height = self.roi
        crop = normalize_bgr(frame)[y : y + height, x : x + width]
        if crop.size == 0:
            self._last = LootSparkleReading(target_is_lootable=None, confidence=0.0)
            self._throttle.record(now)
            return self._last

        positive, dominance, sparkle_pixels = classify_loot_sparkle(crop)
        confident = (
            positive
            and sparkle_pixels >= self.min_pixels
            and dominance >= self.dominance_thresh
            and dominance >= self.min_confidence
        )
        target_is_lootable: bool | None = None
        if confident and LOOT_CHANNEL_MEASURED:
            target_is_lootable = True

        self._last = LootSparkleReading(
            target_is_lootable=target_is_lootable,
            confidence=dominance,
            sparkle_pixels=sparkle_pixels,
        )
        self._throttle.record(now)
        return self._last
