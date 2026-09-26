"""XP bar channel: bar fill + level label -> ``level_or_xp`` (T-FIX-31).

Implements the **XP bar** row of ``docs/PERCEPTION.md`` §2.1.

**Source channel.** The experience bar — a thin fill bar at the bottom edge
of the viewport — plus its level label.

**Extraction.** The fill fraction reuses
:func:`wow_bot.perception.bars.fill_ratio`, the same colour-segmentation
method the Health/Mana row uses, so there is one bar-fill implementation in
the package rather than two. The integer level is OCR'd from the label
region with the Tesseract word-confidence API
(:mod:`wow_bot.perception._ocr`).

**Emitted value.** ``level + xp_fraction`` — a single monotone scalar, so
the runner's level-then-xp read and the strategist's ``level_or_xp`` field
agree on one number. The integer part is the level and the fractional part
is the progress toward the next one.

**Confidence semantics.** The measured level-OCR score. Below
``min_confidence`` the field is ``None``, never a guessed level, because a
constant level hides XP progress. The bar half is colour segmentation and
carries no score to measure, so it is not given a fabricated ``1.0``
(T-FIX-28's confidence-boundedness precedent); a bar that fills its ROI edge
to edge is treated structurally as an unreadable, occluded measurement and
withholds the value as well. When ``level_roi`` is configured as absent the
OCR half cannot be measured at all, and the field is ``None`` rather than a
level-less fraction.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable
from dataclasses import dataclass

from wow_bot.perception import deps
from wow_bot.perception._ocr import ocr_words, threshold_variants, upscale_gray
from wow_bot.perception.bars import fill_ratio
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["XPBarReader", "XPBarReading", "parse_level_text"]

#: Default minimum confidence for ``level_or_xp`` to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.5

#: Upscale factor before OCR — the level label is small text.
_LEVEL_UPSCALE = 6

#: The first integer in an OCR line. The label is "12" but OCR may return
#: noise around it, so the search is for the number rather than the whole
#: string.
_INTEGER_TOKEN = re.compile(r"\d+")


def parse_level_text(text: str) -> int | None:
    """Return the first integer in ``text``, or ``None`` when there is none.

    The rule is deliberately narrow: exactly one parseable use of the text is
    needed, and an OCR line with no digits cannot name a level. A negative or
    fractional reading is not possible here, because the token pattern
    matches unsigned digits only.
    """
    if not text:
        return None
    match = _INTEGER_TOKEN.search(text)
    if match is None:
        return None
    return int(match.group())


@dataclass(frozen=True)
class XPBarReading:
    """One sample's experience-bar observation.

    ``level_or_xp`` is the monotone ``level + xp_fraction`` scalar or ``None``
    when it was withheld; ``level`` and ``xp_fraction`` are the two measured
    inputs, exposed so a consumer can tell which half was uncertain.
    ``confidence`` is the measured gate value and is reported even when the
    scalar is withheld (ADR-002 Decision 4).
    """

    level_or_xp: float | None
    level: int | None
    xp_fraction: float | None
    confidence: float

    @property
    def found(self) -> bool:
        """True when a ``level_or_xp`` value was emitted."""
        return self.level_or_xp is not None


class XPBarReader:
    """Reads the XP fill fraction and OCRs the level label."""

    def __init__(
        self,
        bar_roi: tuple[int, int, int, int],
        *,
        level_roi: tuple[int, int, int, int] | None = None,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        sampling_hz: float = 1.0,
        upscale: int = _LEVEL_UPSCALE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        if upscale <= 0:
            raise ValueError(f"upscale must be > 0, got {upscale}")
        self.bar_roi = bar_roi
        self.level_roi = level_roi
        self.min_confidence = float(min_confidence)
        self.upscale = int(upscale)
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._tesseract_configured = False
        self._last = XPBarReading(
            level_or_xp=None, level=None, xp_fraction=None, confidence=0.0
        )

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> XPBarReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _read_level(self, frame: FrameLike) -> tuple[int | None, float]:
        """OCR the level label; return ``(level, measured_confidence)``."""
        pytesseract = deps.require_pytesseract()
        roi = self.level_roi
        if roi is None:
            return None, 0.0
        x, y, width, height = roi
        normalized = normalize_bgr(frame)
        crop = normalized[y : y + height, x : x + width]
        if crop.size == 0:
            return None, 0.0

        gray = upscale_gray(crop, self.upscale)
        best_level: int | None = None
        best_confidence = 0.0
        for image in threshold_variants(gray):
            data = pytesseract.image_to_data(image, config="--psm 7", output_type="dict")
            # The level label is a number, but OCR often renders it with a
            # "Lv"/"/" neighbour; averaging every token measures the whole
            # label rather than only its digits.
            text, confidence = ocr_words(data, numeric_only=False)
            level = parse_level_text(text)
            if level is not None and confidence >= best_confidence:
                best_level = level
                best_confidence = confidence
        return best_level, best_confidence

    def read(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> XPBarReading:
        """Read the XP bar, returning the cached sample while throttled.

        ``confidence`` is the measured level-OCR score. The bar half has no
        confidence model to measure — it is colour segmentation, exactly like
        the Health/Mana row — so it is not given a fabricated ``1.0``
        (T-FIX-28's confidence-boundedness precedent): a bar that fills its
        ROI **edge to edge** is instead treated as an unreadable,
        fully-occluded measurement and withholds the value structurally.
        """
        if not self._throttle.allow(now):
            return self._last
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        ratio = fill_ratio(frame, self.bar_roi)
        level, confidence = self._read_level(frame)
        bar_readable = ratio < 1.0

        level_or_xp: float | None = None
        if level is not None and bar_readable and confidence >= self.min_confidence:
            level_or_xp = float(level) + float(ratio)

        self._last = XPBarReading(
            level_or_xp=level_or_xp,
            level=level,
            xp_fraction=ratio,
            confidence=confidence,
        )
        self._throttle.record(now)
        return self._last
