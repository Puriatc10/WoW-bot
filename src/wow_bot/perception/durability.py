"""Character frame channel: per-slot durability -> ``durability_fraction``
(T-FIX-31).

Implements the **Character frame** row of ``docs/PERCEPTION.md`` §2.1.

**Source channel.** The character sheet's per-slot durability readouts, read
only while the sheet is open — another opportunistic observation.

**Extraction.** OCR of each configured slot's percentage text (the Tesseract
word-confidence API of :mod:`wow_bot.perception._ocr`), then a mean over the
slots that read confidently. The slot regions are configured explicitly
rather than derived from a grid, because the character sheet's slot positions
are not a uniform grid; each entry is an ``(x, y, w, h)`` region.

**Emitted scale.** ``[0, 1]``: each readout is an integer percentage, divided
by 100 in exactly one place (:func:`parse_percent_text`).

**Confidence semantics.** The mean is weighted by per-slot OCR confidence,
and the field is emitted only when a *quorum* of the configured slots read
above ``min_score``. Below quorum the field is ``None``, which routes to the
existing safe path: the vendor consumer treats ``None`` as
``"durability_unknown"`` and skips repairing. A partial durability is never
emitted, because a wrong "everything is fine" answer silently suppresses
repairing.
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass

from wow_bot.perception import deps
from wow_bot.perception._ocr import ocr_words, threshold_variants, upscale_gray
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["DurabilityReader", "DurabilityReading", "parse_percent_text"]

#: Default minimum per-slot OCR confidence for a slot to join the mean.
_DEFAULT_MIN_SCORE = 0.5

#: Default share of configured slots that must read above ``min_score``.
_DEFAULT_QUORUM_FRACTION = 0.5

#: Default minimum confidence-weighted mean for the field to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.6

#: Upscale factor before OCR — durability text is small.
_UPSCALE = 6

#: Unsigned integer percentage, with an optional ``%`` suffix. The leading
#: ``(?<![\d-])`` rejects a value glued to a preceding digit or sign, so an
#: OCR ``-5`` or ``1000%`` is not silently read as ``5`` / ``100``.
_PERCENT_TOKEN = re.compile(r"(?<![\d-])(\d{1,3})\s*%?")


def parse_percent_text(text: str) -> int | None:
    """Return the percentage integer in ``text``, or ``None``.

    The value must be a whole number of percent in ``0..100``: a readout
    outside that range is an OCR error (``1000%``, ``-5``, ``7oo``), not a
    durability, so it is rejected rather than clamped into a plausible value.
    A number with a further digit glued to it is rejected too — ``1000%`` must
    not be read as ``100``.
    """
    if not text:
        return None
    match = _PERCENT_TOKEN.search(text)
    if match is None:
        return None
    if match.end() < len(text) and text[match.end()].isdigit():
        return None
    value = int(match.group(1))
    if not 0 <= value <= 100:
        return None
    return value


@dataclass(frozen=True)
class DurabilityReading:
    """One sample's durability observation.

    ``durability_fraction`` is the confidence-weighted mean in ``[0, 1]`` or
    ``None`` when the quorum was not met. ``confidence`` is that mean's
    weight and is reported even when the fraction is withheld; ``read_slots``
    and ``total_slots`` expose the quorum arithmetic so a consumer can see
    how much of the sheet was legible.
    """

    durability_fraction: float | None
    confidence: float
    read_slots: int = 0
    total_slots: int = 0

    @property
    def found(self) -> bool:
        """True when a ``durability_fraction`` was emitted."""
        return self.durability_fraction is not None


class DurabilityReader:
    """OCRs the configured character-sheet slots and means their durability."""

    def __init__(
        self,
        slot_rois: Sequence[tuple[int, int, int, int]],
        *,
        min_score: float = _DEFAULT_MIN_SCORE,
        quorum_fraction: float = _DEFAULT_QUORUM_FRACTION,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        sampling_hz: float = 1.0,
        upscale: int = _UPSCALE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not slot_rois:
            raise ValueError("slot_rois must not be empty")
        if not 0.0 <= min_score <= 1.0:
            raise ValueError(f"min_score must be within [0, 1], got {min_score}")
        if not 0.0 <= quorum_fraction <= 1.0:
            raise ValueError(
                f"quorum_fraction must be within [0, 1], got {quorum_fraction}"
            )
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        if upscale <= 0:
            raise ValueError(f"upscale must be > 0, got {upscale}")
        self.slot_rois = tuple(slot_rois)
        self.min_score = float(min_score)
        self.quorum_fraction = float(quorum_fraction)
        self.min_confidence = float(min_confidence)
        self.upscale = int(upscale)
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._tesseract_configured = False
        self._last = DurabilityReading(
            durability_fraction=None,
            confidence=0.0,
            total_slots=len(self.slot_rois),
        )

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> DurabilityReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _read_slot(self, frame: FrameLike, roi: tuple[int, int, int, int]) -> tuple[int | None, float]:
        """OCR one slot; return ``(percent, measured_confidence)``."""
        pytesseract = deps.require_pytesseract()
        x, y, width, height = roi
        crop = frame[y : y + height, x : x + width]
        if crop.size == 0:
            return None, 0.0

        gray = upscale_gray(crop, self.upscale)
        best_percent: int | None = None
        best_confidence = 0.0
        for image in threshold_variants(gray):
            data = pytesseract.image_to_data(image, config="--psm 7", output_type="dict")
            text, confidence = ocr_words(data)
            percent = parse_percent_text(text)
            if percent is not None and confidence >= best_confidence:
                best_percent = percent
                best_confidence = confidence
        return best_percent, best_confidence

    def read(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> DurabilityReading:
        """Read the configured slots, returning the cache while throttled."""
        if not self._throttle.allow(now):
            return self._last
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        normalized = normalize_bgr(frame)
        weighted_total = 0.0
        weight_total = 0.0
        read_slots = 0
        for roi in self.slot_rois:
            percent, confidence = self._read_slot(normalized, roi)
            if percent is None or confidence < self.min_score:
                continue
            weighted_total += (percent / 100.0) * confidence
            weight_total += confidence
            read_slots += 1

        # ``quorum_fraction = 0.0`` is a configured "zero slots is enough",
        # so the guard falls back to the weighted mean of nothing, which is
        # explicitly ``0.0`` rather than a division by zero.
        confidence = weight_total / read_slots if read_slots else 0.0
        quorum = self.quorum_fraction * len(self.slot_rois)
        fraction: float | None = None
        if read_slots >= quorum and confidence >= self.min_confidence:
            fraction = weighted_total / weight_total if weight_total > 0.0 else 0.0

        self._last = DurabilityReading(
            durability_fraction=fraction,
            confidence=confidence,
            read_slots=read_slots,
            total_slots=len(self.slot_rois),
        )
        self._throttle.record(now)
        return self._last
