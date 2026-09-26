"""Enemy cast bar channel: cast bar -> ``incoming_casts`` (T-FIX-31).

Implements the **Enemy cast bar** row of ``docs/PERCEPTION.md`` §2.1.

**Source channel.** The cast bar that appears on the target frame (and above
enemy nameplates) while a mob is casting.

**Extraction.** OCR for the spell name, a template-free border-colour test for
the interruptibility indicator, and the bar's fill fraction for the remaining
cast time. ``spell_id`` is the OCR'd name mapped through the configured
name→id table; ``remaining_cast_time_s`` is the fill fraction times the
configured longest cast duration. A name that is absent from the table drops
the entry, because ``spell_id`` is required by
:class:`~wow_bot.shared.interfaces.IncomingCast` and a guessed id would make
the reactive layer interrupt the wrong spell.

**Bounded to one cast.** This reader observes **one** bar — the target
frame's — so it emits either a one-entry tuple or the empty tuple. It does
not claim to enumerate every concurrently-cast spell; multi-plate
enumeration is deliberately out of scope here.

**Interruptibility is dropped, not defaulted.** The border strip renders the
non-interruptible state in this client (``border_interruptible = false``,
configurable because the convention is client-dependent). When the strip is
absent or shows no coloured pixels, the whole cast is dropped: the interrupt
is gated on ``is_interruptible`` directly, so a fabricated boolean is not an
acceptable substitute.

**An empty tuple is honest.** With no confident cast, ``()`` means "no casts
observed", which is a legitimate observation, not a fabricated absence
(ADR-002 Decision 4's measured-versus-not-attempted distinction).
"""

from __future__ import annotations

import re
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception._ocr import ocr_words, threshold_variants, upscale_gray
from wow_bot.perception.bars import fill_ratio
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr
from wow_bot.shared.interfaces import IncomingCast

__all__ = ["CastingReader", "CastingReading", "normalise_spell_name"]

#: Default minimum OCR confidence for a cast to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.6

#: Default longest cast duration for the fill-fraction conversion.
_DEFAULT_MAX_REMAINING_S = 30.0

#: Default upscale factor before OCR.
_UPSCALE = 6

#: Letters, digits, and single spaces only — the shape of a spell name.
_SPELL_TEXT = re.compile(r"[A-Za-z][A-Za-z' ]*")


def normalise_spell_name(text: str) -> str:
    """Normalise an OCR spell name for the id-table lookup.

    Lower-cased, whitespace-collapsed, punctuation stripped: OCR routinely
    returns ``"Shadow  Bolt."`` for ``"Shadow Bolt"``, and a table lookup
    that is defeated by punctuation would drop real casts for no reason.
    """
    letters = _SPELL_TEXT.search(text or "")
    if letters is None:
        return ""
    words = letters.group().split()
    return " ".join(words).lower()


@dataclass(frozen=True)
class CastingReading:
    """One sample's enemy-cast observation.

    ``casts`` is a one-entry tuple when a cast was observed and the empty
    tuple otherwise. ``confidence`` is the measured OCR confidence and is
    reported even when the cast was withheld; the per-bar detail
    (``spell_name``, ``bar_fill``, ``interruptible``) is exposed so a
    consumer can see which part of the reading was uncertain.
    """

    casts: tuple[IncomingCast, ...] = ()
    confidence: float = 0.0
    spell_name: str | None = None
    bar_fill: float | None = None
    interruptible: bool | None = None


class CastingReader:
    """Reads the target-frame cast bar into at most one ``IncomingCast``."""

    def __init__(
        self,
        cast_roi: tuple[int, int, int, int],
        *,
        spell_ids: Mapping[str, str],
        border_roi: tuple[int, int, int, int] | None = None,
        border_interruptible: bool = False,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        max_remaining_s: float = _DEFAULT_MAX_REMAINING_S,
        sampling_hz: float = 2.0,
        upscale: int = _UPSCALE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        if max_remaining_s <= 0.0:
            raise ValueError(f"max_remaining_s must be > 0, got {max_remaining_s}")
        if upscale <= 0:
            raise ValueError(f"upscale must be > 0, got {upscale}")
        self.cast_roi = cast_roi
        self.border_roi = border_roi
        self.border_interruptible = bool(border_interruptible)
        self.min_confidence = float(min_confidence)
        self.max_remaining_s = float(max_remaining_s)
        self.upscale = int(upscale)
        # The table is keyed by the same normalised form the reader looks up.
        self.spell_ids: dict[str, str] = {
            normalise_spell_name(name): str(value) for name, value in spell_ids.items()
        }
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._tesseract_configured = False
        self._last = CastingReading()

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> CastingReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _read_spell_name(self, frame: FrameLike) -> tuple[str | None, float]:
        """OCR the cast bar's spell name; return ``(name, measured confidence)``."""
        pytesseract = deps.require_pytesseract()
        x, y, width, height = self.cast_roi
        crop = normalize_bgr(frame)[y : y + height, x : x + width]
        if crop.size == 0:
            return None, 0.0

        big = upscale_gray(crop, self.upscale)
        best_name: str | None = None
        best_confidence = 0.0
        for image in threshold_variants(big):
            data = pytesseract.image_to_data(image, config="--psm 7", output_type="dict")
            # A spell name's tokens carry no digits, so the numeric-only rule
            # would report 0.0 confidence for every readable name.
            text, confidence = ocr_words(data, numeric_only=False)
            name = normalise_spell_name(text)
            if name and name in self.spell_ids and confidence >= best_confidence:
                best_name = name
                best_confidence = confidence
        return best_name, best_confidence

    @staticmethod
    def _blue_mask(bgr: FrameLike) -> FrameLike:
        cv2 = deps.require_cv2()
        hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
        mask: FrameLike = cv2.inRange(hsv, (95, 80, 80), (135, 255, 255))
        return mask

    def _read_interruptibility(self, frame: FrameLike) -> bool | None:
        """Return whether the cast is interruptible, or ``None`` when unknown.

        The client renders the interruptibility state on the bar's border. A
        strip with no coloured pixels is "not observed", not "not
        interruptible": the interrupt decision is gated on this boolean
        directly, so an absent indicator drops the cast instead of defaulting.
        """
        if self.border_roi is None:
            return None
        x, y, width, height = self.border_roi
        crop = normalize_bgr(frame)[y : y + height, x : x + width]
        if crop.size == 0:
            return None
        blue = int(np.count_nonzero(self._blue_mask(crop) > 0))
        if blue == 0:
            return None
        return bool(self.border_interruptible)

    def read(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        caster_entity_id: str = "target",
        tesseract_cmd: str | None = None,
    ) -> CastingReading:
        """Read the cast bar, returning the cached sample while throttled.

        ``caster_entity_id`` names the entity the single observed bar belongs
        to; it is required by :class:`IncomingCast` and is supplied by the
        caller (the target the state builder already resolved), never
        invented here.
        """
        if not self._throttle.allow(now):
            return self._last
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        name, confidence = self._read_spell_name(frame)
        fill = fill_ratio(frame, self.cast_roi)
        interruptible = self._read_interruptibility(frame)

        casts: tuple[IncomingCast, ...] = ()
        if name is not None and confidence >= self.min_confidence and interruptible is not None:
            casts = (
                IncomingCast(
                    caster_entity_id=caster_entity_id,
                    spell_id=self.spell_ids[name],
                    remaining_cast_time_s=float(fill) * self.max_remaining_s,
                    is_interruptible=interruptible,
                ),
            )

        self._last = CastingReading(
            casts=casts,
            confidence=confidence,
            spell_name=name,
            bar_fill=float(fill),
            interruptible=interruptible,
        )
        self._throttle.record(now)
        return self._last
