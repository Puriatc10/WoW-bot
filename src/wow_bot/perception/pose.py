"""World-pose reader: OCR of an on-screen addon coordinate frame (T-FIX-30).

Implements decision **A** of ``docs/lab_phase/HAMBERGER_PORT_PLAN.md``
finding F-1, recorded in
``docs/decisions/ADR-003-world-pose-channel.md``: the player's world
position is read from a coordinate addon (TomTom/MapCoords style) that
renders it as a short text line at a fixed screen position.

Why not the minimap? :class:`wow_bot.perception.minimap.MinimapTracker`
returns the matched arrow's centre *inside the minimap ROI* — screen pixels,
not world coordinates — and in the live client that arrow is pinned to the
minimap centre, so the match encodes facing rather than position. Feeding
those pixels into ``GameState.position`` would rebuild the world graph around
a ~200-unit screen rectangle (plan doc finding F-1, option D, rejected).

**Units.** The parsed pair is in world coordinate units, the unit the addon
frame prints and the unit ``world/sync.py`` consumes. No conversion happens
here, so there is exactly one place where a unit could be confused and it is
the addon's own output.

**Honest absence.** ``GameState.position`` is non-Optional, but this reader
still reports ``None`` when it cannot read a confident pose: a wrong pose
writes a wrong node into the world model permanently, so a guess is not an
acceptable substitute. Callers raise rather than fabricate (see
:mod:`wow_bot.perception.observations`).

**Height is not observed.** The frame prints two coordinates; the reader does
not invent a third. ``player_z`` stays ``None`` (ADR-003 Decision 2).

**Locale limitation, recorded rather than hidden.** ``.`` is the decimal
separator and ``,``/whitespace/``;`` are the separators between the two
numbers. A locale that prints ``57,3 42,1`` yields four number tokens and the
parse fails to ``None`` rather than guessing which comma is which.
"""

from __future__ import annotations

import math
import re
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = [
    "PoseReading",
    "WorldPoseReader",
    "parse_coordinate_text",
]

#: Upscale factor before OCR — the coordinate text is small.
_DEFAULT_UPSCALE = 4

#: Default minimum OCR confidence for a pose to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.6

#: One decimal number, dot as the decimal separator.
_NUMBER_TOKEN = re.compile(r"[-+]?\d+(?:\.\d+)?")


def parse_coordinate_text(text: str) -> tuple[float, float] | None:
    """Parse an addon coordinate line into ``(x, y)``, or ``None``.

    The rules are deliberately narrow, because a misread coordinate is a
    wrong world node rather than an imprecise one:

    * exactly **two** decimal numbers must be present — zero, one, or three
      or more means the line was not understood;
    * the two numbers must be separated by at least one character other than
      a dot, which rejects a stray decimal point turning one number into two
      (``57.342.1``);
    * both values must be finite.

    ``.`` is the decimal separator. A comma-decimal line such as
    ``"57,3 42,1"`` contains four numbers and is rejected rather than
    guessed at (ADR-003 Decision 1).
    """
    if not text:
        return None
    matches = list(_NUMBER_TOKEN.finditer(text))
    if len(matches) != 2:
        return None
    gap = text[matches[0].end() : matches[1].start()]
    if not any(character not in "." for character in gap):
        return None
    try:
        x = float(matches[0].group())
        y = float(matches[1].group())
    except ValueError:  # pragma: no cover - the regex cannot emit this
        return None
    if not (math.isfinite(x) and math.isfinite(y)):
        return None
    return (x, y)


def _words_and_confidence(data: Mapping[str, Sequence[Any]]) -> tuple[str, float]:
    """Join OCR word tokens and average the confidences of digit tokens.

    ``pytesseract.image_to_data`` reports a confidence in ``[0, 100]`` per
    word, and ``-1`` for rows that are not words. Only tokens carrying a
    digit contribute, because those are the tokens the parse consumes; a
    confidence averaged over punctuation noise would be a different quantity.
    No confidence is invented when no digit token exists: the result is
    ``0.0``, which is "not measured", not "certainly wrong".
    """
    texts = data.get("text", ())
    confidences = data.get("conf", ())
    words: list[str] = []
    scores: list[float] = []
    for index, raw_text in enumerate(texts):
        token = str(raw_text).strip()
        if not token:
            continue
        words.append(token)
        if not any(character.isdigit() for character in token):
            continue
        raw_confidence = confidences[index] if index < len(confidences) else -1
        try:
            value = float(raw_confidence)
        except (TypeError, ValueError):
            value = -1.0
        if value >= 0.0:
            scores.append(min(value, 100.0) / 100.0)
    confidence = sum(scores) / len(scores) if scores else 0.0
    return " ".join(words), confidence


@dataclass(frozen=True)
class PoseReading:
    """One sample's world-pose observation.

    ``position`` is world units, never screen pixels. ``confidence`` is the
    measured OCR confidence of the coordinate tokens, and is reported even
    when ``position`` is ``None``, so "seen but uncertain" is distinguishable
    from "not attempted" (ADR-002 Decision 4). ``sampled_at`` is the
    monotonic time of the sample this reading came from; while the reader is
    throttled it re-serves the last sample unchanged, so ``sampled_at`` is
    how a consumer detects staleness (T-FIX-32 owns that gate).
    """

    position: tuple[float, float] | None
    confidence: float
    sampled_at: float | None = None


class WorldPoseReader:
    """OCRs an addon coordinate frame and reports a world ``(x, y)``."""

    def __init__(
        self,
        coordinate_roi: tuple[int, int, int, int],
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        sampling_hz: float = 1.0,
        upscale: int = _DEFAULT_UPSCALE,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        if upscale <= 0:
            raise ValueError(f"upscale must be > 0, got {upscale}")
        self.roi = coordinate_roi
        self.min_confidence = float(min_confidence)
        self.upscale = int(upscale)
        self._clock = clock
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._tesseract_configured = False
        self._last = PoseReading(position=None, confidence=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> PoseReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def _configure_tesseract(self, tesseract_cmd: str) -> None:
        if self._tesseract_configured:
            return
        pytesseract = deps.require_pytesseract()
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self._tesseract_configured = True

    def _sample_time(self, now: float | None) -> float:
        return float(self._clock()) if now is None else float(now)

    def _ocr_candidates(self, crop: FrameLike) -> list[tuple[str, float]]:
        """OCR the crop under three thresholdings; return ``(text, conf)``.

        The same three-way thresholding T-FIX-27's target reader uses
        (Otsu, fixed, inverted) is applied because the addon frame may render
        light-on-dark or dark-on-light depending on the operator's UI. The
        word-confidence API is used instead of a lexical heuristic so the
        reported confidence is measured, not inferred.
        """
        cv2 = deps.require_cv2()
        pytesseract = deps.require_pytesseract()
        big = cv2.resize(
            crop,
            (0, 0),
            fx=self.upscale,
            fy=self.upscale,
            interpolation=cv2.INTER_CUBIC,
        )
        gray = cv2.cvtColor(big, cv2.COLOR_BGR2GRAY)
        _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        _, fixed = cv2.threshold(gray, 130, 255, cv2.THRESH_BINARY)
        inverted = cv2.bitwise_not(otsu)

        candidates: list[tuple[str, float]] = []
        for image in (otsu, fixed, inverted):
            data = pytesseract.image_to_data(image, config="--psm 7", output_type="dict")
            candidates.append(_words_and_confidence(data))
        return candidates

    def read(
        self,
        frame: FrameLike,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> PoseReading:
        """Read the pose, returning the cached sample while throttled.

        A throttled frame re-serves the previous sample with its original
        ``sampled_at``; only a real sample updates the timestamp. A stale
        sample is never upgraded to ``None`` here, because the consumer, not
        the reader, owns the freshness policy.
        """
        if not self._throttle.allow(now):
            return self._last
        if tesseract_cmd is not None:
            self._configure_tesseract(tesseract_cmd)

        normalized = normalize_bgr(frame)
        x, y, width, height = self.roi
        crop = normalized[y : y + height, x : x + width]
        if crop.size == 0:
            self._last = PoseReading(
                position=None,
                confidence=0.0,
                sampled_at=self._sample_time(now),
            )
            self._throttle.record(now)
            return self._last

        best_position: tuple[float, float] | None = None
        best_confidence = 0.0
        for text, confidence in self._ocr_candidates(crop):
            parsed = parse_coordinate_text(text)
            if parsed is not None and confidence >= best_confidence:
                best_position = parsed
                best_confidence = confidence

        if best_position is not None and best_confidence < self.min_confidence:
            # Seen but uncertain: the measured confidence is kept, the value
            # is withheld (ADR-002 Decision 4).
            best_position = None

        self._last = PoseReading(
            position=best_position,
            confidence=best_confidence,
            sampled_at=self._sample_time(now),
        )
        self._throttle.record(now)
        return self._last
