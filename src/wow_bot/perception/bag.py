"""Bag frame channel: slot grid -> ``inventory_count`` / ``inventory_max``
(T-FIX-31).

Implements the **Bag frame** row of ``docs/PERCEPTION.md`` §2.1.

**Source channel.** The bag window's slot grid, read only while the window
is open. This is an *opportunistic* observation, not a per-frame one: when
the window is absent the caller simply does not read this channel, and both
fields stay ``None`` (the consumer's existing absolute-threshold path).

**Extraction.** A per-slot template match against an injected empty-slot and
occupied-slot template. Two counter-layout properties fall out of the grid
geometry and are *cached for the session* rather than re-derived per frame:
``inventory_max`` is the slot count of the grid, and the slot rectangles are
computed once from the origin/size/gap constants.

**Confidence semantics.** The channel reports the mean of each slot's
winning template-match score. A slot whose best match does not clear
``min_confidence`` is *unknown*, and the whole count is then withheld
(``None``) rather than emitted partially — a partial ``inventory_count``
silently suppresses full-bag handling, which is precisely the decision the
consumer gates on it. ``inventory_max`` needs one high-confidence reading;
because it is geometry, a successful grid read yields it, and a failed one
yields ``None`` alongside the count.

**Rejected alternative.** Accumulating "You receive item" lines from the
Chat/events channel drifts, because chat lines scroll out of the OCR region
while it is occluded. A per-frame absolute count does not drift.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from wow_bot.perception import deps
from wow_bot.perception.capture import FrameLike, Throttle, normalize_bgr

__all__ = ["BagFrameReader", "BagReading", "slot_rectangles"]

#: Default minimum mean per-slot match score for a count to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.7


def slot_rectangles(
    origin: tuple[int, int],
    columns: int,
    rows: int,
    slot_size: tuple[int, int],
    gap: int,
) -> tuple[tuple[int, int, int, int], ...]:
    """Return the ``(x, y, w, h)`` rectangle of every grid slot, row-major.

    ``origin`` is the **interior top-left corner of the first slot**, so a
    slot rectangle is exactly its interior; a title bar or a frame border is
    deliberately not part of the geometry and must be calibrated away.
    """
    x0, y0 = origin
    width, height = slot_size
    pitch_x = width + gap
    pitch_y = height + gap
    return tuple(
        (x0 + column * pitch_x, y0 + row * pitch_y, width, height)
        for row in range(rows)
        for column in range(columns)
    )


@dataclass(frozen=True)
class BagReading:
    """One sample's bag observation.

    ``inventory_count`` is the occupied-slot count or ``None`` when it was
    withheld; ``inventory_max`` is the grid capacity or ``None`` when the
    grid could not be read; ``confidence`` is the mean per-slot winning
    match score, reported even when the count is withheld so "seen but
    uncertain" stays distinguishable from "not attempted" (ADR-002
    Decision 4). ``slots`` is the number of slots that were actually read.
    """

    inventory_count: int | None
    inventory_max: int | None
    confidence: float
    slots: int = 0


class BagFrameReader:
    """Template-matches each bag-grid slot against empty/occupied templates."""

    def __init__(
        self,
        origin: tuple[int, int],
        columns: int,
        rows: int,
        slot_size: tuple[int, int],
        *,
        empty_slot_template: str | Path,
        occupied_slot_template: str | Path,
        gap: int = 2,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
        sampling_hz: float = 1.0,
        base_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if columns <= 0 or rows <= 0:
            raise ValueError("columns and rows must be > 0")
        if gap < 0:
            raise ValueError("gap must be >= 0")
        if slot_size[0] <= 0 or slot_size[1] <= 0:
            raise ValueError("slot_size must be positive")
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")

        self.origin = (int(origin[0]), int(origin[1]))
        self.columns = int(columns)
        self.rows = int(rows)
        self.slot_size = (int(slot_size[0]), int(slot_size[1]))
        self.gap = int(gap)
        self.min_confidence = float(min_confidence)

        # The capacity is a layout property: it is stable for a session and
        # is cached rather than re-derived on every frame.
        self.inventory_max = self.columns * self.rows
        self.slots = slot_rectangles(
            self.origin, self.columns, self.rows, self.slot_size, self.gap
        )

        self._empty_template = self._load_template(empty_slot_template, base_dir)
        self._occupied_template = self._load_template(occupied_slot_template, base_dir)
        for kind, template in (
            ("empty_slot_template", self._empty_template),
            ("occupied_slot_template", self._occupied_template),
        ):
            # The slot rectangles come from config while the templates come
            # from disk; a mismatch between the two is a wiring error, and
            # catching it here keeps it from surfacing as a silent 0.0 score.
            if (
                int(template.shape[0]) != self.slot_size[1]
                or int(template.shape[1]) != self.slot_size[0]
            ):
                raise ValueError(
                    f"[bag].slot_size {self.slot_size} does not match the "
                    f"{kind} template size "
                    f"{(int(template.shape[1]), int(template.shape[0]))}"
                )
        self._empty_gray = self._to_gray(self._empty_template)
        self._occupied_gray = self._to_gray(self._occupied_template)

        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = BagReading(
            inventory_count=None, inventory_max=None, confidence=0.0
        )

    @staticmethod
    def _resolve(path: str | Path, base_dir: Path | None) -> Path:
        candidate = Path(path)
        if base_dir is not None and not candidate.is_absolute():
            candidate = base_dir / candidate
        return candidate

    @classmethod
    def _load_template(cls, path: str | Path, base_dir: Path | None) -> FrameLike:
        cv2 = deps.require_cv2()
        resolved = cls._resolve(path, base_dir)
        template = cv2.imread(str(resolved))
        if template is None:
            raise deps.PerceptionDependencyError(
                f"Bag slot template not found or unreadable: {resolved}. "
                "Point [bag].empty_slot_template / [bag].occupied_slot_template "
                "at a readable PNG; MOCK_MODE never loads templates."
            )
        return np.asarray(template)

    @staticmethod
    def _to_gray(image: FrameLike) -> FrameLike:
        cv2 = deps.require_cv2()
        gray: FrameLike = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        return gray

    def _match_template(self, crop: FrameLike) -> tuple[str, float]:
        """Return ``(kind, score)`` for one slot: the best of the two templates.

        ``kind`` is ``"empty"`` or ``"occupied"``. The score is the normalised
        cross-correlation of the winning template, which is a measured value
        even when it falls below the confidence floor. Construction already
        proved both templates fit inside a slot, so every template is
        eligible for the match.
        """
        cv2 = deps.require_cv2()
        gray = self._to_gray(normalize_bgr(crop))
        best_kind = "empty"
        best_score = -1.0
        for kind, template in (
            ("empty", self._empty_gray),
            ("occupied", self._occupied_gray),
        ):
            result = cv2.matchTemplate(gray, template, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, _ = cv2.minMaxLoc(result)
            score = float(max_val)
            if score > best_score:
                best_kind = kind
                best_score = score
        return best_kind, best_score

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> BagReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def read(self, frame: FrameLike, *, now: float | None = None) -> BagReading:
        """Read the bag grid, returning the cached sample while throttled.

        An empty ROI (the bag window is closed) is not a failure: it means
        "not attempted", so both fields are ``None`` with zero confidence.
        """
        if not self._throttle.allow(now):
            return self._last

        normalized = normalize_bgr(frame)
        height, width = normalized.shape[:2]
        occupied = 0
        scored = 0
        score_total = 0.0
        resolvable = True
        for x, y, slot_w, slot_h in self.slots:
            if x + slot_w > width or y + slot_h > height:
                # The grid runs off the frame: this read cannot answer the
                # question, so it does not answer it partially.
                resolvable = False
                break
            crop = normalized[y : y + slot_h, x : x + slot_w]
            if crop.size == 0:
                resolvable = False
                break
            kind, score = self._match_template(crop)
            scored += 1
            score_total += score
            if score < self.min_confidence:
                # One uncertain slot makes the whole count uncertain: a
                # partial count silently suppresses full-bag handling.
                resolvable = False
                break
            if kind == "occupied":
                occupied += 1

        confidence = score_total / scored if scored else 0.0
        if resolvable and scored == len(self.slots):
            self._last = BagReading(
                inventory_count=occupied,
                inventory_max=self.inventory_max,
                confidence=confidence,
                slots=scored,
            )
        else:
            self._last = BagReading(
                inventory_count=None,
                inventory_max=None,
                confidence=confidence,
                slots=scored,
            )
        self._throttle.record(now)
        return self._last
