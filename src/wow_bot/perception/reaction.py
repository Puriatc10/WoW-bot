"""Target-reaction channel: nameplate text colour -> reaction (T-FIX-30).

``TargetInfo.reaction`` has no direct sensor, and ``TargetInfo.__post_init__``
rejects an empty label, so without this channel a target cannot be built at
all. The client already renders the reaction in the nameplate text colour —
red for hostile, yellow for neutral, green for friendly — so a colour
classifier over ``[reaction].nameplate_roi`` reads a value the client
rendered, in the same way the bar readers read rendered bars.

**Honest absence.** Reaction is emitted only when enough coloured pixels are
present and one class clearly dominates. Below either floor the field is
``None``, and the state builder then omits the target instead of defaulting,
because a fabricated reaction silently re-gates targeting and combat.

**Accuracy.** Medium to high when the text is legible; it fails low when the
ROI holds no nameplate (few coloured pixels), or when the frame is heavily
tinted so two classes are close (dominance below threshold). Both failures
produce ``None``, never a wrong label. Precision and recall are recorded as
still unmeasurable in ``docs/PERCEPTION.md`` §6, because the frozen frame
corpus does not exist in the repository.

The colour predicates below are this channel's own, and are deliberately not
shared with :mod:`wow_bot.perception.target`: that module classifies
HP-bar *fill* pixels (green fill against a golden border), a different
question from classifying a *text* label whose three classes must be
separated from each other and from white.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from wow_bot.perception.capture import PER_FRAME_HZ, FrameLike, Throttle, normalize_bgr
from wow_bot.shared.interfaces import REACTIONS

__all__ = [
    "TargetReactionReader",
    "TargetReactionReading",
    "classify_reaction",
]

#: Minimum coloured pixels inside the ROI before a label is attempted.
_DEFAULT_MIN_PIXELS = 12

#: Minimum share of the coloured pixels the winning class must hold.
_DEFAULT_DOMINANCE_THRESH = 0.6


def classify_reaction(crop: FrameLike) -> tuple[str | None, float, int]:
    """Classify a BGR crop into ``(reaction, confidence, coloured_pixels)``.

    ``reaction`` is the dominant label or ``None``; ``confidence`` is the
    dominant share of the coloured pixels (``0.0`` when none are coloured);
    ``coloured_pixels`` is the count the caller gates on. Neutral is decided
    first so that a gold nameplate pixel is not also read as hostile by the
    broader red predicate.
    """
    pixels = np.asarray(crop)
    if pixels.ndim != 3 or pixels.size == 0:
        return None, 0.0, 0
    b = pixels[:, :, 0].astype(np.int16)
    g = pixels[:, :, 1].astype(np.int16)
    r = pixels[:, :, 2].astype(np.int16)

    neutral = (r >= 150) & (g >= 120) & (b <= 110) & (np.abs(r - g) <= 60)
    hostile = (r >= 120) & ((r - np.maximum(g, b)) >= 50) & ~neutral
    friendly = (g >= 90) & ((g - np.maximum(r, b)) >= 20) & ~neutral & ~hostile

    counts = {
        "hostile": int(np.count_nonzero(hostile)),
        "neutral": int(np.count_nonzero(neutral)),
        "friendly": int(np.count_nonzero(friendly)),
    }
    coloured = sum(counts.values())
    if coloured == 0:
        return None, 0.0, 0
    winner = max(counts, key=lambda label: counts[label])
    confidence = counts[winner] / coloured
    return winner, confidence, coloured


@dataclass(frozen=True)
class TargetReactionReading:
    """One frame's reaction observation.

    ``confidence`` is reported even when ``reaction`` is ``None``, so "seen
    but uncertain" stays distinguishable from "not attempted"; that is the
    ADR-002 Decision 4 convention.
    """

    reaction: str | None
    confidence: float
    coloured_pixels: int = 0


class TargetReactionReader:
    """Classifies the nameplate text colour inside a configured ROI."""

    def __init__(
        self,
        nameplate_roi: tuple[int, int, int, int],
        *,
        min_pixels: int = _DEFAULT_MIN_PIXELS,
        dominance_thresh: float = _DEFAULT_DOMINANCE_THRESH,
        sampling_hz: float = PER_FRAME_HZ,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if min_pixels <= 0:
            raise ValueError(f"min_pixels must be > 0, got {min_pixels}")
        if not 0.0 <= dominance_thresh <= 1.0:
            raise ValueError(
                f"dominance_thresh must be within [0, 1], got {dominance_thresh}"
            )
        self.roi = nameplate_roi
        self.min_pixels = int(min_pixels)
        self.dominance_thresh = float(dominance_thresh)
        self._throttle = Throttle(sampling_hz, clock=clock)
        self._last = TargetReactionReading(reaction=None, confidence=0.0)

    @property
    def sampling_hz(self) -> float:
        """Configured sampling budget in Hz."""
        return self._throttle.hz

    @property
    def last_reading(self) -> TargetReactionReading:
        """The most recent reading (also the cached value while throttled)."""
        return self._last

    def read(self, frame: FrameLike, *, now: float | None = None) -> TargetReactionReading:
        """Read the reaction, returning the cached reading while throttled."""
        if not self._throttle.allow(now):
            return self._last

        normalized = normalize_bgr(frame)
        x, y, width, height = self.roi
        crop = normalized[y : y + height, x : x + width]
        if crop.size == 0:
            self._last = TargetReactionReading(reaction=None, confidence=0.0)
            self._throttle.record(now)
            return self._last

        winner, confidence, coloured = classify_reaction(crop)
        reaction: str | None = None
        if (
            winner is not None
            and winner in REACTIONS
            and coloured >= self.min_pixels
            and confidence >= self.dominance_thresh
        ):
            reaction = winner

        self._last = TargetReactionReading(
            reaction=reaction,
            confidence=confidence,
            coloured_pixels=coloured,
        )
        self._throttle.record(now)
        return self._last
