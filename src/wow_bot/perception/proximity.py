"""Target-distance channel: observed nameplate width -> yards (T-FIX-30).

``TargetInfo.distance_estimate`` has no direct sensor. The per-frame
observable that scales with distance in the ported reader set is the
rendered size of the target nameplate, and T-FIX-27's
:class:`~wow_bot.perception.target.TargetReader` already locates both ends of
the nameplate HP bar by colour segmentation. The observed width in pixels is
therefore a measurement the port already makes; it is exposed additively as
:attr:`~wow_bot.perception.target.TargetReading.bar_width_px` and converted
here.

**One conversion site.** This module is the only place where a nameplate
pixel width becomes yards, exactly as
:mod:`wow_bot.perception.minimap` is the only degrees-to-radians site.

**The model is an inference, and it says so.** The conversion is
inverse-proportional (pinhole):

    distance_yd = reference_distance_yd * reference_width_px / width_px

It is exact only at the calibration pair, its absolute error grows with
distance (a one-pixel error is a large relative error on a small plate), and
the client may clamp nameplate scale at short and long range. It is
documented as low-to-medium accuracy in ``docs/PERCEPTION.md`` §2.1 and its
precision/recall is recorded there as still unmeasurable, because the frozen
frame corpus does not exist in the repository.

**Honest absence.** A width is only reported when the bar's right edge was
actually observed; the target reader's right-edge fallback keeps the HP
fraction working but is not a measurement. An unobserved width, or a
nameplate match below the confidence floor, yields ``None`` — and the state
builder then omits the target rather than letting ``target_in_range`` be
derived from a guessed distance.
"""

from __future__ import annotations

from dataclasses import dataclass

from wow_bot.perception.target import TargetReading

__all__ = [
    "TargetDistanceEstimator",
    "TargetDistanceModel",
    "TargetDistanceReading",
]

#: Default minimum nameplate-match confidence for a distance to be emitted.
_DEFAULT_MIN_CONFIDENCE = 0.5


@dataclass(frozen=True)
class TargetDistanceModel:
    """Inverse-proportional calibration from nameplate width to yards.

    ``reference_width_px`` is the HP-bar pixel width observed when the
    nameplate is ``reference_distance_yd`` yards away. Both are calibration
    constants from ``[proximity]``, never guesses.
    """

    reference_width_px: float
    reference_distance_yd: float

    def __post_init__(self) -> None:
        if self.reference_width_px <= 0.0:
            raise ValueError(
                f"reference_width_px must be > 0, got {self.reference_width_px}"
            )
        if self.reference_distance_yd <= 0.0:
            raise ValueError(
                f"reference_distance_yd must be > 0, got {self.reference_distance_yd}"
            )

    def distance_from_width(self, width_px: float) -> float | None:
        """Convert an observed pixel width to yards, or ``None`` if unusable."""
        if width_px <= 0.0:
            return None
        return self.reference_distance_yd * self.reference_width_px / width_px


@dataclass(frozen=True)
class TargetDistanceReading:
    """One frame's target-distance observation.

    ``confidence`` is the nameplate template-match score of the same read,
    and is reported even when ``distance_estimate`` is ``None``, so "seen but
    uncertain" stays distinguishable from "not attempted".
    """

    distance_estimate: float | None
    bar_width_px: float | None
    confidence: float


class TargetDistanceEstimator:
    """Converts an observed nameplate width into a yards estimate."""

    def __init__(
        self,
        model: TargetDistanceModel,
        *,
        min_confidence: float = _DEFAULT_MIN_CONFIDENCE,
    ) -> None:
        if not 0.0 <= min_confidence <= 1.0:
            raise ValueError(f"min_confidence must be within [0, 1], got {min_confidence}")
        self.model = model
        self.min_confidence = float(min_confidence)

    def estimate(
        self,
        reading: TargetReading,
        *,
        min_confidence: float | None = None,
    ) -> TargetDistanceReading:
        """Estimate the target's distance from one target-reader reading.

        ``min_confidence`` overrides the constructor value for this call;
        it exists so a caller holding a stricter policy can tighten the gate
        without rebuilding the estimator.
        """
        floor = self.min_confidence if min_confidence is None else float(min_confidence)
        width = reading.bar_width_px
        confidence = float(reading.frame_confidence)
        distance: float | None = None
        if width is not None and confidence >= floor:
            distance = self.model.distance_from_width(float(width))
        return TargetDistanceReading(
            distance_estimate=distance,
            bar_width_px=None if width is None else float(width),
            confidence=confidence,
        )
