"""T-FIX-30 — target-distance channel: nameplate width -> yards.

Pure conversion tests plus the honest-absence rules. No frame, no client, and
no model weight is needed: the observation is
``TargetReading.bar_width_px``, measured by the T-FIX-27 target reader.
"""

from __future__ import annotations

import pytest

from wow_bot.perception.builder import build_target_info
from wow_bot.perception.proximity import (
    TargetDistanceEstimator,
    TargetDistanceModel,
    TargetDistanceReading,
)
from wow_bot.perception.target import TargetReading

MODEL = TargetDistanceModel(reference_width_px=120.0, reference_distance_yd=10.0)


def reading(width: int | None, confidence: float = 0.9) -> TargetReading:
    return TargetReading(
        name="Defias Thug",
        hp_pct=50,
        frame_confidence=confidence,
        bar_width_px=width,
    )


# ----------------------------------------------------------------------
# model
# ----------------------------------------------------------------------
def test_model_is_inverse_proportional() -> None:
    assert MODEL.distance_from_width(120.0) == pytest.approx(10.0)
    assert MODEL.distance_from_width(60.0) == pytest.approx(20.0)
    assert MODEL.distance_from_width(240.0) == pytest.approx(5.0)


def test_model_rejects_non_positive_widths() -> None:
    assert MODEL.distance_from_width(0.0) is None
    assert MODEL.distance_from_width(-4.0) is None


def test_model_validates_its_calibration_constants() -> None:
    with pytest.raises(ValueError, match="reference_width_px"):
        TargetDistanceModel(reference_width_px=0.0, reference_distance_yd=10.0)
    with pytest.raises(ValueError, match="reference_distance_yd"):
        TargetDistanceModel(reference_width_px=120.0, reference_distance_yd=0.0)


# ----------------------------------------------------------------------
# estimator
# ----------------------------------------------------------------------
def test_observed_width_becomes_yards() -> None:
    estimate = TargetDistanceEstimator(MODEL, min_confidence=0.5).estimate(reading(120))
    assert estimate.distance_estimate == pytest.approx(10.0)
    assert estimate.bar_width_px == pytest.approx(120.0)
    assert estimate.confidence == pytest.approx(0.9)


def test_unobserved_width_yields_no_distance_but_keeps_the_width_slot_none() -> None:
    estimate = TargetDistanceEstimator(MODEL).estimate(reading(None))
    assert estimate.distance_estimate is None
    assert estimate.bar_width_px is None
    assert estimate == TargetDistanceReading(
        distance_estimate=None, bar_width_px=None, confidence=0.9
    )


def test_low_match_confidence_withholds_the_distance_but_reports_the_score() -> None:
    estimator = TargetDistanceEstimator(MODEL, min_confidence=0.5)
    estimate = estimator.estimate(reading(120, confidence=0.3))
    assert estimate.distance_estimate is None
    assert estimate.bar_width_px == pytest.approx(120.0)
    assert estimate.confidence == pytest.approx(0.3)


def test_per_call_confidence_override_tightens_the_gate() -> None:
    estimator = TargetDistanceEstimator(MODEL, min_confidence=0.2)
    assert estimator.estimate(reading(120, confidence=0.4)).distance_estimate is not None
    assert (
        estimator.estimate(reading(120, confidence=0.4), min_confidence=0.9).distance_estimate
        is None
    )


def test_construction_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        TargetDistanceEstimator(MODEL, min_confidence=-0.1)


# ----------------------------------------------------------------------
# it feeds the target contract, and only when observed
# ----------------------------------------------------------------------
def test_distance_completes_a_target_info_alongside_a_reaction() -> None:
    estimate = TargetDistanceEstimator(MODEL).estimate(reading(120))
    info = build_target_info(
        reading(120),
        reaction="hostile",
        distance_estimate=estimate.distance_estimate,
    )
    assert info is not None
    assert info.distance_estimate == pytest.approx(10.0)
    assert info.reaction == "hostile"


def test_an_unobserved_distance_leaves_the_target_absent() -> None:
    """A guessed distance would silently derive ``target_in_range``."""
    estimate = TargetDistanceEstimator(MODEL).estimate(reading(None))
    assert (
        build_target_info(
            reading(None),
            reaction="hostile",
            distance_estimate=estimate.distance_estimate,
        )
        is None
    )
