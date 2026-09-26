"""T-FIX-30 — target-reaction reader: nameplate text colour -> reaction.

All frames here are synthesised NumPy canvases, so no test needs a live
client or a recorded corpus. The last group pins the honest-absence rules:
below the pixel floor, or below the dominance floor, the label is ``None``
rather than a guess.
"""

from __future__ import annotations

import numpy as np
import pytest

from wow_bot.perception.reaction import (
    TargetReactionReader,
    TargetReactionReading,
    classify_reaction,
)
from wow_bot.shared.interfaces import REACTIONS

HOSTILE = (0, 0, 220)  # BGR red
NEUTRAL = (0, 200, 245)  # BGR gold
FRIENDLY = (0, 200, 0)  # BGR green
WHITE = (255, 255, 255)
DARK = (20, 20, 20)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now


def canvas(colour: tuple[int, int, int], width: int = 40, height: int = 12) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = colour
    return frame


def mixed(left: tuple[int, int, int], right: tuple[int, int, int], width: int = 40) -> np.ndarray:
    frame = np.zeros((12, width, 3), dtype=np.uint8)
    frame[:, : width // 2] = left
    frame[:, width // 2 :] = right
    return frame


# ----------------------------------------------------------------------
# classifier
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("colour", "expected"),
    [
        (HOSTILE, "hostile"),
        (NEUTRAL, "neutral"),
        (FRIENDLY, "friendly"),
    ],
)
def test_solid_colour_crops_classify(colour: tuple[int, int, int], expected: str) -> None:
    label, confidence, coloured = classify_reaction(canvas(colour))
    assert label == expected
    assert confidence == pytest.approx(1.0)
    assert coloured == 40 * 12


@pytest.mark.parametrize("colour", [WHITE, DARK])
def test_unclassified_colours_yield_no_reaction(colour: tuple[int, int, int]) -> None:
    label, confidence, coloured = classify_reaction(canvas(colour))
    assert label is None
    assert confidence == 0.0
    assert coloured == 0


def test_every_emitted_label_is_a_schema_reaction() -> None:
    for colour in (HOSTILE, NEUTRAL, FRIENDLY):
        label, _, _ = classify_reaction(canvas(colour))
        assert label in REACTIONS


def test_gold_is_not_also_read_as_hostile() -> None:
    """Neutral is decided first, so a gold pixel is not classified twice."""
    label, _, _ = classify_reaction(canvas(NEUTRAL))
    assert label == "neutral"


def test_empty_crop_is_rejected() -> None:
    assert classify_reaction(np.zeros((0, 0, 3), dtype=np.uint8)) == (None, 0.0, 0)


# ----------------------------------------------------------------------
# reader behaviour
# ----------------------------------------------------------------------
def test_reader_emits_a_confident_reaction() -> None:
    reader = TargetReactionReader((0, 0, 40, 12), min_pixels=12, dominance_thresh=0.6)
    reading = reader.read(canvas(FRIENDLY), now=1.0)
    assert reading.reaction == "friendly"
    assert reading.confidence == pytest.approx(1.0)
    assert reading.coloured_pixels == 480


def test_too_few_coloured_pixels_withholds_the_label() -> None:
    reader = TargetReactionReader((0, 0, 2, 2), min_pixels=12)
    reading = reader.read(canvas(FRIENDLY, width=2, height=2), now=1.0)
    assert reading.reaction is None
    assert reading.coloured_pixels == 4
    assert reading.confidence == pytest.approx(1.0), "seen but uncertain is reported"


def test_close_classes_withhold_the_label() -> None:
    reader = TargetReactionReader((0, 0, 40, 12), min_pixels=12, dominance_thresh=0.6)
    reading = reader.read(mixed(HOSTILE, FRIENDLY), now=1.0)
    assert reading.reaction is None
    assert reading.confidence == pytest.approx(0.5)


def test_a_plain_frame_yields_no_reaction_without_raising() -> None:
    reader = TargetReactionReader((0, 0, 40, 12), min_pixels=12)
    reading = reader.read(canvas(DARK), now=1.0)
    assert reading.reaction is None
    assert reading.confidence == 0.0


def test_empty_roi_yields_no_reaction() -> None:
    reader = TargetReactionReader((500, 500, 40, 12), min_pixels=12)
    reading = reader.read(canvas(DARK), now=1.0)
    assert reading.reaction is None
    assert reading.coloured_pixels == 0


def test_throttled_frame_re_serves_the_cached_reading() -> None:
    reader = TargetReactionReader((0, 0, 40, 12), sampling_hz=1.0, clock=FakeClock())
    first = reader.read(canvas(HOSTILE), now=1.0)
    cached = reader.read(canvas(FRIENDLY), now=1.2)
    assert cached == first, "the sample is not recomputed while throttled"


def test_construction_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="min_pixels"):
        TargetReactionReader((0, 0, 1, 1), min_pixels=0)
    with pytest.raises(ValueError, match="dominance_thresh"):
        TargetReactionReader((0, 0, 1, 1), dominance_thresh=1.5)


def test_sampling_hz_defaults_to_every_frame() -> None:
    assert TargetReactionReader((0, 0, 1, 1)).sampling_hz == float("inf")


def test_last_reading_defaults_to_not_observed() -> None:
    reader = TargetReactionReader((0, 0, 1, 1))
    assert reader.last_reading == TargetReactionReading(reaction=None, confidence=0.0)
