"""T-FIX-30 — world-pose reader: OCR of an addon coordinate frame.

Covers the parse rules (including the recorded comma-decimal limitation), the
confidence gate, the throttle/cache behaviour, and the fact that nothing in
this channel ever invents a height. The OCR engine is stubbed, so no test here
needs a Tesseract install, a live client, or a recorded corpus.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.perception import deps
from wow_bot.perception.pose import (
    PoseReading,
    WorldPoseReader,
    parse_coordinate_text,
)

ROI = (0, 0, 60, 16)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class FakeTesseractData:
    """Stands in for ``pytesseract`` with a scripted word-confidence answer."""

    def __init__(self, text: str = "57.3, 42.1", confidence: float = 88.0) -> None:
        self.text = text
        self.confidence = confidence
        self.calls = 0
        self.tesseract_cmd = ""
        self.pytesseract = self

    def image_to_data(self, _image: Any, config: str = "", output_type: str = "") -> dict[str, list[Any]]:
        self.calls += 1
        assert config == "--psm 7"
        assert output_type == "dict"
        tokens = self.text.split()
        return {"text": tokens, "conf": [self.confidence] * len(tokens)}


@pytest.fixture()
def tesseract(monkeypatch: pytest.MonkeyPatch) -> FakeTesseractData:
    module = FakeTesseractData()
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return module


def frame(width: int = 200, height: int = 120, colour: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    canvas = np.zeros((height, width, 3), dtype=np.uint8)
    canvas[:, :] = colour
    return canvas


def make_reader(
    *,
    min_confidence: float = 0.6,
    sampling_hz: float = 1.0,
    roi: tuple[int, int, int, int] = ROI,
    clock: FakeClock | None = None,
) -> WorldPoseReader:
    return WorldPoseReader(
        roi,
        min_confidence=min_confidence,
        sampling_hz=sampling_hz,
        clock=clock or FakeClock(),
    )


# ----------------------------------------------------------------------
# parse rules
# ----------------------------------------------------------------------
@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("57.3, 42.1", (57.3, 42.1)),
        ("57.3 42.1", (57.3, 42.1)),
        ("57.3;42.1", (57.3, 42.1)),
        ("(57.3, 42.1)", (57.3, 42.1)),
        ("[57.3, 42.1]", (57.3, 42.1)),
        ("x=57.3 y=42.1", (57.3, 42.1)),
        ("  57.3,42.1  ", (57.3, 42.1)),
        ("-57.3, 42.1", (-57.3, 42.1)),
    ],
)
def test_two_numbers_are_parsed_regardless_of_separator_or_label(
    text: str, expected: tuple[float, float]
) -> None:
    assert parse_coordinate_text(text) == pytest.approx(expected)


@pytest.mark.parametrize(
    "text",
    [
        "",
        "no coordinates here",
        "57.3",
        "57.3, 42.1, 13.9",
        # Comma decimals carry four number tokens: rejected rather than
        # guessed at (ADR-003 Decision 1).
        "57,3 42,1",
        # A stray decimal point is not a separator between two coordinates.
        "57.342.1",
    ],
)
def test_ambiguous_or_unreadable_lines_are_rejected(text: str) -> None:
    assert parse_coordinate_text(text) is None


def test_nan_and_inf_never_enter_a_pose() -> None:
    assert parse_coordinate_text("nan, inf") is None


# ----------------------------------------------------------------------
# reader behaviour
# ----------------------------------------------------------------------
def test_confident_ocr_emits_a_world_pair(tesseract: FakeTesseractData) -> None:
    reader = make_reader()
    reading = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")

    assert reading.position == pytest.approx((57.3, 42.1))
    assert reading.confidence == pytest.approx(0.88)
    assert reading.sampled_at == 1.0
    assert tesseract.calls == 3, "three thresholdings per sample"


def test_low_confidence_keeps_the_score_but_withholds_the_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = FakeTesseractData(confidence=40.0)
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    reader = make_reader(min_confidence=0.6)

    reading = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.position is None
    assert reading.confidence == pytest.approx(0.40), "seen but uncertain is reported, not hidden"


def test_unparsable_text_yields_no_pose_and_zero_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = FakeTesseractData(text="zzz")
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    reader = make_reader()

    reading = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.position is None
    assert reading.confidence == 0.0, "not measured is 0.0, not a fabricated 1.0"


def test_throttled_frame_reserves_the_sample_without_re_ocr(tesseract: FakeTesseractData) -> None:
    clock = FakeClock()
    reader = make_reader(sampling_hz=1.0, clock=clock)
    first = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")
    calls_after_first = tesseract.calls

    cached = reader.read(frame(), now=1.2, tesseract_cmd="C:/fake/tesseract.exe")
    assert cached == first
    assert tesseract.calls == calls_after_first, "no OCR while throttled"

    clock.advance(2.0)
    reader.read(frame(), now=3.0, tesseract_cmd="C:/fake/tesseract.exe")
    assert tesseract.calls > calls_after_first


def test_reader_re_serves_the_last_sample_while_throttled_with_its_own_timestamp(
    tesseract: FakeTesseractData,
) -> None:
    reader = make_reader(sampling_hz=1.0)
    sample = reader.read(frame(), now=10.0, tesseract_cmd="C:/fake/tesseract.exe")
    cached = reader.read(frame(), now=10.5, tesseract_cmd="C:/fake/tesseract.exe")
    assert cached.sampled_at == sample.sampled_at == 10.0, "staleness stays visible"


def test_empty_roi_yields_no_pose(tesseract: FakeTesseractData) -> None:
    reader = make_reader(roi=(500, 500, 60, 16))
    reading = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.position is None
    assert reading.confidence == 0.0
    assert tesseract.calls == 0


def test_tesseract_cmd_is_set_at_read_time_not_construction(
    tesseract: FakeTesseractData,
) -> None:
    reader = make_reader()
    assert tesseract.tesseract_cmd == ""
    reader.read(frame(), now=1.0, tesseract_cmd="C:/Program Files/Tesseract/tesseract.exe")
    assert tesseract.tesseract_cmd == "C:/Program Files/Tesseract/tesseract.exe"


def test_missing_pytesseract_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    reader = make_reader()
    with pytest.raises(deps.PerceptionDependencyError, match="pytesseract"):
        reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")


def test_construction_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        WorldPoseReader(ROI, min_confidence=1.5)
    with pytest.raises(ValueError, match="upscale"):
        WorldPoseReader(ROI, upscale=0)


def test_sampling_hz_is_reported() -> None:
    assert make_reader(sampling_hz=2.5).sampling_hz == 2.5


def test_last_reading_defaults_to_not_observed() -> None:
    reader = make_reader()
    assert reader.last_reading == PoseReading(position=None, confidence=0.0)


def test_a_present_but_unreadable_roi_reports_no_pose(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-empty ROI with no readable text keeps ``position`` None."""
    module = FakeTesseractData(text="")
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    reader = make_reader(roi=(0, 0, 4, 4), min_confidence=0.0)
    reading = reader.read(frame(), now=1.0, tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.position is None
    assert reading.confidence == 0.0


def test_pose_module_has_no_hardcoded_absolute_path() -> None:
    source = Path("src/wow_bot/perception/pose.py").read_text(encoding="utf-8")
    assert "C:\\" not in source
    assert "Program Files" not in source


def test_pose_reader_never_reports_a_height() -> None:
    """The channel reads two coordinates; a third is not invented."""
    reading = PoseReading(position=(57.3, 42.1), confidence=0.9)
    assert not hasattr(reading, "player_z")
    assert math.isfinite(reading.position[0]) and math.isfinite(reading.position[1])
