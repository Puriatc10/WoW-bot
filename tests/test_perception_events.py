"""T-FIX-27 — event detector: OCR fan-out, dedup and the OCR budget.

Every test replaces ``pytesseract`` and calls the OCR entry point with a
stub, so nothing here needs a Tesseract install or a live client.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np
import pytest

from wow_bot.perception import deps
from wow_bot.perception.events import (
    COMBAT_KEYWORDS,
    EVENT_PRIORITY,
    EventDetector,
)

COMBAT_REGION = (0, 0, 100, 40)
CHAT_REGION = (0, 50, 100, 40)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class FakeTesseractModule:
    """Records every OCR call and replays a scripted answer."""

    def __init__(self, answer: str = "") -> None:
        self.answer = answer
        self.calls: list[tuple[Any, str]] = []
        self.tesseract_cmd = ""
        self.pytesseract = self

    def image_to_string(self, image: Any, config: str = "") -> str:
        self.calls.append((image, config))
        return self.answer


@pytest.fixture()
def tesseract(monkeypatch: pytest.MonkeyPatch) -> FakeTesseractModule:
    module = FakeTesseractModule()
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return module


def frame(width: int = 120, height: int = 100) -> np.ndarray:
    return np.full((height, width, 3), 30, dtype=np.uint8)


def make_detector(clock: FakeClock | None = None, sampling_hz: float = 1.0) -> EventDetector:
    return EventDetector(
        COMBAT_REGION,
        CHAT_REGION,
        sampling_hz=sampling_hz,
        clock=clock or FakeClock(),
    )


def test_ocr_fan_out_is_five_calls_per_sample(
    tesseract: FakeTesseractModule,
) -> None:
    detector = make_detector()
    assert detector.ocr_calls_per_sample == 5
    detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert len(tesseract.calls) == 5
    assert all(config == "--psm 6" for _, config in tesseract.calls)


def test_ocr_calls_per_second_stay_within_the_configured_budget(
    tesseract: FakeTesseractModule,
) -> None:
    """Acceptance: OCR calls per second stay under the configured budget."""
    clock = FakeClock()
    sampling_hz = 1.0
    detector = make_detector(clock, sampling_hz=sampling_hz)
    tesseract.answer = "You receive loot: Linen Cloth"

    fps = 30
    simulated_seconds = 10.0
    for _ in range(int(fps * simulated_seconds)):
        detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
        clock.advance(1.0 / fps)

    max_calls = detector.ocr_calls_per_sample * sampling_hz * simulated_seconds
    assert len(tesseract.calls) <= max_calls, (
        f"{len(tesseract.calls)} OCR calls exceeds the {sampling_hz} Hz budget "
        f"({max_calls} allowed)"
    )


def test_classification_uses_the_priority_table(
    monkeypatch: pytest.MonkeyPatch, tesseract: FakeTesseractModule
) -> None:
    detector = make_detector()
    monkeypatch.setattr(
        detector,
        "_read_region_multi",
        lambda *args, **kwargs: "Defias Thug dies.\nYou receive loot: Linen Cloth.",
    )
    events = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    types = {event.type for event in events}
    assert "death" in types
    assert "loot" in types
    assert detector.combat_keywords.keys() == COMBAT_KEYWORDS.keys()
    assert EVENT_PRIORITY[0] == "death"


def test_events_carry_type_text_and_source_but_no_timestamp(
    monkeypatch: pytest.MonkeyPatch, tesseract: FakeTesseractModule
) -> None:
    detector = make_detector()
    monkeypatch.setattr(
        detector,
        "_read_region_multi",
        lambda *args, **kwargs: "You receive loot: Linen Cloth",
    )
    events = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert len(events) >= 1
    event = events[0]
    assert event.type == "loot"
    assert event.text == "You receive loot: Linen Cloth"
    assert event.source in {"combat", "chat"}
    # The monotonic timestamp is the builder's job (T-FIX-28).
    assert not hasattr(event, "timestamp")


def test_dedup_reports_a_line_only_once(
    monkeypatch: pytest.MonkeyPatch, tesseract: FakeTesseractModule
) -> None:
    clock = FakeClock()
    detector = make_detector(clock)
    monkeypatch.setattr(
        detector,
        "_read_region_multi",
        lambda _frame, region, _filters: (
            "Defias Thug dies."
            if region == COMBAT_REGION
            else "You receive loot: Linen Cloth"
        ),
    )
    first = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert {event.source for event in first} == {"combat", "chat"}
    clock.advance(1.0)
    second = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert second == [], "an already-reported line must not repeat"


_INVALID_LINES = (
        "abc"  # too short
        "\n[DAMAGE][DAMAGE] you hit"  # too many damage markers
        "\n|||||||||||||||"  # not enough alphanumeric content
        "\nYou receive loot: Linen Cloth"  # valid
    )


def test_invalid_lines_are_dropped(
    monkeypatch: pytest.MonkeyPatch, tesseract: FakeTesseractModule
) -> None:
    detector = make_detector()
    monkeypatch.setattr(
        detector,
        "_read_region_multi",
        lambda *args, **kwargs: _INVALID_LINES,
    )
    events = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    texts = {event.text for event in events}
    assert texts == {"You receive loot: Linen Cloth"}


def test_unclassifiable_text_produces_no_event(
    monkeypatch: pytest.MonkeyPatch, tesseract: FakeTesseractModule
) -> None:
    detector = make_detector()
    monkeypatch.setattr(
        detector,
        "_read_region_multi",
        lambda *args, **kwargs: "Nothing interesting happened here",
    )
    assert detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe") == []


def test_events_are_cached_while_throttled(
    tesseract: FakeTesseractModule,
) -> None:
    clock = FakeClock()
    detector = make_detector(clock, sampling_hz=1.0)
    tesseract.answer = "You receive loot: Linen Cloth"
    first = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert first
    calls_after_first = len(tesseract.calls)

    cached = detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert cached == first
    assert len(tesseract.calls) == calls_after_first

    clock.advance(1.0)
    detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert len(tesseract.calls) > calls_after_first


def test_tesseract_cmd_is_applied_at_detect_time(
    tesseract: FakeTesseractModule,
) -> None:
    detector = make_detector()
    assert tesseract.tesseract_cmd == ""
    detector.detect(frame(), tesseract_cmd="C:/Program Files/Tesseract/tesseract.exe")
    assert tesseract.tesseract_cmd == "C:/Program Files/Tesseract/tesseract.exe"


def test_region_outside_the_frame_is_ignored(
    tesseract: FakeTesseractModule,
) -> None:
    detector = EventDetector((500, 500, 10, 10), (500, 600, 10, 10), sampling_hz=1.0)
    assert detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe") == []


def test_missing_pytesseract_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    detector = make_detector()
    with pytest.raises(deps.PerceptionDependencyError, match="pytesseract"):
        detector.detect(frame(), tesseract_cmd="C:/fake/tesseract.exe")


def test_detector_rejects_invalid_dedup_limit() -> None:
    with pytest.raises(ValueError, match="dedup_limit"):
        EventDetector(COMBAT_REGION, CHAT_REGION, dedup_limit=0)


def test_regions_come_from_injected_config_not_module_constants() -> None:
    detector = EventDetector((1, 2, 3, 4), (5, 6, 7, 8))
    assert detector.combat_region == (1, 2, 3, 4)
    assert detector.chat_region == (5, 6, 7, 8)