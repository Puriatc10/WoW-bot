"""T-FIX-31 — UI panel channels over synthesised frames.

One section per ``docs/PERCEPTION.md`` §2.1 channel: Bag frame, XP bar,
Character frame, Enemy cast bar, and Lootable-corpse indicator, plus the panel
composition.

Nothing here needs a live client, a Tesseract install, a YOLO weight, or the
frozen frame corpus: the OCR engine is stubbed and every frame is synthesised
with NumPy. The loot channel's precision/recall gate cannot be measured
without the corpus, so its tests prove the classification rule and the
*withheld assertion* rather than a measured accuracy.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from wow_bot.perception.bag import BagFrameReader, BagReading, slot_rectangles
from wow_bot.perception.cast import CastingReader, normalise_spell_name
from wow_bot.perception.durability import DurabilityReader, parse_percent_text
from wow_bot.perception.loot import (
    LOOT_CHANNEL_MEASURED,
    LootSparkleReader,
    LootSparkleReading,
    classify_loot_sparkle,
)
from wow_bot.perception.panels import (
    PanelObservations,
    PanelReaders,
    observe_panels,
    readers_from_config,
)
from wow_bot.perception.xp import XPBarReader, parse_level_text

FRAME_W, FRAME_H = 240, 160

SLOT_SIZE = (20, 20)
GRID_ORIGIN = (10, 10)
GAP = 2

GREEN = (0, 200, 0)
ORANGE = (0, 140, 255)
GOLD = (230, 225, 255)
BLUE_WHITE = (255, 240, 220)
BLUE_BORDER = (255, 0, 0)
GREY_BORDER = (110, 110, 110)


def noise_slot(rng_seed: int) -> np.ndarray:
    """A slot with no template likeness at all, drawn from a fixed seed."""
    rng = np.random.default_rng(rng_seed)
    return rng.integers(0, 256, size=(SLOT_SIZE[1], SLOT_SIZE[0], 3), dtype=np.uint8)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class FakeTesseractData:
    """Stands in for ``pytesseract`` with a per-ROI scripted answer.

    ``answers`` maps a region of interest to ``(text, confidence_percent)``;
    the confidence is reported in ``[0, 100]`` exactly as
    ``pytesseract.image_to_data`` reports it. A region with no entry reads as
    empty text at ``-1`` confidence — the engine's own "not a word" value.
    """

    def __init__(
        self,
        answers: dict[tuple[int, int, int, int], tuple[str, float]] | None = None,
    ) -> None:
        self.answers = dict(answers or {})
        self.calls = 0
        self.tesseract_cmd = ""
        self.pytesseract = self

    def image_to_data(
        self, _image: Any, config: str = "", output_type: str = ""
    ) -> dict[str, list[Any]]:
        self.calls += 1
        assert config == "--psm 7"
        assert output_type == "dict"
        text, confidence = self._answer_for_current_crop()
        tokens = text.split()
        if not tokens:
            # The engine's own shape for a row with no detected word.
            return {"text": [""], "conf": [-1.0]}
        # One confidence per token; the reader averages the digit tokens.
        return {"text": tokens, "conf": [confidence] * len(tokens)}

    def image_to_string(self, _image: Any, config: str = "") -> str:
        assert config == "--psm 7"
        return ""

    def _answer_for_current_crop(self) -> tuple[str, float]:
        """Return the scripted answer for the current crop.

        Each reader runs three thresholded OCR passes over one crop, so three
        consecutive calls belong to the same region: the answer is chosen by
        ``(calls - 1) // 3`` and a reader that walks its slots in order gets
        them in order. A single-answer fake answers every region with that
        text, which is what the one-region channels need.
        """
        if not self.answers:
            # The engine's own "no word here" value, not an invented score.
            return "", -1.0
        if len(self.answers) == 1:
            return next(iter(self.answers.values()))
        keys = sorted(self.answers)
        index = min((self.calls - 1) // 3, len(keys) - 1)
        return self.answers[keys[index]]


@pytest.fixture()
def tesseract(monkeypatch: pytest.MonkeyPatch) -> FakeTesseractData:
    module = FakeTesseractData()
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return module


def frame(colour: tuple[int, int, int] = (18, 18, 18)) -> np.ndarray:
    canvas = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    canvas[:, :] = colour
    return canvas


def patch(image: np.ndarray, roi: tuple[int, int, int, int], colour: tuple[int, int, int]) -> None:
    x, y, width, height = roi
    image[y : y + height, x : x + width] = colour


# ======================================================================
# Bag frame — template match per slot
# ======================================================================
def slot_texture() -> np.ndarray:
    """A textured empty-slot pattern.

    Uniform slots are a degenerate template-match input (zero variance), so
    the synthetic slot carries a fixed checker pattern; the real occupied
    slot is the same pattern plus an item-coloured block.
    """
    pattern = np.zeros((SLOT_SIZE[1], SLOT_SIZE[0], 3), dtype=np.uint8)
    pattern[:, :] = (60, 60, 60)
    for row in range(0, SLOT_SIZE[1], 4):
        for column in range(0, SLOT_SIZE[0], 4):
            if (row // 4 + column // 4) % 2 == 0:
                pattern[row : row + 2, column : column + 2] = (110, 110, 110)
    return pattern


def occupied_slot() -> np.ndarray:
    pattern = slot_texture()
    pattern[6:14, 6:14] = ORANGE
    return pattern


@pytest.fixture()
def bag_templates(tmp_path: Path) -> tuple[Path, Path]:
    """Synthetic slot templates at the paths the example config names."""
    models = tmp_path / "models"
    models.mkdir(exist_ok=True)
    empty_path = models / "bag_empty_slot_template.png"
    occupied_path = models / "bag_occupied_slot_template.png"
    assert cv2.imwrite(str(empty_path), slot_texture())
    assert cv2.imwrite(str(occupied_path), occupied_slot())
    return empty_path, occupied_path


def bag_frame(occupied_flags: list[bool]) -> np.ndarray:
    canvas = frame()
    for (x, y, _w, _h), occupied in zip(
        slot_rectangles(GRID_ORIGIN, 2, 2, SLOT_SIZE, GAP), occupied_flags, strict=True
    ):
        canvas[y : y + SLOT_SIZE[1], x : x + SLOT_SIZE[0]] = (
            occupied_slot() if occupied else slot_texture()
        )
    return canvas


def make_bag_reader(
    bag_templates: tuple[Path, Path],
    *,
    columns: int = 2,
    rows: int = 2,
    min_confidence: float = 0.7,
    clock: FakeClock | None = None,
) -> BagFrameReader:
    empty_path, occupied_path = bag_templates
    return BagFrameReader(
        GRID_ORIGIN,
        columns,
        rows,
        SLOT_SIZE,
        empty_slot_template=empty_path,
        occupied_slot_template=occupied_path,
        gap=GAP,
        min_confidence=min_confidence,
        sampling_hz=1.0,
        clock=clock or FakeClock(),
    )


def test_slot_rectangles_are_row_major_and_gap_aware() -> None:
    rects = slot_rectangles((10, 20), 2, 2, (6, 4), 2)
    assert rects == ((10, 20, 6, 4), (18, 20, 6, 4), (10, 26, 6, 4), (18, 26, 6, 4))


def test_occupied_slots_are_counted_and_capacity_comes_from_the_grid(
    bag_templates: tuple[Path, Path],
) -> None:
    reading = make_bag_reader(bag_templates).read(bag_frame([True, True, True, False]))
    assert reading.inventory_count == 3
    assert reading.inventory_max == 4
    assert reading.slots == 4
    assert reading.confidence >= 0.9


def test_an_empty_grid_is_zero_not_none(bag_templates: tuple[Path, Path]) -> None:
    reading = make_bag_reader(bag_templates).read(bag_frame([False, False, False, False]))
    assert reading.inventory_count == 0
    assert reading.inventory_max == 4


def test_one_uncertain_slot_withholds_the_whole_count(
    bag_templates: tuple[Path, Path],
) -> None:
    """A partial count silently suppresses full-bag handling."""
    canvas = bag_frame([True, True, True, False])
    x, y, _w, _h = slot_rectangles(GRID_ORIGIN, 2, 2, SLOT_SIZE, GAP)[3]
    canvas[y : y + SLOT_SIZE[1], x : x + SLOT_SIZE[0]] = noise_slot(1234)

    reading = make_bag_reader(bag_templates).read(canvas)
    assert reading.inventory_count is None
    assert reading.inventory_max is None
    # The gate is the per-slot score, so the count is withheld even though
    # the mean over four mostly-perfect slots still clears the threshold.
    assert reading.confidence > 0.7
    assert reading.slots == 4


def test_template_size_must_match_the_configured_slot(
    bag_templates: tuple[Path, Path],
) -> None:
    empty_path, occupied_path = bag_templates
    with pytest.raises(ValueError, match="slot_size"):
        BagFrameReader(
            GRID_ORIGIN,
            2,
            2,
            (30, 30),
            empty_slot_template=empty_path,
            occupied_slot_template=occupied_path,
        )


def test_a_grid_that_runs_off_the_frame_is_not_answered_partially(
    bag_templates: tuple[Path, Path],
) -> None:
    reader = make_bag_reader(bag_templates, columns=10, rows=10)
    reading = reader.read(bag_frame([True, True, True, False]))
    assert reading.inventory_count is None
    assert reading.slots < 100


def test_a_throttled_frame_re_serves_the_cached_sample(
    bag_templates: tuple[Path, Path],
) -> None:
    clock = FakeClock()
    reader = make_bag_reader(bag_templates, clock=clock)
    first = reader.read(bag_frame([True, True, True, True]))
    clock.advance(0.5)
    second = reader.read(bag_frame([False, False, False, False]))
    assert second == first


def test_bag_reader_validates_its_arguments(bag_templates: tuple[Path, Path]) -> None:
    empty_path, occupied_path = bag_templates
    with pytest.raises(ValueError, match="columns and rows"):
        BagFrameReader(
            GRID_ORIGIN,
            0,
            2,
            SLOT_SIZE,
            empty_slot_template=empty_path,
            occupied_slot_template=occupied_path,
        )
    with pytest.raises(ValueError, match="min_confidence"):
        BagFrameReader(
            GRID_ORIGIN,
            2,
            2,
            SLOT_SIZE,
            empty_slot_template=empty_path,
            occupied_slot_template=occupied_path,
            min_confidence=1.5,
        )


# ======================================================================
# XP bar — bar fill + level label
# ======================================================================
XP_BAR_ROI = (10, 10, 120, 8)
XP_LEVEL_ROI = (10, 30, 40, 12)


def xp_frame(fill_columns: int = 60) -> np.ndarray:
    canvas = frame()
    patch(canvas, XP_BAR_ROI, (30, 30, 30))
    patch(canvas, (10, 10, fill_columns, 8), GREEN)
    return canvas


@pytest.mark.parametrize(
    ("text", "expected"),
    [("17", 17), ("Lv 17", 17), ("  42  ", 42), ("17 / 80", 17)],
)
def test_level_text_extracts_the_integer(text: str, expected: int) -> None:
    assert parse_level_text(text) == expected


@pytest.mark.parametrize("text", ["", "no digits", "%"])
def test_level_text_without_digits_is_none(text: str) -> None:
    assert parse_level_text(text) is None


def test_level_and_fill_form_one_monotone_scalar(
    tesseract: FakeTesseractData, monkeypatch: pytest.MonkeyPatch
) -> None:
    tesseract.answers = {XP_LEVEL_ROI: ("17", 92.0)}
    reader = XPBarReader(
        XP_BAR_ROI,
        level_roi=XP_LEVEL_ROI,
        min_confidence=0.5,
        sampling_hz=1.0,
        clock=FakeClock(),
    )
    reading = reader.read(xp_frame(60))
    assert reading.level == 17
    assert reading.xp_fraction == pytest.approx(0.5)
    assert reading.level_or_xp == pytest.approx(17.5)
    assert reading.confidence == pytest.approx(0.92)


def test_a_low_confidence_level_withholds_the_scalar_but_reports_the_score(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {XP_LEVEL_ROI: ("17", 45.0)}
    reader = XPBarReader(
        XP_BAR_ROI, level_roi=XP_LEVEL_ROI, min_confidence=0.5, sampling_hz=1.0
    )
    reading = reader.read(xp_frame(60))
    assert reading.level_or_xp is None
    assert reading.level == 17
    assert reading.confidence == pytest.approx(0.45)


def test_no_level_region_means_no_scalar(tesseract: FakeTesseractData) -> None:
    reader = XPBarReader(XP_BAR_ROI, level_roi=None, sampling_hz=1.0)
    reading = reader.read(xp_frame(60))
    assert reading.level_or_xp is None
    assert reading.confidence == 0.0


def test_a_bar_that_fills_its_region_edge_to_edge_is_treated_as_occluded(
    tesseract: FakeTesseractData,
) -> None:
    """Colour segmentation has no score to report, so it is not given ``1.0``."""
    tesseract.answers = {XP_LEVEL_ROI: ("17", 92.0)}
    reader = XPBarReader(
        XP_BAR_ROI, level_roi=XP_LEVEL_ROI, min_confidence=0.5, sampling_hz=1.0
    )
    reading = reader.read(xp_frame(120))
    assert reading.xp_fraction == pytest.approx(1.0)
    assert reading.level_or_xp is None
    assert reading.confidence == pytest.approx(0.92)


# ======================================================================
# Character frame — per-slot durability
# ======================================================================
DURABILITY_SLOTS = ((10, 60, 30, 14), (60, 60, 30, 14))


@pytest.mark.parametrize(
    ("text", "expected"),
    [("80", 80), ("80%", 80), (" 100 ", 100), ("0", 0), ("57%", 57)],
)
def test_percent_text_is_parsed(text: str, expected: int) -> None:
    assert parse_percent_text(text) == expected


@pytest.mark.parametrize("text", ["", "no digits", "1000%", "-5"])
def test_impossible_percentages_are_rejected(text: str) -> None:
    """An out-of-range readout is an OCR error, not a durability."""
    assert parse_percent_text(text) is None

def test_durability_is_a_confidence_weighted_mean(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {
        DURABILITY_SLOTS[0]: ("80", 90.0),
        DURABILITY_SLOTS[1]: ("60", 70.0),
    }
    reader = DurabilityReader(
        DURABILITY_SLOTS,
        min_score=0.5,
        quorum_fraction=0.5,
        min_confidence=0.6,
        sampling_hz=1.0,
        clock=FakeClock(),
    )
    reading = reader.read(frame())
    assert reading.read_slots == 2
    assert reading.confidence == pytest.approx(0.80)
    # (0.80 * 0.90 + 0.60 * 0.70) / (0.90 + 0.70)
    assert reading.durability_fraction == pytest.approx(0.7125)


def test_below_quorum_withholds_durability_entirely(
    tesseract: FakeTesseractData,
) -> None:
    """``None`` routes to the safe path: the vendor skips repairing."""
    tesseract.answers = {
        DURABILITY_SLOTS[0]: ("80", 90.0),
        DURABILITY_SLOTS[1]: ("60", 30.0),
    }
    reader = DurabilityReader(
        DURABILITY_SLOTS,
        min_score=0.5,
        quorum_fraction=0.6,
        min_confidence=0.6,
        sampling_hz=1.0,
    )
    reading = reader.read(frame())
    assert reading.read_slots == 1
    assert reading.durability_fraction is None


def test_no_legible_slot_yields_no_fraction(tesseract: FakeTesseractData) -> None:
    tesseract.answers = {}
    reader = DurabilityReader(
        DURABILITY_SLOTS, min_score=0.5, quorum_fraction=0.5, sampling_hz=1.0
    )
    reading = reader.read(frame())
    assert reading.read_slots == 0
    assert reading.durability_fraction is None
    assert reading.confidence == 0.0


def test_durability_reader_rejects_an_empty_slot_list() -> None:
    with pytest.raises(ValueError, match="slot_rois"):
        DurabilityReader([])


# ======================================================================
# Enemy cast bar — one cast, never a guessed spell id
# ======================================================================
CAST_ROI = (20, 90, 160, 14)
BORDER_ROI = (20, 86, 160, 3)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Shadow Bolt", "shadow bolt"),
        ("  Shadow   Bolt. ", "shadow bolt"),
        ("SHADOW BOLT", "shadow bolt"),
    ],
)
def test_spell_names_are_normalised_for_the_id_table(raw: str, expected: str) -> None:
    assert normalise_spell_name(raw) == expected


def test_a_name_without_letters_normalises_to_nothing() -> None:
    assert normalise_spell_name("---") == ""


def cast_frame(
    *,
    fill_columns: int = 80,
    border: tuple[int, int, int] = BLUE_BORDER,
) -> np.ndarray:
    canvas = frame()
    patch(canvas, CAST_ROI, (25, 25, 25))
    patch(canvas, (20, 90, fill_columns, 14), (0, 120, 200))
    patch(canvas, BORDER_ROI, border)
    return canvas


def make_cast_reader(**overrides: Any) -> CastingReader:
    options: dict[str, Any] = {
        "spell_ids": {"Shadow Bolt": "686"},
        "border_roi": BORDER_ROI,
        "border_interruptible": False,
        "min_confidence": 0.6,
        "max_remaining_s": 20.0,
        "sampling_hz": 1.0,
        "clock": FakeClock(),
    }
    options.update(overrides)
    return CastingReader(CAST_ROI, **options)


def test_a_blue_border_marks_a_non_interruptible_cast(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 88.0)}
    reading = make_cast_reader().read(cast_frame(fill_columns=80))
    assert len(reading.casts) == 1
    cast = reading.casts[0]
    assert cast.spell_id == "686"
    assert cast.is_interruptible is False
    assert cast.caster_entity_id == "target"
    assert 0.0 < cast.remaining_cast_time_s <= 20.0


def test_no_blue_border_means_interruptible_when_configured_so(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 88.0)}
    reading = make_cast_reader(border_interruptible=True).read(
        cast_frame(fill_columns=80)
    )
    assert len(reading.casts) == 1
    assert reading.casts[0].is_interruptible is True


def test_an_unobserved_border_drops_the_whole_cast(
    tesseract: FakeTesseractData,
) -> None:
    """The interrupt is gated on the boolean, so it is never defaulted."""
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 88.0)}
    reading = make_cast_reader().read(cast_frame(border=GREY_BORDER))
    assert reading.casts == ()
    assert reading.spell_name == "shadow bolt"
    assert reading.interruptible is None


def test_a_missing_border_region_drops_the_whole_cast(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 88.0)}
    reading = make_cast_reader(border_roi=None).read(cast_frame())
    assert reading.casts == ()
    assert reading.interruptible is None


def test_a_low_confidence_name_yields_an_empty_tuple(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 40.0)}
    reading = make_cast_reader().read(cast_frame())
    assert reading.casts == ()
    assert reading.confidence == pytest.approx(0.40)


def test_a_name_absent_from_the_id_table_is_dropped(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Frostbolt", 95.0)}
    reading = make_cast_reader().read(cast_frame())
    assert reading.casts == ()
    assert reading.spell_name is None


def test_an_empty_id_table_drops_every_cast(tesseract: FakeTesseractData) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 95.0)}
    reading = make_cast_reader(spell_ids={}).read(cast_frame())
    assert reading.casts == ()


def test_no_cast_observed_is_the_honest_empty_tuple(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {}
    reading = make_cast_reader().read(frame())
    assert reading.casts == ()
    assert reading.bar_fill is not None


def test_cast_id_table_keys_are_normalised_at_construction(
    tesseract: FakeTesseractData,
) -> None:
    tesseract.answers = {CAST_ROI: ("Shadow Bolt", 95.0)}
    reader = make_cast_reader(spell_ids={"  Shadow   Bolt. ": 686})
    assert reader.spell_ids == {"shadow bolt": "686"}
    assert reader.read(cast_frame()).casts[0].spell_id == "686"


def test_cast_reader_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="min_confidence"):
        CastingReader(CAST_ROI, spell_ids={}, min_confidence=2.0)
    with pytest.raises(ValueError, match="max_remaining_s"):
        CastingReader(CAST_ROI, spell_ids={}, max_remaining_s=0.0)


# ======================================================================
# Lootable-corpse indicator — measured, and the assertion stays off
# ======================================================================
LOOT_ROI = (40, 120, 120, 30)


def test_the_channel_may_not_assert_while_the_corpus_is_absent() -> None:
    """§6's precision/recall target is unmeasurable, so ``True`` is withheld."""
    assert LOOT_CHANNEL_MEASURED is False


def test_a_bright_sparkle_yields_a_full_dominance() -> None:
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :] = GOLD
    positive, dominance, pixels = classify_loot_sparkle(crop)
    assert positive is True
    assert dominance == pytest.approx(1.0)
    assert pixels == 100


def test_an_ordinary_bright_object_is_not_a_sparkle() -> None:
    """A white/grey highlight is bright but belongs to no sparkle class."""
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    crop[:, :] = (240, 240, 240)
    positive, dominance, pixels = classify_loot_sparkle(crop)
    assert positive is False
    assert dominance == pytest.approx(0.0)
    assert pixels == 0


def test_a_dark_region_is_not_measured_at_all() -> None:
    crop = np.zeros((10, 10, 3), dtype=np.uint8)
    positive, dominance, pixels = classify_loot_sparkle(crop)
    assert positive is False
    assert dominance == 0.0
    assert pixels == 0


def test_a_cool_sparkle_counts_as_sparkle() -> None:
    crop = np.zeros((4, 4, 3), dtype=np.uint8)
    crop[:, :] = BLUE_WHITE
    positive, dominance, _pixels = classify_loot_sparkle(crop)
    assert positive is True
    assert dominance == pytest.approx(1.0)


def test_a_bright_gold_icon_is_not_a_sparkle() -> None:
    """Bright and warm is not enough: the sparkle is *whiter* than a gold icon."""
    crop = np.zeros((8, 8, 3), dtype=np.uint8)
    crop[:, :] = ORANGE
    positive, dominance, pixels = classify_loot_sparkle(crop)
    assert positive is False
    assert dominance == pytest.approx(0.0)
    assert pixels == 0


def test_a_confident_sparkle_is_measured_but_withheld() -> None:
    """The measurement is observable; the assertion is not made."""
    reader = LootSparkleReader(
        LOOT_ROI,
        min_pixels=10,
        dominance_thresh=0.6,
        min_confidence=0.9,
        sampling_hz=1.0,
        clock=FakeClock(),
    )
    canvas = frame()
    patch(canvas, LOOT_ROI, GOLD)
    reading = reader.read(canvas)
    assert reading.target_is_lootable is None
    assert reading.confidence == pytest.approx(1.0)
    assert reading.sparkle_pixels == LOOT_ROI[2] * LOOT_ROI[3]


def test_a_sparkle_below_the_thresholds_is_not_confident() -> None:
    reader = LootSparkleReader(
        LOOT_ROI,
        min_pixels=10,
        dominance_thresh=0.6,
        min_confidence=0.9,
        sampling_hz=1.0,
        clock=FakeClock(),
    )
    canvas = frame()
    # Half a bright non-sparkle block and half gold: 50% dominance, below the
    # configured floor.
    patch(canvas, LOOT_ROI, (240, 240, 240))
    patch(canvas, (LOOT_ROI[0], LOOT_ROI[1], LOOT_ROI[2] // 2, LOOT_ROI[3]), GOLD)
    reading = reader.read(canvas)
    assert reading.target_is_lootable is None
    assert reading.confidence < 0.9


def test_an_empty_sparkle_region_is_not_measured() -> None:
    reader = LootSparkleReader(LOOT_ROI, sampling_hz=1.0, clock=FakeClock())
    reading = reader.read(frame())
    assert reading.target_is_lootable is None
    assert reading.confidence == 0.0
    assert reading.sparkle_pixels == 0


def test_loot_reader_validates_its_arguments() -> None:
    with pytest.raises(ValueError, match="min_pixels"):
        LootSparkleReader(LOOT_ROI, min_pixels=0)
    with pytest.raises(ValueError, match="dominance_thresh"):
        LootSparkleReader(LOOT_ROI, dominance_thresh=1.5)


# ======================================================================
# Panel composition
# ======================================================================
class StubBag:
    """Minimal bag stand-in: no OpenCV, no templates, one scripted reading."""

    def __init__(self) -> None:
        self.reads = 0

    def read(self, _frame: np.ndarray, *, now: float | None = None) -> BagReading:
        del now
        self.reads += 1
        return BagReading(inventory_count=7, inventory_max=16, confidence=0.8, slots=16)


class StubLoot:
    """Minimal loot stand-in: a confident measurement with a withheld verdict."""

    def __init__(self) -> None:
        self._reading = LootSparkleReading(
            target_is_lootable=None, confidence=0.95, sparkle_pixels=44
        )

    def read(self, _frame: np.ndarray, *, now: float | None = None) -> LootSparkleReading:
        del now
        return self._reading


def test_unwired_channels_are_not_attempted() -> None:
    observations = observe_panels(frame(), readers=PanelReaders())
    assert observations == PanelObservations()
    assert observations.inventory_count is None
    assert observations.inventory_max is None
    assert observations.level_or_xp is None
    assert observations.durability_fraction is None
    assert observations.incoming_casts == ()
    assert observations.target_is_lootable is None
    assert observations.confidence == {}


def test_wired_channels_are_projected_and_scored() -> None:
    readers = PanelReaders(bag=StubBag())  # type: ignore[arg-type]
    observations = observe_panels(frame(), readers=readers)
    assert observations.inventory_count == 7
    assert observations.inventory_max == 16
    assert observations.level_or_xp is None
    # The emitted count contributes its measured score.
    assert observations.confidence == {"inventory_count": 0.8}


def test_a_channel_that_measured_nothing_contributes_no_score() -> None:
    readers = PanelReaders(loot=StubLoot())  # type: ignore[arg-type]
    observations = observe_panels(frame(), readers=readers)
    assert observations.confidence == {}
    assert observations.loot.sparkle_pixels == 44
    assert observations.target_is_lootable is None


def test_readers_are_built_from_the_validated_config(
    bag_templates: tuple[Path, Path],
) -> None:
    """The wiring path constructs all five readers from real config values."""
    from dataclasses import replace

    from wow_bot.perception.perception_config import load_perception_config

    config = load_perception_config(Path("config/perception.example.toml"))
    # The synthetic fixture slots are 20x20 rather than the example config's
    # 36x36, so only the geometry is adjusted; every other value, and the
    # template path resolution, comes from the real file.
    config = replace(config, bag_slot_size=SLOT_SIZE)
    readers = readers_from_config(config, base_dir=bag_templates[0].parent.parent)
    assert isinstance(readers.bag, BagFrameReader)
    assert isinstance(readers.xp, XPBarReader)
    assert isinstance(readers.durability, DurabilityReader)
    assert isinstance(readers.casting, CastingReader)
    assert isinstance(readers.loot, LootSparkleReader)
    assert readers.bag.slots == slot_rectangles(
        config.bag_grid_origin,
        config.bag_columns,
        config.bag_rows,
        config.bag_slot_size,
        config.bag_gap,
    )


def test_a_missing_bag_template_fails_at_wiring_time(tmp_path: Path) -> None:
    """Fail-closed at construction, not on the first frame."""
    from wow_bot.perception.deps import PerceptionDependencyError
    from wow_bot.perception.perception_config import load_perception_config

    config = load_perception_config(Path("config/perception.example.toml"))
    with pytest.raises(PerceptionDependencyError, match="Bag slot template"):
        readers_from_config(config, base_dir=tmp_path)


# ======================================================================
# Static import guards for the new modules
# ======================================================================
def test_new_panel_modules_avoid_forbidden_imports() -> None:
    forbidden_modules = {
        "aiosqlite",
        "asyncio",
        "random",
        "threading",
        "pydirectinput",
        "pyautogui",
        "pynput",
        "keyboard",
        "mss",
        "ultralytics",
        "torch",
        "wow_bot.main",
        "wow_bot.lab.runner_v2",
    }
    forbidden_substrings = {"ollama", "openai", "anthropic", "llm"}

    for name in (
        "_ocr.py",
        "bag.py",
        "xp.py",
        "durability.py",
        "cast.py",
        "loot.py",
        "panels.py",
    ):
        path = Path("src/wow_bot/perception") / name
        assert path.exists(), f"{path} must exist"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                root = module.split(".")[0]
                assert root not in forbidden_modules, f"{name} imports {module}"
                assert not any(sub in module.lower() for sub in forbidden_substrings), (
                    f"{name} imports {module}"
                )


def test_panel_readers_never_read_files_at_call_time() -> None:
    """Only template *construction* touches disk, and only for the bag."""
    source = Path("src/wow_bot/perception/durability.py").read_text(encoding="utf-8")
    assert "open(" not in source
    assert "imread" not in source
