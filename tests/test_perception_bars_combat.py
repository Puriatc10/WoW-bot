"""T-FIX-27 — bars and combat: the cheap per-frame channels.

Acceptance items covered here:
* ``BarReader`` returns a ratio in ``[0, 1]`` for a synthetic bar of known
  fill;
* ``CombatDetector`` returns ``True`` for a red edge strip and ``False``
  for a neutral one.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest

from wow_bot.perception.bars import BarReader, fill_ratio
from wow_bot.perception.combat import CombatDetector


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def blank(width: int = 200, height: int = 60, color: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def put_rect(
    frame: np.ndarray,
    x: int,
    y: int,
    width: int,
    height: int,
    color: tuple[int, int, int],
) -> None:
    frame[y : y + height, x : x + width] = color


def bgr_to_hsv_frame(bgr: tuple[int, int, int]) -> np.ndarray:
    pixel = np.array([[bgr]], dtype=np.uint8)
    return cv2.cvtColor(pixel, cv2.COLOR_BGR2HSV)[0, 0]


# ----------------------------------------------------------------------
# bar colours
# ----------------------------------------------------------------------
GREEN_BGR = (0, 200, 0)
MANA_BGR = (200, 100, 0)  # blue-ish, saturated and bright
DARK_BGR = (40, 40, 40)


def test_bar_colours_satisfy_the_fill_gate() -> None:
    """Guards the synthetic fixtures themselves against a bad colour pick."""
    for name, bgr in (("green", GREEN_BGR), ("mana", MANA_BGR), ("dark", DARK_BGR)):
        _, saturation, value = bgr_to_hsv_frame(bgr)
        if name == "dark":
            assert not (saturation > 80 and value > 80)
        else:
            assert saturation > 80 and value > 80


def make_bar_frame(fill: float, *, hp_roi=(10, 10, 100, 8), mana_roi=(10, 30, 100, 8)) -> np.ndarray:
    """Synthesise a frame with a known HP fill and a known mana fill."""
    frame = blank(200, 60, DARK_BGR)
    hx, hy, hw, hh = hp_roi
    put_rect(frame, hx, hy, round(hw * fill), hh, GREEN_BGR)
    mx, my, mw, mh = mana_roi
    put_rect(frame, mx, my, round(mw * 0.25), mh, MANA_BGR)
    return frame


HP_ROI = (10, 10, 100, 8)
MANA_ROI = (10, 30, 100, 8)


@pytest.mark.parametrize("fill", [0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0])
def test_fill_ratio_matches_known_fill(fill: float) -> None:
    frame = make_bar_frame(fill)
    ratio = fill_ratio(frame, HP_ROI)
    assert 0.0 <= ratio <= 1.0
    if fill == 0.0:
        assert ratio == 0.0
    else:
        assert ratio == pytest.approx(fill, abs=0.02)


def test_bar_reader_returns_both_ratios_in_unit_range() -> None:
    reader = BarReader(HP_ROI, MANA_ROI)
    reading = reader.read_all(make_bar_frame(0.6))
    assert 0.0 <= reading.hp <= 1.0
    assert 0.0 <= reading.mana <= 1.0
    assert reading.hp == pytest.approx(0.6, abs=0.02)
    assert reading.mana == pytest.approx(0.25, abs=0.02)
    # read_hp/read_mana must agree with read_all on the same frame.
    assert reader.read_hp(make_bar_frame(0.6)) == pytest.approx(reading.hp, abs=0.02)


def test_bar_reader_clamps_miscalibrated_roi() -> None:
    """A fill starting at column 0 must never yield more than 1.0."""
    frame = blank(200, 60, DARK_BGR)
    put_rect(frame, 0, 10, 200, 8, GREEN_BGR)
    assert fill_ratio(frame, HP_ROI) == 1.0


def test_bar_reader_accepts_bgra_frames() -> None:
    bgr = make_bar_frame(0.5)
    bgra = np.dstack([bgr, np.full(bgr.shape[:2], 255, dtype=np.uint8)])
    assert fill_ratio(bgra, HP_ROI) == pytest.approx(fill_ratio(bgr, HP_ROI))


def test_bar_reader_handles_out_of_bounds_roi() -> None:
    frame = blank(20, 20)
    assert fill_ratio(frame, (100, 100, 50, 50)) == 0.0
    assert fill_ratio(frame, (0, 0, 0, 0)) == 0.0


def test_bar_reader_caches_while_throttled() -> None:
    clock = FakeClock()
    reader = BarReader(HP_ROI, MANA_ROI, sampling_hz=2.0, clock=clock)
    full = reader.read_hp(make_bar_frame(1.0))
    assert full == pytest.approx(1.0)
    # A different frame at the same instant must not be re-read.
    assert reader.read_hp(make_bar_frame(0.0)) == pytest.approx(full)
    clock.advance(0.5)
    assert reader.read_hp(make_bar_frame(0.0)) == pytest.approx(0.0)


def test_bar_reader_default_budget_runs_every_frame() -> None:
    reader = BarReader(HP_ROI, MANA_ROI)
    assert reader.sampling_hz == float("inf")
    assert reader.read_hp(make_bar_frame(1.0)) == pytest.approx(1.0)
    assert reader.read_hp(make_bar_frame(0.0)) == pytest.approx(0.0)


def test_bar_reader_never_imports_ocr() -> None:
    import ast

    source = Path("src/wow_bot/perception/bars.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    imported: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported += [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            imported.append((node.module or "").split(".")[0])
    assert "pytesseract" not in imported
    assert "cv2" not in imported, "cv2 must stay behind deps.require_cv2()"


# ----------------------------------------------------------------------
# combat
# ----------------------------------------------------------------------
def make_edge_frame(
    *,
    width: int = 400,
    height: int = 300,
    edge: int = 30,
    color: tuple[int, int, int] = (0, 0, 200),
) -> np.ndarray:
    frame = blank(width, height, (30, 30, 30))
    put_rect(frame, 0, 0, width, edge, color)
    put_rect(frame, 0, height - edge, width, edge, color)
    put_rect(frame, 0, 0, edge, height, color)
    put_rect(frame, width - edge, 0, edge, height, color)
    return frame


def test_combat_detects_red_edges() -> None:
    detector = CombatDetector(edge_size=30, red_ratio_thresh=0.02, cooldown_s=1.0)
    frame = blank(400, 300, (30, 30, 30))
    put_rect(frame, 0, 0, 400, 30, (0, 0, 200))
    in_combat, ratio = detector.detect(frame, now=0.0)
    assert in_combat is True
    assert ratio > 0.02


def test_combat_is_false_for_neutral_edges() -> None:
    detector = CombatDetector(edge_size=30, red_ratio_thresh=0.02, cooldown_s=1.0)
    in_combat, ratio = detector.detect(blank(400, 300, (30, 30, 30)), now=0.0)
    assert in_combat is False
    assert ratio == 0.0


def test_combat_ignores_red_outside_the_edges() -> None:
    detector = CombatDetector(edge_size=30, red_ratio_thresh=0.02, cooldown_s=0.0)
    frame = blank(400, 300, (30, 30, 30))
    put_rect(frame, 150, 120, 100, 60, (0, 0, 255))
    in_combat, ratio = detector.detect(frame, now=0.0)
    assert in_combat is False
    assert ratio == 0.0


def test_combat_cooldown_latch_uses_injected_clock() -> None:
    clock = FakeClock()
    detector = CombatDetector(edge_size=30, red_ratio_thresh=0.02, cooldown_s=1.0, clock=clock)
    red = blank(400, 300, (30, 30, 30))
    put_rect(red, 0, 0, 400, 30, (0, 0, 200))
    neutral = blank(400, 300, (30, 30, 30))

    assert detector.detect(red)[0] is True
    clock.advance(0.5)
    assert detector.detect(neutral)[0] is True, "latch must hold inside the cooldown"
    clock.advance(0.6)
    assert detector.detect(neutral)[0] is False, "latch must expire after the cooldown"


def test_combat_latch_never_fires_without_a_red_frame() -> None:
    detector = CombatDetector(cooldown_s=5.0)
    neutral = blank(400, 300, (30, 30, 30))
    assert detector.detect(neutral, now=100.0)[0] is False
    assert detector.detect(neutral, now=101.0)[0] is False


def test_combat_ratio_threshold_gates_detection() -> None:
    frame = blank(400, 300, (30, 30, 30))
    put_rect(frame, 0, 0, 400, 30, (0, 0, 200))
    permissive = CombatDetector(edge_size=30, red_ratio_thresh=0.001)
    strict = CombatDetector(edge_size=30, red_ratio_thresh=0.99)
    assert permissive.detect(frame, now=0.0)[0] is True
    assert strict.detect(frame, now=0.0)[0] is False


def test_combat_accepts_bgra_and_uses_injected_clock_only() -> None:
    frame = make_edge_frame(color=(0, 0, 200))
    bgra = np.dstack([frame, np.full(frame.shape[:2], 255, dtype=np.uint8)])
    detector = CombatDetector(edge_size=30)
    assert detector.detect(bgra, now=0.0)[0] is True


def test_combat_rejects_invalid_tuning() -> None:
    with pytest.raises(ValueError):
        CombatDetector(edge_size=0)
    with pytest.raises(ValueError):
        CombatDetector(red_ratio_thresh=1.5)
    with pytest.raises(ValueError):
        CombatDetector(cooldown_s=-1.0)


def test_combat_module_has_no_wall_clock_call() -> None:
    import ast

    source = Path("src/wow_bot/perception/combat.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node.func.attr
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "time"
        and node.func.attr in {"time", "perf_counter"}
    ]
    assert calls == []
