"""T-FIX-27 — minimap tracker and the single degrees→radians site.

Acceptance item covered here: degrees→radians conversion exists in exactly
one location. The rest guards the arrow match, circular smoothing and the
fact that the returned point is screen pixels, never a world pose.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import cv2
import numpy as np
import pytest

from wow_bot.perception import deps
from wow_bot.perception.minimap import (
    MinimapTracker,
    degrees_to_facing_radians,
)

MINIMAP_ROI = (0, 0, 120, 120)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


def blank(width: int = 200, height: int = 200, color: tuple[int, int, int] = (20, 20, 20)) -> np.ndarray:
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:, :] = color
    return frame


def arrow_mask(angle_deg: float, size: int = 24) -> np.ndarray:
    """White triangle with its tip pointing along ``angle_deg``."""
    mask = np.zeros((size, size), dtype=np.uint8)
    center = (size - 1) / 2.0
    tip = (center + math.cos(math.radians(angle_deg)) * (size / 2.0 - 1.0),
           center - math.sin(math.radians(angle_deg)) * (size / 2.0 - 1.0))
    perp = math.radians(angle_deg + 90.0)
    half = size / 4.0
    base = (center - math.cos(math.radians(angle_deg)) * (size / 2.0 - 5.0),
            center + math.sin(math.radians(angle_deg)) * (size / 2.0 - 5.0))
    points = np.array(
        [
            [tip[0], tip[1]],
            [base[0] + math.cos(perp) * half, base[1] - math.sin(perp) * half],
            [base[0] - math.cos(perp) * half, base[1] + math.sin(perp) * half],
        ],
        dtype=np.float32,
    )
    cv2.fillPoly(mask, [np.round(points).astype(np.int32)], 255)
    return mask


def write_arrow_template(path: Path, angle_deg: float = 0.0, size: int = 24) -> Path:
    cv2.imwrite(str(path), arrow_mask(angle_deg, size))
    return path


def stamp(frame: np.ndarray, template: np.ndarray, x: int, y: int) -> None:
    height, width = template.shape[:2]
    frame[y : y + height, x : x + width] = cv2.cvtColor(template, cv2.COLOR_GRAY2BGR)


# ----------------------------------------------------------------------
# conversion — exactly one site
# ----------------------------------------------------------------------
def test_degrees_to_facing_radians_uses_half_open_range() -> None:
    assert degrees_to_facing_radians(0.0) == pytest.approx(0.0)
    assert degrees_to_facing_radians(90.0) == pytest.approx(math.pi / 2)
    assert degrees_to_facing_radians(270.0) == pytest.approx(-math.pi / 2)
    # The range is half-open: 180 degrees is -pi, never +pi.
    assert degrees_to_facing_radians(180.0) == pytest.approx(-math.pi)
    assert degrees_to_facing_radians(180.0) < math.pi
    assert degrees_to_facing_radians(-180.0) == pytest.approx(-math.pi)
    assert degrees_to_facing_radians(540.0) == pytest.approx(-math.pi)


def test_facing_convention_matches_mock_perception_projection() -> None:
    """``cos(facing)*v`` / ``sin(facing)*v`` must point where the arrow points.

    MockPerception projects with that pair, so a minimap arrow pointing up
    must yield ``facing = pi/2`` and therefore a positive ``y`` step.
    """
    facing = degrees_to_facing_radians(90.0)
    assert math.cos(facing) * 10.0 == pytest.approx(0.0, abs=1e-9)
    assert math.sin(facing) * 10.0 == pytest.approx(10.0)


def test_conversion_stays_in_half_open_range() -> None:
    for degrees in range(0, 360, 5):
        radians = degrees_to_facing_radians(float(degrees))
        assert -math.pi <= radians < math.pi


def test_degrees_to_radians_conversion_exists_in_exactly_one_file() -> None:
    """Acceptance: the conversion must live in one location only."""
    hits: list[str] = []
    for path in sorted(Path("src/wow_bot/perception").rglob("*.py")):
        source = path.read_text(encoding="utf-8")
        if "math.radians" in source or "np.radians" in source:
            hits.append(path.name)
    # minimap.py owns the facing conversion and its circular smoothing;
    # no other reader converts an angle at all.
    assert hits == ["minimap.py"], f"angle conversion found in {hits}"


def test_conversion_function_is_defined_once() -> None:
    definitions: list[str] = []
    for path in sorted(Path("src/wow_bot/perception").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.FunctionDef)
                and "facing_radians" in node.name
                and not node.decorator_list  # the @property accessors, not the converter
            ):
                definitions.append(f"{path.name}:{node.name}")
    assert definitions == ["minimap.py:degrees_to_facing_radians"], definitions


# ----------------------------------------------------------------------
# arrow tracking
# ----------------------------------------------------------------------
def test_tracker_finds_arrow_position(tmp_path: Path) -> None:
    template = arrow_mask(0.0)
    path = write_arrow_template(tmp_path / "arrow.png")
    frame = blank(200, 200)
    stamp(frame, template, 40, 60)

    tracker = MinimapTracker(MINIMAP_ROI, path, base_dir=tmp_path)
    reading = tracker.update(frame)
    assert reading.confidence > 0.5
    assert reading.position_px is not None
    x, y = reading.position_px
    assert x == pytest.approx(40 + template.shape[1] / 2.0, abs=1.0)
    assert y == pytest.approx(60 + template.shape[0] / 2.0, abs=1.0)


def test_tracker_reports_no_position_when_arrow_is_absent(tmp_path: Path) -> None:
    path = write_arrow_template(tmp_path / "arrow.png")
    tracker = MinimapTracker(MINIMAP_ROI, path, base_dir=tmp_path)
    reading = tracker.update(blank(200, 200))
    assert reading.position_px is None
    assert reading.angle_degrees is None
    assert reading.facing_radians is None


def test_tracker_keeps_heading_but_drops_position_when_arrow_is_lost(
    tmp_path: Path,
) -> None:
    """A lost arrow must not be reported at a stale screen position."""
    clock = FakeClock()
    template = arrow_mask(0.0)
    stamp_path = write_arrow_template(tmp_path / "arrow.png")
    tracker = MinimapTracker(MINIMAP_ROI, stamp_path, sampling_hz=100.0, clock=clock)
    frame = blank(200, 200)
    stamp(frame, template, 40, 60)
    first = tracker.update(frame)
    assert first.position_px is not None

    clock.advance(1.0)
    lost = tracker.update(blank(200, 200))
    assert lost.position_px is None
    assert lost.confidence == 0.0
    # Orientation is smoothed, so the last heading survives the loss.
    assert lost.angle_degrees == first.angle_degrees


def test_tracker_heading_follows_the_arrow_tip(tmp_path: Path) -> None:
    template = arrow_mask(0.0)
    path = write_arrow_template(tmp_path / "arrow.png")
    tracker = MinimapTracker(MINIMAP_ROI, path, base_dir=tmp_path)

    # Tip already points along +x, so the estimate must be near zero
    # (either side of the wrap point).
    frame = blank(200, 200)
    stamp(frame, template, 40, 60)
    reading = tracker.update(frame)
    assert reading.angle_degrees is not None
    assert min(reading.angle_degrees, 360 - reading.angle_degrees) < 25.0
    assert reading.facing_radians is not None
    assert abs(reading.facing_radians) < math.radians(25.0)

    # A template rotated to point up must be reported near 90 degrees,
    # because the tracker flips the screen's downward y axis.
    up_path = write_arrow_template(tmp_path / "arrow_up.png", angle_deg=90.0)
    up_tracker = MinimapTracker(MINIMAP_ROI, up_path, base_dir=tmp_path)
    up_frame = blank(200, 200)
    stamp(up_frame, arrow_mask(90.0), 40, 60)
    up_reading = up_tracker.update(up_frame)
    assert up_reading.angle_degrees is not None
    assert abs(up_reading.angle_degrees - 90.0) < 25.0
    assert up_reading.facing_radians == pytest.approx(math.pi / 2, abs=math.radians(25.0))


def test_tracker_smooths_angle_circularly(tmp_path: Path) -> None:
    """5-degree steps must not be averaged across the 0/360 wrap."""
    path = write_arrow_template(tmp_path / "arrow.png")
    tracker = MinimapTracker(MINIMAP_ROI, path, smoothing_frames=5, base_dir=tmp_path)
    smoothed = tracker._smooth_angle(355.0)
    assert smoothed == pytest.approx(355.0)
    smoothed = tracker._smooth_angle(2.0)
    assert smoothed is not None
    # A naive mean of [355, 2] would be ~178.5; the circular mean is ~358.5.
    assert smoothed > 350.0 or smoothed < 10.0


def test_tracker_caches_while_throttled(tmp_path: Path) -> None:
    clock = FakeClock()
    template = arrow_mask(0.0)
    path = write_arrow_template(tmp_path / "arrow.png")
    tracker = MinimapTracker(MINIMAP_ROI, path, sampling_hz=5.0, clock=clock)
    frame = blank(200, 200)
    stamp(frame, template, 40, 60)
    first = tracker.update(frame)
    assert first.confidence > 0.5

    # Same instant, empty frame: the cached reading is returned unchanged.
    cached = tracker.update(blank(200, 200))
    assert cached.position_px == first.position_px
    assert cached.confidence == first.confidence

    clock.advance(0.2)
    after = tracker.update(blank(200, 200))
    assert after.position_px is None
    assert after.confidence == 0.0


def test_tracker_missing_template_fails_closed(tmp_path: Path) -> None:
    tracker = MinimapTracker(MINIMAP_ROI, tmp_path / "absent.png", base_dir=tmp_path)
    # Construction is lazy, the failure is explicit at first use.
    with pytest.raises(deps.PerceptionDependencyError, match="arrow template"):
        tracker.update(blank(60, 60))


def test_tracker_rejects_bad_smoothing_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        MinimapTracker(MINIMAP_ROI, tmp_path / "a.png", smoothing_frames=0)


def test_tracker_position_is_screen_pixels_documented_as_such() -> None:
    source = Path("src/wow_bot/perception/minimap.py").read_text(encoding="utf-8")
    assert "not a world coordinate" in source
    assert "F-1" in source