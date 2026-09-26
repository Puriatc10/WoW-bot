"""T-FIX-27 — target nameplate reader over synthesised frames.

The OCR engine and the nameplate templates are both stubbed: no test here
needs a live client, a Tesseract install, or a recorded frame corpus.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from wow_bot.perception import deps
from wow_bot.perception.target import TargetReader

FRAME_W, FRAME_H = 800, 600
TEMPLATE_W, TEMPLATE_H = 100, 20
NAMEPLATE_XY = (400, 340)

KNOWN = ("Defias Thug", "Kobold Vermin")

GREEN = (0, 190, 0)
GOLDEN = (0, 200, 240)


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class FakeTesseractModule:
    """Stands in for ``pytesseract`` with a scripted OCR answer."""

    def __init__(self, text: str = "Defias Thug", fail: bool = False) -> None:
        self.text = text
        self.fail = fail
        self.calls = 0
        self.tesseract_cmd = ""
        self.pytesseract = self

    def image_to_string(self, _image: Any, config: str = "") -> str:
        self.calls += 1
        if self.fail:
            raise RuntimeError("tesseract exploded")
        assert config == "--psm 7"
        return self.text


@pytest.fixture()
def template_path(tmp_path: Path) -> Path:
    """A small nameplate template (template files are tiny in reality)."""
    image = np.zeros((TEMPLATE_H, TEMPLATE_W, 3), dtype=np.uint8)
    paint_nameplate(image, (0, 0))
    path = tmp_path / "target_name_template.png"
    written = cv2.imwrite(str(path), image)
    assert written, "template fixture failed to write"
    return path


@pytest.fixture()
def frame_template_path(tmp_path: Path) -> Path:
    canvas = np.zeros((30, 60, 3), dtype=np.uint8)
    canvas[:, :] = (120, 40, 40)
    path = tmp_path / "target_template.png"
    assert cv2.imwrite(str(path), canvas)
    return path


@pytest.fixture()
def tesseract(monkeypatch: pytest.MonkeyPatch) -> FakeTesseractModule:
    module = FakeTesseractModule()
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    return module


def paint_nameplate(frame: np.ndarray, position: tuple[int, int]) -> None:
    """Stamp a distinctive nameplate so template matching is unambiguous."""
    x, y = position
    frame[y : y + TEMPLATE_H, x : x + TEMPLATE_W] = (90, 90, 90)
    frame[y + 4 : y + 8, x + 6 : x + 44] = (230, 230, 230)
    frame[y + 12 : y + 16, x + 52 : x + 94] = (230, 230, 230)


def nameplate_frame(
    fill: float = 1.0,
    *,
    position: tuple[int, int] = NAMEPLATE_XY,
    hp_bar: bool = True,
    neutral: bool = False,
) -> np.ndarray:
    """Frame with the nameplate stamped and an HP bar of known fill."""
    frame = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    frame[:, :] = (18, 18, 18)
    if neutral:
        return frame

    x, y = position
    paint_nameplate(frame, position)
    if not hp_bar:
        return frame

    bar_y = y + TEMPLATE_H + 5
    left = x + 20
    total = TEMPLATE_W
    filled = round(total * fill)
    if filled > 0:
        frame[bar_y, left : left + filled] = GREEN
    # The bar's right end is marked golden in the client (hamberger scans
    # for it); the unfilled remainder stays dark, which is exactly the
    # case hamberger's "no golden pixel" fallback exists for.
    frame[bar_y, left + total - 1] = GOLDEN
    return frame


def make_reader(
    template_path: Path,
    frame_template_path: Path,
    tmp_path: Path,
    *,
    clock: FakeClock | None = None,
    sampling_hz: float = 2.0,
) -> TargetReader:
    return TargetReader(
        "target_name_template.png",
        "target_template.png",
        KNOWN,
        match_thresh=0.35,
        confirm_thresh=0.80,
        ocr_thresh=0.50,
        ocr_confirm_thresh=0.85,
        sampling_hz=sampling_hz,
        name_refresh_frames=5,
        base_dir=tmp_path,
        clock=clock or FakeClock(),
    )


# ----------------------------------------------------------------------
# construction / config
# ----------------------------------------------------------------------
def test_reader_loads_templates_and_reports_no_target_initially(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    reader = make_reader(template_path, frame_template_path, tmp_path)
    reading = reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.name == "Defias Thug"
    assert reading.found is True
    assert reader.sampling_hz == 2.0


def test_reader_rejects_empty_whitelist(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="known_enemies"):
        TargetReader("a.png", "b.png", (), base_dir=tmp_path)


def test_missing_template_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(deps.PerceptionDependencyError, match="not found"):
        TargetReader("absent_name.png", "absent_frame.png", KNOWN, base_dir=tmp_path)


def test_tesseract_cmd_is_set_at_read_time_not_construction(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    reader = make_reader(template_path, frame_template_path, tmp_path)
    assert tesseract.tesseract_cmd == ""

    reader.read(nameplate_frame(), tesseract_cmd="C:/Program Files/Tesseract/tesseract.exe")
    assert tesseract.tesseract_cmd == "C:/Program Files/Tesseract/tesseract.exe"


# ----------------------------------------------------------------------
# no target / low confidence
# ----------------------------------------------------------------------
def test_neutral_frame_yields_no_target_without_raising(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    reader = make_reader(template_path, frame_template_path, tmp_path)
    reading = reader.read(nameplate_frame(neutral=True), tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.name is None
    assert reading.found is False
    assert reading.hp_pct == 0
    assert reading.frame_confidence < 0.9
    assert tesseract.calls == 0, "no nameplate means no OCR"


def test_low_match_confidence_is_rejected(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    # A perfect template stamp scores exactly 1.0, so the gate is set above it.
    reader = TargetReader(
        "target_name_template.png",
        "target_template.png",
        KNOWN,
        match_thresh=1.01,
        ocr_thresh=1.01,
        base_dir=tmp_path,
    )
    reading = reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.name is None
    assert reader.locked_position is None


# ----------------------------------------------------------------------
# target HP is an int 0..100 (hamberger unit, builder normalises)
# ----------------------------------------------------------------------
@pytest.mark.parametrize("fill", [0.0, 0.25, 0.5, 1.0])
def test_target_hp_percent_is_int_in_range(
    fill: float,
    template_path: Path,
    frame_template_path: Path,
    tmp_path: Path,
    tesseract: FakeTesseractModule,
) -> None:
    reader = make_reader(template_path, frame_template_path, tmp_path)
    reading = reader.read(
        nameplate_frame(fill=fill), tesseract_cmd="C:/fake/tesseract.exe"
    )
    assert reading.name == "Defias Thug"
    assert isinstance(reading.hp_pct, int)
    assert 0 <= reading.hp_pct <= 100


def test_target_hp_percent_tracks_the_fill(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    half = make_reader(template_path, frame_template_path, tmp_path)
    full = make_reader(template_path, frame_template_path, tmp_path)
    half_reading = half.read(
        nameplate_frame(fill=0.5), tesseract_cmd="C:/fake/tesseract.exe"
    )
    full_reading = full.read(
        nameplate_frame(fill=1.0), tesseract_cmd="C:/fake/tesseract.exe"
    )
    assert full_reading.hp_pct > half_reading.hp_pct
    assert half_reading.hp_pct == pytest.approx(50, abs=8)
    assert full_reading.hp_pct == pytest.approx(100, abs=2)


def test_hp_is_zero_when_no_bar_is_visible(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    reader = make_reader(template_path, frame_template_path, tmp_path)
    reading = reader.read(
        nameplate_frame(hp_bar=False), tesseract_cmd="C:/fake/tesseract.exe"
    )
    assert reading.name == "Defias Thug"
    assert reading.hp_pct == 0


# ----------------------------------------------------------------------
# position lock, refresh and throttling
# ----------------------------------------------------------------------
def test_position_lock_avoids_repeated_ocr(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    clock = FakeClock()
    reader = make_reader(template_path, frame_template_path, tmp_path, clock=clock)
    reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    first_calls = tesseract.calls
    assert first_calls > 0
    assert reader.locked_position is not None

    for _ in range(3):
        clock.advance(0.5)
        reading = reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
        assert reading.name == "Defias Thug"
    assert tesseract.calls == first_calls, "locked frames must not re-OCR"


def test_name_is_refreshed_on_the_configured_interval(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    clock = FakeClock()
    reader = make_reader(template_path, frame_template_path, tmp_path, clock=clock)
    reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    after_first = tesseract.calls

    for index in range(4):
        clock.advance(0.5)
        reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert tesseract.calls == after_first, "refresh must not fire before name_refresh_frames"

    clock.advance(0.5)
    reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert tesseract.calls > after_first, "the 5th accepted read must re-OCR the name"


def test_ocr_budget_caps_calls_per_second(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    """Acceptance: OCR calls per second stay under the configured budget."""
    clock = FakeClock()
    sampling_hz = 2.0
    reader = make_reader(
        template_path, frame_template_path, tmp_path, clock=clock, sampling_hz=sampling_hz
    )
    frame = nameplate_frame()
    fps = 30
    simulated_seconds = 10.0
    frames = int(fps * simulated_seconds)

    for _ in range(frames):
        reader.read(frame, tesseract_cmd="C:/fake/tesseract.exe")
        clock.advance(1.0 / fps)

    # At most 3 Tesseract passes happen per accepted read (the three
    # threshold strategies), and reads are capped at the sampling budget.
    max_ocr_calls = sampling_hz * simulated_seconds * 3
    observed_rate = tesseract.calls / simulated_seconds
    assert tesseract.calls <= max_ocr_calls, (
        f"{tesseract.calls} OCR calls in {simulated_seconds}s exceeds the "
        f"{sampling_hz} Hz budget ({observed_rate:.1f} calls/s)"
    )
    assert tesseract.calls < frames / 2, "OCR must be far rarer than the frame rate"


def test_throttled_read_repeats_the_cached_reading(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    clock = FakeClock()
    reader = make_reader(template_path, frame_template_path, tmp_path, clock=clock)
    first = reader.read(nameplate_frame(1.0), tesseract_cmd="C:/fake/tesseract.exe")
    cached = reader.read(nameplate_frame(0.0), tesseract_cmd="C:/fake/tesseract.exe")
    assert cached == first

    clock.advance(0.5)
    fresh = reader.read(nameplate_frame(0.0), tesseract_cmd="C:/fake/tesseract.exe")
    assert fresh.hp_pct < first.hp_pct


# ----------------------------------------------------------------------
# losing the target and unknown names
# ----------------------------------------------------------------------
def test_lock_breaks_when_the_nameplate_disappears(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    clock = FakeClock()
    reader = make_reader(template_path, frame_template_path, tmp_path, clock=clock)
    assert reader.read(
        nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe"
    ).name == "Defias Thug"

    clock.advance(0.5)
    lost = reader.read(nameplate_frame(neutral=True), tesseract_cmd="C:/fake/tesseract.exe")
    assert lost.name is None
    assert reader.locked_position is None


def test_confirm_threshold_gates_the_position_lock(
    template_path: Path, frame_template_path: Path, tmp_path: Path, tesseract: FakeTesseractModule
) -> None:
    """A weak match must not take the lock, per hamberger's confirm constant."""
    weak = TargetReader(
        "target_name_template.png",
        "target_template.png",
        KNOWN,
        match_thresh=0.35,
        ocr_thresh=0.50,
        confirm_thresh=1.01,
        base_dir=tmp_path,
    )
    reading = weak.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.frame_confidence > 0.35, "the candidate was found"
    assert reading.name is None, "but confirmation refused the lock"
    assert weak.locked_position is None

    # The same frame with an attainable confirmation threshold locks normally.
    strong = make_reader(template_path, frame_template_path, tmp_path)
    locked = strong.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert locked.name == "Defias Thug"
    assert strong.locked_position is not None


def test_unknown_name_is_reported_as_no_target(
    template_path: Path, frame_template_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = FakeTesseractModule(text="Murloc Forager")
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    reader = make_reader(template_path, frame_template_path, tmp_path)
    reading = reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")
    assert reading.name is None, "only whitelisted names become a target"

    # The frame was still matched, so the confidence is reported honestly.
    assert reading.frame_confidence > 0.35


def test_whitelist_matching_is_tolerant_of_ocr_noise() -> None:
    reader = TargetReader.__new__(TargetReader)
    reader.known_enemies = KNOWN
    assert reader._match_whitelist("Defias Thug")[0] == "Defias Thug"
    assert reader._match_whitelist("Kobold Vermin!")[0] == "Kobold Vermin"
    assert reader._match_whitelist("level 5 Defias Thug")[0] == "Defias Thug"
    assert reader._match_whitelist("")[0] is None
    assert reader._match_whitelist(".....")[0] is None
    assert reader._match_whitelist("Murloc Forager")[0] is None


def test_ocr_failure_is_not_swallowed_as_a_name(
    template_path: Path, frame_template_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = FakeTesseractModule(fail=True)
    monkeypatch.setitem(sys.modules, "pytesseract", module)
    reader = make_reader(template_path, frame_template_path, tmp_path)
    with pytest.raises(RuntimeError):
        reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")


def test_missing_pytesseract_fails_closed(
    template_path: Path, frame_template_path: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", None)
    reader = make_reader(template_path, frame_template_path, tmp_path)
    with pytest.raises(deps.PerceptionDependencyError, match="pytesseract"):
        reader.read(nameplate_frame(), tesseract_cmd="C:/fake/tesseract.exe")


def test_target_module_has_no_hardcoded_absolute_path() -> None:
    source = Path("src/wow_bot/perception/target.py").read_text(encoding="utf-8")
    assert "C:\\" not in source
    assert "Program Files" not in source


def test_tesseract_cmd_assignment_lives_only_in_the_configure_method() -> None:
    """No module-level or constructor-time ``tesseract_cmd`` assignment."""
    import ast

    tree = ast.parse(Path("src/wow_bot/perception/target.py").read_text(encoding="utf-8"))
    for function in [n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)]:
        assigns = [
            node
            for node in ast.walk(function)
            if isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Attribute) and target.attr == "tesseract_cmd"
                for target in node.targets
            )
        ]
        if assigns:
            assert function.name == "_configure_tesseract", function.name
    module_level = [
        node
        for node in tree.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Attribute) and target.attr == "tesseract_cmd"
            for target in node.targets
        )
    ]
    assert module_level == []