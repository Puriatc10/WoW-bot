"""T-FIX-27 — YOLO enemy detector with the optional stack absent.

The ``yolo`` extra (ultralytics + torch) is not installed in the default
environment, which is exactly the MOCK_MODE/CI case the task requires:
importing the module and constructing the detector must fail *explicitly*
and never silently.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.perception import deps
from wow_bot.perception.enemies import EnemyDetection, EnemyDetector


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


# ----------------------------------------------------------------------
# optional-dependency behaviour
# ----------------------------------------------------------------------
def test_import_needs_no_optional_stack() -> None:
    import subprocess
    import sys as _sys

    code = (
        "import sys;"
        "import wow_bot.perception.enemies as e;"
        "loaded=[n for n in ('ultralytics','torch') if sys.modules.get(n) is not None];"
        "assert not loaded, loaded;"
        "print('OK')"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code], capture_output=True, text=True, check=False
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_missing_weights_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(deps.PerceptionDependencyError, match="weights not found"):
        EnemyDetector(tmp_path / "absent.pt", base_dir=tmp_path)


def test_missing_ultralytics_fails_closed(tmp_path: Path) -> None:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"not a real model")
    original = deps.require_ultralytics
    try:

        def raiser() -> Any:
            raise deps.PerceptionDependencyError(
                "ultralytics is required for real perception but is not installed."
            )

        deps.require_ultralytics = raiser  # type: ignore[assignment]
        with pytest.raises(deps.PerceptionDependencyError, match="ultralytics"):
            EnemyDetector(weights, base_dir=tmp_path)
    finally:
        deps.require_ultralytics = original  # type: ignore[assignment]


def test_offline_env_is_forced_before_the_lazy_import(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    for key in deps.YOLO_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"stub")
    seen: dict[str, bool] = {}

    def check_offline() -> Any:
        seen["offline"] = all(key in __import__("os").environ for key in deps.YOLO_OFFLINE_ENV)
        raise deps.PerceptionDependencyError("stop here")

    monkeypatch.setattr(deps, "require_ultralytics", check_offline)
    with pytest.raises(deps.PerceptionDependencyError):
        EnemyDetector(weights, base_dir=tmp_path)
    assert seen["offline"] is True


def test_invalid_confidence_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="confidence"):
        EnemyDetector(tmp_path / "a.pt", confidence=1.5, base_dir=tmp_path)
    with pytest.raises(ValueError, match="confidence"):
        EnemyDetector(tmp_path / "a.pt", confidence=-0.1, base_dir=tmp_path)


# ----------------------------------------------------------------------
# detection shapes, over a stubbed model
# ----------------------------------------------------------------------
class _FakeTensor:
    def __init__(self, values: list[list[float]]) -> None:
        self._values = np.asarray(values, dtype=float)

    def cpu(self) -> _FakeTensor:
        return self

    def numpy(self) -> np.ndarray:
        return self._values


class _FakeBox:
    def __init__(self, coords: list[float], confidence: float, class_id: int) -> None:
        self.xyxy = [_FakeTensor([coords])]
        self.conf = [confidence]
        self.cls = [class_id]


class _FakeResult:
    def __init__(self, boxes: list[_FakeBox]) -> None:
        self.boxes = boxes


class _FakeYOLO:
    def __init__(self, weights: str) -> None:
        self.weights = weights
        self.names = {0: "Defias Thug", 1: "Kobold Vermin"}
        self.calls = 0
        self.frame_shapes: list[tuple[int, ...]] = []

    def __call__(self, frame: np.ndarray, conf: float = 0.5, verbose: bool = False) -> list[_FakeResult]:
        self.calls += 1
        self.frame_shapes.append(frame.shape)
        return [
            _FakeResult(
                [
                    _FakeBox([10.0, 20.0, 60.0, 100.0], 0.9, 0),
                    _FakeBox([200.0, 50.0, 260.0, 90.0], 0.7, 1),
                ]
            )
        ]


@pytest.fixture()
def detector(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> EnemyDetector:
    weights = tmp_path / "model.pt"
    weights.write_bytes(b"stub weights")
    fake_module = type("FakeUltralytics", (), {"YOLO": _FakeYOLO})
    monkeypatch.setattr(deps, "require_ultralytics", lambda: fake_module)
    monkeypatch.setattr(
        deps, "apply_yolo_offline_env", lambda: dict(deps.YOLO_OFFLINE_ENV)
    )
    return EnemyDetector(weights, confidence=0.5, sampling_hz=2.0, base_dir=tmp_path)


def test_detections_keep_the_corner_pair_convention(detector: EnemyDetector) -> None:
    detections = detector.detect(np.zeros((120, 320, 3), dtype=np.uint8))
    assert len(detections) == 2
    first = detections[0]
    # (x1, y1, x2, y2), NOT (x, y, w, h) — the builder converts (trap 3).
    assert first.bbox == (10, 20, 60, 100)
    assert first.width == 50
    assert first.height == 80
    assert first.area == 4000
    assert first.center == (35, 60)
    assert first.name == "Defias Thug"
    assert first.confidence == pytest.approx(0.9)


def test_closest_enemy_is_the_largest_box(detector: EnemyDetector) -> None:
    closest = detector.closest_enemy(np.zeros((120, 320, 3), dtype=np.uint8))
    assert closest is not None
    assert closest.bbox == (10, 20, 60, 100)


def test_no_detections_yields_an_empty_list(detector: EnemyDetector) -> None:
    detector.model.__class__ = type(
        "EmptyYOLO", (), {"__call__": lambda self, *a, **k: [_FakeResult([])], "names": {0: "x"}}
    )
    assert detector.detect(np.zeros((120, 320, 3), dtype=np.uint8)) == []
    assert detector.closest_enemy(np.zeros((120, 320, 3), dtype=np.uint8)) is None


def test_detector_normalises_bgra_frames_before_inference(
    detector: EnemyDetector,
) -> None:
    detector.detect(np.zeros((120, 320, 4), dtype=np.uint8))
    assert detector.model.frame_shapes[-1] == (120, 320, 3)  # type: ignore[attr-defined]


def test_detect_is_throttled_with_cached_detections(tmp_path: Path) -> None:
    clock = FakeClock()
    weights_path = tmp_path / "model.pt"
    weights_path.write_bytes(b"stub")
    fake_module = type("FakeUltralytics", (), {"YOLO": _FakeYOLO})
    original = deps.require_ultralytics
    original_env = deps.apply_yolo_offline_env
    deps.require_ultralytics = lambda: fake_module  # type: ignore[assignment]
    deps.apply_yolo_offline_env = lambda: dict(deps.YOLO_OFFLINE_ENV)  # type: ignore[assignment]
    try:
        detector = EnemyDetector(
            weights_path, sampling_hz=2.0, clock=clock, base_dir=tmp_path
        )
        frame = np.zeros((120, 320, 3), dtype=np.uint8)
        first = detector.detect(frame)
        assert detector.model.calls == 1  # type: ignore[attr-defined]

        cached = detector.detect(frame)
        assert detector.model.calls == 1, "throttled frames must not run YOLO again"  # type: ignore[attr-defined]
        assert cached == first

        clock.advance(0.5)
        detector.detect(frame)
        assert detector.model.calls == 2  # type: ignore[attr-defined]
    finally:
        deps.require_ultralytics = original  # type: ignore[assignment]
        deps.apply_yolo_offline_env = original_env  # type: ignore[assignment]


def test_detections_are_immutable_value_objects() -> None:
    import dataclasses

    detection = EnemyDetection(name="x", bbox=(1, 2, 3, 4), confidence=0.5)
    with pytest.raises(dataclasses.FrozenInstanceError):
        detection.name = "y"  # type: ignore[misc]


# ----------------------------------------------------------------------
# static guards from the task contract
# ----------------------------------------------------------------------
def test_no_os_input_library_is_ported() -> None:
    """AGENTS.md §7: no pydirectinput (or any OS input) in perception."""
    source = Path("src/wow_bot/perception/enemies.py").read_text(encoding="utf-8")
    for forbidden in ("pydirectinput", "pyautogui", "pynput", "keyboard"):
        assert forbidden not in source


def test_no_hardcoded_absolute_path_in_perception_package() -> None:
    """Acceptance: no hardcoded absolute path under perception/."""
    offenders: list[str] = []
    for path in sorted(Path("src/wow_bot/perception").rglob("*.py")):
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            stripped = line.strip()
            if stripped.startswith("#") or '"""' in stripped:
                continue
            if "C:\\" in line or "C:/" in line or "/home/" in line or "/Users/" in line:
                if "Program Files" in line and "remedy" in line.lower():
                    continue
                offenders.append(f"{path.name}:{number}: {stripped}")
    assert offenders == [], offenders


def test_no_reader_owns_a_module_level_capture_singleton() -> None:
    """Contract: every reader is frame-in; composition is T-FIX-28's job."""
    for name in ("bars", "combat", "target", "enemies", "minimap", "events"):
        path = Path(f"src/wow_bot/perception/{name}.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                names = [t.id for t in node.targets if isinstance(t, ast.Name)]
                assert "capture" not in names, f"{name}.py defines a capture singleton"
        source = path.read_text(encoding="utf-8")
        assert "ScreenCapture(" not in source, f"{name}.py constructs a capture device"


def test_modules_never_import_actuation_or_lab() -> None:
    for name in ("capture", "bars", "combat", "target", "enemies", "minimap", "events"):
        path = Path(f"src/wow_bot/perception/{name}.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            module = ""
            if isinstance(node, ast.Import):
                module = node.names[0].name
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
            assert not module.startswith("wow_bot.actuation"), f"{name} imports actuation"
            assert not module.startswith("wow_bot.lab"), f"{name} imports the lab runner"
            assert not module.startswith("wow_bot.main"), f"{name} imports main"


def test_ultralytics_import_is_lazy() -> None:
    tree = ast.parse(Path("src/wow_bot/perception/enemies.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
            assert "ultralytics" not in names
        elif isinstance(node, ast.ImportFrom):
            assert (node.module or "") != "ultralytics"
    assert "require_ultralytics" in Path(
        "src/wow_bot/perception/enemies.py"
    ).read_text(encoding="utf-8")


def test_import_succeeds_without_cv2_either(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "cv2", None)
    import importlib

    module = importlib.import_module("wow_bot.perception.enemies")
    assert module.EnemyDetection is EnemyDetection


# ----------------------------------------------------------------------
# T-FIX-27 acceptance checklist, in one place
# ----------------------------------------------------------------------
READER_MODULES = ("capture", "bars", "combat", "target", "enemies", "minimap", "events")


def test_acceptance_all_deliverable_modules_exist_and_import() -> None:
    import importlib

    for name in READER_MODULES:
        module = importlib.import_module(f"wow_bot.perception.{name}")
        assert module is not None, name


def test_acceptance_no_hardcoded_absolute_path_under_perception() -> None:
    for name in READER_MODULES:
        path = Path(f"src/wow_bot/perception/{name}.py")
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
            if line.lstrip().startswith("#"):
                continue
            for marker in ("C:\\Program Files", "C:/Program Files", "/home/", "/Users/"):
                assert marker not in line, f"{name}.py:{number} hardcodes {marker}"


def test_acceptance_no_module_level_tesseract_assignment() -> None:
    for name in READER_MODULES:
        path = Path(f"src/wow_bot/perception/{name}.py")
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in tree.body:
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    assert not isinstance(target, ast.Attribute), (
                        f"{name}.py assigns {target.attr} at module level"
                    )


def test_acceptance_readers_never_construct_a_capture_device() -> None:
    for name in READER_MODULES:
        source = Path(f"src/wow_bot/perception/{name}.py").read_text(encoding="utf-8")
        if name == "capture":
            continue
        assert "ScreenCapture" not in source, f"{name}.py references the capture device"


def test_acceptance_every_throttled_channel_reads_its_budget_from_config() -> None:
    """The OCR/YOLO channels must expose the injected sampling budget."""
    from wow_bot.perception.bars import BarReader
    from wow_bot.perception.combat import CombatDetector
    from wow_bot.perception.events import EventDetector
    from wow_bot.perception.minimap import MinimapTracker

    assert BarReader((0, 0, 1, 1), (0, 0, 1, 1), sampling_hz=3.0).sampling_hz == 3.0
    assert CombatDetector(sampling_hz=4.0).sampling_hz == 4.0
    assert EventDetector((0, 0, 1, 1), (0, 0, 1, 1), sampling_hz=1.0).sampling_hz == 1.0
    assert (
        MinimapTracker(
            (0, 0, 1, 1), Path("models/arrow_template.png"), sampling_hz=6.0
        ).sampling_hz
        == 6.0
    )
    # TargetReader needs templates on disk, covered in test_perception_target.py.