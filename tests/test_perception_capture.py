"""T-FIX-27 — capture helpers and the per-channel frame budget.

Covers plan doc §6.6 (OCR/YOLO cannot run per frame) and the "no hardcoded
absolute path / no global capture singleton" contract items.
"""

from __future__ import annotations

import numpy as np
import pytest

from wow_bot.perception.capture import (
    PER_FRAME_HZ,
    ScreenCapture,
    Throttle,
    normalize_bgr,
)


class FakeClock:
    """Deterministic monotonic clock; tests advance it explicitly."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


# ----------------------------------------------------------------------
# normalize_bgr
# ----------------------------------------------------------------------
def test_normalize_bgr_accepts_bgr_bgra_and_gray() -> None:
    bgr = np.zeros((4, 5, 3), dtype=np.uint8)
    bgra = np.zeros((4, 5, 4), dtype=np.uint8)
    gray = np.zeros((4, 5), dtype=np.uint8)

    assert normalize_bgr(bgr).shape == (4, 5, 3)
    assert normalize_bgr(bgra).shape == (4, 5, 3)
    assert normalize_bgr(gray).shape == (4, 5, 3)
    # A BGRA frame is sliced, not reordered: channel order is preserved.
    marker = np.zeros((2, 2, 4), dtype=np.uint8)
    marker[:, :, 0] = 7
    assert int(normalize_bgr(marker)[0, 0, 0]) == 7


def test_normalize_bgr_rejects_bad_input() -> None:
    with pytest.raises(TypeError, match="None"):
        normalize_bgr(None)  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="numpy"):
        normalize_bgr("frame")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="dtype"):
        normalize_bgr(np.zeros((2, 2, 3), dtype=np.float32))
    with pytest.raises(ValueError, match="3D"):
        normalize_bgr(np.zeros((2, 2, 2, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="channels"):
        normalize_bgr(np.zeros((2, 2, 5), dtype=np.uint8))


# ----------------------------------------------------------------------
# Throttle
# ----------------------------------------------------------------------
def test_per_frame_budget_always_allows() -> None:
    clock = FakeClock()
    throttle = Throttle(PER_FRAME_HZ, clock=clock)
    assert throttle.hz == PER_FRAME_HZ
    assert throttle.interval_s == 0.0
    for _ in range(5):
        assert throttle.allow() is True
        throttle.record()
    # The clock never moved, so a finite budget would have blocked long ago.
    assert clock.now == 0.0


def test_finite_budget_samples_at_configured_rate() -> None:
    clock = FakeClock()
    throttle = Throttle(2.0, clock=clock)
    assert throttle.interval_s == pytest.approx(0.5)

    assert throttle.allow() is True
    throttle.record()
    assert throttle.allow() is False
    clock.advance(0.25)
    assert throttle.allow() is False
    clock.advance(0.25)
    assert throttle.allow() is True
    throttle.record()
    assert throttle.allow() is False


def test_allow_is_pure_until_recorded() -> None:
    clock = FakeClock()
    throttle = Throttle(2.0, clock=clock)
    # allow() must not consume the budget on its own.
    clock.advance(10.0)
    assert throttle.allow() is True
    assert throttle.allow() is True
    assert throttle.allow(now=999.0) is True


def test_record_accepts_explicit_time() -> None:
    throttle = Throttle(1.0, clock=FakeClock())
    throttle.record(now=100.0)
    assert throttle.allow(now=100.5) is False
    assert throttle.allow(now=101.0) is True


def test_throttle_rejects_invalid_budgets() -> None:
    with pytest.raises(ValueError):
        Throttle(0.0)
    with pytest.raises(ValueError):
        Throttle(-1.0)
    with pytest.raises(ValueError):
        Throttle(float("nan"))
    with pytest.raises(TypeError):
        Throttle("fast")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# ScreenCapture
# ----------------------------------------------------------------------
def test_capture_module_import_loads_no_optional_stack() -> None:
    """Importing capture.py must not pull mss/cv2/pytesseract/ultralytics."""
    import subprocess
    import sys as _sys

    code = (
        "import sys;"
        "import wow_bot.perception.capture as c;"
        "c.ScreenCapture(idle_fps=10, combat_fps=30);"
        "names=('mss','cv2','pytesseract','ultralytics','torch');"
        "loaded=[n for n in names if sys.modules.get(n) is not None];"
        "assert not loaded, loaded;"
        "print('OK')"
    )
    result = subprocess.run(
        [_sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    assert "OK" in result.stdout


def test_capture_set_mode_switches_rate() -> None:
    capture = ScreenCapture(idle_fps=10, combat_fps=30)
    assert capture.set_mode("combat") == 30
    assert capture.current_fps == 30
    assert capture.set_mode("idle") == 10
    assert capture.current_fps == 10
    # An unknown mode is idle, never an error: the caller's combat verdict
    # is what drives this.
    assert capture.set_mode("nonsense") == 10


def test_capture_rejects_bad_rates() -> None:
    with pytest.raises(ValueError):
        ScreenCapture(idle_fps=0)
    with pytest.raises(ValueError):
        ScreenCapture(combat_fps=-1)
    with pytest.raises(ValueError):
        ScreenCapture(pool_size=0)


def test_capture_grab_without_device_raises_dependency_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import sys

    monkeypatch.setitem(sys.modules, "mss", None)
    capture = ScreenCapture()
    with pytest.raises(ImportError):
        capture.grab()


def test_capture_stop_is_safe_without_start() -> None:
    capture = ScreenCapture()
    capture.stop()
    assert capture.running is False
    assert capture._sct is None


def test_capture_cooldown_uses_injected_clock_not_time_time() -> None:
    """The contract forbids ``time.time()`` inside readers/capture."""
    import ast
    from pathlib import Path

    path = Path("src/wow_bot/perception/capture.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden_calls: list[str] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if (
            isinstance(func, ast.Attribute)
            and isinstance(func.value, ast.Name)
            and func.value.id == "time"
            and func.attr in {"time", "perf_counter"}
        ):
            forbidden_calls.append(func.attr)
    assert forbidden_calls == [], f"wall-clock calls found: {forbidden_calls}"
