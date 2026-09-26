"""Tests for T-FIX-29: guarded perception dependencies.

MOCK_MODE must import without cv2/mss/pytesseract/ultralytics installed.
Missing optionals raise PerceptionDependencyError with a named remedy.
"""

from __future__ import annotations

import ast
import os
import sys
from pathlib import Path

import pytest

from wow_bot.perception.deps import (
    YOLO_OFFLINE_ENV,
    PerceptionDependencyError,
    apply_yolo_offline_env,
    has_cv2,
    has_mss,
    has_pytesseract,
    has_tesseract_binary,
    has_ultralytics,
    require_cv2,
    require_mss,
    require_pytesseract,
    require_tesseract_binary,
    require_ultralytics,
    verify_yolo_offline,
)


def _hide_modules(monkeypatch: pytest.MonkeyPatch, *names: str) -> None:
    for name in names:
        monkeypatch.setitem(sys.modules, name, None)


def test_error_is_import_error() -> None:
    assert issubclass(PerceptionDependencyError, ImportError)


def test_import_has_no_side_effects() -> None:
    import subprocess
    import sys as _sys

    code = (
        "import sys;"
        "import wow_bot.perception.deps as d;"
        "names=('cv2','mss','pytesseract','ultralytics');"
        "missing=[n for n in names if n in sys.modules "
        "and sys.modules[n] is not None];"
        "assert not missing, missing;"
        "assert 'tesseract_cmd' not in dir(d);"
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


def test_missing_base_deps_raise_with_remedy(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _hide_modules(monkeypatch, "cv2", "mss", "pytesseract")
    with pytest.raises(PerceptionDependencyError, match="cv2"):
        require_cv2()
    with pytest.raises(PerceptionDependencyError, match="mss"):
        require_mss()
    with pytest.raises(PerceptionDependencyError, match="pytesseract"):
        require_pytesseract()
    assert has_cv2() is False
    assert has_mss() is False
    assert has_pytesseract() is False


def test_missing_tesseract_binary_raises(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("shutil.which", lambda _name: None)
    with pytest.raises(PerceptionDependencyError, match="Tesseract"):
        require_tesseract_binary()
    assert has_tesseract_binary() is False


def test_yolo_requires_offline_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in YOLO_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    _hide_modules(monkeypatch, "ultralytics")
    with pytest.raises(PerceptionDependencyError, match="offline"):
        require_ultralytics()
    with pytest.raises(PerceptionDependencyError, match="offline"):
        verify_yolo_offline()


def test_yolo_offline_env_applies(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    for key in YOLO_OFFLINE_ENV:
        monkeypatch.delenv(key, raising=False)
    applied = apply_yolo_offline_env()
    assert applied == YOLO_OFFLINE_ENV
    for key, value in YOLO_OFFLINE_ENV.items():
        assert os.environ[key] == value
    verify_yolo_offline()
    _hide_modules(monkeypatch, "ultralytics")
    with pytest.raises(PerceptionDependencyError, match="ultralytics"):
        require_ultralytics()
    assert has_ultralytics() is False


def test_deps_avoids_forbidden_imports() -> None:
    path = Path("src/wow_bot/perception/deps.py")
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    forbidden = {
        "aiosqlite",
        "asyncio",
        "random",
        "threading",
        "time",
        "cv2",
        "mss",
        "pytesseract",
        "ultralytics",
        "torch",
        "wow_bot.main",
        "wow_bot.lab.runner_v2",
    }
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules = [alias.name.split(".")[0] for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            modules = [(node.module or "").split(".")[0]]
        else:
            continue
        for module in modules:
            assert module not in forbidden, f"deps.py imports {module}"
            assert "llm" not in module.lower()
            assert "tesseract_cmd" not in module
