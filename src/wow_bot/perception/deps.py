"""Guarded optional imports for the real-perception stack (T-FIX-29).

MOCK_MODE and the default suite import without the heavy vision packages.
Each require helper lazily imports at call time and raises with a remedy.
Module import has no side effects: no cv2/mss/pytesseract/ultralytics import
happens here, no tesseract_cmd is assigned, no YOLO settings are touched.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = [
    "YOLO_EXTRA_NAME",
    "YOLO_OFFLINE_ENV",
    "PerceptionDependencyError",
    "apply_yolo_offline_env",
    "has_cv2",
    "has_mss",
    "has_pytesseract",
    "has_tesseract_binary",
    "has_ultralytics",
    "require_cv2",
    "require_mss",
    "require_pytesseract",
    "require_tesseract_binary",
    "require_ultralytics",
    "verify_yolo_offline",
]


class PerceptionDependencyError(ImportError):
    """Missing optional perception dependency with a named install remedy."""


YOLO_EXTRA_NAME = "yolo"

YOLO_OFFLINE_ENV: dict[str, str] = {
    "YOLO_OFFLINE": "1",
    "YOLO_VERBOSE": "False",
    "WANDB_DISABLED": "true",
    "WANDB_MODE": "offline",
    "WANDB_ANONYMOUS": "must",
}


def apply_yolo_offline_env() -> dict[str, str]:
    """Force the YOLO offline env and return the applied mapping."""
    for key, value in YOLO_OFFLINE_ENV.items():
        os.environ[key] = value
    return dict(YOLO_OFFLINE_ENV)


def verify_yolo_offline() -> None:
    """Raise when the offline env mapping has not been applied."""
    missing = [key for key in YOLO_OFFLINE_ENV if key not in os.environ]
    if missing:
        raise PerceptionDependencyError(
            "YOLO offline environment is not applied "
            f"(missing: {', '.join(sorted(missing))}). "
            "Call perception.deps.apply_yolo_offline_env() before "
            "constructing ultralytics.YOLO."
        )


def _import_or_raise(module_name: str, remedy: str) -> Any:
    """Import module_name lazily, raising with remedy when absent."""
    try:
        return __import__(module_name)
    except ImportError as exc:
        message = f"{module_name} is required for real perception "
        message += f"but is not installed. {remedy}"
        raise PerceptionDependencyError(message) from exc


def _is_importable(module_name: str) -> bool:
    """Return True when module_name imports, False on ImportError."""
    try:
        __import__(module_name)
    except ImportError:
        return False
    return True


def require_cv2() -> Any:
    """Return cv2 or raise with the install remedy."""
    return _import_or_raise(
        "cv2",
        "Install the base dependencies (opencv-python-headless).",
    )


def has_cv2() -> bool:
    """Return True when cv2 is importable without raising."""
    return _is_importable("cv2")


def require_mss() -> Any:
    """Return mss or raise with the install remedy."""
    return _import_or_raise("mss", "Install the base dependencies (mss).")


def has_mss() -> bool:
    """Return True when mss is importable without raising."""
    return _is_importable("mss")


def require_pytesseract() -> Any:
    """Return pytesseract or raise with the install remedy."""
    return _import_or_raise(
        "pytesseract",
        "Install the base dependencies (pytesseract) plus the separate "
        "Tesseract system binary; MOCK_MODE never requires either.",
    )


def has_pytesseract() -> bool:
    """Return True when pytesseract is importable without raising."""
    return _is_importable("pytesseract")


def require_tesseract_binary() -> str:
    """Return the tesseract path or raise with the install remedy."""
    from shutil import which

    found = which("tesseract")
    if found is None:
        raise PerceptionDependencyError(
            "Tesseract system binary is required for real perception OCR "
            "but was not found on PATH. Install Tesseract OCR separately; "
            "MOCK_MODE never requires it."
        )
    return found


def has_tesseract_binary() -> bool:
    """Return True when a tesseract binary is found on PATH."""
    from shutil import which

    return which("tesseract") is not None


def require_ultralytics() -> Any:
    """Return ultralytics or raise with the optional-extra remedy."""
    verify_yolo_offline()
    return _import_or_raise(
        "ultralytics",
        f"Install the optional {YOLO_EXTRA_NAME!r} extra "
        "(uv sync --extra yolo); MOCK_MODE never requires it.",
    )


def has_ultralytics() -> bool:
    """Return True when ultralytics is importable without raising."""
    return _is_importable("ultralytics")
