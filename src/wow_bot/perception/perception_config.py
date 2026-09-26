"""Standalone perception configuration loader (T-FIX-29).

The lab ``Config`` (``src/wow_bot/config.py``) is frozen and rejects
unknown keys, so perception settings live in their own TOML file,
``config/perception.example.toml``, with their own loader and error
type — the same pattern as farm profiles (``farm/profile.py``) and
rotations (``combat/rotation.py`` + ``scripts/lab/full_soak.py``).

Every hardcoded value from ``docs/lab_phase/HAMBERGER_PORT_PLAN.md``
§6.3 becomes a validated key here. The loader only validates *shape*
(types, ranges, unknown keys); it never checks that weight/template
files exist or that binaries are installed, so MOCK_MODE and CI load
the example without Tesseract, YOLO weights, or template PNGs present.
Existence is enforced fail-closed at reader construction time by the
future T-FIX-27 readers via ``perception.deps`` guards.
"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

__all__ = [
    "PERCEPTION_SCHEMA_VERSION",
    "PerceptionConfig",
    "PerceptionConfigError",
    "load_perception_config",
    "load_perception_config_from_dict",
]


class PerceptionConfigError(Exception):
    """Raised when a perception config file fails validation or parsing."""


PERCEPTION_SCHEMA_VERSION: int = 1


ALLOWED_TOP_LEVEL_KEYS: frozenset[str] = frozenset(
    {
        "perception",
        "capture",
        "bars",
        "combat",
        "target",
        "enemies",
        "events",
        "minimap",
    }
)

ALLOWED_PERCEPTION_KEYS: frozenset[str] = frozenset(
    {
        "schema_version",
        "tesseract_cmd",
        "calibrated_resolution",
    }
)

ALLOWED_CAPTURE_KEYS: frozenset[str] = frozenset(
    {
        "idle_fps",
        "combat_fps",
    }
)

ALLOWED_BARS_KEYS: frozenset[str] = frozenset(
    {
        "hp_roi",
        "mana_roi",
        "sampling_hz",
    }
)

ALLOWED_COMBAT_KEYS: frozenset[str] = frozenset(
    {
        "edge_size",
        "red_ratio_thresh",
        "cooldown_s",
        "sampling_hz",
    }
)

ALLOWED_TARGET_KEYS: frozenset[str] = frozenset(
    {
        "name_template",
        "frame_template",
        "known_enemies",
        "match_thresh",
        "confirm_thresh",
        "ocr_thresh",
        "ocr_confirm_thresh",
        "sampling_hz",
        "name_refresh_frames",
    }
)

ALLOWED_ENEMIES_KEYS: frozenset[str] = frozenset(
    {
        "yolo_weights",
        "confidence",
        "sampling_hz",
    }
)

ALLOWED_EVENTS_KEYS: frozenset[str] = frozenset(
    {
        "combat_region",
        "chat_region",
        "sampling_hz",
    }
)

ALLOWED_MINIMAP_KEYS: frozenset[str] = frozenset(
    {
        "minimap_roi",
        "arrow_template",
        "smoothing_frames",
        "sampling_hz",
        "match_thresh",
    }
)


@dataclass(frozen=True)
class PerceptionConfig:
    """Validated real-perception settings; paths are strings, not checks."""

    schema_version: int
    tesseract_cmd: str
    calibrated_resolution: tuple[int, int]
    idle_fps: int
    combat_fps: int
    hp_roi: tuple[int, int, int, int]
    mana_roi: tuple[int, int, int, int]
    bars_sampling_hz: float
    combat_edge_size: int
    combat_red_ratio_thresh: float
    combat_cooldown_s: float
    combat_sampling_hz: float
    target_name_template: str
    target_frame_template: str
    known_enemies: tuple[str, ...]
    target_match_thresh: float
    target_confirm_thresh: float
    target_ocr_thresh: float
    target_ocr_confirm_thresh: float
    target_sampling_hz: float
    target_name_refresh_frames: int
    yolo_weights: str
    yolo_confidence: float
    enemies_sampling_hz: float
    combat_region: tuple[int, int, int, int]
    chat_region: tuple[int, int, int, int]
    events_sampling_hz: float
    minimap_roi: tuple[int, int, int, int]
    minimap_arrow_template: str
    minimap_smoothing_frames: int
    minimap_sampling_hz: float
    minimap_match_thresh: float


def _require_section(data: dict[str, Any], name: str) -> dict[str, Any]:
    section = data.get(name)
    if not isinstance(section, dict):
        raise PerceptionConfigError(f"Missing required section [{name}]")
    return section


def _check_keys(section: dict[str, Any], name: str, allowed: frozenset[str]) -> None:
    for key in section:
        if key not in allowed:
            raise PerceptionConfigError(f"Unknown key [{name}].{key}")


def _as_roi(value: Any, name: str) -> tuple[int, int, int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 4
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        raise PerceptionConfigError(f"{name} must be [x, y, w, h] ints")
    x, y, w, h = value
    if x < 0 or y < 0 or w <= 0 or h <= 0:
        raise PerceptionConfigError(f"{name} needs x,y >= 0 and w,h > 0")
    return (x, y, w, h)


def _as_resolution(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        raise PerceptionConfigError(f"{name} must be [width, height] ints")
    width, height = value
    if width <= 0 or height <= 0:
        raise PerceptionConfigError(f"{name} needs width,height > 0")
    return (width, height)


def _as_positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise PerceptionConfigError(f"{name} must be a positive int")
    return value


def _as_non_negative_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerceptionConfigError(f"{name} must be a number")
    result = float(value)
    if result < 0.0:
        raise PerceptionConfigError(f"{name} must be >= 0")
    return result


def _as_unit_float(value: Any, name: str) -> float:
    result = _as_non_negative_float(value, name)
    if result > 1.0:
        raise PerceptionConfigError(f"{name} must be within [0, 1]")
    return result


def _as_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerceptionConfigError(f"{name} must be a non-empty string")
    return value


def _as_sampling_hz(value: Any, name: str) -> float:
    """Validate a per-channel sampling rate.

    ``inf`` is the explicit "run every frame" budget for the cheap
    channels (plan doc §6.6); NaN and non-positive rates are rejected
    because they would silently disable a channel.
    """
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerceptionConfigError(f"{name} must be a number")
    result = float(value)
    if math.isnan(result):
        raise PerceptionConfigError(f"{name} must not be NaN")
    if result <= 0.0:
        raise PerceptionConfigError(f"{name} must be > 0")
    return result


def load_perception_config_from_dict(data: dict[str, Any]) -> PerceptionConfig:
    """Validate a parsed TOML mapping into a frozen PerceptionConfig."""
    if not isinstance(data, dict):
        raise PerceptionConfigError("TOML root must be a table")
    for key in data:
        if key not in ALLOWED_TOP_LEVEL_KEYS:
            raise PerceptionConfigError(f"Unknown top-level key: '{key}'")

    perception = _require_section(data, "perception")
    capture = _require_section(data, "capture")
    bars = _require_section(data, "bars")
    combat = _require_section(data, "combat")
    target = _require_section(data, "target")
    enemies = _require_section(data, "enemies")
    events = _require_section(data, "events")
    minimap = _require_section(data, "minimap")

    _check_keys(perception, "perception", ALLOWED_PERCEPTION_KEYS)
    _check_keys(capture, "capture", ALLOWED_CAPTURE_KEYS)
    _check_keys(bars, "bars", ALLOWED_BARS_KEYS)
    _check_keys(combat, "combat", ALLOWED_COMBAT_KEYS)
    _check_keys(target, "target", ALLOWED_TARGET_KEYS)
    _check_keys(enemies, "enemies", ALLOWED_ENEMIES_KEYS)
    _check_keys(events, "events", ALLOWED_EVENTS_KEYS)
    _check_keys(minimap, "minimap", ALLOWED_MINIMAP_KEYS)

    version = perception.get("schema_version")
    if (
        isinstance(version, bool)
        or not isinstance(version, int)
        or version != PERCEPTION_SCHEMA_VERSION
    ):
        raise PerceptionConfigError(
            f"[perception].schema_version must equal {PERCEPTION_SCHEMA_VERSION}"
        )

    known = target.get("known_enemies")
    if not isinstance(known, list) or not known:
        raise PerceptionConfigError("[target].known_enemies must be a non-empty list")
    for entry in known:
        if not isinstance(entry, str) or not entry.strip() or entry != entry.strip():
            raise PerceptionConfigError(
                "[target].known_enemies entries must be stripped non-empty strings"
            )

    return PerceptionConfig(
        schema_version=version,
        tesseract_cmd=_as_non_empty_str(
            perception.get("tesseract_cmd"), "[perception].tesseract_cmd"
        ),
        calibrated_resolution=_as_resolution(
            perception.get("calibrated_resolution"),
            "[perception].calibrated_resolution",
        ),
        idle_fps=_as_positive_int(capture.get("idle_fps"), "[capture].idle_fps"),
        combat_fps=_as_positive_int(capture.get("combat_fps"), "[capture].combat_fps"),
        hp_roi=_as_roi(bars.get("hp_roi"), "[bars].hp_roi"),
        mana_roi=_as_roi(bars.get("mana_roi"), "[bars].mana_roi"),
        bars_sampling_hz=_as_sampling_hz(bars.get("sampling_hz"), "[bars].sampling_hz"),
        combat_edge_size=_as_positive_int(combat.get("edge_size"), "[combat].edge_size"),
        combat_red_ratio_thresh=_as_unit_float(
            combat.get("red_ratio_thresh"), "[combat].red_ratio_thresh"
        ),
        combat_cooldown_s=_as_non_negative_float(combat.get("cooldown_s"), "[combat].cooldown_s"),
        combat_sampling_hz=_as_sampling_hz(combat.get("sampling_hz"), "[combat].sampling_hz"),
        target_name_template=_as_non_empty_str(
            target.get("name_template"), "[target].name_template"
        ),
        target_frame_template=_as_non_empty_str(
            target.get("frame_template"), "[target].frame_template"
        ),
        known_enemies=tuple(known),
        target_match_thresh=_as_unit_float(target.get("match_thresh"), "[target].match_thresh"),
        target_confirm_thresh=_as_unit_float(
            target.get("confirm_thresh"), "[target].confirm_thresh"
        ),
        target_ocr_thresh=_as_unit_float(target.get("ocr_thresh"), "[target].ocr_thresh"),
        target_ocr_confirm_thresh=_as_unit_float(
            target.get("ocr_confirm_thresh"), "[target].ocr_confirm_thresh"
        ),
        target_sampling_hz=_as_sampling_hz(target.get("sampling_hz"), "[target].sampling_hz"),
        target_name_refresh_frames=_as_positive_int(
            target.get("name_refresh_frames"), "[target].name_refresh_frames"
        ),
        yolo_weights=_as_non_empty_str(enemies.get("yolo_weights"), "[enemies].yolo_weights"),
        yolo_confidence=_as_unit_float(enemies.get("confidence"), "[enemies].confidence"),
        enemies_sampling_hz=_as_sampling_hz(
            enemies.get("sampling_hz"), "[enemies].sampling_hz"
        ),
        combat_region=_as_roi(events.get("combat_region"), "[events].combat_region"),
        chat_region=_as_roi(events.get("chat_region"), "[events].chat_region"),
        events_sampling_hz=_as_sampling_hz(events.get("sampling_hz"), "[events].sampling_hz"),
        minimap_roi=_as_roi(minimap.get("minimap_roi"), "[minimap].minimap_roi"),
        minimap_arrow_template=_as_non_empty_str(
            minimap.get("arrow_template"), "[minimap].arrow_template"
        ),
        minimap_smoothing_frames=_as_positive_int(
            minimap.get("smoothing_frames"), "[minimap].smoothing_frames"
        ),
        minimap_sampling_hz=_as_sampling_hz(
            minimap.get("sampling_hz"), "[minimap].sampling_hz"
        ),
        minimap_match_thresh=_as_unit_float(
            minimap.get("match_thresh"), "[minimap].match_thresh"
        ),
    )


def load_perception_config(path: Path) -> PerceptionConfig:
    """Load and validate a perception TOML file at path."""
    if not isinstance(path, Path):
        path = Path(path)
    if not path.is_file():
        raise PerceptionConfigError(f"Perception config file not found: {path}")
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except tomllib.TOMLDecodeError as exc:
        raise PerceptionConfigError(f"TOML parse error in {path}: {exc}") from exc
    except OSError as exc:
        raise PerceptionConfigError(f"Failed to read {path}: {exc}") from exc
    return load_perception_config_from_dict(data)
