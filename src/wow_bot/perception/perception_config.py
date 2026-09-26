"""Standalone perception configuration loader (T-FIX-29).

The lab ``Config`` (``src/wow_bot/config.py``) is frozen and rejects
unknown keys, so perception settings live in their own TOML file,
``config/perception.example.toml``, with their own loader and error
type — the same pattern as farm profiles (``farm/profile.py``) and
rotations (``combat/rotation.py`` + ``scripts/lab/full_soak.py``).

Every hardcoded value from ``docs/lab_phase/HAMBERGER_PORT_PLAN.md``
§6.3 becomes a validated key here, the T-FIX-30 channels
(``[pose]``, ``[reaction]``, ``[proximity]``) add theirs under the same
rules, and the T-FIX-31 UI panel channels (``[bag]``, ``[xp]``,
``[durability]``, ``[cast]``, ``[loot]``) do too. The loader only validates *shape* (types, ranges, unknown keys); it
never checks that weight/template files exist or that binaries are
installed, so MOCK_MODE and CI load the example without Tesseract, YOLO
weights, or template PNGs present. Existence is enforced fail-closed at
reader construction time by the T-FIX-27 readers via ``perception.deps``
guards.
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
        "pose",
        "reaction",
        "proximity",
        "bag",
        "xp",
        "durability",
        "cast",
        "loot",
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

#: T-FIX-30 world-pose channel: OCR of an addon coordinate frame (ADR-003).
ALLOWED_POSE_KEYS: frozenset[str] = frozenset(
    {
        "coordinate_roi",
        "min_confidence",
        "sampling_hz",
    }
)

#: T-FIX-30 target-reaction channel: nameplate text colour.
ALLOWED_REACTION_KEYS: frozenset[str] = frozenset(
    {
        "nameplate_roi",
        "min_pixels",
        "dominance_thresh",
        "sampling_hz",
    }
)

#: T-FIX-30 target-distance channel: nameplate HP-bar width -> yards.
ALLOWED_PROXIMITY_KEYS: frozenset[str] = frozenset(
    {
        "reference_width_px",
        "reference_distance_yd",
        "min_confidence",
        "sampling_hz",
    }
)

#: T-FIX-31 Bag frame channel: slot grid -> inventory_count / inventory_max.
ALLOWED_BAG_KEYS: frozenset[str] = frozenset(
    {
        "grid_origin",
        "columns",
        "rows",
        "slot_size",
        "gap",
        "empty_slot_template",
        "occupied_slot_template",
        "min_confidence",
        "sampling_hz",
    }
)

#: T-FIX-31 XP bar channel: fill fraction + level label -> level_or_xp.
ALLOWED_XP_KEYS: frozenset[str] = frozenset(
    {
        "bar_roi",
        "level_roi",
        "min_confidence",
        "sampling_hz",
    }
)

#: T-FIX-31 Character frame channel: per-slot durability -> durability_fraction.
ALLOWED_DURABILITY_KEYS: frozenset[str] = frozenset(
    {
        "slot_rois",
        "min_score",
        "quorum_fraction",
        "min_confidence",
        "sampling_hz",
    }
)

#: T-FIX-31 Enemy cast bar channel: cast bar -> incoming_casts.
ALLOWED_CAST_KEYS: frozenset[str] = frozenset(
    {
        "cast_roi",
        "border_roi",
        "spell_ids",
        "border_interruptible",
        "min_confidence",
        "sampling_hz",
        "max_remaining_s",
    }
)

#: T-FIX-31 Lootable-corpse channel: loot sparkle -> target_is_lootable.
ALLOWED_LOOT_KEYS: frozenset[str] = frozenset(
    {
        "sparkle_roi",
        "min_pixels",
        "dominance_thresh",
        "min_confidence",
        "sampling_hz",
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
    pose_coordinate_roi: tuple[int, int, int, int]
    pose_min_confidence: float
    pose_sampling_hz: float
    reaction_nameplate_roi: tuple[int, int, int, int]
    reaction_min_pixels: int
    reaction_dominance_thresh: float
    reaction_sampling_hz: float
    proximity_reference_width_px: float
    proximity_reference_distance_yd: float
    proximity_min_confidence: float
    proximity_sampling_hz: float
    # T-FIX-31 UI panel channels (docs/PERCEPTION.md §2.1).
    bag_grid_origin: tuple[int, int]
    bag_columns: int
    bag_rows: int
    bag_slot_size: tuple[int, int]
    bag_gap: int
    bag_empty_slot_template: str
    bag_occupied_slot_template: str
    bag_min_confidence: float
    bag_sampling_hz: float
    xp_bar_roi: tuple[int, int, int, int]
    xp_level_roi: tuple[int, int, int, int] | None
    xp_min_confidence: float
    xp_sampling_hz: float
    durability_slot_rois: tuple[tuple[int, int, int, int], ...]
    durability_min_score: float
    durability_quorum_fraction: float
    durability_min_confidence: float
    durability_sampling_hz: float
    cast_roi: tuple[int, int, int, int]
    cast_border_roi: tuple[int, int, int, int] | None
    cast_spell_ids: dict[str, str]
    cast_border_interruptible: bool
    cast_min_confidence: float
    cast_sampling_hz: float
    cast_max_remaining_s: float
    loot_sparkle_roi: tuple[int, int, int, int]
    loot_min_pixels: int
    loot_dominance_thresh: float
    loot_min_confidence: float
    loot_sampling_hz: float


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


def _as_non_negative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise PerceptionConfigError(f"{name} must be an int >= 0")
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


def _as_positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise PerceptionConfigError(f"{name} must be a number")
    result = float(value)
    if result <= 0.0:
        raise PerceptionConfigError(f"{name} must be > 0")
    return result


def _as_non_empty_str(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise PerceptionConfigError(f"{name} must be a non-empty string")
    return value


def _as_point(value: Any, name: str) -> tuple[int, int]:
    if (
        not isinstance(value, list)
        or len(value) != 2
        or any(isinstance(v, bool) or not isinstance(v, int) for v in value)
    ):
        raise PerceptionConfigError(f"{name} must be [x, y] ints")
    x, y = value
    if x < 0 or y < 0:
        raise PerceptionConfigError(f"{name} needs x,y >= 0")
    return (x, y)


def _as_slot_rois(value: Any, name: str) -> tuple[tuple[int, int, int, int], ...]:
    """Validate the Character frame's explicit per-slot regions.

    The layout is explicit rather than a grid, because the character sheet's
    slot positions are not a uniform grid (T-FIX-31 design note); at least
    one slot is required, since a durability mean over zero slots would be a
    fabricated ``1.0``.
    """
    if not isinstance(value, list) or not value:
        raise PerceptionConfigError(f"{name} must be a non-empty list of ROIs")
    return tuple(_as_roi(entry, f"{name}[{index}]") for index, entry in enumerate(value))


def _as_bool(value: Any, name: str) -> bool:
    if not isinstance(value, bool):
        raise PerceptionConfigError(f"{name} must be a boolean")
    return value


def _as_optional_roi(value: Any, name: str) -> tuple[int, int, int, int] | None:
    """Validate an ROI that may be present or explicitly absent (``None``)."""
    if value is None:
        return None
    if isinstance(value, list) and not value:
        return None
    return _as_roi(value, name)


def _as_spell_ids(value: Any, name: str) -> dict[str, str]:
    """Validate the spell-name -> spell-id table.

    An empty table is allowed and means "no spell id is resolvable", which
    makes the cast reader drop every cast rather than emit a guessed id
    (``docs/PERCEPTION.md`` §2.1, Enemy cast bar).
    """
    if not isinstance(value, dict):
        raise PerceptionConfigError(f"{name} must be a table of name = id pairs")
    table: dict[str, str] = {}
    for raw_name, raw_id in value.items():
        if not isinstance(raw_name, str) or not raw_name.strip():
            raise PerceptionConfigError(f"{name} keys must be non-empty strings")
        if isinstance(raw_id, bool) or not isinstance(raw_id, (str, int)):
            raise PerceptionConfigError(f"{name}.{raw_name} must be a string or int id")
        table[raw_name.strip().lower()] = str(raw_id)
    return table


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
    pose = _require_section(data, "pose")
    reaction = _require_section(data, "reaction")
    proximity = _require_section(data, "proximity")
    bag = _require_section(data, "bag")
    xp = _require_section(data, "xp")
    durability = _require_section(data, "durability")
    cast = _require_section(data, "cast")
    loot = _require_section(data, "loot")

    _check_keys(perception, "perception", ALLOWED_PERCEPTION_KEYS)
    _check_keys(capture, "capture", ALLOWED_CAPTURE_KEYS)
    _check_keys(bars, "bars", ALLOWED_BARS_KEYS)
    _check_keys(combat, "combat", ALLOWED_COMBAT_KEYS)
    _check_keys(target, "target", ALLOWED_TARGET_KEYS)
    _check_keys(enemies, "enemies", ALLOWED_ENEMIES_KEYS)
    _check_keys(events, "events", ALLOWED_EVENTS_KEYS)
    _check_keys(minimap, "minimap", ALLOWED_MINIMAP_KEYS)
    _check_keys(pose, "pose", ALLOWED_POSE_KEYS)
    _check_keys(reaction, "reaction", ALLOWED_REACTION_KEYS)
    _check_keys(proximity, "proximity", ALLOWED_PROXIMITY_KEYS)
    _check_keys(bag, "bag", ALLOWED_BAG_KEYS)
    _check_keys(xp, "xp", ALLOWED_XP_KEYS)
    _check_keys(durability, "durability", ALLOWED_DURABILITY_KEYS)
    _check_keys(cast, "cast", ALLOWED_CAST_KEYS)
    _check_keys(loot, "loot", ALLOWED_LOOT_KEYS)

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
        pose_coordinate_roi=_as_roi(
            pose.get("coordinate_roi"), "[pose].coordinate_roi"
        ),
        pose_min_confidence=_as_unit_float(
            pose.get("min_confidence"), "[pose].min_confidence"
        ),
        pose_sampling_hz=_as_sampling_hz(pose.get("sampling_hz"), "[pose].sampling_hz"),
        reaction_nameplate_roi=_as_roi(
            reaction.get("nameplate_roi"), "[reaction].nameplate_roi"
        ),
        reaction_min_pixels=_as_positive_int(
            reaction.get("min_pixels"), "[reaction].min_pixels"
        ),
        reaction_dominance_thresh=_as_unit_float(
            reaction.get("dominance_thresh"), "[reaction].dominance_thresh"
        ),
        reaction_sampling_hz=_as_sampling_hz(
            reaction.get("sampling_hz"), "[reaction].sampling_hz"
        ),
        proximity_reference_width_px=_as_positive_float(
            proximity.get("reference_width_px"), "[proximity].reference_width_px"
        ),
        proximity_reference_distance_yd=_as_positive_float(
            proximity.get("reference_distance_yd"),
            "[proximity].reference_distance_yd",
        ),
        proximity_min_confidence=_as_unit_float(
            proximity.get("min_confidence"), "[proximity].min_confidence"
        ),
        proximity_sampling_hz=_as_sampling_hz(
            proximity.get("sampling_hz"), "[proximity].sampling_hz"
        ),
        bag_grid_origin=_as_point(bag.get("grid_origin"), "[bag].grid_origin"),
        bag_columns=_as_positive_int(bag.get("columns"), "[bag].columns"),
        bag_rows=_as_positive_int(bag.get("rows"), "[bag].rows"),
        bag_slot_size=_as_resolution(bag.get("slot_size"), "[bag].slot_size"),
        bag_gap=_as_non_negative_int(bag.get("gap"), "[bag].gap"),
        bag_empty_slot_template=_as_non_empty_str(
            bag.get("empty_slot_template"), "[bag].empty_slot_template"
        ),
        bag_occupied_slot_template=_as_non_empty_str(
            bag.get("occupied_slot_template"), "[bag].occupied_slot_template"
        ),
        bag_min_confidence=_as_unit_float(
            bag.get("min_confidence"), "[bag].min_confidence"
        ),
        bag_sampling_hz=_as_sampling_hz(bag.get("sampling_hz"), "[bag].sampling_hz"),
        xp_bar_roi=_as_roi(xp.get("bar_roi"), "[xp].bar_roi"),
        xp_level_roi=_as_optional_roi(xp.get("level_roi"), "[xp].level_roi"),
        xp_min_confidence=_as_unit_float(
            xp.get("min_confidence"), "[xp].min_confidence"
        ),
        xp_sampling_hz=_as_sampling_hz(xp.get("sampling_hz"), "[xp].sampling_hz"),
        durability_slot_rois=_as_slot_rois(
            durability.get("slot_rois"), "[durability].slot_rois"
        ),
        durability_min_score=_as_unit_float(
            durability.get("min_score"), "[durability].min_score"
        ),
        durability_quorum_fraction=_as_unit_float(
            durability.get("quorum_fraction"), "[durability].quorum_fraction"
        ),
        durability_min_confidence=_as_unit_float(
            durability.get("min_confidence"), "[durability].min_confidence"
        ),
        durability_sampling_hz=_as_sampling_hz(
            durability.get("sampling_hz"), "[durability].sampling_hz"
        ),
        cast_roi=_as_roi(cast.get("cast_roi"), "[cast].cast_roi"),
        cast_border_roi=_as_optional_roi(cast.get("border_roi"), "[cast].border_roi"),
        cast_spell_ids=_as_spell_ids(cast.get("spell_ids"), "[cast].spell_ids"),
        cast_border_interruptible=_as_bool(
            cast.get("border_interruptible"), "[cast].border_interruptible"
        ),
        cast_min_confidence=_as_unit_float(
            cast.get("min_confidence"), "[cast].min_confidence"
        ),
        cast_sampling_hz=_as_sampling_hz(cast.get("sampling_hz"), "[cast].sampling_hz"),
        cast_max_remaining_s=_as_positive_float(
            cast.get("max_remaining_s"), "[cast].max_remaining_s"
        ),
        loot_sparkle_roi=_as_roi(loot.get("sparkle_roi"), "[loot].sparkle_roi"),
        loot_min_pixels=_as_positive_int(
            loot.get("min_pixels"), "[loot].min_pixels"
        ),
        loot_dominance_thresh=_as_unit_float(
            loot.get("dominance_thresh"), "[loot].dominance_thresh"
        ),
        loot_min_confidence=_as_unit_float(
            loot.get("min_confidence"), "[loot].min_confidence"
        ),
        loot_sampling_hz=_as_sampling_hz(loot.get("sampling_hz"), "[loot].sampling_hz"),
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
