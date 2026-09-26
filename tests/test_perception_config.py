"""Tests for T-FIX-29 perception configuration placement.

The frozen lab Config rejects unknown keys, so perception settings live
in config/perception.example.toml with the perception_config loader.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from wow_bot.perception.perception_config import (
    PERCEPTION_SCHEMA_VERSION,
    PerceptionConfig,
    PerceptionConfigError,
    load_perception_config,
    load_perception_config_from_dict,
)


def test_example_loads_without_touching_disk_assets() -> None:
    config = load_perception_config(Path("config/perception.example.toml"))
    assert isinstance(config, PerceptionConfig)
    assert config.schema_version == PERCEPTION_SCHEMA_VERSION == 1
    assert config.hp_roi == (514, 771, 123, 18)
    assert config.mana_roi == (514, 793, 123, 17)
    assert config.minimap_roi == (1705, 33, 206, 217)
    assert config.target_name_template == "models/target_name_template.png"
    assert config.minimap_arrow_template == "models/arrow_template.png"
    assert config.yolo_weights == "models/weights/best.pt"
    assert config.yolo_confidence == 0.5
    assert config.combat_edge_size == 150
    assert config.combat_red_ratio_thresh == 0.02
    assert config.combat_cooldown_s == 1.0
    assert config.target_match_thresh == 0.35
    assert config.target_confirm_thresh == 0.80
    assert config.target_ocr_thresh == 0.50
    assert config.target_ocr_confirm_thresh == 0.85
    assert config.calibrated_resolution == (1920, 1080)
    assert len(config.known_enemies) >= 1
    # T-FIX-30 channels.
    assert config.pose_coordinate_roi == (1720, 40, 180, 24)
    assert config.pose_min_confidence == 0.6
    assert config.reaction_nameplate_roi == (560, 300, 800, 40)
    assert config.reaction_min_pixels == 12
    assert config.reaction_dominance_thresh == 0.6
    assert config.proximity_reference_width_px == 120.0
    assert config.proximity_reference_distance_yd == 10.0
    assert config.proximity_min_confidence == 0.5
    assert config.pose_sampling_hz == 1.0
    assert config.reaction_sampling_hz == float("inf")
    # T-FIX-31 UI panel channels.
    assert config.bag_grid_origin == (1600, 700)
    assert (config.bag_columns, config.bag_rows) == (4, 4)
    assert config.bag_slot_size == (36, 36)
    assert config.bag_gap == 2
    assert config.bag_empty_slot_template == "models/bag_empty_slot_template.png"
    assert config.bag_occupied_slot_template == (
        "models/bag_occupied_slot_template.png"
    )
    assert config.bag_min_confidence == 0.7
    assert config.xp_bar_roi == (600, 1060, 720, 12)
    assert config.xp_level_roi == (560, 1050, 36, 20)
    assert config.xp_min_confidence == 0.5
    assert config.durability_slot_rois == ((900, 300, 26, 26), (940, 300, 26, 26))
    assert config.durability_min_score == 0.5
    assert config.durability_quorum_fraction == 0.5
    assert config.durability_min_confidence == 0.6
    assert config.cast_roi == (560, 260, 800, 26)
    assert config.cast_border_roi == (560, 258, 800, 3)
    assert config.cast_spell_ids == {"shadow bolt": "686"}
    assert config.cast_border_interruptible is False
    assert config.cast_min_confidence == 0.6
    assert config.cast_max_remaining_s == 30.0
    assert config.loot_sparkle_roi == (760, 400, 320, 200)
    assert config.loot_min_pixels == 40
    assert config.loot_dominance_thresh == 0.6
    assert config.loot_min_confidence == 0.9


def test_config_is_frozen() -> None:
    import dataclasses

    config = load_perception_config(Path("config/perception.example.toml"))
    with pytest.raises(dataclasses.FrozenInstanceError):
        config.idle_fps = 99  # type: ignore[misc]


def _example_dict() -> dict[str, object]:
    import tomllib

    with Path("config/perception.example.toml").open("rb") as handle:
        return tomllib.load(handle)


def test_unknown_top_level_key_raises() -> None:
    data = _example_dict()
    data["bogus"] = 1
    with pytest.raises(PerceptionConfigError, match="Unknown top-level"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_unknown_section_key_raises() -> None:
    data = _example_dict()
    data["combat"]["bogus"] = 1  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match="Unknown key"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_missing_section_raises() -> None:
    data = _example_dict()
    del data["enemies"]
    with pytest.raises(PerceptionConfigError, match="Missing required section"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_bad_roi_and_thresh_raise() -> None:
    data = _example_dict()
    data["bars"]["hp_roi"] = [1, 2, 3]  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match="hp_roi"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]
    data = _example_dict()
    data["enemies"]["confidence"] = 1.5  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match="confidence"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_bad_t_fix_30_channel_values_raise() -> None:
    data = _example_dict()
    data["pose"]["min_confidence"] = 1.5  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[pose\].min_confidence"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["reaction"]["min_pixels"] = 0  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[reaction\].min_pixels"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["proximity"]["reference_width_px"] = 0.0  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[proximity\].reference_width_px"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_unknown_t_fix_30_channel_key_raises() -> None:
    data = _example_dict()
    data["proximity"]["bogus"] = 1  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match="Unknown key"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_bad_t_fix_31_channel_values_raise() -> None:
    data = _example_dict()
    data["bag"]["grid_origin"] = [1, 2, 3]  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[bag\].grid_origin"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["bag"]["gap"] = -1  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[bag\].gap"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["durability"]["slot_rois"] = []  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[durability\].slot_rois"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["cast"]["border_interruptible"] = "yes"  # type: ignore[index]
    with pytest.raises(
        PerceptionConfigError, match=r"\[cast\].border_interruptible"
    ):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["cast"]["spell_ids"]["Shadow Bolt"] = 1.5  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[cast\].spell_ids"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]

    data = _example_dict()
    data["loot"]["min_pixels"] = 0  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match=r"\[loot\].min_pixels"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_an_absent_optional_panel_roi_is_allowed() -> None:
    """The XP level and cast border regions are opportunistic UI elements."""
    data = _example_dict()
    del data["xp"]["level_roi"]  # type: ignore[index]
    del data["cast"]["border_roi"]  # type: ignore[index]
    config = load_perception_config_from_dict(data)  # type: ignore[arg-type]
    assert config.xp_level_roi is None
    assert config.cast_border_roi is None


def test_an_empty_spell_id_table_is_allowed() -> None:
    """No table means no id is resolvable, which drops every cast loudly."""
    data = _example_dict()
    data["cast"]["spell_ids"] = {}  # type: ignore[index]
    config = load_perception_config_from_dict(data)  # type: ignore[arg-type]
    assert config.cast_spell_ids == {}


def test_unknown_t_fix_31_channel_key_raises() -> None:
    data = _example_dict()
    data["bag"]["bogus"] = 1  # type: ignore[index]
    with pytest.raises(PerceptionConfigError, match="Unknown key"):
        load_perception_config_from_dict(data)  # type: ignore[arg-type]


def test_missing_file_and_bad_toml_raise(tmp_path: Path) -> None:
    with pytest.raises(PerceptionConfigError, match="not found"):
        load_perception_config(tmp_path / "nope.toml")
    bad = tmp_path / "bad.toml"
    bad.write_text("invalid = [toml,\n", encoding="utf-8")
    with pytest.raises(PerceptionConfigError, match="TOML parse error"):
        load_perception_config(bad)


def test_example_comments_above_every_key() -> None:
    lines = Path("config/perception.example.toml").read_text(encoding="utf-8").splitlines()
    for i, line in enumerate(lines):
        if "=" in line:
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            assert j >= 0, f"Line {i + 1} has no preceding line"
            assert lines[j].strip().startswith("#"), f"Line {i + 1} missing comment"


def test_new_module_avoids_forbidden_imports() -> None:
    path = Path("src/wow_bot/perception/perception_config.py")
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
            assert module not in forbidden
            assert "llm" not in module.lower()
