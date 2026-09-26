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
