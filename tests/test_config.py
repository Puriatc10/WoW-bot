"""Acceptance tests for lab mode configuration loader (wow_bot.config)."""

from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest

from wow_bot.config import Config, ConfigError, load_config

VALID_TOML_CONTENT = """
# Valid example configuration
lab_mode = false
server_allowlist = ["127.0.0.1:8080", "10.0.0.1"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
log_level = "INFO"
"""


def test_valid_config_loads_without_error_and_all_fields_match(tmp_path: Path) -> None:
    """Acceptance: Valid config loads without error and all fields match."""
    config_file = tmp_path / "lab.toml"
    config_file.write_text(VALID_TOML_CONTENT, encoding="utf-8")

    cfg = load_config(config_file)

    assert isinstance(cfg, Config)
    assert cfg.lab_mode is False
    assert cfg.server_allowlist == ("127.0.0.1:8080", "10.0.0.1")
    assert cfg.isolation_sentinel == "1.1.1.1:80"
    assert cfg.kill_switch_key == "F12"
    assert cfg.session_root == Path("runs/lab")
    assert cfg.dry_run is True
    assert cfg.max_session_seconds == 3600
    assert cfg.log_level == "INFO"


def test_missing_required_key_raises_config_error(tmp_path: Path) -> None:
    """Acceptance: Missing required key raises ConfigError."""
    content = """
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
# log_level is missing
"""
    config_file = tmp_path / "missing_key.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="Missing required config key"):
        load_config(config_file)


def test_unknown_key_raises_config_error(tmp_path: Path) -> None:
    """Acceptance: Unknown key raises ConfigError."""
    content = (
        VALID_TOML_CONTENT
        + """
extra_unknown_key = "not_allowed"
"""
    )
    config_file = tmp_path / "unknown_key.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="Unknown config key"):
        load_config(config_file)


def test_lab_example_tomls_stay_loadable() -> None:
    """T-FIX-29: perception keys are comments, so the example stays loadable."""
    example = Path("config/lab.example.toml")
    cfg = load_config(example)
    assert cfg.lab_mode is False
    assert cfg.dry_run is True
    text = example.read_text(encoding="utf-8")
    for key in (
        "tesseract_cmd",
        "hp_roi",
        "mana_roi",
        "minimap_roi",
        "yolo_weights",
        "edge_size",
        "red_ratio_thresh",
        "cooldown_s",
        "match_thresh",
        "confirm_thresh",
        "ocr_thresh",
        "ocr_confirm_thresh",
        "combat_region",
        "chat_region",
        "known_enemies",
        "smoothing_frames",
        "calibrated_resolution",
        "idle_fps",
        "combat_fps",
        "name_template",
        "frame_template",
        "arrow_template",
    ):
        assert key in text


@pytest.mark.parametrize(
    "forbidden_domain",
    [
        "blizzard.com",
        "battle.net",
        "worldofwarcraft.com",
        "us.BlIzZaRd.CoM:80",
        "auth.BATTLE.NET",
    ],
)
def test_server_allowlist_containing_blizzard_domain_raises_config_error(
    tmp_path: Path, forbidden_domain: str
) -> None:
    """Acceptance: server_allowlist containing a known Blizzard domain raises ConfigError."""
    content = f"""
lab_mode = false
server_allowlist = ["127.0.0.1:8080", "{forbidden_domain}"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
log_level = "INFO"
"""
    config_file = tmp_path / "blizzard.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="forbidden domain"):
        load_config(config_file)


def test_log_level_with_invalid_value_raises_config_error(tmp_path: Path) -> None:
    """Acceptance: log_level with invalid value raises ConfigError."""
    content = """
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
log_level = "TRACE"
"""
    config_file = tmp_path / "invalid_log_level.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="'log_level' must be one of"):
        load_config(config_file)


@pytest.mark.parametrize(
    "invalid_sentinel",
    [
        "1.1.1.1",
        "invalid_host_no_port",
        "1.1.1.1:port",
        "1.1.1.1:70000",
        "1.1.1.1:0",
        ":8080",
    ],
)
def test_isolation_sentinel_with_invalid_format_raises_config_error(
    tmp_path: Path, invalid_sentinel: str
) -> None:
    """Acceptance: isolation_sentinel with invalid format raises ConfigError."""
    content = f"""
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "{invalid_sentinel}"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
log_level = "INFO"
"""
    config_file = tmp_path / "invalid_sentinel.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError):
        load_config(config_file)


def test_config_is_frozen_attempting_to_set_field_raises(tmp_path: Path) -> None:
    """Acceptance: Config is frozen: attempting to set a field raises."""
    config_file = tmp_path / "lab.toml"
    config_file.write_text(VALID_TOML_CONTENT, encoding="utf-8")

    cfg = load_config(config_file)

    with pytest.raises(FrozenInstanceError):
        cfg.lab_mode = True  # type: ignore[misc]


def test_server_allowlist_is_tuple_not_list(tmp_path: Path) -> None:
    """Acceptance: server_allowlist is a tuple, not a list."""
    config_file = tmp_path / "lab.toml"
    config_file.write_text(VALID_TOML_CONTENT, encoding="utf-8")

    cfg = load_config(config_file)

    assert isinstance(cfg.server_allowlist, tuple)
    assert not isinstance(cfg.server_allowlist, list)


def test_file_does_not_exist_raises_config_error(tmp_path: Path) -> None:
    """Test that a non-existent file raises ConfigError."""
    non_existent = tmp_path / "does_not_exist.toml"
    with pytest.raises(ConfigError, match="does not exist"):
        load_config(non_existent)


def test_invalid_toml_syntax_raises_config_error(tmp_path: Path) -> None:
    """Test that malformed TOML raises ConfigError."""
    config_file = tmp_path / "broken.toml"
    config_file.write_text("lab_mode = [unclosed bracket", encoding="utf-8")

    with pytest.raises(ConfigError, match="not valid TOML"):
        load_config(config_file)


@pytest.mark.parametrize("bad_seconds", [0, -10, "3600", true_val := True])
def test_max_session_seconds_invalid_raises_config_error(
    tmp_path: Path, bad_seconds: object
) -> None:
    """Test that max_session_seconds <= 0 or not int raises ConfigError."""
    toml_seconds = "true" if bad_seconds is True else repr(bad_seconds)
    content = f"""
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = "runs/lab"
dry_run = true
max_session_seconds = {toml_seconds}
log_level = "INFO"
"""
    config_file = tmp_path / "bad_seconds.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="max_session_seconds"):
        load_config(config_file)


def test_empty_session_root_raises_config_error(tmp_path: Path) -> None:
    """Test that an empty session_root string raises ConfigError."""
    content = """
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = "F12"
session_root = ""
dry_run = true
max_session_seconds = 3600
log_level = "INFO"
"""
    config_file = tmp_path / "empty_session_root.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="session_root"):
        load_config(config_file)


def test_empty_kill_switch_key_raises_config_error(tmp_path: Path) -> None:
    """Test that an empty kill_switch_key string raises ConfigError."""
    content = """
lab_mode = false
server_allowlist = ["127.0.0.1:8080"]
isolation_sentinel = "1.1.1.1:80"
kill_switch_key = ""
session_root = "runs/lab"
dry_run = true
max_session_seconds = 3600
log_level = "INFO"
"""
    config_file = tmp_path / "empty_kill_switch.toml"
    config_file.write_text(content, encoding="utf-8")

    with pytest.raises(ConfigError, match="kill_switch_key"):
        load_config(config_file)
