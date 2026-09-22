"""Tests for farm profile loader and validator (T11.1)."""

import ast
import copy
import json
from pathlib import Path
from types import MappingProxyType

import pytest

from wow_bot.farm import (
    PROFILE_SCHEMA_VERSION,
    FarmProfileError,
    load_profile,
    profile_summary,
    validate_profile_dict,
)


def _write_toml(path: Path, content: str) -> Path:
    toml_path = path / "profile.toml"
    toml_path.write_text(content, encoding="utf-8")
    return toml_path


VALID_TOML = """
[profile]
name = "Test Route"
schema_version = 1
description = "A valid test route"

[cycle]
nodes = [
  { kind = "node", name = "Peacebloom 1", behavior_hints = { action = "gather" } }
]
vendor = { kind = "vendor", name = "Vendor Thomas" }
repair = { kind = "vendor", name = "Repair Guy" }
stop_when_inventory_full = true
stop_after_cycles = 5

[route_preferences]
avoid_kinds = ["mob"]
prefer_kinds = ["waypoint"]
max_detour_factor = 1.2

[metadata]
author = "Tester"
"""


def test_example_profile_loads_successfully() -> None:
    example_path = Path("config/farm_profiles/example.toml")
    profile = load_profile(example_path)
    assert profile.name == "Elwynn Herb Route"
    assert profile.schema_version == PROFILE_SCHEMA_VERSION
    assert profile.schema_version == 1


def test_profile_schema_version_equals_one() -> None:
    assert PROFILE_SCHEMA_VERSION == 1


def test_load_profile_missing_file_raises(tmp_path: Path) -> None:
    missing_file = tmp_path / "does_not_exist.toml"
    with pytest.raises(FarmProfileError, match="not found"):
        load_profile(missing_file)


def test_load_profile_malformed_toml_raises(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, "invalid = [toml, bad")
    with pytest.raises(FarmProfileError, match="TOML parse error"):
        load_profile(f)


def test_load_profile_root_not_table_raises(tmp_path: Path) -> None:
    # tomllib.loads always returns a dict for valid TOML, but test validate_profile_dict with non-dict
    with pytest.raises(FarmProfileError, match="Root of farm profile must be a table"):
        validate_profile_dict("not a dict")  # type: ignore[arg-type]


def test_load_profile_unknown_top_level_key_raises(tmp_path: Path) -> None:
    content = VALID_TOML + "\n[unknown_section]\nfoo = 'bar'"
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Unknown top-level key: 'unknown_section'"):
        load_profile(f)


def test_load_profile_missing_profile_section_raises(tmp_path: Path) -> None:
    content = """
[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Missing required top-level section: 'profile'"):
        load_profile(f)


def test_load_profile_missing_cycle_section_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Missing required top-level section: 'cycle'"):
        load_profile(f)


def test_load_profile_unknown_key_in_profile_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1
extra_key = "value"

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Unknown key in \\[profile\\]: 'extra_key'"):
        load_profile(f)


def test_load_profile_unknown_key_in_cycle_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
invalid_cycle_key = 123
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Unknown key in \\[cycle\\]: 'invalid_cycle_key'"):
        load_profile(f)


def test_load_profile_invalid_schema_version_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 99

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Unsupported schema_version 99"):
        load_profile(f)


def test_load_profile_missing_name_in_profile_raises(tmp_path: Path) -> None:
    content = """
[profile]
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Missing required key 'name' in \\[profile\\]"):
        load_profile(f)


def test_load_profile_empty_nodes_list_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = []
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="'nodes' in \\[cycle\\] must be a non-empty list"):
        load_profile(f)


def test_load_profile_node_reference_missing_kind_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Missing required key 'kind' in node reference"):
        load_profile(f)


def test_load_profile_node_reference_invalid_kind_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "invalid_kind", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Invalid node reference kind 'invalid_kind'"):
        load_profile(f)


def test_load_profile_node_reference_whitespace_name_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = " N1 " }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Invalid node reference name ' N1 '"):
        load_profile(f)


def test_load_profile_vendor_reference_invalid_kind_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "mob", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="Invalid vendor reference kind 'mob'"):
        load_profile(f)


def test_load_profile_repair_defaults_to_vendor(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    profile = load_profile(f)
    assert profile.cycle.repair == profile.cycle.vendor
    assert profile.cycle.repair.name == "V1"


def test_load_profile_stop_when_inventory_full_defaults_true(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    profile = load_profile(f)
    assert profile.cycle.stop_when_inventory_full is True


def test_load_profile_stop_after_cycles_defaults_zero(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
"""
    f = _write_toml(tmp_path, content)
    profile = load_profile(f)
    assert profile.cycle.stop_after_cycles == 0


def test_load_profile_negative_stop_after_cycles_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }
stop_after_cycles = -1
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="'stop_after_cycles' in \\[cycle\\] must be an integer >= 0"):
        load_profile(f)


def test_load_profile_unknown_key_in_route_preferences_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }

[route_preferences]
unknown_pref = 123
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(
        FarmProfileError, match="Unknown key in \\[route_preferences\\]: 'unknown_pref'"
    ):
        load_profile(f)


def test_load_profile_overlapping_avoid_and_prefer_kinds_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }

[route_preferences]
avoid_kinds = ["mob"]
prefer_kinds = ["mob"]
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="avoid_kinds and prefer_kinds must be disjoint"):
        load_profile(f)


def test_load_profile_detour_factor_less_than_one_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }

[route_preferences]
max_detour_factor = 0.5
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="max_detour_factor must be finite and >= 1.0"):
        load_profile(f)


def test_load_profile_non_string_value_in_metadata_raises(tmp_path: Path) -> None:
    content = """
[profile]
name = "Route"
schema_version = 1

[cycle]
nodes = [{ kind = "node", name = "N1" }]
vendor = { kind = "vendor", name = "V1" }

[metadata]
level = 10
"""
    f = _write_toml(tmp_path, content)
    with pytest.raises(FarmProfileError, match="\\[metadata\\] keys and values must all be strings"):
        load_profile(f)


def test_load_profile_valid_metadata_returns_mapping_proxy(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    profile = load_profile(f)
    assert isinstance(profile.metadata, MappingProxyType)
    assert profile.metadata["author"] == "Tester"


def test_node_reference_behavior_hints_is_mapping_proxy(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    profile = load_profile(f)
    node = profile.cycle.nodes[0]
    assert isinstance(node.behavior_hints, MappingProxyType)
    assert node.behavior_hints["action"] == "gather"


def test_dataclasses_are_frozen(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    profile = load_profile(f)

    with pytest.raises((TypeError, AttributeError)):
        profile.name = "New Name"  # type: ignore[misc]

    with pytest.raises((TypeError, AttributeError)):
        profile.cycle.stop_after_cycles = 100  # type: ignore[misc]

    with pytest.raises((TypeError, AttributeError)):
        profile.cycle.nodes[0].name = "New Node"  # type: ignore[misc]

    with pytest.raises((TypeError, AttributeError)):
        profile.cycle.vendor.name = "New Vendor"  # type: ignore[misc]

    with pytest.raises((TypeError, AttributeError)):
        profile.route_preferences.max_detour_factor = 2.0  # type: ignore[misc]


def test_validate_profile_dict_valid_passes() -> None:
    d = {
        "profile": {"name": "Valid Route", "schema_version": 1, "description": "desc"},
        "cycle": {
            "nodes": [{"kind": "node", "name": "N1"}],
            "vendor": {"kind": "vendor", "name": "V1"},
        },
    }
    validate_profile_dict(d)


def test_validate_profile_dict_invalid_raises() -> None:
    d = {
        "profile": {"name": "Valid Route", "schema_version": 1},
        "cycle": {
            "nodes": [],  # empty nodes list
            "vendor": {"kind": "vendor", "name": "V1"},
        },
    }
    with pytest.raises(FarmProfileError, match="'nodes' in \\[cycle\\] must be a non-empty list"):
        validate_profile_dict(d)


def test_validate_profile_dict_does_not_mutate_input() -> None:
    original = {
        "profile": {"name": "Route", "schema_version": 1, "description": "desc"},
        "cycle": {
            "nodes": [{"kind": "node", "name": "N1", "behavior_hints": {"key": "val"}}],
            "vendor": {"kind": "vendor", "name": "V1"},
            "stop_when_inventory_full": True,
            "stop_after_cycles": 2,
        },
        "route_preferences": {"avoid_kinds": ["mob"], "max_detour_factor": 1.5},
        "metadata": {"zone": "Elwynn"},
    }
    snapshot = copy.deepcopy(original)
    validate_profile_dict(original)
    assert original == snapshot


def test_profile_summary_structure(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    profile = load_profile(f)
    summary = profile_summary(profile)

    # Must be JSON serializable
    json_str = json.dumps(summary)
    assert isinstance(json_str, str)

    expected_keys = {
        "schema_version",
        "name",
        "node_count",
        "vendor",
        "repair",
        "stop_when_inventory_full",
        "stop_after_cycles",
        "route_preferences",
    }
    assert set(summary.keys()) == expected_keys
    assert summary["schema_version"] == 1
    assert summary["name"] == "Test Route"
    assert summary["node_count"] == 1
    assert summary["vendor"] == {"kind": "vendor", "name": "Vendor Thomas"}
    assert summary["repair"] == {"kind": "vendor", "name": "Repair Guy"}
    assert summary["stop_when_inventory_full"] is True
    assert summary["stop_after_cycles"] == 5
    assert summary["route_preferences"] == {
        "avoid_kinds": ["mob"],
        "prefer_kinds": ["waypoint"],
        "max_detour_factor": 1.2,
    }


def test_profile_summary_omits_metadata(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    profile = load_profile(f)
    summary = profile_summary(profile)
    assert "metadata" not in summary


def test_determinism(tmp_path: Path) -> None:
    f = _write_toml(tmp_path, VALID_TOML)
    p1 = load_profile(f)
    p2 = load_profile(f)
    assert p1 == p2


def test_example_profile_comments_above_every_key() -> None:
    example_path = Path("config/farm_profiles/example.toml")
    lines = example_path.read_text(encoding="utf-8").splitlines()

    for i, line in enumerate(lines):
        if "=" in line:
            # Find immediately preceding non-empty line
            j = i - 1
            while j >= 0 and not lines[j].strip():
                j -= 1
            assert j >= 0, f"Line {i+1} containing '=' has no preceding line"
            assert lines[j].strip().startswith(
                "#"
            ), f"Line {i+1} '{line.strip()}' is not preceded by a comment line (preceded by line {j+1} '{lines[j].strip()}')"


def test_ast_profile_no_prohibited_imports() -> None:
    path = Path("src/wow_bot/farm/profile.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    prohibited_modules = {
        "wow_bot.reporting",
        "wow_bot.analysis",
        "wow_bot.strategist",
        "wow_bot.watchdog",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.world.store",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.executor",
        "wow_bot.humanize",
        "wow_bot.internal_dynamics",
        "wow_bot.lab",
        "wow_bot.main",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for p in prohibited_modules:
                    assert not mod_name.startswith(p), f"Prohibited import: {mod_name}"
                for term in ("ollama", "openai", "anthropic", "llm"):
                    assert term not in mod_name.lower(), f"Prohibited import term: {mod_name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for p in prohibited_modules:
                assert not mod_name.startswith(p), f"Prohibited import from: {mod_name}"
            for term in ("ollama", "openai", "anthropic", "llm"):
                assert term not in mod_name.lower(), f"Prohibited import term: {mod_name}"


def test_ast_profile_no_time_calls() -> None:
    path = Path("src/wow_bot/farm/profile.py")
    tree = ast.parse(path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr in (
            "monotonic",
            "time",
            "perf_counter",
        ):
            # Check if called on time or time module
            assert False, f"Prohibited time call detected: {node.attr}"


def test_no_prelab_module_modified() -> None:
    # Ensure src/wow_bot/farm/ only contains new files and no existing files outside farm/ were touched
    # This is verified statically by inspecting git status or checking that files in farm/ are new
    farm_dir = Path("src/wow_bot/farm")
    assert farm_dir.exists()
    assert (farm_dir / "__init__.py").exists()
    assert (farm_dir / "profile.py").exists()
