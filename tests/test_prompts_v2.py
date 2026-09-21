"""Tests for Strategy v2 grounded prompt builder (wow_bot.strategist.prompts_v2)."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping

import pytest

from wow_bot.executor.states import FSMState
from wow_bot.strategist.prompts_v2 import (
    ALLOWED_GOALS,
    GameStateView,
    MetaStateLike,
    PromptBundle,
    PromptConfig,
    build_prompt,
    render_schema,
)
from wow_bot.world.summary import CombatSummary, NodeSummary, WorldSummary


@dataclass
class FakeMeta(MetaStateLike):
    drives: Mapping[str, float] = field(
        default_factory=lambda: {"hunger": 0.2, "fatigue": 0.5, "curiosity": 0.8}
    )
    memory_summary: str = "Explored Goldshire inn and talked to trainer."


@dataclass
class FakeGameState(GameStateView):
    player_x: float = 100.5
    player_y: float = 200.25
    player_z: float = 10.0
    self_hp_percent: float = 85.0
    resource: float = 60.0
    resource_max: float = 100.0
    current_target_id: str | None = "mob_123"
    target_hp_percent: float | None = 45.0
    inventory_count: int = 12
    level_or_xp: float = 5.0
    fsm_state: FSMState = FSMState.IDLE


def make_empty_world_summary() -> WorldSummary:
    """Helper constructing an empty WorldSummary without DB access."""
    return WorldSummary(
        around_xy=(100.5, 200.25),
        radius=100.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=(),
        nearest_trainers=(),
        nearest_nodes=(),
        nearest_mobs=(),
        nearest_waypoints=(),
        recent_combats=(),
        total_nodes=0,
    )


def make_populated_world_summary() -> WorldSummary:
    """Helper constructing a populated WorldSummary with multiple nodes and combats."""
    node1 = NodeSummary(1, 105.0, 205.0, 10.0, "vendor", 7.1, "2026-09-21T11:50:00Z")
    node2 = NodeSummary(2, 110.0, 210.0, 10.0, "vendor", 14.1, "2026-09-21T11:51:00Z")
    node3 = NodeSummary(3, 115.0, 215.0, 10.0, "trainer", 21.2, "2026-09-21T11:52:00Z")
    node4 = NodeSummary(4, 120.0, 220.0, 10.0, "node", 28.3, "2026-09-21T11:53:00Z")
    node5 = NodeSummary(5, 125.0, 225.0, 10.0, "mob", 35.4, "2026-09-21T11:54:00Z")
    node6 = NodeSummary(6, 130.0, 230.0, 10.0, "waypoint", 42.4, "2026-09-21T11:55:00Z")

    combat1 = CombatSummary(101, "mob_123", "VICTORY", "2026-09-21T11:50:00Z", "2026-09-21T11:51:00Z")

    return WorldSummary(
        around_xy=(100.5, 200.25),
        radius=100.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=(node1, node2),
        nearest_trainers=(node3,),
        nearest_nodes=(node4,),
        nearest_mobs=(node5,),
        nearest_waypoints=(node6,),
        recent_combats=(combat1,),
        total_nodes=6,
    )


def test_prompt_config_validation() -> None:
    """Acceptance: PromptConfig parameter validation."""
    with pytest.raises(ValueError, match="schema_version"):
        PromptConfig(schema_version=0)

    with pytest.raises(ValueError, match="max_memory_chars"):
        PromptConfig(max_memory_chars=-1)

    with pytest.raises(ValueError, match="max_vendors"):
        PromptConfig(max_vendors=-1)

    with pytest.raises(ValueError, match="max_recent_combats"):
        PromptConfig(max_recent_combats=-5)


def test_build_prompt_non_empty_and_sections() -> None:
    """Acceptance: build_prompt returns PromptBundle with non-empty text and all documented sections in order."""
    meta = FakeMeta()
    world = make_populated_world_summary()
    state = FakeGameState()
    config = PromptConfig(schema_version=1)

    bundle = build_prompt(meta, world, state, config=config)

    assert isinstance(bundle, PromptBundle)
    assert len(bundle.text) > 0

    # Verify sections separated by blank lines
    sections = bundle.text.split("\n\n")
    assert len(sections) == 8

    # 1. system
    assert "You are a planning agent" in sections[0]
    assert "Prompt schema version: 1." in sections[0]

    # 2. allowed_goals
    for goal in ALLOWED_GOALS:
        assert f"- {goal}:" in sections[1]

    # 3. drives
    assert "curiosity: 0.800" in sections[2]
    assert "fatigue: 0.500" in sections[2]
    assert "hunger: 0.200" in sections[2]

    # 4. memory
    assert "Explored Goldshire inn" in sections[3]

    # 5. world
    assert "vendors:" in sections[4]

    # 6. state
    assert "fsm_state: IDLE" in sections[5]
    assert "position: (100.50,200.25,10.00)" in sections[5]

    # 7. response_schema
    parsed_schema = json.loads(sections[6])
    assert parsed_schema["goal"] == "<one of the allowed goals>"

    # 8. instructions
    assert "Reply with a single JSON object" in sections[7]


def test_build_prompt_allowed_goals_count() -> None:
    """Acceptance: allowed_goals section contains one line per ALLOWED_GOALS entry."""
    meta = FakeMeta()
    world = make_empty_world_summary()
    state = FakeGameState()

    bundle = build_prompt(meta, world, state)
    sections = bundle.text.split("\n\n")
    goals_lines = sections[1].split("\n")

    assert len(goals_lines) == len(ALLOWED_GOALS)
    for line, expected_goal in zip(goals_lines, ALLOWED_GOALS, strict=True):
        assert line.startswith(f"- {expected_goal}:")


def test_build_prompt_drives_sorting_and_formatting() -> None:
    """Acceptance: drives section lists keys in sorted order formatted with 3 decimals."""
    meta = FakeMeta(drives={"social": 0.1, "aggression": 0.99, "fatigue": 0.33333})
    world = make_empty_world_summary()
    state = FakeGameState()

    bundle = build_prompt(meta, world, state)
    sections = bundle.text.split("\n\n")
    drive_lines = sections[2].split("\n")

    assert drive_lines == [
        "  aggression: 0.990",
        "  fatigue: 0.333",
        "  social: 0.100",
    ]


def test_build_prompt_memory_truncation_and_empty() -> None:
    """Acceptance: memory truncation and empty summary handling."""
    # Truncation
    meta_long = FakeMeta(memory_summary="A" * 50)
    config = PromptConfig(max_memory_chars=10)
    bundle_long = build_prompt(meta_long, make_empty_world_summary(), FakeGameState(), config=config)
    mem_section = bundle_long.text.split("\n\n")[3]
    assert mem_section == "AAAAAAAAAA…"

    # Empty
    meta_empty = FakeMeta(memory_summary="")
    bundle_empty = build_prompt(meta_empty, make_empty_world_summary(), FakeGameState())
    mem_section_empty = bundle_empty.text.split("\n\n")[3]
    assert mem_section_empty == "(none)"


def test_build_prompt_world_max_entries_and_empty() -> None:
    """Acceptance: world section includes up to max entries per kind and shows (no world data) when empty."""
    # Populate multiple vendors
    nodes = tuple(
        NodeSummary(i, 100.0 + i, 200.0 + i, 10.0, "vendor", float(i), "2026-09-21T12:00:00Z")
        for i in range(1, 10)
    )
    world_multi = WorldSummary(
        around_xy=(100.0, 200.0),
        radius=100.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=nodes,
        nearest_trainers=(),
        nearest_nodes=(),
        nearest_mobs=(),
        nearest_waypoints=(),
        recent_combats=(),
        total_nodes=9,
    )

    config = PromptConfig(max_vendors=2)
    bundle = build_prompt(FakeMeta(), world_multi, FakeGameState(), config=config)
    world_section = bundle.text.split("\n\n")[4]
    vendor_lines = [line for line in world_section.split("\n") if "id=" in line]
    assert len(vendor_lines) == 2

    # Completely empty world
    bundle_empty = build_prompt(FakeMeta(), make_empty_world_summary(), FakeGameState())
    world_section_empty = bundle_empty.text.split("\n\n")[4]
    assert world_section_empty == "(no world data)"


def test_build_prompt_state_formatting_and_none_values() -> None:
    """Acceptance: state section formatting, including handling None target/hp."""
    # Active target
    state_active = FakeGameState(
        current_target_id="mob_456",
        target_hp_percent=50.0,
    )
    bundle_active = build_prompt(FakeMeta(), make_empty_world_summary(), state_active)
    state_sec = bundle_active.text.split("\n\n")[5]
    assert "  target: mob_456" in state_sec
    assert "  target_hp: 50.0%" in state_sec

    # None target & hp
    state_none = FakeGameState(
        current_target_id=None,
        target_hp_percent=None,
    )
    bundle_none = build_prompt(FakeMeta(), make_empty_world_summary(), state_none)
    state_sec_none = bundle_none.text.split("\n\n")[5]
    assert "  target: none" in state_sec_none
    assert "  target_hp: unknown" in state_sec_none


def test_build_prompt_raw_state_json_flag() -> None:
    """Acceptance: include_raw_state_json flag controls raw state JSON fence block."""
    meta = FakeMeta()
    world = make_empty_world_summary()
    state = FakeGameState()

    # include_raw_state_json = False (default)
    bundle_false = build_prompt(meta, world, state, config=PromptConfig(include_raw_state_json=False))
    assert "```json" not in bundle_false.text

    # include_raw_state_json = True
    bundle_true = build_prompt(meta, world, state, config=PromptConfig(include_raw_state_json=True))
    assert "```json" in bundle_true.text
    state_sec = bundle_true.text.split("\n\n")[5]
    assert "```json" in state_sec


def test_render_schema_and_prompt_schema_match() -> None:
    """Acceptance: render_schema returns valid JSON and matches response_schema section in build_prompt."""
    schema_str = render_schema()
    parsed = json.loads(schema_str)
    assert parsed["goal"] == "<one of the allowed goals>"

    bundle = build_prompt(FakeMeta(), make_empty_world_summary(), FakeGameState())
    prompt_schema_sec = bundle.text.split("\n\n")[6]

    assert schema_str == prompt_schema_sec


def test_prompt_hash_stability_and_sensitivity() -> None:
    """Acceptance: prompt_hash properties (32-char hex, stability, input & config sensitivity)."""
    meta = FakeMeta()
    world = make_empty_world_summary()
    state = FakeGameState()

    bundle1 = build_prompt(meta, world, state)
    bundle2 = build_prompt(meta, world, state)

    # 32-char hex string (blake2b digest size 16)
    assert len(bundle1.prompt_hash) == 32
    assert all(c in "0123456789abcdef" for c in bundle1.prompt_hash)

    # Stability
    assert bundle1.prompt_hash == bundle2.prompt_hash

    # Input sensitivity (MetaState change)
    meta_diff = FakeMeta(memory_summary="Different memory")
    bundle_meta_diff = build_prompt(meta_diff, world, state)
    assert bundle_meta_diff.prompt_hash != bundle1.prompt_hash

    # Config schema_version sensitivity
    config_v2 = PromptConfig(schema_version=2)
    bundle_cfg_diff = build_prompt(meta, world, state, config=config_v2)
    assert bundle_cfg_diff.prompt_hash != bundle1.prompt_hash


def test_token_estimate_and_section_lengths() -> None:
    """Acceptance: token_estimate equals len(text) // 4 and section_lengths structure."""
    bundle = build_prompt(FakeMeta(), make_populated_world_summary(), FakeGameState())

    assert bundle.token_estimate == len(bundle.text) // 4

    assert isinstance(bundle.section_lengths, Mapping)
    # Sum of section character lengths + separators == len(text)
    section_chars_sum = sum(bundle.section_lengths.values())
    assert section_chars_sum <= len(bundle.text)


def test_prompt_bundle_to_json() -> None:
    """Acceptance: PromptBundle.to_json is JSON-serializable and includes all fields."""
    bundle = build_prompt(FakeMeta(), make_empty_world_summary(), FakeGameState())
    data = bundle.to_json()

    # Serialization check
    json_str = json.dumps(data)
    assert isinstance(json_str, str)

    assert data["text"] == bundle.text
    assert data["prompt_hash"] == bundle.prompt_hash
    assert data["schema_version"] == bundle.schema_version
    assert data["token_estimate"] == bundle.token_estimate
    assert isinstance(data["section_lengths"], dict)


def test_determinism_and_input_immutability() -> None:
    """Acceptance: Determinism across runs and non-mutation of inputs."""
    drives = {"hunger": 0.4, "fatigue": 0.6}
    meta = FakeMeta(drives=drives, memory_summary="Initial summary")
    state = FakeGameState(player_x=12.34)

    bundle1 = build_prompt(meta, make_empty_world_summary(), state)
    bundle2 = build_prompt(meta, make_empty_world_summary(), state)

    # Byte-for-byte identical
    assert bundle1.text.encode("utf-8") == bundle2.text.encode("utf-8")
    assert bundle1.prompt_hash == bundle2.prompt_hash

    # Input non-mutability
    assert meta.drives == {"hunger": 0.4, "fatigue": 0.6}
    assert meta.memory_summary == "Initial summary"
    assert state.player_x == 12.34


def test_unicode_preservation() -> None:
    """Acceptance: Memory summary with unicode (Persian, emoji) is preserved in output and hash."""
    unicode_memory = "Visited tavern in Stormwind 🏰 | یادداشت: موفقیت‌آمیز بود"
    meta = FakeMeta(memory_summary=unicode_memory)

    bundle = build_prompt(meta, make_empty_world_summary(), FakeGameState())

    assert unicode_memory in bundle.text
    assert len(bundle.prompt_hash) == 32


def test_static_ast_forbidden_imports_and_time_calls() -> None:
    """Acceptance: Static AST check ensuring prompts_v2.py does not import forbidden modules or call time functions."""
    target_path = Path("src/wow_bot/strategist/prompts_v2.py")
    assert target_path.exists()

    tree = ast.parse(target_path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist.llm_client",
        "wow_bot.strategist.orchestrator",
        "wow_bot.strategist.parser",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.world.store",
        "wow_bot.world.loader",
        "wow_bot.world.sync",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")
    forbidden_time_funcs = ("monotonic", "time", "perf_counter")

    for node in ast.walk(tree):
        # Import checks
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                assert name not in forbidden_modules, f"Forbidden import found: {name}"
                for sub in forbidden_substrings:
                    assert sub not in name.lower(), f"Forbidden substring in import: {name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            assert mod_name not in forbidden_modules, f"Forbidden import from found: {mod_name}"
            for sub in forbidden_substrings:
                assert sub not in mod_name.lower(), f"Forbidden substring in import from: {mod_name}"

        # Call checks for time functions
        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                if isinstance(node.func.value, ast.Name) and node.func.value.id == "time":
                    assert node.func.attr not in forbidden_time_funcs, (
                        f"Forbidden time call found: time.{node.func.attr}"
                    )
