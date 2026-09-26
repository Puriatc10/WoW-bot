"""Tests for T-FIX-21: adapter derivation, runtime context, resource table.

These cover the quantities ADR-002 classifies **derived** (computed by the
adapter, never stored on ``GameState``) and **internal** (answered by a
runtime context, not observed).

The projection-expectation restructure across all eight views is T-FIX-24's
job; this file proves the derivation rules themselves.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from wow_bot.perception.adapter import (
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
)
from wow_bot.perception.context import RuntimeContext, StaticRuntimeContext
from wow_bot.perception.resource_table import EMPTY_RESOURCE_TABLE, ResourceTable
from wow_bot.perception.views import CombatView, ReactiveView, TargetView
from wow_bot.shared.interfaces import EnemyInfo, GameState, TargetInfo

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_entity(
    *,
    entity_id: str = "mob-a",
    distance: float = 8.0,
    hp_fraction: float = 0.70,
    threat: float | None = 12.0,
    is_attackable: bool | None = True,
    is_alive: bool | None = True,
    in_combat_with_self: bool | None = True,
) -> EnemyInfo:
    """Build one canonically-populated entity observation."""
    return EnemyInfo(
        bbox=(1, 2, 3, 4),
        confidence=0.9,
        distance_estimate=distance,
        entity_id=entity_id,
        kind="mob",
        x=10.0,
        y=20.0,
        z=None,  # the mock world model is 2-D; no height channel
        hp_fraction=hp_fraction,
        threat=threat,
        is_attackable=is_attackable,
        is_alive=is_alive,
        is_in_combat_with_self=in_combat_with_self,
    )


def _make_populated_state(
    *,
    entities: tuple[EnemyInfo, ...] | None = None,
    target: TargetInfo | None = None,
) -> GameState:
    """A GameState with every observed field populated, as a real backend would."""
    if entities is None:
        entities = (_make_entity(),)
    if target is None:
        target = TargetInfo(name="mob-a", hp_pct=0.55, reaction="hostile", distance_estimate=8.0)
    return GameState(
        timestamp=1.0,
        hp_pct=0.90,
        mana_pct=0.50,
        position=(100.0, 200.0),
        facing=0.0,
        in_combat=True,
        target=target,
        enemies=list(entities),
        events=[],
        player_z=1.5,
        entities=entities,
    )


def _config(distance: float = 30.0) -> AdapterDerivationConfig:
    return AdapterDerivationConfig(engage_distance_units=distance)


# ---------------------------------------------------------------------------
# AdapterDerivationConfig
# ---------------------------------------------------------------------------


def test_derivation_config_rejects_non_positive_distance() -> None:
    with pytest.raises(ValueError, match="engage_distance_units must be positive"):
        AdapterDerivationConfig(engage_distance_units=0.0)

    with pytest.raises(ValueError, match="engage_distance_units must be positive"):
        AdapterDerivationConfig(engage_distance_units=-1.0)


# ---------------------------------------------------------------------------
# target_in_range
# ---------------------------------------------------------------------------


def test_target_in_range_derived_within_threshold() -> None:
    adapter = GameStateAdapter(_make_populated_state(), config=_config(30.0))

    assert adapter.to_reactive_view().target_in_range is True


def test_target_in_range_derived_outside_threshold() -> None:
    target = TargetInfo(name="mob-a", hp_pct=0.55, reaction="hostile", distance_estimate=44.0)
    adapter = GameStateAdapter(_make_populated_state(target=target), config=_config(30.0))

    assert adapter.to_reactive_view().target_in_range is False


def test_target_in_range_not_derivable_without_config() -> None:
    """The adapter must not invent an engagement threshold."""
    adapter = GameStateAdapter(_make_populated_state())

    with pytest.raises(AdapterIncompleteError, match="non-Optional target_in_range"):
        adapter.to_reactive_view()


def test_explicit_target_in_range_overrides_derivation() -> None:
    """An explicitly supplied value wins, so injected test channels still work."""
    state = _make_populated_state()
    object.__setattr__(state, "target_in_range", False)
    adapter = GameStateAdapter(state, config=_config(30.0))

    assert adapter.to_reactive_view().target_in_range is False


# ---------------------------------------------------------------------------
# target_is_alive
# ---------------------------------------------------------------------------


def _loot_state(target: TargetInfo | None) -> GameState:
    state = _make_populated_state()
    object.__setattr__(state, "target", target)
    object.__setattr__(state, "target_is_lootable", True)
    object.__setattr__(state, "inventory_count", 3)
    return state


def test_target_is_alive_derived_from_hp_fraction() -> None:
    alive = TargetInfo("mob-a", 0.55, "hostile", 8.0)
    dead = TargetInfo("mob-b", 0.0, "hostile", 8.0)

    assert GameStateAdapter(_loot_state(alive)).to_loot_view().target_is_alive is True
    assert GameStateAdapter(_loot_state(dead)).to_loot_view().target_is_alive is False


def test_target_is_alive_not_derivable_without_target() -> None:
    with pytest.raises(AdapterIncompleteError, match="non-Optional target_is_alive"):
        GameStateAdapter(_loot_state(None)).to_loot_view()


def test_target_in_range_not_derivable_without_target() -> None:
    state = _make_populated_state()
    object.__setattr__(state, "target", None)

    with pytest.raises(AdapterIncompleteError, match="non-Optional target_in_range"):
        GameStateAdapter(state, config=_config()).to_reactive_view()


# ---------------------------------------------------------------------------
# adds_count
# ---------------------------------------------------------------------------


def test_adds_count_counts_engaged_entities() -> None:
    entities = (
        _make_entity(entity_id="a", in_combat_with_self=True),
        _make_entity(entity_id="b", in_combat_with_self=True),
        _make_entity(entity_id="c", in_combat_with_self=False),
    )
    adapter = GameStateAdapter(_make_populated_state(entities=entities), config=_config())

    assert adapter.to_flee_view().adds_count == 2


def test_adds_count_none_when_entity_channel_empty() -> None:
    """Empty means 'no observations of this kind', not 'zero adds'."""
    adapter = GameStateAdapter(_make_populated_state(entities=()), config=_config())

    with pytest.raises(AdapterIncompleteError, match="non-Optional adds_count"):
        adapter.to_flee_view()


def test_adds_count_none_when_any_entity_status_unknown() -> None:
    """A partial count would under-report adds and suppress fleeing."""
    entities = (
        _make_entity(entity_id="a", in_combat_with_self=True),
        _make_entity(entity_id="b", in_combat_with_self=None),
    )
    adapter = GameStateAdapter(_make_populated_state(entities=entities), config=_config())

    with pytest.raises(AdapterIncompleteError, match="non-Optional adds_count"):
        adapter.to_flee_view()


# ---------------------------------------------------------------------------
# ResourceTable + resource_max
# ---------------------------------------------------------------------------


def test_resource_table_lookup_and_misses() -> None:
    table = ResourceTable(entries={("mage", 60): 4200.0})

    assert table.max_resource("mage", 60) == 4200.0
    assert table.max_resource("mage", 61) is None  # no entry: not guessed
    assert table.max_resource(None, 60) is None  # class unobserved
    assert table.max_resource("mage", None) is None  # level unobserved


def test_resource_table_rejects_malformed_entries() -> None:
    with pytest.raises(ValueError, match="non-empty class name"):
        ResourceTable(entries={("", 60): 100.0})

    with pytest.raises(ValueError, match="levels must be positive"):
        ResourceTable(entries={("mage", 0): 100.0})

    with pytest.raises(ValueError, match="values must be positive"):
        ResourceTable(entries={("mage", 60): 0.0})


def test_resource_max_underivable_with_default_empty_table() -> None:
    """ADR-002 unresolved question 2: provenance data is absent, so it is None."""
    adapter = GameStateAdapter(_make_populated_state(), config=_config())

    with pytest.raises(AdapterIncompleteError, match="non-Optional resource_max"):
        adapter.to_combat_view()

    assert EMPTY_RESOURCE_TABLE.entries == {}


def test_resource_max_derives_once_a_class_channel_exists() -> None:
    """Proves the mechanism without inventing game data.

    ``char_class`` is not a GameState field yet, so the test injects it the
    same way the adapter reads it. When a character-class channel is added and
    ``docs/PERCEPTION.md`` documents it, no further adapter change is needed
    beyond that channel being populated.
    """
    state = _make_populated_state()
    object.__setattr__(state, "char_class", "mage")
    object.__setattr__(state, "level_or_xp", 60.4)
    object.__setattr__(state, "inventory_count", 4)
    table = ResourceTable(entries={("mage", 60): 4200.0})

    adapter = GameStateAdapter(state, config=_config(), resource_table=table)
    view = adapter.to_strategist_view(fsm_state="COMBAT")

    assert view.resource_max == 4200.0
    assert view.level_or_xp == 60.4


# ---------------------------------------------------------------------------
# Runtime context (ADR-002 "internal" class)
# ---------------------------------------------------------------------------


def test_adapter_remains_constructible_with_snapshot_only() -> None:
    """Backwards compatibility: the single-argument construction still works."""
    adapter = GameStateAdapter(_make_populated_state())

    assert adapter.config is None
    assert adapter.context is None
    assert adapter.resource_table is EMPTY_RESOURCE_TABLE


def test_static_runtime_context_defaults() -> None:
    context = StaticRuntimeContext()

    assert context.gcd_ready() is True
    assert context.spell_cooldown_ready("anything") is False
    assert context.target_has_debuff("anything") is False
    assert isinstance(context, RuntimeContext)


def test_combat_view_delegates_to_runtime_context() -> None:
    state = _make_populated_state()
    object.__setattr__(state, "resource_max", 100.0)
    context = StaticRuntimeContext(
        gcd_is_ready=False,
        ready_spells=frozenset({"heroic_strike"}),
        target_debuffs=frozenset({"rend"}),
    )

    view = GameStateAdapter(state, config=_config(), context=context).to_combat_view()

    assert isinstance(view, CombatView)
    assert view.gcd_ready is False
    assert view.target_has_debuff("rend") is True
    assert view.target_has_debuff("sunder") is False
    assert view.spell_cooldown_ready("heroic_strike") is True
    assert view.spell_cooldown_ready("execute") is False


def test_reactive_view_delegates_to_runtime_context() -> None:
    context = StaticRuntimeContext(ready_spells=frozenset({"shield"}))
    view = GameStateAdapter(
        _make_populated_state(), config=_config(), context=context
    ).to_reactive_view()

    assert isinstance(view, ReactiveView)
    assert view.spell_cooldown_ready("shield") is True
    assert view.spell_cooldown_ready("interrupt") is False


def test_reactive_view_without_context_fails_loudly() -> None:
    """No context means no fabricated cooldown answer."""
    view = GameStateAdapter(_make_populated_state(), config=_config()).to_reactive_view()

    with pytest.raises(NotImplementedError, match="requires a RuntimeContext"):
        view.spell_cooldown_ready("shield")


# ---------------------------------------------------------------------------
# Entity channel -> TargetView projection
# ---------------------------------------------------------------------------


def test_entity_channel_projects_to_target_views() -> None:
    entities = (_make_entity(entity_id="a", hp_fraction=0.4, distance=6.0, threat=3.0),)

    views = GameStateAdapter(_make_populated_state(entities=entities)).to_targeting_views()

    assert len(views) == 1
    assert isinstance(views[0], TargetView)
    assert views[0].entity_id == "a"
    assert views[0].distance == 6.0
    assert views[0].threat == 3.0
    assert views[0].hp_percent == 40.0  # derived from the canonical fraction
    assert views[0].is_alive is True


def test_entity_is_alive_derived_when_unobserved() -> None:
    entities = (_make_entity(hp_fraction=0.0, is_alive=None),)

    views = GameStateAdapter(_make_populated_state(entities=entities)).to_targeting_views()

    assert views[0].is_alive is False


def test_entity_without_observed_threat_blocks_targeting() -> None:
    """ADR-002 unresolved question 3: no threat derivation exists, so it raises."""
    entities = (_make_entity(threat=None),)

    with pytest.raises(AdapterIncompleteError, match="non-Optional threat"):
        GameStateAdapter(_make_populated_state(entities=entities)).to_targeting_views()


def test_empty_entity_channel_falls_back_to_selected_target() -> None:
    """An empty channel is 'not observed', so the real target is still projected."""
    state = _make_populated_state(entities=())
    object.__setattr__(state, "threat", 5.0)
    object.__setattr__(state, "is_attackable", True)
    object.__setattr__(state, "is_alive", True)
    object.__setattr__(state, "is_in_combat_with_self", True)

    views = GameStateAdapter(state).to_targeting_views()

    assert len(views) == 1
    assert views[0].entity_id == "mob-a"


# ---------------------------------------------------------------------------
# Determinism
# ---------------------------------------------------------------------------


def test_derivation_is_deterministic() -> None:
    """Identical snapshot + config + context must yield identical views."""
    context = StaticRuntimeContext(ready_spells=frozenset({"shield"}))
    first = GameStateAdapter(_make_populated_state(), config=_config(), context=context)
    second = GameStateAdapter(_make_populated_state(), config=_config(), context=context)

    assert first.to_reactive_view() == second.to_reactive_view()
    assert first.to_flee_view() == second.to_flee_view()
    assert first.to_targeting_views() == second.to_targeting_views()


def test_derivation_does_not_mutate_the_snapshot() -> None:
    state = _make_populated_state()
    before = state.to_dict()
    adapter = GameStateAdapter(state, config=_config(), context=StaticRuntimeContext())

    adapter.to_world_sync_view()
    adapter.to_targeting_views()
    adapter.to_reactive_view()
    adapter.to_flee_view()

    assert state.to_dict() == before


# ---------------------------------------------------------------------------
# Honest projection outcome for a fully-observed snapshot
# ---------------------------------------------------------------------------


def test_projection_outcome_with_fully_populated_snapshot() -> None:
    """Records exactly which views a fully-observed snapshot drives today.

    Four of eight now project. The other four are each blocked by one specific
    unobserved field, and stay fail-loud rather than inventing a value:

    - ``resource_max``      -- ADR-002 unresolved question 2 (no class channel)
    - ``target_is_lootable``-- channel-pending, no vision channel yet (T-FIX-23)
    - ``inventory_count``   -- channel-pending, no vision channel yet (T-FIX-23)
    """
    adapter = GameStateAdapter(
        _make_populated_state(), config=_config(), context=StaticRuntimeContext()
    )

    # Projectable: every non-Optional field is observed or derivable.
    assert adapter.to_world_sync_view() is not None
    assert len(adapter.to_targeting_views()) == 1
    assert adapter.to_reactive_view() is not None
    assert adapter.to_flee_view() is not None

    with pytest.raises(AdapterIncompleteError, match="non-Optional resource_max"):
        adapter.to_strategist_view(fsm_state="COMBAT")
    with pytest.raises(AdapterIncompleteError, match="non-Optional resource_max"):
        adapter.to_combat_view()
    with pytest.raises(AdapterIncompleteError, match="non-Optional target_is_lootable"):
        adapter.to_loot_view()
    with pytest.raises(AdapterIncompleteError, match="non-Optional inventory_count"):
        adapter.to_vendor_view()


def test_mock_shaped_entities_block_targeting_on_threat() -> None:
    """MockPerception emits ``threat=None``, so its entity channel is not convertible."""
    entities = (_make_entity(threat=None),)
    adapter = GameStateAdapter(_make_populated_state(entities=entities), config=_config())

    with pytest.raises(AdapterIncompleteError, match="non-Optional threat"):
        adapter.to_targeting_views()


# ---------------------------------------------------------------------------
# Static import guards for the new modules
# ---------------------------------------------------------------------------


def test_new_perception_modules_avoid_forbidden_imports() -> None:
    forbidden_modules = {
        "aiosqlite",
        "asyncio",
        "random",
        "threading",
        "time",
        "wow_bot.main",
        "wow_bot.lab.runner_v2",
    }
    forbidden_substrings = {"ollama", "openai", "anthropic", "llm"}

    for name in ("context.py", "resource_table.py"):
        path = Path("src/wow_bot/perception") / name
        assert path.exists(), f"{path} must exist"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                modules = [node.module or ""]
            else:
                continue
            for module in modules:
                assert module not in forbidden_modules, f"{name} imports {module}"
                assert not any(sub in module.lower() for sub in forbidden_substrings), (
                    f"{name} imports {module}"
                )
                if module.startswith("wow_bot.") and not module.startswith("wow_bot.perception"):
                    assert module == "wow_bot.shared.interfaces", (
                        f"{name} cannot import {module}"
                    )



