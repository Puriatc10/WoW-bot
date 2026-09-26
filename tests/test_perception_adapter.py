"""Unit tests for PerceptionBackend protocol and GameStateAdapter."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from wow_bot.combat.flee import FleeStateView
from wow_bot.combat.loop import CombatStateView
from wow_bot.executor.states import FSMState
from wow_bot.farm.vendor import VendorStateView
from wow_bot.perception.adapter import (
    AdapterIncompleteError,
    GameStateAdapter,
    fraction_to_percent,
)
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.perception.views import (
    CombatView,
    FleeView,
    LootView,
    ReactiveView,
    StrategistView,
    TargetView,
    VendorView,
    WorldSyncView,
)
from wow_bot.shared.interfaces import GameState, TargetInfo
from wow_bot.world.sync import GameStateLike


def _make_game_state(
    *,
    hp_pct: float = 0.85,
    mana_pct: float = 0.60,
    position: tuple[float, float] = (100.0, 200.0),
    facing: float = 1.57,
    in_combat: bool = False,
    target: TargetInfo | None = None,
) -> GameState:
    """Helper creating a canonical GameState instance."""
    return GameState(
        timestamp=1000.0,
        hp_pct=hp_pct,
        mana_pct=mana_pct,
        position=position,
        facing=facing,
        in_combat=in_combat,
        target=target,
        enemies=[],
        events=[],
    )


_SENTINEL = object()


def _make_well_formed_state(
    *,
    hp_pct: float = 0.85,
    mana_pct: float = 0.60,
    position: tuple[float, float] = (100.0, 200.0),
    facing: float = 1.57,
    in_combat: bool = False,
    target: object = _SENTINEL,
) -> GameState:
    """Helper creating a well-formed GameState with all fields required by lab consumer protocols."""
    actual_target: TargetInfo | None
    if target is _SENTINEL:
        actual_target = TargetInfo(
            name="TargetMob", hp_pct=0.50, reaction="hostile", distance_estimate=10.0
        )
    elif isinstance(target, TargetInfo):
        actual_target = target
    else:
        actual_target = None

    state = _make_game_state(
        hp_pct=hp_pct,
        mana_pct=mana_pct,
        position=position,
        facing=facing,
        in_combat=in_combat,
        target=actual_target,
    )
    # Supply attributes required by consumer protocols
    object.__setattr__(state, "player_z", 0.0)
    object.__setattr__(state, "resource_max", 100.0)
    object.__setattr__(state, "inventory_count", 5)
    object.__setattr__(state, "inventory_max", 20)
    object.__setattr__(state, "level_or_xp", 1000.0)
    object.__setattr__(state, "target_in_range", True)
    object.__setattr__(state, "gcd_ready", True)
    object.__setattr__(state, "adds_count", 0)
    object.__setattr__(state, "target_is_alive", False)
    object.__setattr__(state, "target_is_lootable", True)
    object.__setattr__(state, "threat", 10.0)
    object.__setattr__(state, "is_attackable", True)
    object.__setattr__(state, "is_alive", True)
    object.__setattr__(state, "is_in_combat_with_self", True)
    return state


# ---------------------------------------------------------------------------
# 1. Structural and Type Compatibility Tests (Well-formed State)
# ---------------------------------------------------------------------------


def test_well_formed_game_state_does_not_raise() -> None:
    """A well-formed GameState does NOT raise for any of the eight projections."""
    state = _make_well_formed_state()
    adapter = GameStateAdapter(state)

    ws_view = adapter.to_world_sync_view()
    assert isinstance(ws_view, WorldSyncView)

    strat_view = adapter.to_strategist_view(fsm_state="IDLE")
    assert isinstance(strat_view, StrategistView)

    combat_view = adapter.to_combat_view()
    assert isinstance(combat_view, CombatView)

    targeting_views = adapter.to_targeting_views()
    assert len(targeting_views) == 1
    assert isinstance(targeting_views[0], TargetView)

    reactive_view = adapter.to_reactive_view()
    assert isinstance(reactive_view, ReactiveView)

    flee_view = adapter.to_flee_view()
    assert isinstance(flee_view, FleeView)

    loot_view = adapter.to_loot_view()
    assert isinstance(loot_view, LootView)

    vendor_view = adapter.to_vendor_view()
    assert isinstance(vendor_view, VendorView)


def test_world_sync_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(position=(12.5, 34.5))
    adapter = GameStateAdapter(state)
    view = adapter.to_world_sync_view()

    assert isinstance(view, WorldSyncView)
    assert isinstance(view, GameStateLike)
    assert isinstance(view.player_x, float)
    assert isinstance(view.player_y, float)
    assert isinstance(view.player_z, float)
    assert view.player_x == 12.5
    assert view.player_y == 34.5
    assert view.player_z == 0.0
    assert view.entities == ()
    assert view.target_entity_id == "TargetMob"


def test_strategist_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(hp_pct=0.80, mana_pct=0.50, position=(10.0, 20.0))
    adapter = GameStateAdapter(state)
    view = adapter.to_strategist_view(fsm_state="SCANNING")

    assert isinstance(view, StrategistView)
    assert isinstance(view.player_x, float)
    assert isinstance(view.player_y, float)
    assert isinstance(view.player_z, float)
    assert isinstance(view.self_hp_percent, float)
    assert isinstance(view.resource, float)
    assert isinstance(view.resource_max, float)
    assert isinstance(view.inventory_count, int)
    assert isinstance(view.level_or_xp, float)
    assert isinstance(view.fsm_state, FSMState)
    assert view.fsm_state == FSMState.SCANNING
    assert view.fsm_state.value == "SCANNING"

    # Static Protocol attribute presence
    for attr in [
        "player_x",
        "player_y",
        "player_z",
        "self_hp_percent",
        "resource",
        "resource_max",
        "current_target_id",
        "target_hp_percent",
        "inventory_count",
        "level_or_xp",
        "fsm_state",
    ]:
        assert hasattr(view, attr)


def test_combat_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(hp_pct=0.90, mana_pct=0.70, position=(5.0, 15.0))
    adapter = GameStateAdapter(state)
    view = adapter.to_combat_view()

    assert isinstance(view, CombatView)
    assert isinstance(view, CombatStateView)
    assert isinstance(view.target_in_range, bool)
    assert isinstance(view.target_hp_percent, float)
    assert isinstance(view.self_hp_percent, float)
    assert isinstance(view.resource, float)
    assert isinstance(view.resource_max, float)
    assert isinstance(view.gcd_ready, bool)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert len(view.entities) == 1
    assert isinstance(view.entities[0], TargetView)
    assert view.current_target_id == "TargetMob"

    with pytest.raises(NotImplementedError):
        view.target_has_debuff("rend")
    with pytest.raises(NotImplementedError):
        view.spell_cooldown_ready("heroic_strike")


def test_targeting_views_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state()
    adapter = GameStateAdapter(state)
    views = adapter.to_targeting_views()

    assert len(views) == 1
    t_view = views[0]
    assert isinstance(t_view, TargetView)
    assert isinstance(t_view.entity_id, str)
    assert isinstance(t_view.distance, float)
    assert isinstance(t_view.threat, float)
    assert isinstance(t_view.hp_percent, float)
    assert isinstance(t_view.is_attackable, bool)
    assert isinstance(t_view.is_alive, bool)
    assert isinstance(t_view.is_in_combat_with_self, bool)

    for attr in [
        "entity_id",
        "distance",
        "threat",
        "hp_percent",
        "is_attackable",
        "is_alive",
        "is_in_combat_with_self",
    ]:
        assert hasattr(t_view, attr)


def test_reactive_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(hp_pct=0.40, position=(1.0, 2.0), in_combat=True)
    adapter = GameStateAdapter(state)
    view = adapter.to_reactive_view()

    assert isinstance(view, ReactiveView)
    assert isinstance(view.self_hp_percent, float)
    assert isinstance(view.self_in_combat, bool)
    assert isinstance(view.target_in_range, bool)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert isinstance(view.incoming_casts, tuple)
    assert callable(view.spell_cooldown_ready)
    with pytest.raises(NotImplementedError):
        view.spell_cooldown_ready("shield")


def test_flee_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(hp_pct=0.15, position=(50.0, 60.0))
    adapter = GameStateAdapter(state)
    view = adapter.to_flee_view()

    assert isinstance(view, FleeView)
    assert isinstance(view, FleeStateView)
    assert isinstance(view.self_hp_percent, float)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert isinstance(view.adds_count, int)
    assert view.self_hp_percent == 15.0
    assert view.self_x == 50.0
    assert view.self_y == 60.0
    assert view.adds_count == 0


def test_loot_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(position=(70.0, 80.0))
    adapter = GameStateAdapter(state)
    view = adapter.to_loot_view()

    assert isinstance(view, LootView)
    assert isinstance(view.target_is_alive, bool)
    assert isinstance(view.target_is_lootable, bool)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert isinstance(view.inventory_count, int)
    assert view.self_x == 70.0
    assert view.self_y == 80.0
    assert view.inventory_count == 5


def test_vendor_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(position=(90.0, 95.0))
    adapter = GameStateAdapter(state)
    view = adapter.to_vendor_view()

    assert isinstance(view, VendorView)
    assert isinstance(view, VendorStateView)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert isinstance(view.inventory_count, int)
    assert view.self_x == 90.0
    assert view.self_y == 95.0
    assert view.inventory_count == 5


# ---------------------------------------------------------------------------
# 2. Fail-Loud Missing Non-Optional Fields (AdapterIncompleteError)
# ---------------------------------------------------------------------------


def test_world_sync_view_missing_position_raises() -> None:
    """Missing position raises AdapterIncompleteError and does not return None."""
    state = _make_well_formed_state()
    object.__setattr__(state, "position", None)
    adapter = GameStateAdapter(state)

    with pytest.raises(AdapterIncompleteError, match="WorldSyncView requires non-Optional position"):
        adapter.to_world_sync_view()


def test_strategist_view_missing_non_optional_fields_raises() -> None:
    # 1. Missing position
    state = _make_well_formed_state()
    object.__setattr__(state, "position", None)
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional position"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 2. Missing hp_pct
    state = _make_well_formed_state()
    object.__setattr__(state, "hp_pct", None)
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional self_hp_percent"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 3. Missing mana_pct
    state = _make_well_formed_state()
    object.__setattr__(state, "mana_pct", None)
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional resource"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 4. Missing resource_max
    state = _make_well_formed_state()
    delattr(state, "resource_max")
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional resource_max"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 5. Missing inventory_count
    state = _make_well_formed_state()
    delattr(state, "inventory_count")
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional inventory_count"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 6. Missing level_or_xp
    state = _make_well_formed_state()
    delattr(state, "level_or_xp")
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional level_or_xp"):
        GameStateAdapter(state).to_strategist_view(fsm_state="IDLE")

    # 7. Invalid or missing fsm_state
    state = _make_well_formed_state()
    with pytest.raises(AdapterIncompleteError, match="Invalid fsm_state"):
        GameStateAdapter(state).to_strategist_view(fsm_state="INVALID_STATE")


def test_combat_view_missing_non_optional_fields_raises() -> None:
    # Missing target_in_range
    state = _make_well_formed_state()
    delattr(state, "target_in_range")
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional target_in_range"):
        GameStateAdapter(state).to_combat_view()

    # Missing gcd_ready
    state = _make_well_formed_state()
    delattr(state, "gcd_ready")
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional gcd_ready"):
        GameStateAdapter(state).to_combat_view()

    # Missing resource_max
    state = _make_well_formed_state()
    delattr(state, "resource_max")
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional resource_max"):
        GameStateAdapter(state).to_combat_view()

    # Missing target_hp_percent when no target present
    state = _make_well_formed_state(target=None)
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional target_hp_percent"):
        GameStateAdapter(state).to_combat_view()


def test_targeting_views_missing_non_optional_fields_raises() -> None:
    # When target is present, missing threat raises
    state = _make_well_formed_state()
    delattr(state, "threat")
    with pytest.raises(AdapterIncompleteError, match="TargetEntityLike requires non-Optional threat"):
        GameStateAdapter(state).to_targeting_views()

    # Missing is_attackable raises
    state = _make_well_formed_state()
    delattr(state, "is_attackable")
    with pytest.raises(AdapterIncompleteError, match="TargetEntityLike requires non-Optional is_attackable"):
        GameStateAdapter(state).to_targeting_views()

    # When target is None, returns empty tuple without raising
    state_no_target = _make_well_formed_state(target=None)
    assert GameStateAdapter(state_no_target).to_targeting_views() == ()


def test_reactive_view_missing_non_optional_fields_raises() -> None:
    # Missing in_combat
    state = _make_well_formed_state()
    object.__setattr__(state, "in_combat", None)
    with pytest.raises(AdapterIncompleteError, match="ReactiveStateView requires non-Optional self_in_combat"):
        GameStateAdapter(state).to_reactive_view()

    # Missing target_in_range
    state = _make_well_formed_state()
    delattr(state, "target_in_range")
    with pytest.raises(AdapterIncompleteError, match="ReactiveStateView requires non-Optional target_in_range"):
        GameStateAdapter(state).to_reactive_view()


def test_flee_view_missing_non_optional_fields_raises() -> None:
    # Missing adds_count
    state = _make_well_formed_state()
    delattr(state, "adds_count")
    with pytest.raises(AdapterIncompleteError, match="FleeStateView requires non-Optional adds_count"):
        GameStateAdapter(state).to_flee_view()


def test_loot_view_missing_non_optional_fields_raises() -> None:
    # target_is_alive is DERIVED from the canonical target HP fraction
    # (ADR-002 "derived" class, implemented by T-FIX-21), so removing an
    # injected override no longer raises: the derivation supplies the value.
    # The fixture's target has hp_pct == 0.50, so the target is alive.
    state = _make_well_formed_state()
    delattr(state, "target_is_alive")
    assert GameStateAdapter(state).to_loot_view().target_is_alive is True

    # With no target at all the derivation has no input, so the projection
    # still fails loudly rather than guessing liveness.
    state_no_target = _make_well_formed_state(target=None)
    delattr(state_no_target, "target_is_alive")
    with pytest.raises(AdapterIncompleteError, match="LootStateView requires non-Optional target_is_alive"):
        GameStateAdapter(state_no_target).to_loot_view()

    # Missing target_is_lootable
    state = _make_well_formed_state()
    delattr(state, "target_is_lootable")
    with pytest.raises(AdapterIncompleteError, match="LootStateView requires non-Optional target_is_lootable"):
        GameStateAdapter(state).to_loot_view()

    # Missing inventory_count
    state = _make_well_formed_state()
    delattr(state, "inventory_count")
    with pytest.raises(AdapterIncompleteError, match="LootStateView requires non-Optional inventory_count"):
        GameStateAdapter(state).to_loot_view()


def test_vendor_view_missing_non_optional_fields_raises() -> None:
    # Missing inventory_count
    state = _make_well_formed_state()
    delattr(state, "inventory_count")
    with pytest.raises(AdapterIncompleteError, match="VendorStateView requires non-Optional inventory_count"):
        GameStateAdapter(state).to_vendor_view()


# ---------------------------------------------------------------------------
# 3. HP/MP Unit Conversion & Rounding Policy
# ---------------------------------------------------------------------------


def test_fraction_to_percent_rounding_policy() -> None:
    """Document and verify round-half-to-even at 1 decimal place."""
    assert fraction_to_percent(None) is None
    assert fraction_to_percent(0.0) == 0.0
    assert fraction_to_percent(1.0) == 100.0
    assert fraction_to_percent(0.5) == 50.0

    # Half to even cases at 1 decimal place
    # 0.1255 * 100 = 12.55 -> rounds to 12.6 (nearest even at .5)
    assert fraction_to_percent(0.1255) == 12.6
    # 0.1245 * 100 = 12.45 -> rounds to 12.4 (nearest even at .5)
    assert fraction_to_percent(0.1245) == 12.4

    # Further even vs odd test cases
    assert fraction_to_percent(0.0125) == 1.2
    assert fraction_to_percent(0.0135) == 1.4
    assert fraction_to_percent(0.0145) == 1.4
    assert fraction_to_percent(0.0155) == 1.6


def test_adapter_hp_mp_conversion() -> None:
    target = TargetInfo(name="Target", hp_pct=0.1245, reaction="hostile", distance_estimate=10.0)
    state = _make_well_formed_state(hp_pct=0.1255, mana_pct=0.9999, target=target)
    adapter = GameStateAdapter(state)

    strat = adapter.to_strategist_view(fsm_state="IDLE")
    assert strat.self_hp_percent == 12.6
    assert strat.resource == 100.0
    assert strat.target_hp_percent == 12.4


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    state = _make_well_formed_state()
    adapter = GameStateAdapter(state)

    assert adapter.to_world_sync_view() == adapter.to_world_sync_view()
    assert (
        adapter.to_strategist_view(fsm_state="IDLE")
        == adapter.to_strategist_view(fsm_state="IDLE")
    )
    assert adapter.to_combat_view() == adapter.to_combat_view()
    assert adapter.to_targeting_views() == adapter.to_targeting_views()
    assert adapter.to_reactive_view() == adapter.to_reactive_view()
    assert adapter.to_flee_view() == adapter.to_flee_view()
    assert adapter.to_loot_view() == adapter.to_loot_view()
    assert adapter.to_vendor_view() == adapter.to_vendor_view()


# ---------------------------------------------------------------------------
# 5. Immutability / No Mutation
# ---------------------------------------------------------------------------


def test_no_mutation() -> None:
    state = _make_well_formed_state(hp_pct=0.75, mana_pct=0.60, position=(1.0, 2.0))
    adapter = GameStateAdapter(state)

    for _ in range(3):
        _ = adapter.to_world_sync_view()
        _ = adapter.to_strategist_view(fsm_state="IDLE")
        _ = adapter.to_combat_view()
        _ = adapter.to_targeting_views()
        _ = adapter.to_reactive_view()
        _ = adapter.to_flee_view()
        _ = adapter.to_loot_view()
        _ = adapter.to_vendor_view()

    assert state.hp_pct == 0.75
    assert state.mana_pct == 0.60
    assert state.position == (1.0, 2.0)
    assert state.target is not None
    assert state.target.hp_pct == 0.50


# ---------------------------------------------------------------------------
# 6. No File Reads at Import or Call Time
# ---------------------------------------------------------------------------


def test_no_file_reads() -> None:
    def forbidden_read(*args: object, **kwargs: object) -> None:
        raise RuntimeError("Forbidden file access detected")

    state = _make_well_formed_state()
    adapter = GameStateAdapter(state)

    with (
        patch.object(Path, "read_text", side_effect=forbidden_read),
        patch.object(Path, "read_bytes", side_effect=forbidden_read),
        patch.object(builtins, "open", side_effect=forbidden_read),
    ):
        _ = adapter.to_world_sync_view()
        _ = adapter.to_strategist_view(fsm_state="IDLE")
        _ = adapter.to_combat_view()
        _ = adapter.to_targeting_views()
        _ = adapter.to_reactive_view()
        _ = adapter.to_flee_view()
        _ = adapter.to_loot_view()
        _ = adapter.to_vendor_view()


# ---------------------------------------------------------------------------
# 7. Static AST Import Checks
# ---------------------------------------------------------------------------


def test_static_ast_import_restrictions() -> None:
    forbidden_modules = {
        "aiosqlite",
        "threading",
        "time",
        "random",
        "wow_bot.main",
        "wow_bot.lab.runner_v2",
    }
    forbidden_substrings = {"ollama", "openai", "anthropic", "llm"}

    files_to_check = [
        Path("src/wow_bot/perception/protocol.py"),
        Path("src/wow_bot/perception/adapter.py"),
        Path("src/wow_bot/perception/views.py"),
    ]

    for file_path in files_to_check:
        assert file_path.exists(), f"File {file_path} must exist"
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod_name = alias.name
                    assert mod_name not in forbidden_modules, (
                        f"{file_path} imports forbidden module {mod_name}"
                    )
                    assert not any(sub in mod_name.lower() for sub in forbidden_substrings), (
                        f"{file_path} imports forbidden module {mod_name}"
                    )
                    if file_path.name == "adapter.py":
                        assert mod_name != "asyncio", "adapter.py must not import asyncio"
            elif isinstance(node, ast.ImportFrom):
                mod_name = node.module or ""
                assert mod_name not in forbidden_modules, (
                    f"{file_path} imports from forbidden module {mod_name}"
                )
                assert not any(sub in mod_name.lower() for sub in forbidden_substrings), (
                    f"{file_path} imports from forbidden module {mod_name}"
                )
                if file_path.name == "adapter.py":
                    assert mod_name != "asyncio", "adapter.py must not import asyncio"

                # Check perception consumers and pre-lab modules outside shared.interfaces
                if mod_name.startswith("wow_bot.") and not mod_name.startswith("wow_bot.perception"):
                    allowed_modules = {"wow_bot.shared.interfaces"}
                    if file_path.name == "views.py":
                        # views.py imports exact FSMState enum
                        allowed_modules.add("wow_bot.executor.states")
                    assert mod_name in allowed_modules, (
                        f"{file_path} cannot import {mod_name}; only {allowed_modules} allowed"
                    )


# ---------------------------------------------------------------------------
# 8. PerceptionBackend ABC Test
# ---------------------------------------------------------------------------


def test_perception_backend_abc() -> None:
    with pytest.raises(TypeError):
        # PerceptionBackend is an ABC with abstract snapshot()
        cls = PerceptionBackend
        cls()

    class ConcreteBackend(PerceptionBackend):
        async def snapshot(self) -> GameState:
            return _make_game_state()

    backend = ConcreteBackend()
    assert isinstance(backend, PerceptionBackend)


# ---------------------------------------------------------------------------
# 9. End-to-End Projection From a Canonical GameState
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=True,
    reason=(
        "A strictly canonical GameState cannot supply several fields that the "
        "consumer Protocols declare non-Optional (player_z, resource_max, "
        "inventory_count, level_or_xp, target_in_range, gcd_ready, adds_count, "
        "target_is_alive, target_is_lootable, threat, is_attackable, is_alive, "
        "is_in_combat_with_self). Every projection therefore raises "
        "AdapterIncompleteError. See ADR-001, section 'The adapter cannot "
        "produce a successful projection today'. Remove this xfail once "
        "GameState is extended or a supplementary data channel exists, without "
        "which T-FIX-04 (MockPerceptionAdapter) has nothing to project."
    ),
)
def test_canonical_game_state_end_to_end_projection() -> None:
    """Every projection must succeed end-to-end starting from a canonical GameState.

    The fixture below uses only fields defined in ``wow_bot.shared.interfaces``:
    no extra attributes, no fakes, no monkeypatched attributes. It carries a
    target, matching the shape ``MockPerception`` emits during combat.
    """
    state = GameState(
        timestamp=1000.0,
        hp_pct=0.85,
        mana_pct=0.60,
        position=(100.0, 200.0),
        facing=1.57,
        in_combat=True,
        target=TargetInfo(
            name="TargetMob", hp_pct=0.50, reaction="hostile", distance_estimate=10.0
        ),
        enemies=[],
        events=[],
    )
    adapter = GameStateAdapter(state)

    assert isinstance(adapter.to_world_sync_view(), WorldSyncView)
    assert isinstance(adapter.to_strategist_view(fsm_state="IDLE"), StrategistView)
    assert isinstance(adapter.to_combat_view(), CombatView)
    assert len(adapter.to_targeting_views()) == 1
    assert isinstance(adapter.to_reactive_view(), ReactiveView)
    assert isinstance(adapter.to_flee_view(), FleeView)
    assert isinstance(adapter.to_loot_view(), LootView)
    assert isinstance(adapter.to_vendor_view(), VendorView)
