"""Unit tests for PerceptionBackend protocol and GameStateAdapter."""

from __future__ import annotations

import ast
import builtins
from pathlib import Path
from unittest.mock import patch

import pytest

from wow_bot.combat.flee import FleeStateView
from wow_bot.executor.states import FSMState
from wow_bot.farm.vendor import VendorStateView
from wow_bot.perception.adapter import (
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
    fraction_to_percent,
)
from wow_bot.perception.context import StaticRuntimeContext
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.perception.views import (
    FleeView,
    LootView,
    ReactiveView,
    TargetView,
    VendorView,
    WorldSyncView,
)
from wow_bot.shared.interfaces import EnemyInfo, GameState, TargetInfo
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
    """Helper creating a well-formed GameState with all fields natively populated."""
    actual_target: TargetInfo | None
    entities: list[EnemyInfo] = []
    
    if target is _SENTINEL:
        actual_target = TargetInfo(
            name="TargetMob", hp_pct=0.50, reaction="hostile", distance_estimate=10.0
        )
        entities.append(EnemyInfo(
            bbox=(10, 20, 30, 40), confidence=0.95, distance_estimate=10.0,
            entity_id="TargetMob", kind="mob", x=110.0, y=210.0, z=0.0,
            hp_fraction=0.50, threat=10.0, is_attackable=True, is_alive=True,
            is_in_combat_with_self=True
        ))
    elif isinstance(target, TargetInfo):
        actual_target = target
        entities.append(EnemyInfo(
            bbox=(10, 20, 30, 40), confidence=0.95, distance_estimate=target.distance_estimate,
            entity_id=target.name, kind="mob", x=position[0] + 10.0, y=position[1] + 10.0,
            z=0.0, hp_fraction=target.hp_pct, threat=10.0, is_attackable=True,
            is_alive=target.hp_pct > 0, is_in_combat_with_self=True
        ))
    else:
        actual_target = None

    return GameState(
        timestamp=1000.0, hp_pct=hp_pct, mana_pct=mana_pct, position=position,
        facing=facing, in_combat=in_combat, target=actual_target, enemies=[], events=[],
        player_z=0.0, inventory_count=5, inventory_max=20, level_or_xp=1000.0,
        target_is_lootable=True, entities=tuple(entities)
    )


# ---------------------------------------------------------------------------
# 1. Structural and Type Compatibility Tests (Well-formed State)
# ---------------------------------------------------------------------------


def test_well_formed_game_state_does_not_raise() -> None:
    """A well-formed GameState does NOT raise for any of the eight projections."""
    state = _make_well_formed_state()
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())

    ws_view = adapter.to_world_sync_view()
    assert isinstance(ws_view, WorldSyncView)

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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
    view = adapter.to_world_sync_view()

    assert isinstance(view, WorldSyncView)
    assert isinstance(view, GameStateLike)
    assert isinstance(view.player_x, float)
    assert isinstance(view.player_y, float)
    assert isinstance(view.player_z, float)
    assert view.player_x == 12.5
    assert view.player_y == 34.5
    assert view.player_z == 0.0
    assert len(view.entities) == 1
    assert view.target_entity_id == "TargetMob"








def test_targeting_views_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state()
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
    view = adapter.to_reactive_view()

    assert isinstance(view, ReactiveView)
    assert isinstance(view.self_hp_percent, float)
    assert isinstance(view.self_in_combat, bool)
    assert isinstance(view.target_in_range, bool)
    assert isinstance(view.self_x, float)
    assert isinstance(view.self_y, float)
    assert isinstance(view.incoming_casts, tuple)
    assert callable(view.spell_cooldown_ready)


def test_flee_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(hp_pct=0.15, position=(50.0, 60.0))
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
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
    assert view.adds_count == 1


def test_loot_view_structural_and_type_compatibility() -> None:
    state = _make_well_formed_state(position=(70.0, 80.0))
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())

    with pytest.raises(AdapterIncompleteError, match="WorldSyncView requires non-Optional position"):
        adapter.to_world_sync_view()


def test_vendor_view_missing_non_optional_fields_raises() -> None:
    # Missing inventory_count
    state = _make_well_formed_state()
    object.__setattr__(state, "inventory_count", None)
    with pytest.raises(AdapterIncompleteError, match="VendorStateView requires non-Optional inventory_count"):
        GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext()).to_vendor_view()


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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())
    rx = adapter.to_reactive_view()
    assert rx.self_hp_percent == 12.6


# ---------------------------------------------------------------------------
# 4. Determinism
# ---------------------------------------------------------------------------


def test_determinism() -> None:
    state = _make_well_formed_state()
    adapter = _canonical_adapter(state)
    assert adapter.to_world_sync_view() == adapter.to_world_sync_view()
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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())

    for _ in range(3):
        _ = adapter.to_world_sync_view()
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
    adapter = GameStateAdapter(state, config=AdapterDerivationConfig(engage_distance_units=30.0), context=StaticRuntimeContext())

    with (
        patch.object(Path, "read_text", side_effect=forbidden_read),
        patch.object(Path, "read_bytes", side_effect=forbidden_read),
        patch.object(builtins, "open", side_effect=forbidden_read),
    ):
        _ = adapter.to_world_sync_view()
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
# 9. Per-View Projection Expectations (T-FIX-24)
# ---------------------------------------------------------------------------


def _make_canonical_snapshot(
    *,
    entities: tuple[EnemyInfo, ...] | None = None,
    target: TargetInfo | None = None,
    player_z: float = 1.5,
) -> GameState:
    """Construct a canonical extended GameState without monkeypatching or fakes.

    Uses only the fields defined in ``wow_bot.shared.interfaces.GameState`` and
    ``EnemyInfo``, reflecting the post-ADR-002 extended schema.
    """
    if target is None:
        target = TargetInfo(
            name="TargetMob",
            hp_pct=0.50,
            reaction="hostile",
            distance_estimate=10.0,
        )
    if entities is None:
        entities = (
            EnemyInfo(
                bbox=(10, 20, 30, 40),
                confidence=0.95,
                distance_estimate=10.0,
                entity_id="mob-1",
                kind="mob",
                x=110.0,
                y=210.0,
                z=1.5,
                hp_fraction=0.50,
                threat=100.0,
                is_attackable=True,
                is_alive=True,
                is_in_combat_with_self=True,
            ),
        )
    return GameState(
        timestamp=1000.0,
        hp_pct=0.85,
        mana_pct=0.60,
        position=(100.0, 200.0),
        facing=1.57,
        in_combat=True,
        target=target,
        enemies=list(entities),
        events=[],
        player_z=player_z,
        entities=entities,
    )


def _canonical_adapter(
    state: GameState | None = None,
) -> GameStateAdapter:
    if state is None:
        state = _make_canonical_snapshot()
    return GameStateAdapter(
        state,
        config=AdapterDerivationConfig(engage_distance_units=30.0),
        context=StaticRuntimeContext(),
    )


def test_projection_to_world_sync_view_succeeds() -> None:
    adapter = _canonical_adapter()
    view = adapter.to_world_sync_view()
    assert isinstance(view, WorldSyncView)
    assert isinstance(view, GameStateLike)
    assert view.player_x == 100.0
    assert view.player_y == 200.0
    assert view.player_z == 1.5
    assert view.target_entity_id == "TargetMob"
    assert len(view.entities) == 1


def test_projection_to_targeting_views_succeeds() -> None:
    adapter = _canonical_adapter()
    views = adapter.to_targeting_views()
    assert len(views) == 1
    t_view = views[0]
    assert isinstance(t_view, TargetView)
    assert t_view.entity_id == "mob-1"
    assert t_view.distance == 10.0
    assert t_view.threat == 100.0
    assert t_view.hp_percent == 50.0
    assert t_view.is_attackable is True
    assert t_view.is_alive is True
    assert t_view.is_in_combat_with_self is True


def test_projection_to_reactive_view_succeeds() -> None:
    adapter = _canonical_adapter()
    view = adapter.to_reactive_view()
    assert isinstance(view, ReactiveView)
    assert view.self_hp_percent == 85.0
    assert view.self_in_combat is True
    assert view.target_in_range is True
    assert view.self_x == 100.0
    assert view.self_y == 200.0
    assert view.incoming_casts == ()
    assert view.current_target_id == "TargetMob"


def test_projection_to_flee_view_succeeds() -> None:
    adapter = _canonical_adapter()
    view = adapter.to_flee_view()
    assert isinstance(view, FleeView)
    assert view.self_hp_percent == 85.0
    assert view.self_x == 100.0
    assert view.self_y == 200.0
    assert view.adds_count == 1
    assert view.current_target_id == "TargetMob"


def test_projection_to_strategist_view_fails_loud_on_missing_resource_max() -> None:
    adapter = _canonical_adapter()
    with pytest.raises(
        AdapterIncompleteError,
        match="StrategistView requires non-Optional resource_max",
    ):
        adapter.to_strategist_view(fsm_state=FSMState.IDLE)


def test_projection_to_combat_view_fails_loud_on_missing_resource_max() -> None:
    adapter = _canonical_adapter()
    with pytest.raises(
        AdapterIncompleteError,
        match="CombatStateView requires non-Optional resource_max",
    ):
        adapter.to_combat_view()


def test_projection_to_loot_view_fails_loud_on_missing_target_is_lootable() -> None:
    adapter = _canonical_adapter()
    with pytest.raises(
        AdapterIncompleteError,
        match="LootStateView requires non-Optional target_is_lootable",
    ):
        adapter.to_loot_view()


def test_projection_to_vendor_view_fails_loud_on_missing_inventory_count() -> None:
    adapter = _canonical_adapter()
    with pytest.raises(
        AdapterIncompleteError,
        match="VendorStateView requires non-Optional inventory_count",
    ):
        adapter.to_vendor_view()


# ---------------------------------------------------------------------------
# 10. End-to-end Projection (T-FIX-04)
# ---------------------------------------------------------------------------

from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.perception.mock_backend import MockPerceptionBackend


@pytest.mark.asyncio
async def test_end_to_end_projection_from_mock() -> None:
    mock = MockPerception(combat_on_duration=10.0, max_enemies=1)
    backend = MockPerceptionBackend(mock)
    
    while True:
        snapshot = await backend.snapshot()
        if snapshot.in_combat and snapshot.target is not None and snapshot.entities:
            break
            
    adapter = GameStateAdapter(
        snapshot, 
        config=AdapterDerivationConfig(engage_distance_units=30.0),
        context=StaticRuntimeContext()
    )
    
    # 1. WorldSyncView (blocked by player_z = None in mock)
    with pytest.raises(AdapterIncompleteError, match="WorldSyncView requires non-Optional player_z"):
        adapter.to_world_sync_view()
        
    # 2. StrategistView (blocked by resource_max)
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional resource_max"):
        adapter.to_strategist_view(fsm_state="IDLE")
        
    # 3. CombatView (blocked by resource_max)
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional resource_max"):
        adapter.to_combat_view()
        
    # 4. TargetView (blocked by threat, because MockPerception sets threat=None)
    try:
        views = adapter.to_targeting_views()
        raise AssertionError(f"Expected to raise, but got: {views} with entities: {snapshot.entities}")
    except AdapterIncompleteError as e:
        assert "TargetEntityLike requires non-Optional threat" in str(e)
        
    # 5. ReactiveView (unblocked)
    rx_view = adapter.to_reactive_view()
    assert isinstance(rx_view, ReactiveView)
    
    # 6. FleeView (unblocked)
    flee_view = adapter.to_flee_view()
    assert isinstance(flee_view, FleeView)
    
    # 7. LootView
    try:
        adapter.to_loot_view()
    except AdapterIncompleteError as e:
        assert "LootStateView requires non-Optional" in str(e)
        
    # 8. VendorView
    try:
        adapter.to_vendor_view()
    except AdapterIncompleteError as e:
        assert "VendorStateView requires non-Optional" in str(e)

@pytest.mark.asyncio
async def test_mock_scenario_dead_target() -> None:
    mock = MockPerception(combat_on_duration=10.0, max_enemies=1)
    mock._scenario = "dead_target_scenario"
    backend = MockPerceptionBackend(mock)
    for _ in range(200):
        snapshot = await backend.snapshot()
        if snapshot.in_combat and snapshot.target is not None:
            assert snapshot.target.hp_pct == 0.0
            break
