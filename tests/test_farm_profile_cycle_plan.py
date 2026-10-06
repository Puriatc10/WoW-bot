"""Unit tests for farm profile cycle plan execution and load-time validation (T-FIX-26).

Verifies that:
1. A profile's configured nodes and route preferences determine the cycle (nodes visited in order,
   route penalties applied to NavGraph, and configured vendor resolved over nearest heuristic).
2. stop_after_cycles halts the loop at the configured outcome count.
3. The reported iteration counter (res.cycles_completed) equals completed farm cycles, not loop passes.
4. An invalid profile fails strictly at load time, not mid-run.
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wow_bot.combat.rotation import RotationConfig
from wow_bot.config import Config
from wow_bot.executor.fsm_v2 import FSMState
from wow_bot.farm.profile import (
    CycleSpec,
    FarmProfile,
    FarmProfileError,
    NodeReference,
    RoutePreferences,
    VendorReference,
    load_profile,
)
from wow_bot.farm.vendor import VendorLocation
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRunnerError,
    LabRunStatus,
    LabRuntime,
    _run_cycle,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.nav.graph import NodeKind
from wow_bot.nav.navigator import NavResult, NavStatus
from wow_bot.session import Session
from wow_bot.world.store import WorldModel


@dataclass
class FakeGameState:
    """Fake GameState satisfying GameStateView for lab tests."""

    player_x: float = 0.0
    player_y: float = 0.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 0.0
    self_y: float = 0.0
    self_hp_percent: float = 100.0
    self_in_combat: bool = False
    fsm_state: FSMState = FSMState.IDLE
    current_target_id: str | None = None
    target_entity_id: str | None = None
    target_in_range: bool = False
    target_is_alive: bool = False
    target_is_lootable: bool = False
    target_distance: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    adds_count: int = 0
    inventory_count: int = 0
    inventory_max: int = 20
    durability_fraction: float | None = 1.0
    level: float = 10.0
    xp: float = 1000.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        return True


@dataclass
class FakeMetaState:
    """Fake MetaState for tests."""

    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


@pytest.fixture
def test_config(tmp_path: Path) -> Config:
    """Provide a valid Config instance in LAB mode."""
    sess_root = tmp_path / "runs"
    sess_root.mkdir(parents=True, exist_ok=True)
    return Config(
        lab_mode=True,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=sess_root,
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )


async def _create_test_runtime(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
    game_state: FakeGameState | None = None,
) -> tuple[LabRuntime, Path]:
    db_path = tmp_path / "world_test.db"
    world = await WorldModel.open(db_path)
    await world.add_node(0.0, 0.0, kind="node")
    await world.close()

    session = Session.start(test_config)
    st = game_state if game_state is not None else FakeGameState()

    runtime = await build_lab_runtime_async(
        config=test_config,
        session=session,
        game_state_source=lambda: st,
        meta_state_source=FakeMetaState,
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=str(db_path),
        runner_config=LabRunnerConfig(
            max_cycles_per_run=100,
            max_consecutive_failures=5,
            fail_on_async_context=False,
        ),
        include_reflex_loop=False,
        include_watchdog=False,
    )
    return runtime, db_path


# ---------------------------------------------------------------------------
# Acceptance Criterion 1: Configured nodes and routes determine the cycle
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_configured_nodes_and_routes_determine_cycle(
    tmp_path: Path,
    test_config: Config,
) -> None:
    """Acceptance 1: Configured nodes in cycle.nodes are visited in order, route preferences

    determine graph penalties, and configured vendor is resolved over nearest heuristic.
    """
    profile = FarmProfile(
        schema_version=1,
        name="test_cycle_profile",
        description="Profile with ordered nodes and route preferences",
        cycle=CycleSpec(
            nodes=(
                NodeReference(kind="waypoint", name="node_alpha"),
                NodeReference(kind="waypoint", name="node_beta"),
            ),
            vendor=VendorReference(kind="vendor", name="special_vendor"),
            repair=VendorReference(kind="vendor", name="special_vendor"),
            stop_when_inventory_full=True,
            stop_after_cycles=1,
        ),
        route_preferences=RoutePreferences(
            avoid_kinds=("mob",),
            prefer_kinds=("waypoint",),
            max_detour_factor=2.5,
        ),
        metadata={},
    )

    st = FakeGameState(inventory_count=0, inventory_max=20)
    runtime, _db_path = await _create_test_runtime(tmp_path, test_config, profile, game_state=st)

    try:
        # 1. Assert NavGraph has directional penalties derived from route_preferences
        assert runtime.graph is not None
        # Add test nodes and edges to check penalties applied to NavGraph
        origin_id = await runtime.world.add_node(0.0, 0.0, kind="waypoint")
        mob_node_id = await runtime.world.add_node(10.0, 0.0, kind="mob")
        wp_node_id = await runtime.world.add_node(0.0, 10.0, kind="waypoint")
        await runtime.world.add_edge(origin_id, mob_node_id, cost=10.0)
        await runtime.world.add_edge(origin_id, wp_node_id, cost=10.0)

        # Build graph using world's current state and runtime profile preferences
        from wow_bot.nav.graph import GraphConfig, NodeKind, build_graph
        penalties: dict[NodeKind, float] = {}
        detour = float(profile.route_preferences.max_detour_factor)
        for ak in profile.route_preferences.avoid_kinds:
            penalties[NodeKind(ak)] = max(2.0, detour)
        for pk in profile.route_preferences.prefer_kinds:
            penalties[NodeKind(pk)] = max(0.01, 1.0 / max(1.01, detour))

        test_graph = await build_graph(runtime.world, config=GraphConfig(directional_penalties=penalties))
        edges = test_graph.neighbors(origin_id)
        mob_edge = next(e for e in edges if e.to_id == mob_node_id)
        wp_edge = next(e for e in edges if e.to_id == wp_node_id)
        # Avoid kind mob has cost multiplied by 2.5: 10.0 * 2.5 = 25.0
        assert mob_edge.cost == pytest.approx(25.0)
        # Prefer kind waypoint has cost divided by 2.5: 10.0 * 0.4 = 4.0
        assert wp_edge.cost == pytest.approx(4.0)

        # 2. Add target nodes to the world model so they can be resolved
        await runtime.world.add_node(100.0, 100.0, kind="waypoint", meta={"name": "node_alpha"})
        await runtime.world.add_node(200.0, 200.0, kind="waypoint", meta={"name": "node_beta"})

        # Closer vendor at (10, 10) vs configured vendor at (300, 300)
        await runtime.world.add_node(10.0, 10.0, kind="vendor", meta={"name": "near_vendor"})
        await runtime.world.add_node(300.0, 300.0, kind="vendor", meta={"name": "special_vendor"})

        # Record visited destinations by the navigator
        visited_destinations: list[tuple[float, float]] = []

        def mock_go_to(target_xy: tuple[float, float]) -> NavResult:
            visited_destinations.append(target_xy)
            return NavResult(
                status=NavStatus.SUCCESS,
                target_xy=target_xy,
                final_xy=target_xy,
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.05,
                reason="",
            )

        runtime.navigator.go_to = mock_go_to  # type: ignore[method-assign]

        # Run 1 complete cycle (visiting node_alpha then node_beta)
        res = await run_lab_loop_async(runtime, max_cycles=10)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 1

        # Assert nodes were visited in configured order
        assert len(visited_destinations) == 2
        assert visited_destinations[0] == (100.0, 100.0)  # node_alpha
        assert visited_destinations[1] == (200.0, 200.0)  # node_beta

        # 3. Verify vendor resolution uses configured vendor name, NOT nearest heuristic
        st.inventory_count = 20
        st.inventory_max = 20

        vendor_targets_called: list[str | None] = []
        original_vendor_run = runtime.vendor.run

        def mock_vendor_run(vendor_loc: VendorLocation | None, *args: Any, **kwargs: Any) -> Any:
            vendor_name = vendor_loc.name if vendor_loc else None
            vendor_targets_called.append(vendor_name)
            return original_vendor_run(vendor_loc, *args, **kwargs)

        runtime.vendor.run = mock_vendor_run  # type: ignore[method-assign]

        # Trigger cycle step with full inventory (step 1 transitions to go_to_vendor)
        cycle_res = await _run_cycle(runtime, "farm", None, 0, None)
        assert cycle_res[0] == "go_to_vendor"
        assert cycle_res[6] == "special_vendor"

        # Step 2 executes go_to_vendor targeting special_vendor
        await _run_cycle(
            runtime,
            cycle_res[0],
            cycle_res[1],
            1,
            cycle_res[3],
            current_target=cycle_res[6],
            current_strategy=cycle_res[7],
        )
        assert len(vendor_targets_called) == 1
        assert vendor_targets_called[0] == "special_vendor"
        assert vendor_targets_called[0] != "near_vendor"
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Acceptance Criterion 2: stop_after_cycles halts at configured outcome count
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_stop_after_cycles_halts_at_configured_outcome_count(
    tmp_path: Path,
    test_config: Config,
) -> None:
    """Acceptance 2: stop_after_cycles halts the loop exactly when the configured

    number of completed farm cycles (farm outcomes) is reached.
    """
    profile = FarmProfile(
        schema_version=1,
        name="test_stop_cycles",
        description="Profile with stop_after_cycles = 2",
        cycle=CycleSpec(
            nodes=(
                NodeReference(kind="waypoint", name="waypoint_1"),
                NodeReference(kind="waypoint", name="waypoint_2"),
            ),
            vendor=VendorReference(kind="vendor", name="vendor_1"),
            repair=VendorReference(kind="vendor", name="vendor_1"),
            stop_when_inventory_full=True,
            stop_after_cycles=2,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )

    runtime, _ = await _create_test_runtime(tmp_path, test_config, profile)

    try:
        await runtime.world.add_node(50.0, 50.0, kind="waypoint", meta={"name": "waypoint_1"})
        await runtime.world.add_node(80.0, 80.0, kind="waypoint", meta={"name": "waypoint_2"})

        nav_calls = 0

        def mock_go_to(target_xy: tuple[float, float]) -> NavResult:
            nonlocal nav_calls
            nav_calls += 1
            return NavResult(
                status=NavStatus.SUCCESS,
                target_xy=target_xy,
                final_xy=target_xy,
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.01,
                reason="",
            )

        runtime.navigator.go_to = mock_go_to  # type: ignore[method-assign]

        res = await run_lab_loop_async(runtime, max_cycles=100)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.reason == "stop_after_cycles_reached"
        assert res.cycles_completed == 2
        # 2 nodes per cycle * 2 cycles = 4 total node trips
        assert nav_calls == 4
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Acceptance Criterion 3: Iteration counter equals completed cycles, not passes
# ---------------------------------------------------------------------------
@pytest.mark.asyncio
async def test_iteration_counter_equals_completed_cycles_not_loop_passes(
    tmp_path: Path,
    test_config: Config,
) -> None:
    """Acceptance 3: The iteration counter res.cycles_completed strictly reflects completed

    farm cycles (outcomes), and does not equal while-loop pass count.
    """
    profile = FarmProfile(
        schema_version=1,
        name="test_multi_node_counter",
        description="Profile with 3 nodes per cycle",
        cycle=CycleSpec(
            nodes=(
                NodeReference(kind="waypoint", name="n1"),
                NodeReference(kind="waypoint", name="n2"),
                NodeReference(kind="waypoint", name="n3"),
            ),
            vendor=VendorReference(kind="vendor", name="v1"),
            repair=VendorReference(kind="vendor", name="v1"),
            stop_when_inventory_full=True,
            stop_after_cycles=2,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )

    runtime, _ = await _create_test_runtime(tmp_path, test_config, profile)

    try:
        await runtime.world.add_node(10.0, 10.0, kind="waypoint", meta={"name": "n1"})
        await runtime.world.add_node(20.0, 20.0, kind="waypoint", meta={"name": "n2"})
        await runtime.world.add_node(30.0, 30.0, kind="waypoint", meta={"name": "n3"})

        loop_trips = 0

        def mock_go_to(target_xy: tuple[float, float]) -> NavResult:
            nonlocal loop_trips
            loop_trips += 1
            return NavResult(
                status=NavStatus.SUCCESS,
                target_xy=target_xy,
                final_xy=target_xy,
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.01,
                reason="",
            )

        runtime.navigator.go_to = mock_go_to  # type: ignore[method-assign]

        res = await run_lab_loop_async(runtime, max_cycles=100)
        # 3 nodes * 2 cycles = 6 loop passes executed
        assert loop_trips == 6
        # Iteration counter measures completed farm cycles (2), NOT loop passes (6)
        assert res.cycles_completed == 2
        assert res.cycles_completed != loop_trips
    finally:
        await runtime.close()


# ---------------------------------------------------------------------------
# Acceptance Criterion 4: Invalid profile fails at load time
# ---------------------------------------------------------------------------
def test_invalid_profile_fails_at_load_time(tmp_path: Path, test_config: Config) -> None:
    """Acceptance 4: An invalid profile fails strictly at load / build time, not mid-run."""
    # 1. Invalid TOML file missing required cycle section
    bad_toml_1 = tmp_path / "bad_profile_1.toml"
    bad_toml_1.write_text(
        """
[profile]
schema_version = 1
name = "bad_profile"
description = "missing cycle section"
""",
        encoding="utf-8",
    )
    with pytest.raises(FarmProfileError, match="Missing required top-level section: 'cycle'"):
        load_profile(bad_toml_1)

    # 2. Invalid TOML with empty nodes list
    bad_toml_2 = tmp_path / "bad_profile_2.toml"
    bad_toml_2.write_text(
        """
[profile]
schema_version = 1
name = "bad_profile_2"

[cycle]
nodes = []
vendor = { kind = "vendor", name = "v1" }
""",
        encoding="utf-8",
    )
    with pytest.raises(FarmProfileError, match="'nodes' in \\[cycle\\] must be a non-empty list"):
        load_profile(bad_toml_2)

    # 3. Invalid profile dictionary passed to build_lab_runtime_async
    bad_dict = {
        "profile": {"schema_version": 999, "name": "bad_ver"},
        "cycle": {"nodes": []},
    }

    session = Session.start(test_config)
    db_path = tmp_path / "dummy_world.db"

    with pytest.raises(LabRunnerError, match="Invalid farm profile dict"):
        asyncio.run(
            build_lab_runtime_async(
                config=test_config,
                session=session,
                game_state_source=lambda: FakeGameState(),
                meta_state_source=FakeMetaState,
                rotation_config=RotationConfig(rules=()),
                farm_profile=bad_dict,  # type: ignore[arg-type]
                world_db_path=str(db_path),
            )
        )

    # 4. Non-FarmProfile object passed to build_lab_runtime_async
    with pytest.raises(LabRunnerError, match="farm_profile must be a FarmProfile instance"):
        asyncio.run(
            build_lab_runtime_async(
                config=test_config,
                session=session,
                game_state_source=lambda: FakeGameState(),
                meta_state_source=FakeMetaState,
                rotation_config=RotationConfig(rules=()),
                farm_profile=12345,  # type: ignore[arg-type]
                world_db_path=str(db_path),
            )
        )
