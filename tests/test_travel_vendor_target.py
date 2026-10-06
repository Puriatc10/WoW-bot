"""Unit tests for travel_to and vendor target resolution correctness (T-FIX-17)."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from wow_bot.combat.rotation import RotationConfig
from wow_bot.config import Config
from wow_bot.executor.fsm_v2 import FSMState
from wow_bot.farm.profile import (
    CycleSpec,
    FarmProfile,
    NodeReference,
    RoutePreferences,
    VendorReference,
)
from wow_bot.farm.vendor import VendorLocation, VendorStatus
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRuntime,
    _run_cycle,
    build_lab_runtime_async,
)
from wow_bot.nav.navigator import NavConfig, Navigator, NavResult, NavStatus
from wow_bot.session import Session
from wow_bot.strategist.orchestrator_v2 import (
    JsonStrategy,
    OrchestratorOutcome,
    OrchestratorResult,
)
from wow_bot.world.store import WorldModel


@dataclass
class FakeGameState:
    """Fake GameState satisfying GameStateView for lab tests."""

    player_x: float = 5.0
    player_y: float = 5.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 5.0
    self_y: float = 5.0
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
    inventory_count: int = 5
    inventory_max: int = 30
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


def read_session_events(session: Session) -> list[dict[str, Any]]:
    events_file = session.path / "events.jsonl"
    events = []
    if events_file.exists():
        with open(events_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
    return events


@pytest.fixture
def farm_profile() -> FarmProfile:
    return FarmProfile(
        schema_version=1,
        name="test_profile",
        description="test",
        cycle=CycleSpec(
            nodes=(NodeReference(kind="mob", name="m1"),),
            vendor=VendorReference(kind="vendor", name="v1"),
            repair=VendorReference(kind="vendor", name="v1"),
            stop_when_inventory_full=True,
            stop_after_cycles=0,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )


async def _create_test_runtime(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
    game_state: FakeGameState | None = None,
) -> tuple[LabRuntime, Path]:
    db_path = tmp_path / "world_test.db"
    world = await WorldModel.open(db_path)
    # Add a base node so NavGraph isn't empty
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
            max_cycles_per_run=1,
            max_consecutive_failures=2,
            fail_on_async_context=False,
        ),
        include_reflex_loop=False,
        include_watchdog=False,
    )
    return runtime, db_path


@pytest.mark.asyncio
async def test_strategy_with_target_changes_travel_destination(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 1: A strategy carrying a target changes the travel destination."""
    runtime, _ = await _create_test_runtime(tmp_path, test_config, farm_profile)

    try:
        # Insert target node in the world model
        node_id = await runtime.world.add_node(
            150.0, 250.0, kind="waypoint", meta={"entity_id": "wp_special"}
        )

        # Mock orchestrator to return travel_to strategy targeting the node
        nav_target_called: list[tuple[float, float]] = []

        def mock_go_to(target_xy: tuple[float, float]) -> NavResult:
            nav_target_called.append(target_xy)
            return NavResult(
                status=NavStatus.SUCCESS,
                target_xy=target_xy,
                final_xy=target_xy,
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.1,
                reason="",
            )

        runtime.navigator.go_to = mock_go_to  # type: ignore[method-assign]

        # 1. Test using node_id as target
        mock_strategy = JsonStrategy(
            goal="travel_to",
            target=str(node_id),
            rationale="head to waypoint",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="h1",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        res = await _run_cycle(runtime, "farm", None, 0, None)
        assert len(nav_target_called) == 1
        assert nav_target_called[0] == (150.0, 250.0)
        assert nav_target_called[0] != (0.0, 0.0)
        assert res[0] == "farm"  # transitioned back to farm on success

        # 2. Test using entity_id as target
        nav_target_called.clear()
        mock_strategy_entity = JsonStrategy(
            goal="travel_to",
            target="wp_special",
            rationale="head to named waypoint",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy_entity,
                prompt_hash="h2",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        await _run_cycle(runtime, "farm", None, 1, None)
        assert len(nav_target_called) == 1
        assert nav_target_called[0] == (150.0, 250.0)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_unresolvable_target_produces_explicit_logged_fallback(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 2: An unresolvable target produces an explicit, logged fallback (not a silent origin trip)."""
    runtime, _ = await _create_test_runtime(tmp_path, test_config, farm_profile)

    try:
        nav_target_called: list[tuple[float, float]] = []

        def mock_go_to(target_xy: tuple[float, float]) -> NavResult:
            nav_target_called.append(target_xy)
            return NavResult(
                status=NavStatus.SUCCESS,
                target_xy=target_xy,
                final_xy=target_xy,
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.1,
                reason="",
            )

        runtime.navigator.go_to = mock_go_to  # type: ignore[method-assign]

        unresolvable_target = "nonexistent_destination_9999"
        mock_strategy = JsonStrategy(
            goal="travel_to",
            target=unresolvable_target,
            rationale="lost target",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="h3",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        await _run_cycle(runtime, "farm", None, 0, None)

        # Navigated to fallback destination (0.0, 0.0)
        assert len(nav_target_called) == 1
        assert nav_target_called[0] == (0.0, 0.0)

        # Explicit fallback event must be recorded in session events
        events = read_session_events(runtime.session)
        fallback_events = [e for e in events if e.get("event") == "target_fallback"]
        assert len(fallback_events) >= 1
        evt = fallback_events[-1]
        assert evt["target"] == unresolvable_target
        assert evt["fallback_xy"] == [0.0, 0.0]
        assert "unresolvable_target" in evt["reason"]
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_named_vendor_entity_selected_over_merely_nearer_one(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 3: A named vendor entity is selected over a merely nearer one."""
    game_state = FakeGameState(player_x=5.0, player_y=5.0)
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, game_state=game_state
    )

    try:
        # Add Vendor 1 (near: (10, 10), dist ~7.0)
        near_id = await runtime.world.add_node(
            10.0,
            10.0,
            kind="vendor",
            meta={"name": "Near Bob", "entity_id": "vendor_near"},
        )
        # Add Vendor 2 (far: (50, 50), dist ~63.6)
        far_id = await runtime.world.add_node(
            50.0,
            50.0,
            kind="vendor",
            meta={"name": "Far Alice", "entity_id": "vendor_far"},
        )

        # Mock vendor controller to record which VendorLocation was received
        selected_vendors: list[VendorLocation] = []

        def mock_vendor_run(location: VendorLocation, *args: Any, **kwargs: Any) -> Any:
            selected_vendors.append(location)
            res = MagicMock()
            res.status = VendorStatus.SUCCESS
            return res

        runtime.vendor.run = mock_vendor_run  # type: ignore[method-assign]

        # 1. When target names "vendor_far", select Vendor 2 (far) over Vendor 1 (near)
        mock_strategy = JsonStrategy(
            goal="sell_vendor",
            target="vendor_far",
            rationale="prefer Far Alice",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="h4",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        await _run_cycle(runtime, "farm", None, 0, None)
        assert len(selected_vendors) == 1
        assert selected_vendors[0].node_id == far_id
        assert selected_vendors[0].name == "Far Alice"
        assert selected_vendors[0].node_id != near_id

        # 2. When target is None, standard resolution selects the nearer vendor (Vendor 1)
        selected_vendors.clear()
        mock_strategy_none = JsonStrategy(
            goal="sell_vendor",
            target=None,
            rationale="no target, take nearest",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy_none,
                prompt_hash="h5",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        await _run_cycle(runtime, "farm", None, 1, None)
        assert len(selected_vendors) == 1
        assert selected_vendors[0].node_id == near_id
        assert selected_vendors[0].name == "Near Bob"

        # 3. When an unresolvable vendor target is given, an explicit logged fallback occurs
        selected_vendors.clear()
        mock_strategy_unres = JsonStrategy(
            goal="sell_vendor",
            target="vendor_missing_ghost",
            rationale="ghost vendor",
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy_unres,
                prompt_hash="h6",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        await _run_cycle(runtime, "farm", None, 2, None)
        assert len(selected_vendors) == 1
        # Fallback selects nearest vendor
        assert selected_vendors[0].node_id == near_id
        # Session logs vendor_target_fallback event
        events = read_session_events(runtime.session)
        v_fallback = [e for e in events if e.get("event") == "vendor_target_fallback"]
        assert len(v_fallback) >= 1
        assert v_fallback[-1]["target"] == "vendor_missing_ghost"
        assert "unresolvable_vendor_target" in v_fallback[-1]["reason"]
    finally:
        await runtime.close()


def test_navigator_go_to_node_delegation() -> None:
    """Test Navigator.go_to_node looks up coordinates in graph and delegates to go_to."""
    from wow_bot.actuation.mapper import ActionStatus
    from wow_bot.nav.graph import NavGraph

    # Build a simple NavGraph
    graph = NavGraph(
        nodes={
            1: MagicMock(id=1, x=10.0, y=20.0, z=0.0),
            2: MagicMock(id=2, x=30.0, y=40.0, z=0.0),
        },
        adjacency={1: (), 2: ()},
    )
    actuator = MagicMock()
    actuator.is_aborted.return_value = False
    actuator.execute.return_value = MagicMock(status=ActionStatus.SUCCESS)

    nav = Navigator(
        graph=graph,
        actuator=actuator,
        position_source=lambda: (30.0, 40.0),
        config=NavConfig(arrival_tolerance_units=1.0),
    )

    res = nav.go_to_node(2)
    assert res.status == NavStatus.SUCCESS
    assert res.target_xy == (30.0, 40.0)
