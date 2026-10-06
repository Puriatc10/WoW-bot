"""Tests for T-FIX-25: Strategist plan refresh and lifetime.

Verifies:
1. An expired strategy (now >= valid_until) triggers a new planning request.
2. A valid strategy (now < valid_until) is not re-requested before valid_until.
3. The 'farm' sentinel is not required for refresh (e.g., active goal is 'explore').
4. The strategist cooldown still prevents a request storm when strategy is expired.
5. Planning requests are logged to session with the prompt hash (AGENTS.md §6.3).
"""

from __future__ import annotations

import json
from collections.abc import Callable
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
from wow_bot.lab.runner_v2 import (
    LabRunnerConfig,
    LabRuntime,
    _run_cycle,
    build_lab_runtime_async,
)
from wow_bot.session import Session
from wow_bot.shared.interfaces import Strategy
from wow_bot.strategist.cooldown_v2 import CooldownDecision
from wow_bot.strategist.orchestrator_v2 import (
    OrchestratorConfig,
    OrchestratorOutcome,
    OrchestratorResult,
)
from wow_bot.strategist.vocab_v2 import ValidatedStrategy
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
    inventory_count: int = 0
    inventory_max: int = 16
    durability_fraction: float | None = 1.0
    level: float = 10.0
    xp: float = 1000.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        return True


@dataclass
class FakeMetaState:
    """Fake MetaState satisfying MetaStateLike for lab tests."""

    drives: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0, 0.0)
    current_behavior_name: str = "idle"
    mood: str = "calm"
    urgency: float = 0.0
    dominant_drive: str = "curiosity"
    energy: float = 1.0
    stress: float = 0.0
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


async def _create_test_runtime(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
    game_state: FakeGameState | None = None,
    clock: Callable[[], float] | None = None,
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
            max_cycles_per_run=1,
            max_consecutive_failures=2,
            fail_on_async_context=False,
            world_sync_interval_cycles=100,
            summarize_every_cycles=100,
        ),
        clock=clock,
        include_reflex_loop=False,
        include_watchdog=False,
    )
    return runtime, db_path


@pytest.mark.asyncio
async def test_expired_strategy_triggers_new_planning_request(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 1: An expired strategy (now >= valid_until) triggers a new planning request."""
    sim_time = [100.0]
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, clock=lambda: sim_time[0]
    )

    try:
        mock_strategy = ValidatedStrategy(
            goal="grind_humans",
            target="m1",
            rationale="expired plan replaced",
            valid_until=200.0,
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="ph_expired",
                attempts=1,
                latency_ms=12.0,
                reason="",
            )
        )

        # Active strategy expired: valid_until was 50.0, but clock is 100.0
        expired_strat = ValidatedStrategy(
            goal="grind_humans",
            target="m1",
            rationale="old plan",
            valid_until=50.0,
        )

        res = await _run_cycle(
            runtime=runtime,
            prev_goal="grind_humans",
            cached_vendor=None,
            cycle_index=0,
            last_summary=None,
            current_target="m1",
            current_strategy=expired_strat,
        )

        assert runtime.orchestrator.decide.call_count == 1
        # The new active strategy returned has valid_until=200.0
        assert res[7] == mock_strategy
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_valid_strategy_not_rerequested_before_valid_until(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 2: A valid strategy (now < valid_until) is NOT re-requested before valid_until."""
    sim_time = [100.0]
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, clock=lambda: sim_time[0]
    )

    try:
        runtime.orchestrator.decide = MagicMock()  # type: ignore[method-assign]

        valid_strat = ValidatedStrategy(
            goal="grind_humans",
            target="m1",
            rationale="active plan",
            valid_until=500.0,
        )

        res = await _run_cycle(
            runtime=runtime,
            prev_goal="grind_humans",
            cached_vendor=None,
            cycle_index=0,
            last_summary=None,
            current_target="m1",
            current_strategy=valid_strat,
        )

        # Decide must NOT have been called because strategy is still valid
        assert runtime.orchestrator.decide.call_count == 0
        assert res[7] == valid_strat
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_farm_sentinel_not_required_for_refresh(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 3: The 'farm' sentinel is not required for refresh.

    A non-'farm' goal (e.g. 'explore' or Strategy dataclass) triggers refresh when expired.
    """
    sim_time = [100.0]
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, clock=lambda: sim_time[0]
    )

    try:
        mock_strategy = ValidatedStrategy(
            goal="farm_herbs",
            target="peacebloom",
            rationale="switched from explore",
            valid_until=300.0,
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="ph_switch",
                attempts=1,
                latency_ms=15.0,
                reason="",
            )
        )

        # Previous goal is 'explore', strategy is an expired Strategy dataclass
        expired_strat = Strategy(
            goal="explore",
            region="elwynn",
            risk_tolerance=0.5,
            priority=["explore"],
            constraints={"target": None},
            valid_until=40.0,
        )

        res = await _run_cycle(
            runtime=runtime,
            prev_goal="explore",  # Not "farm"!
            cached_vendor=None,
            cycle_index=0,
            last_summary=None,
            current_target=None,
            current_strategy=expired_strat,
        )

        # Decide MUST be called despite prev_goal != "farm"
        assert runtime.orchestrator.decide.call_count == 1
        assert res[0] == "farm_herbs"
        assert res[7] == mock_strategy
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_cooldown_prevents_request_storm(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Acceptance 4: Cooldown prevents request storm when an expired strategy triggers refresh."""
    sim_time = [100.0]
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, clock=lambda: sim_time[0]
    )

    try:
        # Mock orchestrator to simulate BLOCKED_COOLDOWN
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.BLOCKED_BY_COOLDOWN,
                strategy=None,
                prompt_hash="",
                attempts=0,
                latency_ms=0.0,
                reason=CooldownDecision.BLOCKED_COOLDOWN.value,
            )
        )

        expired_strat = ValidatedStrategy(
            goal="explore",
            target=None,
            rationale="old",
            valid_until=50.0,
        )

        # First cycle attempts refresh, gets blocked by cooldown
        res1 = await _run_cycle(
            runtime=runtime,
            prev_goal="explore",
            cached_vendor=None,
            cycle_index=0,
            last_summary=None,
            current_target=None,
            current_strategy=expired_strat,
        )
        # Goal remains 'explore' without crashing or erroring
        assert res1[0] == "explore"
        assert runtime.orchestrator.decide.call_count == 1

        # Second cycle also blocked by cooldown
        res2 = await _run_cycle(
            runtime=runtime,
            prev_goal="explore",
            cached_vendor=None,
            cycle_index=1,
            last_summary=None,
            current_target=None,
            current_strategy=expired_strat,
        )
        assert res2[0] == "explore"
        assert runtime.orchestrator.decide.call_count == 2
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_planning_request_logged_with_prompt_hash(
    tmp_path: Path,
    test_config: Config,
    farm_profile: FarmProfile,
) -> None:
    """Contract verification: Planning requests are logged to session with prompt_hash."""
    sim_time = [100.0]
    runtime, _ = await _create_test_runtime(
        tmp_path, test_config, farm_profile, clock=lambda: sim_time[0]
    )

    try:
        mock_strategy = ValidatedStrategy(
            goal="explore",
            target=None,
            rationale="plan",
            valid_until=150.0,
        )
        runtime.orchestrator.decide = MagicMock(  # type: ignore[method-assign]
            return_value=OrchestratorResult(
                outcome=OrchestratorOutcome.SUCCESS,
                strategy=mock_strategy,
                prompt_hash="hash_alpha_999",
                attempts=1,
                latency_ms=10.0,
                reason="",
            )
        )

        # Absent strategy triggers planning
        await _run_cycle(
            runtime=runtime,
            prev_goal="farm",
            cached_vendor=None,
            cycle_index=0,
            last_summary=None,
            current_target=None,
            current_strategy=None,
        )

        assert runtime.session is not None
        events = read_session_events(runtime.session)
        planning_events = [e for e in events if e.get("event") == "planning_request"]
        assert len(planning_events) == 1
        assert planning_events[0]["prompt_hash"] == "hash_alpha_999"
    finally:
        await runtime.close()


def test_validated_strategy_and_config_ttl() -> None:
    """Test ValidatedStrategy and OrchestratorConfig valid_until and TTL validation."""
    strat = ValidatedStrategy(
        goal="explore",
        target=None,
        rationale="test",
        valid_until=123.45,
    )
    assert strat.valid_until == 123.45

    with pytest.raises(ValueError, match="valid_until must be a float"):
        ValidatedStrategy(
            goal="explore",
            target=None,
            rationale="test",
            valid_until="invalid",  # type: ignore[arg-type]
        )

    cfg = OrchestratorConfig(strategy_ttl_seconds=600.0)
    assert cfg.strategy_ttl_seconds == 600.0

    with pytest.raises(ValueError, match="strategy_ttl_seconds"):
        OrchestratorConfig(strategy_ttl_seconds=-10.0)
