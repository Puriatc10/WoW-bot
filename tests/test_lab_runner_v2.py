"""Tests for LAB mode runner_v2 module (Task 11.4)."""

from __future__ import annotations

import ast
import asyncio
import json
import sys
from dataclasses import dataclass, is_dataclass
from pathlib import Path
from typing import Any

import pytest
from wow_bot.actuation.backends.focus_null import NullFocusBackend
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
    LabRunnerError,
    LabRunResult,
    LabRunStatus,
    LabRuntime,
    _CompositeBehavior,
    _NullLlmClient,
    build_lab_runtime,
    build_lab_runtime_async,
    run_lab_loop,
    run_lab_loop_async,
)
from wow_bot.session import Session


@dataclass
class FakeGameState:
    """Fake GameState satisfying GameStateView, LootStateView, VendorStateView, and CombatStateView."""

    player_x: float = 10.0
    player_y: float = 20.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 10.0
    self_y: float = 20.0
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
    """Fake MetaState satisfying MetaStateLike."""

    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


class FakeLlmClient:
    """Scripted fake LLM client for tests."""

    def __init__(self, responses: list[str] | None = None) -> None:
        self.responses = responses or [
            '{"goal": "farm", "target": null, "rationale": "continue farming"}'
        ]
        self.call_count = 0

    def complete(self, prompt: str) -> str:
        res = self.responses[self.call_count % len(self.responses)]
        self.call_count += 1
        return res


class FakeClock:
    """Fake clock advancing time deterministically."""

    def __init__(self, start: float = 0.0, step: float = 0.01) -> None:
        self.current = start
        self.step = step

    def __call__(self) -> float:
        val = self.current
        self.current += self.step
        return val


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Provide a valid Config instance in LAB mode."""
    sess_root = tmp_path / "runs"
    sess_root.mkdir(parents=True, exist_ok=True)
    return Config(
        lab_mode=True,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=sess_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


@pytest.fixture
def temp_session(mock_config: Config) -> Session:
    """Provide a temporary Session instance."""
    return Session.start(mock_config)


@pytest.fixture
def mock_profile() -> FarmProfile:
    """Provide a valid FarmProfile instance."""
    return FarmProfile(
        schema_version=1,
        name="test_profile",
        description="A test profile",
        cycle=CycleSpec(
            nodes=(NodeReference(kind="mob", name="mob_1"),),
            vendor=VendorReference(kind="vendor", name="vendor_1"),
            repair=VendorReference(kind="vendor", name="vendor_1"),
            stop_when_inventory_full=True,
            stop_after_cycles=0,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )


# Parameter validation tests
def test_runner_config_validation() -> None:
    """Verify bounds validation in LabRunnerConfig."""
    with pytest.raises(ValueError, match="max_cycles_per_run"):
        LabRunnerConfig(max_cycles_per_run=0)

    with pytest.raises(ValueError, match="max_consecutive_failures"):
        LabRunnerConfig(max_consecutive_failures=0)

    with pytest.raises(ValueError, match="world_sync_interval_cycles"):
        LabRunnerConfig(world_sync_interval_cycles=0)

    with pytest.raises(ValueError, match="summarize_every_cycles"):
        LabRunnerConfig(summarize_every_cycles=0)

    with pytest.raises(ValueError, match="vendor_repair_threshold"):
        LabRunnerConfig(vendor_repair_threshold=-0.1)

    with pytest.raises(ValueError, match="vendor_repair_threshold"):
        LabRunnerConfig(vendor_repair_threshold=1.5)


def test_lab_run_result_frozen_and_json() -> None:
    """Verify LabRunResult is frozen and serializes to JSON."""
    res = LabRunResult(
        status=LabRunStatus.COMPLETED,
        cycles_completed=10,
        failures=0,
        duration_s=1.23,
        shutdown_report=None,
        reason="completed",
    )
    with pytest.raises(AttributeError):
        res.cycles_completed = 20  # type: ignore[misc]

    data = res.to_json()
    assert json.dumps(data)
    assert data["status"] == "completed"
    assert data["cycles_completed"] == 10


def test_lab_runtime_is_frozen(temp_session: Session) -> None:
    """Verify LabRuntime is a dataclass."""
    assert is_dataclass(LabRuntime)


# Builder tests
@pytest.mark.asyncio
async def test_build_lab_runtime_async_basic(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify build_lab_runtime_async constructs a valid LabRuntime."""
    db_path = str(tmp_path / "world.db")
    state = FakeGameState()
    meta = FakeMetaState()
    llm = FakeLlmClient()

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=lambda: state,
        meta_state_source=lambda: meta,
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=db_path,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=llm,
    )

    assert isinstance(runtime, LabRuntime)
    assert runtime.mode == "LAB"
    assert runtime.world is not None
    assert runtime.graph is not None
    assert runtime.reflex_loop is not None
    await runtime.world.close()


@pytest.mark.asyncio
async def test_build_lab_runtime_async_lab_mode_false_raises(
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify build_lab_runtime_async raises when lab_mode is False."""
    cfg = Config(
        lab_mode=False,
        server_allowlist=[],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=Path("runs/lab"),
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    with pytest.raises(LabRunnerError, match="lab_mode"):
        await build_lab_runtime_async(
            config=cfg,
            session=temp_session,
            game_state_source=lambda: FakeGameState(),
            meta_state_source=lambda: FakeMetaState(),
            rotation_config=RotationConfig(rules=()),
            farm_profile=mock_profile,
            world_db_path=str(tmp_path / "world.db"),
            include_watchdog=False,
            focus_backend=NullFocusBackend(),
        )


@pytest.mark.asyncio
async def test_build_lab_runtime_async_loop_threads_not_started(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify builder does not start the Reflex loop."""
    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=lambda: FakeGameState(),
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
    )

    assert runtime.reflex_loop is not None
    assert not runtime.reflex_loop.is_running()
    await runtime.world.close()


@pytest.mark.asyncio
async def test_build_lab_runtime_async_flags_exclude_components(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify include_reflex_loop and include_watchdog flags set attributes to None."""
    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=lambda: FakeGameState(),
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        include_reflex_loop=False,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
    )

    assert runtime.reflex_loop is None
    assert runtime.watchdog is None
    await runtime.world.close()


@pytest.mark.asyncio
async def test_build_lab_runtime_in_running_event_loop_raises(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify sync build_lab_runtime raises LabRunnerError when called inside event loop."""
    with pytest.raises(LabRunnerError, match="running event loop"):
        build_lab_runtime(
            config=mock_config,
            session=temp_session,
            game_state_source=lambda: FakeGameState(),
            meta_state_source=lambda: FakeMetaState(),
            rotation_config=RotationConfig(rules=()),
            farm_profile=mock_profile,
            world_db_path=str(tmp_path / "world.db"),
            include_watchdog=False,
            focus_backend=NullFocusBackend(),
        )


# Run loop tests
@pytest.mark.asyncio
async def test_run_lab_loop_async_max_cycles(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify run_lab_loop_async runs exactly max_cycles and returns MAX_CYCLES_REACHED."""
    clock = FakeClock(step=0.01)
    llm = FakeLlmClient()
    state = FakeGameState(inventory_count=0, inventory_max=100)

    def moving_state_source() -> FakeGameState:
        state.player_x += 2.0
        state.player_y += 2.0
        state.self_x += 2.0
        state.self_y += 2.0
        state.xp += 10.0
        state.inventory_count += 1
        return state

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=moving_state_source,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=clock,
        sleep=lambda _: None,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=llm,
    )

    res = await run_lab_loop_async(runtime, max_cycles=10)
    assert res.status == LabRunStatus.MAX_CYCLES_REACHED
    assert res.cycles_completed == 10
    assert res.failures == 0
    await runtime.world.close()


@pytest.mark.asyncio
async def test_run_lab_loop_async_stop_event(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify run_lab_loop_async stops when stop_event is set."""
    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=lambda: FakeGameState(),
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        sleep=lambda _: None,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    stop_evt = asyncio.Event()
    stop_evt.set()

    res = await run_lab_loop_async(runtime, max_cycles=100, stop_event=stop_evt)
    assert res.status == LabRunStatus.STOP_EVENT_SET
    assert res.cycles_completed == 0
    await runtime.world.close()


@pytest.mark.asyncio
async def test_run_lab_loop_sync_in_running_event_loop_raises(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify sync run_lab_loop raises when called inside running event loop."""
    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=lambda: FakeGameState(),
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    with pytest.raises(LabRunnerError, match="running event loop"):
        run_lab_loop(runtime, max_cycles=5)
    await runtime.world.close()


def test_time_and_sleep_purity(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify runtime uses injected clock/sleep."""
    clock = FakeClock(step=0.01)
    sleep_calls = 0

    def custom_sleep(s: float) -> None:
        nonlocal sleep_calls
        sleep_calls += 1

    state = FakeGameState(inventory_count=0, inventory_max=100)

    def moving_state_source() -> FakeGameState:
        state.player_x += 2.0
        state.player_y += 2.0
        state.self_x += 2.0
        state.self_y += 2.0
        state.xp += 10.0
        state.inventory_count += 1
        return state

    runtime = build_lab_runtime(
        config=mock_config,
        session=temp_session,
        game_state_source=moving_state_source,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=clock,
        sleep=custom_sleep,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    res = run_lab_loop(runtime, max_cycles=5)
    assert res.status == LabRunStatus.MAX_CYCLES_REACHED
    assert res.cycles_completed == 5
    assert sleep_calls == 5


# Composite behavior tests
def test_composite_behavior_dispatch() -> None:
    """Verify _CompositeBehavior dispatch order across FSM states."""
    import random

    from wow_bot.combat.loop import CombatLoop, CombatLoopConfig
    from wow_bot.combat.rotation import RotationConfig, RotationTable
    from wow_bot.combat.targeting import TargetConfig, TargetSelector
    from wow_bot.executor.recovery import RecoveryBehavior, RecoveryConfig
    from wow_bot.farm.loot import LootConfig, LootController

    rec = RecoveryBehavior(config=RecoveryConfig())
    combat = CombatLoop(
        config=CombatLoopConfig(),
        rotation=RotationTable(RotationConfig(rules=())),
        targeting=TargetSelector(config=TargetConfig()),
    )
    loot = LootController(config=LootConfig())

    comp = _CompositeBehavior(
        recovery_behavior=rec,
        combat_loop=combat,
        loot_controller=loot,
    )

    state = FakeGameState()
    meta = FakeMetaState()
    rng = random.Random(0)

    # STUCK_RECOVERY
    res_rec = comp.decide(FSMState.STUCK_RECOVERY, state, meta, 1.0, rng)
    assert res_rec is not None

    # COMBAT
    state.fsm_state = FSMState.COMBAT
    res_comb = comp.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert res_comb is None  # empty rotation returns None

    # LOOTING
    state.fsm_state = FSMState.LOOTING
    res_loot = comp.decide(FSMState.LOOTING, state, meta, 1.0, rng)
    assert res_loot is None  # no target returns None

    # IDLE
    res_idle = comp.decide(FSMState.IDLE, state, meta, 1.0, rng)
    assert res_idle is None


def test_null_llm_client_raises() -> None:
    """Verify _NullLlmClient raises clear exception on complete()."""
    client = _NullLlmClient()
    with pytest.raises(LabRunnerError, match="No LlmClient provided"):
        client.complete("prompt")


# Static AST tests
def test_static_ast_checks() -> None:
    """Verify static AST constraints on runner_v2.py."""
    filepath = Path("src/wow_bot/lab/runner_v2.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    forbidden_exact = {
        "wow_bot.main",
        "wow_bot.executor.fsm",
        "wow_bot.executor.controller",
        "wow_bot.strategist.orchestrator",
        "wow_bot.strategist.llm_client",
        "wow_bot.reporting.scenario",
        "wow_bot.analysis",
        "wow_bot.internal_dynamics",
        "openai",
        "anthropic",
        "gemini",
        "cohere",
        "bedrock",
    }

    forbidden_calls = {"exit", "_exit", "kill"}

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_exact:
                    assert alias.name != forbidden and not alias.name.startswith(forbidden + "."), f"Forbidden import: {alias.name}"
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                for forbidden in forbidden_exact:
                    assert node.module != forbidden and not node.module.startswith(forbidden + "."), f"Forbidden import from: {node.module}"

        elif isinstance(node, ast.Call):
            if isinstance(node.func, ast.Attribute):
                assert node.func.attr not in forbidden_calls, f"Forbidden call attribute: {node.func.attr}"
                assert node.func.attr not in ("start", "stop"), f"Runner calls start/stop on thread: {node.func.attr}"
            elif isinstance(node.func, ast.Name):
                assert node.func.id not in forbidden_calls, f"Forbidden call name: {node.func.id}"


# CLI script tests
def test_cli_script_help() -> None:
    """Verify CLI script --help exits with code 0."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "scripts/lab/run_farm_v2.py", "--help"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    assert "usage:" in res.stdout


def test_cli_script_mock_run() -> None:
    """Verify CLI script runs in MOCK mode and outputs valid LabRunResult JSON."""
    import subprocess
    res = subprocess.run(
        [sys.executable, "scripts/lab/run_farm_v2.py", "--max-cycles", "5", "--mode", "MOCK"],
        capture_output=True,
        text=True,
    )
    assert res.returncode == 0
    data = json.loads(res.stdout)
    assert data["status"] in ("completed", "max_cycles_reached")
    assert data["cycles_completed"] == 5
