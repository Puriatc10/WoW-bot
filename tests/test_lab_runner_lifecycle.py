"""Acceptance tests for T-FIX-06: Reflex/Watchdog lifecycle in lab runner."""

from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass
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
    LabRunStatus,
    _CancellableSleep,
    _MockReflexClock,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.reflex.loop import ReflexLoop
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session
from wow_bot.watchdog.health import HealthState
from wow_bot.watchdog.watchdog import (
    HeartbeatMessage,
    WatchdogMonitor,
    WatchdogProcess,
)


@dataclass
class FakeGameState:
    """Fake GameState satisfying runner protocols."""

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

    def complete(self, prompt: str) -> str:
        return '{"goal": "farm", "target": null, "rationale": "continue farming"}'


class FakeWatchdogClock:
    """Deterministic monotonic clock for watchdog tests."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.time = initial_time

    def monotonic(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


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
        dry_run=False,
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


@pytest.mark.asyncio
async def test_reflex_ticks_produced_and_stops_cleanly(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Criterion 1: With reflex enabled, run_lab_loop_async produces > 0 ticks and stops cleanly."""
    state = FakeGameState()

    def state_source() -> FakeGameState:
        state.player_x += 0.5
        state.player_y += 0.5
        state.self_x += 0.5
        state.self_y += 0.5
        return state

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=state_source,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        include_reflex_loop=True,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    assert runtime.reflex_loop is not None
    assert not runtime.reflex_loop.is_running()
    assert runtime.reflex_loop.tick_index == 0

    res = await run_lab_loop_async(runtime, max_cycles=3)

    assert res.status == LabRunStatus.MAX_CYCLES_REACHED
    assert res.cycles_completed == 3
    # Reflex loop ran in background and ticked
    assert runtime.reflex_loop.tick_index > 0
    # Reflex loop stopped cleanly on exit
    assert not runtime.reflex_loop.is_running()

    await runtime.close()


@pytest.mark.asyncio
async def test_no_reflex_or_watchdog_thread_alive_after_stop(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Criterion 2: After stop, no reflex or watchdog thread/process is alive."""
    state = FakeGameState()

    def state_source() -> FakeGameState:
        state.player_x += 0.5
        state.player_y += 0.5
        state.self_x += 0.5
        state.self_y += 0.5
        return state

    # Create WatchdogProcess with short poll interval for fast test shutdown
    watchdog = WatchdogProcess(poll_interval=0.05)

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        game_state_source=state_source,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        include_reflex_loop=True,
        watchdog=watchdog,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    res = await run_lab_loop_async(runtime, max_cycles=2)
    assert res.status == LabRunStatus.MAX_CYCLES_REACHED

    # Verify reflex thread stopped
    assert runtime.reflex_loop is not None
    assert not runtime.reflex_loop.is_running()

    # Verify watchdog process and monitor thread stopped
    assert runtime.watchdog is not None
    assert not runtime.watchdog.is_alive
    if runtime.watchdog._monitor_thread is not None:
        assert not runtime.watchdog._monitor_thread.is_alive()

    # Check global active threads
    alive_thread_names = [t.name for t in threading.enumerate() if t.is_alive()]
    assert "watchdog_shutdown_monitor" not in alive_thread_names

    await runtime.close()


def test_stop_issued_during_blocked_sleep_returns_bounded_time(tmp_path: Path) -> None:
    """Criterion 3: A stop issued during a blocked sleep returns within bounded time."""
    # Test ReflexLoop stop timeout bound
    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=lambda _sigs, _idx, _rng: [],
    )
    loop.start()
    assert loop.is_running()

    t0 = time.monotonic()
    loop.stop(timeout_s=0.5)
    t1 = time.monotonic()

    assert (t1 - t0) < 1.0
    assert not loop.is_running()

    # Test _CancellableSleep returns promptly when stop_event is set
    cfg = Config(
        lab_mode=True,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )
    safety = SafetyLayer(cfg)
    stop_event = threading.Event()
    sleep_fn = _CancellableSleep(safety=safety, stop_event=stop_event)

    stop_event.set()
    t_start = time.monotonic()
    sleep_fn(10.0)  # Would block 10 seconds without cancellation
    t_end = time.monotonic()

    assert (t_end - t_start) < 0.2


@pytest.mark.asyncio
async def test_both_components_start_in_mock_mode_with_fake_clock() -> None:
    """Criterion 4: Both reflex loop and watchdog start in MOCK_MODE with fake clocks."""
    # 1. Reflex loop with _MockReflexClock
    mock_clk = _MockReflexClock(initial_time=0.0)
    reflex = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=lambda _sigs, _idx, _rng: [],
        clock=mock_clk,
    )
    assert not reflex.is_running()
    reflex.start()
    assert reflex.is_running()
    await asyncio.sleep(0.05)
    reflex.stop(timeout_s=0.5)
    assert not reflex.is_running()
    assert reflex.tick_index > 0

    # 2. Watchdog with FakeWatchdogClock
    fake_watchdog_clock = FakeWatchdogClock(initial_time=100.0)
    monitor = WatchdogMonitor(clock=fake_watchdog_clock)
    hb = HeartbeatMessage(
        monotonic_sent_at=100.0,
        simulation_timestamp=100.0,
        fsm_state="IDLE",
        progress_token=1,
    )
    monitor.process_message(hb)
    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY

    # 3. WatchdogProcess starts and stops cleanly with custom queues
    wp = WatchdogProcess(poll_interval=0.05)
    wp.start()
    assert wp.is_alive
    wp.message_queue.put_nowait(hb)
    wp.stop()
    wp.join(timeout=1.0)
    assert not wp.is_alive
    wp.close()
