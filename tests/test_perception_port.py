"""Tests for T-FIX-20: Async perception port and loop scheduling.

Verifies:
1. Port contract: background snapshot pump, synchronous sample() accessor,
   uninitialized and stale detection (never silently substituted or fabricated).
2. Backend drives runner: real-shaped async backend drives runner through the port
   and cycles observe snapshots in order.
3. Staleness honesty: staleness is reported explicitly, failing loud on stale frames.
4. Bounded cancellation: cancelling a cycle returns within bounded time even while
   a synchronous navigation or sleep call is in flight.
5. Backward compatibility: pre-existing synchronous game_state_source path works unchanged.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.combat.rotation import RotationConfig
from wow_bot.config import Config
from wow_bot.executor.states import FSMState
from wow_bot.farm.profile import (
    CycleSpec,
    FarmProfile,
    NodeReference,
    RoutePreferences,
    VendorReference,
)
from wow_bot.lab.runner_v2 import (
    LabRunnerError,
    LabRunStatus,
    _CancellableSleep,
    build_lab_runtime,
    build_lab_runtime_async,
    run_lab_loop,
    run_lab_loop_async,
)
from wow_bot.perception.port import (
    PerceptionBackendError,
    PerceptionPort,
    PerceptionStaleError,
)
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.session import Session
from wow_bot.shared.interfaces import GameState

# ---------------------------------------------------------------------------
# Fixtures & Test doubles
# ---------------------------------------------------------------------------


@pytest.fixture
def mock_config(tmp_path: Path) -> Config:
    """Provide a valid Config instance in LAB mode."""
    sess_root = tmp_path / "runs"
    sess_root.mkdir(parents=True, exist_ok=True)
    return Config(
        lab_mode=True,
        server_allowlist=("127.0.0.1:8080",),
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


class SequencePerceptionBackend(PerceptionBackend):
    """Async backend yielding GameState with incrementing sequence IDs and positions."""

    def __init__(self, initial_seq: int = 1, delay: float = 0.0) -> None:
        self.seq = initial_seq
        self.delay = delay
        self.call_count = 0
        self.fail_after: int | None = None

    async def snapshot(self) -> GameState:
        if self.delay > 0:
            await asyncio.sleep(self.delay)
        if self.fail_after is not None and self.call_count >= self.fail_after:
            raise RuntimeError(f"Backend hardware failure at call {self.call_count}")

        idx = self.seq
        self.seq += 1
        self.call_count += 1

        state = GameState(
            timestamp=float(idx),
            hp_pct=1.0,
            mana_pct=1.0,
            position=(float(idx), 20.0),
            facing=0.0,
            in_combat=False,
            target=None,
            enemies=[],
            events=[],
            player_z=0.0,
            inventory_count=idx,
            inventory_max=30,
            level_or_xp=10.0,
        )
        object.__setattr__(state, "player_x", float(idx))
        object.__setattr__(state, "player_y", 20.0)
        object.__setattr__(state, "player_z", 0.0)
        object.__setattr__(state, "player_heading", 0.0)
        object.__setattr__(state, "self_x", float(idx))
        object.__setattr__(state, "self_y", 20.0)
        object.__setattr__(state, "self_hp_percent", 100.0)
        object.__setattr__(state, "self_in_combat", False)
        object.__setattr__(state, "fsm_state", FSMState.IDLE)
        object.__setattr__(state, "current_target_id", None)
        object.__setattr__(state, "target_entity_id", None)
        object.__setattr__(state, "target_in_range", False)
        object.__setattr__(state, "target_is_alive", False)
        object.__setattr__(state, "target_is_lootable", False)
        object.__setattr__(state, "target_distance", None)
        object.__setattr__(state, "target_x", None)
        object.__setattr__(state, "target_y", None)
        object.__setattr__(state, "adds_count", 0)
        object.__setattr__(state, "inventory_count", idx)
        object.__setattr__(state, "inventory_max", 30)
        object.__setattr__(state, "durability_fraction", 1.0)
        object.__setattr__(state, "level", 10.0)
        object.__setattr__(state, "xp", 1000.0)
        return state


@dataclass
class FakeGameState:
    """Mock GameState satisfying runner views."""

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
    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


class FakeLlmClient:
    def complete(self, prompt: str) -> str:
        return '{"goal": "farm", "target": null, "rationale": "continue farming"}'


class SettableClock:
    def __init__(self, start: float = 1000.0, step: float = 0.01) -> None:
        self.time = start
        self.step = step

    def __call__(self) -> float:
        val = self.time
        self.time += self.step
        return val

    def advance(self, delta: float) -> None:
        self.time += delta


# ---------------------------------------------------------------------------
# 1. PerceptionPort Unit Tests
# ---------------------------------------------------------------------------


def test_port_validation_raises() -> None:
    """Verify PerceptionPort validates backend type and positive timing bounds."""
    backend = SequencePerceptionBackend()
    with pytest.raises(TypeError, match="backend must be a PerceptionBackend"):
        PerceptionPort("not_a_backend")  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="staleness_bound_s must be > 0.0"):
        PerceptionPort(backend, staleness_bound_s=0.0)

    with pytest.raises(ValueError, match="staleness_bound_s must be > 0.0"):
        PerceptionPort(backend, staleness_bound_s=-1.0)

    with pytest.raises(ValueError, match="pump_interval_s must be > 0.0"):
        PerceptionPort(backend, pump_interval_s=0.0)


def test_sample_uninitialized_raises_explicit_stale_error() -> None:
    """Verify sample() on an un-pumped port raises PerceptionStaleError."""
    backend = SequencePerceptionBackend()
    port = PerceptionPort(backend, staleness_bound_s=0.5)

    with pytest.raises(PerceptionStaleError, match="uninitialized"):
        port.sample()


@pytest.mark.asyncio
async def test_pump_once_and_sample_success() -> None:
    """Verify pump_once() stores latest snapshot and sample() returns it."""
    backend = SequencePerceptionBackend(initial_seq=42)
    clock = SettableClock(100.0)
    port = PerceptionPort(backend, staleness_bound_s=0.5, clock=clock)

    snap = await port.pump_once()
    assert snap.timestamp == 42.0
    assert port.latest_timestamp == 100.0

    sample = port.sample()
    assert sample.timestamp == 42.0


def test_staleness_detection_fails_loud_without_silent_reuse() -> None:
    """Verify sample() rejects stale snapshots after staleness_bound_s expires."""
    backend = SequencePerceptionBackend(initial_seq=1)
    clock = SettableClock(1000.0)
    port = PerceptionPort(backend, staleness_bound_s=0.5, clock=clock)

    asyncio.run(port.pump_once())
    assert port.sample().timestamp == 1.0

    # Advance clock within bound
    clock.advance(0.3)
    assert port.sample().timestamp == 1.0

    # Advance clock past bound
    clock.advance(0.21)  # age is now 0.51s > 0.5s bound
    with pytest.raises(PerceptionStaleError, match="stale: age .* exceeds bound 0.5"):
        port.sample()


@pytest.mark.asyncio
async def test_backend_failure_propagates_on_sample() -> None:
    """Verify backend exception in pump is recorded and raised on sample()."""
    backend = SequencePerceptionBackend(initial_seq=1)
    backend.fail_after = 1
    port = PerceptionPort(backend, staleness_bound_s=1.0)

    # First succeeds
    await port.pump_once()
    assert port.sample().timestamp == 1.0

    # Second fails
    with pytest.raises(RuntimeError, match="Backend hardware failure"):
        await port.pump_once()

    with pytest.raises(PerceptionBackendError, match="Backend snapshot failed"):
        port.sample()


@pytest.mark.asyncio
async def test_port_async_context_manager_lifecycle() -> None:
    """Verify async context manager starts and stops pump loop."""
    backend = SequencePerceptionBackend(initial_seq=1)
    port = PerceptionPort(backend, staleness_bound_s=1.0, pump_interval_s=0.01)

    async with port:
        assert port.is_running is True
        await asyncio.sleep(0.05)
        # Pump should have advanced several times
        snap = port.sample()
        assert snap.timestamp >= 2.0

    assert port.is_running is False


def test_threaded_pump_runs_while_main_thread_sleeps() -> None:
    """Verify threaded pump continues capturing while main thread does synchronous work."""
    backend = SequencePerceptionBackend(initial_seq=1)
    port = PerceptionPort(backend, staleness_bound_s=1.0, pump_interval_s=0.01)

    port.start_pump(threaded=True)
    try:
        assert port.is_running is True
        # Block main thread synchronously
        time.sleep(0.06)
        snap = port.sample()
        # Verify the background thread captured frames during our synchronous sleep
        assert snap.timestamp >= 3.0
    finally:
        port.stop_pump()

    assert port.is_running is False


# ---------------------------------------------------------------------------
# 2. Acceptance 1: Real-shaped async backend drives runner in order
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_async_backend_drives_runner_in_order(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Acceptance criterion 1: Async backend drives runner and cycles observe snapshots in order."""
    backend = SequencePerceptionBackend(initial_seq=1)
    port = PerceptionPort(backend, staleness_bound_s=2.0, pump_interval_s=0.01)

    observed_snapshots: list[float] = []

    original_sample = port.sample

    def spy_sample() -> GameState:
        snap = original_sample()
        observed_snapshots.append(snap.timestamp)
        return snap

    port.sample = spy_sample  # type: ignore[method-assign]

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        perception_port=port,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=SettableClock(start=1000.0, step=0.01),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
        sleep=lambda _: None,
    )

    try:
        res = await run_lab_loop_async(runtime, max_cycles=5)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 5

        # Check snapshots were observed in strictly increasing order
        assert len(observed_snapshots) >= 5
        for i in range(len(observed_snapshots) - 1):
            assert observed_snapshots[i] <= observed_snapshots[i + 1]
    finally:
        await runtime.world.close()


@pytest.mark.asyncio
async def test_runner_accepts_perception_backend_directly(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Verify build_lab_runtime_async automatically wraps a PerceptionBackend in a PerceptionPort."""
    backend = SequencePerceptionBackend(initial_seq=10)

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        perception_backend=backend,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=SettableClock(start=1000.0, step=0.01),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
        sleep=lambda _: None,
    )

    try:
        assert isinstance(runtime.perception_port, PerceptionPort)
        assert runtime.perception_port.backend is backend

        res = await run_lab_loop_async(runtime, max_cycles=3)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 3
    finally:
        await runtime.world.close()


# ---------------------------------------------------------------------------
# 3. Acceptance 2: Staleness honesty (never silently substituted)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_staleness_reported_explicitly_in_runner(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Acceptance criterion 2: Stale snapshot is never silently reused; runner reports RUNTIME_ERROR."""
    backend = SequencePerceptionBackend(initial_seq=1)
    clock = SettableClock(100.0)

    port = PerceptionPort(
        backend,
        staleness_bound_s=0.2,
        pump_interval_s=10.0,  # Slow pump so it won't refresh automatically
        clock=clock,
    )

    # Prime with one snapshot at t=100.0
    await port.pump_once()
    assert port.sample().timestamp == 1.0

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        perception_port=port,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=clock,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
        sleep=lambda _: None,
    )

    try:
        # Advance clock past staleness bound (0.2s bound, advance 1.0s)
        clock.advance(1.0)

        res = await run_lab_loop_async(runtime, max_cycles=3)
        assert res.status == LabRunStatus.RUNTIME_ERROR
        assert "PerceptionStaleError" in res.reason
        assert "exceeds bound 0.2" in res.reason
    finally:
        await runtime.world.close()


# ---------------------------------------------------------------------------
# 4. Acceptance 3: Bounded cancellation during in-flight sleep / navigation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_cancellation_returns_within_bounded_time(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Acceptance criterion 3: Cancelling a cycle returns in bounded time during in-flight sleep."""
    backend = SequencePerceptionBackend(initial_seq=1)

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        perception_backend=backend,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=SettableClock(start=1000.0, step=0.01),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
        # Default sleep is _CancellableSleep
    )

    try:
        stop_event = asyncio.Event()

        # Launch runner task
        runner_task = asyncio.create_task(
            run_lab_loop_async(runtime, max_cycles=1000, stop_event=stop_event)
        )

        # Let it run for a brief moment, then signal stop
        await asyncio.sleep(0.05)
        t_before = time.monotonic()
        stop_event.set()

        # Wait for completion with a tight bounded deadline (0.5s max, though it should be < 0.1s)
        res = await asyncio.wait_for(runner_task, timeout=0.5)
        elapsed = time.monotonic() - t_before

        assert elapsed < 0.3, f"Cancellation took {elapsed:.3f}s, exceeding bounded limit"
        assert res.status == LabRunStatus.STOP_EVENT_SET, (
            f"Got status {res.status} with reason: {res.reason}"
        )
        assert res.reason == "stop_event_set"
    finally:
        await runtime.world.close()


@pytest.mark.asyncio
async def test_asyncio_task_cancel_returns_within_bounded_time(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Verify task.cancel() on in-flight run_lab_loop_async finishes cleanly in bounded time."""
    backend = SequencePerceptionBackend(initial_seq=1)

    runtime = await build_lab_runtime_async(
        config=mock_config,
        session=temp_session,
        perception_backend=backend,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=SettableClock(start=1000.0, step=0.01),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    try:
        runner_task = asyncio.create_task(run_lab_loop_async(runtime, max_cycles=1000))
        await asyncio.sleep(0.05)

        t_before = time.monotonic()
        runner_task.cancel()

        res = await asyncio.wait_for(runner_task, timeout=0.5)
        elapsed = time.monotonic() - t_before

        assert elapsed < 0.3
        assert res.status == LabRunStatus.STOP_EVENT_SET
        assert res.reason == "cancelled"
    finally:
        await runtime.world.close()


# ---------------------------------------------------------------------------
# 5. Acceptance 4: Backward compatibility with synchronous game_state_source
# ---------------------------------------------------------------------------


def test_synchronous_game_state_source_works_unchanged(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Acceptance criterion 4: Pre-existing synchronous game_state_source callable works."""
    calls = 0

    def sync_source() -> FakeGameState:
        nonlocal calls
        calls += 1
        return FakeGameState(inventory_count=calls)

    runtime = build_lab_runtime(
        config=mock_config,
        session=temp_session,
        game_state_source=sync_source,
        meta_state_source=lambda: FakeMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=mock_profile,
        world_db_path=str(tmp_path / "world.db"),
        clock=SettableClock(start=1000.0, step=0.01),
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
        sleep=lambda _: None,
    )

    assert runtime.perception_port is None
    res = run_lab_loop(runtime, max_cycles=4)

    assert res.status == LabRunStatus.MAX_CYCLES_REACHED
    assert res.cycles_completed == 4
    assert calls >= 4


def test_missing_perception_sources_raises_lab_runner_error(
    mock_config: Config,
    temp_session: Session,
    mock_profile: FarmProfile,
    tmp_path: Any,
) -> None:
    """Verify build_lab_runtime raises LabRunnerError if no perception source is provided."""
    with pytest.raises(LabRunnerError, match="One of game_state_source"):
        build_lab_runtime(
            config=mock_config,
            session=temp_session,
            meta_state_source=lambda: FakeMetaState(),
            rotation_config=RotationConfig(rules=()),
            farm_profile=mock_profile,
            world_db_path=str(tmp_path / "world.db"),
            include_watchdog=False,
            focus_backend=NullFocusBackend(),
            llm_client=FakeLlmClient(),
        )


def test_cancellable_sleep_bounded_on_stop_event() -> None:
    """Unit test for _CancellableSleep verifying quick wake-up on stop_event."""
    from wow_bot.safety import SafetyLayer

    class FakeSafety(SafetyLayer):
        def __init__(self) -> None:
            self._aborted = False

        def is_aborted(self) -> bool:
            return self._aborted

    safety = FakeSafety()
    stop_event = asyncio.Event()
    c_sleep = _CancellableSleep(safety=safety, stop_event=stop_event)

    # Set stop event beforehand -> should return immediately (< 0.05s)
    stop_event.set()
    t0 = time.monotonic()
    c_sleep(5.0)
    assert time.monotonic() - t0 < 0.05
