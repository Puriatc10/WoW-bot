"""Tests for Safety lifecycle wiring (Task T-FIX-07).

Validates the mandatory calls from AGENTS.md §4.4:
- safety.check_isolation() at LAB_MODE startup (fails if sentinel is reachable,
  with no actuator constructed).
- unconditional driver input release on every loop exit path (success, stop, exception).
- dry_run=False in MOCK_MODE raises ModeError.
- critical failures invoke safety.abort with a reason.
- SafetyLayer is armed before actuator construction.
"""

from __future__ import annotations

import asyncio
import socket
from collections.abc import Generator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import wow_bot.lab.runner_v2 as runner_module
from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.driver import MouseButton
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
    LabRunnerConfig,
    LabRunStatus,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.mode import ModeError, enter_mode
from wow_bot.safety import IsolationViolation, SafetyLayer
from wow_bot.session import Session


@dataclass
class DummyGameState:
    """Minimal fake game state for safety tests."""

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
    inventory_count: int = 2
    inventory_max: int = 20
    durability_fraction: float | None = 1.0
    level: float = 1.0
    xp: float = 100.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        return True


@dataclass
class DummyMetaState:
    """Minimal fake meta state."""

    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


def _find_unused_port() -> int:
    """Find an unused TCP port on localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


@pytest.fixture
def test_profile() -> FarmProfile:
    """Create a minimal valid FarmProfile."""
    return FarmProfile(
        schema_version=1,
        name="safety_test_profile",
        description="Profile for safety lifecycle tests",
        cycle=CycleSpec(
            nodes=(NodeReference(kind="mob", name="mob_1"),),
            vendor=VendorReference(kind="vendor", name="vendor_1"),
            repair=VendorReference(kind="vendor", name="vendor_1"),
            stop_when_inventory_full=False,
            stop_after_cycles=0,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )


@pytest.fixture
def lab_config(tmp_path: Path) -> Config:
    """Create a valid LAB_MODE configuration."""
    unused_port = _find_unused_port()
    sess_root = tmp_path / "runs"
    sess_root.mkdir(parents=True, exist_ok=True)
    return Config(
        lab_mode=True,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel=f"127.0.0.1:{unused_port}",
        kill_switch_key="F12",
        session_root=sess_root,
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )


@pytest.fixture
def lab_session(lab_config: Config) -> Generator[Session, None, None]:
    """Create a temporary Session."""
    sess = Session.start(lab_config)
    try:
        yield sess
    finally:
        sess.close("test_finished")


@pytest.mark.asyncio
async def test_lab_mode_startup_fails_on_isolation_violation_no_actuator_constructed(
    tmp_path: Path,
    test_profile: FarmProfile,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 1: LAB_MODE startup fails when isolation check fails, with no actuator constructed."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    reachable_port = int(listener.getsockname()[1])

    actuator_constructed = False
    original_make_actuator = runner_module.make_actuator

    def spy_make_actuator(*args: Any, **kwargs: Any) -> Any:
        nonlocal actuator_constructed
        actuator_constructed = True
        return original_make_actuator(*args, **kwargs)

    monkeypatch.setattr(runner_module, "make_actuator", spy_make_actuator)

    try:
        cfg = Config(
            lab_mode=True,
            server_allowlist=["127.0.0.1:8080"],
            isolation_sentinel=f"127.0.0.1:{reachable_port}",
            kill_switch_key="F12",
            session_root=tmp_path / "runs",
            dry_run=False,
            max_session_seconds=3600,
            log_level="INFO",
        )
        sess = Session.start(cfg)
        try:
            with pytest.raises(IsolationViolation):
                await build_lab_runtime_async(
                    config=cfg,
                    session=sess,
                    game_state_source=lambda: DummyGameState(),
                    meta_state_source=lambda: DummyMetaState(),
                    rotation_config=RotationConfig(rules=()),
                    farm_profile=test_profile,
                    world_db_path=str(tmp_path / "world.db"),
                    driver_name="null",
                    include_watchdog=False,
                    include_reflex_loop=False,
                    focus_backend=NullFocusBackend(),
                )

            assert actuator_constructed is False, "No actuator should be constructed when isolation check fails"
        finally:
            sess.close("test_finished")
    finally:
        listener.close()


@pytest.mark.asyncio
async def test_all_held_keys_and_buttons_released_on_success(
    lab_config: Config,
    lab_session: Session,
    test_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Acceptance 2a: All held keys and buttons are released on successful loop exit."""
    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=lambda: DummyGameState(),
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=test_profile,
        world_db_path=str(tmp_path / "world.db"),
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        # Simulate held inputs on the driver prior to loop completion
        runtime.driver.key_down("w")
        runtime.driver.key_down("shift")
        runtime.driver.mouse_down(MouseButton.LEFT)

        assert runtime.driver.held_keys() == frozenset({"w", "shift"})
        assert runtime.driver.held_buttons() == frozenset({MouseButton.LEFT})

        res = await run_lab_loop_async(runtime, max_cycles=1)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED

        # Unconditionally released on loop exit
        assert runtime.driver.held_keys() == frozenset()
        assert runtime.driver.held_buttons() == frozenset()

        records = runtime.driver.records()
        assert any(sample.action == "release_all" for sample in records)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_all_held_keys_and_buttons_released_on_stop(
    lab_config: Config,
    lab_session: Session,
    test_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Acceptance 2b: All held keys and buttons are released when loop is stopped via stop_event."""
    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=lambda: DummyGameState(),
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=test_profile,
        world_db_path=str(tmp_path / "world.db"),
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        runtime.driver.key_down("a")
        runtime.driver.mouse_down(MouseButton.RIGHT)

        assert runtime.driver.held_keys() == frozenset({"a"})
        assert runtime.driver.held_buttons() == frozenset({MouseButton.RIGHT})

        stop_event = asyncio.Event()
        stop_event.set()

        res = await run_lab_loop_async(runtime, max_cycles=10, stop_event=stop_event)
        assert res.status == LabRunStatus.STOP_EVENT_SET

        # All inputs released
        assert runtime.driver.held_keys() == frozenset()
        assert runtime.driver.held_buttons() == frozenset()
        records = runtime.driver.records()
        assert any(sample.action == "release_all" for sample in records)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_all_held_keys_and_buttons_released_on_injected_exception(
    lab_config: Config,
    lab_session: Session,
    test_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Acceptance 2c: All held keys and buttons are released when an exception is raised inside the loop."""
    def _faulty_state() -> DummyGameState:
        raise RuntimeError("injected fault in state reading")

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=_faulty_state,
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=test_profile,
        world_db_path=str(tmp_path / "world.db"),
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        runtime.driver.key_down("space")
        runtime.driver.mouse_down(MouseButton.MIDDLE)

        assert runtime.driver.held_keys() == frozenset({"space"})
        assert runtime.driver.held_buttons() == frozenset({MouseButton.MIDDLE})

        res = await run_lab_loop_async(runtime, max_cycles=5)
        assert res.status == LabRunStatus.RUNTIME_ERROR

        # All inputs unconditionally released despite runtime error
        assert runtime.driver.held_keys() == frozenset()
        assert runtime.driver.held_buttons() == frozenset()
        records = runtime.driver.records()
        assert any(sample.action == "release_all" for sample in records)
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_dry_run_false_in_mock_mode_raises(
    tmp_path: Path,
    test_profile: FarmProfile,
) -> None:
    """Acceptance 3: dry_run=False in MOCK_MODE raises ModeError."""
    cfg = Config(
        lab_mode=False,
        server_allowlist=["127.0.0.1:8080"],
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )
    sess = Session.start(cfg)
    try:
        # Direct enter_mode test
        with pytest.raises(ModeError, match="MOCK_MODE requires dry_run=True"):
            enter_mode(cfg, sess)

        # Lab builder test
        with pytest.raises(ModeError, match="MOCK_MODE requires dry_run=True"):
            await build_lab_runtime_async(
                config=cfg,
                session=sess,
                game_state_source=lambda: DummyGameState(),
                meta_state_source=lambda: DummyMetaState(),
                rotation_config=RotationConfig(rules=()),
                farm_profile=test_profile,
                world_db_path=str(tmp_path / "world.db"),
                driver_name="null",
                include_watchdog=False,
                include_reflex_loop=False,
                focus_backend=NullFocusBackend(),
            )
    finally:
        sess.close("test_finished")


@pytest.mark.asyncio
async def test_critical_failure_invokes_safety_abort_with_reason(
    lab_config: Config,
    lab_session: Session,
    test_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Acceptance 4: Critical failure invokes safety.abort with a descriptive reason."""
    def _exploding_state() -> DummyGameState:
        raise RuntimeError("critical hardware sensor failure")

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=_exploding_state,
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=test_profile,
        world_db_path=str(tmp_path / "world.db"),
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        assert runtime.safety.is_aborted() is False

        res = await run_lab_loop_async(runtime, max_cycles=5)
        assert res.status == LabRunStatus.RUNTIME_ERROR

        assert runtime.safety.is_aborted() is True
        reason = runtime.safety.abort_reason()
        assert reason is not None
        assert "critical hardware sensor failure" in reason
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_consecutive_failures_invokes_safety_abort(
    lab_config: Config,
    lab_session: Session,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 4b: Reaching max consecutive failures triggers safety.abort."""
    # Force _run_cycle to fail repeatedly
    fail_count = 0

    async def _failing_cycle(*args: Any, **kwargs: Any) -> tuple[str, Any, bool, Any]:
        nonlocal fail_count
        fail_count += 1
        return ("farm", None, True, None)

    monkeypatch.setattr(runner_module, "_run_cycle", _failing_cycle)

    prof = FarmProfile(
        schema_version=1,
        name="test",
        description="",
        cycle=CycleSpec(
            nodes=(NodeReference(kind="mob", name="m1"),),
            vendor=VendorReference(kind="vendor", name="v1"),
            repair=VendorReference(kind="vendor", name="v1"),
            stop_when_inventory_full=False,
            stop_after_cycles=0,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )
    rcfg = LabRunnerConfig(max_consecutive_failures=3)
    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=lambda: DummyGameState(),
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=prof,
        world_db_path=str(tmp_path / "world.db"),
        runner_config=rcfg,
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        res = await run_lab_loop_async(runtime, max_cycles=10)
        assert res.status == LabRunStatus.MAX_FAILURES_REACHED
        assert runtime.safety.is_aborted() is True
        assert runtime.safety.abort_reason() == "max_consecutive_failures_reached"
    finally:
        await runtime.close()


@pytest.mark.asyncio
async def test_safety_layer_armed_before_actuator_exists(
    lab_config: Config,
    lab_session: Session,
    test_profile: FarmProfile,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance 5: SafetyLayer is armed before the actuator exists (ordering asserted, not assumed)."""
    event_order: list[str] = []

    original_arm = SafetyLayer.arm

    def spy_arm(self: SafetyLayer) -> None:
        event_order.append("safety_armed")
        original_arm(self)

    monkeypatch.setattr(SafetyLayer, "arm", spy_arm)

    original_make_actuator = runner_module.make_actuator

    def spy_make_actuator(*args: Any, **kwargs: Any) -> Any:
        safety = kwargs.get("safety")
        assert safety is not None, "safety must be passed to make_actuator"
        assert safety._armed is True, "SafetyLayer must already be armed when actuator is created"
        event_order.append("actuator_constructed")
        return original_make_actuator(*args, **kwargs)

    monkeypatch.setattr(runner_module, "make_actuator", spy_make_actuator)

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=lab_session,
        game_state_source=lambda: DummyGameState(),
        meta_state_source=lambda: DummyMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=test_profile,
        world_db_path=str(tmp_path / "world.db"),
        driver_name="null",
        include_watchdog=False,
        include_reflex_loop=False,
        focus_backend=NullFocusBackend(),
    )
    try:
        assert event_order == ["safety_armed", "actuator_constructed"]
        assert runtime.safety._armed is True
    finally:
        await runtime.close()
