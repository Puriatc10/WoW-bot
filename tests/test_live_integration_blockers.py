"""Tests for Phase 13 pre-live integration and safety blockers.

Verifies:
1. Win32 focus backend selection/injection behavior.
2. Focus-loss actuation abort.
3. Physical kill switch wiring via a fake backend.
4. Valid MetaState reaching strategist without PROMPT_BUILD_ERROR.
5. TOML rotation loading in run_farm_v2 matching live_soak.
6. Live pre-flight PASS/FAIL/NEEDS_LIVE_CALIBRATION semantics.
7. Runtime cleanup on normal completion.
8. Runtime cleanup on exception.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.mapper import ActionStatus, Turn
from wow_bot.combat.rotation import RotationConfig, load_rotation_from_dict
from wow_bot.config import Config
from wow_bot.executor.fsm_v2 import FSMState
from wow_bot.farm.profile import (
    CycleSpec,
    FarmProfile,
    NodeReference,
    RoutePreferences,
    VendorReference,
)
from wow_bot.kill_switch import NullBackend
from wow_bot.lab.runner_v2 import (
    LabRunnerError,
    LabRunStatus,
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.session import Session
from wow_bot.strategist.cooldown_v2 import CooldownConfig, CooldownGate
from wow_bot.strategist.orchestrator_v2 import (
    OrchestratorConfig,
    OrchestratorOutcome,
    OrchestratorV2,
)
from wow_bot.strategist.prompts_v2 import LabMetaStateAdapter, build_prompt
from wow_bot.strategist.vocab_v2 import VocabConfig, VocabularyGuard
from wow_bot.world.summary import WorldSummary


@dataclass
class FakeGameState:
    """Fake GameState for runner tests."""

    player_x: float = 10.0
    player_y: float = 20.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 10.0
    self_y: float = 20.0
    self_hp_percent: float = 100.0
    self_mana_percent: float = 100.0
    self_in_combat: bool = False
    fsm_state: FSMState = FSMState.IDLE
    inventory_count: int = 0
    inventory_max: int = 16
    level: float = 1.0
    xp: float = 0.0
    level_or_xp: float = 1.0
    current_target_id: str | None = None
    target_entity_id: str | None = None
    target_in_range: bool = False
    target_is_alive: bool = False
    target_is_lootable: bool = False
    target_distance: float | None = None
    target_hp_percent: float | None = None
    resource: float = 100.0
    resource_max: float = 100.0
    entities: tuple[Any, ...] = ()
    target: Any = None


class FakeFocusBackend:
    """Controllable focus backend for tests."""

    def __init__(self, focused: bool = True, window_handle: object = "fake_hwnd") -> None:
        self._focused = focused
        self._handle = window_handle
        self.closed = False

    def find_window(self, title_substring: str) -> object | None:
        return self._handle

    def get_foreground_handle(self) -> object | None:
        return self._handle if self._focused else "other_hwnd"

    def same_window(self, a: object, b: object) -> bool:
        return a == b and a is not None

    def set_focused(self, focused: bool) -> None:
        self._focused = focused

    def close(self) -> None:
        self.closed = True


class FakeKillSwitchBackend:
    """Controllable kill switch backend for tests."""

    def __init__(self) -> None:
        self.callback: Any = None
        self.started = False
        self.stopped = False
        self.stop_count = 0

    def start(self, on_trigger: Any) -> None:
        self.callback = on_trigger
        self.started = True

    def stop(self) -> None:
        self.stopped = True
        self.stop_count += 1

    def trigger(self) -> None:
        if self.callback is not None:
            self.callback()


def make_dummy_world() -> WorldSummary:
    """Helper constructing an empty WorldSummary."""
    return WorldSummary(
        around_xy=(10.0, 20.0),
        radius=100.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=(),
        nearest_trainers=(),
        nearest_nodes=(),
        nearest_mobs=(),
        nearest_waypoints=(),
        recent_combats=(),
        total_nodes=0,
    )


@pytest.fixture
def lab_config(tmp_path: Path) -> Config:
    """Fixture providing valid LAB Config."""
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
    """Fixture providing valid FarmProfile."""
    return FarmProfile(
        schema_version=1,
        name="test_farm",
        description="test",
        cycle=CycleSpec(
            nodes=(NodeReference(kind="mob", name="mob_1"),),
            vendor=VendorReference(kind="vendor", name="vendor_1"),
            repair=VendorReference(kind="vendor", name="vendor_1"),
            stop_when_inventory_full=False,
            stop_after_cycles=1,
        ),
        route_preferences=RoutePreferences(),
        metadata={},
    )


@pytest.fixture(autouse=True)
def clean_pynput_from_sys_modules() -> Any:
    """Ensure pynput does not leak into sys.modules and fail static import isolation tests."""
    yield
    for mod in list(sys.modules):
        if mod == "pynput" or mod.startswith("pynput."):
            sys.modules.pop(mod, None)


# ---------------------------------------------------------------------------
# 1. Win32 focus backend selection/injection behavior
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_live_runner_selects_win32_focus_on_windows_without_injection(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify live LAB runner selects Win32FocusBackend on Windows when none is supplied."""
    if sys.platform != "win32":
        pytest.skip("Windows only")

    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")

    # When no focus backend is passed in live mode and window is missing -> fails closed!
    with pytest.raises(LabRunnerError, match="Failed to initialize focus manager"):
        await build_lab_runtime_async(
            config=lab_config,
            session=session,
            game_state_source=FakeGameState,
            meta_state_source=lambda: LabMetaStateAdapter(),
            rotation_config=RotationConfig(rules=()),
            farm_profile=farm_profile,
            world_db_path=db_path,
            driver_name="null",
            window_title="NonExistentWindow_123456789",
            live_mode=True,
            kill_switch_backend=NullBackend(),
            include_reflex_loop=False,
            include_watchdog=False,
        )


@pytest.mark.asyncio
async def test_live_runner_permits_injected_focus_backend(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify tests can still inject NullFocusBackend or custom fake backends."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")
    fake_backend = FakeFocusBackend(focused=True)

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=True,
        focus_backend=fake_backend,
        kill_switch_backend=NullBackend(),
        include_reflex_loop=False,
        include_watchdog=False,
    )

    assert runtime.focus._backend is fake_backend
    await runtime.close()


@pytest.mark.asyncio
async def test_mock_runner_defaults_to_null_focus_backend(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify MOCK mode runner defaults to NullFocusBackend without error."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=False,
        include_reflex_loop=False,
        include_watchdog=False,
    )

    assert isinstance(runtime.focus._backend, NullFocusBackend)
    await runtime.close()


# ---------------------------------------------------------------------------
# 2. Focus-loss actuation abort
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_focus_loss_causes_actuation_failure_and_abort(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify losing foreground focus causes actuator.execute to fail immediately."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")
    fake_backend = FakeFocusBackend(focused=True)

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=True,
        focus_backend=fake_backend,
        kill_switch_backend=NullBackend(),
        include_reflex_loop=False,
        include_watchdog=False,
    )

    # Initially focused: execute succeeds
    res1 = runtime.actuator.execute(Turn(angle_rad=0.1), position=(10.0, 20.0))
    assert res1.status == ActionStatus.SUCCESS

    # Lose focus!
    fake_backend.set_focused(False)
    # Trigger focus poll update
    runtime.focus._is_focused = False

    res2 = runtime.actuator.execute(Turn(angle_rad=0.1), position=(10.0, 20.0))
    assert res2.status == ActionStatus.FAILED
    assert "focus_lost" in res2.notes

    await runtime.close()


# ---------------------------------------------------------------------------
# 3. Physical kill switch wiring via a fake backend
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_kill_switch_wiring_and_abort(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify KillSwitch triggers SafetyLayer abort, release_all, and actuator abort."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")
    ks_backend = FakeKillSwitchBackend()

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=True,
        focus_backend=NullFocusBackend(),
        kill_switch_backend=ks_backend,
        include_reflex_loop=False,
        include_watchdog=False,
    )

    assert runtime.kill_switch is not None
    assert runtime.kill_switch._key == "F12"
    assert not runtime.safety.is_aborted()

    # Start loop or trigger directly
    runtime.kill_switch.start()
    assert ks_backend.started

    # Trigger kill switch!
    ks_backend.trigger()

    assert runtime.safety.is_aborted()
    assert runtime.safety.abort_reason() == "kill_switch_triggered"
    assert runtime.actuator.is_aborted()

    # Cleanup idempotency
    runtime.kill_switch.stop()
    runtime.kill_switch.stop()
    await runtime.close()
    assert ks_backend.stop_count >= 1


@pytest.mark.asyncio
async def test_mock_mode_does_not_start_physical_keyboard_listener(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify MOCK mode uses NullBackend and does not register a physical listener."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=False,
        include_reflex_loop=False,
        include_watchdog=False,
    )

    assert runtime.kill_switch is not None
    assert isinstance(runtime.kill_switch._backend, NullBackend)
    await runtime.close()


# ---------------------------------------------------------------------------
# 4. Valid MetaState reaching strategist
# ---------------------------------------------------------------------------

def test_lab_metastate_adapter_builds_prompt_without_error() -> None:
    """Verify LabMetaStateAdapter satisfies prompt builder and strategist without PROMPT_BUILD_ERROR."""
    meta = LabMetaStateAdapter(
        drives={"hunger": 0.1, "fatigue": 0.2, "curiosity": 0.5},
        memory_summary="Operating in isolated lab environment.",
        timestamp=100.0,
    )
    state = FakeGameState()
    world = make_dummy_world()

    bundle = build_prompt(meta, world, state)
    assert bundle.text
    assert bundle.prompt_hash
    assert "hunger: 0.100" in bundle.text
    assert "Operating in isolated lab environment." in bundle.text


def test_orchestrator_decide_with_lab_metastate_adapter() -> None:
    """Verify OrchestratorV2 can decide using LabMetaStateAdapter."""
    meta = LabMetaStateAdapter()
    state = FakeGameState()
    world = make_dummy_world()

    class FakeLlm:
        def complete(self, prompt: str) -> str:
            return '{"goal": "explore", "target": null, "rationale": "testing"}'

    orchestrator = OrchestratorV2(
        llm=FakeLlm(),
        cooldown=CooldownGate(config=CooldownConfig()),
        guard=VocabularyGuard(config=VocabConfig()),
        config=OrchestratorConfig(),
    )

    res = orchestrator.decide(
        meta=meta,
        world=world,
        state=state,
        now=10.0,
        fsm_state=FSMState.IDLE,
    )
    assert res.outcome == OrchestratorOutcome.SUCCESS
    assert res.strategy is not None
    assert res.strategy.goal == "explore"


# ---------------------------------------------------------------------------
# 5. TOML rotation loading
# ---------------------------------------------------------------------------

def test_toml_rotation_loading(tmp_path: Path) -> None:
    """Verify TOML combat rotation can be loaded and parsed matching live_soak."""
    toml_content = """
default_spell_id = "shoot"

[[rules]]
priority = 10
spell_id = "shadow_word_pain"

[[rules.conditions]]
kind = "target_in_range"
value = true

[[rules]]
priority = 20
spell_id = "mind_blast"

[[rules.conditions]]
kind = "target_in_range"
value = true
"""
    rot_file = tmp_path / "rotation.toml"
    rot_file.write_text(toml_content, encoding="utf-8")

    import tomllib
    with open(rot_file, "rb") as f:
        data = tomllib.load(f)

    rot_cfg = load_rotation_from_dict(data)
    assert len(rot_cfg.rules) == 2
    assert rot_cfg.rules[0].spell_id == "shadow_word_pain"
    assert rot_cfg.rules[1].spell_id == "mind_blast"


# ---------------------------------------------------------------------------
# 6. Live pre-flight PASS/FAIL/NEEDS_LIVE_CALIBRATION semantics
# ---------------------------------------------------------------------------

def test_verify_perception_calibration_categories() -> None:
    """Verify pre-flight gate distinguishes PASS, FAIL, and NEEDS_LIVE_CALIBRATION."""
    from scripts.lab.verify_perception_calibration import (
        check_dependencies,
        check_detector_assets,
        check_game_window,
        check_live_channels,
        parse_and_validate_capture_region,
    )

    # 1. Dependency checks
    deps = check_dependencies()
    assert "mss" in deps
    assert isinstance(deps["mss"][0], bool)

    # 2. Detector assets
    status, _ = check_detector_assets(Path("."), None)
    assert status == "PASS"

    status, _ = check_detector_assets(Path("."), "non_existent_weights.pt")
    assert status == "FAIL"

    # 3. Capture region validation
    st_pass, _, reg = parse_and_validate_capture_region("10,20,800,600")
    assert st_pass == "PASS"
    assert reg == (10, 20, 800, 600)

    st_fail, _, _ = parse_and_validate_capture_region("-10,20,800,600")
    assert st_fail == "FAIL"

    st_fail_fmt, _, _ = parse_and_validate_capture_region("invalid_format")
    assert st_fail_fmt == "FAIL"

    # 4. Live channels must report NEEDS_LIVE_CALIBRATION
    live_channels = check_live_channels()
    for name, status, detail in live_channels:
        assert status == "NEEDS_LIVE_CALIBRATION"

    # 5. Missing window returns NEEDS_LIVE_CALIBRATION (not a blocking crash)
    win_status, _ = check_game_window("DefinitivelyNonExistentWindow_999999")
    if sys.platform == "win32":
        assert win_status == "NEEDS_LIVE_CALIBRATION"


# ---------------------------------------------------------------------------
# 7. Runtime cleanup on normal completion
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runtime_cleanup_on_normal_completion(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify all managed resources clean up when run_lab_loop_async finishes normally."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")
    fake_focus = FakeFocusBackend(focused=True)
    fake_ks = FakeKillSwitchBackend()

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=FakeGameState,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=True,
        focus_backend=fake_focus,
        kill_switch_backend=fake_ks,
        include_reflex_loop=True,
        include_watchdog=False,
    )

    res = await run_lab_loop_async(runtime, max_cycles=1)
    assert res.status in (LabRunStatus.MAX_CYCLES_REACHED, LabRunStatus.COMPLETED)

    # Focus and kill switch should be stopped after loop completes
    assert fake_ks.stopped
    assert not runtime.reflex_loop.is_running()

    # close() is idempotent
    await runtime.close()
    assert fake_focus.closed


# ---------------------------------------------------------------------------
# 8. Runtime cleanup on exception
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_runtime_cleanup_on_exception(
    lab_config: Config,
    farm_profile: FarmProfile,
    tmp_path: Path,
) -> None:
    """Verify safety abort, release_all, and clean teardown when loop encounters exception."""
    session = Session.start(lab_config)
    db_path = str(tmp_path / "world.db")
    fake_focus = FakeFocusBackend(focused=True)
    fake_ks = FakeKillSwitchBackend()

    def buggy_game_state_source() -> Any:
        raise RuntimeError("Injected perception fault")

    runtime = await build_lab_runtime_async(
        config=lab_config,
        session=session,
        game_state_source=buggy_game_state_source,
        meta_state_source=lambda: LabMetaStateAdapter(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=farm_profile,
        world_db_path=db_path,
        driver_name="null",
        live_mode=True,
        focus_backend=fake_focus,
        kill_switch_backend=fake_ks,
        include_reflex_loop=True,
        include_watchdog=False,
    )

    res = await run_lab_loop_async(runtime, max_cycles=5)
    assert res.status == LabRunStatus.RUNTIME_ERROR
    assert "Injected perception fault" in res.reason

    # Safety should be aborted
    assert runtime.safety.is_aborted()

    # Resources stopped cleanly
    assert fake_ks.stopped
    assert not runtime.reflex_loop.is_running()

    await runtime.close()
    assert fake_focus.closed
