"""Acceptance tests for T-FIX-12: Reflex tick truth.

Validates:
  1. N ticks with zero signals report N (authoritative total_ticks / tick_index).
  2. The jitter series over a synthetic schedule is unbiased (no systematic omission of quiet ticks).
  3. The runner's reported tick count equals the loop's tick_index (and 0 when no loop, no fabrication).
  4. Log volume stays bounded over a long quiet run (periodic summary without per-tick quiet events).
"""

from __future__ import annotations

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
    build_lab_runtime_async,
    run_lab_loop_async,
)
from wow_bot.reflex.controls import ControlKind, ControlSignal, NullControlSink
from wow_bot.reflex.loop import (
    NullClock,
    ReflexLoop,
    compute_jitter_series,
)
from wow_bot.reflex.signals import NullSignalSource, Signal
from wow_bot.session import Session


class ScriptedClock:
    """Clock returning scripted sequence of timestamps in seconds."""

    def __init__(self, timestamps: list[float]) -> None:
        self._timestamps = timestamps
        self._idx = 0

    def now(self) -> float:
        if self._idx < len(self._timestamps):
            val = self._timestamps[self._idx]
            self._idx += 1
            return val
        return self._timestamps[-1] if self._timestamps else 0.0

    def sleep(self, seconds: float) -> None:
        pass


@dataclass
class FakeGameState:
    """Minimal fake game state for runner tests."""

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
        return '{"goal": "farm", "target": null, "rationale": "continue"}'


def test_n_ticks_zero_signals_report_n() -> None:
    """Criterion 1: N ticks with zero signals report N.

    ReflexLoop.tick_index and ReflexTelemetry report N, with no silent zero
    and no omission of quiet ticks.
    """
    n_ticks = 100

    def rules(signals: list[Signal], tick_index: int, rng: Any) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(NullSignalSource(),),
        sinks=(NullControlSink(),),
        rules=rules,
        clock=NullClock(),
    )

    for _ in range(n_ticks):
        loop.tick()

    assert loop.tick_index == n_ticks
    assert loop.total_ticks == n_ticks

    telem = loop.telemetry()
    assert telem.tick_index == n_ticks
    assert telem.total_ticks == n_ticks
    assert telem.signals_processed_total == 0
    assert telem.controls_emitted_total == 0
    assert telem.overruns_total == 0
    assert telem.sink_errors_total == 0
    # For N ticks, N - 1 intervals are recorded in the jitter series
    assert telem.jitter_samples_count == n_ticks - 1
    assert len(loop.get_jitter_series()) == n_ticks - 1


def test_jitter_series_unbiased_over_synthetic_schedule() -> None:
    """Criterion 2: The jitter series over a synthetic schedule is unbiased (no systematic omission).

    A schedule where only 1 tick has signals while others are quiet must yield
    the complete, full-schedule jitter series rather than an event-filtered subset.
    """
    # 10 ticks at 20 Hz (nominal period 0.050s = 50ms)
    # Timestamps in pairs (tick_start, tick_end) for each tick:
    # tick 0: start=0.000, end=0.005
    # tick 1: start=0.052, end=0.057 -> interval 0.052s -> jitter +2.0ms
    # tick 2: start=0.098, end=0.103 -> interval 0.046s -> jitter -4.0ms
    # tick 3: start=0.155, end=0.160 -> interval 0.057s -> jitter +7.0ms (HAS SIGNAL)
    # tick 4: start=0.201, end=0.206 -> interval 0.046s -> jitter -4.0ms
    # tick 5: start=0.248, end=0.253 -> interval 0.047s -> jitter -3.0ms
    # tick 6: start=0.305, end=0.310 -> interval 0.057s -> jitter +7.0ms
    # tick 7: start=0.350, end=0.355 -> interval 0.045s -> jitter -5.0ms
    # tick 8: start=0.402, end=0.407 -> interval 0.052s -> jitter +2.0ms
    # tick 9: start=0.451, end=0.456 -> interval 0.049s -> jitter -1.0ms
    timestamps = [
        0.000, 0.005,
        0.052, 0.057,
        0.098, 0.103,
        0.155, 0.160,
        0.201, 0.206,
        0.248, 0.253,
        0.305, 0.310,
        0.350, 0.355,
        0.402, 0.407,
        0.451, 0.456,
    ]
    expected_jitters = [2.0, -4.0, 7.0, -4.0, -3.0, 7.0, -5.0, 2.0, -1.0]

    clock = ScriptedClock(timestamps)

    # Emit signal ONLY on tick 3
    def rules(signals: list[Signal], tick_index: int, rng: Any) -> list[ControlSignal]:
        if tick_index == 3:
            return [ControlSignal(kind=ControlKind.PAUSE_FSM, reason="test", ts=0.155)]
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=clock,
    )

    stats_list = [loop.tick() for _ in range(10)]

    # 1. Direct loop jitter series
    jitter_from_loop = loop.get_jitter_series()
    assert len(jitter_from_loop) == 9
    for actual, expected in zip(jitter_from_loop, expected_jitters, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6)

    # 2. Standalone compute_jitter_series from stats
    jitter_from_stats = compute_jitter_series(stats_list, rate_hz=20.0)
    assert len(jitter_from_stats) == 9
    for actual, expected in zip(jitter_from_stats, expected_jitters, strict=True):
        assert actual == pytest.approx(expected, abs=1e-6)

    # 3. Prove that an event-filtered subset would have omitted quiet ticks
    filtered_stats = [s for s in stats_list if s.controls_emitted > 0]
    assert len(filtered_stats) == 1
    # Filtering produces only 1 tick and 0 intervals — demonstrating that the unfiltered series
    # is essential for avoiding systematic omission.
    assert len(compute_jitter_series(filtered_stats, rate_hz=20.0)) == 0


def make_profile() -> FarmProfile:
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
async def test_runner_reported_tick_count_equals_loop_tick_index(tmp_path: Path) -> None:
    """Criterion 3: The runner's reported tick count equals the loop's tick_index."""
    cfg = Config(
        lab_mode=True,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(cfg)
    profile = make_profile()
    state = FakeGameState()

    def state_source() -> FakeGameState:
        state.player_x += 0.1
        state.player_y += 0.1
        return state

    # Test 3a: With reflex loop enabled
    runtime = await build_lab_runtime_async(
        config=cfg,
        session=session,
        game_state_source=state_source,
        meta_state_source=FakeMetaState,
        rotation_config=RotationConfig(rules=()),
        farm_profile=profile,
        world_db_path=str(tmp_path / "world.db"),
        include_reflex_loop=True,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    try:
        assert runtime.reflex_loop is not None
        assert runtime.reflex_loop.tick_index == 0

        res = await run_lab_loop_async(runtime, max_cycles=3)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 3

        # Runner reported tick count in result matches the loop's own tick_index
        assert res.reflex_ticks_total == runtime.reflex_loop.tick_index
        assert res.reflex_ticks_total > 0

        # Runner's progress tracker latest sample also matches the loop's own tick_index
        assert runtime.progress._buffer[-1].reflex_ticks_total == runtime.reflex_loop.tick_index
    finally:
        await runtime.world.close()


@pytest.mark.asyncio
async def test_runner_reported_tick_count_zero_when_no_loop(tmp_path: Path) -> None:
    """Criterion 3b: When reflex loop is excluded, reported count is 0 (no synthesized fabrication)."""
    cfg = Config(
        lab_mode=True,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(cfg)
    profile = make_profile()
    state = FakeGameState()

    runtime = await build_lab_runtime_async(
        config=cfg,
        session=session,
        game_state_source=lambda: state,
        meta_state_source=FakeMetaState,
        rotation_config=RotationConfig(rules=()),
        farm_profile=profile,
        world_db_path=str(tmp_path / "world.db"),
        include_reflex_loop=False,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=FakeLlmClient(),
    )

    try:
        assert runtime.reflex_loop is None

        res = await run_lab_loop_async(runtime, max_cycles=3)
        # Must report 0, not fabricated cycles * 20 = 60
        assert res.reflex_ticks_total == 0
        assert not runtime.progress._buffer or runtime.progress._buffer[-1].reflex_ticks_total == 0
    finally:
        await runtime.world.close()


def test_log_volume_stays_bounded_over_long_quiet_run(tmp_path: Path) -> None:
    """Criterion 4: Log volume stays bounded over a long quiet run.

    5000 quiet ticks with summary_interval_ticks=100 emits exactly 50 summary events,
    with no raw per-tick events polluting session logs.
    """
    cfg = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(cfg)

    def rules(signals: list[Signal], tick_index: int, rng: Any) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=NullClock(),
        session=session,
        summary_interval_ticks=100,
    )

    n_ticks = 5000
    for _ in range(n_ticks):
        loop.tick()

    events_file = session.path / "events.jsonl"
    lines = [line for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]

    # 5000 ticks / 100 per summary = 50 summary events
    assert len(lines) == 50
    # Every event is reflex_summary, no uncompressed reflex_tick events
    assert all('"event": "reflex_summary"' in line for line in lines)

    # Verify total_ticks in the final event matches 5000
    import json
    last_event = json.loads(lines[-1])
    assert last_event["payload"]["total_ticks"] == 5000
    assert last_event["payload"]["tick_index"] == 5000
    assert last_event["payload"]["period_ticks"] == 100
    assert last_event["payload"]["signals_processed_total"] == 0
