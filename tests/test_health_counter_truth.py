"""Acceptance tests for T-FIX-13: Health counter truth.

Validates:
  1. A run with no actions yields zero action and health counters (no increments from cycle iteration alone).
  2. A repeated identical action with unchanged state is detected as repetition (loop detector triggers).
  3. Each reported counter matches the underlying event stream count (counters agree with events.jsonl).
  4. The cycle index is NOT part of the action signature (make_action_signature excludes cycle indices).
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.backends.focus_null import NullFocusBackend
from wow_bot.actuation.mapper import ActionStatus, Cast, Intent
from wow_bot.combat.rotation import RotationConfig
from wow_bot.config import Config
from wow_bot.executor.feedback import Feedback
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
    make_action_signature,
    run_lab_loop_async,
)
from wow_bot.session import Session
from wow_bot.watchdog.loops import LoopConfig, LoopDetector


@dataclass
class FakeGameState:
    """Minimal fake game state for health counter truth tests."""

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


class ScriptedFSMBehavior:
    """Custom behavior that returns scripted intents on decide."""

    def __init__(self, intents: list[Intent | None]) -> None:
        self._intents = intents
        self._call_count = 0

    def decide(self, *args: Any, **kwargs: Any) -> Intent | None:
        if self._call_count < len(self._intents):
            intent = self._intents[self._call_count]
        else:
            intent = self._intents[-1] if self._intents else None
        self._call_count += 1
        return intent

    def handle_feedback(self, feedback: Feedback, now: float) -> None:
        pass


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


def make_config(tmp_path: Path) -> Config:
    return Config(
        lab_mode=True,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path / "runs",
        dry_run=False,
        max_session_seconds=3600,
        log_level="INFO",
    )


@pytest.mark.asyncio
async def test_run_with_no_actions_yields_zero_action_and_health_counters(tmp_path: Path) -> None:
    """Acceptance 1: A run with no actions yields zero action and health counters.

    Demonstrates that counters do not increment merely from cycle iteration.
    """
    cfg = make_config(tmp_path)
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

    # Behavior returning None guarantees no action is executed in any cycle
    runtime.fsm._behavior = ScriptedFSMBehavior([None])

    try:
        res = await run_lab_loop_async(runtime, max_cycles=5)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 5

        # 1. Action counter in result must be exactly zero despite 5 completed cycles
        assert res.successful_actions_total == 0

        # 2. Progress snapshot health rate must be 0.0 actions per minute
        snapshot = runtime.progress.snapshot()
        assert snapshot.successful_actions_per_minute == 0.0

        # 3. Buffer samples must have successful_actions_total == 0
        assert runtime.progress.sample_count > 0
        assert runtime.progress._buffer[-1].successful_actions_total == 0

        # 4. Underlying event stream must contain zero action events
        events_file = session.path / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        action_events = [
            e for e in events
            if e.get("event") in ("humanizer_action", "actuator_result")
        ]
        assert len(action_events) == 0

        # 5. lab_run_finished event recorded in session also reports 0 successful actions
        finished_events = [e for e in events if e.get("event") == "lab_run_finished"]
        assert len(finished_events) == 1
        assert finished_events[0]["successful_actions_total"] == 0
    finally:
        await runtime.close()
        session.close("test_complete")


@pytest.mark.asyncio
async def test_repeated_identical_action_detected_as_repetition(tmp_path: Path) -> None:
    """Acceptance 2: A repeated identical action with unchanged state is detected as repetition.

    Because the cycle index is removed from the action signature, identical repeated
    actions form repeating signature patterns which the LoopDetector detects.
    """
    cfg = make_config(tmp_path)
    session = Session.start(cfg)
    profile = make_profile()
    state = FakeGameState()

    # Configure loop detector with cycle_length=2, min_repeats=2 (4 identical actions trigger it)
    fast_loop_detector = LoopDetector(
        config=LoopConfig(
            cycle_length=2,
            min_repeats=2,
            window_s=60.0,
            stagnation_epsilon=0.0,
            emit_on_stagnation_only=True,
        ),
        session=session,
    )

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

    # Inject the fast loop detector via dataclasses.replace
    runtime = dataclasses.replace(runtime, loops=fast_loop_detector)
    cast_intent = Cast(spell_id="shoot", target_id="mob_1")
    runtime.fsm._behavior = ScriptedFSMBehavior([cast_intent])

    try:
        res = await run_lab_loop_async(runtime, max_cycles=10)

        # Repetition must be detected by LoopDetector and trigger LOOP_DETECTED status
        assert res.status == LabRunStatus.LOOP_DETECTED
        assert res.reason == "loop_detected"
        assert runtime.safety.is_aborted() is True
        assert runtime.safety.abort_reason() == "loop_detected"

        # Verify loop detector session event was emitted
        events_file = session.path / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        loop_events = [e for e in events if e.get("event") == "watchdog_loop_detected"]
        assert len(loop_events) >= 1
        assert loop_events[0]["repeats"] >= 2
    finally:
        await runtime.close()
        session.close("test_complete")


@pytest.mark.asyncio
async def test_each_reported_counter_matches_underlying_event_stream_count(tmp_path: Path) -> None:
    """Acceptance 3: Each reported counter matches the underlying event stream count.

    Counters agree with the session event stream: an empty event stream means zero action
    counters, and N successful actions yield exactly N session events and counter=N.
    """
    cfg = make_config(tmp_path)
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

    # Script 3 distinct actions across 3 cycles
    intents = [
        Cast(spell_id="frostbolt", target_id="mob_1"),
        Cast(spell_id="fireball", target_id="mob_1"),
        Cast(spell_id="shoot", target_id="mob_1"),
    ]
    runtime.fsm._behavior = ScriptedFSMBehavior(intents)

    try:
        res = await run_lab_loop_async(runtime, max_cycles=3)
        assert res.status == LabRunStatus.MAX_CYCLES_REACHED
        assert res.cycles_completed == 3
        assert res.successful_actions_total == 3

        # Read events.jsonl and count underlying action result events
        events_file = session.path / "events.jsonl"
        events = [
            json.loads(line)
            for line in events_file.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        humanizer_success_events = [
            e for e in events
            if e.get("event") == "humanizer_action" and e.get("status") == ActionStatus.SUCCESS.value
        ]
        assert len(humanizer_success_events) == 3
        # Strict 1-to-1 match between reported counter and event stream count
        assert res.successful_actions_total == len(humanizer_success_events)

        # Last ProgressSample in tracker also matches
        last_sample = runtime.progress._buffer[-1]
        assert last_sample.successful_actions_total == len(humanizer_success_events)
    finally:
        await runtime.close()
        session.close("test_complete")


def test_cycle_index_is_not_part_of_action_signature() -> None:
    """Acceptance 4: The cycle index is NOT part of the action signature.

    Verifies make_action_signature creates deterministic signatures without
    monotonically increasing quantities such as cycle counts.
    """
    cast_intent = Cast(spell_id="shoot", target_id="mob_1")

    # Invariant across different cycles
    sig_cycle_0 = make_action_signature("farm", FSMState.COMBAT, repr(cast_intent))
    sig_cycle_100 = make_action_signature("farm", FSMState.COMBAT, repr(cast_intent))

    assert sig_cycle_0 == sig_cycle_100
    assert "cycle" not in sig_cycle_0.lower()
    assert sig_cycle_0 == "farm:COMBAT:Cast(spell_id='shoot', target_id='mob_1')"

    # Check idle signature
    idle_sig = make_action_signature("farm", FSMState.IDLE, "no_action")
    assert idle_sig == "farm:IDLE:no_action"
    assert "cycle" not in idle_sig.lower()
