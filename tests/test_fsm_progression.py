"""Acceptance tests for T-FIX-05: FSM IDLE->SCANNING progression.

Verifies:
1. Cycle progresses IDLE -> SCANNING and reaches a subsequent state (MOVING_TO_TARGET/TARGETING),
   emitting at least one intent.
2. Illegal FSM state transitions raise TransitionError.
3. The value the strategist orchestrator sees for fsm_state tracks runtime FSM state.
4. Determinism without global RNG.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.mapper import MoveTo, Turn
from wow_bot.config import Config
from wow_bot.executor.fsm_v2 import FSM, FSMConfig, GameStateLike, MetaStateLike
from wow_bot.executor.states import FSMState, TransitionError
from wow_bot.lab.runner_v2 import _ScanningBehavior
from wow_bot.session import Session
from wow_bot.strategist.cooldown_v2 import CooldownConfig, CooldownGate
from wow_bot.strategist.orchestrator_v2 import (
    OrchestratorConfig,
    OrchestratorOutcome,
    OrchestratorV2,
)
from wow_bot.strategist.vocab_v2 import VocabConfig, VocabularyGuard
from wow_bot.world.summary import WorldSummary


@dataclass
class DummyTarget:
    name: str = "Defias Bandit"
    x: float = 15.0
    y: float = 25.0


@dataclass
class DummyEntity:
    entity_id: str = "mob_1"
    kind: str = "mob"
    x: float = 12.0
    y: float = 34.0
    z: float = 0.0
    distance: float = 10.0
    threat: float = 50.0
    hp_percent: float = 100.0
    is_attackable: bool = True
    is_alive: bool = True
    is_in_combat_with_self: bool = False


class MockGameState(GameStateLike):
    def __init__(
        self,
        *,
        target: DummyTarget | None = None,
        entities: tuple[DummyEntity, ...] = (),
        player_x: float = 0.0,
        player_y: float = 0.0,
        player_z: float = 0.0,
        player_heading: float = 0.0,
        self_hp_percent: float = 100.0,
        resource: float = 100.0,
        resource_max: float = 100.0,
        inventory_count: int = 5,
        level_or_xp: float = 12.0,
        fsm_state: FSMState = FSMState.IDLE,
    ) -> None:
        self.target = target
        self.entities = entities
        self.current_target_id = target.name if target is not None else None
        self.target_entity_id = target.name if target is not None else None
        self.target_hp_percent = 100.0 if target is not None else None
        self.player_x = player_x
        self.player_y = player_y
        self.player_z = player_z
        self.player_heading = player_heading
        self.self_x = player_x
        self.self_y = player_y
        self.self_hp_percent = self_hp_percent
        self.resource = resource
        self.resource_max = resource_max
        self.inventory_count = inventory_count
        self.level_or_xp = level_or_xp
        self.fsm_state = fsm_state


class MockMetaState(MetaStateLike):
    def __init__(self) -> None:
        self.drives: dict[str, float] = {"hunger": 0.1}
        self.memory_summary: str = "ready to scan"


class FakeLlm:
    def __init__(self, response: str = '{"goal": "explore", "target": null, "rationale": "exploring"}') -> None:
        self.response = response
        self.prompts_received: list[str] = []

    def complete(self, prompt: str) -> str:
        self.prompts_received.append(prompt)
        return self.response


def make_test_session(tmp_path: Path) -> Session:
    config = Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=tmp_path,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def _read_session_events(session: Session) -> list[dict[str, Any]]:
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    result = []
    with open(events_file, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                result.append(json.loads(line))
    return result


def make_dummy_world() -> WorldSummary:
    return WorldSummary(
        around_xy=(0.0, 0.0),
        radius=50.0,
        generated_at="2026-09-21T12:00:00Z",
        nearest_vendors=(),
        nearest_trainers=(),
        nearest_nodes=(),
        nearest_mobs=(),
        nearest_waypoints=(),
        recent_combats=(),
        total_nodes=0,
    )


def test_fsm_tick_progresses_idle_to_scanning_and_emits_intent(tmp_path: Path) -> None:
    """Acceptance criterion 1: FSM progresses IDLE -> SCANNING and reaches a subsequent state emitting intent."""
    session = make_test_session(tmp_path)

    scanning_behavior = _ScanningBehavior()
    fsm = FSM(
        config=FSMConfig(seed=42),
        behavior=scanning_behavior,
        session=session,
    )

    assert fsm.current_state.value == FSMState.IDLE.value

    # Entity observed in state
    state = MockGameState(entities=(DummyEntity(entity_id="wolf", x=10.0, y=20.0),))
    meta = MockMetaState()

    # Tick advances IDLE -> SCANNING, behavior selects entity and returns MoveTo(10, 20),
    # FSM advances SCANNING -> MOVING_TO_TARGET and emits intent
    intent = fsm.tick(state, meta, now=100.0)

    assert intent is not None
    assert isinstance(intent, MoveTo)
    assert intent.x == 10.0
    assert intent.y == 20.0

    assert fsm.previous_state == FSMState.SCANNING
    assert fsm.current_state.value == FSMState.MOVING_TO_TARGET.value

    events = _read_session_events(session)
    transitions = [e for e in events if e.get("event") == "fsm_transition"]
    assert len(transitions) == 2
    assert transitions[0]["from"] == FSMState.IDLE.value
    assert transitions[0]["to"] == FSMState.SCANNING.value
    assert transitions[1]["from"] == FSMState.SCANNING.value
    assert transitions[1]["to"] == FSMState.MOVING_TO_TARGET.value

    intents = [e for e in events if e.get("event") == "fsm_intent"]
    assert len(intents) == 1
    assert intents[0]["state"] == FSMState.MOVING_TO_TARGET.value


def test_fsm_tick_scanning_without_entities_emits_turn_sweep_and_reaches_targeting(tmp_path: Path) -> None:
    """When no entities are visible, scanning emits a turn intent and transitions to TARGETING."""
    session = make_test_session(tmp_path)

    fsm = FSM(
        config=FSMConfig(seed=99),
        behavior=_ScanningBehavior(),
        session=session,
    )
    state = MockGameState(entities=())
    meta = MockMetaState()

    intent = fsm.tick(state, meta, now=50.0)

    assert intent is not None
    assert isinstance(intent, Turn)
    assert fsm.current_state.value == FSMState.TARGETING.value
    assert fsm.previous_state == FSMState.SCANNING

    events = _read_session_events(session)
    transitions = [e for e in events if e.get("event") == "fsm_transition"]
    assert transitions[0]["from"] == FSMState.IDLE.value
    assert transitions[0]["to"] == FSMState.SCANNING.value
    assert transitions[1]["from"] == FSMState.SCANNING.value
    assert transitions[1]["to"] == FSMState.TARGETING.value


def test_fsm_illegal_transition_still_raises() -> None:
    """Acceptance criterion 2: illegal transitions defined in transition table raise TransitionError."""
    fsm = FSM()
    assert fsm.current_state.value == FSMState.IDLE.value

    # IDLE only allows SCANNING and PAUSED
    with pytest.raises(TransitionError, match="illegal transition: IDLE -> COMBAT"):
        fsm.transition_to(FSMState.COMBAT, now=10.0)

    with pytest.raises(TransitionError, match="illegal transition: IDLE -> LOOTING"):
        fsm.transition_to(FSMState.LOOTING, now=10.0)

    with pytest.raises(TransitionError, match="illegal transition: IDLE -> WAITING_GCD"):
        fsm.transition_to(FSMState.WAITING_GCD, now=10.0)

    # Legally transition to SCANNING
    fsm.transition_to(FSMState.SCANNING, now=15.0)
    assert fsm.current_state.value == FSMState.SCANNING.value

    # SCANNING does not allow direct COMBAT (must go through TARGETING or MOVING_TO_TARGET)
    with pytest.raises(TransitionError, match="illegal transition: SCANNING -> COMBAT"):
        fsm.transition_to(FSMState.COMBAT, now=20.0)

    with pytest.raises(TransitionError, match="illegal transition: SCANNING -> LOOTING"):
        fsm.transition_to(FSMState.LOOTING, now=20.0)


def test_strategist_fsm_state_tracks_runtime_fsm_state() -> None:
    """Acceptance criterion 3: value the strategist sees for fsm_state tracks runtime FSM state."""
    llm = FakeLlm()
    orchestrator = OrchestratorV2(
        llm=llm,
        cooldown=CooldownGate(config=CooldownConfig(min_interval_s=0.0)),
        guard=VocabularyGuard(config=VocabConfig()),
        config=OrchestratorConfig(),
    )

    state = MockGameState()
    meta = MockMetaState()
    world = make_dummy_world()

    # 1. Evaluate with SCANNING (preferred state: ALLOWED)
    res1 = orchestrator.decide(
        meta=meta,  # type: ignore[arg-type]
        world=world,
        state=state,
        now=100.0,
        fsm_state=FSMState.SCANNING,
    )
    assert res1.outcome == OrchestratorOutcome.SUCCESS
    assert "fsm_state: SCANNING" in llm.prompts_received[0]

    # 2. Evaluate with COMBAT (blocked state: BLOCKED_BY_COOLDOWN)
    res2 = orchestrator.decide(
        meta=meta,  # type: ignore[arg-type]
        world=world,
        state=state,
        now=200.0,
        fsm_state=FSMState.COMBAT,
    )
    assert res2.outcome == OrchestratorOutcome.BLOCKED_BY_COOLDOWN
    assert res2.reason == "blocked_state"

    # 3. With allow_while_busy=True, evaluate MOVING_TO_TARGET -> ALLOWED, prompt contains fsm_state
    orchestrator_busy = OrchestratorV2(
        llm=llm,
        cooldown=CooldownGate(config=CooldownConfig(min_interval_s=0.0, allow_while_busy=True)),
        guard=VocabularyGuard(config=VocabConfig()),
        config=OrchestratorConfig(),
    )
    res3 = orchestrator_busy.decide(
        meta=meta,  # type: ignore[arg-type]
        world=world,
        state=state,
        now=300.0,
        fsm_state=FSMState.MOVING_TO_TARGET,
    )
    assert res3.outcome == OrchestratorOutcome.SUCCESS
    assert "fsm_state: MOVING_TO_TARGET" in llm.prompts_received[1]


def test_fsm_determinism_identical_seed() -> None:
    """Determinism check: identical RNG seeds produce identical intents and transitions."""
    fsm1 = FSM(config=FSMConfig(seed=12345), behavior=_ScanningBehavior())
    fsm2 = FSM(config=FSMConfig(seed=12345), behavior=_ScanningBehavior())

    state = MockGameState(entities=())
    meta = MockMetaState()

    intent1 = fsm1.tick(state, meta, now=10.0)
    intent2 = fsm2.tick(state, meta, now=10.0)

    assert intent1 == intent2
    assert fsm1.current_state == fsm2.current_state


@pytest.mark.asyncio
async def test_runner_cycle_advances_fsm_and_emits_intent(tmp_path: Path) -> None:
    """Full cycle test: run_lab_loop_async advances FSM from IDLE to subsequent state and executes intents."""
    from wow_bot.actuation.backends.focus_null import NullFocusBackend
    from wow_bot.combat.rotation import RotationConfig
    from wow_bot.farm.profile import (
        CycleSpec,
        FarmProfile,
        NodeReference,
        RoutePreferences,
        VendorReference,
    )
    from wow_bot.lab.runner_v2 import build_lab_runtime_async, run_lab_loop_async

    sess_root = tmp_path / "runs"
    sess_root.mkdir(parents=True, exist_ok=True)
    config = Config(
        lab_mode=True,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="F12",
        session_root=sess_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(config)
    profile = FarmProfile(
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
    llm = FakeLlm('{"goal": "farm", "target": null, "rationale": "start"}')
    state = MockGameState(
        entities=(DummyEntity(entity_id="target_dummy", x=15.0, y=25.0),),
    )

    runtime = await build_lab_runtime_async(
        config=config,
        session=session,
        game_state_source=lambda: state,
        meta_state_source=lambda: MockMetaState(),
        rotation_config=RotationConfig(rules=()),
        farm_profile=profile,
        world_db_path=str(tmp_path / "world.db"),
        sleep=lambda _: None,
        include_watchdog=False,
        focus_backend=NullFocusBackend(),
        llm_client=llm,
    )

    assert runtime.fsm.current_state.value == FSMState.IDLE.value

    # Run 1 cycle
    res = await run_lab_loop_async(runtime, max_cycles=1)
    assert res.cycles_completed == 1

    # FSM should have progressed past IDLE to MOVING_TO_TARGET (since entity was present)
    assert runtime.fsm.current_state.value == FSMState.MOVING_TO_TARGET.value

    await runtime.world.close()

