"""Unit tests for ExecutorFSM (Task 5.3).

Validates state transitions, flee thresholds, stuck detection, strategy updates,
controller boundaries, timestamp rules, and determinism (Test cases A - AJ).
"""

from __future__ import annotations

from typing import Any, cast

import pytest
from loguru import logger

from wow_bot.executor.controller import Controller
from wow_bot.executor.fsm import (
    ExecutorFSM,
    State,
)
from wow_bot.shared.config import ExecutorConfig
from wow_bot.shared.interfaces import Event, GameState, Strategy, TargetInfo


def make_game_state(
    *,
    timestamp: float = 100.0,
    hp_pct: float = 1.0,
    mana_pct: float = 1.0,
    position: tuple[float, float] = (0.0, 0.0),
    facing: float = 0.0,
    in_combat: bool = False,
    target: TargetInfo | None = None,
    events: list[Event] | None = None,
) -> GameState:
    """Factory helper for generating valid GameState test instances."""
    return GameState(
        timestamp=timestamp,
        hp_pct=hp_pct,
        mana_pct=mana_pct,
        position=position,
        facing=facing,
        in_combat=in_combat,
        target=target,
        enemies=[],
        events=events if events is not None else [],
    )


def make_strategy(
    *,
    goal: str = "farm_herbs",
    region: str = "elwynn_forest",
    risk_tolerance: float = 0.5,
    valid_until: float = 1000.0,
) -> Strategy:
    """Factory helper for generating valid Strategy test instances."""
    return Strategy(
        goal=goal,
        region=region,
        risk_tolerance=risk_tolerance,
        priority=["herbs"],
        constraints={},
        valid_until=valid_until,
    )


def make_target(name: str = "MockBoar", hp_pct: float = 1.0) -> TargetInfo:
    """Factory helper for generating valid TargetInfo test instances."""
    return TargetInfo(
        name=name,
        hp_pct=hp_pct,
        reaction="hostile",
        distance_estimate=15.0,
    )


@pytest.mark.asyncio
async def test_a_initial_state() -> None:
    """Test A: Newly created FSM starts in IDLE without issuing controller commands."""
    ctrl = Controller(dry_run=True)
    strat = make_strategy()
    cfg = ExecutorConfig()
    fsm = ExecutorFSM(cfg, ctrl, strat)

    assert fsm.state == State.IDLE
    assert len(ctrl.commands) == 0


@pytest.mark.asyncio
async def test_b_idle_to_scanning() -> None:
    """Test B: IDLE transitions to SCANNING when no target and no combat."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs = make_game_state(timestamp=100.0, in_combat=False, target=None)
    await fsm.tick(gs)

    assert fsm.state == State.SCANNING
    assert len(ctrl.commands) == 0


@pytest.mark.asyncio
async def test_c_idle_to_moving_to_target() -> None:
    """Test C: IDLE transitions to MOVING_TO_TARGET when target present and not in combat."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs = make_game_state(timestamp=100.0, in_combat=False, target=make_target())
    await fsm.tick(gs)

    assert fsm.state == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_d_idle_to_combat() -> None:
    """Test D: IDLE transitions to COMBAT when in combat and HP is safe."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    gs = make_game_state(timestamp=100.0, hp_pct=0.80, in_combat=True)
    await fsm.tick(gs)

    assert fsm.state == State.COMBAT


@pytest.mark.asyncio
async def test_e_idle_directly_to_fleeing() -> None:
    """Test E: IDLE transitions directly to FLEEING when in combat and HP <= threshold."""
    ctrl = Controller(dry_run=True)
    # risk_tolerance = 0.5 -> flee_threshold = 0.40
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    gs = make_game_state(timestamp=100.0, hp_pct=0.35, in_combat=True)
    await fsm.tick(gs)

    assert fsm.state == State.FLEEING
    assert len(ctrl.commands) == 1
    assert ctrl.commands[0].action == "stop_all"


@pytest.mark.asyncio
async def test_f_scanning_stays_scanning() -> None:
    """Test F: Repeated ticks in SCANNING generate no additional controller commands."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs1 = make_game_state(timestamp=100.0)
    await fsm.tick(gs1)
    assert fsm.state == State.SCANNING

    gs2 = make_game_state(timestamp=101.0)
    await fsm.tick(gs2)
    assert fsm.state == State.SCANNING
    assert len(ctrl.commands) == 0


@pytest.mark.asyncio
async def test_g_scanning_to_target_movement() -> None:
    """Test G: SCANNING transitions to MOVING_TO_TARGET when target appears."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0))
    s1: State = fsm.state
    assert s1 == State.SCANNING

    await fsm.tick(make_game_state(timestamp=101.0, target=make_target()))
    s2: State = fsm.state
    assert s2 == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_h_movement_to_combat() -> None:
    """Test H: MOVING_TO_TARGET transitions to COMBAT when combat becomes active."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    await fsm.tick(make_game_state(timestamp=100.0, target=make_target()))
    s1: State = fsm.state
    assert s1 == State.MOVING_TO_TARGET

    await fsm.tick(
        make_game_state(timestamp=101.0, target=make_target(), hp_pct=0.80, in_combat=True)
    )
    s2: State = fsm.state
    assert s2 == State.COMBAT


@pytest.mark.asyncio
async def test_i_target_disappears() -> None:
    """Test I: MOVING_TO_TARGET transitions to SCANNING when target disappears."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0, target=make_target()))
    s1: State = fsm.state
    assert s1 == State.MOVING_TO_TARGET

    await fsm.tick(make_game_state(timestamp=101.0, target=None))
    s2: State = fsm.state
    assert s2 == State.SCANNING


@pytest.mark.asyncio
async def test_j_combat_ends() -> None:
    """Test J: COMBAT transitions to LOOTING when in_combat becomes False."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    await fsm.tick(make_game_state(timestamp=100.0, hp_pct=0.80, in_combat=True))
    s1: State = fsm.state
    assert s1 == State.COMBAT

    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.80, in_combat=False))
    s2: State = fsm.state
    assert s2 == State.LOOTING


@pytest.mark.asyncio
async def test_k_looting_is_transient() -> None:
    """Test K: LOOTING transitions to SCANNING on next normal non-combat tick."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0, hp_pct=0.80, in_combat=True))
    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.80, in_combat=False))
    s1: State = fsm.state
    assert s1 == State.LOOTING

    await fsm.tick(make_game_state(timestamp=102.0, hp_pct=0.80, in_combat=False))
    s2: State = fsm.state
    assert s2 == State.SCANNING
    assert len(ctrl.commands) == 0


@pytest.mark.asyncio
async def test_l_combat_interrupts_looting() -> None:
    """Test L: LOOTING transitions to COMBAT if combat starts during looting."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    await fsm.tick(make_game_state(timestamp=100.0, hp_pct=0.80, in_combat=True))
    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.80, in_combat=False))
    s1: State = fsm.state
    assert s1 == State.LOOTING

    await fsm.tick(make_game_state(timestamp=102.0, hp_pct=0.80, in_combat=True))
    s2: State = fsm.state
    assert s2 == State.COMBAT


@pytest.mark.asyncio
async def test_m_low_risk_flees_earlier() -> None:
    """Test M: Low risk tolerance flees at higher HP threshold (e.g. risk=0.1 -> threshold=0.56)."""
    ctrl = Controller(dry_run=True)
    # threshold = 0.60 - 0.40 * 0.1 = 0.56
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.1))

    gs = make_game_state(timestamp=100.0, hp_pct=0.40, in_combat=True)
    await fsm.tick(gs)

    assert fsm.state == State.FLEEING


@pytest.mark.asyncio
async def test_n_high_risk_flees_later() -> None:
    """Test N: High risk tolerance flees at lower HP threshold (e.g. risk=0.9 -> threshold=0.24)."""
    ctrl = Controller(dry_run=True)
    # threshold = 0.60 - 0.40 * 0.9 = 0.24
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.9))

    gs = make_game_state(timestamp=100.0, hp_pct=0.40, in_combat=True)
    await fsm.tick(gs)

    assert fsm.state == State.COMBAT


def test_o_risk_extremes() -> None:
    """Test O: Verify flee threshold formula for risk extremes (0.0 -> 0.60, 1.0 -> 0.20)."""
    ctrl = Controller(dry_run=True)

    fsm_low = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.0))
    assert pytest.approx(fsm_low._flee_hp_threshold()) == 0.60

    fsm_mid = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))
    assert pytest.approx(fsm_mid._flee_hp_threshold()) == 0.40

    fsm_high = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=1.0))
    assert pytest.approx(fsm_high._flee_hp_threshold()) == 0.20


@pytest.mark.asyncio
async def test_p_exact_flee_boundary() -> None:
    """Test P: FSM flees when hp_pct equals exact threshold (<= rule)."""
    ctrl = Controller(dry_run=True)
    # risk = 0.5 -> threshold = 0.40
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    gs = make_game_state(timestamp=100.0, hp_pct=0.40, in_combat=True)
    await fsm.tick(gs)

    assert fsm.state == State.FLEEING


@pytest.mark.asyncio
async def test_q_low_hp_outside_combat() -> None:
    """Test Q: Low HP outside combat does NOT trigger FLEEING."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.1))

    gs = make_game_state(timestamp=100.0, hp_pct=0.10, in_combat=False, target=None)
    await fsm.tick(gs)

    assert fsm.state == State.SCANNING


@pytest.mark.asyncio
async def test_r_fleeing_persists_during_combat() -> None:
    """Test R: FLEEING state persists while in combat without repeatedly calling stop_all()."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    await fsm.tick(make_game_state(timestamp=100.0, hp_pct=0.30, in_combat=True))
    assert fsm.state == State.FLEEING
    assert len(ctrl.commands) == 1

    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.30, in_combat=True))
    assert fsm.state == State.FLEEING
    assert len(ctrl.commands) == 1  # No extra stop_all command


@pytest.mark.asyncio
async def test_s_fleeing_exits_when_safe() -> None:
    """Test S: FLEEING exits to SCANNING when out of combat and hp_pct > threshold."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    await fsm.tick(make_game_state(timestamp=100.0, hp_pct=0.30, in_combat=True))
    s1: State = fsm.state
    assert s1 == State.FLEEING

    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.50, in_combat=False))
    s2: State = fsm.state
    assert s2 == State.SCANNING


@pytest.mark.asyncio
async def test_t_stuck_timer_begins() -> None:
    """Test T: Movement intent starts stuck timer at timestamp 100.0 without immediate recovery."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs = make_game_state(timestamp=100.0, target=make_target("BoarA"))
    await fsm.tick(gs)

    assert fsm.state == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_u_not_stuck_before_five_seconds() -> None:
    """Test U: Movement for 4.9 seconds remains in MOVING_TO_TARGET."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=104.9, target=target))

    assert fsm.state == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_v_stuck_at_exact_five_seconds() -> None:
    """Test V: Repeated movement for exactly 5.0 simulated seconds triggers STUCK_RECOVERY."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=105.0, target=target))

    assert fsm.state == State.STUCK_RECOVERY
    assert len(ctrl.commands) == 1
    assert ctrl.commands[0].action == "stop_all"


@pytest.mark.asyncio
async def test_w_changed_target_resets_stuck_timer() -> None:
    """Test W: Switching targets resets stuck timer and prevents premature recovery."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0, target=make_target("BoarA")))
    await fsm.tick(make_game_state(timestamp=104.0, target=make_target("BoarB")))
    await fsm.tick(make_game_state(timestamp=105.0, target=make_target("BoarB")))

    assert fsm.state == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_x_leaving_movement_clears_stuck_timer() -> None:
    """Test X: Leaving movement state clears stuck timer so future movement gets a fresh 5s window."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=104.0, target=target))

    # Transition to combat briefly
    await fsm.tick(make_game_state(timestamp=104.5, hp_pct=0.90, in_combat=True))
    s1: State = fsm.state
    assert s1 == State.COMBAT

    # Combat ends -> LOOTING
    await fsm.tick(make_game_state(timestamp=105.0, target=target, in_combat=False))
    s2: State = fsm.state
    assert s2 == State.LOOTING

    # Next tick completes LOOTING -> SCANNING
    await fsm.tick(make_game_state(timestamp=105.5, target=target, in_combat=False))
    s3: State = fsm.state
    assert s3 == State.SCANNING

    # Next tick with target present -> MOVING_TO_TARGET
    await fsm.tick(make_game_state(timestamp=106.0, target=target, in_combat=False))
    s4: State = fsm.state
    assert s4 == State.MOVING_TO_TARGET

    # At 109.0 (3s into new movement), should not be stuck
    await fsm.tick(make_game_state(timestamp=109.0, target=target))
    s5: State = fsm.state
    assert s5 == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_y_recovery_exits_cleanly() -> None:
    """Test Y: STUCK_RECOVERY transitions cleanly to SCANNING on next normal safe tick."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=105.0, target=target))
    s1: State = fsm.state
    assert s1 == State.STUCK_RECOVERY

    await fsm.tick(make_game_state(timestamp=106.0, target=None))
    s2: State = fsm.state
    assert s2 == State.SCANNING


@pytest.mark.asyncio
async def test_z_recovery_can_flee() -> None:
    """Test Z: Next tick after STUCK_RECOVERY transitions to FLEEING if low HP in combat."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=105.0, target=target))
    s1: State = fsm.state
    assert s1 == State.STUCK_RECOVERY

    await fsm.tick(make_game_state(timestamp=106.0, hp_pct=0.30, in_combat=True))
    s2: State = fsm.state
    assert s2 == State.FLEEING


@pytest.mark.asyncio
async def test_aa_recovery_does_not_immediately_retrigger() -> None:
    """Test AA: After recovery exits to SCANNING and movement resumes, a fresh 5s timer applies."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    target = make_target("BoarA")
    await fsm.tick(make_game_state(timestamp=100.0, target=target))
    await fsm.tick(make_game_state(timestamp=105.0, target=target))
    s1: State = fsm.state
    assert s1 == State.STUCK_RECOVERY

    # Exits recovery to SCANNING
    await fsm.tick(make_game_state(timestamp=106.0, target=None))
    s2: State = fsm.state
    assert s2 == State.SCANNING

    # Resumes movement
    await fsm.tick(make_game_state(timestamp=107.0, target=target))
    s3: State = fsm.state
    assert s3 == State.MOVING_TO_TARGET

    # At 110.0 (3s into new movement), should not be stuck
    await fsm.tick(make_game_state(timestamp=110.0, target=target))
    s4: State = fsm.state
    assert s4 == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_ab_set_strategy_updates_flee_behavior() -> None:
    """Test AB: set_strategy immediately updates flee threshold for next tick."""
    ctrl = Controller(dry_run=True)
    # Start high risk (risk=0.9 -> flee_threshold = 0.24)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.9))

    gs = make_game_state(timestamp=100.0, hp_pct=0.40, in_combat=True)
    await fsm.tick(gs)
    s1: State = fsm.state
    assert s1 == State.COMBAT

    # Update strategy to low risk (risk=0.1 -> flee_threshold = 0.56)
    fsm.set_strategy(make_strategy(risk_tolerance=0.1))

    # Next tick at 40% HP triggers FLEEING
    await fsm.tick(make_game_state(timestamp=101.0, hp_pct=0.40, in_combat=True))
    s2: State = fsm.state
    assert s2 == State.FLEEING


@pytest.mark.asyncio
async def test_ac_set_strategy_preserves_state() -> None:
    """Test AC: Calling set_strategy preserves current FSM state."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0, target=make_target()))
    assert fsm.state == State.MOVING_TO_TARGET

    fsm.set_strategy(make_strategy(risk_tolerance=0.2))
    assert fsm.state == State.MOVING_TO_TARGET


@pytest.mark.asyncio
async def test_ad_deterministic_replay() -> None:
    """Test AD: Two identical FSM instances processing identical GameState sequence yield identical states."""
    ctrl1 = Controller(dry_run=True)
    ctrl2 = Controller(dry_run=True)
    strat = make_strategy(risk_tolerance=0.3)

    fsm1 = ExecutorFSM(None, ctrl1, strat)
    fsm2 = ExecutorFSM(None, ctrl2, strat)

    seq = [
        make_game_state(timestamp=100.0, target=None, in_combat=False),
        make_game_state(timestamp=101.0, target=make_target("A"), in_combat=False),
        make_game_state(timestamp=102.0, target=make_target("A"), in_combat=True, hp_pct=0.80),
        make_game_state(timestamp=103.0, target=make_target("A"), in_combat=True, hp_pct=0.30),
        make_game_state(timestamp=104.0, target=None, in_combat=False, hp_pct=0.70),
    ]

    states1: list[State] = []
    states2: list[State] = []

    for gs in seq:
        await fsm1.tick(gs)
        await fsm2.tick(gs)
        states1.append(fsm1.state)
        states2.append(fsm2.state)

    assert states1 == states2
    assert ctrl1.commands == ctrl2.commands


@pytest.mark.asyncio
async def test_ae_decreasing_timestamp_rejected() -> None:
    """Test AE: Decreasing timestamp raises ValueError."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0))
    with pytest.raises(ValueError, match="Decreasing timestamp"):
        await fsm.tick(make_game_state(timestamp=99.0))


@pytest.mark.asyncio
async def test_af_same_timestamp_allowed() -> None:
    """Test AF: Ticks sharing the same timestamp are accepted."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0))
    await fsm.tick(make_game_state(timestamp=100.0))

    assert fsm.state == State.SCANNING


@pytest.mark.asyncio
async def test_boolean_timestamp_rejected() -> None:
    """Test rejecting boolean timestamps explicitly."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs_bool = GameState(
        timestamp=cast(Any, True),
        hp_pct=1.0,
        mana_pct=1.0,
        position=(0.0, 0.0),
        facing=0.0,
        in_combat=False,
        target=None,
        enemies=[],
        events=[],
    )
    with pytest.raises(ValueError, match="Timestamp must be a finite numeric value"):
        await fsm.tick(gs_bool)


@pytest.mark.asyncio
async def test_ag_transition_logging(caplog: pytest.LogCaptureFixture) -> None:
    """Test AG: Actual state transition produces log message."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    handler_id = logger.add(caplog.handler, format="{message}")
    try:
        with caplog.at_level("INFO"):
            await fsm.tick(make_game_state(timestamp=100.0))
    finally:
        logger.remove(handler_id)

    assert "FSM transition: IDLE -> SCANNING" in caplog.text


@pytest.mark.asyncio
async def test_ah_self_transition_not_logged(caplog: pytest.LogCaptureFixture) -> None:
    """Test AH: Self-transitions produce no transition log records."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0))

    handler_id = logger.add(caplog.handler, format="{message}")
    caplog.clear()
    try:
        with caplog.at_level("INFO"):
            await fsm.tick(make_game_state(timestamp=101.0))
    finally:
        logger.remove(handler_id)

    assert "FSM transition" not in caplog.text


@pytest.mark.asyncio
async def test_ai_controller_boundary() -> None:
    """Test AI: Across normal ticks, press_key/move_mouse/click are never invoked."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    await fsm.tick(make_game_state(timestamp=100.0))
    await fsm.tick(make_game_state(timestamp=101.0, target=make_target()))
    await fsm.tick(make_game_state(timestamp=102.0, hp_pct=0.80, in_combat=True))
    await fsm.tick(make_game_state(timestamp=103.0, hp_pct=0.80, in_combat=False))

    actions = [cmd.action for cmd in ctrl.commands]
    assert "press_key" not in actions
    assert "move_mouse" not in actions
    assert "click" not in actions


@pytest.mark.asyncio
async def test_aj_no_timing_helper_randomness() -> None:
    """Test AJ: FSM transition behavior is fully deterministic without RNG or sleeps."""
    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    for i in range(10):
        await fsm.tick(make_game_state(timestamp=100.0 + i))

    assert fsm.state == State.SCANNING
    assert len(ctrl.commands) == 0
