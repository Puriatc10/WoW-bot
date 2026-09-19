"""Scenario-driven integration tests for ExecutorFSM with MockPerception (Task 5.3).

Validates FSM state transitions driven by synthetic GameState streams from
MockPerception scenarios (combat_light, stuck_repeatedly) using deterministic seeds
and simulated timestamps.
"""

from __future__ import annotations

import numpy as np
import pytest

from wow_bot.executor.controller import Controller
from wow_bot.executor.fsm import ExecutorFSM, State
from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.shared.interfaces import GameState, Strategy, TargetInfo


def make_strategy(risk_tolerance: float = 0.5) -> Strategy:
    """Helper factory for strategy in integration tests."""
    return Strategy(
        goal="farm_herbs",
        region="elwynn_forest",
        risk_tolerance=risk_tolerance,
        priority=["herbs"],
        constraints={},
        valid_until=10000.0,
    )


@pytest.mark.asyncio
async def test_fsm_combat_light_scenario() -> None:
    """Integration Test: FSM transitions through SCANNING/MOVING -> COMBAT -> LOOTING in combat_light scenario."""
    rng = np.random.default_rng(seed=42)
    current_sim_time = 1000.0

    def mock_clock() -> float:
        return current_sim_time

    mock_p = MockPerception(
        rng=rng,
        scenario="combat_light",
        monotonic_clock=mock_clock,
        timestamp_clock=mock_clock,
    )

    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy(risk_tolerance=0.5))

    observed_states: set[State] = {fsm.state}

    # Step through 100 frames, advancing clock by 0.5s per frame
    for _ in range(100):
        gs = await mock_p.get_state()
        await fsm.tick(gs)
        observed_states.add(fsm.state)
        current_sim_time += 0.5

    # Confirm FSM observed COMBAT, LOOTING, and SCANNING
    assert State.COMBAT in observed_states
    assert State.LOOTING in observed_states
    assert State.SCANNING in observed_states


@pytest.mark.asyncio
async def test_fsm_stuck_detection_with_mock_target() -> None:
    """Integration Test: Continuous movement toward stable mock target triggers STUCK_RECOVERY."""
    rng = np.random.default_rng(seed=123)
    current_sim_time = 2000.0

    def mock_clock() -> float:
        return current_sim_time

    mock_p = MockPerception(
        rng=rng,
        scenario="combat_light",
        monotonic_clock=mock_clock,
        timestamp_clock=mock_clock,
    )

    ctrl = Controller(dry_run=True)
    fsm = ExecutorFSM(None, ctrl, make_strategy())

    gs = await mock_p.get_state()

    stable_target = TargetInfo(
        name="TargetDummy",
        hp_pct=1.0,
        reaction="hostile",
        distance_estimate=20.0,
    )

    states_history: list[State] = []

    # Advance 12 steps (6 seconds total: 0.5s increment)
    for i in range(12):
        frame_time = current_sim_time + (i * 0.5)
        tick_gs = GameState(
            timestamp=frame_time,
            hp_pct=1.0,
            mana_pct=1.0,
            position=gs.position,
            facing=gs.facing,
            in_combat=False,
            target=stable_target,
            enemies=[],
            events=[],
        )
        await fsm.tick(tick_gs)
        states_history.append(fsm.state)

    # First ticks should be MOVING_TO_TARGET
    assert states_history[0] == State.MOVING_TO_TARGET
    # At t >= 5.0s (index >= 10, offset 5.0s), STUCK_RECOVERY must trigger
    assert State.STUCK_RECOVERY in states_history
