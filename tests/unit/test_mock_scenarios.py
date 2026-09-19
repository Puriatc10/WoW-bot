"""Acceptance tests for Task 2.2 — Scenario-driven Mock Perception.

Roadmap acceptance criteria:
- Test runs each scenario for at least 10 frames.
- All relevant event types appear in respective scenarios.

Additional invariant checks derived from the task requirements:
- Scenario selection works via constructor argument.
- Reproducibility: identical seeds produce identical frame sequences per
  scenario (AGENTS.md testing philosophy rule 1).
- Unknown scenario names fail loudly instead of silently falling back.
- Per-scenario semantics hold: peaceful_farm never enters combat;
  combat_light spawns exactly one enemy; death_loop emits deaths every 120 s;
  rare_loot_drought never emits ``rare_loot``;
  stuck_repeatedly emits ``stuck`` events.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass

import numpy as np
import pytest

from wow_bot.mocks.mock_perception import (
    SCENARIO_NAMES,
    SCENARIO_PROFILES,
    MockPerception,
)
from wow_bot.shared.events import DEATH, RARE_LOOT, STUCK
from wow_bot.shared.interfaces import GameState

# Number of frames per scenario run. Comfortably above the roadmap's "at least
# 10" floor while keeping the whole suite well under the <1 s unit budget.
FRAMES = 60


@dataclass
class MockClock:
    """Caller-controlled elapsed time and epoch time for complete replay."""

    elapsed: float = 0.0

    def monotonic(self) -> float:
        return self.elapsed

    def timestamp(self) -> float:
        return 1_000_000.0 + self.elapsed


def _event_types(states: list[GameState]) -> set[str]:
    return {event.type for state in states for event in state.events}


async def _run(
    scenario: str,
    *,
    seed: int = 42,
    frames: int = FRAMES,
    frame_step: float = 1.0,
    event_probability: float | None = None,
) -> list[GameState]:
    """Drive the uncompressed scenario schedule without sleeping."""
    clock = MockClock()
    p = MockPerception(
        rng=np.random.default_rng(seed), scenario=scenario,
        monotonic_clock=clock.monotonic, timestamp_clock=clock.timestamp,
        event_probability=event_probability,
    )
    states: list[GameState] = []
    for _ in range(frames):
        states.append(await p.get_state())
        clock.elapsed += frame_step
    return states


# ---------------------------------------------------------------------------
# Acceptance criterion 1: every scenario runs for >= 10 frames without error
# and keeps emitting contract-valid frames.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
async def test_each_scenario_runs_for_at_least_ten_frames(scenario: str) -> None:
    states = await _run(scenario, frames=10)
    assert len(states) == 10
    for state in states:
        assert 0.0 <= state.hp_pct <= 1.0
        assert 0.3 <= state.mana_pct <= 1.0
        assert isinstance(state.timestamp, float)
        assert math.isfinite(state.facing)
        assert 0.0 <= state.facing < 2.0 * math.pi


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
async def test_every_frame_satisfies_gamestate_contract(scenario: str) -> None:
    """Frames must survive the shared to_dict/from_dict round trip."""
    states = await _run(scenario)
    for state in states:
        restored = GameState.from_dict(state.to_dict())
        assert restored.hp_pct == pytest.approx(state.hp_pct)
        assert restored.in_combat == state.in_combat
        assert len(restored.events) == len(state.events)


@pytest.mark.parametrize("scenario,expected", [
    ("peaceful_farm", {"level_up", "quest_complete", "npc_interact"}),
    ("combat_light", {"pvp_hit", "pvp_kill", "level_up", "quest_complete"}),
    ("death_loop", {DEATH, RARE_LOOT, "pvp_hit", "pvp_kill", STUCK,
                    "level_up", "quest_complete", "npc_interact"}),
    ("rare_loot_drought", {DEATH, "pvp_hit", "pvp_kill", STUCK,
                           "level_up", "quest_complete", "npc_interact"}),
    ("stuck_repeatedly", {STUCK, "npc_interact", "level_up"}),
])
async def test_all_relevant_events_appear(scenario: str, expected: set[str]) -> None:
    states = await _run(scenario, frames=2000)
    assert _event_types(states) == expected


# ---------------------------------------------------------------------------
# Acceptance criterion 2: relevant event types appear in their scenarios.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
def test_scenario_declares_only_registered_event_types(scenario: str) -> None:
    from wow_bot.shared.events import ALL_EVENT_TYPES

    pool = SCENARIO_PROFILES[scenario].event_pool
    assert set(pool) <= set(ALL_EVENT_TYPES), f"{scenario} pool has unregistered types"


async def test_peaceful_farm_emits_benign_events_and_never_combat() -> None:
    states = await _run("peaceful_farm", frames=1000)
    seen = _event_types(states)
    assert seen, "peaceful_farm emitted no events at all"
    assert seen == {"level_up", "quest_complete", "npc_interact"}
    assert not any(s.in_combat for s in states)
    for state in states:
        assert state.target is None
        assert state.enemies == []


async def test_peaceful_farm_keeps_hp_stable() -> None:
    """HP must stay near its starting value — 'no combat, HP stable'."""
    states = await _run("peaceful_farm", frames=10000)
    hp_values = [s.hp_pct for s in states]
    spread = max(hp_values) - min(hp_values)
    assert spread < 0.25, f"HP drifted too much for a peaceful scenario: {spread:.3f}"
    assert all(hp == 0.85 for hp in hp_values)


async def test_combat_light_uses_single_enemy_and_pvp_events() -> None:
    """Default six-second combats alternate with peace and a single enemy."""
    states = await _run(
        "combat_light",
        seed=7,
        frames=1000,
    )
    seen = _event_types(states)
    assert {"pvp_hit", "pvp_kill"} <= seen, f"expected PvP events, got {seen}"
    runs = [(flag, len(list(group))) for flag, group in itertools.groupby(
        s.in_combat for s in states
    )]
    assert len(runs) > 3
    assert all(length == 6 for flag, length in runs[:-1] if flag)

    combat_frames = [s for s in states if s.in_combat]
    assert combat_frames, "combat_light never entered combat"
    for state in combat_frames:
        assert state.target is not None
        assert state.target.reaction == "hostile"
        assert len(state.enemies) == 1, "combat_light must spawn exactly one enemy"


async def test_death_loop_emits_deaths_every_two_minutes() -> None:
    """Use the default interval and leave ambient events enabled."""
    states = await _run(
        "death_loop",
        frames=601,
    )
    deaths = [e for s in states for e in s.events if e.type == DEATH]
    assert [d.timestamp - 1_000_000.0 for d in deaths] == [120, 240, 360, 480, 600]
    assert all(d.data.get("forced") is True for d in deaths)
    for index in (120, 240, 360, 480):
        dead = states[index]
        assert dead.hp_pct == 0.0
        assert not dead.in_combat and dead.target is None and dead.enemies == []
        assert states[index + 1].hp_pct >= 0.3


async def test_rare_loot_drought_never_emits_rare_loot() -> None:
    """Exercise the remaining event pool and check its explicit exclusion."""
    states = await _run(
        "rare_loot_drought",
        seed=13,
        frames=300,
        event_probability=1.0,
    )
    seen = _event_types(states)
    assert seen, "drought scenario emitted nothing to sample from"
    assert RARE_LOOT not in seen, "rare_loot leaked into the drought scenario"
    from wow_bot.shared.events import ALL_EVENT_TYPES

    assert seen == set(ALL_EVENT_TYPES) - {RARE_LOOT}


async def test_stuck_repeatedly_emits_stuck_events() -> None:
    states = await _run("stuck_repeatedly", seed=17, frames=721)
    seen = _event_types(states)
    assert STUCK in seen, f"expected stuck events, got {seen}"
    assert not any(s.in_combat for s in states)
    assert [e.timestamp - 1_000_000.0 for s in states for e in s.events if e.type == STUCK] == [
        180, 360, 540, 720,
    ]


# ---------------------------------------------------------------------------
# Scenario selection and reproducibility
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("scenario", SCENARIO_NAMES)
async def test_same_seed_same_scenario_is_reproducible(scenario: str) -> None:
    """Replay full frames, including entities, timestamps and scheduled events."""
    a = await _run(scenario, seed=999, frames=721)
    b = await _run(scenario, seed=999, frames=721)
    assert [s.to_dict() for s in a] == [s.to_dict() for s in b]


@pytest.mark.parametrize("scenario,event", [("death_loop", DEATH), ("stuck_repeatedly", STUCK)])
async def test_scheduled_events_require_elapsed_time(scenario: str, event: str) -> None:
    states = await _run(scenario, frames=1000, frame_step=0.0, event_probability=1.0)
    assert event not in _event_types(states)


async def test_different_scenarios_diverge_under_the_same_seed() -> None:
    a = await _run("peaceful_farm", seed=5, frames=20)
    b = await _run("combat_light", seed=5, frames=20)
    sig_a = [s.hp_pct for s in a]
    sig_b = [s.hp_pct for s in b]
    assert sig_a != sig_b


def test_unknown_scenario_raises_value_error() -> None:
    with pytest.raises(ValueError, match="unknown scenario"):
        MockPerception(rng=np.random.default_rng(1), scenario="not_a_real_scenario")


def test_exposed_scenario_names_match_profile_table() -> None:
    assert set(SCENARIO_NAMES) == set(SCENARIO_PROFILES)
    assert SCENARIO_NAMES == (
        "peaceful_farm",
        "combat_light",
        "death_loop",
        "rare_loot_drought",
        "stuck_repeatedly",
    )


def test_scenario_property_reflects_constructor_argument() -> None:
    p = MockPerception(rng=np.random.default_rng(2), scenario="combat_light")
    assert p.scenario == "combat_light"
    assert MockPerception(rng=np.random.default_rng(2)).scenario is None


# ---------------------------------------------------------------------------
# Backward compatibility with Task 2.1 (no scenario given)
# ---------------------------------------------------------------------------


async def test_default_construction_preserves_task_2_1_behaviour() -> None:
    """Without a scenario the mock behaves like the original Task 2.1 build:
    full event pool, ~30 s combat cycle, legacy sigmas."""
    p = MockPerception(rng=np.random.default_rng(33))
    states = [await p.get_state() for _ in range(30)]
    for state in states:
        assert 0.3 <= state.hp_pct <= 1.0
        assert 0.3 <= state.mana_pct <= 1.0
    # Legacy path still walks position smoothly.
    for prev, cur in itertools.pairwise([s.position for s in states]):
        assert math.hypot(cur[0] - prev[0], cur[1] - prev[1]) < 5.0


async def test_explicit_overrides_win_over_profile_defaults() -> None:
    """Constructor knobs passed alongside a scenario override its profile."""
    clock = MockClock()
    p = MockPerception(
        rng=np.random.default_rng(44),
        scenario="peaceful_farm",
        combat_on_duration=2.0,
        combat_cycle_period=2.0,
        max_enemies=2,
        monotonic_clock=clock.monotonic,
    )
    saw_combat = False
    for _ in range(80):
        state = await p.get_state()
        if state.in_combat:
            saw_combat = True
            assert 1 <= len(state.enemies) <= 2
        clock.elapsed += 1.0
    assert saw_combat, "override did not re-enable combat for peaceful_farm"


@pytest.mark.parametrize("scenario,event,interval", [
    ("death_loop", DEATH, 120.0), ("stuck_repeatedly", STUCK, 180.0),
])
async def test_delayed_polling_preserves_cadence(
    scenario: str, event: str, interval: float,
) -> None:
    clock = MockClock()
    p = MockPerception(
        scenario=scenario, rng=np.random.default_rng(42), event_probability=0.0,
        monotonic_clock=clock.monotonic, timestamp_clock=clock.timestamp,
    )
    clock.elapsed = interval - 0.01
    assert (await p.get_state()).events == []
    clock.elapsed = interval * 3 + 1
    state = await p.get_state()
    assert [e.type for e in state.events] == [event]
    assert state.events[0].timestamp == state.timestamp
    assert (await p.get_state()).events == []
    clock.elapsed = interval * 4
    assert [e.type for e in (await p.get_state()).events] == [event]


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
def test_invalid_scenario_tuning_rejected(bad: float) -> None:
    with pytest.raises(ValueError, match="death_interval_seconds"):
        MockPerception(death_interval_seconds=bad)
    with pytest.raises(ValueError, match="stuck_interval_seconds"):
        MockPerception(stuck_interval_seconds=bad)
    with pytest.raises(ValueError, match="hp_step_sigma"):
        MockPerception(hp_step_sigma=bad)
    with pytest.raises(ValueError, match="event_probability"):
        MockPerception(event_probability=bad)


def test_invalid_pool_and_combat_settings_rejected() -> None:
    with pytest.raises(ValueError, match="event_pool"):
        MockPerception(event_pool=("unknown",))
    with pytest.raises(ValueError, match="combat_cycle_period"):
        MockPerception(combat_cycle_period=0.0)
    with pytest.raises(ValueError, match="max_enemies"):
        MockPerception(max_enemies=-1)
