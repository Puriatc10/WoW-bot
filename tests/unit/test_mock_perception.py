"""Acceptance tests for Task 2.1 — Simple Mock Perception.

Roadmap acceptance criteria:
    async def test():
        p = MockPerception()
        for _ in range(10):
            state = await p.get_state()
            assert 0 <= state.hp_pct <= 1
            assert isinstance(state.timestamp, float)
            await asyncio.sleep(0.1)
    Runs without error.

Additional invariant checks derived from the task requirements:
- HP/Mana stay within [0.3, 1.0].
- Position performs a random walk (changes across frames).
- Events are drawn only from the registered event-type set.
- Combat entities appear only while ``in_combat`` is True.
- Determinism: identical seeds produce identical frame sequences.
"""

from __future__ import annotations

import asyncio
import itertools
import math

import numpy as np
import pytest

from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.shared.events import ALL_EVENT_TYPES
from wow_bot.shared.interfaces import GameState

# ---------------------------------------------------------------------------
# Canonical acceptance snippet from ROADMAP.md Task 2.1
# ---------------------------------------------------------------------------


async def test_acceptance_snippet_runs_without_error() -> None:
    p = MockPerception(rng=np.random.default_rng(42))
    for _ in range(10):
        state = await p.get_state()
        assert 0 <= state.hp_pct <= 1
        assert isinstance(state.timestamp, float)
        await asyncio.sleep(0.01)


# ---------------------------------------------------------------------------
# Resource bounds
# ---------------------------------------------------------------------------


async def test_hp_and_mana_stay_within_resource_band() -> None:
    p = MockPerception(rng=np.random.default_rng(7))
    for _ in range(200):
        state = await p.get_state()
        assert 0.3 <= state.hp_pct <= 1.0, f"hp_pct out of band: {state.hp_pct}"
        assert 0.3 <= state.mana_pct <= 1.0, f"mana_pct out of band: {state.mana_pct}"


async def test_all_values_are_finite_floats() -> None:
    p = MockPerception(rng=np.random.default_rng(11))
    for _ in range(50):
        state = await p.get_state()
        for value in (state.timestamp, state.hp_pct, state.mana_pct, state.facing):
            assert isinstance(value, float)
            assert math.isfinite(value), f"non-finite value: {value}"


# ---------------------------------------------------------------------------
# GameState contract conformance
# ---------------------------------------------------------------------------


async def test_emitted_frames_satisfy_gamestate_contract() -> None:
    """Every frame must survive a JSON round trip through the shared contract."""
    p = MockPerception(rng=np.random.default_rng(99))
    for _ in range(30):
        state = await p.get_state()
        payload = state.to_dict()
        restored = type(state).from_dict(payload)
        assert restored.hp_pct == pytest.approx(state.hp_pct)
        assert restored.mana_pct == pytest.approx(state.mana_pct)
        assert restored.position == pytest.approx(state.position)
        assert restored.in_combat == state.in_combat
        assert len(restored.events) == len(state.events)


# ---------------------------------------------------------------------------
# Position random walk
# ---------------------------------------------------------------------------


async def test_position_random_walk_moves_over_time() -> None:
    p = MockPerception(rng=np.random.default_rng(3), start_position=(0.0, 0.0))
    positions: list[tuple[float, float]] = []
    for _ in range(50):
        state = await p.get_state()
        positions.append(state.position)

    # The walk should not be frozen at the origin.
    displacements = [math.hypot(x, y) for x, y in positions]
    assert max(displacements) > 0.0
    # Consecutive steps are small (slow walk), never teleporting.
    for prev, cur in itertools.pairwise(positions):
        step = math.hypot(cur[0] - prev[0], cur[1] - prev[1])
        assert step < 5.0, f"position jumped by {step} units in one frame"


async def test_facing_stays_in_full_circle_range() -> None:
    p = MockPerception(rng=np.random.default_rng(5))
    for _ in range(50):
        state = await p.get_state()
        assert 0.0 <= state.facing < 2.0 * math.pi


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


async def test_events_only_use_registered_types() -> None:
    p = MockPerception(rng=np.random.default_rng(21), event_probability=1.0)
    seen_types: set[str] = set()
    for _ in range(100):
        state = await p.get_state()
        assert len(state.events) <= 1, "simple mock emits at most one event per frame"
        for event in state.events:
            assert event.type in ALL_EVENT_TYPES
            assert isinstance(event.timestamp, float)
            assert isinstance(event.data, dict)
            seen_types.add(event.type)

    # With probability 1.0 over 100 draws we expect several distinct types.
    assert len(seen_types) >= 2


async def test_zero_event_probability_emits_nothing() -> None:
    p = MockPerception(rng=np.random.default_rng(22), event_probability=0.0)
    for _ in range(50):
        state = await p.get_state()
        assert state.events == []


# ---------------------------------------------------------------------------
# Combat scheduling and entities
# ---------------------------------------------------------------------------


async def test_combat_entities_present_only_during_combat() -> None:
    """Combat entities must appear only while in_combat is True."""
    p = MockPerception(
        rng=np.random.default_rng(31),
        combat_cycle_period=0.2,
        combat_on_duration=0.1,
    )
    saw_combat = False
    saw_peace = False
    for _ in range(80):
        state = await p.get_state()
        if state.in_combat:
            saw_combat = True
            assert state.target is not None
            assert state.target.reaction == "hostile"
            assert 1 <= len(state.enemies) <= 3
            for enemy in state.enemies:
                assert len(enemy.bbox) == 4
                assert 0.0 <= enemy.confidence <= 1.0
                assert enemy.distance_estimate >= 0.0
        else:
            saw_peace = True
            assert state.target is None
            assert state.enemies == []
        await asyncio.sleep(0.01)

    assert saw_combat and saw_peace


async def test_combat_toggles_on_off_with_short_schedule() -> None:
    """With period ~0.2 s and on-duration ~0.1 s, combat must toggle repeatedly."""
    p = MockPerception(
        rng=np.random.default_rng(41),
        combat_cycle_period=0.2,
        combat_on_duration=0.1,
    )
    flags: list[bool] = []
    for _ in range(80):
        state = await p.get_state()
        flags.append(state.in_combat)
        await asyncio.sleep(0.01)

    transitions = sum(a != b for a, b in itertools.pairwise(flags))
    assert any(flags), "combat never turned on"
    assert not all(flags), "combat never turned off"
    assert transitions >= 2, f"expected toggling, got {transitions} transitions"


# ---------------------------------------------------------------------------
# Determinism under fixed seed (AGENTS.md testing philosophy rule 1)
# ---------------------------------------------------------------------------


def _frame_signature(state: GameState) -> tuple[float, float, tuple[float, float], float, bool, tuple[str, ...]]:
    """Extract comparable fields from a frame, ignoring wall-clock timestamps."""
    return (
        state.hp_pct,
        state.mana_pct,
        state.position,
        state.facing,
        state.in_combat,
        tuple(e.type for e in state.events),
    )


async def test_same_seed_produces_identical_sequences() -> None:
    a = MockPerception(rng=np.random.default_rng(1234), event_probability=0.3)
    b = MockPerception(rng=np.random.default_rng(1234), event_probability=0.3)
    for _ in range(25):
        sa = await a.get_state()
        sb = await b.get_state()
        assert _frame_signature(sa) == _frame_signature(sb)


async def test_different_seeds_diverge() -> None:
    a = MockPerception(rng=np.random.default_rng(1))
    b = MockPerception(rng=np.random.default_rng(2))
    sigs_a = [_frame_signature(await a.get_state()) for _ in range(10)]
    sigs_b = [_frame_signature(await b.get_state()) for _ in range(10)]
    assert sigs_a != sigs_b


# ---------------------------------------------------------------------------
# Default construction (no rng given) — matches the roadmap snippet exactly
# ---------------------------------------------------------------------------


async def test_default_constructor_works() -> None:
    p = MockPerception()
    state = await p.get_state()
    assert 0.0 <= state.hp_pct <= 1.0
    assert isinstance(state.timestamp, float)