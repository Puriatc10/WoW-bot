"""Unit and integration tests for MetaStateGenerator (Task 3.5)."""

from __future__ import annotations

from collections.abc import AsyncGenerator

import numpy as np
import pytest

from wow_bot.internal_dynamics.chaos import LorenzAttractor
from wow_bot.internal_dynamics.drives import Drives
from wow_bot.internal_dynamics.memory import MemoryStore
from wow_bot.internal_dynamics.meta_state import MetaStateGenerator
from wow_bot.internal_dynamics.oscillators import OscillatorBank
from wow_bot.shared.config import InternalDynamicsConfig
from wow_bot.shared.events import DEATH, RARE_LOOT
from wow_bot.shared.interfaces import Event, GameState, MetaState


def make_dummy_game_state(
    timestamp: float = 100.0,
    events: list[Event] | None = None,
) -> GameState:
    """Helper to build a valid GameState."""
    return GameState(
        timestamp=timestamp,
        hp_pct=1.0,
        mana_pct=1.0,
        position=(10.0, 20.0),
        facing=0.0,
        in_combat=False,
        target=None,
        enemies=[],
        events=events if events is not None else [],
    )


@pytest.fixture
async def memory_store() -> AsyncGenerator[MemoryStore, None]:
    """Fixture providing an initialized in-memory MemoryStore."""
    store = MemoryStore(":memory:")
    await store.init()
    try:
        yield store
    finally:
        await store.close()


@pytest.fixture
def generator_and_deps(
    memory_store: MemoryStore,
) -> tuple[MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore]:
    """Fixture returning a MetaStateGenerator with real dependencies."""
    drives = Drives()
    oscillators = OscillatorBank(seed=42)
    chaos = LorenzAttractor()
    config = InternalDynamicsConfig(trigger_threshold_base=0.3)
    generator = MetaStateGenerator(
        config=config,
        drives=drives,
        oscillators=oscillators,
        chaos=chaos,
        memory=memory_store,
    )
    return generator, drives, oscillators, chaos, memory_store


@pytest.mark.asyncio
async def test_a_valid_basic_step(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test A — valid basic step with no events."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps
    gs = make_dummy_game_state(timestamp=123.45, events=[])

    meta = await generator.step(dt=0.1, game_state=gs)

    assert isinstance(meta, MetaState)
    assert meta.vector.shape == (5,)
    assert np.all(np.isfinite(meta.vector))
    assert np.all((meta.vector >= 0.0) & (meta.vector <= 1.0))
    assert meta.timestamp == 123.45
    assert meta.recent_events == []


@pytest.mark.asyncio
async def test_b_100_step_boundedness(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test B — 100-step boundedness criterion."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    for i in range(100):
        gs = make_dummy_game_state(timestamp=100.0 + i * 0.1)
        meta = await generator.step(dt=0.1, game_state=gs)

        assert meta.vector.shape == (5,)
        assert np.all(np.isfinite(meta.vector))
        assert np.all((meta.vector >= 0.0) & (meta.vector <= 1.0))


@pytest.mark.asyncio
async def test_c_event_affects_metastate(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test C — event affects MetaState drive values."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    gs_normal = make_dummy_game_state(timestamp=1.0)
    meta_before = await generator.step(dt=0.1, game_state=gs_normal)

    death_event = Event(type=DEATH, timestamp=2.0)
    gs_death = make_dummy_game_state(timestamp=2.0, events=[death_event])
    meta_after = await generator.step(dt=0.1, game_state=gs_death)

    # DEATH reduces aggression (-0.15) and increases fatigue (+0.10)
    # Aggression index is 3 in [hunger, fatigue, curiosity, aggression, social]
    assert meta_after.vector[3] < meta_before.vector[3]


@pytest.mark.asyncio
async def test_d_and_e_event_persisted_in_memory(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test D & E — events stored in memory and no-event memory behavior."""
    generator, drives, _oscillators, _chaos, memory = generator_and_deps

    # Test E — step with no events produces no memory entries
    gs_no_events = make_dummy_game_state(timestamp=1.0, events=[])
    await generator.step(dt=0.1, game_state=gs_no_events)
    recalled = await memory.recall_similar(drives.vector, k=10)
    assert len(recalled) == 0

    # Test D — step with event persists it
    ev = Event(type=RARE_LOOT, timestamp=2.0, data={"item": "epic_sword"})
    gs_event = make_dummy_game_state(timestamp=2.0, events=[ev])
    meta = await generator.step(dt=0.1, game_state=gs_event)

    recalled = await memory.recall_similar(meta.vector, k=10)
    assert len(recalled) == 1
    assert recalled[0].type == RARE_LOOT
    assert recalled[0].data == {"item": "epic_sword"}


@pytest.mark.asyncio
async def test_f_recent_events_preserved(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test F — recent events preserved in order with copied list."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    ev1 = Event(type=RARE_LOOT, timestamp=10.0)
    ev2 = Event(type=DEATH, timestamp=10.1)
    events_input = [ev1, ev2]

    gs = make_dummy_game_state(timestamp=10.1, events=events_input)
    meta = await generator.step(dt=0.1, game_state=gs)

    assert meta.recent_events == events_input
    assert meta.recent_events is not gs.events


@pytest.mark.asyncio
async def test_g_timestamp_propagation(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test G — timestamp propagation from GameState."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    gs = make_dummy_game_state(timestamp=987654.321)
    meta = await generator.step(dt=0.1, game_state=gs)

    assert meta.timestamp == 987654.321


def test_h_i_j_trigger_behavior() -> None:
    """Test H, I, J — trigger evaluation (small delta, large delta, exact boundary)."""
    drives = Drives()
    oscillators = OscillatorBank(seed=1)
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")
    gen = MetaStateGenerator(
        config=0.3,
        drives=drives,
        oscillators=oscillators,
        chaos=chaos,
        memory=store,
    )

    v_base = np.array([0.2, 0.5, 0.5, 0.5, 0.5])
    last_meta = MetaState(vector=v_base, recent_events=[], timestamp=1.0)

    # Test H — small trigger delta (distance = 0.1 < 0.3) -> False
    v_small = np.array([0.3, 0.5, 0.5, 0.5, 0.5])
    curr_small = MetaState(vector=v_small, recent_events=[], timestamp=2.0)
    assert gen.should_trigger_llm(curr_small, last_meta) is False

    # Test I — large trigger delta (distance = 0.4 > 0.3) -> True
    v_large = np.array([0.6, 0.5, 0.5, 0.5, 0.5])
    curr_large = MetaState(vector=v_large, recent_events=[], timestamp=2.0)
    assert gen.should_trigger_llm(curr_large, last_meta) is True

    # Test J — exact boundary trigger delta (distance = 0.3 == 0.3) -> Strict > comparison -> False
    # 0.5 - 0.2 = 0.3 exactly in float arithmetic
    v_boundary = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    curr_boundary = MetaState(vector=v_boundary, recent_events=[], timestamp=2.0)
    assert float(np.linalg.norm(v_boundary - v_base)) == 0.3
    assert gen.should_trigger_llm(curr_boundary, last_meta) is False


@pytest.mark.asyncio
async def test_k_negative_dt_raises(memory_store: MemoryStore) -> None:
    """Test K — negative dt raises ValueError before mutating collaborators."""

    class SpyChaos(LorenzAttractor):
        def __init__(self) -> None:
            super().__init__()
            self.called = False

        def step(self) -> np.ndarray:
            self.called = True
            return super().step()

    chaos_spy = SpyChaos()
    drives = Drives()
    oscillators = OscillatorBank()
    gen = MetaStateGenerator(
        config=0.3,
        drives=drives,
        oscillators=oscillators,
        chaos=chaos_spy,
        memory=memory_store,
    )

    gs = make_dummy_game_state()
    with pytest.raises(ValueError, match="dt must be non-negative"):
        await gen.step(-0.1, gs)

    # Confirm chaos step was NOT called
    assert chaos_spy.called is False


@pytest.mark.asyncio
async def test_l_vector_isolation(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test L — mutating returned meta.vector does not corrupt Drives."""
    generator, drives, _oscillators, _chaos, _memory = generator_and_deps

    gs = make_dummy_game_state()
    meta = await generator.step(dt=0.1, game_state=gs)

    # Mutate returned vector
    meta.vector[0] = 0.999

    # Verify drives internal state was NOT mutated
    assert drives.vector[0] != 0.999


@pytest.mark.asyncio
async def test_m_recent_events_isolation(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test M — mutating meta.recent_events does not mutate game_state.events."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    ev = Event(type=RARE_LOOT, timestamp=1.0)
    gs = make_dummy_game_state(events=[ev])
    meta = await generator.step(dt=0.1, game_state=gs)

    meta.recent_events.clear()
    assert len(gs.events) == 1


@pytest.mark.asyncio
async def test_n_multiple_events_persisted(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test N — multiple events persisted to memory."""
    generator, drives, _oscillators, _chaos, memory = generator_and_deps

    ev1 = Event(type=RARE_LOOT, timestamp=1.0, data={"id": 1})
    ev2 = Event(type=DEATH, timestamp=2.0, data={"id": 2})
    gs = make_dummy_game_state(events=[ev1, ev2])

    await generator.step(dt=0.1, game_state=gs)

    recalled = await memory.recall_similar(drives.vector, k=10)
    assert len(recalled) == 2


@pytest.mark.asyncio
async def test_o_dependency_evolution(
    generator_and_deps: tuple[
        MetaStateGenerator, Drives, OscillatorBank, LorenzAttractor, MemoryStore
    ],
) -> None:
    """Test O — repeated generator steps advance dynamic state rather than staying static."""
    generator, _drives, _oscillators, _chaos, _memory = generator_and_deps

    gs = make_dummy_game_state()
    meta1 = await generator.step(dt=1.0, game_state=gs)
    meta2 = await generator.step(dt=1.0, game_state=gs)

    # State vectors should evolve over dt=1.0
    assert not np.array_equal(meta1.vector, meta2.vector)


def test_invalid_threshold_config() -> None:
    """Test initialization fails when trigger_threshold_base is non-positive or NaN."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")

    with pytest.raises(
        ValueError, match="trigger_threshold_base must be a positive finite number"
    ):
        MetaStateGenerator(
            config=0.0, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
        )

    with pytest.raises(
        ValueError, match="trigger_threshold_base must be a positive finite number"
    ):
        MetaStateGenerator(
            config=-0.5, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
        )


def test_should_trigger_llm_input_validation() -> None:
    """Test input validation in should_trigger_llm."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
    )

    m_valid = MetaState(vector=np.array([0.5] * 5), recent_events=[], timestamp=1.0)
    m_bad_shape = MetaState(vector=np.array([0.5] * 5), recent_events=[], timestamp=1.0)
    m_bad_shape.vector = np.array([0.5] * 4)  # force bad shape

    with pytest.raises(ValueError, match="MetaState vectors must have shape"):
        gen.should_trigger_llm(m_valid, m_bad_shape)


# --- Task 3.6 Adaptive Trigger Threshold Tests ---


def test_3_6_test_a_initial_threshold() -> None:
    """Test A — immediately after construction, trigger_threshold == base_threshold."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
    )

    assert gen.trigger_threshold == pytest.approx(0.3)
    assert gen.trigger_threshold == gen.trigger_threshold_base


def test_3_6_test_b_getter_purity() -> None:
    """Test B — repeated reads of trigger_threshold do not alter state or threshold."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
    )

    val1 = gen.trigger_threshold
    val2 = gen.trigger_threshold
    val3 = gen.trigger_threshold
    assert val1 == val2 == val3 == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_3_6_test_c_zero_dt(memory_store: MemoryStore) -> None:
    """Test C — calling step with dt=0 does not advance threshold time."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )

    gs = make_dummy_game_state()
    await gen.step(0.0, gs)

    assert gen.trigger_threshold == pytest.approx(0.3)


@pytest.mark.asyncio
async def test_3_6_test_d_negative_dt_preserves_threshold(
    memory_store: MemoryStore,
) -> None:
    """Test D — negative dt raises error and leaves threshold time unchanged."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )

    gs = make_dummy_game_state()
    await gen.step(1800.0, gs)
    threshold_at_1800 = gen.trigger_threshold
    assert threshold_at_1800 == pytest.approx(0.4)

    with pytest.raises(ValueError, match="dt must be non-negative"):
        await gen.step(-10.0, gs)

    assert gen.trigger_threshold == threshold_at_1800


@pytest.mark.asyncio
async def test_3_6_test_e_f_g_h_checkpoints(memory_store: MemoryStore) -> None:
    """Tests E, F, G, H — threshold checkpoint values at quarter, half, 3/4, and full periods."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )
    gs = make_dummy_game_state()

    # t = 0 -> 0.3
    assert gen.trigger_threshold == pytest.approx(0.3, abs=1e-6)

    # Test E: t = 1800 (quarter period) -> base + 0.1 = 0.4
    await gen.step(1800.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.4, abs=1e-6)

    # Test F: t = 3600 (half period) -> base = 0.3
    await gen.step(1800.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.3, abs=1e-6)

    # Test G: t = 5400 (three-quarter period) -> base - 0.1 = 0.2
    await gen.step(1800.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.2, abs=1e-6)

    # Test H: t = 7200 (full period) -> base = 0.3
    await gen.step(1800.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.3, abs=1e-6)


@pytest.mark.asyncio
async def test_3_6_test_i_bounded_range(memory_store: MemoryStore) -> None:
    """Test I — threshold remains bounded in [base - 0.1, base + 0.1] across multiple cycles."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    base = 0.35
    gen = MetaStateGenerator(
        config=base, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )
    gs = make_dummy_game_state()

    # Step in 100s increments for over 3 cycles (21,600 seconds)
    for _ in range(216):
        await gen.step(100.0, gs)
        val = gen.trigger_threshold
        assert base - 0.1 - 1e-9 <= val <= base + 0.1 + 1e-9


@pytest.mark.asyncio
async def test_3_6_test_j_adaptive_trigger_decision(memory_store: MemoryStore) -> None:
    """Test J — identical MetaState delta produces different trigger decisions across phases."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )
    gs = make_dummy_game_state()

    last_meta = MetaState(
        vector=np.array([0.2, 0.5, 0.5, 0.5, 0.5]), recent_events=[], timestamp=1.0
    )
    # Delta = 0.30
    curr_meta = MetaState(
        vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5]), recent_events=[], timestamp=2.0
    )

    # At t = 1800s, threshold is ~0.4. delta (0.3) < threshold (0.4) -> False
    await gen.step(1800.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.4)
    assert gen.should_trigger_llm(curr_meta, last_meta) is False

    # At t = 5400s (advance another 3600s), threshold is ~0.2. delta (0.3) > threshold (0.2) -> True
    await gen.step(3600.0, gs)
    assert gen.trigger_threshold == pytest.approx(0.2)
    assert gen.should_trigger_llm(curr_meta, last_meta) is True


@pytest.mark.asyncio
async def test_3_6_test_k_strict_boundary(memory_store: MemoryStore) -> None:
    """Test K — when delta == current threshold, trigger decision is False."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=memory_store
    )

    # Initial threshold = 0.3
    last_meta = MetaState(
        vector=np.array([0.2, 0.5, 0.5, 0.5, 0.5]), recent_events=[], timestamp=1.0
    )
    curr_meta = MetaState(
        vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5]), recent_events=[], timestamp=2.0
    )

    delta = float(np.linalg.norm(curr_meta.vector - last_meta.vector))
    assert delta == pytest.approx(0.3)
    assert gen.should_trigger_llm(curr_meta, last_meta) is False


@pytest.mark.asyncio
async def test_3_6_test_l_deterministic_evolution(memory_store: MemoryStore) -> None:
    """Test L — two equivalent generators receiving identical dt sequences evolve identically."""
    gen1 = MetaStateGenerator(
        config=0.3,
        drives=Drives(),
        oscillators=OscillatorBank(seed=10),
        chaos=LorenzAttractor(),
        memory=memory_store,
    )
    gen2 = MetaStateGenerator(
        config=0.3,
        drives=Drives(),
        oscillators=OscillatorBank(seed=10),
        chaos=LorenzAttractor(),
        memory=memory_store,
    )
    gs = make_dummy_game_state()

    dt_sequence = [100.0, 250.5, 1800.0, 0.0, 42.0]
    for dt in dt_sequence:
        await gen1.step(dt, gs)
        await gen2.step(dt, gs)
        assert gen1.trigger_threshold == pytest.approx(gen2.trigger_threshold)


def test_3_6_test_m_no_wall_clock_dependence() -> None:
    """Test M — threshold evolution depends purely on dt and not wall clock time."""
    drives = Drives()
    oscillators = OscillatorBank()
    chaos = LorenzAttractor()
    store = MemoryStore(":memory:")
    gen = MetaStateGenerator(
        config=0.3, drives=drives, oscillators=oscillators, chaos=chaos, memory=store
    )

    # Threshold without stepping remains 0.3 regardless of wall-clock time passing
    assert gen.trigger_threshold == pytest.approx(0.3)
