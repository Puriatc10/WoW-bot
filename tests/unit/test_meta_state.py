"""Unit and integration tests for MetaStateGenerator (Task 3.5)."""

from __future__ import annotations

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
async def memory_store() -> MemoryStore:
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
