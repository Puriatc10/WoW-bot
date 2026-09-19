"""Unit tests for SQLite-backed async MemoryStore (Task 3.4)."""

from __future__ import annotations

import asyncio
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from wow_bot.internal_dynamics.memory import MemoryStore
from wow_bot.shared.events import DEATH, PVP_KILL, RARE_LOOT
from wow_bot.shared.interfaces import Event


@pytest.mark.asyncio
async def test_init_creates_schema() -> None:
    """Test A: init creates schema without error and prepares usable store."""
    store = MemoryStore(":memory:")
    await store.init()
    # Verify store is usable by adding an event
    event = Event(type=RARE_LOOT, timestamp=100.0, data={"item": "Sword"})
    vec = np.full(5, 0.5, dtype=np.float64)
    await store.add(event, vec)
    recalled = await store.recall_similar(vec, k=1)
    assert len(recalled) == 1
    assert recalled[0].type == RARE_LOOT
    await store.close()


@pytest.mark.asyncio
async def test_add_and_recall_nearest() -> None:
    """Test B: add and recall nearest vector."""
    store = MemoryStore(":memory:")
    await store.init()

    event_a = Event(type=RARE_LOOT, timestamp=100.0, data={"name": "A"})
    vec_a = np.array([0.50, 0.50, 0.50, 0.50, 0.50], dtype=np.float64)

    event_b = Event(type=DEATH, timestamp=101.0, data={"name": "B"})
    vec_b = np.array([0.90, 0.90, 0.90, 0.90, 0.90], dtype=np.float64)

    await store.add(event_a, vec_a)
    await store.add(event_b, vec_b)

    query_vec = np.array([0.51, 0.51, 0.51, 0.51, 0.51], dtype=np.float64)
    recalled = await store.recall_similar(query_vec, k=1)

    assert len(recalled) == 1
    assert recalled[0].type == RARE_LOOT
    assert recalled[0].data["name"] == "A"

    await store.close()


@pytest.mark.asyncio
async def test_k_nearest_ordering() -> None:
    """Test C: k nearest ordering across multiple separated vectors."""
    store = MemoryStore(":memory:")
    await store.init()

    events_data = [
        ("near", np.array([0.1, 0.1, 0.1, 0.1, 0.1])),
        ("far", np.array([0.9, 0.9, 0.9, 0.9, 0.9])),
        ("mid", np.array([0.4, 0.4, 0.4, 0.4, 0.4])),
    ]

    for label, vec in events_data:
        await store.add(Event(type=RARE_LOOT, timestamp=100.0, data={"label": label}), vec)

    query = np.array([0.0, 0.0, 0.0, 0.0, 0.0])
    results = await store.recall_similar(query, k=3)

    assert len(results) == 3
    assert results[0].data["label"] == "near"
    assert results[1].data["label"] == "mid"
    assert results[2].data["label"] == "far"

    await store.close()


@pytest.mark.asyncio
async def test_deterministic_tie_handling() -> None:
    """Test D: tie handling ordering using distance ASC, id ASC."""
    store = MemoryStore(":memory:")
    await store.init()

    vec = np.array([0.5, 0.5, 0.5, 0.5, 0.5])
    event_1 = Event(type=PVP_KILL, timestamp=10.0, data={"seq": 1})
    event_2 = Event(type=PVP_KILL, timestamp=20.0, data={"seq": 2})

    await store.add(event_1, vec)
    await store.add(event_2, vec)

    recalled = await store.recall_similar(vec, k=2)
    assert len(recalled) == 2
    assert recalled[0].data["seq"] == 1
    assert recalled[1].data["seq"] == 2

    await store.close()


@pytest.mark.asyncio
async def test_empty_database() -> None:
    """Test E: calling recall_similar on empty DB returns []."""
    store = MemoryStore(":memory:")
    await store.init()

    vec = np.full(5, 0.5)
    recalled = await store.recall_similar(vec, k=5)
    assert recalled == []

    await store.close()


@pytest.mark.asyncio
async def test_fewer_rows_than_k() -> None:
    """Test F: requesting k=5 when only 2 rows exist returns 2 unique events."""
    store = MemoryStore(":memory:")
    await store.init()

    vec = np.full(5, 0.5)
    await store.add(Event(type=RARE_LOOT, timestamp=1.0, data={"idx": 1}), vec)
    await store.add(Event(type=DEATH, timestamp=2.0, data={"idx": 2}), vec)

    recalled = await store.recall_similar(vec, k=5)
    assert len(recalled) == 2
    assert recalled[0].data["idx"] == 1
    assert recalled[1].data["idx"] == 2

    await store.close()


@pytest.mark.asyncio
async def test_invalid_vector_shape_on_add() -> None:
    """Test G: invalid vector shape on add raises ValueError."""
    store = MemoryStore(":memory:")
    await store.init()

    event = Event(type=RARE_LOOT, timestamp=1.0)
    bad_vectors: list[Any] = [
        np.array([0.1, 0.2, 0.3, 0.4]),  # (4,)
        np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]),  # (6,)
        np.array([[0.1, 0.2, 0.3, 0.4, 0.5]]),  # (1, 5)
    ]

    for bad_vec in bad_vectors:
        with pytest.raises(ValueError, match="State vector must have shape"):
            await store.add(event, bad_vec)

    await store.close()


@pytest.mark.asyncio
async def test_invalid_vector_shape_on_recall() -> None:
    """Test H: invalid query vector shape on recall raises ValueError."""
    store = MemoryStore(":memory:")
    await store.init()

    bad_vec = np.array([0.1, 0.2, 0.3, 0.4])
    with pytest.raises(ValueError, match="State vector must have shape"):
        await store.recall_similar(bad_vec, k=5)

    await store.close()


@pytest.mark.asyncio
async def test_nan_inf_validation() -> None:
    """Test I: NaN / Inf values rejected on add and recall."""
    store = MemoryStore(":memory:")
    await store.init()

    event = Event(type=RARE_LOOT, timestamp=1.0)
    nan_vec = np.array([0.5, 0.5, np.nan, 0.5, 0.5])
    pos_inf_vec = np.array([0.5, np.inf, 0.5, 0.5, 0.5])
    neg_inf_vec = np.array([-np.inf, 0.5, 0.5, 0.5, 0.5])

    for invalid_vec in (nan_vec, pos_inf_vec, neg_inf_vec):
        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            await store.add(event, invalid_vec)

        with pytest.raises(ValueError, match="must not contain NaN or Inf"):
            await store.recall_similar(invalid_vec)

    await store.close()


@pytest.mark.asyncio
async def test_json_event_data_roundtrip() -> None:
    """Test J: event data JSON round-trip fidelity."""
    store = MemoryStore(":memory:")
    await store.init()

    payload = {
        "source": "mock",
        "count": 2,
        "flag": True,
        "details": {"nested": [1, 2, 3]},
    }
    event = Event(type=RARE_LOOT, timestamp=123.456, data=payload)
    vec = np.full(5, 0.5)

    await store.add(event, vec)
    recalled = await store.recall_similar(vec, k=1)

    assert len(recalled) == 1
    assert recalled[0].data == payload

    await store.close()


@pytest.mark.asyncio
async def test_timestamp_preservation() -> None:
    """Test K: stored event timestamp is exactly preserved."""
    store = MemoryStore(":memory:")
    await store.init()

    exact_time = 1700000000.123456
    event = Event(type=PVP_KILL, timestamp=exact_time)
    vec = np.full(5, 0.5)

    await store.add(event, vec)
    recalled = await store.recall_similar(vec, k=1)

    assert len(recalled) == 1
    assert recalled[0].timestamp == pytest.approx(exact_time)

    await store.close()


@pytest.mark.asyncio
async def test_decay_old_removes_expired_memories() -> None:
    """Test L & M: decay_old removes expired memories based on cutoff time."""
    store = MemoryStore(":memory:")
    await store.init()

    fixed_now = 100000.0  # seconds
    max_age_hours = 24.0

    old_event = Event(type=DEATH, timestamp=13599.0, data={"age": "old"})
    boundary_event = Event(type=DEATH, timestamp=13600.0, data={"age": "boundary"})
    recent_event = Event(type=DEATH, timestamp=50000.0, data={"age": "recent"})

    vec = np.full(5, 0.5)
    await store.add(old_event, vec)
    await store.add(boundary_event, vec)
    await store.add(recent_event, vec)

    with patch("time.time", return_value=fixed_now):
        await store.decay_old(max_age_hours=max_age_hours)

    recalled = await store.recall_similar(vec, k=10)
    ages = [e.data["age"] for e in recalled]

    assert "old" not in ages
    assert "boundary" in ages
    assert "recent" in ages

    await store.close()


@pytest.mark.asyncio
async def test_invalid_max_age_hours() -> None:
    """Test N: negative max_age_hours raises ValueError."""
    store = MemoryStore(":memory:")
    await store.init()

    with pytest.raises(ValueError, match="max_age_hours must be non-negative"):
        await store.decay_old(max_age_hours=-1.0)

    await store.close()


@pytest.mark.asyncio
async def test_invalid_k() -> None:
    """Test O: non-positive k raises ValueError."""
    store = MemoryStore(":memory:")
    await store.init()

    vec = np.full(5, 0.5)
    for invalid_k in (0, -1, -5):
        with pytest.raises(ValueError, match="k must be positive"):
            await store.recall_similar(vec, k=invalid_k)

    await store.close()


@pytest.mark.asyncio
async def test_persistence_after_reopen(tmp_path: Any) -> None:
    """Test P & Reinitialization: persistence across store instances in file DB."""
    db_file = tmp_path / "test_persist.db"
    db_path = str(db_file)

    store1 = MemoryStore(db_path)
    await store1.init()
    vec = np.full(5, 0.5)
    await store1.add(Event(type=RARE_LOOT, timestamp=50.0, data={"persisted": True}), vec)
    await store1.close()

    store2 = MemoryStore(db_path)
    await store2.init()
    recalled = await store2.recall_similar(vec, k=1)
    assert len(recalled) == 1
    assert recalled[0].data["persisted"] is True

    # Support close and re-init on same instance
    await store2.close()
    await store2.init()
    recalled_again = await store2.recall_similar(vec, k=1)
    assert len(recalled_again) == 1

    await store2.close()


@pytest.mark.asyncio
async def test_close_idempotency() -> None:
    """Test Q: calling close repeatedly does not crash."""
    store = MemoryStore(":memory:")
    await store.init()

    await store.close()
    await store.close()


@pytest.mark.asyncio
async def test_operation_before_init_or_after_close() -> None:
    """Test R: operations before init or after close fail clearly with RuntimeError."""
    store = MemoryStore(":memory:")
    vec = np.full(5, 0.5)
    event = Event(type=RARE_LOOT, timestamp=1.0)

    with pytest.raises(RuntimeError, match="MemoryStore is not initialized"):
        await store.add(event, vec)

    with pytest.raises(RuntimeError, match="MemoryStore is not initialized"):
        await store.recall_similar(vec)

    with pytest.raises(RuntimeError, match="MemoryStore is not initialized"):
        await store.decay_old(24)

    await store.init()
    await store.close()

    with pytest.raises(RuntimeError, match="MemoryStore is not initialized"):
        await store.add(event, vec)


@pytest.mark.asyncio
async def test_concurrent_add_and_vector_fidelity() -> None:
    """Test S & Concurrent Add: concurrent asyncio.gather adds and vector fidelity."""
    store = MemoryStore(":memory:")
    await store.init()

    vecs = [
        np.array([i / 10.0] * 5, dtype=np.float64)
        for i in range(1, 6)
    ]
    events = [
        Event(type=RARE_LOOT, timestamp=float(i), data={"i": i})
        for i in range(1, 6)
    ]

    await asyncio.gather(*(store.add(ev, v) for ev, v in zip(events, vecs)))

    recalled = await store.recall_similar(vecs[0], k=10)
    assert len(recalled) == 5

    await store.close()
