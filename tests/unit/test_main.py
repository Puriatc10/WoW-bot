"""Tests for Task 7.1 Async Pipeline in src/wow_bot/main.py."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock

import pytest

from wow_bot.main import (
    PipelineSnapshot,
    _put_latest,
    build_runtime,
    run_pipeline,
)
from wow_bot.shared.interfaces import GameState, MetaState, Strategy


@pytest.mark.asyncio
async def test_put_latest_helper() -> None:
    queue: asyncio.Queue[int] = asyncio.Queue(maxsize=1)
    _put_latest(queue, 1)
    assert queue.full()
    assert queue.qsize() == 1

    _put_latest(queue, 2)
    assert queue.qsize() == 1
    assert await queue.get() == 2


@pytest.mark.asyncio
async def test_build_runtime() -> None:
    components = await build_runtime()
    assert components.config is not None
    assert components.controller.dry_run is True
    assert components.memory is not None

    # Clean up memory store
    await components.memory.close()
    await components.llm_client.close()
    components.watchdog.close()


@pytest.mark.asyncio
async def test_run_pipeline_bounded_duration() -> None:
    components = await build_runtime(scenario="peaceful_farm")

    # Mock Strategist to return a valid Strategy quickly without network/LLM calls
    mock_strategy = Strategy(
        goal="farm_herbs",
        region="elwynn",
        risk_tolerance=0.2,
        priority=["herbs"],
        constraints={},
        valid_until=10000000000.0,
    )
    components.strategist.generate_strategy = AsyncMock(return_value=mock_strategy)  # type: ignore[method-assign]

    # Run for 1.0 wall-clock second
    await run_pipeline(components, run_duration_seconds=1.0)

    # Verify controller stop_all was invoked during cleanup
    assert len(components.controller.commands) > 0
    assert components.controller.commands[-1].action == "stop_all"


@pytest.mark.asyncio
async def test_run_pipeline_invalid_duration() -> None:
    components = await build_runtime()
    with pytest.raises(ValueError, match="run_duration_seconds must be a positive finite float"):
        await run_pipeline(components, run_duration_seconds=-5.0)

    await components.memory.close()
    await components.llm_client.close()
    components.watchdog.close()


@pytest.mark.asyncio
async def test_pipeline_snapshot_dataclass() -> None:
    gs = GameState(
        timestamp=100.0,
        hp_pct=1.0,
        mana_pct=1.0,
        position=(0.0, 0.0),
        facing=0.0,
        in_combat=False,
        target=None,
        enemies=[],
        events=[],
    )
    ms = MetaState(
        vector=[0.5, 0.5, 0.5, 0.5, 0.5],
        recent_events=[],
        timestamp=100.0,
    )
    snapshot = PipelineSnapshot(game_state=gs, meta_state=ms)
    assert snapshot.game_state == gs
    assert snapshot.meta_state == ms
