"""Unit and contract tests for GameState -> World sync layer."""

import ast
import asyncio
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from wow_bot.config import Config
from wow_bot.session import Session
from wow_bot.world.store import WorldModel
from wow_bot.world.sync import SyncConfig, SyncStats, WorldSync


@dataclass(frozen=True)
class FakeEntity:
    entity_id: str
    kind: str
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class FakeEntityWithExtras:
    entity_id: str
    kind: str
    x: float
    y: float
    z: float = 0.0
    extra_field1: str = "secret"
    extra_field2: int = 12345


@dataclass(frozen=True)
class FakeGameState:
    player_x: float = 100.0
    player_y: float = 200.0
    player_z: float = 10.0
    entities: tuple[Any, ...] = field(default_factory=tuple)
    target_entity_id: str | None = None


def make_test_session(tmp_path: Path) -> Session:
    """Construct a real Session rooted in tmp_path."""
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


# Acceptance 1
def test_sync_config_invalid_player_node_kind() -> None:
    with pytest.raises(ValueError, match="Invalid player_node_kind"):
        SyncConfig(player_node_kind="invalid_kind")


# Acceptance 2
def test_sync_config_non_positive_min_distance() -> None:
    with pytest.raises(ValueError, match="player_node_min_distance_units must be > 0"):
        SyncConfig(player_node_min_distance_units=0.0)
    with pytest.raises(ValueError, match="player_node_min_distance_units must be > 0"):
        SyncConfig(player_node_min_distance_units=-1.0)


# Acceptance 3
def test_sync_config_negative_entity_meta_max_keys() -> None:
    with pytest.raises(ValueError, match="entity_meta_max_keys must be >= 0"):
        SyncConfig(entity_meta_max_keys=-1)


# Acceptance 4
@pytest.mark.asyncio
async def test_first_sync_once_creates_player_node(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        state = FakeGameState(player_x=10.0, player_y=20.0, player_z=5.0)

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")

        assert stats.player_node_created is True
        assert stats.player_node_updated is False
        assert await world.count_nodes("waypoint") == 1


# Acceptance 5
@pytest.mark.asyncio
async def test_second_sync_once_same_position_updates_node(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        state = FakeGameState(player_x=10.0, player_y=20.0, player_z=5.0)

        stats1 = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats1.player_node_created is True

        stats2 = await sync.sync_once(state, now="2026-01-01T00:01:00Z")
        assert stats2.player_node_created is False
        assert stats2.player_node_updated is True
        assert await world.count_nodes("waypoint") == 1


# Acceptance 6
@pytest.mark.asyncio
async def test_second_sync_once_within_radius_reuses_node(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        config = SyncConfig(player_node_min_distance_units=5.0)
        sync = WorldSync(world, config=config)

        state1 = FakeGameState(player_x=10.0, player_y=20.0)
        await sync.sync_once(state1, now="2026-01-01T00:00:00Z")

        state2 = FakeGameState(player_x=12.0, player_y=22.0)  # distance = sqrt(4+4) ~= 2.83 < 5.0
        stats2 = await sync.sync_once(state2, now="2026-01-01T00:01:00Z")

        assert stats2.player_node_created is False
        assert stats2.player_node_updated is True
        assert await world.count_nodes("waypoint") == 1


# Acceptance 7
@pytest.mark.asyncio
async def test_second_sync_once_beyond_radius_creates_new_node(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        config = SyncConfig(player_node_min_distance_units=5.0)
        sync = WorldSync(world, config=config)

        state1 = FakeGameState(player_x=10.0, player_y=20.0)
        await sync.sync_once(state1, now="2026-01-01T00:00:00Z")

        state2 = FakeGameState(player_x=20.0, player_y=20.0)  # distance = 10.0 > 5.0
        stats2 = await sync.sync_once(state2, now="2026-01-01T00:01:00Z")

        assert stats2.player_node_created is True
        assert stats2.player_node_updated is False
        assert await world.count_nodes("waypoint") == 2


# Acceptance 8
@pytest.mark.asyncio
async def test_entities_upserted(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        entity = FakeEntity(entity_id="mob_1", kind="mob", x=1.0, y=2.0)
        state = FakeGameState(entities=(entity,))

        stats1 = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats1.entities_upserted == 1

        stats2 = await sync.sync_once(state, now="2026-01-01T00:01:00Z")
        assert stats2.entities_upserted == 1

        row = await world.get_entity("mob_1")
        assert row is not None
        assert row.kind == "mob"


# Acceptance 9
@pytest.mark.asyncio
async def test_entity_empty_kind_skipped(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        entity = FakeEntity(entity_id="mob_1", kind="", x=1.0, y=2.0)
        state = FakeGameState(entities=(entity,))

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats.entities_skipped == 1
        assert stats.entities_upserted == 0
        assert await world.get_entity("mob_1") is None


# Acceptance 10
@pytest.mark.asyncio
async def test_entity_non_string_kind_skipped(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        entity = FakeEntity(entity_id="mob_1", kind=123, x=1.0, y=2.0)  # type: ignore[arg-type]
        state = FakeGameState(entities=(entity,))

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats.entities_skipped == 1
        assert stats.entities_upserted == 0
        assert await world.get_entity("mob_1") is None


# Acceptance 11
@pytest.mark.asyncio
async def test_entity_meta_max_keys_truncation_no_leak(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        entity = FakeEntityWithExtras(
            entity_id="mob_1",
            kind="mob",
            x=1.0,
            y=2.0,
            extra_field1="secret",
            extra_field2=12345,
        )
        state = FakeGameState(entities=(entity,))

        await sync.sync_once(state, now="2026-01-01T00:00:00Z")

        row = await world.get_entity("mob_1")
        assert row is not None
        meta = json.loads(row.meta_json)
        assert set(meta.keys()) == {"kind", "x", "y", "z"}
        assert "extra_field1" not in meta
        assert "extra_field2" not in meta


# Acceptance 12
@pytest.mark.asyncio
async def test_target_seen_when_matched(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        e1 = FakeEntity(entity_id="mob_1", kind="mob", x=1.0, y=2.0)
        e2 = FakeEntity(entity_id="mob_2", kind="mob", x=3.0, y=4.0)
        state = FakeGameState(entities=(e1, e2), target_entity_id="mob_2")

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats.target_seen is True


# Acceptance 13
@pytest.mark.asyncio
async def test_target_seen_false_when_none_or_unmatched(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        e1 = FakeEntity(entity_id="mob_1", kind="mob", x=1.0, y=2.0)

        state1 = FakeGameState(entities=(e1,), target_entity_id=None)
        stats1 = await sync.sync_once(state1, now="2026-01-01T00:00:00Z")
        assert stats1.target_seen is False

        state2 = FakeGameState(entities=(e1,), target_entity_id="mob_999")
        stats2 = await sync.sync_once(state2, now="2026-01-01T00:01:00Z")
        assert stats2.target_seen is False


# Acceptance 14
@pytest.mark.asyncio
async def test_session_attached_emits_world_sync_event(tmp_path: Path) -> None:
    session = make_test_session(tmp_path)

    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world, session=session)
        state = FakeGameState()

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")

        events_file = session.path / "events.jsonl"
        lines = [json.loads(line) for line in events_file.read_text().splitlines()]
        sync_events = [e for e in lines if e.get("event") == "world_sync"]

        assert len(sync_events) == 1
        assert sync_events[0]["stats"] == stats.to_json()


# Acceptance 15
@pytest.mark.asyncio
async def test_no_session_attached_works(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world, session=None)
        state = FakeGameState()

        stats = await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        assert stats.player_node_created is True


# Acceptance 16
@pytest.mark.asyncio
async def test_run_periodic_max_iterations(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    call_count = 0

    def state_source() -> FakeGameState:
        nonlocal call_count
        call_count += 1
        return FakeGameState(player_x=10.0 + call_count * 10.0)

    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        stats_list = await sync.run_periodic(
            state_source,
            rate_hz=10.0,
            max_iterations=3,
        )

        assert call_count == 3
        assert len(stats_list) == 3


# Acceptance 17
@pytest.mark.asyncio
async def test_run_periodic_invalid_rate_hz(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        with pytest.raises(ValueError, match="rate_hz must be in range"):
            await sync.run_periodic(FakeGameState, rate_hz=0.0)
        with pytest.raises(ValueError, match="rate_hz must be in range"):
            await sync.run_periodic(FakeGameState, rate_hz=11.0)


# Acceptance 18
@pytest.mark.asyncio
async def test_run_periodic_invalid_max_iterations(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        with pytest.raises(ValueError, match="max_iterations must be >= 1"):
            await sync.run_periodic(FakeGameState, max_iterations=0)


# Acceptance 19
@pytest.mark.asyncio
async def test_run_periodic_stop_event_early(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    stop_evt = asyncio.Event()
    call_count = 0

    def state_source() -> FakeGameState:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            stop_evt.set()
        return FakeGameState(player_x=10.0)

    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        stats_list = await sync.run_periodic(
            state_source,
            rate_hz=10.0,
            stop_event=stop_evt,
            max_iterations=10,
        )

        assert call_count == 2
        assert len(stats_list) == 2


# Acceptance 20
@pytest.mark.asyncio
async def test_run_periodic_propagates_exceptions(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"

    def state_source() -> FakeGameState:
        raise RuntimeError("State source error")

    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        with pytest.raises(RuntimeError, match="State source error"):
            await sync.run_periodic(state_source, rate_hz=10.0, max_iterations=3)


# Acceptance 21
@pytest.mark.asyncio
async def test_idempotency_identical_gamestate(tmp_path: Path) -> None:
    db_path = tmp_path / "world.db"
    async with await WorldModel.open(db_path) as world:
        sync = WorldSync(world)
        e1 = FakeEntity(entity_id="mob_1", kind="mob", x=1.0, y=2.0)
        e2 = FakeEntity(entity_id="npc_1", kind="npc", x=3.0, y=4.0)
        state = FakeGameState(player_x=50.0, player_y=50.0, entities=(e1, e2))

        await sync.sync_once(state, now="2026-01-01T00:00:00Z")
        await sync.sync_once(state, now="2026-01-01T00:01:00Z")

        assert await world.count_nodes("waypoint") == 1
        assert await world.get_entity("mob_1") is not None
        assert await world.get_entity("npc_1") is not None


# Acceptance 22
def test_sync_stats_to_json() -> None:
    stats = SyncStats(
        player_node_created=True,
        player_node_updated=False,
        entities_upserted=2,
        entities_skipped=1,
        target_seen=True,
    )
    d = stats.to_json()
    assert d == {
        "player_node_created": True,
        "player_node_updated": False,
        "entities_upserted": 2,
        "entities_skipped": 1,
        "target_seen": True,
    }
    # Verify JSON serializability
    serialized = json.dumps(d)
    assert json.loads(serialized) == d


# Acceptance 23
def test_static_ast_no_forbidden_imports() -> None:
    sync_py_path = Path("src/wow_bot/world/sync.py")
    tree = ast.parse(sync_py_path.read_text(encoding="utf-8"))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.internal_dynamics",
    }
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for f_exact in forbidden_exact:
                    assert not mod.startswith(f_exact), f"Forbidden import: {mod}"
                for f_sub in forbidden_substrings:
                    assert f_sub not in mod.lower(), f"Forbidden import containing '{f_sub}': {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            for f_exact in forbidden_exact:
                assert not mod.startswith(f_exact), f"Forbidden import from: {mod}"
            for f_sub in forbidden_substrings:
                assert f_sub not in mod.lower(), f"Forbidden import containing '{f_sub}': {mod}"
