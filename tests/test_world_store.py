"""Tests for the World Model Async Store API."""

from __future__ import annotations

import ast
import sqlite3
import time
from pathlib import Path

import pytest

from wow_bot.world.store import (
    EdgeRow,
    EntityRow,
    NodeRow,
    WorldModel,
    WorldStoreError,
)

FIXED_TS: str = "2026-01-01T00:00:00Z"
FIXED_TS_2: str = "2026-01-02T12:00:00Z"


@pytest.mark.asyncio
async def test_open_succeeds_and_creates_db_file(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    assert not db_file.exists()
    store = await WorldModel.open(db_file)
    try:
        assert db_file.exists()
    finally:
        await store.close()


@pytest.mark.asyncio
async def test_open_applies_migrations(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        cnt = await store.count_nodes()
        assert cnt == 0


@pytest.mark.asyncio
async def test_open_is_idempotent(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    store1 = await WorldModel.open(db_file)
    await store1.add_node(1.0, 2.0, kind="waypoint")
    await store1.close()

    store2 = await WorldModel.open(db_file)
    try:
        assert await store2.count_nodes() == 1
    finally:
        await store2.close()


@pytest.mark.asyncio
async def test_close_is_idempotent(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    store = await WorldModel.open(db_file)
    await store.close()
    await store.close()


@pytest.mark.asyncio
async def test_async_context_manager_closes_on_exit(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.add_node(0.0, 0.0, kind="node")

    with pytest.raises(ValueError):
        await store.count_nodes()


@pytest.mark.asyncio
async def test_add_node_returns_increasing_ids(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        id1 = await store.add_node(1.0, 2.0, kind="waypoint")
        id2 = await store.add_node(3.0, 4.0, kind="node")
        id3 = await store.add_node(5.0, 6.0, kind="vendor")
        assert id1 < id2 < id3


@pytest.mark.asyncio
async def test_add_node_invalid_kind_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        with pytest.raises(WorldStoreError, match="Invalid kind"):
            await store.add_node(1.0, 1.0, kind="invalid_kind")


@pytest.mark.asyncio
async def test_add_node_stores_canonical_json_meta(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        meta = {"b": 2, "a": 1, "unicode": "Elwynn Forest 🌲"}
        nid = await store.add_node(10.0, 20.0, kind="node", meta=meta)
        node = await store.get_node(nid)
        assert node is not None
        assert node.meta_json == '{"a": 1, "b": 2, "unicode": "Elwynn Forest 🌲"}'


@pytest.mark.asyncio
async def test_get_node_matching_fields_and_missing(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        nid = await store.add_node(1.5, 2.5, z=3.5, kind="mob", meta={"hp": 100})
        node = await store.get_node(nid)
        assert isinstance(node, NodeRow)
        assert node.id == nid
        assert node.x == 1.5
        assert node.y == 2.5
        assert node.z == 3.5
        assert node.kind == "mob"
        assert isinstance(node.discovered_at, str)
        assert isinstance(node.last_seen_at, str)
        assert node.meta_json == '{"hp": 100}'

        missing = await store.get_node(9999)
        assert missing is None


@pytest.mark.asyncio
async def test_update_node_last_seen(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        nid = await store.add_node(1.0, 1.0, kind="trainer")
        node_before = await store.get_node(nid)
        assert node_before is not None

        updated = await store.update_node_last_seen(nid, now=FIXED_TS)
        assert updated is True

        node_after = await store.get_node(nid)
        assert node_after is not None
        assert node_after.last_seen_at == FIXED_TS

        missing_updated = await store.update_node_last_seen(8888, now=FIXED_TS)
        assert missing_updated is False


@pytest.mark.asyncio
async def test_add_edge_unidirectional(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        await store.add_edge(n1, n2, cost=10.0, bidirectional=False, now=FIXED_TS)

        edges1 = await store.edges_from(n1)
        assert len(edges1) == 1
        assert edges1[0] == EdgeRow(
            from_id=n1,
            to_id=n2,
            cost=10.0,
            bidirectional=False,
            discovered_at=FIXED_TS,
        )

        edges2 = await store.edges_from(n2)
        assert len(edges2) == 0


@pytest.mark.asyncio
async def test_add_edge_bidirectional(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        await store.add_edge(n1, n2, cost=5.0, bidirectional=True, now=FIXED_TS)

        edges1 = await store.edges_from(n1)
        assert len(edges1) == 1
        assert edges1[0].to_id == n2
        assert edges1[0].bidirectional is True

        edges2 = await store.edges_from(n2)
        assert len(edges2) == 1
        assert edges2[0].to_id == n1
        assert edges2[0].bidirectional is True


@pytest.mark.asyncio
async def test_add_edge_negative_cost_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")
        with pytest.raises(WorldStoreError, match="non-negative"):
            await store.add_edge(n1, n2, cost=-1.0)


@pytest.mark.asyncio
async def test_add_edge_self_loop_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        with pytest.raises(WorldStoreError, match="Self-loop"):
            await store.add_edge(n1, n1, cost=1.0)


@pytest.mark.asyncio
async def test_add_edge_nonexistent_node_raises_fk_violation(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        with pytest.raises(WorldStoreError, match="Foreign key"):
            await store.add_edge(n1, 999, cost=1.0)


@pytest.mark.asyncio
async def test_add_edge_replace_updates_cost_and_timestamp(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        await store.add_edge(n1, n2, cost=10.0, bidirectional=False, now=FIXED_TS)
        await store.add_edge(n1, n2, cost=2.5, bidirectional=False, now=FIXED_TS_2)

        edges = await store.edges_from(n1)
        assert len(edges) == 1
        assert edges[0].cost == 2.5
        assert edges[0].discovered_at == FIXED_TS_2


@pytest.mark.asyncio
async def test_edges_from_sorted_by_to_id_asc(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")
        n3 = await store.add_node(2.0, 2.0, kind="node")
        n4 = await store.add_node(3.0, 3.0, kind="node")

        # Add edges out of order
        await store.add_edge(n1, n4, cost=1.0, bidirectional=False)
        await store.add_edge(n1, n2, cost=1.0, bidirectional=False)
        await store.add_edge(n1, n3, cost=1.0, bidirectional=False)

        edges = await store.edges_from(n1)
        to_ids = [e.to_id for e in edges]
        assert to_ids == sorted(to_ids)
        assert to_ids == [n2, n3, n4]


@pytest.mark.asyncio
async def test_mark_seen_new_entity(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.mark_seen(
            "mob_01",
            kind="mob",
            x=10.0,
            y=20.0,
            z=5.0,
            meta={"level": 5},
            now=FIXED_TS,
        )
        ent = await store.get_entity("mob_01")
        assert isinstance(ent, EntityRow)
        assert ent.entity_id == "mob_01"
        assert ent.kind == "mob"
        assert ent.last_x == 10.0
        assert ent.last_y == 20.0
        assert ent.last_z == 5.0
        assert ent.last_seen_at == FIXED_TS
        assert ent.meta_json == '{"level": 5}'


@pytest.mark.asyncio
async def test_mark_seen_existing_entity_updates(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.mark_seen(
            "npc_vendor", kind="vendor", x=1.0, y=1.0, now=FIXED_TS
        )
        await store.mark_seen(
            "npc_vendor", kind="vendor", x=2.0, y=3.0, z=4.0, now=FIXED_TS_2
        )
        ent = await store.get_entity("npc_vendor")
        assert ent is not None
        assert ent.last_x == 2.0
        assert ent.last_y == 3.0
        assert ent.last_z == 4.0
        assert ent.last_seen_at == FIXED_TS_2


@pytest.mark.asyncio
async def test_mark_seen_merges_meta(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.mark_seen(
            "boss_1",
            kind="mob",
            x=0.0,
            y=0.0,
            meta={"a": 1, "b": 2},
            now=FIXED_TS,
        )
        # Update with new key 'c' and overriding key 'b'
        await store.mark_seen(
            "boss_1",
            kind="mob",
            x=0.0,
            y=0.0,
            meta={"b": 99, "c": 3},
            now=FIXED_TS_2,
        )
        ent = await store.get_entity("boss_1")
        assert ent is not None
        assert ent.meta_json == '{"a": 1, "b": 99, "c": 3}'


@pytest.mark.asyncio
async def test_mark_seen_malformed_json_meta_replaced(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        # Manually insert malformed JSON into DB table
        await store._conn.execute(
            """
            INSERT INTO wm_entities_seen (entity_id, kind, last_x, last_y, last_z, last_seen_at, meta_json)
            VALUES ('corrupt_1', 'unknown', 0.0, 0.0, 0.0, '2026-01-01T00:00:00Z', '{bad json...')
            """
        )
        await store._conn.commit()

        # Call mark_seen which should recover gracefully without raising
        await store.mark_seen(
            "corrupt_1",
            kind="mob",
            x=5.0,
            y=5.0,
            meta={"recovered": True},
            now=FIXED_TS_2,
        )
        ent = await store.get_entity("corrupt_1")
        assert ent is not None
        assert ent.kind == "mob"
        assert ent.meta_json == '{"recovered": true}'


@pytest.mark.asyncio
async def test_get_entity_missing_returns_none(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        ent = await store.get_entity("nonexistent_entity")
        assert ent is None


@pytest.mark.asyncio
async def test_record_route_returns_increasing_ids(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        r1 = await store.record_route(n1, n2, succeeded=True, now=FIXED_TS)
        r2 = await store.record_route(n2, n1, succeeded=False, now=FIXED_TS_2)

        assert isinstance(r1, int)
        assert isinstance(r2, int)
        assert r1 < r2


@pytest.mark.asyncio
async def test_record_combat_invalid_outcome_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        with pytest.raises(WorldStoreError, match="Invalid combat outcome"):
            await store.record_combat(
                "target_1",
                outcome="invalid_outcome",
                started_at=FIXED_TS,
                ended_at=FIXED_TS_2,
            )


@pytest.mark.asyncio
async def test_record_combat_valid_outcome_returns_id(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        for outcome in ("win", "loss", "flee", "timeout", "unknown"):
            cid = await store.record_combat(
                "mob_1",
                outcome=outcome,
                started_at=FIXED_TS,
                ended_at=FIXED_TS_2,
            )
            assert isinstance(cid, int)


@pytest.mark.asyncio
async def test_query_nearest_distance_filtering(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        # Center at (0, 0)
        n_close = await store.add_node(3.0, 4.0, kind="vendor")  # dist = 5.0
        n_exact = await store.add_node(6.0, 8.0, kind="vendor")  # dist = 10.0
        n_far = await store.add_node(9.0, 12.0, kind="vendor")  # dist = 15.0
        n_other_kind = await store.add_node(3.0, 4.0, kind="trainer")  # dist = 5.0

        res = await store.query_nearest("vendor", (0.0, 0.0), radius=10.0)
        res_ids = [n.id for n in res]
        assert n_close in res_ids
        assert n_exact in res_ids
        assert n_far not in res_ids
        assert n_other_kind not in res_ids


@pytest.mark.asyncio
async def test_query_nearest_limit(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        for i in range(15):
            await store.add_node(float(i), 0.0, kind="node")

        res = await store.query_nearest("node", (0.0, 0.0), radius=100.0, limit=5)
        assert len(res) == 5


@pytest.mark.asyncio
async def test_query_nearest_ordering_and_tie_breaking(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        # Create 2 nodes at exact same distance (5.0) from (0,0)
        # Add in reverse id order relative to coordinates to test tie-breaker
        n_id_1 = await store.add_node(3.0, 4.0, kind="mob")  # dist 5, id 1
        n_id_2 = await store.add_node(-3.0, -4.0, kind="mob")  # dist 5, id 2
        n_id_3 = await store.add_node(0.0, 1.0, kind="mob")  # dist 1, id 3

        res = await store.query_nearest("mob", (0.0, 0.0), radius=10.0)
        res_ids = [n.id for n in res]

        assert res_ids[0] == n_id_3
        # Tie-breaker for same distance (5.0): id ASC -> n_id_1 before n_id_2
        assert res_ids[1:] == [n_id_1, n_id_2]


@pytest.mark.asyncio
async def test_query_nearest_invalid_params_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        with pytest.raises(WorldStoreError, match="radius must be > 0"):
            await store.query_nearest("node", (0.0, 0.0), radius=0.0)

        with pytest.raises(WorldStoreError, match="radius must be > 0"):
            await store.query_nearest("node", (0.0, 0.0), radius=-5.0)

        with pytest.raises(WorldStoreError, match="limit must be >= 1"):
            await store.query_nearest("node", (0.0, 0.0), radius=10.0, limit=0)


@pytest.mark.asyncio
async def test_query_nearest_empty_db_returns_empty_list(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        res = await store.query_nearest("vendor", (0.0, 0.0), radius=100.0)
        assert res == []


@pytest.mark.asyncio
async def test_count_nodes(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.add_node(1.0, 1.0, kind="vendor")
        await store.add_node(2.0, 2.0, kind="vendor")
        await store.add_node(3.0, 3.0, kind="trainer")

        assert await store.count_nodes() == 3
        assert await store.count_nodes(kind="vendor") == 2
        assert await store.count_nodes(kind="trainer") == 1
        assert await store.count_nodes(kind="mob") == 0


@pytest.mark.asyncio
async def test_concurrency_gather_add_node(tmp_path: Path) -> None:
    import asyncio

    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        results = await asyncio.gather(
            store.add_node(1.0, 1.0, kind="node"),
            store.add_node(2.0, 2.0, kind="node"),
            store.add_node(3.0, 3.0, kind="node"),
        )
        assert len(results) == 3
        assert len(set(results)) == 3
        assert await store.count_nodes() == 3


@pytest.mark.asyncio
async def test_performance_query_nearest_10k_nodes(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        # Batch insert 10,000 nodes using executemany for fast setup
        nodes = [
            (
                float(i),
                float(i),
                0.0,
                "node",
                "2026-01-01T00:00:00Z",
                "2026-01-01T00:00:00Z",
                "{}",
            )
            for i in range(10000)
        ]
        await store._conn.executemany(
            """
            INSERT INTO wm_map_nodes (x, y, z, kind, discovered_at, last_seen_at, meta_json)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            nodes,
        )
        await store._conn.commit()

        # Query nearest with bounding box returning <= 100 candidates
        t0 = time.perf_counter()
        results = await store.query_nearest(
            "node", (100.0, 100.0), radius=20.0, limit=100
        )
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        assert len(results) <= 100
        # Performance threshold: query_nearest on 10,000 nodes must complete in < 10 ms (allowing up to 50 ms for CI overhead).
        assert elapsed_ms < 50.0, f"query_nearest took {elapsed_ms:.2f} ms"


@pytest.mark.asyncio
async def test_atomicity_of_open_fails_when_non_wm_table_exists(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "wm.db"

    # Simulate pre-existing corrupt DB with a non-wm_ table
    raw = sqlite3.connect(db_file)
    raw.execute("CREATE TABLE non_wm_forbidden (id INTEGER PRIMARY KEY);")
    raw.commit()
    raw.close()

    # WorldModel.open must fail without modifying or destroying the pre-existing DB state
    with pytest.raises(WorldStoreError, match="isolation"):
        await WorldModel.open(db_file)

    # Verify DB was left unchanged
    raw = sqlite3.connect(db_file)
    cursor = raw.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    tables = [r[0] for r in cursor.fetchall()]
    raw.close()

    assert tables == ["non_wm_forbidden"]


def test_store_module_static_ast_checks() -> None:
    store_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wow_bot"
        / "world"
        / "store.py"
    )
    source = store_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(store_path))

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

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for mod in imported_modules:
        assert mod not in forbidden_exact, f"Forbidden import in store.py: {mod}"
        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM import in store.py: {mod}"
