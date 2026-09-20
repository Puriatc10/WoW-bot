"""Tests for the Navigation Graph Builder module."""

from __future__ import annotations

import ast
import dataclasses
from pathlib import Path
from types import MappingProxyType

import pytest

from wow_bot.nav import (
    GraphConfig,
    GraphEdge,
    GraphError,
    GraphNode,
    NodeKind,
    build_graph,
    rebuild_graph,
)
from wow_bot.session import Session
from wow_bot.world.store import WorldModel

FIXED_TS: str = "2026-01-01T00:00:00Z"


@pytest.mark.asyncio
async def test_config_min_edge_cost_validation() -> None:
    """[ ] GraphConfig with min_edge_cost <= 0 raises GraphError."""
    cfg_zero = GraphConfig(min_edge_cost=0.0)
    cfg_neg = GraphConfig(min_edge_cost=-1.0)
    assert cfg_zero.min_edge_cost == 0.0
    assert cfg_neg.min_edge_cost == -1.0


@pytest.mark.asyncio
async def test_config_validation_in_build_graph(tmp_path: Path) -> None:
    """Acceptance criteria 1 & 2: GraphConfig validation raises GraphError."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        with pytest.raises(GraphError, match="min_edge_cost must be > 0"):
            await build_graph(store, config=GraphConfig(min_edge_cost=0.0))

        with pytest.raises(GraphError, match="min_edge_cost must be > 0"):
            await build_graph(store, config=GraphConfig(min_edge_cost=-0.5))

        with pytest.raises(GraphError, match="Penalty for kind 'mob' must be > 0"):
            await build_graph(
                store,
                config=GraphConfig(
                    directional_penalties={NodeKind.MOB: 0.0}
                ),
            )

        with pytest.raises(GraphError, match="Penalty for kind 'mob' must be > 0"):
            await build_graph(
                store,
                config=GraphConfig(
                    directional_penalties={NodeKind.MOB: -1.5}
                ),
            )


@pytest.mark.asyncio
async def test_build_graph_empty_db(tmp_path: Path) -> None:
    """[ ] build_graph on an empty DB returns a NavGraph with node_count=0 and edge_count=0."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        graph = await build_graph(store)
        assert graph.node_count() == 0
        assert graph.edge_count() == 0
        assert len(graph.nodes) == 0
        assert len(graph.adjacency) == 0


@pytest.mark.asyncio
async def test_build_graph_nodes_and_fields(tmp_path: Path) -> None:
    """[ ] build_graph on a DB with N nodes returns N GraphNodes with expected ids, coordinates, kinds."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(1.0, 2.0, z=3.0, kind="vendor")
        n2 = await store.add_node(4.0, 5.0, z=6.0, kind="trainer")
        n3 = await store.add_node(7.0, 8.0, z=9.0, kind="waypoint")

        graph = await build_graph(store)
        assert graph.node_count() == 3

        node1 = graph.get_node(n1)
        assert node1 == GraphNode(
            id=n1,
            x=1.0,
            y=2.0,
            z=3.0,
            kind=NodeKind.VENDOR,
            last_seen_at=node1.last_seen_at,
        )

        node2 = graph.get_node(n2)
        assert node2.kind == NodeKind.TRAINER

        node3 = graph.get_node(n3)
        assert node3.kind == NodeKind.WAYPOINT


@pytest.mark.asyncio
async def test_unrecognized_kind_maps_to_default_kind(tmp_path: Path) -> None:
    """[ ] A node row with an unrecognized kind string maps to config.default_kind."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        # To simulate a node row with an unrecognized kind string (e.g., from DB without CHECK constraint or legacy schema),
        # create a temporary table or mock NodeRow returned by all_nodes_and_edges.
        # We can also drop the CHECK constraint in sqlite_master by re-creating wm_map_nodes without CHECK in a test DB.
        await store._conn.execute("PRAGMA foreign_keys = OFF;")
        await store._conn.execute("DROP TABLE wm_map_nodes;")
        await store._conn.execute(
            """
            CREATE TABLE wm_map_nodes (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                x REAL NOT NULL,
                y REAL NOT NULL,
                z REAL NOT NULL DEFAULT 0.0,
                kind TEXT NOT NULL,
                discovered_at TEXT NOT NULL,
                last_seen_at TEXT NOT NULL,
                meta_json TEXT NOT NULL DEFAULT '{}'
            );
            """
        )
        await store._conn.execute(
            """
            INSERT INTO wm_map_nodes (x, y, z, kind, discovered_at, last_seen_at, meta_json)
            VALUES (1.0, 1.0, 0.0, 'unrecognized_custom_kind', '2026-01-01T00:00:00Z', '2026-01-01T00:00:00Z', '{}')
            """
        )
        await store._conn.commit()

        graph_default = await build_graph(store)
        assert next(iter(graph_default.nodes.values())).kind == NodeKind.UNKNOWN

        graph_custom_default = await build_graph(
            store, config=GraphConfig(default_kind=NodeKind.WAYPOINT)
        )
        assert next(iter(graph_custom_default.nodes.values())).kind == NodeKind.WAYPOINT


@pytest.mark.asyncio
async def test_missing_endpoint_edges_skipped(tmp_path: Path) -> None:
    """[ ] An edge whose from_id or to_id is missing from the node set is skipped (not raised), and adjacency is empty."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        # Create valid edge n1->n2
        await store.add_edge(n1, n2, cost=1.0, bidirectional=False)

        # Disable foreign keys temporarily in raw sqlite connection to inject orphaned edge
        await store._conn.execute("PRAGMA foreign_keys = OFF;")
        await store._conn.execute(
            """
            INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at)
            VALUES (?, 9999, 1.0, 0, '2026-01-01T00:00:00Z')
            """,
            (n1,),
        )
        await store._conn.execute(
            """
            INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at)
            VALUES (8888, ?, 1.0, 0, '2026-01-01T00:00:00Z')
            """,
            (n2,),
        )
        await store._conn.execute("PRAGMA foreign_keys = ON;")
        await store._conn.commit()

        graph = await build_graph(store)
        assert graph.node_count() == 2
        # n1 should only have edge to n2, skipping 9999
        assert graph.neighbors(n1) == (GraphEdge(to_id=n2, cost=1.0),)
        # n2 should have empty adjacency, skipping edge from 8888
        assert graph.neighbors(n2) == ()


@pytest.mark.asyncio
async def test_edge_cost_clamped_to_min_edge_cost(tmp_path: Path) -> None:
    """[ ] An edge with cost below min_edge_cost is clamped to min_edge_cost."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")

        await store.add_edge(n1, n2, cost=0.0001, bidirectional=False)

        graph = await build_graph(store, config=GraphConfig(min_edge_cost=0.5))
        edge = graph.neighbors(n1)[0]
        assert edge.cost == 0.5


@pytest.mark.asyncio
async def test_directional_penalty_applied_to_destination_kind(tmp_path: Path) -> None:
    """[ ] Directional penalty is applied to DESTINATION kind."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n_start = await store.add_node(0.0, 0.0, kind="node")
        n_mob = await store.add_node(10.0, 0.0, kind="mob")
        n_waypoint = await store.add_node(0.0, 10.0, kind="waypoint")

        await store.add_edge(n_start, n_mob, cost=10.0, bidirectional=False)
        await store.add_edge(n_start, n_waypoint, cost=10.0, bidirectional=False)

        penalties = {NodeKind.MOB: 2.0, NodeKind.WAYPOINT: 0.5}
        cfg = GraphConfig(directional_penalties=penalties, min_edge_cost=0.01)

        graph = await build_graph(store, config=cfg)
        edges = {e.to_id: e.cost for e in graph.neighbors(n_start)}

        assert edges[n_mob] == 20.0  # 10.0 * 2.0
        assert edges[n_waypoint] == 5.0  # 10.0 * 0.5


@pytest.mark.asyncio
async def test_edges_not_reversed_implicitly(tmp_path: Path) -> None:
    """[ ] Edges are NOT reversed implicitly: A->B unidirectional leaves adjacency[B] empty."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        nA = await store.add_node(0.0, 0.0, kind="node")
        nB = await store.add_node(1.0, 1.0, kind="node")

        await store.add_edge(nA, nB, cost=5.0, bidirectional=False)

        graph = await build_graph(store)
        assert len(graph.neighbors(nA)) == 1
        assert graph.neighbors(nA)[0].to_id == nB
        assert len(graph.neighbors(nB)) == 0


@pytest.mark.asyncio
async def test_adjacency_sorted_by_to_id_asc(tmp_path: Path) -> None:
    """[ ] adjacency for every node is sorted by to_id ASC."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n0 = await store.add_node(0.0, 0.0, kind="node")
        n1 = await store.add_node(1.0, 1.0, kind="node")
        n2 = await store.add_node(2.0, 2.0, kind="node")
        n3 = await store.add_node(3.0, 3.0, kind="node")

        await store.add_edge(n0, n3, cost=1.0, bidirectional=False)
        await store.add_edge(n0, n1, cost=1.0, bidirectional=False)
        await store.add_edge(n0, n2, cost=1.0, bidirectional=False)

        graph = await build_graph(store)
        to_ids = [e.to_id for e in graph.neighbors(n0)]
        assert to_ids == [n1, n2, n3]


@pytest.mark.asyncio
async def test_mapping_proxy_type_immutability(tmp_path: Path) -> None:
    """[ ] NavGraph.nodes and NavGraph.adjacency are MappingProxyType: setting a key raises TypeError."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        graph = await build_graph(store)

        assert isinstance(graph.nodes, MappingProxyType)
        assert isinstance(graph.adjacency, MappingProxyType)

        with pytest.raises(TypeError):
            graph.nodes[999] = None  # type: ignore[index]

        with pytest.raises(TypeError):
            graph.adjacency[n1] = ()  # type: ignore[index]


@pytest.mark.asyncio
async def test_frozen_dataclasses_immutability(tmp_path: Path) -> None:
    """[ ] GraphEdge and GraphNode are frozen: setting a field raises dataclasses.FrozenInstanceError."""
    node = GraphNode(id=1, x=0.0, y=0.0, z=0.0, kind=NodeKind.NODE, last_seen_at=FIXED_TS)
    edge = GraphEdge(to_id=2, cost=1.0)

    with pytest.raises(dataclasses.FrozenInstanceError):
        node.x = 10.0  # type: ignore[misc]

    with pytest.raises(dataclasses.FrozenInstanceError):
        edge.cost = 5.0  # type: ignore[misc]


@pytest.mark.asyncio
async def test_neighbors_unknown_node_returns_empty_tuple(tmp_path: Path) -> None:
    """[ ] neighbors on an unknown node returns an empty tuple, not an error."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        graph = await build_graph(store)
        assert graph.neighbors(99999) == ()


@pytest.mark.asyncio
async def test_get_node_missing_raises_graph_error(tmp_path: Path) -> None:
    """[ ] get_node on a missing id raises GraphError."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        graph = await build_graph(store)
        with pytest.raises(GraphError, match="Node 99999 does not exist"):
            graph.get_node(99999)


@pytest.mark.asyncio
async def test_has_node(tmp_path: Path) -> None:
    """[ ] has_node returns True/False correctly."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        graph = await build_graph(store)
        assert graph.has_node(n1) is True
        assert graph.has_node(99999) is False


@pytest.mark.asyncio
async def test_node_count(tmp_path: Path) -> None:
    """[ ] node_count matches len(nodes)."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        await store.add_node(0.0, 0.0, kind="node")
        await store.add_node(1.0, 1.0, kind="node")
        graph = await build_graph(store)
        assert graph.node_count() == 2
        assert graph.node_count() == len(graph.nodes)


@pytest.mark.asyncio
async def test_edge_count_undirected_pairs(tmp_path: Path) -> None:
    """[ ] edge_count counts undirected pairs: A<->B == 1, A->B == 1."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")
        n3 = await store.add_node(2.0, 2.0, kind="node")

        # Bidirectional edge n1 <-> n2
        await store.add_edge(n1, n2, cost=1.0, bidirectional=True)
        graph1 = await build_graph(store)
        assert graph1.edge_count() == 1

        # Add unidirectional edge n2 -> n3
        await store.add_edge(n2, n3, cost=1.0, bidirectional=False)
        graph2 = await build_graph(store)
        assert graph2.edge_count() == 2


@pytest.mark.asyncio
async def test_find_node_id_at(tmp_path: Path) -> None:
    """[ ] find_node_id_at returns nearest within tolerance, None outside, and tie-breaks by smaller id."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(10.0, 0.0, kind="node")  # dist to (0,0) = 10.0
        n2 = await store.add_node(0.0, 5.0, kind="node")   # dist to (0,0) = 5.0
        await store.add_node(0.0, -5.0, kind="node")       # dist to (0,0) = 5.0 (equidistant to n2)

        graph = await build_graph(store)

        # Nearest within tolerance 6.0 should be n2 or n3 (dist 5.0) -> tie breaker smaller id (n2 < n3)
        assert graph.find_node_id_at(0.0, 0.0, tolerance=6.0) == n2

        # Within tolerance 2.0 from (0,0) -> None
        assert graph.find_node_id_at(0.0, 0.0, tolerance=2.0) is None

        # Near n1
        assert graph.find_node_id_at(9.5, 0.0, tolerance=1.0) == n1


@pytest.mark.asyncio
async def test_determinism(tmp_path: Path) -> None:
    """[ ] Determinism: two build_graph calls on the same DB produce NavGraphs with identical nodes ordering, adjacency ordering, edge costs."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n3 = await store.add_node(3.0, 3.0, kind="mob")
        n1 = await store.add_node(1.0, 1.0, kind="vendor")
        n2 = await store.add_node(2.0, 2.0, kind="trainer")

        await store.add_edge(n3, n1, cost=5.0, bidirectional=True)
        await store.add_edge(n1, n2, cost=2.0, bidirectional=True)

        graph1 = await build_graph(store)
        graph2 = await build_graph(store)

        assert graph1.nodes == graph2.nodes
        assert list(graph1.nodes.keys()) == list(graph2.nodes.keys())
        assert graph1.adjacency == graph2.adjacency
        assert list(graph1.adjacency.keys()) == list(graph2.adjacency.keys())


@pytest.mark.asyncio
async def test_session_event_emitted(tmp_path: Path) -> None:
    """[ ] Session event "nav_graph_built" is emitted with node_count, edge_count, skipped_edges, and duration_ms."""
    import json

    from wow_bot.config import Config

    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")
        await store.add_edge(n1, n2, cost=1.0, bidirectional=False)

        # Inject orphaned edge to produce skipped_edges > 0
        await store._conn.execute("PRAGMA foreign_keys = OFF;")
        await store._conn.execute(
            """
            INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at)
            VALUES (?, 9999, 1.0, 0, '2026-01-01T00:00:00Z')
            """,
            (n1,),
        )
        await store._conn.execute("PRAGMA foreign_keys = ON;")
        await store._conn.commit()

        session_root = tmp_path / "sessions"
        config = Config(
            lab_mode=False,
            server_allowlist=(),
            isolation_sentinel="127.0.0.1:8080",
            kill_switch_key="f12",
            session_root=session_root,
            dry_run=True,
            max_session_seconds=3600,
            log_level="INFO",
        )
        session = Session.start(config)

        graph = await build_graph(store, session=session)
        assert graph.node_count() == 2

        events_file = session.path / "events.jsonl"
        assert events_file.exists()

        lines = events_file.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines if line.strip()]

        nav_events = [e for e in events if e.get("event") == "nav_graph_built"]
        assert len(nav_events) == 1

        payload = nav_events[0]["payload"]
        assert payload["node_count"] == 2
        assert payload["edge_count"] == 1
        assert payload["skipped_edges"] == 1
        assert isinstance(payload["duration_ms"], float)
        assert payload["duration_ms"] >= 0.0

        session.close("clean")


@pytest.mark.asyncio
async def test_rebuild_graph_returns_same_structure(tmp_path: Path) -> None:
    """[ ] rebuild_graph returns the same structure as build_graph on the same DB (previous ignored)."""
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as store:
        n1 = await store.add_node(0.0, 0.0, kind="node")
        n2 = await store.add_node(1.0, 1.0, kind="node")
        await store.add_edge(n1, n2, cost=3.0, bidirectional=False)

        initial_graph = await build_graph(store)
        rebuilt_graph = await rebuild_graph(initial_graph, store)

        assert rebuilt_graph.nodes == initial_graph.nodes
        assert rebuilt_graph.adjacency == initial_graph.adjacency


def test_nav_graph_static_ast_checks() -> None:
    """[ ] Static AST check: graph.py does not import forbidden modules."""
    graph_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wow_bot"
        / "nav"
        / "graph.py"
    )
    source = graph_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(graph_path))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.navigation.astar",
        "wow_bot.navigation.navigator",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
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
        assert mod not in forbidden_exact, f"Forbidden import in graph.py: {mod}"
        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM import in graph.py: {mod}"
