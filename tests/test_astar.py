"""Unit tests for A* pathfinding implementation."""

import ast
from dataclasses import FrozenInstanceError
from types import MappingProxyType

import pytest

from wow_bot.nav.astar import (
    AStarConfig,
    HeuristicKind,
    PathError,
    PathResult,
    find_path,
    find_path_through,
    heuristic_cost,
)
from wow_bot.nav.graph import GraphEdge, GraphNode, NavGraph, NodeKind


def make_graph(
    nodes: list[tuple[int, float, float, float, NodeKind, str]],
    edges: list[tuple[int, int, float]],
) -> NavGraph:
    """Helper builder returning a NavGraph matching T5.1's layout."""
    nodes_dict: dict[int, GraphNode] = {}
    for nid, x, y, z, kind, last_seen in nodes:
        nodes_dict[nid] = GraphNode(
            id=nid,
            x=x,
            y=y,
            z=z,
            kind=kind,
            last_seen_at=last_seen,
        )

    adj: dict[int, list[GraphEdge]] = {nid: [] for nid in nodes_dict}
    for from_id, to_id, cost in edges:
        if from_id in adj:
            adj[from_id].append(GraphEdge(to_id=to_id, cost=cost))

    sorted_nodes = dict(sorted(nodes_dict.items(), key=lambda item: item[0]))
    sorted_adj: dict[int, tuple[GraphEdge, ...]] = {}
    for nid in sorted_nodes:
        edges_list = adj.get(nid, [])
        edges_list.sort(key=lambda e: e.to_id)
        sorted_adj[nid] = tuple(edges_list)

    return NavGraph(
        nodes=MappingProxyType(sorted_nodes),
        adjacency=MappingProxyType(sorted_adj),
    )


def test_astar_config_validation() -> None:
    """AStarConfig validation raises ValueError on non-positive bounds."""
    with pytest.raises(ValueError, match="max_expansions must be >= 1"):
        AStarConfig(max_expansions=0)

    with pytest.raises(ValueError, match="max_path_length must be >= 1"):
        AStarConfig(max_path_length=0)


def test_astar_config_frozen() -> None:
    """AStarConfig is frozen."""
    cfg = AStarConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.max_expansions = 50  # type: ignore[misc]


def test_path_result_invariants() -> None:
    """PathResult validates constructor invariants and raises ValueError."""
    # found=True with empty node_ids
    with pytest.raises(ValueError, match="found == True implies len\\(node_ids\\) >= 1"):
        PathResult(found=True, node_ids=(), total_cost=0.0, expansions=0)

    # found=True with negative cost
    with pytest.raises(ValueError, match="found == True implies total_cost >= 0.0"):
        PathResult(found=True, node_ids=(1,), total_cost=-1.0, expansions=0)

    # found=True with non-empty reason
    with pytest.raises(ValueError, match="found == True implies reason == ''"):
        PathResult(found=True, node_ids=(1,), total_cost=1.0, expansions=0, reason="err")

    # found=False with non-empty node_ids
    with pytest.raises(ValueError, match="found == False implies node_ids == \\(\\)"):
        PathResult(found=False, node_ids=(1,), total_cost=0.0, expansions=0, reason="failed")

    # found=False with non-zero total_cost
    with pytest.raises(ValueError, match="found == False implies total_cost == 0.0"):
        PathResult(found=False, node_ids=(), total_cost=5.0, expansions=0, reason="failed")

    # found=False with empty reason
    with pytest.raises(ValueError, match="found == False implies reason is a non-empty string"):
        PathResult(found=False, node_ids=(), total_cost=0.0, expansions=0, reason="")


def test_path_result_frozen() -> None:
    """PathResult is frozen."""
    res = PathResult(found=True, node_ids=(1,), total_cost=0.0, expansions=0)
    with pytest.raises(FrozenInstanceError):
        res.found = False  # type: ignore[misc]


def test_find_path_start_not_in_graph() -> None:
    """find_path returns reason='start_not_in_graph' when start_id missing."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    res = find_path(graph, start_id=99, goal_id=1)
    assert not res.found
    assert res.reason == "start_not_in_graph"
    assert res.expansions >= 0


def test_find_path_goal_not_in_graph() -> None:
    """find_path returns reason='goal_not_in_graph' when goal_id missing."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    res = find_path(graph, start_id=1, goal_id=99)
    assert not res.found
    assert res.reason == "goal_not_in_graph"
    assert res.expansions >= 0


def test_find_path_same_start_and_goal() -> None:
    """find_path with start_id == goal_id returns cost 0 and 0 expansions."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    res = find_path(graph, start_id=1, goal_id=1)
    assert res.found
    assert res.node_ids == (1,)
    assert res.total_cost == 0.0
    assert res.expansions == 0


def test_find_path_two_node_graph() -> None:
    """find_path on two-node graph A->B returns path and correct edge cost."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 10.0)]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=2)
    assert res.found
    assert res.node_ids == (1, 2)
    assert res.total_cost == 10.0
    assert res.expansions >= 0


def test_find_path_linear_chain() -> None:
    """find_path on chain A->B->C->D returns full chain and cumulative cost."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 15.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (2, 3, 5.0),
        (3, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=4)
    assert res.found
    assert res.node_ids == (1, 2, 3, 4)
    assert res.total_cost == 15.0
    assert res.expansions >= 0


def test_find_path_diamond_equal_cost_tiebreak() -> None:
    """Equal cost paths in diamond graph tie-break by smaller intermediate node id."""
    # Nodes: 1 (start), 2, 3, 4 (goal)
    # Both 1->2->4 and 1->3->4 have total cost 10.0
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (1, 3, 5.0),
        (2, 4, 5.0),
        (3, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=4)
    assert res.found
    assert res.node_ids == (1, 2, 4)
    assert res.total_cost == 10.0


def test_find_path_diamond_unequal_cost() -> None:
    """Diamond graph selects the strictly cheaper path."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    # Path via 3 is cheaper (3.0 + 3.0 = 6.0) vs via 2 (5.0 + 5.0 = 10.0)
    edges = [
        (1, 2, 5.0),
        (1, 3, 3.0),
        (2, 4, 5.0),
        (3, 4, 3.0),
    ]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=4)
    assert res.found
    assert res.node_ids == (1, 3, 4)
    assert res.total_cost == 6.0


def test_find_path_unreachable() -> None:
    """Unreachable goal returns found=False with reason='unreachable'."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges: list[tuple[int, int, float]] = []
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=2)
    assert not res.found
    assert res.reason == "unreachable"
    assert res.expansions >= 0


def test_find_path_one_directional_edge() -> None:
    """One-directional edge A->B does not allow path B->A."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 5.0)]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=2, goal_id=1)
    assert not res.found
    assert res.reason == "unreachable"


def test_find_path_cycle_handling() -> None:
    """Search terminates correctly and finds optimal path on cyclic graph."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 2.0),
        (2, 1, 2.0),
        (2, 3, 2.0),
        (3, 2, 2.0),
        (1, 3, 10.0),
    ]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=3)
    assert res.found
    assert res.node_ids == (1, 2, 3)
    assert res.total_cost == 4.0


def test_find_path_self_loop_ignored() -> None:
    """Self-loop edge A->A does not cause infinite expansion loop."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 1, 1.0),
        (1, 2, 5.0),
    ]
    graph = make_graph(nodes, edges)

    res = find_path(graph, start_id=1, goal_id=2)
    assert res.found
    assert res.node_ids == (1, 2)
    assert res.total_cost == 5.0


def test_find_path_max_expansions_exceeded() -> None:
    """Exceeding max_expansions returns found=False with reason='max_expansions_exceeded'."""
    # Create a chain of 10 nodes: 1->2->3->...->10
    nodes = [(i, float(i), 0.0, 0.0, NodeKind.WAYPOINT, "t") for i in range(1, 11)]
    edges = [(i, i + 1, 1.0) for i in range(1, 10)]
    graph = make_graph(nodes, edges)

    cfg = AStarConfig(max_expansions=3)
    res = find_path(graph, start_id=1, goal_id=10, config=cfg)
    assert not res.found
    assert res.reason == "max_expansions_exceeded"
    assert res.expansions > 3


def test_find_path_max_path_length_exceeded() -> None:
    """Exceeding max_path_length returns found=False with reason='max_path_length_exceeded'."""
    nodes = [(i, float(i), 0.0, 0.0, NodeKind.WAYPOINT, "t") for i in range(1, 6)]
    edges = [(i, i + 1, 1.0) for i in range(1, 5)]
    graph = make_graph(nodes, edges)

    cfg = AStarConfig(max_path_length=3)  # Path 1->2->3->4->5 has len 5
    res = find_path(graph, start_id=1, goal_id=5, config=cfg)
    assert not res.found
    assert res.reason == "max_path_length_exceeded"


def test_heuristic_zero_vs_euclidean() -> None:
    """HeuristicKind.ZERO (Dijkstra) produces the same optimal path as EUCLIDEAN."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (1, 3, 3.0),
        (2, 4, 5.0),
        (3, 4, 3.0),
    ]
    graph = make_graph(nodes, edges)

    res_euc = find_path(graph, 1, 4, config=AStarConfig(heuristic=HeuristicKind.EUCLIDEAN))
    res_zero = find_path(graph, 1, 4, config=AStarConfig(heuristic=HeuristicKind.ZERO))

    assert res_euc.found and res_zero.found
    assert res_euc.node_ids == res_zero.node_ids
    assert res_euc.total_cost == res_zero.total_cost


def test_heuristic_manhattan_vs_euclidean() -> None:
    """HeuristicKind.MANHATTAN produces optimal path on grid graph."""
    # Grid graph: (0,0)->(1,0)->(1,1) vs (0,0)->(0,1)->(1,1)
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 1.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 0.0, 1.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 1.0, 1.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 1.0),
        (1, 3, 1.0),
        (2, 4, 1.0),
        (3, 4, 1.0),
    ]
    graph = make_graph(nodes, edges)

    res_man = find_path(graph, 1, 4, config=AStarConfig(heuristic=HeuristicKind.MANHATTAN))
    res_euc = find_path(graph, 1, 4, config=AStarConfig(heuristic=HeuristicKind.EUCLIDEAN))

    assert res_man.found and res_euc.found
    assert res_man.total_cost == res_euc.total_cost
    assert res_man.node_ids == res_euc.node_ids


def test_heuristic_cost_evaluations() -> None:
    """heuristic_cost evaluates EUCLIDEAN, MANHATTAN, and ZERO accurately."""
    a = GraphNode(id=1, x=0.0, y=0.0, z=0.0, kind=NodeKind.WAYPOINT, last_seen_at="t")
    b = GraphNode(id=2, x=3.0, y=4.0, z=0.0, kind=NodeKind.WAYPOINT, last_seen_at="t")

    assert heuristic_cost(HeuristicKind.EUCLIDEAN, a, b) == 5.0
    assert heuristic_cost(HeuristicKind.MANHATTAN, a, b) == 7.0
    assert heuristic_cost(HeuristicKind.ZERO, a, b) == 0.0


def test_find_path_through_empty_waypoints() -> None:
    """find_path_through with empty waypoints raises PathError."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    with pytest.raises(PathError, match="waypoints tuple cannot be empty"):
        find_path_through(graph, ())


def test_find_path_through_single_waypoint() -> None:
    """find_path_through with single waypoint returns (waypoint,) with cost 0."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    res = find_path_through(graph, (1,))
    assert res.found
    assert res.node_ids == (1,)
    assert res.total_cost == 0.0
    assert res.expansions == 0


def test_find_path_through_concatenation_and_deduplication() -> None:
    """find_path_through concatenates legs and deduplicates shared endpoints."""
    # Waypoints: 1, 3, 5
    # Path 1->3: 1->2->3
    # Path 3->5: 3->4->5
    nodes = [(i, float(i), 0.0, 0.0, NodeKind.WAYPOINT, "t") for i in range(1, 6)]
    edges = [(i, i + 1, 2.0) for i in range(1, 5)]
    graph = make_graph(nodes, edges)

    res = find_path_through(graph, (1, 3, 5))
    assert res.found
    assert res.node_ids == (1, 2, 3, 4, 5)
    assert res.total_cost == 8.0
    assert res.expansions >= 0


def test_find_path_through_failing_leg() -> None:
    """find_path_through returns leg_failed error reason if a leg cannot be traversed."""
    # Graph has 1->2, but no path from 2->3
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 1.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 2.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 1.0)]
    graph = make_graph(nodes, edges)

    res = find_path_through(graph, (1, 2, 3))
    assert not res.found
    assert res.reason.startswith("leg_failed:1:")
    assert "unreachable" in res.reason


def test_determinism_across_runs() -> None:
    """100 consecutive find_path calls on same inputs produce identical PathResult."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (1, 3, 5.0),
        (2, 4, 5.0),
        (3, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    first = find_path(graph, 1, 4)
    for _ in range(100):
        current = find_path(graph, 1, 4)
        assert current == first


def test_expansions_non_negative_int() -> None:
    """Expansions is a non-negative int for every successful or failed search."""
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph_unreachable = make_graph(nodes, [])
    graph_reachable = make_graph(nodes, [(1, 2, 10.0)])

    res1 = find_path(graph_unreachable, 1, 2)
    assert isinstance(res1.expansions, int)
    assert res1.expansions >= 0

    res2 = find_path(graph_reachable, 1, 2)
    assert isinstance(res2.expansions, int)
    assert res2.expansions >= 0


def test_static_ast_import_isolation() -> None:
    """Verify static import safety of astar.py using AST analysis."""
    with open("src/wow_bot/nav/astar.py", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="src/wow_bot/nav/astar.py")

    prohibited_exact = {
        "wow_bot.world",
        "wow_bot.world.store",
        "wow_bot.world.schema",
        "wow_bot.strategist",
        "wow_bot.navigation.navigator",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in prohibited_exact, f"Prohibited module imported: {mod}"
                assert not any(
                    kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
                ), f"Prohibited LLM module imported: {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert mod not in prohibited_exact, f"Prohibited module imported: {mod}"
            assert not any(
                kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
            ), f"Prohibited LLM module imported: {mod}"
