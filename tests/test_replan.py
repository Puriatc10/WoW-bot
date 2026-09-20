"""Unit tests for the pure navigation replanner in wow_bot.nav.replan."""

import ast
from dataclasses import FrozenInstanceError, dataclass
from types import MappingProxyType

import pytest

from wow_bot.nav.astar import PathResult
from wow_bot.nav.graph import GraphEdge, GraphNode, NavGraph, NodeKind
from wow_bot.nav.replan import (
    NEAR_COST_TOLERANCE,
    ReplanConfig,
    ReplanError,
    Replanner,
    ReplanOutcome,
    ReplanResult,
)


def make_graph(
    nodes: list[tuple[int, float, float, float, NodeKind, str]],
    edges: list[tuple[int, int, float]],
) -> NavGraph:
    """Helper builder returning a NavGraph directly in memory."""
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


def test_replan_config_validation_max_attempts() -> None:
    """ReplanConfig with max_attempts < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        ReplanConfig(max_attempts=0)
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        ReplanConfig(max_attempts=-1)


def test_replan_config_validation_node_snap_tolerance() -> None:
    """ReplanConfig with node_snap_tolerance_units <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="node_snap_tolerance_units must be > 0"):
        ReplanConfig(node_snap_tolerance_units=0.0)
    with pytest.raises(ValueError, match="node_snap_tolerance_units must be > 0"):
        ReplanConfig(node_snap_tolerance_units=-1.0)


def test_replan_config_validation_min_path_length() -> None:
    """ReplanConfig with min_path_length_nodes < 1 raises ValueError."""
    with pytest.raises(ValueError, match="min_path_length_nodes must be >= 1"):
        ReplanConfig(min_path_length_nodes=0)
    with pytest.raises(ValueError, match="min_path_length_nodes must be >= 1"):
        ReplanConfig(min_path_length_nodes=-2)


def test_replan_result_invariants_ok_without_path() -> None:
    """ReplanResult with OK and path=None or path.found=False raises ValueError."""
    with pytest.raises(ValueError, match="outcome == OK implies path is not None"):
        ReplanResult(
            outcome=ReplanOutcome.OK,
            path=None,
            attempt_index=0,
            reason="",
        )

    failed_path = PathResult(found=False, node_ids=(), total_cost=0.0, expansions=0, reason="failed")
    with pytest.raises(ValueError, match="outcome == OK implies path is not None and path.found is True"):
        ReplanResult(
            outcome=ReplanOutcome.OK,
            path=failed_path,
            attempt_index=0,
            reason="",
        )


def test_replan_result_invariants_non_ok_empty_reason() -> None:
    """ReplanResult with non-OK and empty reason raises ValueError."""
    with pytest.raises(ValueError, match="outcome != OK implies reason != ''"):
        ReplanResult(
            outcome=ReplanOutcome.NO_PATH,
            path=None,
            attempt_index=0,
            reason="",
        )


def test_replan_result_invariants_negative_attempt_index() -> None:
    """ReplanResult with negative attempt_index raises ValueError."""
    with pytest.raises(ValueError, match="attempt_index must be >= 0"):
        ReplanResult(
            outcome=ReplanOutcome.MAX_ATTEMPTS,
            path=None,
            attempt_index=-1,
            reason="max_attempts_exceeded",
        )


def test_replan_negative_attempt_raises_replan_error() -> None:
    """replan with attempt < 0 raises ReplanError."""
    replanner = Replanner()
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    with pytest.raises(ReplanError, match="attempt must be >= 0"):
        replanner.replan(
            from_xy=(0.0, 0.0),
            goal_xy=(0.0, 0.0),
            graph=graph,
            attempt=-1,
        )


def test_replan_attempt_exceeds_max_attempts() -> None:
    """replan with attempt >= max_attempts returns outcome=MAX_ATTEMPTS, path=None, and non-empty reason."""
    config = ReplanConfig(max_attempts=3)
    replanner = Replanner(config=config)
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(0.0, 0.0),
        graph=graph,
        attempt=3,
    )
    assert res.outcome == ReplanOutcome.MAX_ATTEMPTS
    assert res.path is None
    assert res.attempt_index == 3
    assert res.reason == "max_attempts_exceeded"


def test_replan_unsnappable_from_xy() -> None:
    """replan with unsnappable from_xy returns SNAP_FAILED with reason='start_not_snapped'."""
    config = ReplanConfig(node_snap_tolerance_units=1.0)
    replanner = Replanner(config=config)
    graph = make_graph([(1, 10.0, 10.0, 0.0, NodeKind.WAYPOINT, "t")], [])

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 10.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.SNAP_FAILED
    assert res.path is None
    assert res.reason == "start_not_snapped"


def test_replan_unsnappable_goal_xy() -> None:
    """replan with unsnappable goal_xy returns SNAP_FAILED with reason='goal_not_snapped'."""
    config = ReplanConfig(node_snap_tolerance_units=1.0)
    replanner = Replanner(config=config)
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 10.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.SNAP_FAILED
    assert res.path is None
    assert res.reason == "goal_not_snapped"


def test_replan_start_equals_goal() -> None:
    """replan when start_id == goal_id returns OK with single-node path and total_cost=0.0."""
    replanner = Replanner()
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(0.0, 0.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.found is True
    assert res.path.node_ids == (1,)
    assert res.path.total_cost == 0.0
    assert res.path.expansions == 0
    assert res.reason == ""


def test_replan_unreachable_goal() -> None:
    """replan on unreachable goal returns NO_PATH with path.found False and matching reason."""
    replanner = Replanner()
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph = make_graph(nodes, [])

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(5.0, 0.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.NO_PATH
    assert res.path is not None
    assert res.path.found is False
    assert res.reason == "unreachable"
    assert res.path.reason == "unreachable"


def test_replan_valid_graph_multinode_path() -> None:
    """replan on valid graph returns OK with multi-node path matching start and goal endpoints."""
    replanner = Replanner()
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 5.0), (2, 3, 5.0)]
    graph = make_graph(nodes, edges)

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.found is True
    assert res.path.node_ids == (1, 2, 3)
    assert res.path.node_ids[0] == 1
    assert res.path.node_ids[-1] == 3
    assert res.path.total_cost == 10.0
    assert res.reason == ""


def test_replan_min_path_length_too_short() -> None:
    """replan with min_path_length_nodes=3 on a 2-node path returns PATH_TOO_SHORT."""
    config = ReplanConfig(min_path_length_nodes=3)
    replanner = Replanner(config=config)
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 5.0)]
    graph = make_graph(nodes, edges)

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(5.0, 0.0),
        graph=graph,
        attempt=0,
    )
    assert res.outcome == ReplanOutcome.PATH_TOO_SHORT
    assert res.path is not None
    assert res.path.found is True
    assert res.reason == "path_length=2"


def test_replan_prefer_different_first_hop_success() -> None:
    """replan with prefer_different_first_hop=True selects alternative first hop of near-equal cost."""
    config = ReplanConfig(prefer_different_first_hop=True)
    replanner = Replanner(config=config)
    # Start=1, Goal=4.
    # Path 1: 1->2->4 (cost 10.0 + 10.0 = 20.0)
    # Path 2: 1->3->4 (cost 10.1 + 10.0 = 20.1 <= 20.0 * 1.10 = 22.0)
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 10.0),
        (1, 3, 10.1),
        (2, 4, 10.0),
        (3, 4, 10.0),
    ]
    graph = make_graph(nodes, edges)

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.node_ids == (1, 3, 4)
    assert res.path.node_ids[1] == 3


def test_replan_prefer_different_first_hop_no_viable_alternative() -> None:
    """replan with prefer_different_first_hop=True and no viable alternative falls back to original path."""
    config = ReplanConfig(prefer_different_first_hop=True)
    replanner = Replanner(config=config)
    # Start=1, Goal=4. Only single path 1->2->4 exists.
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (2, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.node_ids == (1, 2, 4)


def test_replan_prefer_different_first_hop_false_ignores_previous_path() -> None:
    """replan with prefer_different_first_hop=False ignores previous_path entirely."""
    config = ReplanConfig(prefer_different_first_hop=False)
    replanner = Replanner(config=config)
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    # Path via 2 is cheaper (10.0) than via 3 (10.1)
    edges = [
        (1, 2, 5.0),
        (1, 3, 5.1),
        (2, 4, 5.0),
        (3, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.node_ids == (1, 2, 4)


def test_replan_near_cost_tolerance_exceeded() -> None:
    """Alternative costing > 110% of reference is rejected and original path returned."""
    config = ReplanConfig(prefer_different_first_hop=True)
    replanner = Replanner(config=config)
    # Start=1, Goal=4.
    # Original via 2: cost 10.0
    # Alternative via 3: cost 15.0 > 10.0 * 1.10 = 11.0
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [
        (1, 2, 5.0),
        (1, 3, 10.0),
        (2, 4, 5.0),
        (3, 4, 5.0),
    ]
    graph = make_graph(nodes, edges)

    assert NEAR_COST_TOLERANCE == 1.10

    res = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )
    assert res.outcome == ReplanOutcome.OK
    assert res.path is not None
    assert res.path.node_ids == (1, 2, 4)


def test_replan_determinism() -> None:
    """Two replan calls with identical inputs return identical ReplanResult values."""
    replanner = Replanner()
    nodes = [
        (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 5.0, -5.0, 0.0, NodeKind.WAYPOINT, "t"),
        (4, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    edges = [(1, 2, 5.0), (1, 3, 5.0), (2, 4, 5.0), (3, 4, 5.0)]
    graph = make_graph(nodes, edges)

    res1 = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )
    res2 = replanner.replan(
        from_xy=(0.0, 0.0),
        goal_xy=(10.0, 0.0),
        graph=graph,
        attempt=0,
        previous_path=(1, 2, 4),
    )

    assert res1.outcome == res2.outcome
    assert res1.reason == res2.reason
    assert res1.attempt_index == res2.attempt_index
    assert res1.path is not None and res2.path is not None
    assert res1.path.node_ids == res2.path.node_ids
    assert res1.path.total_cost == res2.path.total_cost
    assert res1.path.expansions == res2.path.expansions


def test_replan_config_and_result_frozen() -> None:
    """ReplanConfig and ReplanResult are frozen: setting fields raises FrozenInstanceError."""
    cfg = ReplanConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.max_attempts = 10  # type: ignore[misc]

    path = PathResult(found=True, node_ids=(1,), total_cost=0.0, expansions=0)
    res = ReplanResult(outcome=ReplanOutcome.OK, path=path, attempt_index=0)
    with pytest.raises(FrozenInstanceError):
        res.outcome = ReplanOutcome.SNAP_FAILED  # type: ignore[misc]


def test_from_navigator_config_matching_fields() -> None:
    """from_navigator_config builds Replanner matching max_replans and snap tolerance."""
    @dataclass
    class LocalNavConfig:
        max_replans: int = 5
        node_snap_tolerance_units: float = 12.5

    nav_cfg = LocalNavConfig()
    replanner = Replanner.from_navigator_config(nav_cfg)

    assert replanner._config.max_attempts == 5
    assert replanner._config.node_snap_tolerance_units == 12.5


def test_from_navigator_config_duck_typing() -> None:
    """from_navigator_config works via duck typing with any object exposing those two attributes."""
    class CustomObject:
        def __init__(self) -> None:
            self.max_replans = 8
            self.node_snap_tolerance_units = 7.5

    replanner = Replanner.from_navigator_config(CustomObject())
    assert replanner._config.max_attempts == 8
    assert replanner._config.node_snap_tolerance_units == 7.5


def test_static_ast_import_isolation() -> None:
    """Verify static import safety of replan.py using AST analysis."""
    with open("src/wow_bot/nav/replan.py", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="src/wow_bot/nav/replan.py")

    prohibited_exact = {
        "wow_bot.world",
        "wow_bot.strategist",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    allowed_nav_imports = {"wow_bot.nav.graph", "wow_bot.nav.astar", "wow_bot.nav.navigator"}

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
            # If importing from wow_bot.nav.*, confirm it is in allowed_nav_imports
            if mod.startswith("wow_bot.nav"):
                assert mod in allowed_nav_imports, f"Prohibited nav import: {mod}"
