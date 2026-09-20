"""A* pathfinding algorithm operating on NavGraph."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_bot.nav.graph import GraphNode, NavGraph


class PathError(Exception):
    """Exception raised for pathfinding errors or invalid parameters."""


class HeuristicKind(str, Enum):
    """Supported heuristic cost functions for A* pathfinding."""

    EUCLIDEAN = "euclidean"
    MANHATTAN = "manhattan"
    ZERO = "zero"


@dataclass(frozen=True)
class AStarConfig:
    """Configuration options for A* path search."""

    heuristic: HeuristicKind = HeuristicKind.EUCLIDEAN
    max_expansions: int = 100_000
    max_path_length: int = 5_000

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_expansions < 1:
            raise ValueError(f"max_expansions must be >= 1, got {self.max_expansions}")
        if self.max_path_length < 1:
            raise ValueError(f"max_path_length must be >= 1, got {self.max_path_length}")


@dataclass(frozen=True)
class PathResult:
    """Result of an A* path search or multi-leg search."""

    found: bool
    node_ids: tuple[int, ...]
    total_cost: float
    expansions: int
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.found:
            if len(self.node_ids) < 1:
                raise ValueError("found == True implies len(node_ids) >= 1")
            if self.total_cost < 0.0:
                raise ValueError("found == True implies total_cost >= 0.0")
            if self.reason != "":
                raise ValueError("found == True implies reason == ''")
        else:
            if self.node_ids != ():
                raise ValueError("found == False implies node_ids == ()")
            if self.total_cost != 0.0:
                raise ValueError("found == False implies total_cost == 0.0")
            if not self.reason:
                raise ValueError("found == False implies reason is a non-empty string")


def heuristic_cost(
    kind: HeuristicKind,
    a: GraphNode,
    b: GraphNode,
) -> float:
    """Compute the heuristic cost estimate between two graph nodes.

    Admissibility Argument:
    -----------------------
    For A* search to guarantee finding an optimal (shortest) path, the heuristic
    function h(a, b) must be admissible; that is, it must never overestimate the
    true minimum cost to reach the goal `b` from `a`.

    - EUCLIDEAN: Straight-line 2D Euclidean distance sqrt((a.x - b.x)^2 + (a.y - b.y)^2).
      The shortest possible physical path between two points in 2D Euclidean space
      is a straight line. Since edge costs produced by the graph builder are defined as
      max(distance, min_edge_cost) (multiplied by optional directional penalties >= 1.0),
      every edge cost e(u, v) is at least the physical Euclidean distance between u and v.
      Therefore, the straight-line distance between any node `a` and goal `b` is always
      <= the true edge-weighted path cost, making EUCLIDEAN strictly admissible.

    - MANHATTAN: L1 distance |a.x - b.x| + |a.y - b.y|.
      On grid-like or orthogonal movement graphs where motion is restricted to axis-aligned
      steps with edge costs equal to step length, L1 distance equals the minimum possible
      path cost and is admissible. On general 2D graphs where diagonal movement is allowed,
      MANHATTAN may overestimate Euclidean distance unless the graph grid topology forces
      orthogonal edges.

    - ZERO: Constant 0.0 heuristic.
      Always 0.0, which is trivially <= true cost (since all edge costs are positive).
      This turns A* into Dijkstra's algorithm, guaranteeing admissibility on all graphs.
    """
    if kind == HeuristicKind.EUCLIDEAN:
        return math.hypot(a.x - b.x, a.y - b.y)
    if kind == HeuristicKind.MANHATTAN:
        return abs(a.x - b.x) + abs(a.y - b.y)
    if kind == HeuristicKind.ZERO:
        return 0.0
    raise ValueError(f"Unknown HeuristicKind: {kind}")


def find_path(
    graph: NavGraph,
    start_id: int,
    goal_id: int,
    *,
    config: AStarConfig | None = None,
) -> PathResult:
    """Find the shortest path between start_id and goal_id in graph using A* search."""
    cfg = config if config is not None else AStarConfig()

    if not graph.has_node(start_id):
        return PathResult(
            found=False,
            node_ids=(),
            total_cost=0.0,
            expansions=0,
            reason="start_not_in_graph",
        )

    if not graph.has_node(goal_id):
        return PathResult(
            found=False,
            node_ids=(),
            total_cost=0.0,
            expansions=0,
            reason="goal_not_in_graph",
        )

    if start_id == goal_id:
        return PathResult(
            found=True,
            node_ids=(start_id,),
            total_cost=0.0,
            expansions=0,
            reason="",
        )

    goal_node = graph.get_node(goal_id)
    start_node = graph.get_node(start_id)

    start_h = heuristic_cost(cfg.heuristic, start_node, goal_node)
    # Min-heap tuples: (f_score, g_score, node_id)
    # node_id serves as total tiebreaker
    open_set: list[tuple[float, float, int]] = [(start_h, 0.0, start_id)]

    g_score: dict[int, float] = {start_id: 0.0}
    came_from: dict[int, int] = {}
    closed: set[int] = set()
    expansions = 0

    while open_set:
        _f, current_g, current = heapq.heappop(open_set)

        if current in closed:
            continue

        if current == goal_id:
            path_rev = [goal_id]
            curr = goal_id
            while curr in came_from:
                curr = came_from[curr]
                path_rev.append(curr)
            path = tuple(reversed(path_rev))

            if len(path) > cfg.max_path_length:
                return PathResult(
                    found=False,
                    node_ids=(),
                    total_cost=0.0,
                    expansions=expansions,
                    reason="max_path_length_exceeded",
                )

            return PathResult(
                found=True,
                node_ids=path,
                total_cost=g_score[goal_id],
                expansions=expansions,
                reason="",
            )

        closed.add(current)
        expansions += 1

        if expansions > cfg.max_expansions:
            return PathResult(
                found=False,
                node_ids=(),
                total_cost=0.0,
                expansions=expansions,
                reason="max_expansions_exceeded",
            )

        for edge in graph.neighbors(current):
            to_id = edge.to_id
            cost = edge.cost

            tentative_g = current_g + cost
            if tentative_g < g_score.get(to_id, math.inf):
                g_score[to_id] = tentative_g
                came_from[to_id] = current
                to_node = graph.get_node(to_id)
                h = heuristic_cost(cfg.heuristic, to_node, goal_node)
                heapq.heappush(open_set, (tentative_g + h, tentative_g, to_id))

    return PathResult(
        found=False,
        node_ids=(),
        total_cost=0.0,
        expansions=expansions,
        reason="unreachable",
    )


def find_path_through(
    graph: NavGraph,
    waypoints: tuple[int, ...],
    *,
    config: AStarConfig | None = None,
) -> PathResult:
    """Find a composite path passing through an ordered sequence of waypoints."""
    if not waypoints:
        raise PathError("waypoints tuple cannot be empty")

    if len(waypoints) == 1:
        return PathResult(
            found=True,
            node_ids=(waypoints[0],),
            total_cost=0.0,
            expansions=0,
            reason="",
        )

    cfg = config if config is not None else AStarConfig()
    combined_nodes: list[int] = []
    accumulated_cost = 0.0
    accumulated_expansions = 0

    for i in range(len(waypoints) - 1):
        leg_start = waypoints[i]
        leg_goal = waypoints[i + 1]

        leg_res = find_path(graph, leg_start, leg_goal, config=cfg)
        accumulated_expansions += leg_res.expansions

        if not leg_res.found:
            return PathResult(
                found=False,
                node_ids=(),
                total_cost=0.0,
                expansions=accumulated_expansions,
                reason=f"leg_failed:{i}:{leg_res.reason}",
            )

        accumulated_cost += leg_res.total_cost
        if i == 0:
            combined_nodes.extend(leg_res.node_ids)
        else:
            combined_nodes.extend(leg_res.node_ids[1:])

    if len(combined_nodes) > cfg.max_path_length:
        return PathResult(
            found=False,
            node_ids=(),
            total_cost=0.0,
            expansions=accumulated_expansions,
            reason="max_path_length_exceeded",
        )

    return PathResult(
        found=True,
        node_ids=tuple(combined_nodes),
        total_cost=accumulated_cost,
        expansions=accumulated_expansions,
        reason="",
    )
