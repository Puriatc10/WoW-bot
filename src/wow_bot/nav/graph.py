"""Immutable in-memory navigation graph built from the World Model for A* pathfinding."""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from wow_bot.session import Session
    from wow_bot.world.store import WorldModel


class GraphError(Exception):
    """Exception raised for errors in navigation graph construction or query."""


class NodeKind(str, Enum):
    """Node classification kinds stored in World Model and graph nodes."""

    VENDOR = "vendor"
    TRAINER = "trainer"
    NODE = "node"
    MOB = "mob"
    WAYPOINT = "waypoint"
    UNKNOWN = "unknown"


@dataclass(frozen=True)
class GraphNode:
    """Immutable graph node representation."""

    id: int
    x: float
    y: float
    z: float
    kind: NodeKind
    last_seen_at: str


@dataclass(frozen=True)
class GraphEdge:
    """Immutable outgoing graph edge representation."""

    to_id: int
    cost: float


@dataclass(frozen=True)
class GraphConfig:
    """Configuration for graph building and edge cost penalties."""

    directional_penalties: Mapping[NodeKind, float] = field(
        default_factory=lambda: MappingProxyType({})
    )
    min_edge_cost: float = 0.01
    default_kind: NodeKind = NodeKind.UNKNOWN


@dataclass(frozen=True)
class NavGraph:
    """Immutable navigation graph with proxy-wrapped mappings."""

    nodes: Mapping[int, GraphNode]
    adjacency: Mapping[int, tuple[GraphEdge, ...]]

    def has_node(self, node_id: int) -> bool:
        """Check whether a node ID exists in the graph."""
        return node_id in self.nodes

    def get_node(self, node_id: int) -> GraphNode:
        """Get GraphNode by node_id or raise GraphError if missing."""
        try:
            return self.nodes[node_id]
        except KeyError:
            raise GraphError(f"Node {node_id} does not exist in graph") from None

    def neighbors(self, node_id: int) -> tuple[GraphEdge, ...]:
        """Return outgoing edges for node_id, or empty tuple for unknown nodes."""
        return self.adjacency.get(node_id, ())

    def node_count(self) -> int:
        """Return total number of nodes in the graph."""
        return len(self.nodes)

    def edge_count(self) -> int:
        """Return count of undirected node pairs connected by edges.

        An edge present in both directions (u->v and v->u) or in a single direction (u->v)
        counts as one undirected pair {u, v}.
        """
        pairs: set[frozenset[int]] = set()
        for u, edges in self.adjacency.items():
            for edge in edges:
                pairs.add(frozenset({u, edge.to_id}))
        return len(pairs)

    def find_node_id_at(self, x: float, y: float, *, tolerance: float) -> int | None:
        """Find the ID of the nearest node within tolerance 2D distance.

        If multiple nodes are equidistant at the minimum distance within tolerance,
        returns the smallest node ID.
        """
        best_node_id: int | None = None
        best_dist: float | None = None

        for node in self.nodes.values():
            dist = math.hypot(node.x - x, node.y - y)
            if dist <= tolerance and (
                best_dist is None
                or dist < best_dist
                or (dist == best_dist and (best_node_id is None or node.id < best_node_id))
            ):
                best_dist = dist
                best_node_id = node.id

        return best_node_id


async def build_graph(
    world: WorldModel,
    *,
    config: GraphConfig | None = None,
    session: Session | None = None,
) -> NavGraph:
    """Build an immutable, deterministic NavGraph from the World Model database."""
    cfg = config if config is not None else GraphConfig()

    if cfg.min_edge_cost <= 0:
        raise GraphError(f"min_edge_cost must be > 0, got {cfg.min_edge_cost}")

    for k, penalty in cfg.directional_penalties.items():
        if penalty <= 0:
            raise GraphError(f"Penalty for kind {k.value!r} must be > 0, got {penalty}")

    t0_ns = time.perf_counter_ns()
    node_rows, edge_rows = await world.all_nodes_and_edges()

    nodes_dict: dict[int, GraphNode] = {}
    for r in node_rows:
        try:
            kind = NodeKind(r.kind)
        except ValueError:
            kind = cfg.default_kind

        nodes_dict[r.id] = GraphNode(
            id=r.id,
            x=r.x,
            y=r.y,
            z=r.z,
            kind=kind,
            last_seen_at=r.last_seen_at,
        )

    adjacency_lists: dict[int, list[GraphEdge]] = {nid: [] for nid in nodes_dict}
    skipped_edges = 0

    for er in edge_rows:
        if er.from_id not in nodes_dict or er.to_id not in nodes_dict:
            skipped_edges += 1
            continue

        base_cost = max(er.cost, cfg.min_edge_cost)
        if cfg.directional_penalties:
            dest_kind = nodes_dict[er.to_id].kind
            multiplier = cfg.directional_penalties.get(dest_kind, 1.0)
            final_cost = base_cost * multiplier
        else:
            final_cost = base_cost

        final_cost = max(final_cost, cfg.min_edge_cost)
        adjacency_lists[er.from_id].append(GraphEdge(to_id=er.to_id, cost=final_cost))

    sorted_nodes = dict(sorted(nodes_dict.items(), key=lambda item: item[0]))
    sorted_adjacency: dict[int, tuple[GraphEdge, ...]] = {}

    for nid in sorted_nodes:
        edges = adjacency_lists[nid]
        edges.sort(key=lambda e: e.to_id)
        sorted_adjacency[nid] = tuple(edges)

    graph = NavGraph(
        nodes=MappingProxyType(sorted_nodes),
        adjacency=MappingProxyType(sorted_adjacency),
    )

    t1_ns = time.perf_counter_ns()
    duration_ms = (t1_ns - t0_ns) / 1_000_000.0

    if session is not None:
        session.write_event({
            "event": "nav_graph_built",
            "payload": {
                "node_count": graph.node_count(),
                "edge_count": graph.edge_count(),
                "skipped_edges": skipped_edges,
                "duration_ms": duration_ms,
            },
        })

    return graph


async def rebuild_graph(
    previous: NavGraph,
    world: WorldModel,
    *,
    config: GraphConfig | None = None,
    session: Session | None = None,
) -> NavGraph:
    """Rebuild navigation graph from World Model.

    The `previous` argument is accepted for API symmetry and future
    incremental rebuilds; in this task it is unused. This function performs
    a full graph rebuild by delegating to build_graph.
    """
    return await build_graph(world, config=config, session=session)
