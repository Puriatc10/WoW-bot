"""Pure deterministic replanner for calculating or recalculating navigation paths."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from wow_bot.nav.astar import AStarConfig, HeuristicKind, PathResult, find_path
from wow_bot.nav.graph import NavGraph

if TYPE_CHECKING:
    from wow_bot.nav.navigator import NavConfig

NEAR_COST_TOLERANCE: float = 1.10


class ReplanError(Exception):
    """Exception raised for errors during replanning."""


class ReplanOutcome(str, Enum):
    """Outcome status of a path replan calculation."""

    OK = "ok"
    SNAP_FAILED = "snap_failed"
    NO_PATH = "no_path"
    MAX_ATTEMPTS = "max_attempts"
    PATH_TOO_SHORT = "path_too_short"


@dataclass(frozen=True)
class ReplanConfig:
    """Configuration options for Replanner."""

    max_attempts: int = 3
    node_snap_tolerance_units: float = 5.0
    min_path_length_nodes: int = 1
    prefer_different_first_hop: bool = True
    heuristic: HeuristicKind = HeuristicKind.EUCLIDEAN

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.node_snap_tolerance_units <= 0:
            raise ValueError(
                f"node_snap_tolerance_units must be > 0, got {self.node_snap_tolerance_units}"
            )
        if self.min_path_length_nodes < 1:
            raise ValueError(
                f"min_path_length_nodes must be >= 1, got {self.min_path_length_nodes}"
            )


@dataclass(frozen=True)
class ReplanResult:
    """Result summary of a path replanning attempt."""

    outcome: ReplanOutcome
    path: PathResult | None
    attempt_index: int
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.outcome == ReplanOutcome.OK:
            if self.path is None or not self.path.found:
                raise ValueError("outcome == OK implies path is not None and path.found is True")
        else:
            if not self.reason:
                raise ValueError("outcome != OK implies reason != ''")

        if self.attempt_index < 0:
            raise ValueError(f"attempt_index must be >= 0, got {self.attempt_index}")


class Replanner:
    """Pure, deterministic navigation replanner."""

    def __init__(
        self,
        *,
        config: ReplanConfig | None = None,
        astar_config: AStarConfig | None = None,
    ) -> None:
        self._config = config if config is not None else ReplanConfig()
        self._astar_config = (
            astar_config
            if astar_config is not None
            else AStarConfig(heuristic=self._config.heuristic)
        )

    @classmethod
    def from_navigator_config(
        cls,
        nav_config: NavConfig | Any,
    ) -> Replanner:
        """Build a Replanner whose ReplanConfig matches nav_config parameters.

        Reads max_replans and node_snap_tolerance_units from nav_config via duck typing.
        """
        max_replans: int = getattr(nav_config, "max_replans")  # noqa: B009
        node_snap_tolerance_units: float = nav_config.node_snap_tolerance_units
        return cls(
            config=ReplanConfig(
                max_attempts=max_replans,
                node_snap_tolerance_units=node_snap_tolerance_units,
            )
        )

    def _try_alternative_first_hop(
        self,
        *,
        graph: NavGraph,
        start_id: int,
        goal_id: int,
        forbidden_hop: int,
        reference_cost: float,
    ) -> PathResult | None:
        """Try to find an alternative first hop to avoid retracing previous path.

        For each direct neighbor of start_id in graph.neighbors(start_id) (iterated in
        order sorted by to_id ASC), skip forbidden_hop and compute fresh A* path from
        neighbor to goal_id. If total cost <= reference_cost * NEAR_COST_TOLERANCE (1.10),
        returns a PathResult with node_ids = (start_id,) + fresh.node_ids, total_cost,
        and expansions.
        """
        for edge in graph.neighbors(start_id):
            n_id = edge.to_id
            if n_id == forbidden_hop:
                continue

            cost_leg1 = edge.cost
            fresh = find_path(graph, n_id, goal_id, config=self._astar_config)
            if fresh.found:
                total = cost_leg1 + fresh.total_cost
                if total <= reference_cost * NEAR_COST_TOLERANCE:
                    return PathResult(
                        found=True,
                        node_ids=(start_id,) + fresh.node_ids,
                        total_cost=total,
                        expansions=fresh.expansions,
                        reason="",
                    )
        return None

    def replan(
        self,
        *,
        from_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        graph: NavGraph,
        attempt: int,
        previous_path: tuple[int, ...] | None = None,
    ) -> ReplanResult:
        """Calculate or recalculate navigation path given current state and attempt index."""
        if attempt < 0:
            raise ReplanError(f"attempt must be >= 0, got {attempt}")

        if attempt >= self._config.max_attempts:
            return ReplanResult(
                outcome=ReplanOutcome.MAX_ATTEMPTS,
                path=None,
                attempt_index=attempt,
                reason="max_attempts_exceeded",
            )

        start_id = graph.find_node_id_at(
            from_xy[0], from_xy[1], tolerance=self._config.node_snap_tolerance_units
        )
        if start_id is None:
            return ReplanResult(
                outcome=ReplanOutcome.SNAP_FAILED,
                path=None,
                attempt_index=attempt,
                reason="start_not_snapped",
            )

        goal_id = graph.find_node_id_at(
            goal_xy[0], goal_xy[1], tolerance=self._config.node_snap_tolerance_units
        )
        if goal_id is None:
            return ReplanResult(
                outcome=ReplanOutcome.SNAP_FAILED,
                path=None,
                attempt_index=attempt,
                reason="goal_not_snapped",
            )

        if start_id == goal_id:
            return ReplanResult(
                outcome=ReplanOutcome.OK,
                path=PathResult(
                    found=True,
                    node_ids=(start_id,),
                    total_cost=0.0,
                    expansions=0,
                    reason="",
                ),
                attempt_index=attempt,
            )

        path = find_path(graph, start_id, goal_id, config=self._astar_config)
        if not path.found:
            return ReplanResult(
                outcome=ReplanOutcome.NO_PATH,
                path=path,
                attempt_index=attempt,
                reason=path.reason,
            )

        if len(path.node_ids) < self._config.min_path_length_nodes:
            return ReplanResult(
                outcome=ReplanOutcome.PATH_TOO_SHORT,
                path=path,
                attempt_index=attempt,
                reason=f"path_length={len(path.node_ids)}",
            )

        if (
            self._config.prefer_different_first_hop
            and previous_path is not None
            and len(path.node_ids) >= 2
            and len(previous_path) >= 2
        ):
            alt = self._try_alternative_first_hop(
                graph=graph,
                start_id=start_id,
                goal_id=goal_id,
                forbidden_hop=previous_path[1],
                reference_cost=path.total_cost,
            )
            if alt is not None and alt.found:
                path = alt

        return ReplanResult(
            outcome=ReplanOutcome.OK,
            path=path,
            attempt_index=attempt,
            reason="",
        )
