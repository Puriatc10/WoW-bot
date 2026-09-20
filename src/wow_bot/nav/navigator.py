"""Navigator component for moving the agent via NavGraph and A* pathfinding."""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from wow_bot.actuation.actuator import Actuator
from wow_bot.actuation.mapper import ActionStatus, MoveTo
from wow_bot.nav.astar import AStarConfig, HeuristicKind, PathResult, find_path

if TYPE_CHECKING:
    from wow_bot.nav.graph import NavGraph
    from wow_bot.session import Session


class NavigationError(Exception):
    """Exception raised for navigation configuration or runtime errors."""


class NavStatus(str, Enum):
    """Status outcomes for navigation requests."""

    SUCCESS = "success"
    FAILED = "failed"
    HARD_FAILURE = "hard_failure"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class NavConfig:
    """Configuration options for the Navigator."""

    arrival_tolerance_units: float = 0.5
    node_snap_tolerance_units: float = 5.0
    deviation_threshold_units: float = 3.0
    segment_max_steps: int = 50
    step_interval_s: float = 0.1
    max_replans: int = 3
    max_total_seconds: float = 120.0
    goal_snap_required: bool = True

    def __post_init__(self) -> None:
        """Validate configuration parameters."""
        if self.arrival_tolerance_units <= 0:
            raise ValueError(
                f"arrival_tolerance_units must be > 0, got {self.arrival_tolerance_units}"
            )
        if self.node_snap_tolerance_units <= 0:
            raise ValueError(
                f"node_snap_tolerance_units must be > 0, got {self.node_snap_tolerance_units}"
            )
        if self.deviation_threshold_units < 0:
            raise ValueError(
                f"deviation_threshold_units must be >= 0, got {self.deviation_threshold_units}"
            )
        if self.segment_max_steps < 1:
            raise ValueError(
                f"segment_max_steps must be >= 1, got {self.segment_max_steps}"
            )
        if self.step_interval_s < 0:
            raise ValueError(
                f"step_interval_s must be >= 0, got {self.step_interval_s}"
            )
        if self.max_replans < 0:
            raise ValueError(
                f"max_replans must be >= 0, got {self.max_replans}"
            )
        if self.max_total_seconds <= 0:
            raise ValueError(
                f"max_total_seconds must be > 0, got {self.max_total_seconds}"
            )


@dataclass(frozen=True)
class NavResult:
    """Result summary of a go_to navigation attempt."""

    status: NavStatus
    target_xy: tuple[float, float]
    final_xy: tuple[float, float]
    iterations: int
    replans: int
    path_attempts: int
    duration_s: float
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.status == NavStatus.SUCCESS:
            if self.reason != "":
                raise ValueError("status == SUCCESS implies reason == ''")
        else:
            if not self.reason:
                raise ValueError("status != SUCCESS implies non-empty reason")

        if self.iterations < 0:
            raise ValueError(f"iterations must be >= 0, got {self.iterations}")
        if self.replans < 0:
            raise ValueError(f"replans must be >= 0, got {self.replans}")
        if self.path_attempts < 0:
            raise ValueError(f"path_attempts must be >= 0, got {self.path_attempts}")
        if self.duration_s < 0.0:
            raise ValueError(f"duration_s must be >= 0.0, got {self.duration_s}")

    def to_json(self) -> dict[str, Any]:
        """Convert result into JSON-serializable dictionary representation."""
        return {
            "status": self.status.value,
            "target_xy": [self.target_xy[0], self.target_xy[1]],
            "final_xy": [self.final_xy[0], self.final_xy[1]],
            "iterations": self.iterations,
            "replans": self.replans,
            "path_attempts": self.path_attempts,
            "duration_s": self.duration_s,
            "reason": self.reason,
        }


@runtime_checkable
class Replanner(Protocol):
    """Protocol for calculating or recalculating navigation paths."""

    def replan(
        self,
        *,
        from_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        graph: NavGraph,
        config: NavConfig,
    ) -> PathResult | None:
        """Return a PathResult from current position to goal, or None if no path possible."""
        ...


class SimpleReplanner:
    """Default pure replanner snapping endpoints to graph nodes and running A*."""

    def __init__(self, *, heuristic: HeuristicKind = HeuristicKind.EUCLIDEAN) -> None:
        self._heuristic = heuristic

    def replan(
        self,
        *,
        from_xy: tuple[float, float],
        goal_xy: tuple[float, float],
        graph: NavGraph,
        config: NavConfig,
    ) -> PathResult | None:
        """Snap positions to node IDs and call A* find_path."""
        start_id = graph.find_node_id_at(
            from_xy[0], from_xy[1], tolerance=config.node_snap_tolerance_units
        )
        goal_id = graph.find_node_id_at(
            goal_xy[0], goal_xy[1], tolerance=config.node_snap_tolerance_units
        )

        if start_id is None or goal_id is None:
            return None

        return find_path(
            graph,
            start_id,
            goal_id,
            config=AStarConfig(heuristic=self._heuristic),
        )


class Navigator:
    """Synchronous component moving the agent toward target coordinates using NavGraph and Actuator."""

    def __init__(
        self,
        graph: NavGraph,
        actuator: Actuator,
        position_source: Callable[[], tuple[float, float]],
        *,
        config: NavConfig | None = None,
        replanner: Replanner | None = None,
        on_segment_start: Callable[[int], None] | None = None,
        on_segment_end: Callable[[int, bool], None] | None = None,
        session: Session | None = None,
        clock: Callable[[], float] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._graph = graph
        self._actuator = actuator
        self._position_source = position_source
        self._config = config if config is not None else NavConfig()
        self._replanner = replanner if replanner is not None else SimpleReplanner()
        self._on_segment_start = on_segment_start
        self._on_segment_end = on_segment_end
        self._session = session
        self._clock = clock if clock is not None else time.monotonic
        self._sleep = sleep if sleep is not None else time.sleep

    def _walk_segment(self, node_id: int, deadline: float) -> bool:
        """Execute movement steps toward graph node_id until reached, aborted, or timed out."""
        target_node = self._graph.get_node(node_id)
        nx, ny = target_node.x, target_node.y

        for _step_index in range(self._config.segment_max_steps):
            if self._clock() >= deadline:
                return False

            if self._actuator.is_aborted():
                return False

            pos = self._position_source()
            if math.hypot(pos[0] - nx, pos[1] - ny) <= self._config.arrival_tolerance_units:
                return True

            result = self._actuator.execute(MoveTo(x=nx, y=ny), position=pos)
            if result.status == ActionStatus.FAILED:
                pass

            if self._config.step_interval_s > 0:
                self._sleep(self._config.step_interval_s)

        return False

    def go_to(self, target_xy: tuple[float, float]) -> NavResult:
        """Move agent to target_xy by pathfinding and stepping through graph segments."""
        start_time = self._clock()
        deadline = start_time + self._config.max_total_seconds
        iterations = 0
        replans = 0
        path_attempts = 0

        def _finish(res: NavResult) -> NavResult:
            if self._session is not None:
                evt = {"event": "nav_completed"}
                evt.update(res.to_json())
                self._session.write_event(evt)
            return res

        if self._session is not None:
            self._session.write_event({
                "event": "nav_started",
                "target_xy": [target_xy[0], target_xy[1]],
            })

        goal_id = self._graph.find_node_id_at(
            target_xy[0],
            target_xy[1],
            tolerance=self._config.node_snap_tolerance_units,
        )
        if goal_id is None:
            current_pos = self._position_source()
            duration_s = max(0.0, self._clock() - start_time)
            return _finish(
                NavResult(
                    status=NavStatus.HARD_FAILURE,
                    target_xy=target_xy,
                    final_xy=current_pos,
                    iterations=iterations,
                    replans=replans,
                    path_attempts=path_attempts,
                    duration_s=duration_s,
                    reason="goal_not_snapped",
                )
            )

        replan_reason = ""

        while replans <= self._config.max_replans:
            if self._clock() >= deadline:
                current_pos = self._position_source()
                duration_s = max(0.0, self._clock() - start_time)
                return _finish(
                    NavResult(
                        status=NavStatus.TIMEOUT,
                        target_xy=target_xy,
                        final_xy=current_pos,
                        iterations=iterations,
                        replans=replans,
                        path_attempts=path_attempts,
                        duration_s=duration_s,
                        reason="max_total_seconds_exceeded",
                    )
                )

            current_pos = self._position_source()
            if (
                math.hypot(current_pos[0] - target_xy[0], current_pos[1] - target_xy[1])
                <= self._config.arrival_tolerance_units
            ):
                duration_s = max(0.0, self._clock() - start_time)
                return _finish(
                    NavResult(
                        status=NavStatus.SUCCESS,
                        target_xy=target_xy,
                        final_xy=current_pos,
                        iterations=iterations,
                        replans=replans,
                        path_attempts=path_attempts,
                        duration_s=duration_s,
                        reason="",
                    )
                )

            start_id = self._graph.find_node_id_at(
                current_pos[0],
                current_pos[1],
                tolerance=self._config.node_snap_tolerance_units,
            )
            if start_id is None:
                duration_s = max(0.0, self._clock() - start_time)
                return _finish(
                    NavResult(
                        status=NavStatus.HARD_FAILURE,
                        target_xy=target_xy,
                        final_xy=current_pos,
                        iterations=iterations,
                        replans=replans,
                        path_attempts=path_attempts,
                        duration_s=duration_s,
                        reason="start_not_snapped",
                    )
                )

            if replans > 0 and self._session is not None:
                self._session.write_event({
                    "event": "nav_replan",
                    "attempt": replans,
                    "from_xy": [current_pos[0], current_pos[1]],
                    "reason": replan_reason,
                })

            path = self._replanner.replan(
                from_xy=current_pos,
                goal_xy=target_xy,
                graph=self._graph,
                config=self._config,
            )
            path_attempts += 1

            if path is None or not path.found:
                reason_str = (
                    "no_path:"
                    + (path.reason if path is not None else "replanner_returned_none")
                )
                duration_s = max(0.0, self._clock() - start_time)
                return _finish(
                    NavResult(
                        status=NavStatus.HARD_FAILURE,
                        target_xy=target_xy,
                        final_xy=current_pos,
                        iterations=iterations,
                        replans=replans,
                        path_attempts=path_attempts,
                        duration_s=duration_s,
                        reason=reason_str,
                    )
                )

            break_to_replan = False
            for k, node_id in enumerate(path.node_ids[1:]):
                if self._clock() >= deadline:
                    curr_pos = self._position_source()
                    duration_s = max(0.0, self._clock() - start_time)
                    return _finish(
                        NavResult(
                            status=NavStatus.TIMEOUT,
                            target_xy=target_xy,
                            final_xy=curr_pos,
                            iterations=iterations,
                            replans=replans,
                            path_attempts=path_attempts,
                            duration_s=duration_s,
                            reason="max_total_seconds_exceeded",
                        )
                    )

                if self._session is not None:
                    self._session.write_event({
                        "event": "nav_segment",
                        "node_id": node_id,
                        "index": k,
                    })

                if self._on_segment_start is not None:
                    self._on_segment_start(node_id)

                reached = self._walk_segment(node_id, deadline)

                if self._on_segment_end is not None:
                    self._on_segment_end(node_id, reached)

                if not reached:
                    if self._clock() >= deadline:
                        curr_pos = self._position_source()
                        duration_s = max(0.0, self._clock() - start_time)
                        return _finish(
                            NavResult(
                                status=NavStatus.TIMEOUT,
                                target_xy=target_xy,
                                final_xy=curr_pos,
                                iterations=iterations,
                                replans=replans,
                                path_attempts=path_attempts,
                                duration_s=duration_s,
                                reason="max_total_seconds_exceeded",
                            )
                        )

                    curr_pos = self._position_source()
                    duration_s = max(0.0, self._clock() - start_time)
                    return _finish(
                        NavResult(
                            status=NavStatus.FAILED,
                            target_xy=target_xy,
                            final_xy=curr_pos,
                            iterations=iterations,
                            replans=replans,
                            path_attempts=path_attempts,
                            duration_s=duration_s,
                            reason="segment_failed",
                        )
                    )

                expected = self._graph.get_node(node_id)
                actual = self._position_source()
                dev = math.hypot(actual[0] - expected.x, actual[1] - expected.y)

                if dev > self._config.deviation_threshold_units:
                    replans += 1
                    replan_reason = "deviation"
                    if replans > self._config.max_replans:
                        duration_s = max(0.0, self._clock() - start_time)
                        return _finish(
                            NavResult(
                                status=NavStatus.HARD_FAILURE,
                                target_xy=target_xy,
                                final_xy=actual,
                                iterations=iterations,
                                replans=replans,
                                path_attempts=path_attempts,
                                duration_s=duration_s,
                                reason="max_replans_exceeded",
                            )
                        )
                    break_to_replan = True
                    break

                iterations += 1

            if break_to_replan:
                continue

        duration_s = max(0.0, self._clock() - start_time)
        return _finish(
            NavResult(
                status=NavStatus.HARD_FAILURE,
                target_xy=target_xy,
                final_xy=self._position_source(),
                iterations=iterations,
                replans=replans,
                path_attempts=path_attempts,
                duration_s=duration_s,
                reason="max_replans_exceeded",
            )
        )
