"""Flee logic implementation for WoW-bot combat subsystem.

Provides a deterministic decision layer that selects safe retreat waypoints
from the navigation graph and delegates movement execution to the Navigator.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from wow_bot.nav.graph import NodeKind
from wow_bot.nav.navigator import NavStatus

if TYPE_CHECKING:
    from wow_bot.nav.graph import NavGraph
    from wow_bot.nav.navigator import Navigator
    from wow_bot.session import Session
    from wow_bot.world.store import WorldModel


class FleeError(Exception):
    """Exception raised for errors in flee logic calculation or execution."""


@runtime_checkable
class FleeStateView(Protocol):
    """Protocol defining required state fields for flee evaluations."""

    self_hp_percent: float
    self_x: float
    self_y: float
    current_target_id: str | None
    adds_count: int
    target_x: float | None
    target_y: float | None


class FleeTrigger(str, Enum):
    """Triggers that cause the bot to flee."""

    NONE = "none"
    LOW_HP = "low_hp"
    TOO_MANY_ADDS = "too_many_adds"


class FleeStatus(str, Enum):
    """Outcome status of flee execution."""

    NOT_TRIGGERED = "not_triggered"
    STARTED = "started"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass(frozen=True)
class FleeConfig:
    """Configuration options for flee thresholding and candidate selection."""

    hp_threshold: float = 25.0
    adds_threshold: int = 3
    search_radius_units: float = 60.0
    min_distance_from_target_units: float = 15.0
    candidates_limit: int = 8
    max_flee_seconds: float = 30.0
    require_waypoint_kind: bool = True
    fallback_direction_units: float = 20.0

    def __post_init__(self) -> None:
        """Validate configuration thresholds and bounds."""
        if not (0.0 <= self.hp_threshold <= 100.0):
            raise ValueError(
                f"hp_threshold must be between 0.0 and 100.0, got {self.hp_threshold}"
            )
        if self.adds_threshold < 1:
            raise ValueError(f"adds_threshold must be >= 1, got {self.adds_threshold}")
        if self.search_radius_units <= 0:
            raise ValueError(
                f"search_radius_units must be > 0, got {self.search_radius_units}"
            )
        if self.min_distance_from_target_units < 0:
            raise ValueError(
                "min_distance_from_target_units must be >= 0, got "
                f"{self.min_distance_from_target_units}"
            )
        if self.candidates_limit < 1:
            raise ValueError(
                f"candidates_limit must be >= 1, got {self.candidates_limit}"
            )
        if self.max_flee_seconds <= 0:
            raise ValueError(
                f"max_flee_seconds must be > 0, got {self.max_flee_seconds}"
            )
        if self.fallback_direction_units <= 0:
            raise ValueError(
                "fallback_direction_units must be > 0, got "
                f"{self.fallback_direction_units}"
            )


@dataclass(frozen=True)
class FleeDecision:
    """Deterministic flee target decision output."""

    trigger: FleeTrigger
    target_xy: tuple[float, float] | None
    reason: str

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if self.trigger == FleeTrigger.NONE and self.target_xy is not None:
            raise ValueError("trigger == NONE implies target_xy is None")
        if self.trigger != FleeTrigger.NONE and self.target_xy is None:
            raise ValueError("trigger != NONE implies target_xy is not None")
        if not self.reason:
            raise ValueError("reason is a non-empty string")


@dataclass(frozen=True)
class FleeResult:
    """Summary of a flee execution attempt."""

    status: FleeStatus
    trigger: FleeTrigger
    target_xy: tuple[float, float] | None
    navigator_status: str
    duration_s: float
    reason: str

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.status == FleeStatus.NOT_TRIGGERED and (
            self.trigger != FleeTrigger.NONE
            or self.target_xy is not None
            or self.navigator_status != ""
        ):
            raise ValueError(
                "status == NOT_TRIGGERED implies trigger == NONE and "
                "target_xy is None and navigator_status == ''"
            )
        if self.status in (FleeStatus.STARTED, FleeStatus.COMPLETED) and (
            self.target_xy is None or self.navigator_status == ""
        ):
            raise ValueError(
                "status == STARTED or COMPLETED implies target_xy is not None "
                "and navigator_status != ''"
            )
        if self.status == FleeStatus.FAILED and not self.reason:
            raise ValueError("status == FAILED implies reason != ''")
        if self.duration_s < 0.0:
            raise ValueError(f"duration_s must be >= 0.0, got {self.duration_s}")

    def to_json(self) -> dict[str, Any]:
        """Convert flee result to a JSON-serializable dictionary."""
        return {
            "status": self.status.value,
            "trigger": self.trigger.value,
            "target_xy": list(self.target_xy) if self.target_xy is not None else None,
            "navigator_status": self.navigator_status,
            "duration_s": self.duration_s,
            "reason": self.reason,
        }


class FleeController:
    """Controller managing flee evaluation, retreat point selection, and navigation."""

    def __init__(
        self,
        navigator: Navigator,
        world: WorldModel,
        graph: NavGraph,
        *,
        config: FleeConfig | None = None,
        session: Session | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._navigator = navigator
        self._world = world
        self._graph = graph
        self._config = config if config is not None else FleeConfig()
        self._session = session
        self._clock = clock if clock is not None else time.monotonic

    def should_flee(self, state: FleeStateView) -> FleeTrigger:
        """Evaluate state thresholds to determine whether fleeing is triggered."""
        if state.self_hp_percent <= self._config.hp_threshold:
            return FleeTrigger.LOW_HP
        if state.adds_count >= self._config.adds_threshold:
            return FleeTrigger.TOO_MANY_ADDS
        return FleeTrigger.NONE

    def decide(self, state: FleeStateView) -> FleeDecision:
        """Select a safe retreat waypoint or fallback target point deterministically."""
        trigger = self.should_flee(state)
        if trigger == FleeTrigger.NONE:
            return FleeDecision(
                trigger=FleeTrigger.NONE,
                target_xy=None,
                reason="not_triggered",
            )

        target_known = state.target_x is not None and state.target_y is not None

        candidates = []
        for node in self._graph.nodes.values():
            if self._config.require_waypoint_kind and node.kind != NodeKind.WAYPOINT:
                continue

            dist_self = math.hypot(node.x - state.self_x, node.y - state.self_y)
            if dist_self > self._config.search_radius_units:
                continue

            if target_known:
                assert state.target_x is not None
                assert state.target_y is not None
                dist_target = math.hypot(
                    node.x - state.target_x, node.y - state.target_y
                )
                if dist_target < self._config.min_distance_from_target_units:
                    continue

            candidates.append((dist_self, node.id, node))

        candidates.sort(key=lambda item: (item[0], item[1]))
        limited_candidates = candidates[: self._config.candidates_limit]

        if limited_candidates:
            chosen = limited_candidates[0][2]
            return FleeDecision(
                trigger=trigger,
                target_xy=(chosen.x, chosen.y),
                reason=f"waypoint:{chosen.id}",
            )

        if target_known:
            assert state.target_x is not None
            assert state.target_y is not None
            dx = state.self_x - state.target_x
            dy = state.self_y - state.target_y
            norm = math.sqrt(dx * dx + dy * dy) or 1.0
            fx = state.self_x + (dx / norm) * self._config.fallback_direction_units
            fy = state.self_y + (dy / norm) * self._config.fallback_direction_units
        else:
            fx = state.self_x - self._config.fallback_direction_units
            fy = state.self_y

        return FleeDecision(
            trigger=trigger,
            target_xy=(fx, fy),
            reason="fallback_point",
        )

    def execute(self, state: FleeStateView) -> FleeResult:
        """Execute flee movement using Navigator and write session events.

        Session events emitted (if session is attached):
          - "flee_started":
              {"event": "flee_started", "trigger": str, "target_xy": list[float]}
          - "flee_completed":
              {"event": "flee_completed", "navigator_status": str,
               "duration_s": float, "target_xy": list[float]}
          - "flee_failed":
              * On navigator exception:
                {"event": "flee_failed", "reason": str}
              * On navigator failure / timeout:
                {"event": "flee_failed", "navigator_status": str,
                 "duration_s": float, "target_xy": list[float], "reason": str}
        """
        start = self._clock()
        decision = self.decide(state)
        if decision.trigger == FleeTrigger.NONE:
            return FleeResult(
                status=FleeStatus.NOT_TRIGGERED,
                trigger=FleeTrigger.NONE,
                target_xy=None,
                navigator_status="",
                duration_s=0.0,
                reason="not_triggered",
            )

        assert decision.target_xy is not None
        target_xy_list = [decision.target_xy[0], decision.target_xy[1]]

        if self._session is not None:
            self._session.write_event({
                "event": "flee_started",
                "trigger": decision.trigger.value,
                "target_xy": target_xy_list,
            })

        try:
            nav_result = self._navigator.go_to(decision.target_xy)
        except Exception as exc:  # noqa: BLE001
            duration = max(0.0, self._clock() - start)
            reason_str = f"navigator_error:{type(exc).__name__}"
            if self._session is not None:
                self._session.write_event({
                    "event": "flee_failed",
                    "reason": repr(exc),
                })
            return FleeResult(
                status=FleeStatus.FAILED,
                trigger=decision.trigger,
                target_xy=decision.target_xy,
                navigator_status="",
                duration_s=duration,
                reason=reason_str,
            )

        duration = max(0.0, self._clock() - start)

        if nav_result.status == NavStatus.SUCCESS:
            status = FleeStatus.COMPLETED
        else:
            status = FleeStatus.FAILED

        reason = nav_result.reason or ""

        if duration > self._config.max_flee_seconds:
            status = FleeStatus.FAILED
            reason = "flee_timeout"

        if self._session is not None:
            if status == FleeStatus.COMPLETED:
                self._session.write_event({
                    "event": "flee_completed",
                    "navigator_status": nav_result.status.value,
                    "duration_s": duration,
                    "target_xy": target_xy_list,
                })
            else:
                self._session.write_event({
                    "event": "flee_failed",
                    "navigator_status": nav_result.status.value,
                    "duration_s": duration,
                    "target_xy": target_xy_list,
                    "reason": reason,
                })

        return FleeResult(
            status=status,
            trigger=decision.trigger,
            target_xy=decision.target_xy,
            navigator_status=nav_result.status.value,
            duration_s=duration,
            reason=reason,
        )
