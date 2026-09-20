"""Recovery primitives and planning for FSM state STUCK_RECOVERY.

Provides deterministic, pure recovery intent generation (backstep, camera sweep, jump,
and alternative waypoint) without OS input injection, clock time reads, or side-effects.
"""

import math
import random
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum

from wow_bot.actuation.mapper import Intent, MoveTo, Turn
from wow_bot.executor.fsm_v2 import GameStateLike, MetaStateLike
from wow_bot.executor.states import FSMState


class RecoveryError(Exception):
    """Raised when recovery planning fails or maximum recovery attempts are exceeded."""


class RecoveryBehaviorKind(str, Enum):
    """Enumeration of recovery behavior primitive kinds."""

    BACKSTEP = "backstep"
    CAMERA_SWEEP = "camera_sweep"
    JUMP = "jump"
    ALTERNATIVE_WAYPOINT = "alternative_waypoint"


_BEHAVIOR_KINDS_ORDER: tuple[RecoveryBehaviorKind, ...] = (
    RecoveryBehaviorKind.BACKSTEP,
    RecoveryBehaviorKind.CAMERA_SWEEP,
    RecoveryBehaviorKind.JUMP,
    RecoveryBehaviorKind.ALTERNATIVE_WAYPOINT,
)


@dataclass(frozen=True)
class RecoveryConfig:
    """Configuration options for stuck recovery behavior planning."""

    max_attempts: int = 3
    backstep_distance_units: float = 1.0
    camera_sweep_rad: float = 0.6
    alternative_radius_units: float = 5.0
    alternative_samples: int = 8
    behavior_weights: tuple[float, ...] = (1.0, 1.0, 1.0, 1.0)

    def __post_init__(self) -> None:
        if self.max_attempts < 1:
            raise ValueError(f"max_attempts must be >= 1, got {self.max_attempts}")
        if self.backstep_distance_units <= 0:
            raise ValueError(
                f"backstep_distance_units must be > 0, got {self.backstep_distance_units}"
            )
        if self.camera_sweep_rad <= 0:
            raise ValueError(
                f"camera_sweep_rad must be > 0, got {self.camera_sweep_rad}"
            )
        if self.alternative_radius_units <= 0:
            raise ValueError(
                f"alternative_radius_units must be > 0, got {self.alternative_radius_units}"
            )
        if self.alternative_samples < 2:
            raise ValueError(
                f"alternative_samples must be >= 2, got {self.alternative_samples}"
            )
        if len(self.behavior_weights) != 4:
            raise ValueError(
                f"behavior_weights must have length 4, got {len(self.behavior_weights)}"
            )
        if any(w < 0 for w in self.behavior_weights):
            raise ValueError(
                f"behavior_weights must be non-negative, got {self.behavior_weights}"
            )
        if sum(self.behavior_weights) <= 0:
            raise ValueError(
                f"sum of behavior_weights must be > 0, got {sum(self.behavior_weights)}"
            )


def pick_behavior(
    rng: random.Random,
    config: RecoveryConfig,
) -> RecoveryBehaviorKind:
    """Select a recovery behavior kind using weighted random choice."""
    selected = rng.choices(
        _BEHAVIOR_KINDS_ORDER,
        weights=config.behavior_weights,
        k=1,
    )[0]
    return selected


def backstep_intent(
    *,
    position: tuple[float, float],
    heading: float,
    config: RecoveryConfig,
) -> Intent:
    """Generate a MoveTo intent to a position behind current position given heading."""
    bx = position[0] - math.cos(heading) * config.backstep_distance_units
    by = position[1] - math.sin(heading) * config.backstep_distance_units
    return MoveTo(x=bx, y=by)


def camera_sweep_intent(
    rng: random.Random,
    *,
    config: RecoveryConfig,
) -> Intent:
    """Generate a Turn intent with a uniform random angle in [-camera_sweep_rad, +camera_sweep_rad]."""
    angle = rng.uniform(-config.camera_sweep_rad, config.camera_sweep_rad)
    return Turn(angle_rad=angle)


def jump_intent() -> Intent:
    """Generate a jump intent placeholder.

    The InputDriver protocol from T1.1 does not expose a jump primitive;
    a no-op turn is the safe placeholder used until a later phase extends
    the mapper with a Jump intent.
    """
    return Turn(angle_rad=0.0)


def alternative_waypoint_intent(
    rng: random.Random,
    *,
    position: tuple[float, float],
    config: RecoveryConfig,
) -> Intent:
    """Generate a MoveTo intent to a randomly selected nearby candidate waypoint."""
    candidates: list[MoveTo] = []
    jitter_limit = math.pi / config.alternative_samples
    for i in range(config.alternative_samples):
        base_angle = (2.0 * math.pi * i) / config.alternative_samples
        jitter = rng.uniform(-jitter_limit, jitter_limit)
        angle = base_angle + jitter
        cx = position[0] + math.cos(angle) * config.alternative_radius_units
        cy = position[1] + math.sin(angle) * config.alternative_radius_units
        candidates.append(MoveTo(x=cx, y=cy))
    return rng.choice(candidates)


class RecoveryPlanner:
    """Planner that selects and generates recovery intents based on state and attempt counts."""

    def __init__(
        self,
        *,
        config: RecoveryConfig | None = None,
        position_source: Callable[[], tuple[float, float]] | None = None,
        heading_source: Callable[[], float] | None = None,
    ) -> None:
        self._config = config if config is not None else RecoveryConfig()
        self._position_source = (
            position_source if position_source is not None else (lambda: (0.0, 0.0))
        )
        self._heading_source = (
            heading_source if heading_source is not None else (lambda: 0.0)
        )
        self._last_behavior: RecoveryBehaviorKind | None = None

    def plan(
        self,
        rng: random.Random,
        *,
        attempts_so_far: int,
    ) -> Intent:
        """Plan a recovery intent for the current recovery attempt step."""
        if attempts_so_far >= self._config.max_attempts:
            raise RecoveryError(
                f"Maximum recovery attempts reached ({attempts_so_far} >= {self._config.max_attempts})"
            )

        kind = pick_behavior(rng, self._config)

        if kind == RecoveryBehaviorKind.BACKSTEP:
            pos = self._position_source()
            heading = self._heading_source()
            intent = backstep_intent(position=pos, heading=heading, config=self._config)
        elif kind == RecoveryBehaviorKind.CAMERA_SWEEP:
            intent = camera_sweep_intent(rng, config=self._config)
        elif kind == RecoveryBehaviorKind.JUMP:
            intent = jump_intent()
        elif kind == RecoveryBehaviorKind.ALTERNATIVE_WAYPOINT:
            pos = self._position_source()
            intent = alternative_waypoint_intent(
                rng, position=pos, config=self._config
            )
        else:
            raise RecoveryError(f"Unhandled recovery behavior kind: {kind}")

        self._last_behavior = kind
        return intent

    def last_behavior(self) -> RecoveryBehaviorKind | None:
        """Return the recovery behavior kind chosen during the most recent plan call."""
        return self._last_behavior


class RecoveryBehavior:
    """Behavior protocol adapter delegating STUCK_RECOVERY decisions to RecoveryPlanner."""

    def __init__(
        self,
        *,
        config: RecoveryConfig | None = None,
        position_source: Callable[[], tuple[float, float]] | None = None,
        heading_source: Callable[[], float] | None = None,
    ) -> None:
        self._planner = RecoveryPlanner(
            config=config,
            position_source=position_source,
            heading_source=heading_source,
        )
        self._attempts: int = 0

    def decide(
        self,
        state: FSMState,
        game_state: GameStateLike,
        meta_state: MetaStateLike,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        """Decide a recovery intent when in state STUCK_RECOVERY, or return None."""
        if state != FSMState.STUCK_RECOVERY:
            return None

        intent = self._planner.plan(rng, attempts_so_far=self._attempts)
        self._attempts += 1
        return intent

    def reset_attempts(self) -> None:
        """Reset internal recovery attempts counter to 0.

        The FSM engine or its owner is responsible for calling reset when leaving STUCK_RECOVERY.
        No automatic reset is wired inside this behavior class.
        """
        self._attempts = 0

    def attempts(self) -> int:
        """Return current recovery attempt count."""
        return self._attempts
