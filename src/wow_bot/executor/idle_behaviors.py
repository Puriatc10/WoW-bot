"""Synthetic idle behavior intents for research simulation (Task 5.4).

This module models high-level idle behavior decisions as symbolic intents
(e.g., camera wander, sudden pause, inventory check, social emote, angled movement).
It performs mathematical and stochastic decision modeling only; it does not execute
physical keyboard or mouse inputs, sleep, or interact with game processes or controllers.

Public API:
    - :class:`IdleBehavior`
    - :class:`IdleIntent`
    - :class:`IdleBehaviorEngine`
    - :func:`camera_wander`
    - :func:`sudden_pause`
    - :func:`inventory_check`
    - :func:`social_emote`
    - :func:`angled_movement`
    - :data:`IDLE_TRIGGER_PROBABILITY`
    - :data:`SOCIAL_EMOTES`
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Final

import numpy as np

from wow_bot.shared.logger import get_logger

log = get_logger("IDLE_BEHAVIORS")

#: Base trigger probability per eligible synthetic tick.
IDLE_TRIGGER_PROBABILITY: Final[float] = 0.08

#: Allowed symbolic emotes for social behavior modeling.
SOCIAL_EMOTES: Final[tuple[str, ...]] = ("wave", "laugh")


class IdleBehavior(Enum):
    """Enumeration of supported synthetic idle behavior types."""

    CAMERA_WANDER = auto()
    SUDDEN_PAUSE = auto()
    INVENTORY_CHECK = auto()
    SOCIAL_EMOTE = auto()
    ANGLED_MOVEMENT = auto()


@dataclass(frozen=True)
class IdleIntent:
    """Immutable representation of a symbolic idle behavior decision."""

    behavior: IdleBehavior
    emote: str | None = None


def camera_wander() -> IdleIntent:
    """Construct an abstract camera-look-around intent.

    Returns:
        IdleIntent with behavior CAMERA_WANDER.
    """
    return IdleIntent(behavior=IdleBehavior.CAMERA_WANDER)


def sudden_pause() -> IdleIntent:
    """Construct an intent to temporarily pause current activity.

    Returns:
        IdleIntent with behavior SUDDEN_PAUSE.
    """
    return IdleIntent(behavior=IdleBehavior.SUDDEN_PAUSE)


def inventory_check() -> IdleIntent:
    """Construct an intent to briefly inspect inventory in simulation.

    Returns:
        IdleIntent with behavior INVENTORY_CHECK.
    """
    return IdleIntent(behavior=IdleBehavior.INVENTORY_CHECK)


def social_emote(
    *,
    rng: np.random.Generator | None = None,
) -> IdleIntent:
    """Construct a symbolic social emote intent selecting from allowed emotes.

    Args:
        rng: Optional explicit NumPy Generator for deterministic sampling.

    Returns:
        IdleIntent with behavior SOCIAL_EMOTE and chosen symbolic emote.
    """
    generator = rng if rng is not None else np.random.default_rng()
    chosen_emote = str(generator.choice(np.array(SOCIAL_EMOTES, dtype=object)))
    return IdleIntent(behavior=IdleBehavior.SOCIAL_EMOTE, emote=chosen_emote)


def angled_movement() -> IdleIntent:
    """Construct an intent preferring non-direct movement style in future paths.

    Returns:
        IdleIntent with behavior ANGLED_MOVEMENT.
    """
    return IdleIntent(behavior=IdleBehavior.ANGLED_MOVEMENT)


def _validate_curiosity(curiosity: object) -> float:
    """Validate that curiosity is a finite numeric float in [0.0, 1.0]."""
    if isinstance(curiosity, bool) or not isinstance(curiosity, (int, float)):
        raise ValueError(  # noqa: TRY004
            f"curiosity must be a numeric float, got {type(curiosity).__name__} ({curiosity!r})"
        )
    float_val = float(curiosity)
    if not math.isfinite(float_val):
        raise ValueError(f"curiosity must be a finite number, got {curiosity!r}")
    if float_val < 0.0 or float_val > 1.0:
        raise ValueError(
            f"curiosity must be in range [0.0, 1.0], got {float_val}"
        )
    return float_val


class IdleBehaviorEngine:
    """Stochastic selector and engine for synthetic idle behavior intents."""

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
    ) -> None:
        """Initialize IdleBehaviorEngine.

        Args:
            rng: Optional explicit NumPy Generator for deterministic testing.
        """
        self._rng: np.random.Generator = (
            rng if rng is not None else np.random.default_rng()
        )

    def maybe_generate(
        self,
        *,
        curiosity: float,
    ) -> IdleIntent | None:
        """Evaluate one synthetic tick and decide whether to fire an idle intent.

        Args:
            curiosity: Curiosity drive level in range [0.0, 1.0].

        Returns:
            IdleIntent if triggered on this tick, or None.

        Raises:
            ValueError: If curiosity fails validation.
        """
        valid_curiosity = _validate_curiosity(curiosity)

        if float(self._rng.random()) >= IDLE_TRIGGER_PROBABILITY:
            return None

        # Calculate behavior selection weights
        camera_weight = 0.20 + 0.50 * (1.0 - valid_curiosity)
        pause_weight = 0.25
        inventory_weight = 0.20
        emote_weight = 0.15
        angled_move_weight = 0.20

        raw_weights = np.array(
            [camera_weight, pause_weight, inventory_weight, emote_weight, angled_move_weight],
            dtype=float,
        )
        norm_weights = raw_weights / np.sum(raw_weights)

        behaviors: Sequence[IdleBehavior] = (
            IdleBehavior.CAMERA_WANDER,
            IdleBehavior.SUDDEN_PAUSE,
            IdleBehavior.INVENTORY_CHECK,
            IdleBehavior.SOCIAL_EMOTE,
            IdleBehavior.ANGLED_MOVEMENT,
        )

        chosen_behavior = self._rng.choice(
            np.array(behaviors, dtype=object),
            p=norm_weights,
        )

        intent: IdleIntent
        if chosen_behavior == IdleBehavior.CAMERA_WANDER:
            intent = camera_wander()
        elif chosen_behavior == IdleBehavior.SUDDEN_PAUSE:
            intent = sudden_pause()
        elif chosen_behavior == IdleBehavior.INVENTORY_CHECK:
            intent = inventory_check()
        elif chosen_behavior == IdleBehavior.SOCIAL_EMOTE:
            intent = social_emote(rng=self._rng)
        elif chosen_behavior == IdleBehavior.ANGLED_MOVEMENT:
            intent = angled_movement()
        else:  # pragma: no cover
            raise ValueError(f"Unknown behavior: {chosen_behavior}")

        log.debug(
            f"Fired idle behavior intent: behavior={intent.behavior.name} "
            f"curiosity={valid_curiosity:.2f} emote={intent.emote}"
        )
        return intent
