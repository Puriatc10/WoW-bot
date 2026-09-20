"""
Reactive combat logic for WoW-bot fast loop.

Provides interrupt, defensive, and retreat evaluation as a SignalSource
and FSM Behavior.
"""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Any, Protocol

from wow_bot.actuation.mapper import Intent, MoveTo
from wow_bot.executor.states import FSMState
from wow_bot.reflex.signals import Signal, SignalSource


class ReactiveCombatError(Exception):
    """Raised when an error occurs in reactive combat operations."""


class EnemyCastLike(Protocol):
    """Protocol representing an enemy cast in progress."""

    caster_entity_id: str
    spell_id: str
    remaining_cast_time_s: float  # >= 0.0
    is_interruptible: bool


class ReactiveStateView(Protocol):
    """Protocol representing the state view consumed by ReactiveCombat."""

    self_hp_percent: float  # 0.0 .. 100.0
    self_in_combat: bool
    current_target_id: str | None
    target_in_range: bool
    incoming_casts: tuple[EnemyCastLike, ...]
    spell_cooldown_ready: Callable[[str], bool]
    self_x: float
    self_y: float


@dataclass(frozen=True)
class ReactiveCombatConfig:
    """Configuration options for ReactiveCombat."""

    interrupt_spell_id: str = "interrupt"
    interrupt_min_remaining_s: float = 0.05
    interrupt_max_remaining_s: float = 5.0
    defensive_spell_id: str = "shield"
    defensive_hp_threshold: float = 35.0
    defensive_cooldown_s: float = 30.0
    retreat_distance_units: float = 3.0

    def __post_init__(self) -> None:
        if (
            not isinstance(self.interrupt_spell_id, str)
            or len(self.interrupt_spell_id.strip()) == 0
            or not isinstance(self.defensive_spell_id, str)
            or len(self.defensive_spell_id.strip()) == 0
        ):
            raise ValueError("Spell IDs must be non-empty strings")

        if (
            isinstance(self.interrupt_min_remaining_s, bool)
            or not isinstance(self.interrupt_min_remaining_s, (int, float))
            or float(self.interrupt_min_remaining_s) < 0.0
        ):
            raise ValueError("interrupt_min_remaining_s must be >= 0.0")

        if (
            isinstance(self.interrupt_max_remaining_s, bool)
            or not isinstance(self.interrupt_max_remaining_s, (int, float))
            or float(self.interrupt_max_remaining_s) <= float(self.interrupt_min_remaining_s)
        ):
            raise ValueError("interrupt_max_remaining_s must be > interrupt_min_remaining_s")

        if (
            isinstance(self.defensive_hp_threshold, bool)
            or not isinstance(self.defensive_hp_threshold, (int, float))
            or float(self.defensive_hp_threshold) < 0.0
            or float(self.defensive_hp_threshold) > 100.0
        ):
            raise ValueError("defensive_hp_threshold must be between 0.0 and 100.0")

        if (
            isinstance(self.defensive_cooldown_s, bool)
            or not isinstance(self.defensive_cooldown_s, (int, float))
            or float(self.defensive_cooldown_s) < 0.0
        ):
            raise ValueError("defensive_cooldown_s must be >= 0.0")

        if (
            isinstance(self.retreat_distance_units, bool)
            or not isinstance(self.retreat_distance_units, (int, float))
            or float(self.retreat_distance_units) < 0.0
        ):
            raise ValueError("retreat_distance_units must be >= 0.0")


class ReactiveAction(str, Enum):
    """Action decisions produced by reactive combat evaluation."""

    NONE = "none"
    INTERRUPT = "interrupt"
    DEFENSIVE = "defensive"
    RETREAT = "retreat"


@dataclass(frozen=True)
class ReactiveDecision:
    """Decision output of reactive combat evaluation."""

    action: ReactiveAction
    spell_id: str | None
    intent: Intent | None
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.reason, str) or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")

        if self.action == ReactiveAction.NONE:
            if self.spell_id is not None or self.intent is not None:
                raise ValueError("action == NONE implies spell_id is None and intent is None")

        elif self.action in (ReactiveAction.INTERRUPT, ReactiveAction.DEFENSIVE):
            if self.spell_id is None or self.intent is None:
                raise ValueError(
                    f"action == {self.action.value} implies spell_id is not None and intent is not None"
                )

        elif self.action == ReactiveAction.RETREAT:
            if self.spell_id is not None or self.intent is None:
                raise ValueError("action == RETREAT implies spell_id is None and intent is not None")


class ReactiveCombat:
    """Pure and deterministic reactive combat evaluator."""

    def __init__(
        self,
        *,
        config: ReactiveCombatConfig | None = None,
    ) -> None:
        self._config = config if config is not None else ReactiveCombatConfig()

    def evaluate(
        self,
        state: ReactiveStateView,
    ) -> ReactiveDecision:
        """Evaluate combat state and return a ReactiveDecision based on strict priority."""
        cfg = self._config

        # 1. NOT_APPLICABLE
        if not state.self_in_combat or state.current_target_id is None:
            return ReactiveDecision(
                action=ReactiveAction.NONE,
                spell_id=None,
                intent=None,
                reason="not_in_combat",
            )

        # 2. RETREAT
        if state.self_hp_percent <= cfg.defensive_hp_threshold:
            if state.spell_cooldown_ready(cfg.defensive_spell_id):
                intent = MoveTo(
                    x=state.self_x - cfg.retreat_distance_units,
                    y=state.self_y,
                )
                return ReactiveDecision(
                    action=ReactiveAction.RETREAT,
                    spell_id=None,
                    intent=intent,
                    reason="low_hp_retreat",
                )

        # 3. INTERRUPT
        if state.spell_cooldown_ready(cfg.interrupt_spell_id):
            for entry in state.incoming_casts:
                if (
                    entry.is_interruptible
                    and cfg.interrupt_min_remaining_s
                    <= entry.remaining_cast_time_s
                    <= cfg.interrupt_max_remaining_s
                ):
                    intent = MoveTo(x=state.self_x, y=state.self_y)
                    return ReactiveDecision(
                        action=ReactiveAction.INTERRUPT,
                        spell_id=cfg.interrupt_spell_id,
                        intent=intent,
                        reason=f"interrupt:{entry.spell_id}",
                    )

        # 4. DEFENSIVE
        if state.self_hp_percent <= cfg.defensive_hp_threshold:
            if state.spell_cooldown_ready(cfg.defensive_spell_id):
                intent = MoveTo(x=state.self_x, y=state.self_y)
                return ReactiveDecision(
                    action=ReactiveAction.DEFENSIVE,
                    spell_id=cfg.defensive_spell_id,
                    intent=intent,
                    reason="defensive_low_hp",
                )
            return ReactiveDecision(
                action=ReactiveAction.NONE,
                spell_id=None,
                intent=None,
                reason="defensive_on_cooldown",
            )

        # 5. NONE
        return ReactiveDecision(
            action=ReactiveAction.NONE,
            spell_id=None,
            intent=None,
            reason="nothing_to_do",
        )

    def to_signal(
        self,
        decision: ReactiveDecision,
        *,
        ts: float,
    ) -> Signal | None:
        """Convert a ReactiveDecision into a reflex Signal."""
        if decision.action == ReactiveAction.NONE:
            return None

        if decision.action == ReactiveAction.INTERRUPT:
            return Signal(
                name=self.name_interrupt(),
                payload={"spell_id": decision.spell_id},
                ts=ts,
            )

        if decision.action == ReactiveAction.DEFENSIVE:
            return Signal(
                name=self.name_defensive(),
                payload={"spell_id": decision.spell_id},
                ts=ts,
            )

        if decision.action == ReactiveAction.RETREAT:
            return Signal(
                name=self.name_retreat(),
                payload={"distance": self._config.retreat_distance_units},
                ts=ts,
            )

        return None

    @classmethod
    def name_interrupt(cls) -> str:
        """Return signal name for interrupt action."""
        return "combat_interrupt"

    @classmethod
    def name_defensive(cls) -> str:
        """Return signal name for defensive action."""
        return "combat_defensive"

    @classmethod
    def name_retreat(cls) -> str:
        """Return signal name for retreat action."""
        return "combat_retreat"


class ReactiveCombatSource(SignalSource):
    """SignalSource producing reactive combat signals for the reflex loop."""

    def __init__(
        self,
        state_source: Callable[[], ReactiveStateView],
        *,
        config: ReactiveCombatConfig | None = None,
    ) -> None:
        self._state_source = state_source
        self._combat = ReactiveCombat(config=config)
        self._last_decision: ReactiveDecision | None = None

    @property
    def last_decision(self) -> ReactiveDecision | None:
        """Return the last evaluated ReactiveDecision."""
        return self._last_decision

    def poll(self, now: float) -> list[Signal]:
        """Poll the state source and return combat signals."""
        state = self._state_source()
        decision = self._combat.evaluate(state)
        self._last_decision = decision
        signal = self._combat.to_signal(decision, ts=now)
        if signal is not None:
            return [signal]
        return []


class ReactiveCombatBehavior:
    """FSM Behavior implementation executing reactive combat decisions."""

    def __init__(
        self,
        *,
        config: ReactiveCombatConfig | None = None,
    ) -> None:
        self._combat = ReactiveCombat(config=config)
        self._last_decision: ReactiveDecision | None = None

    @property
    def last_decision(self) -> ReactiveDecision | None:
        """Return the last evaluated ReactiveDecision."""
        return self._last_decision

    def decide(
        self,
        state: FSMState,
        game_state: Any,
        meta_state: Any,
        now: float,
        rng: Any,
    ) -> Intent | None:
        """Decide an intent for FSM COMBAT state."""
        if state != FSMState.COMBAT:
            return None

        decision = self._combat.evaluate(game_state)
        self._last_decision = decision
        return decision.intent
