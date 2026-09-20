"""
Combat loop behavior for WoW-bot FSM COMBAT state.
"""

import random
from dataclasses import dataclass
from typing import Protocol, runtime_checkable

from wow_bot.actuation.mapper import Intent, MoveTo
from wow_bot.combat.rotation import RotationTable
from wow_bot.combat.targeting import TargetEntityLike, TargetSelector
from wow_bot.executor.states import FSMState


class CombatLoopError(Exception):
    """Raised when an error occurs in the combat loop."""


@runtime_checkable
class CombatStateView(Protocol):
    """
    Unified view of game state consumed by CombatLoop.
    Composes fields required for both targeting and rotation evaluation.
    """

    current_target_id: str | None
    entities: tuple[TargetEntityLike, ...]
    target_in_range: bool
    target_hp_percent: float
    self_hp_percent: float
    resource: float
    resource_max: float
    gcd_ready: bool
    self_x: float
    self_y: float
    target_x: float | None
    target_y: float | None

    def target_has_debuff(self, debuff_id: str, /) -> bool: ...

    def spell_cooldown_ready(self, spell_id: str, /) -> bool: ...


@runtime_checkable
class CombatMetaView(Protocol):
    """
    Structural view of meta state consumed by CombatLoop.
    Optional fields read via getattr.
    """


@dataclass(frozen=True)
class CombatLoopConfig:
    """Configuration options for CombatLoop."""

    engage_distance_units: float = 30.0
    approach_target_distance: float = 2.0
    attackable_spell_id: str = "auto_attack"
    wait_for_gcd_after_cast: bool = True
    consider_target_switch_on_kill: bool = True

    def __post_init__(self) -> None:
        if (
            isinstance(self.engage_distance_units, bool)
            or not isinstance(self.engage_distance_units, (int, float))
            or float(self.engage_distance_units) <= 0.0
        ):
            raise ValueError("engage_distance_units must be > 0")

        if (
            isinstance(self.approach_target_distance, bool)
            or not isinstance(self.approach_target_distance, (int, float))
            or float(self.approach_target_distance) < 0.0
        ):
            raise ValueError("approach_target_distance must be >= 0")

        if float(self.approach_target_distance) >= float(
            self.engage_distance_units
        ):
            raise ValueError(
                "approach_target_distance must be < engage_distance_units"
            )

        if (
            not isinstance(self.attackable_spell_id, str)
            or len(self.attackable_spell_id) == 0
        ):
            raise ValueError("attackable_spell_id must be a non-empty string")


class CombatLoop:
    """
    Combat loop implementation fulfilling the FSM Behavior protocol.

    Pure and deterministic combat decision component.
    """

    def __init__(
        self,
        rotation: RotationTable,
        targeting: TargetSelector,
        *,
        config: CombatLoopConfig | None = None,
    ) -> None:
        if not isinstance(rotation, RotationTable):
            raise TypeError("rotation must be a RotationTable instance")
        if not isinstance(targeting, TargetSelector):
            raise TypeError("targeting must be a TargetSelector instance")
        if config is not None and not isinstance(config, CombatLoopConfig):
            raise TypeError("config must be a CombatLoopConfig instance or None")

        self._rotation = rotation
        self._targeting = targeting
        self._config = config if config is not None else CombatLoopConfig()
        self._last_spell_id: str | None = None
        self._last_reason: str = ""

    def last_spell_id(self) -> str | None:
        """Return the spell id chosen by the most recent successful cast decision, or None."""
        return self._last_spell_id

    def last_reason(self) -> str:
        """Return a short string explaining the last decision path."""
        return self._last_reason

    def _find_entity(
        self,
        entities: tuple[TargetEntityLike, ...],
        entity_id: str,
    ) -> TargetEntityLike | None:
        for entity in entities:
            if entity.entity_id == entity_id:
                return entity
        return None

    def _cast_intent(self, spell_id: str, x: float, y: float) -> MoveTo:
        """
        Return a MoveTo whose x, y equal the caster's current position (a "stand and cast" intent).

        Rationale: the current mapper (T1.3) has no Cast intent; a zero-distance MoveTo is
        a deterministic no-op that the actuator can execute without moving.
        When a future phase adds a Cast intent to the mapper, this helper will be the single place to change.
        """
        return MoveTo(x=x, y=y)

    def decide(
        self,
        state: FSMState,
        game_state: CombatStateView,
        meta_state: CombatMetaView,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        # Safely access optional combat_lock metadata without raising
        _ = getattr(meta_state, "combat_lock", False)

        if state != FSMState.COMBAT:
            self._last_reason = "wrong_state"
            self._last_spell_id = None
            return None

        cfg = self._config
        current_target_id = game_state.current_target_id

        if current_target_id is None:
            decision = self._targeting.select(game_state)
            if decision.entity_id is None:
                self._last_reason = "no_target"
                self._last_spell_id = None
                return None

            target = self._find_entity(game_state.entities, decision.entity_id)
            if (
                target is None
                or game_state.target_x is None
                or game_state.target_y is None
            ):
                self._last_reason = "no_target"
                self._last_spell_id = None
                return None

            self._last_reason = "out_of_range_move"
            self._last_spell_id = None
            return MoveTo(x=game_state.target_x, y=game_state.target_y)

        # current_target_id is set
        target = self._find_entity(game_state.entities, current_target_id)
        if target is None or not target.is_alive:
            self._last_reason = "target_gone"
            self._last_spell_id = None
            return None

        # Distance check
        if target.distance > cfg.engage_distance_units:
            if (
                game_state.target_x is None
                or game_state.target_y is None
            ):
                self._last_reason = "nothing_to_do"
                self._last_spell_id = None
                return None
            self._last_reason = "out_of_range_move"
            self._last_spell_id = None
            return MoveTo(x=game_state.target_x, y=game_state.target_y)

        # In range: consult the rotation
        spell = self._rotation.select(game_state)
        if spell is None:
            if not game_state.gcd_ready and cfg.wait_for_gcd_after_cast:
                self._last_reason = "waiting_gcd"
                self._last_spell_id = None
                return None

            if game_state.target_in_range:
                self._last_reason = "auto_attack"
                self._last_spell_id = cfg.attackable_spell_id
                return self._cast_intent(
                    cfg.attackable_spell_id,
                    game_state.self_x,
                    game_state.self_y,
                )

            self._last_reason = "nothing_to_do"
            self._last_spell_id = None
            return None

        self._last_reason = "cast"
        self._last_spell_id = spell
        return self._cast_intent(spell, game_state.self_x, game_state.self_y)
