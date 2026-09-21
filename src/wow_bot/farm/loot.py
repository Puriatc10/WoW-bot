"""Loot and inventory management module for WoW-bot farm subsystem.

Provides corpse looting behavior for FSMState.LOOTING and an edge-triggered inventory capacity tracker.

Design notes:
- Stand-and-loot convention:
  Looting in the current mapper (T1.3) has no dedicated "loot" primitive. Following the same
  convention as T6.3's "stand and cast" MoveTo, the loot behavior expresses looting as a
  zero-distance MoveTo at the corpse position (i.e. self position, once the agent is close
  enough). This is a deterministic no-op for the actuator that will be replaced in a future phase
  when the mapper gains a ClickAt or Loot intent.
- Attempt pruning policy:
  The LootController tracks per-corpse attempt counts in a bounded dictionary. Entries are pruned
  when a corpse is no longer the current target for more than `max_attempts_per_corpse * 4` calls.
  This bound is enforced internally to prevent unbounded memory growth.
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from wow_bot.actuation.mapper import Intent, MoveTo
from wow_bot.executor.states import FSMState

if TYPE_CHECKING:
    import random

    from wow_bot.session import Session


class LootError(Exception):
    """Raised when a loot or inventory tracking error occurs."""


class LootStateView(Protocol):
    """Protocol defining state fields required for corpse looting evaluation."""

    target_entity_id: str | None
    target_is_alive: bool
    target_is_lootable: bool
    target_distance: float | None
    self_x: float
    self_y: float
    target_x: float | None
    target_y: float | None
    inventory_count: int
    inventory_max: int | None


@dataclass(frozen=True)
class LootConfig:
    """Configuration options for looting reach, retry bounds, and inventory thresholds."""

    loot_reach_units: float = 2.5
    max_attempts_per_corpse: int = 5
    inventory_full_absolute: int = 30
    inventory_clear_fraction: float = 0.8

    def __post_init__(self) -> None:
        """Validate configuration bounds and types."""
        if (
            isinstance(self.loot_reach_units, bool)
            or not isinstance(self.loot_reach_units, (int, float))
            or not math.isfinite(self.loot_reach_units)
            or float(self.loot_reach_units) <= 0.0
        ):
            raise ValueError("loot_reach_units must be > 0.0")

        if (
            isinstance(self.max_attempts_per_corpse, bool)
            or not isinstance(self.max_attempts_per_corpse, int)
            or self.max_attempts_per_corpse < 1
        ):
            raise ValueError("max_attempts_per_corpse must be >= 1")

        if (
            isinstance(self.inventory_full_absolute, bool)
            or not isinstance(self.inventory_full_absolute, int)
            or self.inventory_full_absolute < 1
        ):
            raise ValueError("inventory_full_absolute must be >= 1")

        if (
            isinstance(self.inventory_clear_fraction, bool)
            or not isinstance(self.inventory_clear_fraction, (int, float))
            or not math.isfinite(self.inventory_clear_fraction)
            or not (0.0 < float(self.inventory_clear_fraction) < 1.0)
        ):
            raise ValueError("inventory_clear_fraction must be in (0.0, 1.0)")


class LootStatus(str, enum.Enum):
    """Outcome status of a loot evaluation step."""

    SUCCESS = "success"
    NO_TARGET = "no_target"
    TARGET_ALIVE = "target_alive"
    TARGET_NOT_LOOTABLE = "target_not_lootable"
    OUT_OF_REACH = "out_of_reach"
    ATTEMPTS_EXCEEDED = "attempts_exceeded"


@dataclass(frozen=True)
class LootDecision:
    """Deterministic loot decision output."""

    status: LootStatus
    intent: Intent | None
    reason: str

    def __post_init__(self) -> None:
        """Validate decision invariants."""
        if self.status in (LootStatus.SUCCESS, LootStatus.OUT_OF_REACH) and self.intent is None:
            raise ValueError(f"status {self.status.value} requires a non-None intent")

        if self.status in (
            LootStatus.NO_TARGET,
            LootStatus.TARGET_ALIVE,
            LootStatus.TARGET_NOT_LOOTABLE,
            LootStatus.ATTEMPTS_EXCEEDED,
        ) and self.intent is not None:
            raise ValueError(f"status {self.status.value} requires intent to be None")

        if not isinstance(self.reason, str) or len(self.reason.strip()) == 0:
            raise ValueError("reason is a non-empty string")


class LootController:
    """FSM behavior controller for corpse looting in FSMState.LOOTING."""

    def __init__(
        self,
        *,
        config: LootConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self._config = config if config is not None else LootConfig()
        self._session = session
        self._attempts: dict[str, int] = {}
        self._stale_ticks: dict[str, int] = {}
        self._last_status: LootStatus | None = None
        self._last_entity_id: str | None = None

    def last_status(self) -> LootStatus | None:
        """Return the most recent LootStatus or None."""
        return self._last_status

    def last_entity_id(self) -> str | None:
        """Return the most recent target entity_id or None."""
        return self._last_entity_id

    def attempts_for(self, entity_id: str) -> int:
        """Return the attempt count for a given entity_id."""
        return self._attempts.get(entity_id, 0)

    def reset(self) -> None:
        """Clear attempt history and last status/entity fields."""
        self._attempts.clear()
        self._stale_ticks.clear()
        self._last_status = None
        self._last_entity_id = None

    def _prune_stale(self, current_id: str | None) -> None:
        """Prune entity attempt entries that have been inactive for too many calls."""
        max_stale = self._config.max_attempts_per_corpse * 4
        to_delete: list[str] = []

        for entity_id in list(self._attempts.keys()):
            if entity_id != current_id:
                stale_cnt = self._stale_ticks.get(entity_id, 0) + 1
                self._stale_ticks[entity_id] = stale_cnt
                if stale_cnt > max_stale:
                    to_delete.append(entity_id)
            else:
                self._stale_ticks[current_id] = 0

        for entity_id in to_delete:
            self._attempts.pop(entity_id, None)
            self._stale_ticks.pop(entity_id, None)

    def decide(
        self,
        state: FSMState,
        game_state: Any,
        meta_state: Any,
        now: float,
        rng: random.Random,
    ) -> Intent | None:
        """Decide looting action based on current FSMState and LootStateView."""
        if state != FSMState.LOOTING:
            return None

        current_id = game_state.target_entity_id
        self._last_entity_id = current_id
        self._prune_stale(current_id)

        if current_id is None:
            self._last_status = LootStatus.NO_TARGET
            if self._session is not None:
                self._session.write_event({"event": "loot_skipped", "reason": "no_target"})
            return None

        if game_state.target_is_alive:
            self._last_status = LootStatus.TARGET_ALIVE
            if self._session is not None:
                self._session.write_event({
                    "event": "loot_skipped",
                    "reason": "target_alive",
                    "entity_id": current_id,
                })
            return None

        if not game_state.target_is_lootable:
            self._last_status = LootStatus.TARGET_NOT_LOOTABLE
            if self._session is not None:
                self._session.write_event({
                    "event": "loot_skipped",
                    "reason": "target_not_lootable",
                    "entity_id": current_id,
                })
            return None

        distance = game_state.target_distance
        if distance is None:
            self._last_status = LootStatus.OUT_OF_REACH
            return None

        if distance > self._config.loot_reach_units:
            if game_state.target_x is None or game_state.target_y is None:
                self._last_status = LootStatus.OUT_OF_REACH
                return None
            intent: Intent = MoveTo(x=game_state.target_x, y=game_state.target_y)
            self._last_status = LootStatus.OUT_OF_REACH
            if self._session is not None:
                self._session.write_event({
                    "event": "loot_approach",
                    "entity_id": current_id,
                    "distance": distance,
                })
            return intent

        attempts = self._attempts.get(current_id, 0) + 1
        self._attempts[current_id] = attempts

        if attempts > self._config.max_attempts_per_corpse:
            self._last_status = LootStatus.ATTEMPTS_EXCEEDED
            if self._session is not None:
                self._session.write_event({
                    "event": "loot_failed",
                    "entity_id": current_id,
                    "attempts": attempts,
                })
            return None

        intent = MoveTo(x=game_state.self_x, y=game_state.self_y)
        self._last_status = LootStatus.SUCCESS
        if self._session is not None:
            self._session.write_event({
                "event": "loot_attempt",
                "entity_id": current_id,
                "attempts": attempts,
            })
        return intent


class InventoryTracker:
    """Stateful observer tracking inventory capacity and emitting hysteresis edge events."""

    def __init__(
        self,
        *,
        config: LootConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self._config = config if config is not None else LootConfig()
        self._session = session
        self._is_full: bool = False

    def is_full(self) -> bool:
        """Return True if inventory is currently considered full."""
        return self._is_full

    def reset(self) -> None:
        """Reset state to not-full."""
        self._is_full = False

    def full_threshold(self, inventory_max: int | None) -> int:
        """Return the full_at threshold for the given inventory_max."""
        if inventory_max is not None:
            return inventory_max
        return self._config.inventory_full_absolute

    def observe(
        self,
        *,
        inventory_count: int,
        inventory_max: int | None,
        now: float,
    ) -> bool:
        """Observe inventory count and max, update full status, and emit edge events."""
        if (
            isinstance(inventory_count, bool)
            or not isinstance(inventory_count, int)
            or inventory_count < 0
        ):
            raise LootError(f"inventory_count must be >= 0, got {inventory_count}")

        if inventory_max is not None and (
            isinstance(inventory_max, bool)
            or not isinstance(inventory_max, int)
            or inventory_max <= 0
        ):
            raise LootError(f"inventory_max must be > 0 if specified, got {inventory_max}")

        full_at = self.full_threshold(inventory_max)
        if inventory_max is not None:
            clear_at = math.floor(inventory_max * self._config.inventory_clear_fraction)
        else:
            clear_at = max(0, full_at - 1)

        if not self._is_full:
            full_now = inventory_count >= full_at
        else:
            full_now = inventory_count > clear_at

        if not self._is_full and full_now and self._session is not None:
            self._session.write_event({
                "event": "inventory_full",
                "inventory_count": inventory_count,
                "inventory_max": inventory_max,
                "full_at": full_at,
            })
        elif self._is_full and not full_now and self._session is not None:
            self._session.write_event({
                "event": "inventory_cleared",
                "inventory_count": inventory_count,
            })

        self._is_full = full_now
        return self._is_full
