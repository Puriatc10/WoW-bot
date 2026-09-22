"""Frozen data container views projecting canonical GameState into consumer shapes.

These dataclasses structurally and type-wise satisfy the eight consumer Protocols
without coupling to them via direct inheritance or imports.
"""

from __future__ import annotations

from dataclasses import dataclass

from wow_bot.executor.states import FSMState

__all__ = [
    "CombatView",
    "EnemyCastView",
    "FSMState",
    "FleeView",
    "LootView",
    "ReactiveView",
    "StrategistView",
    "TargetView",
    "VendorView",
    "WorldSyncEntityView",
    "WorldSyncView",
]


@dataclass(frozen=True)
class WorldSyncEntityView:
    """View of an entity observation consumed by the world sync layer."""

    entity_id: str
    kind: str
    x: float
    y: float
    z: float


@dataclass(frozen=True)
class WorldSyncView:
    """View of GameState structurally compatible with world.sync.GameStateLike.

    player_x, player_y, player_z, and entities are non-Optional in the Protocol.
    target_entity_id is Optional in the Protocol.
    """

    player_x: float
    player_y: float
    player_z: float
    entities: tuple[WorldSyncEntityView, ...] = ()
    target_entity_id: str | None = None


@dataclass(frozen=True)
class StrategistView:
    """View of GameState structurally compatible with strategist.prompts_v2.GameStateView.

    fsm_state uses the exact FSMState enum required by the Protocol.
    current_target_id and target_hp_percent are Optional in the Protocol.
    All other fields are non-Optional in the Protocol.
    """

    player_x: float
    player_y: float
    player_z: float
    self_hp_percent: float
    resource: float
    resource_max: float
    inventory_count: int
    level_or_xp: float
    fsm_state: FSMState
    current_target_id: str | None = None
    target_hp_percent: float | None = None


@dataclass(frozen=True)
class TargetView:
    """View of a target entity structurally compatible with combat.targeting.TargetEntityLike.

    All attributes are non-Optional in the Protocol.
    """

    entity_id: str
    distance: float
    threat: float
    hp_percent: float
    is_attackable: bool
    is_alive: bool
    is_in_combat_with_self: bool


@dataclass(frozen=True)
class CombatView:
    """View of GameState structurally compatible with combat.loop.CombatStateView.

    current_target_id, target_x, and target_y are Optional in the Protocol.
    All other attributes are non-Optional in the Protocol.
    """

    entities: tuple[TargetView, ...]
    target_in_range: bool
    target_hp_percent: float
    self_hp_percent: float
    resource: float
    resource_max: float
    gcd_ready: bool
    self_x: float
    self_y: float
    current_target_id: str | None = None
    target_x: float | None = None
    target_y: float | None = None

    def target_has_debuff(self, debuff_id: str, /) -> bool:
        """Check debuff status; fails loudly as debuffs are unknown to GameState."""
        raise NotImplementedError("target_has_debuff is not supported by canonical GameState")

    def spell_cooldown_ready(self, spell_id: str, /) -> bool:
        """Check cooldown status; fails loudly as cooldowns are unknown to GameState."""
        raise NotImplementedError("spell_cooldown_ready is not supported by canonical GameState")


@dataclass(frozen=True)
class EnemyCastView:
    """View of an incoming enemy cast consumed by reactive combat."""

    caster_entity_id: str
    spell_id: str
    remaining_cast_time_s: float
    is_interruptible: bool


@dataclass(frozen=True)
class ReactiveView:
    """View of GameState structurally compatible with combat.reactive.ReactiveStateView.

    current_target_id is Optional in the Protocol.
    All other attributes are non-Optional in the Protocol.
    """

    self_hp_percent: float
    self_in_combat: bool
    target_in_range: bool
    self_x: float
    self_y: float
    incoming_casts: tuple[EnemyCastView, ...] = ()
    current_target_id: str | None = None

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        """Check cooldown status; fails loudly as cooldowns are unknown to GameState."""
        raise NotImplementedError("spell_cooldown_ready is not supported by canonical GameState")


@dataclass(frozen=True)
class FleeView:
    """View of GameState structurally compatible with combat.flee.FleeStateView.

    current_target_id, target_x, and target_y are Optional in the Protocol.
    All other attributes are non-Optional in the Protocol.
    """

    self_hp_percent: float
    self_x: float
    self_y: float
    adds_count: int
    current_target_id: str | None = None
    target_x: float | None = None
    target_y: float | None = None


@dataclass(frozen=True)
class LootView:
    """View of GameState structurally compatible with farm.loot.LootStateView.

    target_entity_id, target_distance, target_x, target_y, and inventory_max are
    Optional in the Protocol. All other attributes are non-Optional.
    """

    target_is_alive: bool
    target_is_lootable: bool
    self_x: float
    self_y: float
    inventory_count: int
    target_entity_id: str | None = None
    target_distance: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    inventory_max: int | None = None


@dataclass(frozen=True)
class VendorView:
    """View of GameState structurally compatible with farm.vendor.VendorStateView.

    durability_fraction is Optional in the Protocol.
    self_x, self_y, and inventory_count are non-Optional in the Protocol.
    """

    self_x: float
    self_y: float
    inventory_count: int
    durability_fraction: float | None = None
