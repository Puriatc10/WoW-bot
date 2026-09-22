"""GameStateAdapter projecting one canonical GameState snapshot to consumer views.

HP/MP Unit Conversion Policy:
Canonical GameState tracks health and mana as fractions in the closed unit
interval [0.0, 1.0] (hp_pct, mana_pct). Downstream consumer views expect
percentage values in [0.0, 100.0] (e.g. self_hp_percent, resource).
Conversion uses round-half-to-even (banker's rounding) at 1 decimal place,
implemented via Decimal with ROUND_HALF_EVEN to guarantee mathematically
exact rounding without binary floating-point representation artifacts.

Fail-Loud Unknown / Missing Values Policy:
The adapter never fabricates plausible defaults (e.g. no (0,0) fallback for
missing position, no zero heading, no synthetic entity IDs). Whenever any
field that the target Protocol declares as non-Optional cannot be supplied,
the adapter raises AdapterIncompleteError at projection time. Fields that the
target Protocol declares as Optional (| None) remain Optional in the returned
views and may be None.
"""

from __future__ import annotations

from dataclasses import dataclass
from decimal import ROUND_HALF_EVEN, Decimal

from wow_bot.perception.views import (
    CombatView,
    EnemyCastView,
    FleeView,
    FSMState,
    LootView,
    ReactiveView,
    StrategistView,
    TargetView,
    VendorView,
    WorldSyncEntityView,
    WorldSyncView,
)
from wow_bot.shared.interfaces import GameState


class AdapterIncompleteError(Exception):
    """Raised when a non-Optional field required by a consumer view cannot be supplied."""


def fraction_to_percent(fraction: float | None) -> float | None:
    """Convert a [0.0, 1.0] fraction to [0.0, 100.0] percent using round-half-to-even.

    Rounds to 1 decimal place using Decimal.quantize with ROUND_HALF_EVEN.
    Returns None if input is None.
    """
    if fraction is None:
        return None
    d = Decimal(str(fraction)) * Decimal(100)
    return float(d.quantize(Decimal("0.1"), rounding=ROUND_HALF_EVEN))


@dataclass(frozen=True)
class GameStateAdapter:
    """Pure adapter projecting a canonical GameState into consumer views."""

    snapshot: GameState

    def _extract_required_coords(self, view_name: str) -> tuple[float, float]:
        """Extract (x, y) coordinates from snapshot or raise AdapterIncompleteError."""
        pos = getattr(self.snapshot, "position", None)
        if pos is None or not isinstance(pos, (tuple, list)) or len(pos) < 2:
            raise AdapterIncompleteError(
                f"{view_name} requires non-Optional position coordinates (x, y)"
            )
        try:
            return float(pos[0]), float(pos[1])
        except (ValueError, TypeError) as exc:
            raise AdapterIncompleteError(
                f"{view_name} has invalid non-numeric position coordinates: {pos}"
            ) from exc

    def to_world_sync_view(self) -> WorldSyncView:
        """Project snapshot to WorldSyncView satisfying world.sync.GameStateLike.

        Raises AdapterIncompleteError if position or the non-Optional player_z
        cannot be supplied from GameState.
        """
        x, y = self._extract_required_coords("WorldSyncView")

        z = getattr(self.snapshot, "player_z", None)
        if z is None:
            raise AdapterIncompleteError("WorldSyncView requires non-Optional player_z")
        z = float(z)

        # `entities` has no canonical channel: GameState exposes `enemies`, whose
        # EnemyInfo shape is disjoint from EntityLike. An empty tuple is reported
        # as "no entity observations available", never fabricated entity values.
        raw_entities = getattr(self.snapshot, "entities", ())
        entities: tuple[WorldSyncEntityView, ...] = tuple(raw_entities)

        target = getattr(self.snapshot, "target", None)
        target_id: str | None = None
        if target is not None and getattr(target, "name", None):
            target_id = str(target.name)
        else:
            raw_tid = getattr(self.snapshot, "target_entity_id", None)
            if raw_tid is not None:
                target_id = str(raw_tid)

        return WorldSyncView(
            player_x=x,
            player_y=y,
            player_z=z,
            entities=entities,
            target_entity_id=target_id,
        )

    def to_strategist_view(self, *, fsm_state: FSMState | str) -> StrategistView:
        """Project snapshot to StrategistView satisfying strategist.prompts_v2.GameStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("StrategistView")

        if fsm_state is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional fsm_state")
        try:
            state_enum = FSMState(fsm_state)
        except (ValueError, TypeError) as exc:
            raise AdapterIncompleteError(f"Invalid fsm_state for StrategistView: {fsm_state}") from exc

        hp = fraction_to_percent(getattr(self.snapshot, "hp_pct", None))
        if hp is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional self_hp_percent")

        mana = fraction_to_percent(getattr(self.snapshot, "mana_pct", None))
        if mana is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional resource")

        res_max = getattr(self.snapshot, "resource_max", None)
        if res_max is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional resource_max")

        inv = getattr(self.snapshot, "inventory_count", None)
        if inv is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional inventory_count")

        lvl = getattr(self.snapshot, "level_or_xp", None)
        if lvl is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional level_or_xp")

        z = getattr(self.snapshot, "player_z", None)
        if z is None:
            raise AdapterIncompleteError("StrategistView requires non-Optional player_z")
        z = float(z)
        target = getattr(self.snapshot, "target", None)
        target_id = str(target.name) if target and getattr(target, "name", None) else None
        target_hp = fraction_to_percent(getattr(target, "hp_pct", None)) if target else None

        return StrategistView(
            player_x=x,
            player_y=y,
            player_z=z,
            self_hp_percent=hp,
            resource=mana,
            resource_max=float(res_max),
            inventory_count=int(inv),
            level_or_xp=float(lvl),
            fsm_state=state_enum,
            current_target_id=target_id,
            target_hp_percent=target_hp,
        )

    def to_combat_view(self) -> CombatView:
        """Project snapshot to CombatView satisfying combat.loop.CombatStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("CombatView")

        hp = fraction_to_percent(getattr(self.snapshot, "hp_pct", None))
        if hp is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional self_hp_percent")

        mana = fraction_to_percent(getattr(self.snapshot, "mana_pct", None))
        if mana is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional resource")

        res_max = getattr(self.snapshot, "resource_max", None)
        if res_max is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional resource_max")

        in_range = getattr(self.snapshot, "target_in_range", None)
        if in_range is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional target_in_range")

        gcd = getattr(self.snapshot, "gcd_ready", None)
        if gcd is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional gcd_ready")

        target = getattr(self.snapshot, "target", None)
        target_hp = (
            fraction_to_percent(getattr(target, "hp_pct", None))
            if target
            else getattr(self.snapshot, "target_hp_percent", None)
        )
        if target_hp is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional target_hp_percent")

        raw_entities = getattr(self.snapshot, "entities", None)
        if raw_entities is not None:
            entities = tuple(raw_entities)
        else:
            # No try/except here on purpose: if the targeting projection cannot
            # supply entities, that structural mismatch must surface instead of
            # being silently replaced with an empty tuple.
            entities = self.to_targeting_views()

        target_id = str(target.name) if target and getattr(target, "name", None) else None
        target_x = getattr(self.snapshot, "target_x", None)
        target_y = getattr(self.snapshot, "target_y", None)

        return CombatView(
            entities=entities,
            target_in_range=bool(in_range),
            target_hp_percent=float(target_hp),
            self_hp_percent=hp,
            resource=mana,
            resource_max=float(res_max),
            gcd_ready=bool(gcd),
            self_x=x,
            self_y=y,
            current_target_id=target_id,
            target_x=float(target_x) if target_x is not None else None,
            target_y=float(target_y) if target_y is not None else None,
        )

    def to_targeting_views(self) -> tuple[TargetView, ...]:
        """Project snapshot targets to TargetView tuple satisfying TargetEntityLike.

        Raises AdapterIncompleteError if target is present but missing non-Optional fields.
        """
        target = getattr(self.snapshot, "target", None)
        if target is None:
            return ()

        target_name = getattr(target, "name", None)
        if target_name is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional entity_id")

        dist = getattr(target, "distance_estimate", getattr(self.snapshot, "distance", None))
        if dist is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional distance")

        threat = getattr(target, "threat", getattr(self.snapshot, "threat", None))
        if threat is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional threat")

        hp_raw = getattr(target, "hp_pct", getattr(self.snapshot, "hp_percent", None))
        hp_pct = fraction_to_percent(hp_raw)
        if hp_pct is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional hp_percent")

        is_attackable = getattr(target, "is_attackable", getattr(self.snapshot, "is_attackable", None))
        if is_attackable is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional is_attackable")

        is_alive = getattr(target, "is_alive", getattr(self.snapshot, "is_alive", None))
        if is_alive is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional is_alive")

        is_in_combat = getattr(
            target, "is_in_combat_with_self", getattr(self.snapshot, "is_in_combat_with_self", None)
        )
        if is_in_combat is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional is_in_combat_with_self")

        return (
            TargetView(
                entity_id=str(target_name),
                distance=float(dist),
                threat=float(threat),
                hp_percent=float(hp_pct),
                is_attackable=bool(is_attackable),
                is_alive=bool(is_alive),
                is_in_combat_with_self=bool(is_in_combat),
            ),
        )

    def to_reactive_view(self) -> ReactiveView:
        """Project snapshot to ReactiveView satisfying combat.reactive.ReactiveStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("ReactiveView")

        hp = fraction_to_percent(getattr(self.snapshot, "hp_pct", None))
        if hp is None:
            raise AdapterIncompleteError("ReactiveStateView requires non-Optional self_hp_percent")

        in_combat = getattr(self.snapshot, "in_combat", None)
        if in_combat is None:
            raise AdapterIncompleteError("ReactiveStateView requires non-Optional self_in_combat")

        in_range = getattr(self.snapshot, "target_in_range", None)
        if in_range is None:
            raise AdapterIncompleteError("ReactiveStateView requires non-Optional target_in_range")

        raw_casts = getattr(self.snapshot, "incoming_casts", ())
        incoming_casts: tuple[EnemyCastView, ...] = tuple(raw_casts)

        target = getattr(self.snapshot, "target", None)
        target_id = str(target.name) if target and getattr(target, "name", None) else None

        return ReactiveView(
            self_hp_percent=hp,
            self_in_combat=bool(in_combat),
            target_in_range=bool(in_range),
            self_x=x,
            self_y=y,
            incoming_casts=incoming_casts,
            current_target_id=target_id,
        )

    def to_flee_view(self) -> FleeView:
        """Project snapshot to FleeView satisfying combat.flee.FleeStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("FleeView")

        hp = fraction_to_percent(getattr(self.snapshot, "hp_pct", None))
        if hp is None:
            raise AdapterIncompleteError("FleeStateView requires non-Optional self_hp_percent")

        adds = getattr(self.snapshot, "adds_count", None)
        if adds is None:
            raise AdapterIncompleteError("FleeStateView requires non-Optional adds_count")

        target = getattr(self.snapshot, "target", None)
        target_id = str(target.name) if target and getattr(target, "name", None) else None
        target_x = getattr(self.snapshot, "target_x", None)
        target_y = getattr(self.snapshot, "target_y", None)

        return FleeView(
            self_hp_percent=hp,
            self_x=x,
            self_y=y,
            adds_count=int(adds),
            current_target_id=target_id,
            target_x=float(target_x) if target_x is not None else None,
            target_y=float(target_y) if target_y is not None else None,
        )

    def to_loot_view(self) -> LootView:
        """Project snapshot to LootView satisfying farm.loot.LootStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("LootView")

        alive = getattr(self.snapshot, "target_is_alive", None)
        if alive is None:
            raise AdapterIncompleteError("LootStateView requires non-Optional target_is_alive")

        lootable = getattr(self.snapshot, "target_is_lootable", None)
        if lootable is None:
            raise AdapterIncompleteError("LootStateView requires non-Optional target_is_lootable")

        inv = getattr(self.snapshot, "inventory_count", None)
        if inv is None:
            raise AdapterIncompleteError("LootStateView requires non-Optional inventory_count")

        target = getattr(self.snapshot, "target", None)
        target_id = str(target.name) if target and getattr(target, "name", None) else None
        target_dist = (
            getattr(target, "distance_estimate", getattr(self.snapshot, "target_distance", None))
            if target
            else getattr(self.snapshot, "target_distance", None)
        )
        target_x = getattr(self.snapshot, "target_x", None)
        target_y = getattr(self.snapshot, "target_y", None)
        inv_max = getattr(self.snapshot, "inventory_max", None)

        return LootView(
            target_is_alive=bool(alive),
            target_is_lootable=bool(lootable),
            self_x=x,
            self_y=y,
            inventory_count=int(inv),
            target_entity_id=target_id,
            target_distance=float(target_dist) if target_dist is not None else None,
            target_x=float(target_x) if target_x is not None else None,
            target_y=float(target_y) if target_y is not None else None,
            inventory_max=int(inv_max) if inv_max is not None else None,
        )

    def to_vendor_view(self) -> VendorView:
        """Project snapshot to VendorView satisfying farm.vendor.VendorStateView.

        Raises AdapterIncompleteError if any non-Optional field cannot be supplied.
        """
        x, y = self._extract_required_coords("VendorView")

        inv = getattr(self.snapshot, "inventory_count", None)
        if inv is None:
            raise AdapterIncompleteError("VendorStateView requires non-Optional inventory_count")

        dur = getattr(self.snapshot, "durability_fraction", None)

        return VendorView(
            self_x=x,
            self_y=y,
            inventory_count=int(inv),
            durability_fraction=float(dur) if dur is not None else None,
        )
