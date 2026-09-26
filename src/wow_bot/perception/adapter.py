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
from typing import Any

from wow_bot.perception.context import RuntimeContext
from wow_bot.perception.resource_table import EMPTY_RESOURCE_TABLE, ResourceTable
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
from wow_bot.shared.interfaces import EnemyInfo, GameState


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
class AdapterDerivationConfig:
    """Inputs the adapter needs to derive quantities ``GameState`` does not store.

    ``engage_distance_units`` is combat policy owned by
    ``combat.loop.CombatLoopConfig`` (default ``30.0`` there). It is **injected**
    rather than defaulted here on purpose: a second hardcoded copy would
    silently diverge from the combat engine's, and ADR-002 requires that
    neither ``engage_distance_units`` nor ``adds_threshold`` be hardcoded.

    When no config is supplied, ``target_in_range`` is not derivable and the
    projections that need it stay fail-loud. That is deliberate: the adapter
    must not invent an engagement threshold.
    """

    engage_distance_units: float

    def __post_init__(self) -> None:
        if self.engage_distance_units <= 0.0:
            raise ValueError(
                "engage_distance_units must be positive, "
                f"got {self.engage_distance_units}"
            )


@dataclass(frozen=True)
class GameStateAdapter:
    """Pure adapter projecting a canonical GameState into consumer views.

    Beyond the raw projection, the adapter owns the **derived** quantities that
    ADR-002 deliberately keeps off ``GameState``:

    - ``target_in_range``  <- target distance vs ``config.engage_distance_units``
    - ``target_is_alive``  <- ``target.hp_pct > 0``
    - ``target_hp_percent``<- ``target.hp_pct`` (fraction to percent)
    - ``adds_count``       <- entities engaged with self
    - ``resource_max``     <- class/level through ``resource_table``

    Every derivation returns ``None`` when one of its inputs is unobserved, so
    the projection raises ``AdapterIncompleteError`` exactly as before. A
    derivation never manufactures a missing input.
    """

    snapshot: GameState
    config: AdapterDerivationConfig | None = None
    context: RuntimeContext | None = None
    resource_table: ResourceTable = EMPTY_RESOURCE_TABLE

    # ------------------------------------------------------------------
    # Derived quantities (ADR-002 "derived" class; never stored on GameState)
    # ------------------------------------------------------------------

    def _derive_target_in_range(self, target: object) -> bool | None:
        """Derive range from the target's estimated distance.

        Inputs: ``target.distance_estimate`` and
        ``config.engage_distance_units``. Returns ``None`` when either is
        unavailable, so the fail-loud path is preserved.
        """
        if self.config is None:
            return None
        raw = getattr(target, "distance_estimate", None)
        if raw is None:
            return None
        try:
            distance = float(raw)
        except (TypeError, ValueError):
            return None
        return distance <= self.config.engage_distance_units

    @staticmethod
    def _alive_from_hp_fraction(hp_fraction: Any) -> bool | None:
        """Return whether a ``[0,1]`` HP fraction denotes a living entity.

        ``None`` when the fraction is unobserved or non-numeric. A wrong answer
        is a hard gate -- the loot consumer skips looting a live target and
        treats a dead one as lootable -- so this never guesses.
        """
        if hp_fraction is None:
            return None
        try:
            return float(hp_fraction) > 0.0
        except (TypeError, ValueError):
            return None

    def _derive_target_is_alive(self, target: object) -> bool | None:
        """Derive liveness from the canonical target HP fraction.

        ``TargetInfo.hp_pct`` is already canonical, so no extra channel is
        needed: a target at 0 HP is dead.
        """
        return self._alive_from_hp_fraction(getattr(target, "hp_pct", None))

    def _derive_adds_count(self) -> int | None:
        """Count observed entities engaged with the player.

        Honest-unknown rule, both halves deliberate:

        - an **empty** entity channel means "no observations of this kind are
          available" (ADR-001 decision 4), not "zero adds", so it yields
          ``None``;
        - any entity whose ``is_in_combat_with_self`` is unobserved makes the
          total unknown, so it yields ``None``.

        A fabricated low count permanently suppresses fleeing, so an
        under-count is more dangerous than an honest ``None``.
        """
        raw_entities = getattr(self.snapshot, "entities", None)
        if raw_entities is None:
            return None
        entities = tuple(raw_entities)
        if not entities:
            return None
        engaged = 0
        for entity in entities:
            flag = getattr(entity, "is_in_combat_with_self", None)
            if flag is None:
                return None
            if bool(flag):
                engaged += 1
        return engaged

    def _derive_resource_max(self) -> float | None:
        """Derive max resource from the character class and level.

        Today this always returns ``None``: no ``GameState`` field carries the
        character class, so ``char_class`` is unobserved and the lookup cannot
        be made. That is ADR-002's unresolved question 2, recorded here rather
        than guessed -- a fabricated maximum silently rescales every
        resource-dependent consumer decision.
        """
        raw_level = getattr(self.snapshot, "level_or_xp", None)
        level: int | None = None
        if raw_level is not None:
            try:
                level = int(float(raw_level))
            except (TypeError, ValueError):
                level = None
        raw_class = getattr(self.snapshot, "char_class", None)
        char_class = str(raw_class) if raw_class is not None else None
        return self.resource_table.max_resource(char_class, level)

    def _entity_to_target_view(self, entity: EnemyInfo) -> TargetView:
        """Project one observed entity into a ``TargetEntityLike`` view.

        ``hp_percent`` and ``is_alive`` are derived from the entity's canonical
        ``hp_fraction``, which is a fraction and must be converted; the other
        fields are read as observed.

        Raises ``AdapterIncompleteError`` naming the first unobserved required
        field rather than dropping the entity, because silently removing a
        hostile from the candidate list would change targeting without saying
        so. ``threat`` has no derivation (ADR-002 unresolved question 3), so an
        entity without observed threat blocks the projection loudly.
        """
        entity_id = getattr(entity, "entity_id", None)
        if entity_id is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional entity_id")

        distance = getattr(entity, "distance_estimate", None)
        if distance is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional distance")

        threat = getattr(entity, "threat", None)
        if threat is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional threat")

        hp_fraction = getattr(entity, "hp_fraction", None)
        hp_percent = fraction_to_percent(hp_fraction)
        if hp_percent is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional hp_percent")

        is_attackable = getattr(entity, "is_attackable", None)
        if is_attackable is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional is_attackable")

        is_alive = getattr(entity, "is_alive", None)
        if is_alive is None:
            is_alive = self._alive_from_hp_fraction(hp_fraction)
        if is_alive is None:
            raise AdapterIncompleteError("TargetEntityLike requires non-Optional is_alive")

        in_combat = getattr(entity, "is_in_combat_with_self", None)
        if in_combat is None:
            raise AdapterIncompleteError(
                "TargetEntityLike requires non-Optional is_in_combat_with_self"
            )

        return TargetView(
            entity_id=str(entity_id),
            distance=float(distance),
            threat=float(threat),
            hp_percent=float(hp_percent),
            is_attackable=bool(is_attackable),
            is_alive=bool(is_alive),
            is_in_combat_with_self=bool(in_combat),
        )

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
            res_max = self._derive_resource_max()
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
            res_max = self._derive_resource_max()
        if res_max is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional resource_max")

        target = getattr(self.snapshot, "target", None)
        in_range = getattr(self.snapshot, "target_in_range", None)
        if in_range is None and target is not None:
            in_range = self._derive_target_in_range(target)
        if in_range is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional target_in_range")

        gcd = getattr(self.snapshot, "gcd_ready", None)
        if gcd is None and self.context is not None:
            gcd = self.context.gcd_ready()
        if gcd is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional gcd_ready")

        target_hp = (
            fraction_to_percent(getattr(target, "hp_pct", None))
            if target
            else getattr(self.snapshot, "target_hp_percent", None)
        )
        if target_hp is None:
            raise AdapterIncompleteError("CombatStateView requires non-Optional target_hp_percent")

        # The canonical entity channel is authoritative only when it carries
        # observations. An empty tuple means "no observations of this kind are
        # available" (ADR-001 decision 4), so the adapter falls back to the
        # selected target rather than reporting a fabricated empty world. No
        # try/except here on purpose: if the projection cannot supply entities,
        # that mismatch must surface rather than become an empty tuple.
        raw_entities = getattr(self.snapshot, "entities", None)
        observed_entities = tuple(raw_entities) if raw_entities is not None else ()
        if observed_entities:
            entities: tuple[TargetView, ...] = tuple(
                self._entity_to_target_view(entity) for entity in observed_entities
            )
        else:
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
            runtime_context=self.context,
        )

    def to_targeting_views(self) -> tuple[TargetView, ...]:
        """Project the observed target candidates to TargetEntityLike views.

        The canonical entity channel is authoritative when it carries
        observations. When it is empty, the adapter falls back to the single
        selected ``target``, which is itself a real observation.

        Raises AdapterIncompleteError if a candidate is missing a field the
        protocol declares non-Optional.
        """
        raw_entities = getattr(self.snapshot, "entities", None)
        observed_entities = tuple(raw_entities) if raw_entities is not None else ()
        if observed_entities:
            return tuple(self._entity_to_target_view(entity) for entity in observed_entities)

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

        target = getattr(self.snapshot, "target", None)
        in_range = getattr(self.snapshot, "target_in_range", None)
        if in_range is None and target is not None:
            in_range = self._derive_target_in_range(target)
        if in_range is None:
            raise AdapterIncompleteError("ReactiveStateView requires non-Optional target_in_range")

        raw_casts = getattr(self.snapshot, "incoming_casts", ())
        incoming_casts: tuple[EnemyCastView, ...] = tuple(raw_casts)

        target_id = str(target.name) if target and getattr(target, "name", None) else None

        return ReactiveView(
            self_hp_percent=hp,
            self_in_combat=bool(in_combat),
            target_in_range=bool(in_range),
            self_x=x,
            self_y=y,
            incoming_casts=incoming_casts,
            current_target_id=target_id,
            runtime_context=self.context,
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
            adds = self._derive_adds_count()
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
            current_target = getattr(self.snapshot, "target", None)
            if current_target is not None:
                alive = self._derive_target_is_alive(current_target)
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
