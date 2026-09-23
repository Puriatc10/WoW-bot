"""Mock perception layer (Tasks 2.1 and 2.2).

Emits synthetic :class:`~wow_bot.shared.interfaces.GameState` frames so that
downstream layers (Internal Dynamics, Strategist, Executor) can be developed
and tested without any contact with a real game client or screenshot pipeline.

Design notes
------------
* All randomness flows through an injected :class:`numpy.random.Generator`.
  Tests pass a fixed seed for determinism; production uses entropy (AGENTS.md
  constraint #3 and testing philosophy rule 1).
* Generic HP/Mana fluctuate between ``0.3`` and ``1.0``. Peaceful HP stays
  constant; scheduled deaths emit zero HP, with respawn on the next frame.
* Position follows a slow 2-D random walk (step magnitude ≈ 0.5 units/frame).
* ``in_combat`` toggles on for ~10 s every ~30 s of wall-clock time.
* Ambient events are sampled from the registry. Scheduled mock events carry
  a ``forced`` payload flag and are driven by elapsed time, not frame count.
* The class exposes only ``async def get_state() -> GameState`` as required by
  the roadmap. No keyboard/mouse I/O, no network calls, no game-process access.

Task 2.2 — scenario-driven behaviour
------------------------------------
:class:`ScenarioProfile` freezes per-scenario tuning parameters (combat cadence,
event emission pool/probability, resource stability, death-cycle timing). Five
named profiles ship out of the box:

===================== =========================================================
``peaceful_farm``     no combat at all, HP stable near full, occasional benign
                      events (level ups, quest completes, NPC interactions).
``combat_light``      short frequent combats against a single enemy, PvP hits
                      and kills dominate the event stream.
``death_loop``        the player dies roughly every two minutes of simulated
                      time; deaths carry a small extra payload.
``rare_loot_drought`` normal combat rhythm but the rare-loot event is excluded
                      entirely from the emission pool.
``stuck_repeatedly``  stuck events recur every few minutes alongside light
                      ambient activity.
===================== =========================================================

Profiles configure one shared frame-production algorithm. Unknown scenario
names raise :class:`ValueError` rather than falling back silently. Fixed mock
schedules are synthetic test stimuli, not production decision policies.
"""

from __future__ import annotations

import math
import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from types import MappingProxyType

import numpy as np

from wow_bot.shared.events import ALL_EVENT_TYPES, DEATH, STUCK
from wow_bot.shared.interfaces import EnemyInfo, Event, GameState, TargetInfo

# ---------------------------------------------------------------------------
# Scenario profiles (Task 2.2)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ScenarioProfile:
    """Immutable tuning parameters describing one mock-perception scenario.

    Every field maps onto exactly one behavioural knob inside
    :class:`MockPerception`; adding a new scenario means composing existing
    fields, not extending the simulation algorithm.
    """

    #: Seconds ``in_combat`` stays True once triggered. ``0`` disables combat.
    combat_on_duration: float
    #: Approximate seconds between combat-on triggers. Ignored when
    #: ``combat_on_duration == 0``.
    combat_cycle_period: float
    #: Per-frame probability of emitting an event (must be in [0, 1]).
    event_probability: float
    #: Pool the event types are sampled from. Empty tuple = never emit.
    event_pool: tuple[str, ...]
    #: Standard deviation of the per-frame HP random-walk step.
    hp_step_sigma: float
    #: Standard deviation of the per-frame Mana random-walk step.
    mana_step_sigma: float
    #: If > 0, force a ``death`` event approximately every this many seconds of
    #: monotonic time, independent of the probabilistic event channel.
    death_interval_seconds: float
    #: Maximum number of enemies spawned during combat.
    max_enemies: int
    #: Mock-only elapsed-time cadence; zero disables scheduled stuck events.
    stuck_interval_seconds: float = 0.0


#: Canonical scenario name → profile mapping consumed by MockPerception.
SCENARIO_PROFILES: Mapping[str, ScenarioProfile] = MappingProxyType({
    "peaceful_farm": ScenarioProfile(
        combat_on_duration=0.0,
        combat_cycle_period=float("inf"),
        event_probability=0.08,
        event_pool=("level_up", "quest_complete", "npc_interact"),
        hp_step_sigma=0.0,
        mana_step_sigma=0.006,
        death_interval_seconds=0.0,
        max_enemies=0,
    ),
    "combat_light": ScenarioProfile(
        combat_on_duration=6.0,
        combat_cycle_period=18.0,
        event_probability=0.10,
        event_pool=("pvp_hit", "pvp_kill", "level_up", "quest_complete"),
        hp_step_sigma=0.03,
        mana_step_sigma=0.025,
        death_interval_seconds=0.0,
        max_enemies=1,
    ),
    "death_loop": ScenarioProfile(
        combat_on_duration=10.0,
        combat_cycle_period=30.0,
        event_probability=0.05,
        event_pool=tuple(t for t in ALL_EVENT_TYPES if t != DEATH),
        hp_step_sigma=0.05,
        mana_step_sigma=0.03,
        death_interval_seconds=120.0,
        max_enemies=3,
    ),
    "rare_loot_drought": ScenarioProfile(
        combat_on_duration=10.0,
        combat_cycle_period=30.0,
        event_probability=0.10,
        event_pool=tuple(t for t in ALL_EVENT_TYPES if t != "rare_loot"),
        hp_step_sigma=0.02,
        mana_step_sigma=0.015,
        death_interval_seconds=0.0,
        max_enemies=3,
    ),
    "stuck_repeatedly": ScenarioProfile(
        combat_on_duration=0.0,
        combat_cycle_period=float("inf"),
        event_probability=0.05,
        event_pool=("npc_interact", "level_up"),
        hp_step_sigma=0.01,
        mana_step_sigma=0.01,
        death_interval_seconds=0.0,
        max_enemies=0,
        stuck_interval_seconds=180.0,
    ),
})

#: Ordered list of supported scenario names (stable iteration order).
SCENARIO_NAMES: tuple[str, ...] = tuple(SCENARIO_PROFILES)


def _profile_for(scenario: str | None) -> ScenarioProfile:
    """Resolve a scenario name to its frozen profile.

    ``None`` selects the default generic behaviour used by Task 2.1 callers
    that predate scenarios. An unknown name fails loudly (AGENTS.md: silent
    config errors are the #1 source of weird bot behaviour).
    """
    if scenario is None:
        return ScenarioProfile(
            combat_on_duration=10.0,
            combat_cycle_period=30.0,
            event_probability=0.05,
            event_pool=ALL_EVENT_TYPES,
            hp_step_sigma=0.02,
            mana_step_sigma=0.015,
            death_interval_seconds=0.0,
            max_enemies=3,
        )
    try:
        return SCENARIO_PROFILES[scenario]
    except KeyError:
        raise ValueError(
            f"unknown scenario {scenario!r}; supported scenarios are "
            f"{list(SCENARIO_NAMES)}"
        ) from None


class MockPerception:
    """Synthetic GameState source for development and testing.

    Parameters
    ----------
    rng:
        A NumPy random generator.  When *None*, one is created from OS entropy.
        Tests should pass ``np.random.default_rng(seed)`` for reproducibility.
    start_position:
        Initial (x, y) world position of the mock player.
    scenario:
        Name of a built-in scenario from :data:`SCENARIO_NAMES`. When given, it
        supplies the full behavioural profile; explicit keyword overrides below
        still win over individual profile fields.
    combat_on_duration / combat_cycle_period / event_probability:
        Legacy Task 2.1 knobs kept for backward compatibility; they override
        the corresponding profile values when passed explicitly.
    event_pool:
        Optional override of the profile's event sampling pool.
    death_interval_seconds:
        Optional override of the profile's forced-death cycle period.
    max_enemies:
        Optional override of the profile's combat spawn cap.
    monotonic_clock / timestamp_clock:
        Optional clocks for replay. Supply the same time sequence and seeded
        generator to reproduce complete frames, including timestamps. Defaults
        use real time; get_state never sleeps or advances the injected clocks.
    stuck_interval_seconds:
        Optional override of the mock's scheduled stuck interval.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
        start_position: tuple[float, float] = (0.0, 0.0),
        scenario: str | None = None,
        combat_on_duration: float | None = None,
        combat_cycle_period: float | None = None,
        event_probability: float | None = None,
        event_pool: Sequence[str] | None = None,
        hp_step_sigma: float | None = None,
        mana_step_sigma: float | None = None,
        death_interval_seconds: float | None = None,
        max_enemies: int | None = None,
        stuck_interval_seconds: float | None = None,
        monotonic_clock: Callable[[], float] | None = None,
        timestamp_clock: Callable[[], float] | None = None,
    ) -> None:
        profile = _profile_for(scenario)
        self._scenario = scenario
        self._rng = rng if rng is not None else np.random.default_rng()
        self._position = list(start_position)
        self._hp = 0.85
        self._mana = 0.90
        self._facing = 0.0
        self._monotonic_clock = monotonic_clock or time.monotonic
        self._timestamp_clock = timestamp_clock or time.time
        self._respawn_pending = False

        self._combat_on_duration = (
            profile.combat_on_duration if combat_on_duration is None else combat_on_duration
        )
        self._combat_cycle_period = (
            profile.combat_cycle_period if combat_cycle_period is None else combat_cycle_period
        )
        self._event_probability = (
            profile.event_probability if event_probability is None else event_probability
        )
        pool = profile.event_pool if event_pool is None else tuple(event_pool)
        self._event_pool: tuple[str, ...] = pool
        self._hp_step_sigma = (
            profile.hp_step_sigma if hp_step_sigma is None else hp_step_sigma
        )
        self._mana_step_sigma = (
            profile.mana_step_sigma if mana_step_sigma is None else mana_step_sigma
        )
        self._death_interval_seconds = (
            profile.death_interval_seconds
            if death_interval_seconds is None
            else death_interval_seconds
        )
        self._max_enemies = profile.max_enemies if max_enemies is None else max_enemies
        self._stuck_interval_seconds = (
            profile.stuck_interval_seconds
            if stuck_interval_seconds is None
            else stuck_interval_seconds
        )
        self._validate_parameters()

        # Combat schedule state.
        now = self._monotonic_clock()
        self._next_combat_start: float = now + self._random_combat_offset()
        self._combat_end: float = 0.0
        self._in_combat = False

        # Forced-death schedule state (death_loop scenario).
        self._next_death_at: float = (
            now + self._death_interval_seconds
            if self._death_interval_seconds > 0.0
            else float("inf")
        )
        self._next_stuck_at = (
            now + self._stuck_interval_seconds
            if self._stuck_interval_seconds > 0.0
            else float("inf")
        )

        # Resource bounds per roadmap spec.
        self._resource_low = 0.3
        self._resource_high = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @property
    def scenario(self) -> str | None:
        """The scenario name this instance was constructed with, if any."""
        return self._scenario

    async def get_state(self) -> GameState:
        """Produce one synthetic frame and advance internal state."""
        now_wall = self._monotonic_clock()
        ts = self._timestamp_clock()

        self._update_resources()
        self._update_position()
        self._update_facing()
        self._update_combat(now_wall)
        events = self._collect_events(ts, now_wall)
        target, enemies = self._make_combat_entities()

        # ADR-002 (T-FIX-03.6) observed extensions. Values are derived from
        # synthetic state the mock already models; fields the mock has no
        # model for stay None ("not observed"), never a fabricated default.
        target_x: float | None
        target_y: float | None
        if target is not None:
            # The selected target sits at its estimated distance along the
            # player's facing direction: a deterministic projection of the
            # mock's own position / facing / distance state.
            target_x = self._position[0] + math.cos(self._facing) * target.distance_estimate
            target_y = self._position[1] + math.sin(self._facing) * target.distance_estimate
        else:
            target_x = None
            target_y = None

        return GameState(
            timestamp=ts,
            hp_pct=self._hp,
            mana_pct=self._mana,
            position=(self._position[0], self._position[1]),
            facing=self._facing,
            in_combat=self._in_combat,
            target=target,
            enemies=enemies,
            events=events,
            player_z=None,  # the mock world model is 2-D; no height channel
            target_x=target_x,
            target_y=target_y,
            entities=tuple(enemies),  # same synthetic mobs, world-sync channel
            perception_confidence={},  # map population is task T-FIX-22
            inventory_count=None,  # channel-pending: "Bag frame"
            inventory_max=None,  # channel-pending: "Bag frame"
            level_or_xp=None,  # channel-pending: "XP bar"
            durability_fraction=None,  # channel-pending: "Character frame"
            target_is_lootable=None,  # channel-pending: "Lootable-corpse indicator"
            incoming_casts=(),  # honest "no casts observed this frame"
        )

    # ------------------------------------------------------------------
    # Internal helpers (all synchronous, no I/O)
    # ------------------------------------------------------------------

    def _validate_parameters(self) -> None:
        for name, value in (
            ("combat_on_duration", self._combat_on_duration),
            ("hp_step_sigma", self._hp_step_sigma),
            ("mana_step_sigma", self._mana_step_sigma),
            ("death_interval_seconds", self._death_interval_seconds),
            ("stuck_interval_seconds", self._stuck_interval_seconds),
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")
        if not 0.0 <= self._event_probability <= 1.0:
            raise ValueError("event_probability must be in [0, 1]")
        if self._combat_on_duration > 0.0 and (
            not math.isfinite(self._combat_cycle_period) or self._combat_cycle_period <= 0.0
        ):
            raise ValueError("combat_cycle_period must be finite and positive for combat")
        if (
            isinstance(self._max_enemies, bool)
            or not isinstance(self._max_enemies, int)
            or self._max_enemies < 0
        ):
            raise ValueError("max_enemies must be a non-negative integer")
        if any(event not in ALL_EVENT_TYPES for event in self._event_pool):
            raise ValueError("event_pool must contain only registered event types")

    def _random_combat_offset(self) -> float:
        """Jitter the first combat trigger so it is not perfectly periodic.

        Returns ``inf`` when combat is disabled for this profile, keeping the
        scheduler permanently parked off.
        """
        if self._combat_on_duration <= 0.0:
            return float("inf")
        return float(
            self._rng.uniform(0.5 * self._combat_cycle_period, 1.5 * self._combat_cycle_period)
        )

    def _update_resources(self) -> None:
        """Slow random walk for HP and Mana, clamped to [0.3, 1.0]."""
        if self._respawn_pending:
            self._hp = self._mana = 0.5
            self._respawn_pending = False
        step_hp = float(self._rng.normal(0.0, self._hp_step_sigma))
        step_mana = float(self._rng.normal(0.0, self._mana_step_sigma))
        self._hp = float(np.clip(self._hp + step_hp, self._resource_low, self._resource_high))
        self._mana = float(np.clip(self._mana + step_mana, self._resource_low, self._resource_high))

    def _update_position(self) -> None:
        """Slow 2-D random walk (≈ 0.5 units per frame)."""
        dx = float(self._rng.normal(0.0, 0.25))
        dy = float(self._rng.normal(0.0, 0.25))
        self._position[0] += dx
        self._position[1] += dy

    def _update_facing(self) -> None:
        """Gentle drift in facing angle (radians)."""
        self._facing += float(self._rng.normal(0.0, 0.05))
        # Keep in [0, 2π).
        self._facing %= (2.0 * np.pi)

    def _update_combat(self, now_wall: float) -> None:
        """Toggle in_combat on for N s every M s of wall-clock time."""
        if self._combat_on_duration <= 0.0:
            self._in_combat = False
            return
        if self._in_combat:
            if now_wall >= self._combat_end:
                self._in_combat = False
                self._next_combat_start = now_wall + self._random_combat_offset()
        else:
            if now_wall >= self._next_combat_start:
                self._in_combat = True
                self._combat_end = now_wall + self._combat_on_duration

    def _collect_events(self, ts: float, now_wall: float) -> list[Event]:
        """Gather every event due for this frame.

        Scheduled death/stuck events precede ambient sampled events. If polling
        skips multiple intervals, emit once on the next frame and advance to
        the next original deadline, avoiding duplicate events and cadence drift.
        """
        events: list[Event] = []

        if self._death_interval_seconds > 0.0 and now_wall >= self._next_death_at:
            events.append(Event(type=DEATH, timestamp=ts, data={"forced": True}))
            intervals = math.floor(
                (now_wall - self._next_death_at) / self._death_interval_seconds
            ) + 1
            self._next_death_at += intervals * self._death_interval_seconds
            self._hp = 0.0
            self._respawn_pending = True
            self._in_combat = False
            self._next_combat_start = now_wall + self._random_combat_offset()

        if self._stuck_interval_seconds > 0.0 and now_wall >= self._next_stuck_at:
            events.append(Event(type=STUCK, timestamp=ts, data={"forced": True}))
            intervals = math.floor(
                (now_wall - self._next_stuck_at) / self._stuck_interval_seconds
            ) + 1
            self._next_stuck_at += intervals * self._stuck_interval_seconds

        events.extend(self._maybe_emit_event(ts))
        return events

    def _maybe_emit_event(self, ts: float) -> list[Event]:
        """Occasionally emit one random event from the profile pool."""
        if not self._event_pool:
            return []
        if self._rng.random() < self._event_probability:
            idx = int(self._rng.integers(0, len(self._event_pool)))
            etype = self._event_pool[idx]
            return [Event(type=etype, timestamp=ts, data={})]
        return []

    def _make_combat_entities(self) -> tuple[TargetInfo | None, list[EnemyInfo]]:
        """During combat, populate a single target and 1..max_enemies enemies."""
        if not self._in_combat or self._max_enemies <= 0:
            return None, []

        n_enemies = int(self._rng.integers(1, self._max_enemies + 1))
        enemies: list[EnemyInfo] = []
        for i in range(n_enemies):
            bx = int(self._rng.integers(0, 1920))
            by = int(self._rng.integers(0, 1080))
            bw = int(self._rng.integers(32, 128))
            bh = int(self._rng.integers(32, 128))
            conf = float(self._rng.uniform(0.6, 1.0))
            dist = float(self._rng.uniform(5.0, 40.0))
            hp_fraction = float(self._rng.uniform(0.2, 1.0))
            # Derive the mock entity's world position from the data this
            # method already samples: the detection distance along a
            # per-entity bearing, offset from the player's own position.
            bearing = float(self._rng.uniform(0.0, 2.0 * np.pi))
            ex = self._position[0] + math.cos(bearing) * dist
            ey = self._position[1] + math.sin(bearing) * dist
            # entity_id is deterministic given the detection, and unique per
            # distinct mob: derived from the entity's own sampled bbox rather
            # than an unrelated random. The mock does not track persistent
            # mobs across frames, so it cannot offer ADR-002's ideal
            # cross-frame id stability; that is a mock limitation, not a
            # fabricated value.
            entity_id = f"mock_mob_{bx}_{by}_{bw}_{bh}"
            enemies.append(
                EnemyInfo(
                    bbox=(bx, by, bw, bh),
                    confidence=conf,
                    distance_estimate=dist,
                    entity_id=entity_id,
                    kind="mob",
                    x=ex,
                    y=ey,
                    z=None,  # the mock world model is 2-D; no height channel
                    hp_fraction=hp_fraction,
                    threat=None,  # no vision component observes threat
                    is_attackable=True,
                    is_alive=hp_fraction > 0.0,
                    is_in_combat_with_self=True,
                )
            )

        target = TargetInfo(
            name=f"MockEnemy_{i}",
            hp_pct=float(self._rng.uniform(0.2, 1.0)),
            reaction="hostile",
            distance_estimate=float(self._rng.uniform(5.0, 30.0)),
        )
        return target, enemies
