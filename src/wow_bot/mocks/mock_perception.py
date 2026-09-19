"""Mock perception layer (Task 2.1).

Emits synthetic :class:`~wow_bot.shared.interfaces.GameState` frames so that
downstream layers (Internal Dynamics, Strategist, Executor) can be developed
and tested without any contact with a real game client or screenshot pipeline.

Design notes
------------
* All randomness flows through an injected :class:`numpy.random.Generator`.
  Tests pass a fixed seed for determinism; production uses entropy (AGENTS.md
  constraint #3 and testing philosophy rule 1).
* HP/Mana fluctuate between ``0.3`` and ``1.0`` via a slow random walk clamped
  to that band.
* Position follows a slow 2-D random walk (step magnitude ≈ 0.5 units/frame).
* ``in_combat`` toggles on for ~10 s every ~30 s of wall-clock time.
* Events are drawn occasionally from the registry in
  :mod:`wow_bot.shared.events`; each emitted event carries its type, the
  current timestamp, and an empty payload dict.
* The class exposes only ``async def get_state() -> GameState`` as required by
  the roadmap. No keyboard/mouse I/O, no network calls, no game-process access.
"""

from __future__ import annotations

import time

import numpy as np

from wow_bot.shared.events import ALL_EVENT_TYPES
from wow_bot.shared.interfaces import EnemyInfo, Event, GameState, TargetInfo


class MockPerception:
    """Synthetic GameState source for development and testing.

    Parameters
    ----------
    rng:
        A NumPy random generator.  When *None*, one is created from OS entropy.
        Tests should pass ``np.random.default_rng(seed)`` for reproducibility.
    start_position:
        Initial (x, y) world position of the mock player.
    combat_on_duration:
        Seconds ``in_combat`` stays *True* once triggered.
    combat_cycle_period:
        Approximate seconds between combat-on triggers.
    event_probability:
        Per-frame probability of emitting exactly one random event.
    """

    def __init__(
        self,
        *,
        rng: np.random.Generator | None = None,
        start_position: tuple[float, float] = (0.0, 0.0),
        combat_on_duration: float = 10.0,
        combat_cycle_period: float = 30.0,
        event_probability: float = 0.05,
    ) -> None:
        self._rng = rng if rng is not None else np.random.default_rng()
        self._position = list(start_position)
        self._hp = 0.85
        self._mana = 0.90
        self._facing = 0.0
        self._combat_on_duration = combat_on_duration
        self._combat_cycle_period = combat_cycle_period
        self._event_probability = event_probability

        # Combat schedule state.
        self._next_combat_start: float = time.monotonic() + self._random_combat_offset()
        self._combat_end: float = 0.0
        self._in_combat = False

        # Resource bounds per roadmap spec.
        self._resource_low = 0.3
        self._resource_high = 1.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def get_state(self) -> GameState:
        """Produce one synthetic frame and advance internal state."""
        now_wall = time.monotonic()
        ts = time.time()

        self._update_resources()
        self._update_position()
        self._update_facing()
        self._update_combat(now_wall)
        events = self._maybe_emit_event(ts)
        target, enemies = self._make_combat_entities()

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
        )

    # ------------------------------------------------------------------
    # Internal helpers (all synchronous, no I/O)
    # ------------------------------------------------------------------

    def _random_combat_offset(self) -> float:
        """Jitter the first combat trigger so it is not perfectly periodic."""
        return float(self._rng.uniform(0.5 * self._combat_cycle_period, 1.5 * self._combat_cycle_period))

    def _update_resources(self) -> None:
        """Slow random walk for HP and Mana, clamped to [0.3, 1.0]."""
        step_hp = float(self._rng.normal(0.0, 0.02))
        step_mana = float(self._rng.normal(0.0, 0.015))
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
        """Toggle in_combat on for ~10 s every ~30 s of wall-clock time."""
        if self._in_combat:
            if now_wall >= self._combat_end:
                self._in_combat = False
                self._next_combat_start = now_wall + self._random_combat_offset()
        else:
            if now_wall >= self._next_combat_start:
                self._in_combat = True
                self._combat_end = now_wall + self._combat_on_duration

    def _maybe_emit_event(self, ts: float) -> list[Event]:
        """Occasionally emit one random registered event."""
        if self._rng.random() < self._event_probability:
            idx = int(self._rng.integers(0, len(ALL_EVENT_TYPES)))
            etype = ALL_EVENT_TYPES[idx]
            return [Event(type=etype, timestamp=ts, data={})]
        return []

    def _make_combat_entities(self) -> tuple[TargetInfo | None, list[EnemyInfo]]:
        """During combat, populate a single target and 1-3 enemies."""
        if not self._in_combat:
            return None, []

        n_enemies = int(self._rng.integers(1, 4))
        enemies: list[EnemyInfo] = []
        for i in range(n_enemies):
            bx = int(self._rng.integers(0, 1920))
            by = int(self._rng.integers(0, 1080))
            bw = int(self._rng.integers(32, 128))
            bh = int(self._rng.integers(32, 128))
            conf = float(self._rng.uniform(0.6, 1.0))
            dist = float(self._rng.uniform(5.0, 40.0))
            enemies.append(
                EnemyInfo(bbox=(bx, by, bw, bh), confidence=conf, distance_estimate=dist)
            )

        target = TargetInfo(
            name=f"MockEnemy_{i}",
            hp_pct=float(self._rng.uniform(0.2, 1.0)),
            reaction="hostile",
            distance_estimate=float(self._rng.uniform(5.0, 30.0)),
        )
        return target, enemies