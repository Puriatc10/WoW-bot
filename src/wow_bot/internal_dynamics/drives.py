"""# Pre-lab (MOCK_MODE)
Drives core module (Task 3.1).

Manages the five internal physiological/psychological drive levels:
  - hunger
  - fatigue
  - curiosity
  - aggression
  - social

Invariants & Behavior:
  - Vector order strictly follows ``DRIVE_NAMES``:
    ``[hunger, fatigue, curiosity, aggression, social]``.
  - Each drive component is clamped to ``[0.0, 1.0]``.
  - Baseline level defaults to 0.5.
  - ``step(dt, chaos_component)`` applies natural drift (e.g. fatigue accumulation)
    and chaos-induced fluctuation.
  - ``apply_event(event)`` applies drive deltas from ``EVENT_EFFECTS`` immediately.
    Unknown events are handled gracefully with no state change.
  - ``decay(dt)`` relaxes drive levels back towards baseline (0.5) over time.
  - Invalid ``dt < 0`` raises ``ValueError``.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wow_bot.shared.config import InternalDynamicsConfig, Settings
from wow_bot.shared.events import effects_for, is_known_event
from wow_bot.shared.interfaces import DRIVE_NAMES, Event

# Default baseline value for drives
DEFAULT_BASELINE: float = 0.5

# Default natural drift rates per second (dt scaled)
# fatigue (0.0005) > social (-0.0001) satisfies the requirement fatigue dynamics > social dynamics
DEFAULT_DRIFT_RATES: dict[str, float] = {
    "hunger": 0.0002,
    "fatigue": 0.0005,
    "curiosity": 0.0,
    "aggression": 0.0,
    "social": -0.0001,
}

# Default relaxation (decay) rate towards baseline per second
DEFAULT_DECAY_RATE: float = 0.005


class Drives:
    """Internal drive vector state and dynamics."""

    def __init__(
        self,
        config: InternalDynamicsConfig | Settings | dict[str, Any] | None = None,
        initial_drives: dict[str, float] | None = None,
        baseline: float = DEFAULT_BASELINE,
        decay_rate: float = DEFAULT_DECAY_RATE,
        drift_rates: dict[str, float] | None = None,
    ) -> None:
        """Initialize drives to baseline or supplied initial levels."""
        self._baseline = float(baseline)
        self._decay_rate = float(decay_rate)
        self._drift_rates = dict(drift_rates if drift_rates is not None else DEFAULT_DRIFT_RATES)

        # Parse config if provided
        if isinstance(config, Settings):
            _dyn_cfg = config.internal_dynamics
        elif isinstance(config, InternalDynamicsConfig):
            _dyn_cfg = config
        else:
            _dyn_cfg = None

        self._drives: dict[str, float] = {}
        for name in DRIVE_NAMES:
            val = initial_drives.get(name, self._baseline) if initial_drives else self._baseline
            self._drives[name] = self._clamp(val)

    @staticmethod
    def _clamp(val: float) -> float:
        """Clamp a value to [0.0, 1.0]."""
        if val < 0.0:
            return 0.0
        if val > 1.0:
            return 1.0
        return float(val)

    @property
    def vector(self) -> np.ndarray:
        """Return the current drive levels as a 1D float64 numpy array of shape (5,).

        The component ordering matches ``interfaces.DRIVE_NAMES``:
        [hunger, fatigue, curiosity, aggression, social].
        All values are clamped to [0.0, 1.0].
        A new array instance is returned so callers cannot mutate internal state.
        """
        vals = [self._clamp(self._drives[name]) for name in DRIVE_NAMES]
        return np.array(vals, dtype=np.float64)

    @property
    def drives_dict(self) -> dict[str, float]:
        """Return a copy of the current drive dictionary."""
        return {name: self._clamp(self._drives[name]) for name in DRIVE_NAMES}

    def get_drive(self, name: str) -> float:
        """Get the current level of a single drive by name."""
        if name not in self._drives:
            raise KeyError(f"Unknown drive name {name!r}; valid names are {DRIVE_NAMES}")
        return self._clamp(self._drives[name])

    def step(self, dt: float, chaos_component: float = 0.0) -> None:
        """Advance drive states over duration dt (seconds) with drift and chaos.

        Args:
            dt: Time delta in seconds.
            chaos_component: Scalar chaos input (e.g. from Lorenz attractor) normalized.

        Raises:
            ValueError: If dt < 0.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        if dt == 0.0:
            return

        for name in DRIVE_NAMES:
            drift = self._drift_rates.get(name, 0.0) * dt
            # Chaos exerts a small modulation across drives
            chaos_mod = chaos_component * 0.001 * dt
            new_val = self._drives[name] + drift + chaos_mod
            self._drives[name] = self._clamp(new_val)

    def apply_event(self, event: Event) -> None:
        """Apply drive deltas from an Event immediately.

        Unknown event types leave state unchanged.

        Args:
            event: An Event instance.
        """
        if not is_known_event(event.type):
            return

        effects = effects_for(event.type)
        for name in DRIVE_NAMES:
            delta = effects.get(name, 0.0)
            self._drives[name] = self._clamp(self._drives[name] + delta)

    def decay(self, dt: float) -> None:
        """Relax drive levels back towards baseline (0.5) over time dt.

        Args:
            dt: Time delta in seconds.

        Raises:
            ValueError: If dt < 0.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")
        if dt == 0.0:
            return

        factor = math.exp(-self._decay_rate * dt)
        for name in DRIVE_NAMES:
            current = self._drives[name]
            # Exponential decay towards baseline
            relaxed = self._baseline + (current - self._baseline) * factor
            self._drives[name] = self._clamp(relaxed)
