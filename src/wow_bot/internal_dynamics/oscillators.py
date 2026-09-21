"""# Pre-lab (MOCK_MODE)
Coupled oscillator bank module (Task 3.2).

Provides slow continuous temporal variation using deterministic coupled oscillators
with five canonical incommensurable frequencies and amplitudes.

Invariants & Behavior:
  - Five canonical frequencies (Hz):
    1/18000 (5h), 1/5400 (90m), 1/1200 (20m), 1/300 (5m), 1/60 (1m).
  - Corresponding canonical amplitudes:
    [0.15, 0.08, 0.05, 0.03, 0.01].
  - Aggregate sum output bounded in roughly [-0.32, +0.32].
  - Phase evolution is proportional to simulation elapsed time dt:
    phase_i = (phase_i + 2 * pi * frequency_i * dt) % (2 * pi).
  - dt < 0 raises ValueError.
  - dt == 0 returns the current aggregate output without mutating phases.
  - Local NumPy Generator (default_rng) is used; global RNG state is not modified.
  - Property returns (frequencies, phases) are copies and cannot mutate internal state.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from wow_bot.shared.config import InternalDynamicsConfig, Settings

# Canonical oscillator frequencies in Hz
DEFAULT_FREQUENCIES_HZ: tuple[float, ...] = (
    1.0 / 18000.0,
    1.0 / 5400.0,
    1.0 / 1200.0,
    1.0 / 300.0,
    1.0 / 60.0,
)

# Canonical oscillator amplitudes
DEFAULT_AMPLITUDES: tuple[float, ...] = (
    0.15,
    0.08,
    0.05,
    0.03,
    0.01,
)

TWO_PI: float = 2.0 * math.pi


class OscillatorBank:
    """Deterministic bank of slow oscillatory signals."""

    def __init__(
        self,
        config: InternalDynamicsConfig | Settings | dict[str, Any] | None = None,
        seed: int | None = None,
    ) -> None:
        """Initialize the oscillator bank.

        Args:
            config: Optional configuration object or dictionary.
            seed: Optional integer seed for deterministic initial phases.
        """
        self._frequencies = np.array(DEFAULT_FREQUENCIES_HZ, dtype=np.float64)
        self._amplitudes = np.array(DEFAULT_AMPLITUDES, dtype=np.float64)

        # Priority 1: Explicit configured initial phases if provided
        configured_phases: list[float] | None = None
        if isinstance(config, dict) and "initial_phases" in config:
            configured_phases = config["initial_phases"]
        elif config is not None and hasattr(config, "initial_phases"):
            phases_attr: Any = config.initial_phases
            if phases_attr is not None:
                configured_phases = list(phases_attr)

        if configured_phases is not None and len(configured_phases) == len(self._frequencies):
            phases_arr = np.array(configured_phases, dtype=np.float64) % TWO_PI
        else:
            # Priority 2 & 3: Seeded RNG when seed is provided, or entropy-backed RNG otherwise
            rng = np.random.default_rng(seed)
            phases_arr = rng.uniform(0.0, TWO_PI, size=len(self._frequencies))

        self._phases: np.ndarray = phases_arr.astype(np.float64)

    def step(self, dt: float) -> float:
        """Advance oscillator phases by dt seconds and return aggregate output.

        Args:
            dt: Elapsed simulation time in seconds.

        Returns:
            Aggregate output sum: sum(amplitude_i * sin(phase_i)).

        Raises:
            ValueError: If dt < 0.
        """
        if dt < 0.0:
            raise ValueError(f"dt must be non-negative, got {dt}")

        if dt > 0.0:
            self._phases = (self._phases + TWO_PI * self._frequencies * dt) % TWO_PI

        # Calculate current aggregate value without mutation if dt == 0
        val = np.sum(self._amplitudes * np.sin(self._phases))
        return float(val)

    @property
    def frequencies(self) -> list[float]:
        """Return oscillator frequencies in Hz as a list copy."""
        return [float(f) for f in self._frequencies]

    @property
    def phases(self) -> list[float]:
        """Return current phases in radians as a list copy."""
        return [float(p) for p in self._phases]
