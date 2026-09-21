"""# Pre-lab (MOCK_MODE)
Lorenz chaotic attractor module (Task 3.3).

Provides deterministic chaotic modulation via classical Runge-Kutta 4th order (RK4)
numerical integration of the canonical Lorenz dynamical system.

Lorenz Equations:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z

Public API:
    - :class:`LorenzAttractor`
"""

from __future__ import annotations

import math

import numpy as np

# Standard default scale for x-projection normalization
NORMALIZATION_SCALE_X: float = 20.0


class LorenzAttractor:
    """Deterministic Lorenz attractor integrated with 4th-order Runge-Kutta (RK4)."""

    def __init__(
        self,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 2.667,
        dt: float = 0.001,
        initial_state: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        """Initialize the Lorenz attractor.

        Args:
            sigma: Prandtl number (must be > 0).
            rho: Rayleigh number (must be finite).
            beta: Geometric parameter (must be > 0).
            dt: Integration time step in seconds (must be > 0).
            initial_state: (x0, y0, z0) starting state tuple (all components finite).

        Raises:
            ValueError: If any parameter or initial state value fails validation.
        """
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError(f"sigma must be positive and finite, got {sigma}")
        if not math.isfinite(beta) or beta <= 0:
            raise ValueError(f"beta must be positive and finite, got {beta}")
        if not math.isfinite(dt) or dt <= 0:
            raise ValueError(f"dt must be positive and finite, got {dt}")
        if not math.isfinite(rho):
            raise ValueError(f"rho must be finite, got {rho}")

        if len(initial_state) != 3:
            raise ValueError(f"initial_state must be a 3-tuple, got {initial_state}")

        x0, y0, z0 = initial_state
        if not (math.isfinite(x0) and math.isfinite(y0) and math.isfinite(z0)):
            raise ValueError(f"initial_state components must all be finite, got {initial_state}")

        self._sigma = float(sigma)
        self._rho = float(rho)
        self._beta = float(beta)
        self._dt = float(dt)

        self._state: np.ndarray = np.array([x0, y0, z0], dtype=np.float64)

    @property
    def sigma(self) -> float:
        """Return parameter sigma."""
        return self._sigma

    @property
    def rho(self) -> float:
        """Return parameter rho."""
        return self._rho

    @property
    def beta(self) -> float:
        """Return parameter beta."""
        return self._beta

    @property
    def dt(self) -> float:
        """Return parameter dt."""
        return self._dt

    @property
    def state(self) -> np.ndarray:
        """Return a copy of the current state vector [x, y, z] as np.float64 array of shape (3,)."""
        return self._state.copy()

    def _derivative(self, state: np.ndarray) -> np.ndarray:
        """Calculate vector field derivative f(state) for the Lorenz equations."""
        x, y, z = state[0], state[1], state[2]
        dx = self._sigma * (y - x)
        dy = x * (self._rho - z) - y
        dz = x * y - self._beta * z
        return np.array([dx, dy, dz], dtype=np.float64)

    def step(self) -> np.ndarray:
        """Advance one RK4 integration step and return a copy of the new (x, y, z) state."""
        s = self._state
        dt = self._dt

        k1 = self._derivative(s)
        k2 = self._derivative(s + 0.5 * dt * k1)
        k3 = self._derivative(s + 0.5 * dt * k2)
        k4 = self._derivative(s + dt * k3)

        self._state = s + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return self._state.copy()

    def normalized(self) -> float:
        """Return a bounded scalar chaos signal in [-1.0, +1.0] derived from current x.

        Uses hyperbolic tangent projection math.tanh(x / NORMALIZATION_SCALE_X).
        Does not mutate internal state or advance integration.
        """
        x = float(self._state[0])
        val = math.tanh(x / NORMALIZATION_SCALE_X)
        return float(val)
