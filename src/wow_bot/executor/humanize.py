"""Stochastic timing, error probability, and coordinate jitter simulation.

This module provides stochastic helpers for humanized timing and error modeling
in simulated bot execution. These functions perform mathematical simulation only;
they do not execute waits (sleeps), interact with the operating system, or
call input controllers.

Public API:
    - :func:`human_delay`
    - :func:`human_error_probability`
    - :func:`jitter_coordinates`
    - :data:`MIN_DELAY_MS`
    - :data:`MAX_DELAY_MS`
"""

from __future__ import annotations

import math
from typing import Final

import numpy as np

#: Minimum simulated delay bound in milliseconds.
MIN_DELAY_MS: Final[float] = 50.0

#: Maximum simulated delay bound in milliseconds.
MAX_DELAY_MS: Final[float] = 2000.0


def _validate_int(val: object, name: str) -> int:
    """Validate that a value is an integer and not a boolean."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise ValueError(  # noqa: TRY004
            f"{name} must be an integer, got {type(val).__name__} ({val!r})"
        )
    return val


def _validate_float_range(
    val: object,
    name: str,
    min_val: float,
    max_val: float,
) -> float:
    """Validate that a value is a finite float/int in [min_val, max_val]."""
    if isinstance(val, bool) or not isinstance(val, (int, float)):
        raise ValueError(  # noqa: TRY004
            f"{name} must be a numeric float, got {type(val).__name__} ({val!r})"
        )
    float_val = float(val)
    if not math.isfinite(float_val):
        raise ValueError(f"{name} must be a finite number, got {val!r}")
    if float_val < min_val or float_val > max_val:
        raise ValueError(
            f"{name} must be in range [{min_val}, {max_val}], got {float_val}"
        )
    return float_val


def human_delay(
    base_ms: int = 200,
    fatigue: float = 0.5,
    chaos_component: float = 0.0,
    *,
    rng: np.random.Generator | None = None,
) -> float:
    """Calculate a simulated human action delay in milliseconds.

    Uses a log-normal distribution with variance scaled by fatigue and chaos:
        sigma = 0.4 + 0.2 * chaos_component + 0.3 * fatigue
        mu = log(base_ms)

    The resulting delay is clipped to [50.0, 2000.0] ms.

    Args:
        base_ms: Base median delay in milliseconds (> 0, strictly integer).
        fatigue: Fatigue level in [0.0, 1.0].
        chaos_component: Lorenz chaos component in [-1.0, 1.0].
        rng: Optional explicit NumPy Generator for deterministic testing.

    Returns:
        Simulated delay duration in milliseconds (float).

    Raises:
        ValueError: If any input parameter fails validation.
    """
    valid_base_ms = _validate_int(base_ms, "base_ms")
    if valid_base_ms <= 0:
        raise ValueError(f"base_ms must be strictly positive (> 0), got {valid_base_ms}")

    valid_fatigue = _validate_float_range(fatigue, "fatigue", 0.0, 1.0)
    valid_chaos = _validate_float_range(chaos_component, "chaos_component", -1.0, 1.0)

    sigma = 0.4 + 0.2 * valid_chaos + 0.3 * valid_fatigue
    mu = float(np.log(valid_base_ms))

    generator = rng if rng is not None else np.random.default_rng()
    raw_delay = float(generator.lognormal(mean=mu, sigma=sigma))
    clipped_delay = float(np.clip(raw_delay, MIN_DELAY_MS, MAX_DELAY_MS))
    return clipped_delay


def human_error_probability(
    drives_vector: np.ndarray,
) -> float:
    """Calculate the estimated probability of a human execution error.

    Uses canonical drive vector indexing:
        index 0 = hunger
        index 1 = fatigue
        index 2 = curiosity
        index 3 = aggression
        index 4 = social

    Formula:
        probability = 0.02 + 0.05 * fatigue + 0.03 * (1.0 - curiosity)

    Args:
        drives_vector: 1D NumPy array of shape (5,) with drive values in [0.0, 1.0].

    Returns:
        Estimated error probability scalar in [0.02, 0.10] (roadmap envelope <= 0.15).

    Raises:
        ValueError: If drives_vector is not a valid 1D NumPy array of shape (5,)
            or contains out-of-range/non-finite values.
    """
    if not isinstance(drives_vector, np.ndarray):
        raise ValueError(  # noqa: TRY004
            f"drives_vector must be a NumPy ndarray, got {type(drives_vector).__name__}"
        )

    if drives_vector.shape != (5,):
        raise ValueError(
            f"drives_vector shape must be (5,), got {drives_vector.shape}"
        )

    if not np.issubdtype(drives_vector.dtype, np.number) or np.issubdtype(
        drives_vector.dtype, np.bool_
    ):
        raise ValueError("drives_vector must contain numeric values")

    if not bool(np.all(np.isfinite(drives_vector))):
        raise ValueError("drives_vector entries must all be finite numbers")

    if not bool(np.all((drives_vector >= 0.0) & (drives_vector <= 1.0))):
        raise ValueError("drives_vector entries must all be in range [0.0, 1.0]")

    fatigue = float(drives_vector[1])
    curiosity = float(drives_vector[2])

    prob = 0.02 + 0.05 * fatigue + 0.03 * (1.0 - curiosity)
    return float(prob)


def jitter_coordinates(
    x: int,
    y: int,
    radius: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> tuple[int, int]:
    """Calculate simulated coordinate perturbation for replay/testing.

    Applies independent uniform integer perturbations in [-radius, radius] to x and y.

    Args:
        x: Base integer x coordinate.
        y: Base integer y coordinate.
        radius: Perturbation radius integer (>= 0).
        rng: Optional explicit NumPy Generator for deterministic testing.

    Returns:
        Perturbed coordinate tuple (x_jittered, y_jittered).

    Raises:
        ValueError: If x, y, or radius fail validation.
    """
    valid_x = _validate_int(x, "x")
    valid_y = _validate_int(y, "y")
    valid_radius = _validate_int(radius, "radius")

    if valid_radius < 0:
        raise ValueError(f"radius must be non-negative (>= 0), got {valid_radius}")

    if valid_radius == 0:
        return (valid_x, valid_y)

    generator = rng if rng is not None else np.random.default_rng()
    dx = int(generator.integers(-valid_radius, valid_radius, endpoint=True))
    dy = int(generator.integers(-valid_radius, valid_radius, endpoint=True))

    return (valid_x + dx, valid_y + dy)
