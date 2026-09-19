"""Synthetic curved 2D path generator for simulation, replay, and scenario evaluation.

This module provides pure 2D geometry functions for generating curved trajectories
between two points. These functions perform mathematical geometry simulation only;
they do not move cursors, execute OS inputs, or interact with controllers.

Public API:
    - :func:`generate_path`
    - :data:`MIN_CURVE_AMPLITUDE_PX`
    - :data:`MAX_CURVE_AMPLITUDE_PX`
    - :data:`MIN_RELATIVE_CURVATURE`
    - :data:`MAX_RELATIVE_CURVATURE`
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Final

import numpy as np

#: Point coordinate tuple type alias.
Point = tuple[float, float]

#: Minimum perpendicular curve amplitude bound in pixels.
MIN_CURVE_AMPLITUDE_PX: Final[float] = 20.0

#: Maximum perpendicular curve amplitude bound in pixels.
MAX_CURVE_AMPLITUDE_PX: Final[float] = 80.0

#: Minimum relative amplitude fraction of baseline length.
MIN_RELATIVE_CURVATURE: Final[float] = 0.10

#: Maximum relative amplitude fraction of baseline length.
MAX_RELATIVE_CURVATURE: Final[float] = 0.20


def _validate_num_points(val: object) -> int:
    """Validate that num_points is an integer >= 3 and not a boolean."""
    if isinstance(val, bool) or not isinstance(val, int):
        raise TypeError(
            f"num_points must be an integer, got {type(val).__name__} ({val!r})"
        )
    if val < 3:
        raise ValueError(f"num_points must be >= 3, got {val}")
    return val


def _validate_point(pt: object, name: str) -> Point:
    """Validate that pt is a sequence of exactly two finite non-boolean real numbers."""
    if isinstance(pt, (str, bytes)) or not isinstance(pt, Sequence):
        raise TypeError(f"{name} must be a 2D coordinate sequence (tuple/list), got {type(pt).__name__}")
    if len(pt) != 2:
        raise ValueError(f"{name} must contain exactly 2 coordinates, got {len(pt)}")

    coords: list[float] = []
    for idx, coord in enumerate(pt):
        if isinstance(coord, bool) or not isinstance(coord, (int, float, np.integer, np.floating)):
            raise TypeError(
                f"{name}[{idx}] must be a numeric real coordinate, got {type(coord).__name__} ({coord!r})"
            )
        float_val = float(coord)
        if not math.isfinite(float_val):
            raise ValueError(f"{name}[{idx}] must be a finite number, got {coord!r}")
        coords.append(float_val)

    return (coords[0], coords[1])


def generate_path(
    start: tuple[float, float],
    end: tuple[float, float],
    num_points: int = 5,
    *,
    rng: np.random.Generator | None = None,
) -> list[tuple[float, float]]:
    """Generate a synthetic curved 2D path between start and end.

    Constructs a smooth single-arc trajectory using a baseline vector and
    perpendicular sine perturbation. The path passes exactly through start
    and end without endpoint jitter.

    Args:
        start: Starting (x, y) 2D coordinate pair.
        end: Ending (x, y) 2D coordinate pair.
        num_points: Total number of returned path points including start and end (>= 3).
        rng: Optional explicit NumPy Generator for deterministic testing.

    Returns:
        List of (x, y) coordinate tuples representing the generated curved path.

    Raises:
        TypeError: If num_points or point elements have invalid types.
        ValueError: If num_points < 3, points are non-2D/non-finite, or start == end.
    """
    valid_num_points = _validate_num_points(num_points)
    norm_start = _validate_point(start, "start")
    norm_end = _validate_point(end, "end")

    if norm_start == norm_end:
        raise ValueError(f"start and end points must be distinct, got start={norm_start}, end={norm_end}")

    dx = norm_end[0] - norm_start[0]
    dy = norm_end[1] - norm_start[1]
    length = math.hypot(dx, dy)

    # Unit normal perpendicular vector to direct segment
    normal_x = -dy / length
    normal_y = dx / length

    generator = rng if rng is not None else np.random.default_rng()

    # Random side choice: -1.0 or +1.0
    side = float(generator.choice([-1.0, 1.0]))

    # Relative amplitude relative to baseline distance
    raw_relative = float(generator.uniform(MIN_RELATIVE_CURVATURE, MAX_RELATIVE_CURVATURE))
    relative_amplitude = length * raw_relative

    # Bounded amplitude policy
    amplitude = min(
        MAX_CURVE_AMPLITUDE_PX,
        max(
            MIN_CURVE_AMPLITUDE_PX,
            relative_amplitude,
        ),
    )

    path: list[tuple[float, float]] = []

    for i in range(valid_num_points):
        if i == 0:
            path.append(norm_start)
        elif i == valid_num_points - 1:
            path.append(norm_end)
        else:
            t = i / (valid_num_points - 1)
            base_x = norm_start[0] + t * dx
            base_y = norm_start[1] + t * dy
            offset = side * amplitude * math.sin(math.pi * t)

            x = float(base_x + normal_x * offset)
            y = float(base_y + normal_y * offset)
            path.append((x, y))

    return path
