import math
import random
from dataclasses import dataclass
from typing import Any


class CursorError(Exception):
    """Raised when cursor trajectory generation or utility operations fail."""


@dataclass(frozen=True)
class CursorConfig:
    # Shape parameters
    control_point_jitter_fraction: float = 0.15
    overshoot_fraction: float = 0.08
    overshoot_probability: float = 0.25

    # Timing parameters
    duration_mu: float = -1.6
    duration_sigma: float = 0.5
    duration_clip_low: float = 0.05
    duration_clip_high: float = 1.2

    # Sampling parameters
    points_per_100ms: int = 40
    min_points: int = 4
    max_points: int = 500
    min_distance_units: float = 0.5

    def __post_init__(self) -> None:
        for field_name in (
            "control_point_jitter_fraction",
            "overshoot_fraction",
            "overshoot_probability",
            "duration_mu",
            "duration_sigma",
            "duration_clip_low",
            "duration_clip_high",
            "min_distance_units",
        ):
            val = getattr(self, field_name)
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
            ):
                raise ValueError(f"{field_name} must be a finite float, got {val}")

        if not (0.0 <= self.control_point_jitter_fraction <= 1.0):
            raise ValueError(
                f"control_point_jitter_fraction must be in [0.0, 1.0], got {self.control_point_jitter_fraction}"
            )
        if not (0.0 <= self.overshoot_fraction <= 1.0):
            raise ValueError(
                f"overshoot_fraction must be in [0.0, 1.0], got {self.overshoot_fraction}"
            )
        if not (0.0 <= self.overshoot_probability <= 1.0):
            raise ValueError(
                f"overshoot_probability must be in [0.0, 1.0], got {self.overshoot_probability}"
            )
        if self.duration_sigma <= 0.0:
            raise ValueError(f"duration_sigma must be > 0.0, got {self.duration_sigma}")
        if self.duration_clip_low <= 0.0:
            raise ValueError(
                f"duration_clip_low must be > 0.0, got {self.duration_clip_low}"
            )
        if self.duration_clip_high <= self.duration_clip_low:
            raise ValueError(
                f"duration_clip_high must be > duration_clip_low ({self.duration_clip_low}), got {self.duration_clip_high}"
            )
        if self.points_per_100ms < 1:
            raise ValueError(
                f"points_per_100ms must be >= 1, got {self.points_per_100ms}"
            )
        if self.min_points < 2:
            raise ValueError(f"min_points must be >= 2, got {self.min_points}")
        if self.max_points < self.min_points:
            raise ValueError(
                f"max_points ({self.max_points}) must be >= min_points ({self.min_points})"
            )
        if self.min_distance_units < 0.0:
            raise ValueError(
                f"min_distance_units must be >= 0.0, got {self.min_distance_units}"
            )


@dataclass(frozen=True)
class TrajectoryPoint:
    x: float
    y: float
    t_offset: float

    def __post_init__(self) -> None:
        for field_name in ("x", "y", "t_offset"):
            val = getattr(self, field_name)
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
            ):
                raise ValueError(f"{field_name} must be a finite float, got {val}")


@dataclass(frozen=True)
class Trajectory:
    points: tuple[TrajectoryPoint, ...]
    start: tuple[float, float]
    end: tuple[float, float]
    duration_s: float
    overshoot_applied: bool

    def __post_init__(self) -> None:
        if not isinstance(self.start, tuple) or len(self.start) != 2:
            raise ValueError(f"start must be a 2-tuple, got {self.start}")
        if not isinstance(self.end, tuple) or len(self.end) != 2:
            raise ValueError(f"end must be a 2-tuple, got {self.end}")
        for pt_name, pt_val in (("start", self.start), ("end", self.end)):
            for v in pt_val:
                if (
                    not isinstance(v, (int, float))
                    or isinstance(v, bool)
                    or not math.isfinite(v)
                ):
                    raise ValueError(f"{pt_name} elements must be finite floats")

        if len(self.points) < 1:
            raise ValueError("Trajectory must contain at least 1 point")

        if self.points[0].t_offset != 0.0:
            raise ValueError(
                f"First point t_offset must be exactly 0.0, got {self.points[0].t_offset}"
            )

        if self.duration_s <= 0.0:
            raise ValueError(f"duration_s must be > 0.0, got {self.duration_s}")

        for i in range(len(self.points) - 1):
            if self.points[i].t_offset > self.points[i + 1].t_offset:
                raise ValueError(
                    f"t_offsets must be non-decreasing, but point {i} ({self.points[i].t_offset}) > point {i+1} ({self.points[i+1].t_offset})"
                )

        if len(self.points) == 1:
            if (
                abs(self.points[0].x - self.end[0]) > 1e-9
                or abs(self.points[0].y - self.end[1]) > 1e-9
            ):
                raise ValueError(
                    f"Single point {self.points[0]} must equal end {self.end} within 1e-9"
                )
        else:
            if (
                abs(self.points[0].x - self.start[0]) > 1e-9
                or abs(self.points[0].y - self.start[1]) > 1e-9
            ):
                raise ValueError(
                    f"First point {self.points[0]} must equal start {self.start} within 1e-9"
                )
            if (
                abs(self.points[-1].x - self.end[0]) > 1e-9
                or abs(self.points[-1].y - self.end[1]) > 1e-9
            ):
                raise ValueError(
                    f"Last point {self.points[-1]} must equal end {self.end} within 1e-9"
                )
            if abs(self.points[-1].t_offset - self.duration_s) > 1e-9:
                raise ValueError(
                    f"Last point t_offset ({self.points[-1].t_offset}) must equal duration_s ({self.duration_s}) within 1e-9"
                )

    def to_json(self) -> dict[str, Any]:
        return {
            "points": [
                {"x": p.x, "y": p.y, "t_offset": p.t_offset} for p in self.points
            ],
            "start": list(self.start),
            "end": list(self.end),
            "duration_s": self.duration_s,
            "overshoot_applied": self.overshoot_applied,
        }


def straight_line_points(
    start: tuple[float, float],
    end: tuple[float, float],
    point_count: int,
) -> tuple[TrajectoryPoint, ...]:
    """Returns `point_count` points evenly spaced along segment from start to end.

    t_offset is distributed uniformly from 0.0 to 1.0 (normalized fraction).
    Pure function, no RNG, no side effects.
    """
    if point_count < 2:
        raise CursorError(f"point_count must be >= 2, got {point_count}")

    if (
        not isinstance(start, tuple)
        or len(start) != 2
        or not isinstance(end, tuple)
        or len(end) != 2
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in start + end
        )
    ):
        raise CursorError("start and end must be 2-tuples of finite numbers")

    points: list[TrajectoryPoint] = []
    dx = end[0] - start[0]
    dy = end[1] - start[1]

    for i in range(point_count):
        u = i / (point_count - 1)
        x = round(start[0] + u * dx, 6)
        y = round(start[1] + u * dy, 6)
        t_offset = u
        points.append(TrajectoryPoint(x=x, y=y, t_offset=t_offset))

    # Force exact endpoints
    points[0] = TrajectoryPoint(
        x=round(start[0], 6), y=round(start[1], 6), t_offset=0.0
    )
    points[-1] = TrajectoryPoint(
        x=round(end[0], 6), y=round(end[1], 6), t_offset=1.0
    )

    return tuple(points)


def _cubic_bezier_point(
    p0: tuple[float, float],
    p1: tuple[float, float],
    p2: tuple[float, float],
    p3: tuple[float, float],
    u: float,
) -> tuple[float, float]:
    """Evaluates cubic Bezier curve at parameter u in [0, 1]."""
    u_inv = 1.0 - u
    u_inv_sq = u_inv * u_inv
    u_inv_cube = u_inv_sq * u_inv
    u_sq = u * u
    u_cube = u_sq * u

    x = (
        u_inv_cube * p0[0]
        + 3.0 * u_inv_sq * u * p1[0]
        + 3.0 * u_inv * u_sq * p2[0]
        + u_cube * p3[0]
    )
    y = (
        u_inv_cube * p0[1]
        + 3.0 * u_inv_sq * u * p1[1]
        + 3.0 * u_inv * u_sq * p2[1]
        + u_cube * p3[1]
    )
    return x, y


def _cubic_ease(t: float) -> float:
    """Cubic ease-in-out function: t^2 * (3 - 2*t)."""
    return t * t * (3.0 - 2.0 * t)


def generate_trajectory(
    rng: random.Random,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    config: CursorConfig | None = None,
) -> Trajectory:
    """Generates a human-like Bezier cursor trajectory from start to end.

    Pure and deterministic given (rng state, config, start, end).
    """
    cfg = config if config is not None else CursorConfig()

    if (
        not isinstance(start, tuple)
        or len(start) != 2
        or not isinstance(end, tuple)
        or len(end) != 2
        or not all(
            isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v)
            for v in start + end
        )
    ):
        raise CursorError("start and end must be 2-tuples of finite numbers")

    start_pt = (float(start[0]), float(start[1]))
    end_pt = (float(end[0]), float(end[1]))

    dx = end_pt[0] - start_pt[0]
    dy = end_pt[1] - start_pt[1]
    distance = math.sqrt(dx * dx + dy * dy)

    # Degenerate case: start and end are closer than min_distance_units
    if distance < cfg.min_distance_units:
        # Use duration_clip_low / 2 as epsilon duration, ensuring strictly positive
        epsilon_duration = cfg.duration_clip_low / 2.0
        if epsilon_duration <= 0.0:
            epsilon_duration = 1e-4

        single_point = TrajectoryPoint(
            x=round(end_pt[0], 6),
            y=round(end_pt[1], 6),
            t_offset=0.0,
        )
        return Trajectory(
            points=(single_point,),
            start=start_pt,
            end=end_pt,
            duration_s=epsilon_duration,
            overshoot_applied=False,
        )

    # Sample duration from lognormal using rejection sampling
    duration_s: float | None = None
    for _ in range(1000):
        draw = math.exp(rng.gauss(cfg.duration_mu, cfg.duration_sigma))
        if cfg.duration_clip_low <= draw <= cfg.duration_clip_high:
            duration_s = draw
            break

    if duration_s is None:
        # Rejection attempts exhausted after 1000 tries; fallback to median exp(mu) clamped to range
        median_val = math.exp(cfg.duration_mu)
        duration_s = max(
            cfg.duration_clip_low, min(cfg.duration_clip_high, median_val)
        )

    # Compute target point count
    desired_points = max(
        cfg.min_points,
        round(duration_s * cfg.points_per_100ms * 10),
    )
    point_count = min(desired_points, cfg.max_points)

    # Overshoot decision
    overshoot_applied = rng.random() < cfg.overshoot_probability

    raw_points: list[tuple[float, float]] = []

    if not overshoot_applied:
        # Single cubic Bezier segment
        jitter_max = cfg.control_point_jitter_fraction * distance
        j1 = (
            rng.uniform(-jitter_max, jitter_max),
            rng.uniform(-jitter_max, jitter_max),
        )
        j2 = (
            rng.uniform(-jitter_max, jitter_max),
            rng.uniform(-jitter_max, jitter_max),
        )

        c1 = (start_pt[0] + 0.3 * dx + j1[0], start_pt[1] + 0.3 * dy + j1[1])
        c2 = (start_pt[0] + 0.7 * dx + j2[0], start_pt[1] + 0.7 * dy + j2[1])

        for i in range(point_count):
            u = i / (point_count - 1)
            px, py = _cubic_bezier_point(start_pt, c1, c2, end_pt, u)
            raw_points.append((px, py))
    else:
        # Two cubic Bezier segments (start -> O and O -> end)
        # Overshoot point O beyond end along line start->end
        unit_x = dx / distance
        unit_y = dy / distance
        overshoot_dist = cfg.overshoot_fraction * distance
        o_pt = (
            end_pt[0] + unit_x * overshoot_dist,
            end_pt[1] + unit_y * overshoot_dist,
        )

        # Segment lengths
        d1 = distance + overshoot_dist
        d2 = overshoot_dist
        d_total = d1 + d2

        # Split point count proportional to lengths (end segment gets at least 2 points)
        n2 = max(2, round(point_count * (d2 / d_total)))
        n1 = max(2, point_count - n2 + 1)

        # Segment 1: start_pt -> o_pt
        dx1 = o_pt[0] - start_pt[0]
        dy1 = o_pt[1] - start_pt[1]
        dist1 = math.sqrt(dx1 * dx1 + dy1 * dy1)
        jitter_max1 = cfg.control_point_jitter_fraction * dist1
        j1_seg1 = (
            rng.uniform(-jitter_max1, jitter_max1),
            rng.uniform(-jitter_max1, jitter_max1),
        )
        j2_seg1 = (
            rng.uniform(-jitter_max1, jitter_max1),
            rng.uniform(-jitter_max1, jitter_max1),
        )
        c1_seg1 = (
            start_pt[0] + 0.3 * dx1 + j1_seg1[0],
            start_pt[1] + 0.3 * dy1 + j1_seg1[1],
        )
        c2_seg1 = (
            start_pt[0] + 0.7 * dx1 + j2_seg1[0],
            start_pt[1] + 0.7 * dy1 + j2_seg1[1],
        )

        # Segment 2: o_pt -> end_pt
        dx2 = end_pt[0] - o_pt[0]
        dy2 = end_pt[1] - o_pt[1]
        dist2 = math.sqrt(dx2 * dx2 + dy2 * dy2)
        jitter_max2 = cfg.control_point_jitter_fraction * dist2
        j1_seg2 = (
            rng.uniform(-jitter_max2, jitter_max2),
            rng.uniform(-jitter_max2, jitter_max2),
        )
        j2_seg2 = (
            rng.uniform(-jitter_max2, jitter_max2),
            rng.uniform(-jitter_max2, jitter_max2),
        )
        c1_seg2 = (
            o_pt[0] + 0.3 * dx2 + j1_seg2[0],
            o_pt[1] + 0.3 * dy2 + j1_seg2[1],
        )
        c2_seg2 = (
            o_pt[0] + 0.7 * dx2 + j2_seg2[0],
            o_pt[1] + 0.7 * dy2 + j2_seg2[1],
        )

        total_sampled_points = n1 + n2 - 1
        for idx in range(total_sampled_points):
            if idx < n1:
                u_local = idx / (n1 - 1)
                px, py = _cubic_bezier_point(
                    start_pt, c1_seg1, c2_seg1, o_pt, u_local
                )
            else:
                u_local = (idx - (n1 - 1)) / (n2 - 1)
                px, py = _cubic_bezier_point(
                    o_pt, c1_seg2, c2_seg2, end_pt, u_local
                )
            raw_points.append((px, py))

        point_count = len(raw_points)

    trajectory_points: list[TrajectoryPoint] = []
    for i, (px, py) in enumerate(raw_points):
        t_param = i / (point_count - 1)
        t_offset = duration_s * _cubic_ease(t_param)
        trajectory_points.append(
            TrajectoryPoint(
                x=round(px, 6),
                y=round(py, 6),
                t_offset=t_offset,
            )
        )

    # Force exact endpoints and timing bounds
    trajectory_points[0] = TrajectoryPoint(
        x=round(start_pt[0], 6),
        y=round(start_pt[1], 6),
        t_offset=0.0,
    )
    trajectory_points[-1] = TrajectoryPoint(
        x=round(end_pt[0], 6),
        y=round(end_pt[1], 6),
        t_offset=duration_s,
    )

    return Trajectory(
        points=tuple(trajectory_points),
        start=start_pt,
        end=end_pt,
        duration_s=duration_s,
        overshoot_applied=overshoot_applied,
    )
