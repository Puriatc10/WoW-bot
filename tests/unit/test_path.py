"""Unit tests for synthetic curved path generation in src/wow_bot/executor/path.py."""

from __future__ import annotations

import math
import sys

import numpy as np
import pytest

from wow_bot.executor.path import (
    MAX_CURVE_AMPLITUDE_PX,
    MIN_CURVE_AMPLITUDE_PX,
    generate_path,
)


def _calc_perpendicular_distances(
    path: list[tuple[float, float]],
    start: tuple[float, float],
    end: tuple[float, float],
) -> list[float]:
    """Calculate perpendicular distance of each point in path from the line through start and end."""
    dx = end[0] - start[0]
    dy = end[1] - start[1]
    baseline_length = math.hypot(dx, dy)
    distances: list[float] = []

    for pt in path:
        dist = abs(dx * (pt[1] - start[1]) - dy * (pt[0] - start[0])) / baseline_length
        distances.append(dist)

    return distances


# Test A: default point count
def test_default_point_count() -> None:
    path = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    assert len(path) == 5


# Test B: custom point count
def test_custom_point_count() -> None:
    path = generate_path((0.0, 0.0), (200.0, 0.0), num_points=10, rng=np.random.default_rng(42))
    assert len(path) == 10


# Test C: exact start
def test_exact_start() -> None:
    start = (12.34, -56.78)
    end = (200.0, 100.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    assert path[0] == start


# Test D: exact end
def test_exact_end() -> None:
    start = (0.0, 0.0)
    end = (200.0, 100.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    assert path[-1] == end


# Test E: path is not straight
def test_path_is_not_straight() -> None:
    start = (0.0, 0.0)
    end = (200.0, 0.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    dists = _calc_perpendicular_distances(path, start, end)
    assert any(d > 0.0 for d in dists[1:-1])


# Test F: roadmap deviation criterion (> 5.0 px std dev)
def test_roadmap_deviation_criterion() -> None:
    start = (0.0, 0.0)
    end = (200.0, 0.0)
    num_points = 5
    seed = 42
    rng = np.random.default_rng(seed)

    path = generate_path(start, end, num_points=num_points, rng=rng)
    dists = _calc_perpendicular_distances(path, start, end)
    std_dev = float(np.std(dists))

    assert path[0] == start
    assert path[-1] == end
    assert len(path) == num_points
    assert std_dev > 5.0, f"Expected std dev > 5.0 px, got {std_dev:.4f} px (distances={dists})"


# Test G: reproducibility
def test_reproducibility() -> None:
    path1 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    path2 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    assert path1 == path2


# Test H: different seeds
def test_different_seeds() -> None:
    path1 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    path2 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(43))
    assert path1 != path2


# Test I: vertical baseline
def test_vertical_baseline() -> None:
    start = (0.0, 0.0)
    end = (0.0, 200.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    assert path[0] == start
    assert path[-1] == end
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in path)
    dists = _calc_perpendicular_distances(path, start, end)
    assert any(d > 0.0 for d in dists[1:-1])


# Test J: diagonal baseline
def test_diagonal_baseline() -> None:
    start = (-50.0, 25.0)
    end = (125.0, 150.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    assert path[0] == start
    assert path[-1] == end
    assert all(math.isfinite(x) and math.isfinite(y) for x, y in path)
    dists = _calc_perpendicular_distances(path, start, end)
    assert any(d > 0.0 for d in dists[1:-1])


# Test K: negative coordinates
def test_negative_coordinates() -> None:
    start = (-100.0, -50.0)
    end = (100.0, 50.0)
    path = generate_path(start, end, rng=np.random.default_rng(42))
    assert path[0] == start
    assert path[-1] == end


# Test L: monotonic baseline progress
def test_monotonic_baseline_progress() -> None:
    start = (10.0, 20.0)
    end = (110.0, 120.0)
    path = generate_path(start, end, num_points=10, rng=np.random.default_rng(42))

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    length = math.hypot(dx, dy)
    ux, uy = dx / length, dy / length

    projections = [
        (pt[0] - start[0]) * ux + (pt[1] - start[1]) * uy
        for pt in path
    ]

    for i in range(len(projections) - 1):
        assert projections[i + 1] >= projections[i] - 1e-9


# Test M: no endpoint perturbation across many seeds
def test_no_endpoint_perturbation_across_many_seeds() -> None:
    start = (-10.5, 33.3)
    end = (100.2, 88.8)
    for seed in range(10):
        path = generate_path(start, end, rng=np.random.default_rng(seed))
        assert path[0] == start
        assert path[-1] == end


# Test N: num_points lower bound
def test_num_points_lower_bound() -> None:
    for invalid_val in [0, 1, 2, -1, -5]:
        with pytest.raises(ValueError, match="num_points must be >= 3"):
            generate_path((0, 0), (100, 100), num_points=invalid_val)


# Test O: invalid num_points type
def test_invalid_num_points_type() -> None:
    for invalid_val in [3.5, "5", True, False, None]:
        with pytest.raises(TypeError, match="num_points must be an integer"):
            generate_path((0, 0), (100, 100), num_points=invalid_val)  # type: ignore[arg-type]


# Test P: malformed start/end
def test_malformed_start_end() -> None:
    invalid_pts = [
        (),
        (1,),
        (1, 2, 3),
        ("x", 2),
        (None, 2),
        "string_pt",
        123,
    ]
    for pt in invalid_pts:
        with pytest.raises((TypeError, ValueError)):
            generate_path(pt, (100, 100))  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            generate_path((0, 0), pt)  # type: ignore[arg-type]


# Test Q: non-finite coordinates
def test_non_finite_coordinates() -> None:
    for bad_coord in [math.nan, math.inf, -math.inf]:
        with pytest.raises(ValueError, match="must be a finite number"):
            generate_path((bad_coord, 0.0), (100.0, 100.0))
        with pytest.raises(ValueError, match="must be a finite number"):
            generate_path((0.0, 0.0), (100.0, bad_coord))


# Test R: bool coordinate rejected
def test_bool_coordinate_rejected() -> None:
    with pytest.raises(TypeError, match="must be a numeric real coordinate"):
        generate_path((True, 5.0), (100.0, 100.0))
    with pytest.raises(TypeError, match="must be a numeric real coordinate"):
        generate_path((0.0, 0.0), (100.0, False))


# Test S: identical start/end
def test_identical_start_end() -> None:
    with pytest.raises(ValueError, match="start and end points must be distinct"):
        generate_path((50.0, 50.0), (50.0, 50.0))


# Test T: long segment curvature cap
def test_long_segment_curvature_cap() -> None:
    start = (0.0, 0.0)
    end = (10000.0, 0.0)
    path = generate_path(start, end, num_points=21, rng=np.random.default_rng(42))
    dists = _calc_perpendicular_distances(path, start, end)
    max_dist = max(dists)
    assert max_dist <= MAX_CURVE_AMPLITUDE_PX + 1e-6


# Test U: minimum curvature floor
def test_minimum_curvature_floor() -> None:
    start = (0.0, 0.0)
    end = (5.0, 0.0)  # Very short baseline (5 px), 10% relative = 0.5 px, so floor (20 px) applies
    # Pick odd num_points=5 so t=0.5 exists exactly at index 2
    path = generate_path(start, end, num_points=5, rng=np.random.default_rng(42))
    dists = _calc_perpendicular_distances(path, start, end)
    mid_dist = dists[2]  # sin(pi * 0.5) = 1.0 -> offset = amplitude = 20.0
    assert math.isclose(mid_dist, MIN_CURVE_AMPLITUDE_PX, rel_tol=1e-5)


# Test V: output contains Python tuples
def test_output_contains_python_tuples() -> None:
    path = generate_path((0.0, 0.0), (200.0, 0.0))
    assert isinstance(path, list)
    for pt in path:
        assert isinstance(pt, tuple)
        assert len(pt) == 2
        assert type(pt[0]) is float
        assert type(pt[1]) is float


# Test W: no Controller coupling
def test_no_controller_coupling() -> None:
    import wow_bot.executor.path as path_module

    with open(path_module.__file__, encoding="utf-8") as f:
        content = f.read()

    assert "Controller" not in content
    assert "pynput" not in content
    assert "pyautogui" not in content
    assert "keyboard" not in content
    assert "mouse" not in content


# Test X: no timing coupling
def test_no_timing_coupling() -> None:
    import wow_bot.executor.path as path_module

    with open(path_module.__file__, encoding="utf-8") as f:
        content = f.read()

    assert "human_delay" not in content
    assert "jitter_coordinates" not in content
    assert "sleep" not in content


# Test Y: no path execution
def test_no_path_execution() -> None:
    # Verify path generation leaves sys.modules controller untouched/uninvoked
    initial_controller_imported = "wow_bot.executor.controller" in sys.modules
    generate_path((0.0, 0.0), (200.0, 0.0))
    assert ("wow_bot.executor.controller" in sys.modules) == initial_controller_imported


# Test Z: fresh list
def test_fresh_list() -> None:
    path1 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    path2 = generate_path((0.0, 0.0), (200.0, 0.0), rng=np.random.default_rng(42))
    assert path1 is not path2
