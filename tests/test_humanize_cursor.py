import ast
import json
import math
import random

import pytest

from wow_bot.humanize.cursor import (
    CursorConfig,
    CursorError,
    Trajectory,
    TrajectoryPoint,
    generate_trajectory,
    straight_line_points,
)


def test_cursor_config_jitter_fraction_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="control_point_jitter_fraction"):
        CursorConfig(control_point_jitter_fraction=-0.1)
    with pytest.raises(ValueError, match="control_point_jitter_fraction"):
        CursorConfig(control_point_jitter_fraction=1.1)


def test_cursor_config_overshoot_fraction_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="overshoot_fraction"):
        CursorConfig(overshoot_fraction=-0.01)
    with pytest.raises(ValueError, match="overshoot_fraction"):
        CursorConfig(overshoot_fraction=1.05)


def test_cursor_config_overshoot_probability_out_of_bounds() -> None:
    with pytest.raises(ValueError, match="overshoot_probability"):
        CursorConfig(overshoot_probability=-0.5)
    with pytest.raises(ValueError, match="overshoot_probability"):
        CursorConfig(overshoot_probability=1.5)


def test_cursor_config_duration_sigma_invalid() -> None:
    with pytest.raises(ValueError, match="duration_sigma"):
        CursorConfig(duration_sigma=0.0)
    with pytest.raises(ValueError, match="duration_sigma"):
        CursorConfig(duration_sigma=-0.2)


def test_cursor_config_duration_clip_bounds_invalid() -> None:
    with pytest.raises(ValueError, match="duration_clip_high"):
        CursorConfig(duration_clip_low=0.5, duration_clip_high=0.5)
    with pytest.raises(ValueError, match="duration_clip_high"):
        CursorConfig(duration_clip_low=0.8, duration_clip_high=0.2)


def test_cursor_config_points_per_100ms_invalid() -> None:
    with pytest.raises(ValueError, match="points_per_100ms"):
        CursorConfig(points_per_100ms=0)


def test_cursor_config_min_points_invalid() -> None:
    with pytest.raises(ValueError, match="min_points"):
        CursorConfig(min_points=1)


def test_cursor_config_max_points_invalid() -> None:
    with pytest.raises(ValueError, match="max_points"):
        CursorConfig(min_points=10, max_points=5)


def test_cursor_config_min_distance_units_invalid() -> None:
    with pytest.raises(ValueError, match="min_distance_units"):
        CursorConfig(min_distance_units=-0.1)


def test_cursor_config_non_finite_float() -> None:
    with pytest.raises(ValueError, match="finite float"):
        CursorConfig(control_point_jitter_fraction=float("nan"))
    with pytest.raises(ValueError, match="finite float"):
        CursorConfig(duration_mu=float("inf"))


def test_trajectory_point_non_finite_field() -> None:
    with pytest.raises(ValueError, match="finite float"):
        TrajectoryPoint(x=float("nan"), y=10.0, t_offset=0.0)
    with pytest.raises(ValueError, match="finite float"):
        TrajectoryPoint(x=10.0, y=float("inf"), t_offset=0.0)
    with pytest.raises(ValueError, match="finite float"):
        TrajectoryPoint(x=10.0, y=10.0, t_offset=float("-inf"))


def test_trajectory_invariants() -> None:
    pt1 = TrajectoryPoint(0.0, 0.0, 0.0)
    pt2 = TrajectoryPoint(10.0, 10.0, 0.5)

    # Empty points
    with pytest.raises(ValueError, match="at least 1 point"):
        Trajectory(
            points=(),
            start=(0.0, 0.0),
            end=(10.0, 10.0),
            duration_s=0.5,
            overshoot_applied=False,
        )

    # First t_offset != 0
    bad_pt1 = TrajectoryPoint(0.0, 0.0, 0.1)
    with pytest.raises(ValueError, match="t_offset must be exactly 0.0"):
        Trajectory(
            points=(bad_pt1, pt2),
            start=(0.0, 0.0),
            end=(10.0, 10.0),
            duration_s=0.5,
            overshoot_applied=False,
        )

    # Non-monotone t_offsets
    pt_decrease = TrajectoryPoint(5.0, 5.0, 0.2)
    with pytest.raises(ValueError, match="non-decreasing"):
        Trajectory(
            points=(pt1, pt2, pt_decrease),
            start=(0.0, 0.0),
            end=(5.0, 5.0),
            duration_s=0.2,
            overshoot_applied=False,
        )

    # duration_s <= 0
    with pytest.raises(ValueError, match="duration_s must be > 0.0"):
        Trajectory(
            points=(pt1,),
            start=(0.0, 0.0),
            end=(0.0, 0.0),
            duration_s=0.0,
            overshoot_applied=False,
        )


def test_generate_trajectory_normal_move_endpoints() -> None:
    rng = random.Random(42)
    start = (100.0, 200.0)
    end = (400.0, 600.0)
    traj = generate_trajectory(rng, start, end)

    assert len(traj.points) >= 2
    assert abs(traj.points[0].x - start[0]) < 1e-9
    assert abs(traj.points[0].y - start[1]) < 1e-9
    assert abs(traj.points[-1].x - end[0]) < 1e-9
    assert abs(traj.points[-1].y - end[1]) < 1e-9


def test_generate_trajectory_monotone_t_offsets() -> None:
    rng = random.Random(123)
    traj = generate_trajectory(rng, (0.0, 0.0), (100.0, 100.0))
    for i in range(len(traj.points) - 1):
        assert traj.points[i].t_offset <= traj.points[i + 1].t_offset


def test_generate_trajectory_start_equals_end() -> None:
    rng = random.Random(1)
    start = (50.0, 50.0)
    end = (50.0, 50.0)
    traj = generate_trajectory(rng, start, end)

    assert len(traj.points) == 1
    assert abs(traj.points[0].x - end[0]) < 1e-9
    assert abs(traj.points[0].y - end[1]) < 1e-9
    assert traj.duration_s > 0.0


def test_generate_trajectory_very_close_points() -> None:
    cfg = CursorConfig(min_distance_units=1.0)
    rng = random.Random(2)
    start = (10.0, 10.0)
    end = (10.3, 10.4)  # dist = 0.5 < 1.0
    traj = generate_trajectory(rng, start, end, config=cfg)

    assert len(traj.points) == 1
    assert abs(traj.points[0].x - end[0]) < 1e-9
    assert abs(traj.points[0].y - end[1]) < 1e-9


def test_generate_trajectory_max_points_honored() -> None:
    cfg = CursorConfig(
        max_points=10,
        points_per_100ms=100,
        duration_clip_low=1.0,
        duration_clip_high=1.2,
    )
    rng = random.Random(3)
    traj = generate_trajectory(rng, (0.0, 0.0), (1000.0, 1000.0), config=cfg)
    assert len(traj.points) <= 10


def test_generate_trajectory_min_points_honored() -> None:
    cfg = CursorConfig(
        min_points=15,
        duration_clip_low=0.05,
        duration_clip_high=0.06,
        points_per_100ms=1,
    )
    rng = random.Random(4)
    traj = generate_trajectory(rng, (0.0, 0.0), (100.0, 100.0), config=cfg)
    assert len(traj.points) >= 15


def test_generate_trajectory_jitter_deviation() -> None:
    cfg = CursorConfig(control_point_jitter_fraction=0.3, overshoot_probability=0.0)
    rng = random.Random(99)
    start = (0.0, 0.0)
    end = (100.0, 0.0)
    traj = generate_trajectory(rng, start, end, config=cfg)

    # With start and end on y=0 line, straight line is y=0.
    # Verify at least one intermediate point has |y| > 1e-3.
    max_y_dev = max(abs(pt.y) for pt in traj.points[1:-1])
    assert max_y_dev > 1e-3


def test_generate_trajectory_zero_jitter_straight_line() -> None:
    cfg = CursorConfig(control_point_jitter_fraction=0.0, overshoot_probability=0.0)
    rng = random.Random(100)
    start = (10.0, 20.0)
    end = (110.0, 220.0)
    traj = generate_trajectory(rng, start, end, config=cfg)

    dx = end[0] - start[0]
    dy = end[1] - start[1]
    for pt in traj.points:
        # Check projection / distance to line start-end
        # Line equation: dy*x - dx*y + (start[0]*end[1] - end[0]*start[1]) = 0
        dist_to_line = abs(dy * pt.x - dx * pt.y + (start[0] * end[1] - end[0] * start[1])) / math.sqrt(dx * dx + dy * dy)
        assert dist_to_line < 1e-5


def test_overshoot_applied_flag() -> None:
    cfg1 = CursorConfig(overshoot_probability=1.0, overshoot_fraction=0.1)
    rng1 = random.Random(10)
    traj1 = generate_trajectory(rng1, (0.0, 0.0), (100.0, 0.0), config=cfg1)
    assert traj1.overshoot_applied is True

    cfg0 = CursorConfig(overshoot_probability=0.0)
    rng0 = random.Random(11)
    traj0 = generate_trajectory(rng0, (0.0, 0.0), (100.0, 0.0), config=cfg0)
    assert traj0.overshoot_applied is False


def test_overshoot_midpoint_beyond_end() -> None:
    cfg = CursorConfig(overshoot_probability=1.0, overshoot_fraction=0.2, control_point_jitter_fraction=0.0)
    rng = random.Random(77)
    start = (0.0, 0.0)
    end = (100.0, 0.0)
    traj = generate_trajectory(rng, start, end, config=cfg)

    # At least one intermediate point x should be > end x (100.0 + epsilon)
    max_x = max(pt.x for pt in traj.points)
    assert max_x > 105.0


def test_duration_within_bounds() -> None:
    cfg = CursorConfig(duration_clip_low=0.2, duration_clip_high=0.5)
    rng = random.Random(55)
    for _ in range(10):
        traj = generate_trajectory(rng, (0.0, 0.0), (100.0, 100.0), config=cfg)
        assert cfg.duration_clip_low <= traj.duration_s <= cfg.duration_clip_high


def test_determinism_same_and_different_seeds() -> None:
    start = (10.0, 20.0)
    end = (200.0, 300.0)

    traj_a1 = generate_trajectory(random.Random(12345), start, end)
    traj_a2 = generate_trajectory(random.Random(12345), start, end)

    assert traj_a1 == traj_a2

    traj_b = generate_trajectory(random.Random(54321), start, end)
    assert traj_a1 != traj_b


def test_straight_line_points_functionality() -> None:
    start = (0.0, 0.0)
    end = (100.0, 200.0)
    pts = straight_line_points(start, end, 5)

    assert len(pts) == 5
    assert (pts[0].x, pts[0].y, pts[0].t_offset) == (0.0, 0.0, 0.0)
    assert (pts[-1].x, pts[-1].y, pts[-1].t_offset) == (100.0, 200.0, 1.0)
    assert pts[2].x == 50.0
    assert pts[2].y == 100.0
    assert pts[2].t_offset == 0.5


def test_straight_line_points_invalid_count() -> None:
    with pytest.raises(CursorError, match="point_count must be >= 2"):
        straight_line_points((0.0, 0.0), (10.0, 10.0), 1)


def test_straight_line_points_deterministic_no_rng() -> None:
    pts1 = straight_line_points((0.0, 0.0), (50.0, 50.0), 10)
    pts2 = straight_line_points((0.0, 0.0), (50.0, 50.0), 10)
    assert pts1 == pts2


def test_trajectory_to_json() -> None:
    rng = random.Random(42)
    traj = generate_trajectory(rng, (0.0, 0.0), (100.0, 100.0))
    d = traj.to_json()

    assert isinstance(d, dict)
    assert d["start"] == [0.0, 0.0]
    assert d["end"] == [100.0, 100.0]
    assert isinstance(d["duration_s"], float)
    assert isinstance(d["overshoot_applied"], bool)
    assert isinstance(d["points"], list)
    for p in d["points"]:
        assert set(p.keys()) == {"x", "y", "t_offset"}

    # Verify JSON serializability
    json_str = json.dumps(d)
    assert len(json_str) > 0


def test_generate_trajectory_raises_cursor_error_on_non_finite_inputs() -> None:
    rng = random.Random(0)
    with pytest.raises(CursorError, match="finite numbers"):
        generate_trajectory(rng, (float("nan"), 0.0), (10.0, 10.0))
    with pytest.raises(CursorError, match="finite numbers"):
        generate_trajectory(rng, (0.0, 0.0), (10.0, float("inf")))


def test_bounding_box_no_overshoot() -> None:
    cfg = CursorConfig(control_point_jitter_fraction=0.15, overshoot_probability=0.0)
    rng = random.Random(888)
    start = (100.0, 100.0)
    end = (300.0, 200.0)
    traj = generate_trajectory(rng, start, end, config=cfg)

    dx = abs(end[0] - start[0])
    dy = abs(end[1] - start[1])
    dist = math.sqrt(dx * dx + dy * dy)
    margin = (cfg.control_point_jitter_fraction + 0.01) * dist

    min_x = min(start[0], end[0]) - margin
    max_x = max(start[0], end[0]) + margin
    min_y = min(start[1], end[1]) - margin
    max_y = max(start[1], end[1]) + margin

    for pt in traj.points:
        assert min_x <= pt.x <= max_x
        assert min_y <= pt.y <= max_y


def test_static_ast_checks_forbidden_imports_and_time() -> None:
    with open("src/wow_bot/humanize/cursor.py", "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="src/wow_bot/humanize/cursor.py")

    forbidden_modules = [
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
        "numpy",
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert not any(
                    mod == f or mod.startswith(f + ".") for f in forbidden_modules
                ), f"Forbidden import: {mod}"
                assert not any(
                    kw in mod for kw in ("ollama", "openai", "anthropic", "llm")
                ), f"Forbidden LLM import: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert not any(
                mod == f or mod.startswith(f + ".") for f in forbidden_modules
            ), f"Forbidden import: {mod}"
            assert not any(
                kw in mod for kw in ("ollama", "openai", "anthropic", "llm")
            ), f"Forbidden LLM import: {mod}"
        elif (
            isinstance(node, ast.Attribute)
            and isinstance(node.value, ast.Name)
            and node.value.id == "time"
        ):
            assert node.attr not in (
                "monotonic",
                "time",
                "perf_counter",
            ), f"Forbidden time call: time.{node.attr}"
