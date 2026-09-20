"""Unit tests for wow_bot.executor.recovery module."""

import ast
import math
import random
from pathlib import Path

import pytest

from wow_bot.actuation.mapper import MoveTo, Turn
from wow_bot.executor.fsm_v2 import GameStateLike, MetaStateLike
from wow_bot.executor.recovery import (
    RecoveryBehavior,
    RecoveryBehaviorKind,
    RecoveryConfig,
    RecoveryError,
    RecoveryPlanner,
    alternative_waypoint_intent,
    backstep_intent,
    camera_sweep_intent,
    jump_intent,
    pick_behavior,
)
from wow_bot.executor.states import FSMState


class DummyGameState(GameStateLike):
    """Dummy game state for testing."""


class DummyMetaState(MetaStateLike):
    """Dummy meta state for testing."""


def test_recovery_config_validation_max_attempts() -> None:
    """RecoveryConfig with max_attempts < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_attempts must be >= 1"):
        RecoveryConfig(max_attempts=0)


def test_recovery_config_validation_non_positive_params() -> None:
    """RecoveryConfig with non-positive distance/radius/sweep raises ValueError."""
    with pytest.raises(ValueError, match="backstep_distance_units must be > 0"):
        RecoveryConfig(backstep_distance_units=0.0)

    with pytest.raises(ValueError, match="camera_sweep_rad must be > 0"):
        RecoveryConfig(camera_sweep_rad=-0.1)

    with pytest.raises(ValueError, match="alternative_radius_units must be > 0"):
        RecoveryConfig(alternative_radius_units=0.0)


def test_recovery_config_validation_alternative_samples() -> None:
    """RecoveryConfig with alternative_samples < 2 raises ValueError."""
    with pytest.raises(ValueError, match="alternative_samples must be >= 2"):
        RecoveryConfig(alternative_samples=1)


def test_recovery_config_validation_behavior_weights_length() -> None:
    """RecoveryConfig with wrong-length behavior_weights raises ValueError."""
    with pytest.raises(ValueError, match="behavior_weights must have length 4"):
        RecoveryConfig(behavior_weights=(1.0, 1.0, 1.0))


def test_recovery_config_validation_behavior_weights_negative() -> None:
    """RecoveryConfig with a negative weight raises ValueError."""
    with pytest.raises(ValueError, match="behavior_weights must be non-negative"):
        RecoveryConfig(behavior_weights=(1.0, -0.5, 1.0, 1.0))


def test_recovery_config_validation_behavior_weights_all_zero() -> None:
    """RecoveryConfig with all-zero weights raises ValueError."""
    with pytest.raises(ValueError, match="sum of behavior_weights must be > 0"):
        RecoveryConfig(behavior_weights=(0.0, 0.0, 0.0, 0.0))


def test_pick_behavior_deterministic_weights() -> None:
    """pick_behavior with deterministic single non-zero weights."""
    rng = random.Random(42)
    config_backstep = RecoveryConfig(behavior_weights=(1.0, 0.0, 0.0, 0.0))
    for _ in range(10):
        assert pick_behavior(rng, config_backstep) == RecoveryBehaviorKind.BACKSTEP

    config_alt = RecoveryConfig(behavior_weights=(0.0, 0.0, 0.0, 1.0))
    for _ in range(10):
        assert pick_behavior(rng, config_alt) == RecoveryBehaviorKind.ALTERNATIVE_WAYPOINT


def test_pick_behavior_determinism_sequence() -> None:
    """pick_behavior is deterministic: same rng seed -> same sequence over 100 calls."""
    config = RecoveryConfig(behavior_weights=(1.0, 2.0, 0.5, 1.5))
    seq1 = [pick_behavior(random.Random(12345), config) for _ in range(100)]
    seq2 = [pick_behavior(random.Random(12345), config) for _ in range(100)]
    assert seq1 == seq2


def test_backstep_intent_math() -> None:
    """backstep_intent calculations across various headings and positions."""
    config = RecoveryConfig(backstep_distance_units=2.0)

    # heading=0, position=(0,0), distance=2 returns MoveTo(-2, 0)
    intent1 = backstep_intent(position=(0.0, 0.0), heading=0.0, config=config)
    assert isinstance(intent1, MoveTo)
    assert pytest.approx(intent1.x, abs=1e-9) == -2.0
    assert pytest.approx(intent1.y, abs=1e-9) == 0.0

    # heading=pi/2, position=(0,0), distance=2 returns MoveTo approximately (0, -2) within 1e-9
    intent2 = backstep_intent(position=(0.0, 0.0), heading=math.pi / 2, config=config)
    assert isinstance(intent2, MoveTo)
    assert pytest.approx(intent2.x, abs=1e-9) == 0.0
    assert pytest.approx(intent2.y, abs=1e-9) == -2.0

    # heading=pi, position=(1,1), distance=1 returns MoveTo approximately (2, 1)
    config_dist1 = RecoveryConfig(backstep_distance_units=1.0)
    intent3 = backstep_intent(position=(1.0, 1.0), heading=math.pi, config=config_dist1)
    assert isinstance(intent3, MoveTo)
    assert pytest.approx(intent3.x, abs=1e-9) == 2.0
    assert pytest.approx(intent3.y, abs=1e-9) == 1.0


def test_camera_sweep_intent_bounds_and_determinism() -> None:
    """camera_sweep_intent angle is within [-sweep, +sweep] and deterministic."""
    sweep_rad = 0.6
    config = RecoveryConfig(camera_sweep_rad=sweep_rad)

    rng = random.Random(999)
    for _ in range(1000):
        intent = camera_sweep_intent(rng, config=config)
        assert isinstance(intent, Turn)
        assert -sweep_rad <= intent.angle_rad <= sweep_rad

    rng_a = random.Random(42)
    rng_b = random.Random(42)
    res_a = [camera_sweep_intent(rng_a, config=config) for _ in range(20)]
    res_b = [camera_sweep_intent(rng_b, config=config) for _ in range(20)]
    assert res_a == res_b


def test_jump_intent() -> None:
    """jump_intent returns Turn(angle_rad=0.0)."""
    intent = jump_intent()
    assert isinstance(intent, Turn)
    assert intent.angle_rad == 0.0


def test_alternative_waypoint_intent_distance_and_sampling() -> None:
    """alternative_waypoint_intent returns MoveTo at correct radius, deterministic and distinct."""
    radius = 5.0
    config = RecoveryConfig(alternative_radius_units=radius, alternative_samples=8)
    pos = (10.0, 20.0)

    rng = random.Random(77)
    intent = alternative_waypoint_intent(rng, position=pos, config=config)
    assert isinstance(intent, MoveTo)
    dist = math.hypot(intent.x - pos[0], intent.y - pos[1])
    assert pytest.approx(dist, abs=1e-6) == radius

    # Determinism given seed
    r1 = alternative_waypoint_intent(random.Random(123), position=pos, config=config)
    r2 = alternative_waypoint_intent(random.Random(123), position=pos, config=config)
    assert r1 == r2

    # Samples distinct angles across many calls with same rng
    samples = [
        alternative_waypoint_intent(rng, position=pos, config=config)
        for _ in range(50)
    ]
    unique_coords = {
        (round(m.x, 4), round(m.y, 4)) for m in samples if isinstance(m, MoveTo)
    }
    assert len(unique_coords) > 1


def test_recovery_planner_max_attempts_exceeded() -> None:
    """RecoveryPlanner with attempts_so_far >= max_attempts raises RecoveryError."""
    config = RecoveryConfig(max_attempts=3)
    planner = RecoveryPlanner(config=config)
    rng = random.Random(0)

    with pytest.raises(RecoveryError, match="Maximum recovery attempts reached"):
        planner.plan(rng, attempts_so_far=3)

    with pytest.raises(RecoveryError, match="Maximum recovery attempts reached"):
        planner.plan(rng, attempts_so_far=4)


def test_recovery_planner_dispatch_backstep() -> None:
    """RecoveryPlanner.plan with weights (1,0,0,0) calls sources once each and returns MoveTo."""
    config = RecoveryConfig(behavior_weights=(1.0, 0.0, 0.0, 0.0))
    pos_calls = 0
    heading_calls = 0

    def pos_source() -> tuple[float, float]:
        nonlocal pos_calls
        pos_calls += 1
        return (5.0, 5.0)

    def heading_source() -> float:
        nonlocal heading_calls
        heading_calls += 1
        return 0.0

    planner = RecoveryPlanner(
        config=config,
        position_source=pos_source,
        heading_source=heading_source,
    )
    rng = random.Random(1)
    intent = planner.plan(rng, attempts_so_far=0)

    assert isinstance(intent, MoveTo)
    assert pos_calls == 1
    assert heading_calls == 1
    assert planner.last_behavior() == RecoveryBehaviorKind.BACKSTEP


def test_recovery_planner_dispatch_camera_sweep() -> None:
    """RecoveryPlanner.plan with weights (0,1,0,0) does NOT call sources and returns Turn."""
    config = RecoveryConfig(behavior_weights=(0.0, 1.0, 0.0, 0.0))
    pos_calls = 0
    heading_calls = 0

    def pos_source() -> tuple[float, float]:
        nonlocal pos_calls
        pos_calls += 1
        return (0.0, 0.0)

    def heading_source() -> float:
        nonlocal heading_calls
        heading_calls += 1
        return 0.0

    planner = RecoveryPlanner(
        config=config,
        position_source=pos_source,
        heading_source=heading_source,
    )
    rng = random.Random(1)
    intent = planner.plan(rng, attempts_so_far=0)

    assert isinstance(intent, Turn)
    assert pos_calls == 0
    assert heading_calls == 0
    assert planner.last_behavior() == RecoveryBehaviorKind.CAMERA_SWEEP


def test_recovery_planner_dispatch_jump() -> None:
    """RecoveryPlanner.plan with weights (0,0,1,0) does NOT call sources and returns Turn(0.0)."""
    config = RecoveryConfig(behavior_weights=(0.0, 0.0, 1.0, 0.0))
    pos_calls = 0
    heading_calls = 0

    def pos_source() -> tuple[float, float]:
        nonlocal pos_calls
        pos_calls += 1
        return (0.0, 0.0)

    def heading_source() -> float:
        nonlocal heading_calls
        heading_calls += 1
        return 0.0

    planner = RecoveryPlanner(
        config=config,
        position_source=pos_source,
        heading_source=heading_source,
    )
    rng = random.Random(1)
    intent = planner.plan(rng, attempts_so_far=0)

    assert intent == Turn(angle_rad=0.0)
    assert pos_calls == 0
    assert heading_calls == 0
    assert planner.last_behavior() == RecoveryBehaviorKind.JUMP


def test_recovery_planner_dispatch_alternative_waypoint() -> None:
    """RecoveryPlanner.plan with weights (0,0,0,1) calls position_source once and returns MoveTo."""
    config = RecoveryConfig(
        behavior_weights=(0.0, 0.0, 0.0, 1.0),
        alternative_radius_units=4.0,
    )
    pos_calls = 0
    heading_calls = 0

    def pos_source() -> tuple[float, float]:
        nonlocal pos_calls
        pos_calls += 1
        return (10.0, 10.0)

    def heading_source() -> float:
        nonlocal heading_calls
        heading_calls += 1
        return 0.0

    planner = RecoveryPlanner(
        config=config,
        position_source=pos_source,
        heading_source=heading_source,
    )
    rng = random.Random(1)
    intent = planner.plan(rng, attempts_so_far=0)

    assert isinstance(intent, MoveTo)
    assert pos_calls == 1
    assert heading_calls == 0
    dist = math.hypot(intent.x - 10.0, intent.y - 10.0)
    assert pytest.approx(dist, abs=1e-6) == 4.0
    assert planner.last_behavior() == RecoveryBehaviorKind.ALTERNATIVE_WAYPOINT


def test_recovery_planner_source_exception_propagation() -> None:
    """RecoveryPlanner propagates exceptions from position_source and heading_source."""
    config = RecoveryConfig(behavior_weights=(1.0, 0.0, 0.0, 0.0))

    def bad_pos() -> tuple[float, float]:
        raise RuntimeError("Position unavailable")

    planner = RecoveryPlanner(config=config, position_source=bad_pos)
    with pytest.raises(RuntimeError, match="Position unavailable"):
        planner.plan(random.Random(1), attempts_so_far=0)


def test_recovery_behavior_decide_and_state_filtering() -> None:
    """RecoveryBehavior.decide returns None for non-STUCK_RECOVERY states, intent for STUCK_RECOVERY."""
    behavior = RecoveryBehavior()
    gs = DummyGameState()
    ms = DummyMetaState()
    rng = random.Random(42)

    # Returns None for non-STUCK_RECOVERY states without incrementing attempts
    for state in FSMState:
        if state != FSMState.STUCK_RECOVERY:
            assert behavior.decide(state, gs, ms, 0.0, rng) is None
    assert behavior.attempts() == 0

    # Returns intent for STUCK_RECOVERY and increments attempts
    intent1 = behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)
    assert intent1 is not None
    assert behavior.attempts() == 1

    intent2 = behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)
    assert intent2 is not None
    assert behavior.attempts() == 2


def test_recovery_behavior_reset_attempts() -> None:
    """RecoveryBehavior.reset_attempts clears the attempt counter."""
    behavior = RecoveryBehavior()
    gs = DummyGameState()
    ms = DummyMetaState()
    rng = random.Random(42)

    behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)
    assert behavior.attempts() == 1

    behavior.reset_attempts()
    assert behavior.attempts() == 0


def test_recovery_behavior_exceeds_max_attempts() -> None:
    """RecoveryBehavior raises RecoveryError when attempts exceed config.max_attempts."""
    config = RecoveryConfig(max_attempts=2)
    behavior = RecoveryBehavior(config=config)
    gs = DummyGameState()
    ms = DummyMetaState()
    rng = random.Random(42)

    behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)  # attempt 0 -> 1
    behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)  # attempt 1 -> 2

    with pytest.raises(RecoveryError, match="Maximum recovery attempts reached"):
        behavior.decide(FSMState.STUCK_RECOVERY, gs, ms, 0.0, rng)


def test_recovery_planner_determinism_sequence() -> None:
    """Two RecoveryPlanner instances with identical inputs and seeds produce identical sequences over 100 calls."""
    config = RecoveryConfig(max_attempts=200)

    p1 = RecoveryPlanner(config=config)
    p2 = RecoveryPlanner(config=config)

    rng1 = random.Random(1001)
    rng2 = random.Random(1001)

    seq1 = [p1.plan(rng1, attempts_so_far=i) for i in range(100)]
    seq2 = [p2.plan(rng2, attempts_so_far=i) for i in range(100)]

    assert seq1 == seq2


def test_static_ast_imports() -> None:
    """Static AST check: recovery.py does not import forbidden modules or LLM dependencies."""
    recovery_path = Path(__file__).parent.parent / "src" / "wow_bot" / "executor" / "recovery.py"
    tree = ast.parse(recovery_path.read_text(encoding="utf-8"))

    forbidden_substrings = (
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
        "wow_bot.actuation.actuator",
        "wow_bot.reflex",
        "wow_bot.session",
        "ollama",
        "openai",
        "anthropic",
        "llm",
    )

    imported_names: list[str] = []

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            imported_names.append(module)
            for alias in node.names:
                imported_names.append(f"{module}.{alias.name}")

    for imported in imported_names:
        for forbidden in forbidden_substrings:
            assert forbidden not in imported.lower(), (
                f"Forbidden import '{imported}' found in {recovery_path}"
            )
