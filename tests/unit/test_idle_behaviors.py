"""Unit tests for synthetic idle behaviors and IdleBehaviorEngine (Task 5.4)."""

from __future__ import annotations

import ast
import inspect
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

import wow_bot.executor.idle_behaviors as idle_mod
from wow_bot.executor.idle_behaviors import (
    IDLE_TRIGGER_PROBABILITY,
    SOCIAL_EMOTES,
    IdleBehavior,
    IdleBehaviorEngine,
    IdleIntent,
    angled_movement,
    camera_wander,
    inventory_check,
    social_emote,
    sudden_pause,
)


def test_a_camera_callable() -> None:
    """Test A: camera_wander returns CAMERA_WANDER with no executable metadata."""
    intent = camera_wander()
    assert isinstance(intent, IdleIntent)
    assert intent.behavior == IdleBehavior.CAMERA_WANDER
    assert intent.emote is None


def test_b_sudden_pause_callable() -> None:
    """Test B: sudden_pause returns SUDDEN_PAUSE."""
    intent = sudden_pause()
    assert isinstance(intent, IdleIntent)
    assert intent.behavior == IdleBehavior.SUDDEN_PAUSE
    assert intent.emote is None


def test_c_inventory_callable() -> None:
    """Test C: inventory_check returns INVENTORY_CHECK."""
    intent = inventory_check()
    assert isinstance(intent, IdleIntent)
    assert intent.behavior == IdleBehavior.INVENTORY_CHECK
    assert intent.emote is None


def test_d_social_emote_callable() -> None:
    """Test D: social_emote returns SOCIAL_EMOTE with allowed symbolic emote."""
    rng = np.random.default_rng(42)
    intent = social_emote(rng=rng)
    assert isinstance(intent, IdleIntent)
    assert intent.behavior == IdleBehavior.SOCIAL_EMOTE
    assert intent.emote in SOCIAL_EMOTES


def test_e_angled_movement_callable() -> None:
    """Test E: angled_movement returns ANGLED_MOVEMENT with no coordinates/paths."""
    intent = angled_movement()
    assert isinstance(intent, IdleIntent)
    assert intent.behavior == IdleBehavior.ANGLED_MOVEMENT
    assert intent.emote is None


def test_f_curiosity_boundaries() -> None:
    """Test F: curiosity values 0.0 and 1.0 are accepted."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    # Should not raise exception
    engine.maybe_generate(curiosity=0.0)
    engine.maybe_generate(curiosity=1.0)


@pytest.mark.parametrize(
    "invalid_curiosity",
    [
        -0.01,
        1.01,
        float("nan"),
        float("inf"),
        float("-inf"),
        True,
        False,
        "0.5",
        None,
        [0.5],
    ],
)
def test_g_invalid_curiosity(invalid_curiosity: object) -> None:
    """Test G: invalid curiosity values are rejected with ValueError."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    with pytest.raises(ValueError):
        engine.maybe_generate(curiosity=invalid_curiosity)  # type: ignore[arg-type]


def test_h_no_event_possible() -> None:
    """Test H: maybe_generate can return None (triggering < 100%)."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    none_count = 0
    for _ in range(50):
        res = engine.maybe_generate(curiosity=0.5)
        if res is None:
            none_count += 1
    assert none_count > 0


def test_i_event_possible() -> None:
    """Test I: maybe_generate can return an IdleIntent."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    fired_count = 0
    for _ in range(200):
        res = engine.maybe_generate(curiosity=0.5)
        if res is not None:
            fired_count += 1
            assert isinstance(res, IdleIntent)
    assert fired_count > 0


def test_j_100_tick_acceptance() -> None:
    """Test J: 100-tick canonical acceptance test (seed=42, curiosity=0.5)."""
    rng = np.random.default_rng(42)
    engine = IdleBehaviorEngine(rng=rng)

    fired_intents: list[IdleIntent] = []
    behavior_counts: dict[str, int] = {}

    for _ in range(100):
        intent = engine.maybe_generate(curiosity=0.5)
        if intent is not None:
            fired_intents.append(intent)
            name = intent.behavior.name
            behavior_counts[name] = behavior_counts.get(name, 0) + 1

    fired_count = len(fired_intents)
    assert fired_count >= 3, (
        f"Acceptance test failed: expected >= 3 fired intents in 100 ticks, "
        f"got {fired_count}. Breakdown: {behavior_counts}"
    )
    assert fired_count <= 20, f"Implausibly high event count: {fired_count}"


def test_k_reproducibility() -> None:
    """Test K: two engines with identical seed produce identical intent sequences."""
    engine1 = IdleBehaviorEngine(rng=np.random.default_rng(42))
    engine2 = IdleBehaviorEngine(rng=np.random.default_rng(42))

    curiosity_seq = [0.1, 0.5, 0.9, 0.2, 0.7] * 20
    seq1 = [engine1.maybe_generate(curiosity=c) for c in curiosity_seq]
    seq2 = [engine2.maybe_generate(curiosity=c) for c in curiosity_seq]

    assert seq1 == seq2


def test_l_different_seed() -> None:
    """Test L: distinct seeds produce different output sequences over nontrivial evaluations."""
    engine1 = IdleBehaviorEngine(rng=np.random.default_rng(42))
    engine2 = IdleBehaviorEngine(rng=np.random.default_rng(999))

    seq1 = [engine1.maybe_generate(curiosity=0.5) for _ in range(100)]
    seq2 = [engine2.maybe_generate(curiosity=0.5) for _ in range(100)]

    assert seq1 != seq2


def test_m_camera_more_likely_at_low_curiosity() -> None:
    """Test M: camera wander frequency among fired events is higher at curiosity=0.0 vs curiosity=1.0."""
    engine_low = IdleBehaviorEngine(rng=np.random.default_rng(12345))
    engine_high = IdleBehaviorEngine(rng=np.random.default_rng(12345))

    n_samples = 5000

    low_fired = [
        engine_low.maybe_generate(curiosity=0.0)
        for _ in range(n_samples)
    ]
    high_fired = [
        engine_high.maybe_generate(curiosity=1.0)
        for _ in range(n_samples)
    ]

    low_non_none = [i for i in low_fired if i is not None]
    high_non_none = [i for i in high_fired if i is not None]

    low_camera_ratio = (
        sum(1 for i in low_non_none if i.behavior == IdleBehavior.CAMERA_WANDER)
        / len(low_non_none)
    )
    high_camera_ratio = (
        sum(1 for i in high_non_none if i.behavior == IdleBehavior.CAMERA_WANDER)
        / len(high_non_none)
    )

    assert low_camera_ratio > high_camera_ratio


def test_n_behavior_diversity() -> None:
    """Test N: over a large sample, at least >= 3 distinct behavior types fire."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    observed_behaviors: set[IdleBehavior] = set()

    for _ in range(1000):
        intent = engine.maybe_generate(curiosity=0.5)
        if intent is not None:
            observed_behaviors.add(intent.behavior)

    assert len(observed_behaviors) >= 3


def test_o_only_allowed_behaviors() -> None:
    """Test O: all returned behaviors belong to IdleBehavior enum."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))
    allowed = set(IdleBehavior)

    for _ in range(500):
        intent = engine.maybe_generate(curiosity=0.5)
        if intent is not None:
            assert intent.behavior in allowed


def test_p_emote_vocabulary() -> None:
    """Test P: social emote metadata is strictly wave or laugh."""
    engine = IdleBehaviorEngine(rng=np.random.default_rng(42))

    for _ in range(2000):
        intent = engine.maybe_generate(curiosity=0.5)
        if intent is not None and intent.behavior == IdleBehavior.SOCIAL_EMOTE:
            assert intent.emote in ("wave", "laugh")


def test_q_no_real_io_imports() -> None:
    """Test Q: module does not import physical input drivers or Controller."""
    source = inspect.getsource(idle_mod)
    tree = ast.parse(source)

    imported_names: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_names.add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imported_names.add(node.module)
            for alias in node.names:
                imported_names.add(alias.name)

    forbidden = {"pynput", "pyautogui", "keyboard", "mouse", "Controller", "wow_bot.executor.controller"}
    intersection = imported_names.intersection(forbidden)
    assert not intersection, f"Forbidden imports found: {intersection}"


def test_r_no_sleep() -> None:
    """Test R: module does not call sleep functions."""
    source = inspect.getsource(idle_mod)
    assert "time.sleep" not in source
    assert "asyncio.sleep" not in source


def test_s_no_path_generation() -> None:
    """Test S: angled_movement produces no path or coordinate structure."""
    intent = angled_movement()
    assert not hasattr(intent, "points")
    assert not hasattr(intent, "coordinates")
    assert not hasattr(intent, "path")


def test_t_intent_immutability() -> None:
    """Test T: IdleIntent is immutable."""
    intent = IdleIntent(behavior=IdleBehavior.CAMERA_WANDER)
    with pytest.raises(FrozenInstanceError):
        intent.behavior = IdleBehavior.SUDDEN_PAUSE  # type: ignore[misc]


def test_trigger_probability_constant() -> None:
    """Verify trigger probability constant remains 0.08."""
    assert IDLE_TRIGGER_PROBABILITY == 0.08
