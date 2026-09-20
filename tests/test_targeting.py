"""
Unit and static AST tests for wow_bot.combat.targeting.
"""

import ast
from dataclasses import dataclass
from pathlib import Path

import pytest

from wow_bot.combat.targeting import (
    PriorityMetric,
    TargetConfig,
    TargetEntityLike,
    TargetingStateLike,
    TargetSelector,
)


@dataclass(frozen=True)
class FakeEntity(TargetEntityLike):
    entity_id: str
    distance: float = 10.0
    threat: float = 100.0
    hp_percent: float = 100.0
    is_attackable: bool = True
    is_alive: bool = True
    is_in_combat_with_self: bool = True


@dataclass(frozen=True)
class FakeState(TargetingStateLike):
    entities: tuple[FakeEntity, ...] = ()
    current_target_id: str | None = None


# Acceptance Criteria Tests


def test_target_config_invalid_max_distance() -> None:
    with pytest.raises(ValueError, match="max_distance"):
        TargetConfig(max_distance=0.0)
    with pytest.raises(ValueError, match="max_distance"):
        TargetConfig(max_distance=-5.0)


def test_target_config_invalid_min_hp_percent() -> None:
    with pytest.raises(ValueError, match="min_hp_percent"):
        TargetConfig(min_hp_percent=-0.1)
    with pytest.raises(ValueError, match="min_hp_percent"):
        TargetConfig(min_hp_percent=100.1)


def test_target_config_empty_priority() -> None:
    with pytest.raises(ValueError, match="priority"):
        TargetConfig(priority=())


def test_target_config_duplicate_metrics() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        TargetConfig(
            priority=(
                PriorityMetric.THREAT,
                PriorityMetric.DISTANCE,
                PriorityMetric.THREAT,
            )
        )


def test_target_config_negative_switch_hysteresis() -> None:
    with pytest.raises(ValueError, match="switch_hysteresis"):
        TargetConfig(switch_hysteresis=-1.0)


def test_select_no_entities_returns_none() -> None:
    selector = TargetSelector()
    state = FakeState(entities=())
    decision = selector.select(state)
    assert decision.entity_id is None
    assert decision.reason == "no_candidates"
    assert decision.metric_values == ()


def test_filter_dead_entities() -> None:
    selector = TargetSelector()
    e_dead = FakeEntity(entity_id="e1", is_alive=False)
    e_alive = FakeEntity(entity_id="e2", is_alive=True)
    state = FakeState(entities=(e_dead, e_alive))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_filter_non_attackable_entities_when_required() -> None:
    selector = TargetSelector(TargetConfig(require_attackable=True))
    e1 = FakeEntity(entity_id="e1", is_attackable=False)
    e2 = FakeEntity(entity_id="e2", is_attackable=True)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_keep_non_attackable_entities_when_not_required() -> None:
    selector = TargetSelector(TargetConfig(require_attackable=False))
    e1 = FakeEntity(entity_id="e1", is_attackable=False, threat=200.0)
    e2 = FakeEntity(entity_id="e2", is_attackable=True, threat=100.0)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e1"


def test_filter_not_in_combat_when_required() -> None:
    selector = TargetSelector(TargetConfig(require_in_combat=True))
    e1 = FakeEntity(entity_id="e1", is_in_combat_with_self=False)
    e2 = FakeEntity(entity_id="e2", is_in_combat_with_self=True)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_keep_not_in_combat_when_not_required() -> None:
    selector = TargetSelector(TargetConfig(require_in_combat=False))
    e1 = FakeEntity(entity_id="e1", is_in_combat_with_self=False, threat=200.0)
    e2 = FakeEntity(entity_id="e2", is_in_combat_with_self=True, threat=100.0)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e1"


def test_filter_beyond_max_distance() -> None:
    selector = TargetSelector(TargetConfig(max_distance=20.0))
    e_far = FakeEntity(entity_id="e1", distance=20.1)
    e_near = FakeEntity(entity_id="e2", distance=19.9)
    state = FakeState(entities=(e_far, e_near))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_filter_below_min_hp_percent() -> None:
    selector = TargetSelector(TargetConfig(min_hp_percent=5.0))
    e_low = FakeEntity(entity_id="e1", hp_percent=4.9)
    e_ok = FakeEntity(entity_id="e2", hp_percent=5.0)
    state = FakeState(entities=(e_low, e_ok))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_priority_threat_first() -> None:
    selector = TargetSelector(TargetConfig(priority=(PriorityMetric.THREAT,)))
    e1 = FakeEntity(entity_id="e1", threat=50.0, distance=5.0)
    e2 = FakeEntity(entity_id="e2", threat=100.0, distance=10.0)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_priority_distance_first() -> None:
    selector = TargetSelector(TargetConfig(priority=(PriorityMetric.DISTANCE,)))
    e1 = FakeEntity(entity_id="e1", threat=200.0, distance=15.0)
    e2 = FakeEntity(entity_id="e2", threat=50.0, distance=5.0)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_priority_hp_first() -> None:
    selector = TargetSelector(TargetConfig(priority=(PriorityMetric.HP,)))
    e1 = FakeEntity(entity_id="e1", hp_percent=80.0)
    e2 = FakeEntity(entity_id="e2", hp_percent=20.0)
    state = FakeState(entities=(e1, e2))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_compound_priority_threat_then_distance() -> None:
    cfg = TargetConfig(priority=(PriorityMetric.THREAT, PriorityMetric.DISTANCE))
    selector = TargetSelector(cfg)
    e1 = FakeEntity(entity_id="e1", threat=100.0, distance=20.0)
    e2 = FakeEntity(entity_id="e2", threat=100.0, distance=10.0)
    e3 = FakeEntity(entity_id="e3", threat=50.0, distance=5.0)
    state = FakeState(entities=(e1, e2, e3))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_compound_priority_distance_then_threat() -> None:
    cfg = TargetConfig(priority=(PriorityMetric.DISTANCE, PriorityMetric.THREAT))
    selector = TargetSelector(cfg)
    e1 = FakeEntity(entity_id="e1", distance=10.0, threat=50.0)
    e2 = FakeEntity(entity_id="e2", distance=10.0, threat=150.0)
    e3 = FakeEntity(entity_id="e3", distance=20.0, threat=200.0)
    state = FakeState(entities=(e1, e2, e3))
    decision = selector.select(state)
    assert decision.entity_id == "e2"


def test_tie_breaking_lexicographical_id() -> None:
    cfg = TargetConfig(priority=(PriorityMetric.THREAT, PriorityMetric.DISTANCE))
    selector = TargetSelector(cfg)
    e_z = FakeEntity(entity_id="z_target", threat=100.0, distance=10.0)
    e_a = FakeEntity(entity_id="a_target", threat=100.0, distance=10.0)
    e_m = FakeEntity(entity_id="m_target", threat=100.0, distance=10.0)
    state = FakeState(entities=(e_z, e_a, e_m))
    decision = selector.select(state)
    assert decision.entity_id == "a_target"


def test_metric_values_follows_priority_order() -> None:
    cfg = TargetConfig(
        priority=(
            PriorityMetric.HP,
            PriorityMetric.THREAT,
            PriorityMetric.DISTANCE,
        )
    )
    selector = TargetSelector(cfg)
    e = FakeEntity(entity_id="e1", hp_percent=45.0, threat=120.0, distance=12.5)
    state = FakeState(entities=(e,))
    decision = selector.select(state)
    assert decision.entity_id == "e1"
    assert decision.metric_values == (
        ("hp", 45.0),
        ("threat", 120.0),
        ("distance", 12.5),
    )


def test_metric_values_empty_when_no_candidate() -> None:
    selector = TargetSelector()
    state = FakeState(entities=())
    decision = selector.select(state)
    assert decision.entity_id is None
    assert decision.metric_values == ()


def test_hysteresis_keeps_current_when_within_window() -> None:
    cfg = TargetConfig(
        priority=(PriorityMetric.THREAT, PriorityMetric.DISTANCE),
        switch_hysteresis=20.0,
    )
    selector = TargetSelector(cfg)
    e_best = FakeEntity(entity_id="e_best", threat=100.0, distance=10.0)
    e_curr = FakeEntity(entity_id="e_curr", threat=85.0, distance=10.0)
    state = FakeState(entities=(e_best, e_curr), current_target_id="e_curr")
    decision = selector.select(state)
    # abs(100.0 - 85.0) = 15.0 <= 20.0 -> keep e_curr
    assert decision.entity_id == "e_curr"


def test_hysteresis_switches_when_outside_window() -> None:
    cfg = TargetConfig(
        priority=(PriorityMetric.THREAT, PriorityMetric.DISTANCE),
        switch_hysteresis=10.0,
    )
    selector = TargetSelector(cfg)
    e_best = FakeEntity(entity_id="e_best", threat=100.0, distance=10.0)
    e_curr = FakeEntity(entity_id="e_curr", threat=85.0, distance=10.0)
    state = FakeState(entities=(e_best, e_curr), current_target_id="e_curr")
    decision = selector.select(state)
    # abs(100.0 - 85.0) = 15.0 > 10.0 -> switch to e_best
    assert decision.entity_id == "e_best"


def test_hysteresis_switches_when_current_not_in_candidates() -> None:
    cfg = TargetConfig(
        priority=(PriorityMetric.THREAT,),
        switch_hysteresis=100.0,
    )
    selector = TargetSelector(cfg)
    e_alive = FakeEntity(entity_id="e_alive", threat=50.0)
    e_dead = FakeEntity(entity_id="e_dead", threat=200.0, is_alive=False)
    state = FakeState(entities=(e_alive, e_dead), current_target_id="e_dead")
    decision = selector.select(state)
    assert decision.entity_id == "e_alive"


def test_hysteresis_zero_always_switches_to_top_ranked() -> None:
    cfg = TargetConfig(
        priority=(PriorityMetric.THREAT,),
        switch_hysteresis=0.0,
    )
    selector = TargetSelector(cfg)
    e_best = FakeEntity(entity_id="e_best", threat=100.0)
    e_curr = FakeEntity(entity_id="e_curr", threat=99.9)
    state = FakeState(entities=(e_best, e_curr), current_target_id="e_curr")
    decision = selector.select(state)
    assert decision.entity_id == "e_best"


def test_should_switch_differs_from_current() -> None:
    selector = TargetSelector()
    e1 = FakeEntity(entity_id="e1", threat=100.0)
    state = FakeState(entities=(e1,), current_target_id="e2")
    assert selector.should_switch(state) is True


def test_should_switch_equals_current() -> None:
    selector = TargetSelector()
    e1 = FakeEntity(entity_id="e1", threat=100.0)
    state = FakeState(entities=(e1,), current_target_id="e1")
    assert selector.should_switch(state) is False


def test_should_switch_current_none_candidate_chosen() -> None:
    selector = TargetSelector()
    e1 = FakeEntity(entity_id="e1", threat=100.0)
    state = FakeState(entities=(e1,), current_target_id=None)
    assert selector.should_switch(state) is True


def test_should_switch_current_none_no_candidate() -> None:
    selector = TargetSelector()
    state = FakeState(entities=(), current_target_id=None)
    assert selector.should_switch(state) is False


def test_select_does_not_mutate_state_or_config() -> None:
    cfg = TargetConfig(
        max_distance=30.0,
        min_hp_percent=1.0,
        priority=(PriorityMetric.DISTANCE,),
    )
    selector = TargetSelector(cfg)
    e1 = FakeEntity(entity_id="e1", distance=10.0)
    e2 = FakeEntity(entity_id="e2", distance=15.0)
    state = FakeState(entities=(e1, e2), current_target_id="e1")

    cfg_before = (cfg.max_distance, cfg.min_hp_percent, cfg.priority)
    state_before = (state.entities, state.current_target_id)

    selector.select(state)

    cfg_after = (cfg.max_distance, cfg.min_hp_percent, cfg.priority)
    state_after = (state.entities, state.current_target_id)

    assert cfg_before == cfg_after
    assert state_before == state_after


def test_determinism_100_runs() -> None:
    selector = TargetSelector()
    e1 = FakeEntity(entity_id="e1", threat=80.0, distance=10.0)
    e2 = FakeEntity(entity_id="e2", threat=120.0, distance=15.0)
    e3 = FakeEntity(entity_id="e3", threat=120.0, distance=12.0)
    state = FakeState(entities=(e1, e2, e3), current_target_id="e1")

    first_decision = selector.select(state)
    for _ in range(100):
        decision = selector.select(state)
        assert decision == first_decision


def test_static_ast_import_isolation() -> None:
    filepath = Path("src/wow_bot/combat/targeting.py")
    assert filepath.exists(), "targeting.py file does not exist"

    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "aiosqlite",
        "asyncio",
        "threading",
    }
    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        imported_modules: list[str] = []
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.append(node.module)

        for mod in imported_modules:
            for forbidden in forbidden_exact:
                assert not (
                    mod == forbidden or mod.startswith(forbidden + ".")
                ), f"Forbidden import found in targeting.py: {mod}"

            mod_lower = mod.lower()
            for sub in forbidden_substrings:
                assert (
                    sub not in mod_lower
                ), f"Forbidden LLM/cloud module substring {sub!r} found in import: {mod}"
