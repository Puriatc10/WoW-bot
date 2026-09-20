"""
Unit tests for wow_bot.combat.rotation table.
"""

import ast
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from wow_bot.combat.rotation import (
    CombatStateLike,
    Condition,
    RotationConfig,
    RotationError,
    RotationRule,
    RotationTable,
    load_rotation_from_dict,
)


@dataclass
class FakeCombatState:
    target_in_range: bool = True
    target_hp_percent: float = 100.0
    self_hp_percent: float = 100.0
    resource: float = 100.0
    resource_max: float = 100.0
    gcd_ready: bool = True
    debuffs: set[str] = field(default_factory=set)
    ready_spells: set[str] = field(default_factory=set)

    def target_has_debuff(self, debuff_id: str) -> bool:
        return debuff_id in self.debuffs

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        return spell_id in self.ready_spells


# Check interface compliance
_state_check: CombatStateLike = FakeCombatState()


def test_condition_unknown_kind_raises() -> None:
    with pytest.raises(ValueError, match="Invalid condition kind"):
        Condition(kind="unknown_kind", value=True)


def test_condition_numeric_kind_non_numeric_value_raises() -> None:
    with pytest.raises(ValueError, match="requires int or float value"):
        Condition(kind="target_hp_below", value="fifty")

    # bool is not considered numeric for condition values
    with pytest.raises(ValueError, match="requires int or float value"):
        Condition(kind="target_hp_below", value=True)


def test_condition_target_hp_below_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="requires value in \\[0.0, 100.0\\]"):
        Condition(kind="target_hp_below", value=-1.0)

    with pytest.raises(ValueError, match="requires value in \\[0.0, 100.0\\]"):
        Condition(kind="target_hp_below", value=100.1)


def test_condition_spell_ready_missing_spell_id_raises() -> None:
    with pytest.raises(ValueError, match="requires non-empty str spell_id"):
        Condition(kind="spell_ready", value=True, spell_id=None)

    with pytest.raises(ValueError, match="requires non-empty str spell_id"):
        Condition(kind="spell_ready", value=True, spell_id="")


def test_condition_spell_ready_non_bool_value_raises() -> None:
    with pytest.raises(ValueError, match="requires bool value"):
        Condition(kind="spell_ready", value="True", spell_id="fireball")


def test_condition_non_spell_ready_with_spell_id_raises() -> None:
    with pytest.raises(ValueError, match="must have spell_id=None"):
        Condition(kind="target_in_range", value=True, spell_id="fireball")


def test_condition_target_has_debuff_empty_value_raises() -> None:
    with pytest.raises(ValueError, match="requires non-empty str value"):
        Condition(kind="target_has_debuff", value="")


def test_rule_empty_spell_id_raises() -> None:
    cond = Condition(kind="target_in_range", value=True)
    with pytest.raises(ValueError, match="spell_id must be a non-empty string"):
        RotationRule(spell_id="", conditions=(cond,))


def test_rule_empty_conditions_raises() -> None:
    with pytest.raises(ValueError, match="conditions must be a non-empty tuple"):
        RotationRule(spell_id="fireball", conditions=())


def test_rule_negative_priority_raises() -> None:
    cond = Condition(kind="target_in_range", value=True)
    with pytest.raises(ValueError, match="priority must be an integer >= 0"):
        RotationRule(spell_id="fireball", conditions=(cond,), priority=-1)


def test_config_rules_list_instead_of_tuple_raises() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    with pytest.raises(ValueError, match="rules must be a tuple"):
        RotationConfig(rules=[rule])  # type: ignore[arg-type]


def test_config_duplicate_priority_spell_id_pair_raises() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule1 = RotationRule(spell_id="fireball", conditions=(cond,), priority=5)
    rule2 = RotationRule(spell_id="fireball", conditions=(cond,), priority=5)
    with pytest.raises(ValueError, match="Duplicate \\(priority, spell_id\\) pair"):
        RotationConfig(rules=(rule1, rule2))


def test_config_empty_default_spell_id_raises() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    with pytest.raises(
        ValueError, match="default_spell_id must be a non-empty string or None"
    ):
        RotationConfig(rules=(rule,), default_spell_id="")


def test_table_select_returns_first_matching_rule() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    rule1 = RotationRule(spell_id="fireball", conditions=(cond1,), priority=1)
    rule2 = RotationRule(spell_id="frostbolt", conditions=(cond1,), priority=2)

    table = RotationTable(RotationConfig(rules=(rule1, rule2)))
    state = FakeCombatState(target_in_range=True)
    assert table.select(state) == "fireball"


def test_table_select_returns_default_spell_when_no_match() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    table = RotationTable(
        RotationConfig(rules=(rule,), default_spell_id="auto_attack")
    )
    state = FakeCombatState(target_in_range=False)
    assert table.select(state) == "auto_attack"


def test_table_select_returns_none_when_no_match_and_no_default() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,), default_spell_id=None))
    state = FakeCombatState(target_in_range=False)
    assert table.select(state) is None


def test_table_select_empty_rules_returns_default_or_none() -> None:
    table_def = RotationTable(
        RotationConfig(rules=(), default_spell_id="auto_attack")
    )
    table_none = RotationTable(RotationConfig(rules=(), default_spell_id=None))
    state = FakeCombatState()
    assert table_def.select(state) == "auto_attack"
    assert table_none.select(state) is None


def test_higher_priority_wins() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule_low_prio = RotationRule(
        spell_id="frostbolt", conditions=(cond,), priority=10
    )
    rule_high_prio = RotationRule(
        spell_id="fireball", conditions=(cond,), priority=1
    )

    # Order in tuple is frostbolt first, but fireball has higher priority (1 < 10)
    table = RotationTable(
        RotationConfig(rules=(rule_low_prio, rule_high_prio))
    )
    state = FakeCombatState(target_in_range=True)
    assert table.select(state) == "fireball"


def test_equal_priority_ties_broken_by_original_order() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule1 = RotationRule(spell_id="fireball", conditions=(cond,), priority=5)
    rule2 = RotationRule(spell_id="frostbolt", conditions=(cond,), priority=5)

    table = RotationTable(RotationConfig(rules=(rule1, rule2)))
    state = FakeCombatState(target_in_range=True)
    assert table.select(state) == "fireball"


def test_rule_requires_all_conditions_true() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    cond2 = Condition(kind="gcd_ready", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond1, cond2))

    table = RotationTable(RotationConfig(rules=(rule,)))

    state_both = FakeCombatState(target_in_range=True, gcd_ready=True)
    assert table.select(state_both) == "fireball"

    state_one_false = FakeCombatState(target_in_range=True, gcd_ready=False)
    assert table.select(state_one_false) is None


def test_rule_skipped_if_single_condition_false() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    cond2 = Condition(kind="self_hp_below", value=50.0)
    rule = RotationRule(spell_id="heal", conditions=(cond1, cond2))

    table = RotationTable(RotationConfig(rules=(rule,)))
    state = FakeCombatState(target_in_range=True, self_hp_percent=80.0)
    assert table.select(state) is None


def test_target_hp_below_strict_less_than() -> None:
    cond = Condition(kind="target_hp_below", value=50.0)
    rule = RotationRule(spell_id="execute", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_49 = FakeCombatState(target_hp_percent=49.9)
    assert table.select(state_49) == "execute"

    state_50 = FakeCombatState(target_hp_percent=50.0)
    assert table.select(state_50) is None


def test_target_hp_above_strict_greater_than() -> None:
    cond = Condition(kind="target_hp_above", value=80.0)
    rule = RotationRule(spell_id="snipe", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_81 = FakeCombatState(target_hp_percent=80.1)
    assert table.select(state_81) == "snipe"

    state_80 = FakeCombatState(target_hp_percent=80.0)
    assert table.select(state_80) is None


def test_self_hp_below_strict_less_than() -> None:
    cond = Condition(kind="self_hp_below", value=30.0)
    rule = RotationRule(spell_id="heal", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_29 = FakeCombatState(self_hp_percent=29.9)
    assert table.select(state_29) == "heal"

    state_30 = FakeCombatState(self_hp_percent=30.0)
    assert table.select(state_30) is None


def test_self_hp_above_strict_greater_than() -> None:
    cond = Condition(kind="self_hp_above", value=70.0)
    rule = RotationRule(spell_id="overpower", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_71 = FakeCombatState(self_hp_percent=70.1)
    assert table.select(state_71) == "overpower"

    state_70 = FakeCombatState(self_hp_percent=70.0)
    assert table.select(state_70) is None


def test_resource_at_least_greater_or_equal() -> None:
    cond = Condition(kind="resource_at_least", value=30.0)
    rule = RotationRule(spell_id="mortal_strike", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_30 = FakeCombatState(resource=30.0)
    assert table.select(state_30) == "mortal_strike"

    state_29 = FakeCombatState(resource=29.9)
    assert table.select(state_29) is None


def test_resource_below_strict_less_than() -> None:
    cond = Condition(kind="resource_below", value=20.0)
    rule = RotationRule(spell_id="life_tap", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_19 = FakeCombatState(resource=19.9)
    assert table.select(state_19) == "life_tap"

    state_20 = FakeCombatState(resource=20.0)
    assert table.select(state_20) is None


def test_spell_ready_calls_state_callable() -> None:
    cond = Condition(kind="spell_ready", value=True, spell_id="fireball")
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_ready = FakeCombatState(ready_spells={"fireball"})
    assert table.select(state_ready) == "fireball"

    state_not_ready = FakeCombatState(ready_spells=set())
    assert table.select(state_not_ready) is None


def test_target_has_debuff_calls_state_callable() -> None:
    cond = Condition(kind="target_has_debuff", value="shadow_word_pain")
    rule = RotationRule(spell_id="mind_blast", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_has_debuff = FakeCombatState(debuffs={"shadow_word_pain"})
    assert table.select(state_has_debuff) == "mind_blast"

    state_no_debuff = FakeCombatState(debuffs=set())
    assert table.select(state_no_debuff) is None


def test_target_lacks_debuff_semantics() -> None:
    cond = Condition(kind="target_lacks_debuff", value="shadow_word_pain")
    rule = RotationRule(
        spell_id="shadow_word_pain", conditions=(cond,)
    )
    table = RotationTable(RotationConfig(rules=(rule,)))

    state_no_debuff = FakeCombatState(debuffs=set())
    assert table.select(state_no_debuff) == "shadow_word_pain"

    state_has_debuff = FakeCombatState(debuffs={"shadow_word_pain"})
    assert table.select(state_has_debuff) is None


def test_exceptions_from_spell_cooldown_ready_propagate() -> None:
    cond = Condition(kind="spell_ready", value=True, spell_id="fireball")
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state = FakeCombatState()

    def bad_callable(spell_id: str) -> bool:
        raise RuntimeError("Cooldown system error")

    state.spell_cooldown_ready = bad_callable  # type: ignore[method-assign]

    with pytest.raises(RuntimeError, match="Cooldown system error"):
        table.select(state)


def test_exceptions_from_target_has_debuff_propagate() -> None:
    cond = Condition(kind="target_has_debuff", value="curse")
    rule = RotationRule(spell_id="curse", conditions=(cond,))
    table = RotationTable(RotationConfig(rules=(rule,)))

    state = FakeCombatState()

    def bad_debuff(debuff_id: str) -> bool:
        raise KeyError("Debuff check failed")

    state.target_has_debuff = bad_debuff  # type: ignore[method-assign]

    with pytest.raises(KeyError, match="Debuff check failed"):
        table.select(state)


def test_select_does_not_mutate_state_or_config() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule = RotationRule(spell_id="fireball", conditions=(cond,))
    config = RotationConfig(rules=(rule,), default_spell_id="auto_attack")
    table = RotationTable(config)

    state = FakeCombatState(
        target_in_range=True,
        target_hp_percent=100.0,
        self_hp_percent=100.0,
        resource=100.0,
        resource_max=100.0,
        gcd_ready=True,
        debuffs={"curse"},
        ready_spells={"fireball"},
    )

    state_snap = (
        state.target_in_range,
        state.target_hp_percent,
        state.self_hp_percent,
        state.resource,
        state.resource_max,
        state.gcd_ready,
        set(state.debuffs),
        set(state.ready_spells),
    )

    config_snap = (config.rules, config.default_spell_id)

    table.select(state)

    state_after = (
        state.target_in_range,
        state.target_hp_percent,
        state.self_hp_percent,
        state.resource,
        state.resource_max,
        state.gcd_ready,
        set(state.debuffs),
        set(state.ready_spells),
    )

    config_after = (config.rules, config.default_spell_id)

    assert state_snap == state_after
    assert config_snap == config_after


def test_explain_returns_one_entry_per_rule_in_presorted_order() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    cond2 = Condition(kind="self_hp_below", value=50.0)

    rule1 = RotationRule(
        spell_id="frostbolt", conditions=(cond1,), priority=10
    )
    rule2 = RotationRule(spell_id="heal", conditions=(cond2,), priority=1)

    table = RotationTable(RotationConfig(rules=(rule1, rule2)))
    state = FakeCombatState(target_in_range=True, self_hp_percent=80.0)

    explanation = table.explain(state)
    # Pre-sorted order: heal (prio 1), frostbolt (prio 10)
    assert explanation == [("heal", False), ("frostbolt", True)]


def test_explain_reports_matched_true_when_all_conditions_true() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    cond2 = Condition(kind="gcd_ready", value=True)

    rule1 = RotationRule(spell_id="fireball", conditions=(cond1, cond2))
    rule2 = RotationRule(spell_id="frostbolt", conditions=(cond1,))

    table = RotationTable(RotationConfig(rules=(rule1, rule2)))
    state = FakeCombatState(target_in_range=True, gcd_ready=True)

    explanation = table.explain(state)
    assert explanation == [("fireball", True), ("frostbolt", True)]


def test_explain_does_not_short_circuit() -> None:
    cond = Condition(kind="target_in_range", value=True)

    rule1 = RotationRule(spell_id="fireball", conditions=(cond,), priority=1)
    rule2 = RotationRule(spell_id="frostbolt", conditions=(cond,), priority=2)

    table = RotationTable(RotationConfig(rules=(rule1, rule2)))
    state = FakeCombatState(target_in_range=True)

    explanation = table.explain(state)
    # Even though rule1 matches, rule2 is still evaluated in explain
    assert len(explanation) == 2
    assert explanation == [("fireball", True), ("frostbolt", True)]


def test_rule_order_and_rule_count() -> None:
    cond = Condition(kind="target_in_range", value=True)
    rule_low = RotationRule(
        spell_id="frostbolt", conditions=(cond,), priority=10
    )
    rule_high = RotationRule(
        spell_id="fireball", conditions=(cond,), priority=1
    )

    table = RotationTable(RotationConfig(rules=(rule_low, rule_high)))
    assert table.rule_count == 2
    assert table.rule_order == ("fireball", "frostbolt")


def test_load_rotation_from_dict_valid() -> None:
    data = {
        "rules": [
            {
                "spell_id": "fireball",
                "priority": 1,
                "conditions": [
                    {"kind": "target_in_range", "value": True},
                    {"kind": "gcd_ready", "value": True},
                ],
            }
        ],
        "default_spell_id": "auto_attack",
    }

    config = load_rotation_from_dict(data)
    assert isinstance(config, RotationConfig)
    assert config.default_spell_id == "auto_attack"
    assert len(config.rules) == 1
    assert config.rules[0].spell_id == "fireball"


def test_load_rotation_from_dict_unknown_top_level_key_raises() -> None:
    data = {"rules": [], "unknown_key": 123}
    with pytest.raises(RotationError, match="Unknown top-level key"):
        load_rotation_from_dict(data)


def test_load_rotation_from_dict_missing_rules_key_raises() -> None:
    data = {"default_spell_id": "auto_attack"}
    with pytest.raises(RotationError, match="Missing required key 'rules'"):
        load_rotation_from_dict(data)


def test_load_rotation_from_dict_unknown_key_inside_rule_raises() -> None:
    data = {
        "rules": [
            {
                "spell_id": "fireball",
                "conditions": [{"kind": "gcd_ready", "value": True}],
                "extra_param": True,
            }
        ]
    }
    with pytest.raises(RotationError, match="Unknown key"):
        load_rotation_from_dict(data)


def test_load_rotation_from_dict_unknown_key_inside_condition_raises() -> None:
    data = {
        "rules": [
            {
                "spell_id": "fireball",
                "conditions": [
                    {
                        "kind": "gcd_ready",
                        "value": True,
                        "unknown_cond_param": "foo",
                    }
                ],
            }
        ]
    }
    with pytest.raises(RotationError, match="Unknown key"):
        load_rotation_from_dict(data)


def test_load_rotation_from_dict_type_mismatch_raises() -> None:
    # Priority is str instead of int
    data1 = {
        "rules": [
            {
                "spell_id": "fireball",
                "priority": "one",
                "conditions": [{"kind": "gcd_ready", "value": True}],
            }
        ]
    }
    with pytest.raises(RotationError, match="priority"):
        load_rotation_from_dict(data1)

    # Kind value invalid for percent condition
    data2 = {
        "rules": [
            {
                "spell_id": "fireball",
                "conditions": [{"kind": "target_hp_below", "value": 150.0}],
            }
        ]
    }
    with pytest.raises(RotationError, match="Invalid condition"):
        load_rotation_from_dict(data2)


def test_load_rotation_from_dict_without_default_spell_id() -> None:
    data: dict[str, Any] = {
        "rules": [
            {
                "spell_id": "fireball",
                "conditions": [{"kind": "gcd_ready", "value": True}],
            }
        ]
    }
    config = load_rotation_from_dict(data)
    assert config.default_spell_id is None


def test_determinism_100_calls() -> None:
    cond1 = Condition(kind="target_in_range", value=True)
    cond2 = Condition(kind="gcd_ready", value=True)
    rule1 = RotationRule(spell_id="fireball", conditions=(cond1, cond2))
    config = RotationConfig(rules=(rule1,), default_spell_id="auto_attack")
    table = RotationTable(config)

    state = FakeCombatState(target_in_range=True, gcd_ready=True)

    first_result = table.select(state)
    for _ in range(100):
        assert table.select(state) == first_result


def test_static_ast_check() -> None:
    file_path = Path("src/wow_bot/combat/rotation.py")
    tree = ast.parse(file_path.read_text("utf-8"))

    prohibited_exact = {
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

    prohibited_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert (
                    mod not in prohibited_exact
                ), f"Prohibited module imported: {mod}"
                assert not any(
                    kw in mod.lower() for kw in prohibited_substrings
                ), f"Prohibited LLM module imported: {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert (
                mod not in prohibited_exact
            ), f"Prohibited module imported: {mod}"
            assert not any(
                kw in mod.lower() for kw in prohibited_substrings
            ), f"Prohibited LLM module imported: {mod}"
