"""
Unit tests for WoW-bot combat loop behavior.
"""

import ast
import copy
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import pytest

from wow_bot.actuation.mapper import MoveTo
from wow_bot.combat.loop import (
    CombatLoop,
    CombatLoopConfig,
    CombatLoopError,
    CombatMetaView,
    CombatStateView,
)
from wow_bot.combat.rotation import (
    Condition,
    RotationConfig,
    RotationRule,
    RotationTable,
)
from wow_bot.combat.targeting import TargetConfig, TargetSelector
from wow_bot.executor.fsm_v2 import FSM, Behavior, FSMConfig
from wow_bot.executor.states import FSMState


@dataclass(frozen=True)
class FakeEntity:
    """Fake entity test double implementing TargetEntityLike."""

    entity_id: str
    distance: float = 5.0
    threat: float = 100.0
    hp_percent: float = 100.0
    is_attackable: bool = True
    is_alive: bool = True
    is_in_combat_with_self: bool = True


@dataclass
class FakeCombatState:
    """Fake state test double satisfying CombatStateView, CombatStateLike, and TargetingStateLike."""

    current_target_id: str | None = None
    entities: tuple[Any, ...] = ()
    target_in_range: bool = True
    target_hp_percent: float = 100.0
    self_hp_percent: float = 100.0
    resource: float = 100.0
    resource_max: float = 100.0
    gcd_ready: bool = True
    self_x: float = 10.0
    self_y: float = 20.0
    target_x: float | None = 12.0
    target_y: float | None = 20.0
    debuffs: tuple[str, ...] = ()
    ready_spells: set[str] = field(default_factory=set)

    def target_has_debuff(self, debuff_id: str, /) -> bool:
        return debuff_id in self.debuffs

    def spell_cooldown_ready(self, spell_id: str, /) -> bool:
        return spell_id in self.ready_spells


@dataclass
class FakeCombatMeta:
    """Fake meta state test double implementing CombatMetaView."""

    combat_lock: bool = False


def _make_loop(
    *,
    rotation_rules: tuple[RotationRule, ...] = (),
    default_spell_id: str | None = None,
    target_config: TargetConfig | None = None,
    loop_config: CombatLoopConfig | None = None,
) -> CombatLoop:
    r_cfg = RotationConfig(rules=rotation_rules, default_spell_id=default_spell_id)
    r_tab = RotationTable(r_cfg)
    t_cfg = target_config if target_config is not None else TargetConfig(require_in_combat=False)
    t_sel = TargetSelector(t_cfg)
    return CombatLoop(r_tab, t_sel, config=loop_config)


def test_protocol_and_exception_definitions() -> None:
    """Verify CombatLoopError is an Exception and test doubles satisfy Protocol views."""
    assert issubclass(CombatLoopError, Exception)
    state = FakeCombatState()
    meta = FakeCombatMeta()
    assert isinstance(state, CombatStateView)
    assert isinstance(meta, CombatMetaView)


def test_combat_loop_config_validation() -> None:
    """Verify CombatLoopConfig parameter validation bounds."""
    # engage_distance_units > 0
    with pytest.raises(ValueError, match="engage_distance_units"):
        CombatLoopConfig(engage_distance_units=0.0)
    with pytest.raises(ValueError, match="engage_distance_units"):
        CombatLoopConfig(engage_distance_units=-5.0)

    # approach_target_distance >= 0
    with pytest.raises(ValueError, match="approach_target_distance"):
        CombatLoopConfig(approach_target_distance=-1.0)

    # approach_target_distance < engage_distance_units
    with pytest.raises(ValueError, match="approach_target_distance"):
        CombatLoopConfig(engage_distance_units=10.0, approach_target_distance=10.0)
    with pytest.raises(ValueError, match="approach_target_distance"):
        CombatLoopConfig(engage_distance_units=10.0, approach_target_distance=15.0)

    # attackable_spell_id non-empty
    with pytest.raises(ValueError, match="attackable_spell_id"):
        CombatLoopConfig(attackable_spell_id="")


def test_decide_wrong_state() -> None:
    """Verify decide returns None for every non-COMBAT state and sets last_reason='wrong_state'."""
    loop = _make_loop()
    state = FakeCombatState()
    meta = FakeCombatMeta()
    rng = random.Random(42)

    for st in FSMState:
        if st == FSMState.COMBAT:
            continue
        intent = loop.decide(st, state, meta, 1.0, rng)
        assert intent is None
        assert loop.last_reason() == "wrong_state"
        assert loop.last_spell_id() is None


def test_decide_no_target_no_candidate() -> None:
    """Verify decide returns None when current_target_id is None and targeting finds no candidate."""
    loop = _make_loop()
    state = FakeCombatState(current_target_id=None, entities=())
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "no_target"
    assert loop.last_spell_id() is None


def test_decide_retargeting_move_to_known_position() -> None:
    """Verify decide returns MoveTo toward target when no current target but candidate selected with known coords."""
    loop = _make_loop()
    entity = FakeEntity(entity_id="mob_1", distance=15.0)
    state = FakeCombatState(
        current_target_id=None,
        entities=(entity,),
        target_x=25.0,
        target_y=35.0,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert isinstance(intent, MoveTo)
    assert intent.x == 25.0
    assert intent.y == 35.0
    assert loop.last_reason() == "out_of_range_move"
    assert loop.last_spell_id() is None


def test_decide_retargeting_unknown_coordinates() -> None:
    """Verify decide returns None when candidate is selected but target_x or target_y is None."""
    loop = _make_loop()
    entity = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(
        current_target_id=None,
        entities=(entity,),
        target_x=None,
        target_y=20.0,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "no_target"
    assert loop.last_spell_id() is None


def test_decide_current_target_missing() -> None:
    """Verify decide returns None when current_target_id is missing from entities."""
    loop = _make_loop()
    state = FakeCombatState(current_target_id="mob_1", entities=())
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "target_gone"
    assert loop.last_spell_id() is None


def test_decide_current_target_dead() -> None:
    """Verify decide returns None when current_target_id is dead."""
    loop = _make_loop()
    dead_mob = FakeEntity(entity_id="mob_1", is_alive=False)
    state = FakeCombatState(current_target_id="mob_1", entities=(dead_mob,))
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "target_gone"
    assert loop.last_spell_id() is None


def test_decide_out_of_range_move() -> None:
    """Verify decide returns MoveTo toward target when distance > engage_distance_units."""
    cfg = CombatLoopConfig(engage_distance_units=20.0)
    loop = _make_loop(loop_config=cfg)
    mob = FakeEntity(entity_id="mob_1", distance=25.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_x=50.0,
        target_y=60.0,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert isinstance(intent, MoveTo)
    assert intent.x == 50.0
    assert intent.y == 60.0
    assert loop.last_reason() == "out_of_range_move"
    assert loop.last_spell_id() is None


def test_decide_stand_and_cast_rotation_spell() -> None:
    """Verify decide returns stand and cast MoveTo when in range and rotation selects spell."""
    rule = RotationRule(
        spell_id="fireball",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1", distance=10.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        gcd_ready=True,
        self_x=12.5,
        self_y=14.5,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert isinstance(intent, MoveTo)
    assert intent.x == 12.5
    assert intent.y == 14.5
    assert loop.last_reason() == "cast"
    assert loop.last_spell_id() == "fireball"


def test_decide_auto_attack_when_gcd_ready() -> None:
    """Verify decide returns auto_attack stand-and-cast when rotation returns None, gcd is ready, and target in range."""
    loop = _make_loop()
    mob = FakeEntity(entity_id="mob_1", distance=5.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=True,
        gcd_ready=True,
        self_x=3.0,
        self_y=4.0,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert isinstance(intent, MoveTo)
    assert intent.x == 3.0
    assert intent.y == 4.0
    assert loop.last_reason() == "auto_attack"
    assert loop.last_spell_id() == "auto_attack"


def test_decide_waiting_gcd() -> None:
    """Verify decide returns None when rotation returns None, gcd is False, and wait_for_gcd_after_cast is True."""
    cfg = CombatLoopConfig(wait_for_gcd_after_cast=True)
    loop = _make_loop(loop_config=cfg)
    mob = FakeEntity(entity_id="mob_1", distance=5.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=True,
        gcd_ready=False,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "waiting_gcd"
    assert loop.last_spell_id() is None


def test_decide_auto_attack_when_wait_for_gcd_false() -> None:
    """Verify decide returns auto_attack when gcd is False, wait_for_gcd_after_cast is False, and target_in_range is True."""
    cfg = CombatLoopConfig(wait_for_gcd_after_cast=False)
    loop = _make_loop(loop_config=cfg)
    mob = FakeEntity(entity_id="mob_1", distance=5.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=True,
        gcd_ready=False,
        self_x=8.0,
        self_y=9.0,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert isinstance(intent, MoveTo)
    assert intent.x == 8.0
    assert intent.y == 9.0
    assert loop.last_reason() == "auto_attack"
    assert loop.last_spell_id() == "auto_attack"


def test_decide_nothing_to_do() -> None:
    """Verify decide returns None with last_reason='nothing_to_do' when rotation returns None, not waiting GCD, and target_in_range is False."""
    cfg = CombatLoopConfig(wait_for_gcd_after_cast=False)
    loop = _make_loop(loop_config=cfg)
    mob = FakeEntity(entity_id="mob_1", distance=10.0)
    state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=False,
        gcd_ready=True,
    )
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert intent is None
    assert loop.last_reason() == "nothing_to_do"
    assert loop.last_spell_id() is None


def test_last_reason_and_spell_id_reflection() -> None:
    """Verify last_reason and last_spell_id reflect all 8 decision paths."""
    rule = RotationRule(
        spell_id="frostbolt",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1", distance=10.0)
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,), gcd_ready=True)
    meta = FakeCombatMeta()
    rng = random.Random(42)

    # 1. cast
    loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    assert loop.last_reason() == "cast"
    assert loop.last_spell_id() == "frostbolt"

    # 2. wrong_state
    loop.decide(FSMState.IDLE, state, meta, 1.0, rng)
    assert loop.last_reason() == "wrong_state"
    assert loop.last_spell_id() is None

    # 3. no_target
    empty_state = FakeCombatState(current_target_id=None, entities=())
    loop.decide(FSMState.COMBAT, empty_state, meta, 1.0, rng)
    assert loop.last_reason() == "no_target"
    assert loop.last_spell_id() is None

    # 4. target_gone
    gone_state = FakeCombatState(current_target_id="mob_2", entities=(mob,))
    loop.decide(FSMState.COMBAT, gone_state, meta, 1.0, rng)
    assert loop.last_reason() == "target_gone"
    assert loop.last_spell_id() is None

    # 5. out_of_range_move
    far_mob = FakeEntity(entity_id="mob_1", distance=50.0)
    far_state = FakeCombatState(
        current_target_id="mob_1",
        entities=(far_mob,),
        target_x=100.0,
        target_y=100.0,
    )
    loop.decide(FSMState.COMBAT, far_state, meta, 1.0, rng)
    assert loop.last_reason() == "out_of_range_move"
    assert loop.last_spell_id() is None

    # 6. waiting_gcd (with empty rotation)
    no_rule_loop = _make_loop()
    gcd_state = FakeCombatState(current_target_id="mob_1", entities=(mob,), gcd_ready=False)
    no_rule_loop.decide(FSMState.COMBAT, gcd_state, meta, 1.0, rng)
    assert no_rule_loop.last_reason() == "waiting_gcd"
    assert no_rule_loop.last_spell_id() is None

    # 7. auto_attack
    ready_state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=True,
        gcd_ready=True,
    )
    no_rule_loop.decide(FSMState.COMBAT, ready_state, meta, 1.0, rng)
    assert no_rule_loop.last_reason() == "auto_attack"
    assert no_rule_loop.last_spell_id() == "auto_attack"

    # 8. nothing_to_do
    no_range_state = FakeCombatState(
        current_target_id="mob_1",
        entities=(mob,),
        target_in_range=False,
        gcd_ready=True,
    )
    no_rule_loop.decide(FSMState.COMBAT, no_range_state, meta, 1.0, rng)
    assert no_rule_loop.last_reason() == "nothing_to_do"
    assert no_rule_loop.last_spell_id() is None


def test_decide_does_not_mutate_state() -> None:
    """Verify decide does not mutate game_state or meta_state."""
    rule = RotationRule(
        spell_id="smite",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,))
    meta = FakeCombatMeta(combat_lock=True)
    rng = random.Random(42)

    state_before = copy.deepcopy(state)
    meta_before = copy.deepcopy(meta)

    loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)

    assert state == state_before
    assert meta == meta_before


def test_decide_does_not_consume_rng() -> None:
    """Verify call decide 100 times does not consume or change rng state."""
    rule = RotationRule(
        spell_id="smite",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,))
    meta = FakeCombatMeta()
    rng = random.Random(12345)

    initial_rng_state = rng.getstate()

    for _ in range(100):
        loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)

    assert rng.getstate() == initial_rng_state


def test_decide_does_not_read_now() -> None:
    """Verify call decide with two different values of now produces identical results."""
    rule = RotationRule(
        spell_id="smite",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,))
    meta = FakeCombatMeta()
    rng = random.Random(42)

    intent1 = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
    reason1 = loop.last_reason()
    spell1 = loop.last_spell_id()

    intent2 = loop.decide(FSMState.COMBAT, state, meta, 999999.0, rng)
    reason2 = loop.last_reason()
    spell2 = loop.last_spell_id()

    assert intent1 == intent2
    assert reason1 == reason2
    assert spell1 == spell2


def test_determinism_across_100_calls() -> None:
    """Verify 100 calls with the same state return identical Intent."""
    rule = RotationRule(
        spell_id="fireball",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,))
    meta = FakeCombatMeta()
    rng = random.Random(42)

    first_intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)

    for _ in range(100):
        intent = loop.decide(FSMState.COMBAT, state, meta, 1.0, rng)
        assert intent == first_intent


def test_fsm_behavior_compatibility() -> None:
    """Verify CombatLoop is compatible with FSM's Behavior Protocol."""
    rule = RotationRule(
        spell_id="fireball",
        conditions=(Condition(kind="gcd_ready", value=True),),
    )
    loop = _make_loop(rotation_rules=(rule,))
    mob = FakeEntity(entity_id="mob_1")
    state = FakeCombatState(current_target_id="mob_1", entities=(mob,))
    meta = FakeCombatMeta()

    fsm = FSM(config=FSMConfig(seed=42), behavior=cast(Behavior, loop))

    # Force FSM to COMBAT state by submitting a transition or driving state internally
    # Let's inspect FSM: current_state can be transitioned or tested
    # We can test ticking when FSM is in COMBAT
    fsm._state = FSMState.COMBAT

    intent = fsm.tick(cast(Any, state), cast(Any, meta), now=10.0)
    assert isinstance(intent, MoveTo)
    assert intent.x == state.self_x
    assert intent.y == state.self_y


def test_static_ast_check_imports() -> None:
    """Verify loop.py does not import forbidden modules via static AST inspection."""
    loop_py_path = Path(__file__).parent.parent / "src" / "wow_bot" / "combat" / "loop.py"
    source = loop_py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(loop_py_path))

    forbidden_patterns = [
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation.actuator",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
        "ollama",
        "openai",
        "anthropic",
        "llm",
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                for pattern in forbidden_patterns:
                    assert pattern not in name, f"Forbidden import found: {name} contains {pattern}"
        elif isinstance(node, ast.ImportFrom):
            module = node.module or ""
            for pattern in forbidden_patterns:
                assert pattern not in module, f"Forbidden import from found: {module} contains {pattern}"


def test_static_ast_check_rng_and_now_no_attribute_access() -> None:
    """Verify inside decide method body there is no attribute access on name rng or now."""
    loop_py_path = Path(__file__).parent.parent / "src" / "wow_bot" / "combat" / "loop.py"
    source = loop_py_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(loop_py_path))

    decide_node: ast.FunctionDef | None = None

    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "CombatLoop":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decide":
                    decide_node = item
                    break

    assert decide_node is not None, "decide method not found in CombatLoop class"

    for child in ast.walk(decide_node):
        if isinstance(child, ast.Attribute) and isinstance(child.value, ast.Name):
            attr_obj_id = child.value.id
            assert attr_obj_id not in (
                "rng",
                "now",
            ), f"Attribute access on forbidden parameter '{attr_obj_id}.{child.attr}' in decide()"
