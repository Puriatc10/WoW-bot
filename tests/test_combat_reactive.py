"""
Tests for reactive combat (T6.4).
"""

import ast
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from wow_bot.actuation.mapper import MoveTo
from wow_bot.combat.reactive import (
    EnemyCastLike,
    ReactiveAction,
    ReactiveCombat,
    ReactiveCombatBehavior,
    ReactiveCombatConfig,
    ReactiveCombatSource,
    ReactiveDecision,
    ReactiveStateView,
)
from wow_bot.executor.states import FSMState


@dataclass(frozen=True)
class FakeCast:
    """Fake implementation of EnemyCastLike for testing."""

    caster_entity_id: str = "enemy_1"
    spell_id: str = "fireball"
    remaining_cast_time_s: float = 2.0
    is_interruptible: bool = True


@dataclass(frozen=True)
class FakeReactiveState:
    """Fake implementation of ReactiveStateView for testing."""

    self_hp_percent: float = 100.0
    self_in_combat: bool = True
    current_target_id: str | None = "target_1"
    target_in_range: bool = True
    incoming_casts: tuple[EnemyCastLike, ...] = ()
    cooldown_ready_fn: Callable[[str], bool] | None = None
    self_x: float = 10.0
    self_y: float = 20.0

    def spell_cooldown_ready(self, spell_id: str, /) -> bool:
        if self.cooldown_ready_fn is not None:
            return self.cooldown_ready_fn(spell_id)
        return True


# Acceptance Test 1: Empty spell id raises ValueError
def test_config_empty_spell_id() -> None:
    with pytest.raises(ValueError, match="Spell IDs must be non-empty strings"):
        ReactiveCombatConfig(interrupt_spell_id="")
    with pytest.raises(ValueError, match="Spell IDs must be non-empty strings"):
        ReactiveCombatConfig(defensive_spell_id="   ")


# Acceptance Test 2: Negative interrupt_min_remaining_s raises ValueError
def test_config_negative_interrupt_min_remaining() -> None:
    with pytest.raises(ValueError, match="interrupt_min_remaining_s must be >= 0.0"):
        ReactiveCombatConfig(interrupt_min_remaining_s=-0.1)


# Acceptance Test 3: interrupt_max_remaining_s <= interrupt_min_remaining_s raises ValueError
def test_config_invalid_interrupt_max_remaining() -> None:
    with pytest.raises(ValueError, match="interrupt_max_remaining_s must be > interrupt_min_remaining_s"):
        ReactiveCombatConfig(interrupt_min_remaining_s=1.0, interrupt_max_remaining_s=1.0)
    with pytest.raises(ValueError, match="interrupt_max_remaining_s must be > interrupt_min_remaining_s"):
        ReactiveCombatConfig(interrupt_min_remaining_s=1.0, interrupt_max_remaining_s=0.5)


# Acceptance Test 4: defensive_hp_threshold outside [0, 100] raises ValueError
def test_config_defensive_hp_threshold_bounds() -> None:
    with pytest.raises(ValueError, match="defensive_hp_threshold must be between 0.0 and 100.0"):
        ReactiveCombatConfig(defensive_hp_threshold=-1.0)
    with pytest.raises(ValueError, match="defensive_hp_threshold must be between 0.0 and 100.0"):
        ReactiveCombatConfig(defensive_hp_threshold=100.1)


# Acceptance Test 5: Negative defensive_cooldown_s raises ValueError
def test_config_negative_defensive_cooldown() -> None:
    with pytest.raises(ValueError, match="defensive_cooldown_s must be >= 0.0"):
        ReactiveCombatConfig(defensive_cooldown_s=-1.0)


# Acceptance Test 6: Negative retreat_distance_units raises ValueError
def test_config_negative_retreat_distance() -> None:
    with pytest.raises(ValueError, match="retreat_distance_units must be >= 0.0"):
        ReactiveCombatConfig(retreat_distance_units=-0.5)


# Acceptance Test 7: ReactiveDecision invariants
def test_decision_invariants() -> None:
    intent = MoveTo(x=0.0, y=0.0)
    with pytest.raises(ValueError, match="action == NONE implies spell_id is None and intent is None"):
        ReactiveDecision(ReactiveAction.NONE, "spell", None, reason="test")
    with pytest.raises(ValueError, match="action == NONE implies spell_id is None and intent is None"):
        ReactiveDecision(ReactiveAction.NONE, None, intent, reason="test")

    with pytest.raises(ValueError, match="action == interrupt implies spell_id is not None and intent is not None"):
        ReactiveDecision(ReactiveAction.INTERRUPT, None, intent, reason="test")
    with pytest.raises(ValueError, match="action == interrupt implies spell_id is not None and intent is not None"):
        ReactiveDecision(ReactiveAction.INTERRUPT, "kick", None, reason="test")

    with pytest.raises(ValueError, match="action == RETREAT implies spell_id is None and intent is not None"):
        ReactiveDecision(ReactiveAction.RETREAT, "shield", intent, reason="test")
    with pytest.raises(ValueError, match="action == RETREAT implies spell_id is None and intent is not None"):
        ReactiveDecision(ReactiveAction.RETREAT, None, None, reason="test")

    with pytest.raises(ValueError, match="reason must be a non-empty string"):
        ReactiveDecision(ReactiveAction.NONE, None, None, reason="")


# Acceptance Test 8: evaluate returns NONE with reason="not_in_combat" when self_in_combat is False
def test_evaluate_not_in_combat() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(self_in_combat=False)
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.spell_id is None
    assert decision.intent is None
    assert decision.reason == "not_in_combat"


# Acceptance Test 9: evaluate returns NONE with reason="not_in_combat" when current_target_id is None
def test_evaluate_no_target() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(current_target_id=None)
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.spell_id is None
    assert decision.intent is None
    assert decision.reason == "not_in_combat"


# Acceptance Test 10: RETREAT takes precedence over INTERRUPT and DEFENSIVE
def test_retreat_precedence() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(defensive_hp_threshold=35.0, retreat_distance_units=3.0))
    state = FakeReactiveState(
        self_hp_percent=30.0,
        incoming_casts=(FakeCast(remaining_cast_time_s=1.0, is_interruptible=True),),
        self_x=10.0,
        self_y=20.0,
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.RETREAT
    assert decision.spell_id is None
    assert decision.intent == MoveTo(x=7.0, y=20.0)
    assert decision.reason == "low_hp_retreat"


# Acceptance Test 11: RETREAT intent is a MoveTo at (self_x - retreat_distance, self_y)
def test_retreat_intent_coords() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(defensive_hp_threshold=50.0, retreat_distance_units=5.0))
    state = FakeReactiveState(self_hp_percent=40.0, self_x=100.0, self_y=50.0)
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.RETREAT
    assert decision.intent == MoveTo(x=95.0, y=50.0)


# Acceptance Test 12: RETREAT is returned even when defensive cooldown is ready
def test_retreat_returned_when_defensive_cooldown_ready() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(defensive_hp_threshold=35.0))
    state = FakeReactiveState(self_hp_percent=20.0, cooldown_ready_fn=lambda _s: True)
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.RETREAT


# Acceptance Test 13: INTERRUPT returned when low HP not triggered, cast in window, interrupt ready
def test_interrupt_success() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(interrupt_spell_id="counterspell"))
    state = FakeReactiveState(
        self_hp_percent=100.0,
        incoming_casts=(FakeCast(spell_id="fireball", remaining_cast_time_s=1.5, is_interruptible=True),),
        self_x=12.0,
        self_y=34.0,
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.INTERRUPT
    assert decision.spell_id == "counterspell"
    assert decision.intent == MoveTo(x=12.0, y=34.0)
    assert decision.reason == "interrupt:fireball"


# Acceptance Test 14: INTERRUPT NOT returned when cast remaining time < interrupt_min_remaining_s
def test_interrupt_below_min_remaining() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(interrupt_min_remaining_s=0.1))
    state = FakeReactiveState(
        incoming_casts=(FakeCast(remaining_cast_time_s=0.04, is_interruptible=True),),
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "nothing_to_do"


# Acceptance Test 15: INTERRUPT NOT returned when cast remaining time > interrupt_max_remaining_s
def test_interrupt_above_max_remaining() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(interrupt_max_remaining_s=3.0))
    state = FakeReactiveState(
        incoming_casts=(FakeCast(remaining_cast_time_s=4.0, is_interruptible=True),),
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "nothing_to_do"


# Acceptance Test 16: INTERRUPT NOT returned when cast is not interruptible
def test_interrupt_non_interruptible() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(
        incoming_casts=(FakeCast(remaining_cast_time_s=1.0, is_interruptible=False),),
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "nothing_to_do"


# Acceptance Test 17: INTERRUPT NOT returned when interrupt spell on cooldown; DEFENSIVE evaluated next
def test_interrupt_on_cooldown_falls_through_to_defensive() -> None:
    rc = ReactiveCombat(
        config=ReactiveCombatConfig(
            interrupt_spell_id="kick",
            defensive_spell_id="shield",
            defensive_hp_threshold=35.0,
        )
    )

    def cooldowns(spell: str) -> bool:
        if spell == "kick":
            return False
        if spell == "shield":
            return True
        return True

    state = FakeReactiveState(
        self_hp_percent=30.0,
        incoming_casts=(FakeCast(remaining_cast_time_s=1.0, is_interruptible=True),),
        cooldown_ready_fn=cooldowns,
    )
    # Note: When self_hp_percent is 30.0, step 2 (RETREAT) checks state.spell_cooldown_ready("shield").
    # Since shield is ready, RETREAT returns!
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.RETREAT

    # Now test low HP not triggering RETREAT (e.g. self_hp = 50.0), kick on cooldown, defensive low HP not triggered (50 > 35)
    state2 = FakeReactiveState(
        self_hp_percent=50.0,
        incoming_casts=(FakeCast(remaining_cast_time_s=1.0, is_interruptible=True),),
        cooldown_ready_fn=lambda s: s != "kick",
    )
    decision2 = rc.evaluate(state2)
    assert decision2.action == ReactiveAction.NONE
    assert decision2.reason == "nothing_to_do"

    # Now test low HP (30.0), but defensive spell is ON COOLDOWN. Step 2 (RETREAT) fails. Step 3 (INTERRUPT) fails (kick on cooldown). Step 4 (DEFENSIVE) evaluates and returns NONE with reason "defensive_on_cooldown".
    state3 = FakeReactiveState(
        self_hp_percent=30.0,
        incoming_casts=(FakeCast(remaining_cast_time_s=1.0, is_interruptible=True),),
        cooldown_ready_fn=lambda _s: False,
    )
    decision3 = rc.evaluate(state3)
    assert decision3.action == ReactiveAction.NONE
    assert decision3.reason == "defensive_on_cooldown"


# Acceptance Test 18: INTERRUPT picks FIRST matching cast in incoming_casts order
def test_interrupt_picks_first_matching_cast() -> None:
    rc = ReactiveCombat()
    cast1 = FakeCast(caster_entity_id="e1", spell_id="frostbolt", remaining_cast_time_s=2.0)
    cast2 = FakeCast(caster_entity_id="e2", spell_id="pyroblast", remaining_cast_time_s=1.5)
    state = FakeReactiveState(incoming_casts=(cast1, cast2))
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.INTERRUPT
    assert decision.reason == "interrupt:frostbolt"


# Acceptance Test 19: DEFENSIVE returned when low HP triggered and defensive spell ready (and RETREAT skipped because defensive cd ready triggers RETREAT first!)
# Note: In strict priority order, Step 2 (RETREAT) checks HP <= defensive_hp_threshold AND defensive spell ready -> returns RETREAT. Step 4 (DEFENSIVE) is reached when HP <= defensive_hp_threshold AND defensive spell is ready... wait, if defensive spell is ready, Step 2 RETREAT always handles it!
# Wait, let's verify Step 4 DEFENSIVE logic:
# Step 4 is reached if Step 2 RETREAT did not trigger or if defensive cooldown condition in Step 2 was evaluated.
# Wait! In step 2:
#   if state.self_hp_percent <= config.defensive_hp_threshold:
#       if state.spell_cooldown_ready(config.defensive_spell_id):
#           return RETREAT
# Then Step 3 INTERRUPT.
# Then Step 4 DEFENSIVE:
#   if state.self_hp_percent <= config.defensive_hp_threshold:
#       if state.spell_cooldown_ready(config.defensive_spell_id):
#           return DEFENSIVE
#       else:
#           return NONE(reason="defensive_on_cooldown")
# Note that if defensive spell is ready, Step 2 RETREAT will have already returned RETREAT.
# Thus, Step 4 DEFENSIVE with defensive spell ready is fallback logic (or if Step 2 RETREAT was somehow skipped/overridden, or if defensive spell was ready during Step 4). If defensive spell is not ready, Step 4 returns defensive_on_cooldown!
def test_defensive_branch_on_cooldown() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(defensive_hp_threshold=35.0))
    state = FakeReactiveState(
        self_hp_percent=30.0,
        cooldown_ready_fn=lambda _s: False,
    )
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "defensive_on_cooldown"


def test_defensive_action_directly() -> None:
    # Test DEFENSIVE action representation and decision validity
    intent = MoveTo(x=10.0, y=10.0)
    decision = ReactiveDecision(
        action=ReactiveAction.DEFENSIVE,
        spell_id="shield",
        intent=intent,
        reason="defensive_low_hp",
    )
    assert decision.action == ReactiveAction.DEFENSIVE
    assert decision.spell_id == "shield"
    assert decision.intent == intent


# Acceptance Test 20: DEFENSIVE returns NONE with reason="defensive_on_cooldown" when low HP triggered and defensive not ready
def test_defensive_on_cooldown_reason() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(defensive_hp_threshold=35.0))
    state = FakeReactiveState(self_hp_percent=25.0, cooldown_ready_fn=lambda _s: False)
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "defensive_on_cooldown"


# Acceptance Test 21: NONE with reason="nothing_to_do" when no branch matches
def test_nothing_to_do() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(self_hp_percent=100.0, incoming_casts=())
    decision = rc.evaluate(state)
    assert decision.action == ReactiveAction.NONE
    assert decision.reason == "nothing_to_do"


# Acceptance Test 22: to_signal returns None for ReactiveAction.NONE
def test_to_signal_none() -> None:
    rc = ReactiveCombat()
    decision = ReactiveDecision(ReactiveAction.NONE, None, None, reason="test")
    assert rc.to_signal(decision, ts=123.45) is None


# Acceptance Test 23: to_signal returns Signal for INTERRUPT
def test_to_signal_interrupt() -> None:
    rc = ReactiveCombat()
    intent = MoveTo(x=0.0, y=0.0)
    decision = ReactiveDecision(ReactiveAction.INTERRUPT, "kick", intent, reason="test")
    sig = rc.to_signal(decision, ts=100.0)
    assert sig is not None
    assert sig.name == "combat_interrupt"
    assert sig.payload == {"spell_id": "kick"}
    assert sig.ts == 100.0


# Acceptance Test 24: to_signal returns Signal for DEFENSIVE
def test_to_signal_defensive() -> None:
    rc = ReactiveCombat()
    intent = MoveTo(x=0.0, y=0.0)
    decision = ReactiveDecision(ReactiveAction.DEFENSIVE, "shield", intent, reason="test")
    sig = rc.to_signal(decision, ts=200.0)
    assert sig is not None
    assert sig.name == "combat_defensive"
    assert sig.payload == {"spell_id": "shield"}
    assert sig.ts == 200.0


# Acceptance Test 25: to_signal returns Signal for RETREAT
def test_to_signal_retreat() -> None:
    rc = ReactiveCombat(config=ReactiveCombatConfig(retreat_distance_units=4.5))
    intent = MoveTo(x=0.0, y=0.0)
    decision = ReactiveDecision(ReactiveAction.RETREAT, None, intent, reason="test")
    sig = rc.to_signal(decision, ts=300.0)
    assert sig is not None
    assert sig.name == "combat_retreat"
    assert sig.payload == {"distance": 4.5}
    assert sig.ts == 300.0


# Acceptance Test 26: to_signal sets Signal.ts verbatim
def test_to_signal_verbatim_ts() -> None:
    rc = ReactiveCombat()
    intent = MoveTo(x=0.0, y=0.0)
    decision = ReactiveDecision(ReactiveAction.INTERRUPT, "kick", intent, reason="test")
    sig = rc.to_signal(decision, ts=999.876)
    assert sig is not None
    assert sig.ts == 999.876


# Acceptance Test 27: ReactiveCombatSource.poll returns empty list when decision is NONE
def test_source_poll_empty_when_none() -> None:
    state = FakeReactiveState(self_in_combat=False)
    source = ReactiveCombatSource(state_source=lambda: state)
    signals = source.poll(now=50.0)
    assert signals == []
    assert source.last_decision is not None
    assert source.last_decision.action == ReactiveAction.NONE


# Acceptance Test 28: ReactiveCombatSource.poll returns single-element list with signal when decision not NONE
def test_source_poll_signal_when_not_none() -> None:
    state = FakeReactiveState(
        incoming_casts=(FakeCast(spell_id="fireball", remaining_cast_time_s=1.0),)
    )
    source = ReactiveCombatSource(state_source=lambda: state)
    signals = source.poll(now=75.0)
    assert len(signals) == 1
    assert signals[0].name == "combat_interrupt"
    assert signals[0].ts == 75.0
    assert source.last_decision is not None
    assert source.last_decision.action == ReactiveAction.INTERRUPT


# Acceptance Test 29: ReactiveCombatSource.poll propagates exceptions from state_source
def test_source_poll_propagates_exception() -> None:
    def raise_error() -> ReactiveStateView:
        raise RuntimeError("state failure")

    source = ReactiveCombatSource(state_source=raise_error)
    with pytest.raises(RuntimeError, match="state failure"):
        source.poll(now=10.0)


# Acceptance Test 30: ReactiveCombatBehavior.decide returns None for non-COMBAT FSM states
def test_behavior_decide_non_combat_state() -> None:
    behavior = ReactiveCombatBehavior()
    state = FakeReactiveState(
        incoming_casts=(FakeCast(spell_id="fireball", remaining_cast_time_s=1.0),)
    )
    intent = behavior.decide(FSMState.IDLE, state, None, now=1.0, rng=None)
    assert intent is None


# Acceptance Test 31: ReactiveCombatBehavior.decide returns decision's intent for COMBAT state when action matches
def test_behavior_decide_combat_state_matching_action() -> None:
    behavior = ReactiveCombatBehavior()
    state = FakeReactiveState(
        incoming_casts=(FakeCast(spell_id="fireball", remaining_cast_time_s=1.0),),
        self_x=5.0,
        self_y=15.0,
    )
    intent = behavior.decide(FSMState.COMBAT, state, None, now=1.0, rng=None)
    assert intent == MoveTo(x=5.0, y=15.0)
    assert behavior.last_decision is not None
    assert behavior.last_decision.action == ReactiveAction.INTERRUPT


# Acceptance Test 32: ReactiveCombatBehavior.decide returns None for COMBAT when decision is NONE
def test_behavior_decide_combat_state_none_decision() -> None:
    behavior = ReactiveCombatBehavior()
    state = FakeReactiveState(self_in_combat=False)
    intent = behavior.decide(FSMState.COMBAT, state, None, now=1.0, rng=None)
    assert intent is None
    assert behavior.last_decision is not None
    assert behavior.last_decision.action == ReactiveAction.NONE


# Acceptance Test 33: ReactiveCombatBehavior.decide does not consume rng or read now
def test_behavior_decide_rng_now_invariance() -> None:
    behavior = ReactiveCombatBehavior()
    state = FakeReactiveState(
        incoming_casts=(FakeCast(spell_id="fireball", remaining_cast_time_s=1.0),)
    )

    class DummyRNG:
        def __getattribute__(self, name: str) -> Any:
            raise AssertionError(f"rng attribute '{name}' was accessed!")

    dummy_rng = DummyRNG()

    res1 = behavior.decide(FSMState.COMBAT, state, None, now=10.0, rng=dummy_rng)
    res2 = behavior.decide(FSMState.COMBAT, state, None, now=2000.0, rng=dummy_rng)
    assert res1 == res2


# Acceptance Test 34: Determinism: 100 evaluate calls with same state return same ReactiveDecision
def test_evaluate_determinism() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(
        incoming_casts=(FakeCast(spell_id="frostbolt", remaining_cast_time_s=2.0),)
    )
    first_decision = rc.evaluate(state)
    for _ in range(100):
        assert rc.evaluate(state) == first_decision


# Acceptance Test 35: evaluate does not mutate state
def test_evaluate_state_immutability() -> None:
    rc = ReactiveCombat()
    state = FakeReactiveState(
        self_hp_percent=80.0,
        self_in_combat=True,
        current_target_id="target_100",
        target_in_range=True,
        incoming_casts=(FakeCast(spell_id="smite", remaining_cast_time_s=1.2),),
        self_x=50.0,
        self_y=60.0,
    )
    snapshot = (
        state.self_hp_percent,
        state.self_in_combat,
        state.current_target_id,
        state.target_in_range,
        state.incoming_casts,
        state.self_x,
        state.self_y,
    )
    _ = rc.evaluate(state)
    current_snapshot = (
        state.self_hp_percent,
        state.self_in_combat,
        state.current_target_id,
        state.target_in_range,
        state.incoming_casts,
        state.self_x,
        state.self_y,
    )
    assert snapshot == current_snapshot


# Acceptance Test 36: name_interrupt / name_defensive / name_retreat classmethods
def test_name_classmethods() -> None:
    assert ReactiveCombat.name_interrupt() == "combat_interrupt"
    assert ReactiveCombat.name_defensive() == "combat_defensive"
    assert ReactiveCombat.name_retreat() == "combat_retreat"


# Acceptance Test 37: Static AST check for prohibited imports
def test_static_ast_prohibited_imports() -> None:
    import pathlib

    filepath = pathlib.Path("src/wow_bot/combat/reactive.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    forbidden_modules = [
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.session",
        "aiosqlite",
        "asyncio",
        "threading",
    ]
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                if name == "wow_bot.actuation.actuator":
                    pytest.fail("Direct import of wow_bot.actuation.actuator is prohibited")
                for sub in forbidden_substrings:
                    if sub in name.lower():
                        pytest.fail(f"Prohibited import containing '{sub}': {name}")
                for fmod in forbidden_modules:
                    if name == fmod or name.startswith(fmod + "."):
                        pytest.fail(f"Prohibited module import: {name}")

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            if mod_name == "wow_bot.actuation.actuator":
                # Check imported names
                for alias in node.names:
                    if alias.name != "Actuator":
                        pytest.fail(f"Prohibited import from actuator: {alias.name}")
            else:
                for sub in forbidden_substrings:
                    if sub in mod_name.lower():
                        pytest.fail(f"Prohibited import from '{mod_name}'")
                for fmod in forbidden_modules:
                    if mod_name == fmod or mod_name.startswith(fmod + "."):
                        pytest.fail(f"Prohibited module import from {mod_name}")


# Acceptance Test 38: Static AST check for ReactiveCombatBehavior.decide body (no attribute access on rng or now)
def test_static_ast_decide_no_rng_or_now_access() -> None:
    import pathlib

    filepath = pathlib.Path("src/wow_bot/combat/reactive.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"))

    decide_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ReactiveCombatBehavior":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decide":
                    decide_node = item
                    break

    assert decide_node is not None, "ReactiveCombatBehavior.decide method not found"

    for child in ast.walk(decide_node):
        if (
            isinstance(child, ast.Attribute)
            and isinstance(child.value, ast.Name)
            and child.value.id in ("rng", "now")
        ):
            pytest.fail(
                f"Attribute access '{child.attr}' on parameter '{child.value.id}' in decide()"
            )
