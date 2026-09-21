"""Tests for loot and inventory management module (T11.2)."""

from __future__ import annotations

import ast
import random
from dataclasses import dataclass
from pathlib import Path

import pytest

from wow_bot.actuation.mapper import MoveTo
from wow_bot.executor.fsm_v2 import Behavior
from wow_bot.executor.states import FSMState
from wow_bot.farm.loot import (
    InventoryTracker,
    LootConfig,
    LootController,
    LootDecision,
    LootError,
    LootStatus,
)
from wow_bot.session import Session


@dataclass
class FakeLootState:
    """Fake state implementing LootStateView protocol for testing."""

    target_entity_id: str | None = "mob_1"
    target_is_alive: bool = False
    target_is_lootable: bool = True
    target_distance: float | None = 1.5
    self_x: float = 10.0
    self_y: float = 20.0
    target_x: float | None = 10.0
    target_y: float | None = 20.0
    inventory_count: int = 5
    inventory_max: int | None = 30


@dataclass(frozen=True)
class DummyConfig:
    """Dataclass config mock for Session.start tests."""

    session_root: Path
    lab_mode: bool = False
    dry_run: bool = True


# Acceptance Criteria Tests

def test_loot_config_reach_units_validation() -> None:
    """LootConfig with loot_reach_units <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="loot_reach_units must be > 0.0"):
        LootConfig(loot_reach_units=0.0)
    with pytest.raises(ValueError, match="loot_reach_units must be > 0.0"):
        LootConfig(loot_reach_units=-1.0)


def test_loot_config_max_attempts_validation() -> None:
    """LootConfig with max_attempts_per_corpse < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_attempts_per_corpse must be >= 1"):
        LootConfig(max_attempts_per_corpse=0)


def test_loot_config_inventory_full_absolute_validation() -> None:
    """LootConfig with inventory_full_absolute < 1 raises ValueError."""
    with pytest.raises(ValueError, match="inventory_full_absolute must be >= 1"):
        LootConfig(inventory_full_absolute=0)


def test_loot_config_inventory_clear_fraction_validation() -> None:
    """LootConfig with inventory_clear_fraction outside (0, 1) raises ValueError."""
    with pytest.raises(ValueError, match="inventory_clear_fraction must be in \\(0.0, 1.0\\)"):
        LootConfig(inventory_clear_fraction=0.0)
    with pytest.raises(ValueError, match="inventory_clear_fraction must be in \\(0.0, 1.0\\)"):
        LootConfig(inventory_clear_fraction=1.0)


def test_loot_config_non_finite_floats_validation() -> None:
    """LootConfig with non-finite floats raises ValueError."""
    with pytest.raises(ValueError, match="loot_reach_units must be > 0.0"):
        LootConfig(loot_reach_units=float("nan"))
    with pytest.raises(ValueError, match="loot_reach_units must be > 0.0"):
        LootConfig(loot_reach_units=float("inf"))
    with pytest.raises(ValueError, match="inventory_clear_fraction must be in \\(0.0, 1.0\\)"):
        LootConfig(inventory_clear_fraction=float("nan"))


def test_loot_decision_invariants() -> None:
    """LootDecision invariants: SUCCESS/OUT_OF_REACH require intent; others require None; empty reason raises."""
    # SUCCESS with None intent -> ValueError
    with pytest.raises(ValueError, match="status success requires a non-None intent"):
        LootDecision(status=LootStatus.SUCCESS, intent=None, reason="ok")

    # OUT_OF_REACH with None intent -> ValueError
    with pytest.raises(ValueError, match="status out_of_reach requires a non-None intent"):
        LootDecision(status=LootStatus.OUT_OF_REACH, intent=None, reason="ok")

    # NO_TARGET with non-None intent -> ValueError
    with pytest.raises(ValueError, match="status no_target requires intent to be None"):
        LootDecision(status=LootStatus.NO_TARGET, intent=MoveTo(x=0, y=0), reason="ok")

    # Empty reason -> ValueError
    with pytest.raises(ValueError, match="reason is a non-empty string"):
        LootDecision(status=LootStatus.NO_TARGET, intent=None, reason="")


def test_decide_non_looting_fsm_state_returns_none() -> None:
    """decide with state != LOOTING returns None for every non-LOOTING FSMState member."""
    controller = LootController()
    state = FakeLootState()
    rng = random.Random(42)

    for fsm_state in FSMState:
        if fsm_state == FSMState.LOOTING:
            continue
        res = controller.decide(fsm_state, state, {}, 100.0, rng)
        assert res is None


def test_decide_no_target(tmp_path: Path) -> None:
    """decide with target_entity_id None returns None, sets last_status=NO_TARGET, emits loot_skipped."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    controller = LootController(session=session)
    state = FakeLootState(target_entity_id=None)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.last_status() == LootStatus.NO_TARGET

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 1
    assert '"event": "loot_skipped"' in events[0]
    assert '"reason": "no_target"' in events[0]


def test_decide_target_alive() -> None:
    """decide with target_is_alive True returns None, sets last_status=TARGET_ALIVE."""
    controller = LootController()
    state = FakeLootState(target_is_alive=True)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.last_status() == LootStatus.TARGET_ALIVE


def test_decide_target_not_lootable() -> None:
    """decide with target_is_lootable False returns None, sets last_status=TARGET_NOT_LOOTABLE."""
    controller = LootController()
    state = FakeLootState(target_is_lootable=False)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.last_status() == LootStatus.TARGET_NOT_LOOTABLE


def test_decide_target_distance_none() -> None:
    """decide with target_distance None returns None, sets last_status=OUT_OF_REACH, does NOT emit loot_attempt."""
    controller = LootController()
    state = FakeLootState(target_distance=None)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.last_status() == LootStatus.OUT_OF_REACH


def test_decide_out_of_reach_with_coordinates(tmp_path: Path) -> None:
    """decide with distance > reach and known coordinates returns MoveTo(target_x, target_y), sets OUT_OF_REACH, emits loot_approach."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    controller = LootController(config=LootConfig(loot_reach_units=2.5), session=session)
    state = FakeLootState(target_distance=5.0, target_x=12.0, target_y=25.0)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert isinstance(res, MoveTo)
    assert res.x == 12.0
    assert res.y == 25.0
    assert controller.last_status() == LootStatus.OUT_OF_REACH

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 1
    assert '"event": "loot_approach"' in events[0]
    assert '"distance": 5.0' in events[0]


def test_decide_out_of_reach_without_coordinates() -> None:
    """decide with distance > reach and unknown coordinates returns None."""
    controller = LootController(config=LootConfig(loot_reach_units=2.5))
    state = FakeLootState(target_distance=5.0, target_x=None, target_y=None)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.last_status() == LootStatus.OUT_OF_REACH


def test_decide_in_reach_success(tmp_path: Path) -> None:
    """decide with target_distance <= reach returns MoveTo(self_x, self_y), sets SUCCESS, emits loot_attempt."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    controller = LootController(config=LootConfig(loot_reach_units=2.5), session=session)
    state = FakeLootState(target_distance=1.5, self_x=10.0, self_y=20.0)
    rng = random.Random(42)

    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert isinstance(res, MoveTo)
    assert res.x == 10.0
    assert res.y == 20.0
    assert controller.last_status() == LootStatus.SUCCESS

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 1
    assert '"event": "loot_attempt"' in events[0]
    assert '"attempts": 1' in events[0]


def test_decide_attempts_counter_and_exceeded(tmp_path: Path) -> None:
    """decide increments attempt counter on in-reach calls and returns ATTEMPTS_EXCEEDED when exceeded."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    config = LootConfig(max_attempts_per_corpse=3, loot_reach_units=2.5)
    controller = LootController(config=config, session=session)
    state = FakeLootState(target_entity_id="corpse_1", target_distance=1.0)
    rng = random.Random(42)

    for attempt in range(1, 4):
        res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
        assert isinstance(res, MoveTo)
        assert controller.attempts_for("corpse_1") == attempt
        assert controller.last_status() == LootStatus.SUCCESS

    # 4th attempt > max_attempts_per_corpse (3)
    res = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert res is None
    assert controller.attempts_for("corpse_1") == 4
    assert controller.last_status() == LootStatus.ATTEMPTS_EXCEEDED

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 4
    assert '"event": "loot_failed"' in events[3]
    assert '"attempts": 4' in events[3]


def test_decide_target_switch_resets_attempt_count() -> None:
    """Switching target_entity_id resets per-corpse attempt count for new entity."""
    controller = LootController(config=LootConfig(max_attempts_per_corpse=5))
    rng = random.Random(42)

    state1 = FakeLootState(target_entity_id="mob_1", target_distance=1.0)
    controller.decide(FSMState.LOOTING, state1, {}, 100.0, rng)
    controller.decide(FSMState.LOOTING, state1, {}, 100.0, rng)
    assert controller.attempts_for("mob_1") == 2

    state2 = FakeLootState(target_entity_id="mob_2", target_distance=1.0)
    controller.decide(FSMState.LOOTING, state2, {}, 100.0, rng)
    assert controller.attempts_for("mob_2") == 1


def test_decide_does_not_consume_rng() -> None:
    """decide does not consume rng: call 100 times with captured RNG state before/after."""
    controller = LootController()
    state = FakeLootState()
    rng = random.Random(12345)

    initial_state = rng.getstate()
    for _ in range(100):
        _ = controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)

    assert rng.getstate() == initial_state


def test_decide_does_not_read_now() -> None:
    """decide does not read now: call with two different values of now and same state returns identical result."""
    controller1 = LootController()
    controller2 = LootController()
    state = FakeLootState()
    rng1 = random.Random(42)
    rng2 = random.Random(42)

    res1 = controller1.decide(FSMState.LOOTING, state, {}, 100.0, rng1)
    res2 = controller2.decide(FSMState.LOOTING, state, {}, 2000.0, rng2)
    assert res1 == res2


def test_decide_does_not_mutate_state() -> None:
    """decide does not mutate state (snapshot comparison)."""
    controller = LootController()
    state = FakeLootState()
    snapshot = (
        state.target_entity_id,
        state.target_is_alive,
        state.target_is_lootable,
        state.target_distance,
        state.self_x,
        state.self_y,
        state.target_x,
        state.target_y,
        state.inventory_count,
        state.inventory_max,
    )

    controller.decide(FSMState.LOOTING, state, {}, 100.0, random.Random(42))

    current = (
        state.target_entity_id,
        state.target_is_alive,
        state.target_is_lootable,
        state.target_distance,
        state.self_x,
        state.self_y,
        state.target_x,
        state.target_y,
        state.inventory_count,
        state.inventory_max,
    )
    assert snapshot == current


def test_determinism() -> None:
    """Determinism: 100 calls with 100 new LootControllers and same state return the same Intent."""
    state = FakeLootState()
    intents = [
        LootController().decide(FSMState.LOOTING, state, {}, 100.0, random.Random(i))
        for i in range(100)
    ]

    first = intents[0]
    assert all(intent == first for intent in intents)


def test_reset_clears_attempts_and_status() -> None:
    """reset clears attempt counts and last_status."""
    controller = LootController()
    state = FakeLootState(target_entity_id="mob_1", target_distance=1.0)
    rng = random.Random(42)

    controller.decide(FSMState.LOOTING, state, {}, 100.0, rng)
    assert controller.attempts_for("mob_1") == 1
    assert controller.last_status() == LootStatus.SUCCESS
    assert controller.last_entity_id() == "mob_1"

    controller.reset()
    assert controller.attempts_for("mob_1") == 0
    assert controller.last_status() is None
    assert controller.last_entity_id() is None


def test_inventory_tracker_observe_invalid_count() -> None:
    """InventoryTracker.observe with negative inventory_count raises LootError."""
    tracker = InventoryTracker()
    with pytest.raises(LootError, match="inventory_count must be >= 0"):
        tracker.observe(inventory_count=-1, inventory_max=30, now=100.0)


def test_inventory_tracker_observe_invalid_max() -> None:
    """InventoryTracker.observe with inventory_max <= 0 raises LootError."""
    tracker = InventoryTracker()
    with pytest.raises(LootError, match="inventory_max must be > 0"):
        tracker.observe(inventory_count=5, inventory_max=0, now=100.0)
    with pytest.raises(LootError, match="inventory_max must be > 0"):
        tracker.observe(inventory_count=5, inventory_max=-10, now=100.0)


def test_observe_inventory_max_known_uses_max_as_full_at() -> None:
    """observe with inventory_max known uses inventory_max as full_at."""
    tracker = InventoryTracker()
    assert not tracker.observe(inventory_count=29, inventory_max=30, now=100.0)
    assert tracker.observe(inventory_count=30, inventory_max=30, now=101.0)


def test_observe_inventory_max_unknown_uses_absolute() -> None:
    """observe with inventory_max unknown uses inventory_full_absolute as full_at."""
    tracker = InventoryTracker(config=LootConfig(inventory_full_absolute=15))
    assert not tracker.observe(inventory_count=14, inventory_max=None, now=100.0)
    assert tracker.observe(inventory_count=15, inventory_max=None, now=101.0)


def test_observe_emits_inventory_full_and_cleared_transitions(tmp_path: Path) -> None:
    """observe emits inventory_full on transition to full, and inventory_cleared on transition to not-full."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    tracker = InventoryTracker(config=LootConfig(inventory_clear_fraction=0.8), session=session)

    # Below full
    assert not tracker.observe(inventory_count=20, inventory_max=30, now=100.0)

    # Reaches full -> emits inventory_full
    assert tracker.observe(inventory_count=30, inventory_max=30, now=101.0)

    # Stays full -> no duplicate event
    assert tracker.observe(inventory_count=30, inventory_max=30, now=102.0)

    # Drops to 25 (clear threshold is 30 * 0.8 = 24) -> stays full
    assert tracker.observe(inventory_count=25, inventory_max=30, now=103.0)

    # Drops to 24 -> cleared -> emits inventory_cleared
    assert not tracker.observe(inventory_count=24, inventory_max=30, now=104.0)

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 2
    assert '"event": "inventory_full"' in events[0]
    assert '"event": "inventory_cleared"' in events[1]


def test_hysteresis_behavior() -> None:
    """Hysteresis: after full, inventory must drop below floor(inventory_max * clear_fraction) to be cleared."""
    tracker = InventoryTracker(config=LootConfig(inventory_clear_fraction=0.8))

    # Full at 20 (max=20, clear_at=floor(20*0.8)=16)
    tracker.observe(inventory_count=20, inventory_max=20, now=100.0)
    assert tracker.is_full()

    # Drop to 17 (> 16) -> still full
    tracker.observe(inventory_count=17, inventory_max=20, now=101.0)
    assert tracker.is_full()

    # Drop to 16 (<= 16) -> cleared
    tracker.observe(inventory_count=16, inventory_max=20, now=102.0)
    assert not tracker.is_full()


def test_is_full_and_reset() -> None:
    """is_full reflects state and reset sets is_full to False."""
    tracker = InventoryTracker()
    tracker.observe(inventory_count=30, inventory_max=30, now=100.0)
    assert tracker.is_full()

    tracker.reset()
    assert not tracker.is_full()


def test_full_threshold_purity() -> None:
    """full_threshold returns expected value and is pure (does not change state)."""
    tracker = InventoryTracker(config=LootConfig(inventory_full_absolute=25))
    assert not tracker.is_full()

    assert tracker.full_threshold(50) == 50
    assert tracker.full_threshold(None) == 25
    assert not tracker.is_full()


def test_no_session_attached() -> None:
    """No session attached: all methods work without raising."""
    controller = LootController()
    state = FakeLootState()
    rng = random.Random(42)
    assert controller.decide(FSMState.LOOTING, state, {}, 100.0, rng) is not None

    tracker = InventoryTracker()
    assert tracker.observe(inventory_count=30, inventory_max=30, now=100.0)


def test_behavior_protocol_conformance() -> None:
    """LootController conforms to the executor.fsm_v2 Behavior Protocol (type level)."""
    controller = LootController()
    behavior_instance: Behavior = controller
    assert behavior_instance is not None


# AST Checks

def test_static_ast_forbidden_imports() -> None:
    """Static AST check: loot.py does not import forbidden modules."""
    loot_file = Path("src/wow_bot/farm/loot.py")
    tree = ast.parse(loot_file.read_text(), filename=str(loot_file))

    forbidden_prefixes = (
        "wow_bot.reporting",
        "wow_bot.analysis",
        "wow_bot.watchdog",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.world.store",
        "wow_bot.nav",
        "wow_bot.strategist",
        "wow_bot.humanize",
        "wow_bot.internal_dynamics",
        "wow_bot.lab",
        "wow_bot.main",
        "aiosqlite",
        "asyncio",
        "threading",
    )

    forbidden_keywords = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for prefix in forbidden_prefixes:
                    assert not mod_name.startswith(prefix), f"Forbidden import: {mod_name}"
                for kw in forbidden_keywords:
                    assert kw not in mod_name.lower(), f"Forbidden LLM import: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for prefix in forbidden_prefixes:
                assert not mod_name.startswith(prefix), f"Forbidden import from: {mod_name}"
            for kw in forbidden_keywords:
                assert kw not in mod_name.lower(), f"Forbidden LLM import: {mod_name}"


def test_static_ast_decide_no_rng_or_now_attr_access() -> None:
    """Static AST check: inside LootController.decide, no attribute access on rng or now."""
    loot_file = Path("src/wow_bot/farm/loot.py")
    tree = ast.parse(loot_file.read_text(), filename=str(loot_file))

    decide_node: ast.FunctionDef | None = None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "LootController":
            for item in node.body:
                if isinstance(item, ast.FunctionDef) and item.name == "decide":
                    decide_node = item
                    break

    assert decide_node is not None, "LootController.decide method not found"

    for node in ast.walk(decide_node):
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            assert node.value.id not in ("rng", "now"), f"Forbidden attribute access on '{node.value.id}.{node.attr}' in decide()"
