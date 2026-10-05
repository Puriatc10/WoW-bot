"""Tests for T-FIX-08 gameplay primitives (cast/loot/vendor/jump) and heading-aware mapping."""

from __future__ import annotations

import math
import random
from typing import Any
from unittest.mock import MagicMock

import pytest

from wow_bot.actuation.actuator import NullActuator
from wow_bot.actuation.driver import InputDriver, InputSample, MouseButton
from wow_bot.actuation.mapper import (
    ActionMapper,
    ActionStatus,
    Cast,
    Intent,
    Jump,
    Keymap,
    Loot,
    MoveTo,
    NullDelay,
    Turn,
    VendorInteract,
)
from wow_bot.combat.loop import CombatLoop
from wow_bot.combat.rotation import (
    Condition,
    RotationConfig,
    RotationRule,
    RotationTable,
)
from wow_bot.combat.targeting import TargetSelector
from wow_bot.executor.controller import ControllerCommand, SimulationController
from wow_bot.executor.recovery import (
    RecoveryBehaviorKind,
    RecoveryConfig,
    RecoveryPlanner,
    jump_intent,
)
from wow_bot.executor.states import FSMState
from wow_bot.farm.loot import LootConfig, LootController
from wow_bot.farm.vendor import (
    VendorConfig,
    VendorController,
    VendorLocation,
    VendorStatus,
)


class RecordingDriver(InputDriver):
    """Test fake driver capturing low-level key operations."""

    def __init__(self) -> None:
        self._records: list[InputSample] = []
        self._held: set[str] = set()
        self._now = 0.0

    def key_down(self, key: str) -> None:
        self._held.add(key)
        self._records.append(
            InputSample(key=key, button=None, x=None, y=None, action="key_down", ts=self._now)
        )

    def key_up(self, key: str) -> None:
        self._held.discard(key)
        self._records.append(
            InputSample(key=key, button=None, x=None, y=None, action="key_up", ts=self._now)
        )

    def mouse_move(self, x: int, y: int) -> None:
        pass

    def mouse_down(self, button: MouseButton) -> None:
        pass

    def mouse_up(self, button: MouseButton) -> None:
        pass

    def release_all(self) -> None:
        self._held.clear()
        self._records.append(
            InputSample(key=None, button=None, x=None, y=None, action="release_all", ts=self._now)
        )

    def now(self) -> float:
        return self._now

    def advance(self, dt: float) -> None:
        self._now += dt

    @property
    def records(self) -> list[InputSample]:
        return list(self._records)


class FakeEntity:
    def __init__(self, entity_id: str, distance: float = 1.0, is_alive: bool = True) -> None:
        self.entity_id = entity_id
        self.distance = distance
        self.is_alive = is_alive


class FakeCombatState:
    def __init__(
        self,
        current_target_id: str | None = "mob_1",
        target_in_range: bool = True,
        gcd_ready: bool = True,
    ) -> None:
        self.current_target_id = current_target_id
        self.entities = (FakeEntity(entity_id="mob_1", distance=2.0),)
        self.target_in_range = target_in_range
        self.target_hp_percent = 80.0
        self.self_hp_percent = 100.0
        self.resource = 100.0
        self.resource_max = 100.0
        self.gcd_ready = gcd_ready
        self.self_x = 0.0
        self.self_y = 0.0
        self.target_x = 2.0
        self.target_y = 0.0

    def target_has_debuff(self, debuff_id: str, /) -> bool:
        return False


class FakeLootState:
    def __init__(self) -> None:
        self.target_entity_id = "corpse_42"
        self.target_is_alive = False
        self.target_is_lootable = True
        self.target_distance = 1.0
        self.self_x = 15.0
        self.self_y = 25.0
        self.target_x = 15.0
        self.target_y = 25.0
        self.inventory_count = 2
        self.inventory_max = 30


class FakeVendorState:
    def __init__(self) -> None:
        self.self_x = 10.0
        self.self_y = 20.0
        self.inventory_count = 3
        self.durability_fraction = 0.3


# Acceptance 1: Each primitive produces its own distinct intent/command
def test_distinct_primitive_types_and_parameters() -> None:
    """Each primitive is a distinct dataclass intent carrying explicit parameters."""
    cast = Cast(spell_id="fireball_1", target_id="target_99")
    loot = Loot(target_id="corpse_12")
    vendor = VendorInteract(vendor_entity="vendor_1", action="sell")
    jump = Jump(direction="forward")

    assert isinstance(cast, Cast)
    assert cast.spell_id == "fireball_1"
    assert cast.target_id == "target_99"

    assert isinstance(loot, Loot)
    assert loot.target_id == "corpse_12"

    assert isinstance(vendor, VendorInteract)
    assert vendor.vendor_entity == "vendor_1"
    assert vendor.action == "sell"

    assert isinstance(jump, Jump)
    assert jump.direction == "forward"

    # All primitives are distinct types
    types = {type(cast), type(loot), type(vendor), type(jump), MoveTo, Turn}
    assert len(types) == 6


def test_mapper_execution_of_each_primitive() -> None:
    """Mapper maps each primitive to distinct driver key actions."""
    driver = RecordingDriver()
    keymap = Keymap(
        jump="space",
        interact="f",
        loot="f",
        default_cast_key="1",
        spell_keys={"frostbolt": "2"},
    )
    mapper = ActionMapper(driver, keymap=keymap, delay=NullDelay())

    # 1. Cast mapped spell
    res_cast = mapper.execute(Cast(spell_id="frostbolt", target_id="mob_1"), position=(0.0, 0.0))
    assert res_cast.status == ActionStatus.SUCCESS
    assert ("key_down", "2") in [(r.action, r.key) for r in driver.records]
    assert ("key_up", "2") in [(r.action, r.key) for r in driver.records]

    # 2. Loot
    driver = RecordingDriver()
    mapper = ActionMapper(driver, keymap=keymap, delay=NullDelay())
    res_loot = mapper.execute(Loot(target_id="corpse_1"), position=(0.0, 0.0))
    assert res_loot.status == ActionStatus.SUCCESS
    assert ("key_down", "f") in [(r.action, r.key) for r in driver.records]
    assert ("key_up", "f") in [(r.action, r.key) for r in driver.records]

    # 3. VendorInteract
    driver = RecordingDriver()
    mapper = ActionMapper(driver, keymap=keymap, delay=NullDelay())
    res_vendor = mapper.execute(VendorInteract(vendor_entity=10, action="repair"), position=(0.0, 0.0))
    assert res_vendor.status == ActionStatus.SUCCESS
    assert ("key_down", "f") in [(r.action, r.key) for r in driver.records]
    assert ("key_up", "f") in [(r.action, r.key) for r in driver.records]

    # 4. Jump
    driver = RecordingDriver()
    mapper = ActionMapper(driver, keymap=keymap, delay=NullDelay())
    res_jump = mapper.execute(Jump(direction="forward"), position=(0.0, 0.0))
    assert res_jump.status == ActionStatus.SUCCESS
    keys = [r.key for r in driver.records if r.action == "key_down"]
    assert "space" in keys


# Acceptance 2: Loot and vendor no longer emit a disguised MoveTo
def test_loot_no_longer_emits_disguised_moveto() -> None:
    """LootController emits Loot intent when in reach, not MoveTo."""
    controller = LootController(config=LootConfig(loot_reach_units=2.5))
    state = FakeLootState()
    rng = random.Random(42)

    intent = controller.decide(FSMState.LOOTING, state, {}, 0.0, rng)
    assert not isinstance(intent, MoveTo)
    assert isinstance(intent, Loot)
    assert intent.target_id == "corpse_42"


def test_vendor_no_longer_emits_disguised_moveto() -> None:
    """VendorController emits VendorInteract intent during sell and repair phases, not MoveTo."""
    mock_navigator = MagicMock()
    mock_nav_result = MagicMock()
    mock_nav_result.status.value = "arrived"
    mock_nav_result.status.name = "ARRIVED"
    mock_navigator.go_to.return_value = mock_nav_result

    executed_intents: list[Intent] = []

    class FakeActuator:
        def execute(self, intent: Intent, *, position: tuple[float, float]) -> Any:
            executed_intents.append(intent)
            from wow_bot.actuation.mapper import ActionResult, ActionStatus

            return ActionResult(status=ActionStatus.SUCCESS, latency_ms=0.0)

    controller = VendorController(
        navigator=mock_navigator,
        actuator=FakeActuator(),  # type: ignore[arg-type]
        config=VendorConfig(sell_chunk_size=1, max_sell_steps=1, max_repair_steps=1),
    )
    vendor = VendorLocation(node_id=101, x=50.0, y=50.0, kind="vendor", name="Bob")
    state = FakeVendorState()

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS
    assert len(executed_intents) == 2

    # Neither intent is MoveTo
    for intent in executed_intents:
        assert not isinstance(intent, MoveTo)
        assert isinstance(intent, VendorInteract)

    assert executed_intents[0].action == "sell"
    assert executed_intents[0].vendor_entity == 101
    assert executed_intents[1].action == "repair"
    assert executed_intents[1].vendor_entity == 101


def test_combat_loop_no_longer_emits_disguised_moveto() -> None:
    """CombatLoop emits Cast intent when casting, not MoveTo."""
    rule = RotationRule(spell_id="arcane_shot", conditions=(Condition(kind="gcd_ready", value=True),))
    rot_table = RotationTable(RotationConfig(rules=(rule,)))
    targeting = TargetSelector()
    loop = CombatLoop(rot_table, targeting)
    state = FakeCombatState()
    rng = random.Random(42)

    intent = loop.decide(FSMState.COMBAT, state, {}, 0.0, rng)  # type: ignore[arg-type]
    assert not isinstance(intent, MoveTo)
    assert isinstance(intent, Cast)
    assert intent.spell_id == "arcane_shot"
    assert intent.target_id == "mob_1"


def test_recovery_jump_no_longer_emits_disguised_turn() -> None:
    """jump_intent emits Jump intent, not Turn(0.0)."""
    intent = jump_intent()
    assert not isinstance(intent, Turn)
    assert isinstance(intent, Jump)
    assert intent.direction == "forward"

    planner = RecoveryPlanner(
        config=RecoveryConfig(behavior_weights=(0.0, 0.0, 1.0, 0.0)),
    )
    planned = planner.plan(random.Random(1), attempts_so_far=0)
    assert planner.last_behavior() == RecoveryBehaviorKind.JUMP
    assert isinstance(planned, Jump)


# Acceptance 3: Movement key selection is heading-correct
def test_movement_key_selection_heading_aware() -> None:
    """A rotated facing produces different movement keys for the same destination."""
    driver = RecordingDriver()
    mapper = ActionMapper(driver, delay=NullDelay())

    # Target is to the East: (10.0, 0.0) from (0.0, 0.0)
    target = MoveTo(10.0, 0.0)

    # 1. Facing East (heading = 0.0 rad): Target is straight ahead -> 'w' (forward)
    driver = RecordingDriver()
    mapper = ActionMapper(driver, delay=NullDelay())
    mapper.execute(target, position=(0.0, 0.0), heading=0.0)
    keys_east = {r.key for r in driver.records if r.action == "key_down"}
    assert keys_east == {"w"}

    # 2. Facing North (heading = pi/2 rad): Target is to the right -> 'd' (right)
    driver = RecordingDriver()
    mapper = ActionMapper(driver, delay=NullDelay())
    mapper.execute(target, position=(0.0, 0.0), heading=math.pi / 2)
    keys_north = {r.key for r in driver.records if r.action == "key_down"}
    assert keys_north == {"d"}

    # 3. Facing West (heading = pi rad): Target is behind -> 's' (back)
    driver = RecordingDriver()
    mapper = ActionMapper(driver, delay=NullDelay())
    mapper.execute(target, position=(0.0, 0.0), heading=math.pi)
    keys_west = {r.key for r in driver.records if r.action == "key_down"}
    assert keys_west == {"s"}

    # 4. Facing South (heading = -pi/2 rad): Target is to the left -> 'a' (left)
    driver = RecordingDriver()
    mapper = ActionMapper(driver, delay=NullDelay())
    mapper.execute(target, position=(0.0, 0.0), heading=-math.pi / 2)
    keys_south = {r.key for r in driver.records if r.action == "key_down"}
    assert keys_south == {"a"}

    # Verify that different headings produced different keys for the same movement target
    assert keys_east != keys_north
    assert keys_north != keys_west
    assert keys_west != keys_south


# Acceptance 4: MOCK_MODE records all four primitives symbolically
@pytest.mark.asyncio
async def test_simulation_controller_records_all_primitives_symbolically() -> None:
    """SimulationController records Cast, Loot, VendorInteract, and Jump symbolically."""
    controller = SimulationController(dry_run=True)
    assert controller.dry_run is True

    cast = Cast(spell_id="shoot", target_id="enemy_1")
    loot = Loot(target_id="corpse_1")
    vendor = VendorInteract(vendor_entity="vendor_9", action="sell")
    jump = Jump(direction="forward")

    controller.execute(cast)
    controller.execute(loot)
    controller.execute(vendor)
    controller.execute(jump)

    # 1. Symbolic intents recorded
    assert controller.intents == (cast, loot, vendor, jump)

    # 2. Commands recorded as ControllerCommand
    commands = controller.commands
    assert len(commands) == 4
    assert commands[0] == ControllerCommand(action="cast", params={"spell_id": "shoot", "target_id": "enemy_1"})
    assert commands[1] == ControllerCommand(action="loot", params={"target_id": "corpse_1"})
    assert commands[2] == ControllerCommand(action="vendorinteract", params={"vendor_entity": "vendor_9", "action": "sell"})
    assert commands[3] == ControllerCommand(action="jump", params={"direction": "forward"})


def test_null_actuator_records_all_primitives() -> None:
    """NullActuator in MOCK_MODE records all four primitives with position."""
    actuator = NullActuator()
    cast = Cast(spell_id="heal", target_id="self")
    loot = Loot(target_id="corpse_5")
    vendor = VendorInteract(vendor_entity="smith", action="repair")
    jump = Jump(direction="forward")

    actuator.execute(cast, position=(1.0, 2.0))
    actuator.execute(loot, position=(3.0, 4.0))
    actuator.execute(vendor, position=(5.0, 6.0))
    actuator.execute(jump, position=(7.0, 8.0))

    recorded = actuator.recorded()
    assert len(recorded) == 4
    assert recorded[0] == (cast, (1.0, 2.0))
    assert recorded[1] == (loot, (3.0, 4.0))
    assert recorded[2] == (vendor, (5.0, 6.0))
    assert recorded[3] == (jump, (7.0, 8.0))


# Acceptance 5: dry_run=False in MOCK_MODE remains rejected
def test_simulation_controller_dry_run_false_raises() -> None:
    """dry_run=False in MOCK_MODE Controller/SimulationController is rejected."""
    with pytest.raises(ValueError, match="Real input execution is not supported"):
        SimulationController(dry_run=False)
