"""Tests for ActionMapper and related actuation mapping components."""

ast_module = __import__("ast")
import ast
from pathlib import Path

import pytest

from wow_bot.actuation.driver import InputDriver, InputSample, MouseButton
from wow_bot.actuation.mapper import (
    ActionMapper,
    ActionStatus,
    DelayProvider,
    Keymap,
    MoveTo,
    NullDelay,
    RealDelay,
    Turn,
)


class FakeDriver(InputDriver):
    """Local fake driver for unit testing without OS input or real hardware."""

    def __init__(self, start_time: float = 0.0) -> None:
        self._clock: float = start_time
        self._records: list[InputSample] = []
        self._held_keys: set[str] = set()
        self.raise_on_key_down: str | None = None

    def key_down(self, key: str) -> None:
        if self.raise_on_key_down == key:
            raise RuntimeError(f"Simulated key_down failure for key '{key}'")
        self._held_keys.add(key)
        self._records.append(
            InputSample(
                key=key,
                button=None,
                x=None,
                y=None,
                action="key_down",
                ts=self._clock,
            )
        )

    def key_up(self, key: str) -> None:
        self._held_keys.discard(key)
        self._records.append(
            InputSample(
                key=key,
                button=None,
                x=None,
                y=None,
                action="key_up",
                ts=self._clock,
            )
        )

    def mouse_move(self, x: int, y: int) -> None:
        pass

    def mouse_down(self, button: MouseButton) -> None:
        pass

    def mouse_up(self, button: MouseButton) -> None:
        pass

    def release_all(self) -> None:
        self._held_keys.clear()
        self._records.append(
            InputSample(
                key=None,
                button=None,
                x=None,
                y=None,
                action="release_all",
                ts=self._clock,
            )
        )

    def now(self) -> float:
        return self._clock

    def advance_clock(self, seconds: float) -> None:
        self._clock += seconds

    def records(self) -> list[InputSample]:
        return list(self._records)

    def held_keys(self) -> set[str]:
        return set(self._held_keys)


class ClockAdvancingDelay(DelayProvider):
    """Delay provider that advances FakeDriver clock when wait is called."""

    def __init__(self, driver: FakeDriver) -> None:
        self._driver = driver

    def wait(self, seconds: float) -> None:
        self._driver.advance_clock(seconds)


class FailingDelay(DelayProvider):
    """Delay provider that raises an exception on wait."""

    def wait(self, seconds: float) -> None:
        raise RuntimeError("Simulated delay wait failure")


# Acceptance Test 1: MoveTo within tolerance returns SUCCESS with notes="arrived" and zero calls/waits.
def test_moveto_within_tolerance() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, arrival_tolerance=0.5, delay=delay)

    res = mapper.execute(MoveTo(0.3, 0.4), position=(0.0, 0.0))

    assert res.status == ActionStatus.SUCCESS
    assert res.notes == "arrived"
    assert len(driver.records()) == 0
    assert len(delay.waits()) == 0


# Acceptance Test 2: MoveTo with dy > 0 and dx == 0 presses forward key only.
def test_moveto_forward_only() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    res = mapper.execute(MoveTo(0.0, 1.0), position=(0.0, 0.0))

    assert res.status == ActionStatus.SUCCESS
    actions = [(r.action, r.key) for r in driver.records()]
    assert actions == [("key_down", "w"), ("key_up", "w")]


# Acceptance Test 3: MoveTo with dy > 0 and dx > 0 presses forward and right.
def test_moveto_forward_right() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    res = mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))

    assert res.status == ActionStatus.SUCCESS
    actions = [(r.action, r.key) for r in driver.records()]
    assert actions == [
        ("key_down", "w"),
        ("key_down", "d"),
        ("key_up", "d"),
        ("key_up", "w"),
    ]


# Acceptance Test 4: MoveTo with dy < 0 and dx < 0 presses back and left.
def test_moveto_back_left() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    res = mapper.execute(MoveTo(-1.0, -1.0), position=(0.0, 0.0))

    assert res.status == ActionStatus.SUCCESS
    actions = [(r.action, r.key) for r in driver.records()]
    assert actions == [
        ("key_down", "s"),
        ("key_down", "a"),
        ("key_up", "a"),
        ("key_up", "s"),
    ]


# Acceptance Test 5: key_down calls are emitted before delay.wait and key_up after.
def test_key_events_and_delay_order() -> None:
    events: list[str] = []

    class EventTrackingDelay(DelayProvider):
        def wait(self, seconds: float) -> None:
            events.append(f"wait:{seconds}")

    class EventTrackingDriver(FakeDriver):
        def key_down(self, key: str) -> None:
            events.append(f"key_down:{key}")
            super().key_down(key)

        def key_up(self, key: str) -> None:
            events.append(f"key_up:{key}")
            super().key_up(key)

    tr_driver = EventTrackingDriver()
    mapper = ActionMapper(tr_driver, delay=EventTrackingDelay())

    mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))

    assert events[0].startswith("key_down:")
    assert events[1].startswith("key_down:")
    assert events[2].startswith("wait:")
    assert events[3].startswith("key_up:")
    assert events[4].startswith("key_up:")


# Acceptance Test 6: key_up order is reverse of key_down order.
def test_key_up_reverse_order() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))

    key_downs = [r.key for r in driver.records() if r.action == "key_down"]
    key_ups = [r.key for r in driver.records() if r.action == "key_up"]

    assert key_downs == ["w", "d"]
    assert key_ups == ["d", "w"]
    assert key_ups == list(reversed(key_downs))


# Acceptance Test 7: Step duration equals min(move_step_duration_s, distance / speed).
def test_step_duration_moveto() -> None:
    driver = FakeDriver()
    delay = NullDelay()

    # Case A: distance / speed < move_step_duration_s (0.1 / 5.0 = 0.02s < 0.15s)
    mapper_short = ActionMapper(
        driver,
        arrival_tolerance=0.01,
        move_step_duration_s=0.15,
        assumed_speed_units_per_s=5.0,
        delay=delay,
    )
    res_short = mapper_short.execute(MoveTo(0.1, 0.0), position=(0.0, 0.0))
    assert delay.waits()[-1] == pytest.approx(0.02)
    assert res_short.notes == "step=0.020"

    # Case B: distance / speed > move_step_duration_s (10.0 / 5.0 = 2.0s > 0.15s)
    res_long = mapper_short.execute(MoveTo(10.0, 0.0), position=(0.0, 0.0))
    assert delay.waits()[-1] == pytest.approx(0.15)
    assert res_long.notes == "step=0.150"


# Acceptance Test 8: latency_ms is > 0 when driver.now() advances.
def test_latency_ms_when_clock_advances() -> None:
    driver = FakeDriver(start_time=100.0)
    delay = ClockAdvancingDelay(driver)
    mapper = ActionMapper(driver, delay=delay)

    res = mapper.execute(MoveTo(1.0, 0.0), position=(0.0, 0.0))

    assert res.latency_ms > 0.0
    # step_s for dist=1.0 at speed=5.0 is min(0.15, 0.20) = 0.15s -> 150ms
    assert res.latency_ms == pytest.approx(150.0)


# Acceptance Test 9: Turn with positive angle presses turn_left; negative presses turn_right.
def test_turn_directions() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    # Positive angle -> turn_left ("q")
    res_pos = mapper.execute(Turn(0.5), position=(0.0, 0.0))
    assert res_pos.status == ActionStatus.SUCCESS
    actions_pos = [(r.action, r.key) for r in driver.records()]
    assert actions_pos == [("key_down", "q"), ("key_up", "q")]

    # Negative angle -> turn_right ("e")
    driver_neg = FakeDriver()
    mapper_neg = ActionMapper(driver_neg, delay=delay)
    res_neg = mapper_neg.execute(Turn(-0.5), position=(0.0, 0.0))
    assert res_neg.status == ActionStatus.SUCCESS
    actions_neg = [(r.action, r.key) for r in driver_neg.records()]
    assert actions_neg == [("key_down", "e"), ("key_up", "e")]


# Acceptance Test 10: Turn with abs(angle) <= 1e-6 returns SUCCESS with notes="noop" and zero calls.
def test_turn_noop() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    for angle in [0.0, 1e-7, -1e-7, 1e-6, -1e-6]:
        res = mapper.execute(Turn(angle), position=(0.0, 0.0))
        assert res.status == ActionStatus.SUCCESS
        assert res.notes == "noop"

    assert len(driver.records()) == 0
    assert len(delay.waits()) == 0


# Acceptance Test 11: Turn step duration equals min(turn_step_duration_s, abs(angle) / turn_rate).
def test_turn_step_duration() -> None:
    driver = FakeDriver()
    delay = NullDelay()

    # Case A: abs(angle) / turn_rate < turn_step_duration_s (0.1 / 3.14 = 0.0318...s < 0.10s)
    mapper = ActionMapper(
        driver,
        turn_step_duration_s=0.10,
        assumed_turn_rate_rad_per_s=3.14,
        delay=delay,
    )
    res_small = mapper.execute(Turn(0.1), position=(0.0, 0.0))
    expected_step_small = 0.1 / 3.14
    assert delay.waits()[-1] == pytest.approx(expected_step_small)
    assert res_small.notes == f"step={expected_step_small:.3f}"

    # Case B: abs(angle) / turn_rate > turn_step_duration_s (3.14 / 3.14 = 1.0s > 0.10s)
    res_large = mapper.execute(Turn(3.14), position=(0.0, 0.0))
    assert delay.waits()[-1] == pytest.approx(0.10)
    assert res_large.notes == "step=0.100"


# Acceptance Test 12: If driver raises inside key_down, no key remains held.
def test_driver_raises_in_key_down_releases_held_keys() -> None:
    driver = FakeDriver()
    driver.raise_on_key_down = "d"  # second key in MoveTo(1, 1)
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    with pytest.raises(RuntimeError, match="Simulated key_down failure"):
        mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))

    assert len(driver.held_keys()) == 0


# Acceptance Test 13: If delay.wait raises, no key remains held.
def test_delay_wait_raises_releases_held_keys() -> None:
    driver = FakeDriver()
    delay = FailingDelay()
    mapper = ActionMapper(driver, delay=delay)

    with pytest.raises(RuntimeError, match="Simulated delay wait failure"):
        mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))

    assert len(driver.held_keys()) == 0


# Acceptance Test 14: Two consecutive MoveTo calls are independent: no state leaks.
def test_consecutive_moveto_calls_independent() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    mapper = ActionMapper(driver, delay=delay)

    res1 = mapper.execute(MoveTo(0.0, 1.0), position=(0.0, 0.0))
    rec1_count = len(driver.records())

    res2 = mapper.execute(MoveTo(0.0, 2.0), position=(0.0, 1.0))
    rec2_count = len(driver.records()) - rec1_count

    assert res1.status == ActionStatus.SUCCESS
    assert res2.status == ActionStatus.SUCCESS
    assert len(driver.held_keys()) == 0
    # Second call should only have 2 records (key_down, key_up for 'w')
    assert rec2_count == 2
    second_call_records = driver.records()[rec1_count:]
    assert [(r.action, r.key) for r in second_call_records] == [
        ("key_down", "w"),
        ("key_up", "w"),
    ]


# Acceptance Test 15: Keymap customization is honored.
def test_custom_keymap_honored() -> None:
    driver = FakeDriver()
    delay = NullDelay()
    custom_map = Keymap(
        forward="i",
        back="k",
        left="j",
        right="l",
        turn_left="u",
        turn_right="o",
    )
    mapper = ActionMapper(driver, keymap=custom_map, delay=delay)

    # MoveTo(1, 1) should press 'i' and 'l'
    mapper.execute(MoveTo(1.0, 1.0), position=(0.0, 0.0))
    move_keys = [r.key for r in driver.records()]
    assert move_keys == ["i", "l", "l", "i"]

    # Turn(0.5) should press 'u'
    driver_turn = FakeDriver()
    mapper_turn = ActionMapper(driver_turn, keymap=custom_map, delay=delay)
    mapper_turn.execute(Turn(0.5), position=(0.0, 0.0))
    turn_keys = [r.key for r in driver_turn.records()]
    assert turn_keys == ["u", "u"]


# Acceptance Test 16: Validation: negative or zero arrival_tolerance raises ValueError.
def test_validation_arrival_tolerance() -> None:
    driver = FakeDriver()
    with pytest.raises(ValueError, match="arrival_tolerance must be positive"):
        ActionMapper(driver, arrival_tolerance=0.0)

    with pytest.raises(ValueError, match="arrival_tolerance must be positive"):
        ActionMapper(driver, arrival_tolerance=-0.5)


# Acceptance Test 17: Validation: negative move_step_duration_s raises ValueError.
def test_validation_move_step_duration() -> None:
    driver = FakeDriver()
    with pytest.raises(ValueError, match="move_step_duration_s must be non-negative"):
        ActionMapper(driver, move_step_duration_s=-0.1)

    # 0.0 is allowed
    mapper = ActionMapper(driver, move_step_duration_s=0.0)
    assert mapper is not None


# Acceptance Test 18: Validation: negative turn_step_duration_s raises ValueError.
def test_validation_turn_step_duration() -> None:
    driver = FakeDriver()
    with pytest.raises(ValueError, match="turn_step_duration_s must be non-negative"):
        ActionMapper(driver, turn_step_duration_s=-0.1)

    # 0.0 is allowed
    mapper = ActionMapper(driver, turn_step_duration_s=0.0)
    assert mapper is not None


# Acceptance Test 19: Validation: non-positive assumed_speed_units_per_s raises ValueError.
def test_validation_assumed_speed() -> None:
    driver = FakeDriver()
    with pytest.raises(ValueError, match="assumed_speed_units_per_s must be positive"):
        ActionMapper(driver, assumed_speed_units_per_s=0.0)

    with pytest.raises(ValueError, match="assumed_speed_units_per_s must be positive"):
        ActionMapper(driver, assumed_speed_units_per_s=-5.0)


# Acceptance Test 20: Validation: non-positive assumed_turn_rate_rad_per_s raises ValueError.
def test_validation_assumed_turn_rate() -> None:
    driver = FakeDriver()
    with pytest.raises(ValueError, match="assumed_turn_rate_rad_per_s must be positive"):
        ActionMapper(driver, assumed_turn_rate_rad_per_s=0.0)

    with pytest.raises(ValueError, match="assumed_turn_rate_rad_per_s must be positive"):
        ActionMapper(driver, assumed_turn_rate_rad_per_s=-3.14)


# Acceptance Test 21: NullDelay.waits() returns a copy.
def test_null_delay_waits_copy() -> None:
    delay = NullDelay()
    delay.wait(0.15)
    delay.wait(0.10)

    waits_copy = delay.waits()
    assert waits_copy == [0.15, 0.10]

    # Mutate the returned list
    waits_copy.append(999.0)

    # Internal state should remain unchanged
    assert delay.waits() == [0.15, 0.10]


def test_real_delay() -> None:
    real_delay = RealDelay()
    real_delay.wait(-1.0)
    real_delay.wait(0.0001)


# Acceptance Test 22: ActionMapper static AST check against forbidden imports.
def test_action_mapper_forbidden_imports_ast() -> None:
    mapper_path = Path(__file__).parent.parent / "src" / "wow_bot" / "actuation" / "mapper.py"
    source = mapper_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(mapper_path))

    forbidden_modules = {
        "wow_bot.safety",
        "wow_bot.session",
        "wow_bot.executor",
        "wow_bot.perception",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                for forbidden in forbidden_modules:
                    assert not alias.name.startswith(forbidden), (
                        f"Forbidden import '{alias.name}' found in {mapper_path}"
                    )
        elif isinstance(node, ast_module.ImportFrom) and node.module:
            for forbidden in forbidden_modules:
                assert not node.module.startswith(forbidden), (
                    f"Forbidden importfrom '{node.module}' found in {mapper_path}"
                )
