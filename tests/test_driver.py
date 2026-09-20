"""Tests for input driver protocol, NullDriver, and factory function."""

import sys

import pytest

from wow_bot.actuation.driver import (
    DriverError,
    InputDriver,
    InputSample,
    MouseButton,
    make_driver,
)
from wow_bot.actuation.drivers.null import NullDriver


class FakeDriver:
    """Local fake driver to verify typing.Protocol compliance."""

    def key_down(self, key: str) -> None:
        pass

    def key_up(self, key: str) -> None:
        pass

    def mouse_move(self, x: int, y: int) -> None:
        pass

    def mouse_down(self, button: MouseButton) -> None:
        pass

    def mouse_up(self, button: MouseButton) -> None:
        pass

    def release_all(self) -> None:
        pass

    def now(self) -> float:
        return 0.0

    def records(self) -> list[InputSample]:
        return []


def test_null_driver_records_key_down_key_up_in_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NullDriver records key_down then key_up in order."""
    time_counter = 100.0

    def fake_now(self: NullDriver) -> float:
        nonlocal time_counter
        time_counter += 1.0
        return time_counter

    monkeypatch.setattr(NullDriver, "now", fake_now)
    driver = NullDriver()

    driver.key_down("w")
    driver.key_up("w")

    recs = driver.records()
    assert len(recs) == 2
    assert recs[0].action == "key_down"
    assert recs[0].key == "w"
    assert recs[0].ts == 101.0

    assert recs[1].action == "key_up"
    assert recs[1].key == "w"
    assert recs[1].ts == 102.0


def test_null_driver_records_mouse_move_absolute(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """NullDriver records mouse_move with absolute coordinates."""
    monkeypatch.setattr(NullDriver, "now", lambda self: 50.0)
    driver = NullDriver()

    driver.mouse_move(100, 200)

    recs = driver.records()
    assert len(recs) == 1
    sample = recs[0]
    assert sample.action == "move"
    assert sample.x == 100
    assert sample.y == 200
    assert sample.key is None
    assert sample.button is None
    assert sample.ts == 50.0


def test_null_driver_records_mouse_down_mouse_up_for_each_button() -> None:
    """NullDriver records mouse_down then mouse_up for each MouseButton."""
    driver = NullDriver()

    for button in MouseButton:
        driver.mouse_down(button)
        driver.mouse_up(button)

    recs = driver.records()
    assert len(recs) == 6
    assert recs[0].action == "mouse_down"
    assert recs[0].button == MouseButton.LEFT
    assert recs[1].action == "mouse_up"
    assert recs[1].button == MouseButton.LEFT

    assert recs[2].action == "mouse_down"
    assert recs[2].button == MouseButton.RIGHT
    assert recs[3].action == "mouse_up"
    assert recs[3].button == MouseButton.RIGHT

    assert recs[4].action == "mouse_down"
    assert recs[4].button == MouseButton.MIDDLE
    assert recs[5].action == "mouse_up"
    assert recs[5].button == MouseButton.MIDDLE


def test_null_driver_held_keys_reflects_state() -> None:
    """NullDriver.held_keys reflects key_down and clears on key_up."""
    driver = NullDriver()
    assert driver.held_keys() == frozenset()

    driver.key_down("a")
    driver.key_down("b")
    assert driver.held_keys() == frozenset({"a", "b"})

    driver.key_up("a")
    assert driver.held_keys() == frozenset({"b"})

    driver.key_up("b")
    assert driver.held_keys() == frozenset()


def test_null_driver_release_all_idempotent() -> None:
    """NullDriver.release_all clears held_keys and held_buttons, is idempotent, and records one synthetic sample per call."""
    driver = NullDriver()
    driver.key_down("x")
    driver.mouse_down(MouseButton.LEFT)

    assert driver.held_keys() == frozenset({"x"})
    assert driver.held_buttons() == frozenset({MouseButton.LEFT})

    driver.release_all()
    assert driver.held_keys() == frozenset()
    assert driver.held_buttons() == frozenset()

    recs1 = driver.records()
    assert len(recs1) == 3
    assert recs1[-1].action == "release_all"
    assert recs1[-1].key is None
    assert recs1[-1].button is None

    # Idempotent second call
    driver.release_all()
    assert driver.held_keys() == frozenset()
    assert driver.held_buttons() == frozenset()

    recs2 = driver.records()
    assert len(recs2) == 4
    assert recs2[-1].action == "release_all"


def test_null_driver_records_returns_copy() -> None:
    """NullDriver.records returns a copy: mutating returned list does not affect driver state."""
    driver = NullDriver()
    driver.key_down("a")

    recs = driver.records()
    assert len(recs) == 1

    recs.clear()
    assert len(driver.records()) == 1


def test_make_driver_null() -> None:
    """make_driver("null", lab_mode=False) returns a NullDriver instance."""
    driver = make_driver("null", lab_mode=False)
    assert isinstance(driver, NullDriver)


def test_make_driver_pynput_unavailable_lab_mode_false(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_driver("pynput", lab_mode=False) raises DriverError when pynput is unavailable."""
    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)
    monkeypatch.setitem(sys.modules, "pynput.mouse", None)

    with pytest.raises(DriverError, match="pynput is required"):
        make_driver("pynput", lab_mode=False)


def test_make_driver_pynput_unavailable_lab_mode_true(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """make_driver("pynput", lab_mode=True) raises DriverError when pynput is unavailable."""
    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)
    monkeypatch.setitem(sys.modules, "pynput.mouse", None)

    with pytest.raises(DriverError, match="pynput is required"):
        make_driver("pynput", lab_mode=True)


def test_make_driver_unknown_raises_driver_error() -> None:
    """make_driver("unknown", lab_mode=False) raises DriverError."""
    with pytest.raises(DriverError, match="Unknown input driver name"):
        make_driver("unknown", lab_mode=False)


def test_input_sample_validation() -> None:
    """InputSample has exactly one of key/button set for key/mouse actions, and x/y set for move actions."""
    # Valid samples
    s_key = InputSample(
        key="space", button=None, x=None, y=None, action="key_down", ts=1.0
    )
    assert s_key.key == "space"

    s_btn = InputSample(
        key=None, button=MouseButton.LEFT, x=None, y=None, action="mouse_down", ts=1.0
    )
    assert s_btn.button == MouseButton.LEFT

    s_move = InputSample(key=None, button=None, x=10, y=20, action="move", ts=1.0)
    assert s_move.x == 10 and s_move.y == 20

    s_release = InputSample(
        key=None, button=None, x=None, y=None, action="release_all", ts=1.0
    )
    assert s_release.action == "release_all"

    # Invalid: key action with button set
    with pytest.raises(DriverError):
        InputSample(
            key="a", button=MouseButton.LEFT, x=None, y=None, action="key_down", ts=1.0
        )

    # Invalid: mouse action with key set
    with pytest.raises(DriverError):
        InputSample(
            key="a", button=MouseButton.LEFT, x=None, y=None, action="mouse_down", ts=1.0
        )

    # Invalid: move action missing x/y
    with pytest.raises(DriverError):
        InputSample(key=None, button=None, x=None, y=None, action="move", ts=1.0)

    # Invalid action
    with pytest.raises(DriverError):
        InputSample(
            key=None, button=None, x=None, y=None, action="invalid_action", ts=1.0
        )


def test_driver_now_monotonically_non_decreasing() -> None:
    """now() returns monotonically non-decreasing values across successive calls."""
    driver = NullDriver()
    t1 = driver.now()
    t2 = driver.now()
    t3 = driver.now()
    assert t1 <= t2 <= t3


def test_fake_driver_protocol_acceptance() -> None:
    """A local FakeDriver implementing InputDriver is accepted by a function taking InputDriver."""

    def consume_driver(driver: InputDriver) -> list[InputSample]:
        return driver.records()

    fake = FakeDriver()
    assert isinstance(fake, InputDriver)
    assert consume_driver(fake) == []


def test_null_driver_recording_disabled() -> None:
    """NullDriver with recording disabled returns empty list from records() while tracking held keys correctly."""
    driver = NullDriver(recording_enabled=False)
    driver.key_down("w")
    driver.mouse_down(MouseButton.RIGHT)

    assert driver.held_keys() == frozenset({"w"})
    assert driver.held_buttons() == frozenset({MouseButton.RIGHT})

    assert driver.records() == []

    driver.key_up("w")
    assert driver.held_keys() == frozenset()
    assert driver.records() == []
