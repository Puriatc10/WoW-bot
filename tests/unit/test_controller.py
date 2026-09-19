"""Unit tests for dry-run Controller (Task 5.1)."""

from __future__ import annotations

import sys
from typing import Any

import pytest

from wow_bot.executor.controller import (
    SUPPORTED_MOUSE_BUTTONS,
    Controller,
    ControllerCommand,
)


@pytest.mark.asyncio
async def test_a_construction() -> None:
    """Test A: Construction with default parameters succeeds in dry-run mode."""
    controller = Controller()
    assert controller.dry_run is True
    assert controller.commands == ()


@pytest.mark.asyncio
async def test_b_real_mode_rejected() -> None:
    """Test B: Controller(dry_run=False) raises ValueError."""
    with pytest.raises(
        ValueError,
        match="Real input execution is not supported by this research controller",
    ):
        Controller(dry_run=False)


@pytest.mark.asyncio
async def test_c_press_key_records_command() -> None:
    """Test C: press_key records exactly one PressKeyCommand."""
    controller = Controller()
    await controller.press_key("w", 50)
    assert len(controller.commands) == 1
    cmd = controller.commands[0]
    assert cmd == ControllerCommand(
        action="press_key", params={"key": "w", "duration_ms": 50}
    )


@pytest.mark.asyncio
async def test_d_default_duration() -> None:
    """Test D: press_key defaults duration_ms to 50."""
    controller = Controller()
    await controller.press_key("space")
    assert len(controller.commands) == 1
    assert controller.commands[0].params["duration_ms"] == 50


@pytest.mark.asyncio
async def test_e_invalid_key() -> None:
    """Test E: Reject empty key, whitespace-only key, or non-string key."""
    controller = Controller()

    invalid_keys: list[Any] = ["", "   ", 123, None, True, False, ["w"]]
    for bad_key in invalid_keys:
        with pytest.raises(ValueError):
            await controller.press_key(bad_key, 50)

    assert controller.commands == ()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_duration",
    [0, -1, -100, 1.5, "50", True, False, None, [50]],
)
async def test_f_invalid_duration(bad_duration: Any) -> None:
    """Test F: Reject non-positive, non-int, or boolean duration_ms."""
    controller = Controller()
    with pytest.raises(ValueError):
        await controller.press_key("w", bad_duration)
    assert controller.commands == ()


@pytest.mark.asyncio
async def test_key_normalization() -> None:
    """Test key normalization: strip surrounding whitespace but preserve case."""
    controller = Controller()
    await controller.press_key("  space  ", 50)
    await controller.press_key(" Shift ", 50)
    assert controller.commands[0].params["key"] == "space"
    assert controller.commands[1].params["key"] == "Shift"


@pytest.mark.asyncio
async def test_g_move_mouse_records_command() -> None:
    """Test G: move_mouse records command with exact x, y values."""
    controller = Controller()
    await controller.move_mouse(100, 200)
    assert len(controller.commands) == 1
    assert controller.commands[0] == ControllerCommand(
        action="move_mouse", params={"x": 100, "y": 200}
    )


@pytest.mark.asyncio
async def test_h_negative_coordinates_allowed() -> None:
    """Test H: move_mouse accepts negative coordinates."""
    controller = Controller()
    await controller.move_mouse(-100, 200)
    await controller.move_mouse(50, -300)
    assert controller.commands[0].params == {"x": -100, "y": 200}
    assert controller.commands[1].params == {"x": 50, "y": -300}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "bad_x, bad_y",
    [
        (1.5, 200),
        (100, 2.5),
        ("100", 200),
        (100, "200"),
        (True, 200),
        (100, False),
        (None, 200),
        (100, None),
    ],
)
async def test_i_invalid_coordinates(bad_x: Any, bad_y: Any) -> None:
    """Test I: move_mouse rejects floats, strings, and booleans."""
    controller = Controller()
    with pytest.raises(ValueError):
        await controller.move_mouse(bad_x, bad_y)
    assert controller.commands == ()


@pytest.mark.asyncio
async def test_j_left_click_default() -> None:
    """Test J: click defaults to 'left' button."""
    controller = Controller()
    await controller.click()
    assert len(controller.commands) == 1
    assert controller.commands[0] == ControllerCommand(
        action="click", params={"button": "left"}
    )


@pytest.mark.asyncio
@pytest.mark.parametrize("btn", sorted(SUPPORTED_MOUSE_BUTTONS))
async def test_k_supported_buttons(btn: str) -> None:
    """Test K: left, right, middle clicks are accepted."""
    controller = Controller()
    await controller.click(btn)
    assert controller.commands[0].params["button"] == btn


@pytest.mark.asyncio
@pytest.mark.parametrize("bad_btn", ["side", "wheel", "banana", "", "LEFT", 123, True])
async def test_l_unsupported_buttons(bad_btn: Any) -> None:
    """Test L: Unsupported mouse buttons are rejected without recording history."""
    controller = Controller()
    with pytest.raises(ValueError):
        await controller.click(bad_btn)
    assert controller.commands == ()


@pytest.mark.asyncio
async def test_m_stop_all() -> None:
    """Test M: stop_all records a stop_all command."""
    controller = Controller()
    await controller.stop_all()
    assert len(controller.commands) == 1
    assert controller.commands[0] == ControllerCommand(
        action="stop_all", params={}
    )


@pytest.mark.asyncio
async def test_n_repeated_stop() -> None:
    """Test N: Repeated stop_all calls record multiple explicit stop requests."""
    controller = Controller()
    await controller.stop_all()
    await controller.stop_all()
    assert len(controller.commands) == 2
    assert controller.commands[0].action == "stop_all"
    assert controller.commands[1].action == "stop_all"


@pytest.mark.asyncio
async def test_o_command_ordering() -> None:
    """Test O: Executing press, move, click, stop preserves exact sequence order."""
    controller = Controller()
    await controller.press_key("w", 50)
    await controller.move_mouse(100, 200)
    await controller.click("right")
    await controller.stop_all()

    actions = [cmd.action for cmd in controller.commands]
    assert actions == ["press_key", "move_mouse", "click", "stop_all"]


@pytest.mark.asyncio
async def test_p_history_isolation() -> None:
    """Test P: Caller cannot mutate internal command history or nested params."""
    controller = Controller()
    await controller.press_key("w", 50)

    history = controller.commands
    assert isinstance(history, tuple)

    # Attempt to mutate params on returned command tuple element
    history[0].params["key"] = "hacked"

    # Verify controller's internal state remains untouched
    assert controller.commands[0].params["key"] == "w"


@pytest.mark.asyncio
async def test_q_deterministic_replay_representation() -> None:
    """Test Q: Two fresh controllers executing identical calls produce equal histories."""
    c1 = Controller()
    c2 = Controller()

    for c in (c1, c2):
        await c.press_key("w", 75)
        await c.move_mouse(120, 240)
        await c.click("left")
        await c.stop_all()

    assert c1.commands == c2.commands


def test_r_no_real_input_libraries_imported() -> None:
    """Test R: Verify production controller module does not import OS input libraries."""
    forbidden = ["pynput", "pyautogui", "keyboard", "mouse", "win32api", "quartz"]
    for mod in forbidden:
        assert mod not in sys.modules, f"Forbidden OS input library {mod} is imported!"

    import wow_bot.executor.controller as controller_module

    with open(controller_module.__file__, encoding="utf-8") as f:
        source_code = f.read()

    for mod in forbidden:
        assert f"import {mod}" not in source_code
        assert f"from {mod}" not in source_code


@pytest.mark.asyncio
async def test_s_no_sleep_dependency() -> None:
    """Test S: Controller methods complete immediately without sleeping."""
    controller = Controller()
    # Executing 100 actions should be instantaneous
    for _ in range(100):
        await controller.press_key("w", 1000)
    assert len(controller.commands) == 100


@pytest.mark.asyncio
async def test_t_async_api() -> None:
    """Test T: Primitive methods are awaitable and function in pytest-asyncio."""
    controller = Controller()
    res1 = controller.press_key("a")
    res2 = controller.move_mouse(0, 0)
    res3 = controller.click()
    res4 = controller.stop_all()

    # Confirm all return awaitable coroutines
    for coro in (res1, res2, res3, res4):
        await coro

    assert len(controller.commands) == 4
