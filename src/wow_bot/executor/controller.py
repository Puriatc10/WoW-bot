"""Dry-run action controller for simulated game input.

This module provides an async Controller abstraction used by the Executor
and FSM layers. For research safety, Task 5.1 is dry-run only: no operating system
input injection or physical keyboard/mouse control is performed.

Public API:
    - :class:`ControllerCommand`
    - :class:`Controller`
    - :data:`SUPPORTED_MOUSE_BUTTONS`
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Final

from wow_bot.shared.logger import get_logger

log = get_logger("CONTROLLER")

SUPPORTED_MOUSE_BUTTONS: Final[frozenset[str]] = frozenset(
    {"left", "right", "middle"}
)


@dataclass(frozen=True)
class ControllerCommand:
    """Immutable representation of a recorded controller command."""

    action: str
    params: dict[str, Any]


class Controller:
    """Simulated dry-run controller for recording primitive actions.

    All methods are async to preserve the Executor layer contract, but complete
    immediately without physical delays or OS input injection.
    """

    def __init__(self, *, dry_run: bool = True) -> None:
        """Initialize the dry-run controller.

        Args:
            dry_run: Execution mode flag. Must be ``True``.

        Raises:
            ValueError: If ``dry_run`` is set to ``False``.
        """
        if not dry_run:
            raise ValueError(
                "Real input execution is not supported by this research controller"
            )

        self._dry_run: bool = True
        self._commands: list[ControllerCommand] = []

    @property
    def dry_run(self) -> bool:
        """Return True indicating dry-run mode is active."""
        return self._dry_run

    @property
    def commands(self) -> tuple[ControllerCommand, ...]:
        """Return an immutable defensive copy of recorded commands."""
        return tuple(
            ControllerCommand(action=cmd.action, params=dict(cmd.params))
            for cmd in self._commands
        )

    async def press_key(self, key: str, duration_ms: int = 50) -> None:
        """Simulate pressing a key for a given duration in milliseconds.

        Args:
            key: Symbolic key name (non-empty string).
            duration_ms: Positive integer press duration in milliseconds.

        Raises:
            ValueError: If key or duration_ms fail validation.
        """
        if not isinstance(key, str):
            raise ValueError(f"Key must be a string, got {type(key).__name__}")  # noqa: TRY004

        normalized_key = key.strip()
        if not normalized_key:
            raise ValueError("Key name cannot be empty or whitespace-only")

        if isinstance(duration_ms, bool) or not isinstance(duration_ms, int):
            raise ValueError(  # noqa: TRY004
                f"duration_ms must be an integer, got {type(duration_ms).__name__}"
            )

        if duration_ms <= 0:
            raise ValueError(
                f"duration_ms must be strictly positive (> 0), got {duration_ms}"
            )

        cmd = ControllerCommand(
            action="press_key",
            params={"key": normalized_key, "duration_ms": duration_ms},
        )
        self._commands.append(cmd)
        log.info(
            f"press_key key='{normalized_key}' duration_ms={duration_ms} dry_run=true"
        )

    async def move_mouse(self, x: int, y: int) -> None:
        """Simulate moving the mouse to coordinates (x, y).

        Args:
            x: Integer x coordinate.
            y: Integer y coordinate.

        Raises:
            ValueError: If x or y are not integers.
        """
        if isinstance(x, bool) or not isinstance(x, int):
            raise ValueError(f"x coordinate must be an integer, got {type(x).__name__}")  # noqa: TRY004

        if isinstance(y, bool) or not isinstance(y, int):
            raise ValueError(f"y coordinate must be an integer, got {type(y).__name__}")  # noqa: TRY004

        cmd = ControllerCommand(
            action="move_mouse",
            params={"x": x, "y": y},
        )
        self._commands.append(cmd)
        log.info(f"move_mouse x={x} y={y} dry_run=true")

    async def click(self, button: str = "left") -> None:
        """Simulate clicking a mouse button.

        Args:
            button: Mouse button identifier ('left', 'right', or 'middle').

        Raises:
            ValueError: If button is not in SUPPORTED_MOUSE_BUTTONS.
        """
        if not isinstance(button, str) or button not in SUPPORTED_MOUSE_BUTTONS:
            raise ValueError(
                f"Unsupported mouse button '{button}'. Must be one of {sorted(SUPPORTED_MOUSE_BUTTONS)}"
            )

        cmd = ControllerCommand(
            action="click",
            params={"button": button},
        )
        self._commands.append(cmd)
        log.info(f"click button='{button}' dry_run=true")

    async def stop_all(self) -> None:
        """Simulate stopping all active inputs.

        Safe and idempotent; records a stop_all command each time called.
        """
        cmd = ControllerCommand(
            action="stop_all",
            params={},
        )
        self._commands.append(cmd)
        log.info("stop_all dry_run=true")
