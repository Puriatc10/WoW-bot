"""Low-level input driver protocol and interface definitions."""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable


class DriverError(Exception):
    """Raised when an actuation driver encounters an unrecoverable error or missing dependency."""


class MouseButton(Enum):
    """Enumeration of supported mouse buttons."""

    LEFT = "left"
    RIGHT = "right"
    MIDDLE = "middle"


@dataclass(frozen=True)
class InputSample:
    """Immutable snapshot of a driver input event."""

    key: str | None
    button: MouseButton | None
    x: int | None
    y: int | None
    action: str  # one of: "key_down", "key_up", "move", "mouse_down", "mouse_up", "release_all"
    ts: float  # monotonic seconds from driver.now()

    def __post_init__(self) -> None:
        valid_actions = {"key_down", "key_up", "move", "mouse_down", "mouse_up", "release_all"}
        if self.action not in valid_actions:
            raise DriverError(f"Invalid action: {self.action}")

        if self.action in {"key_down", "key_up"}:
            if self.key is None or self.button is not None or self.x is not None or self.y is not None:
                raise DriverError(
                    f"Action '{self.action}' requires key to be set and button/x/y to be None"
                )
        elif self.action in {"mouse_down", "mouse_up"}:
            if self.button is None or self.key is not None or self.x is not None or self.y is not None:
                raise DriverError(
                    f"Action '{self.action}' requires button to be set and key/x/y to be None"
                )
        elif self.action == "move":
            if (
                self.x is None
                or self.y is None
                or self.key is not None
                or self.button is not None
            ):
                raise DriverError(
                    "Action 'move' requires x and y to be set and key/button to be None"
                )
        elif self.action == "release_all" and (
            self.key is not None
            or self.button is not None
            or self.x is not None
            or self.y is not None
        ):
            raise DriverError(
                "Action 'release_all' requires key, button, x, and y to be None"
            )


@runtime_checkable
class InputDriver(Protocol):
    """Protocol for low-level keyboard and mouse actuation drivers."""

    def key_down(self, key: str) -> None:
        """Press and hold a key."""
        ...

    def key_up(self, key: str) -> None:
        """Release a held key."""
        ...

    def mouse_move(self, x: int, y: int) -> None:
        """Move cursor to absolute coordinates (x, y)."""
        ...

    def mouse_down(self, button: MouseButton) -> None:
        """Press and hold a mouse button."""
        ...

    def mouse_up(self, button: MouseButton) -> None:
        """Release a held mouse button."""
        ...

    def release_all(self) -> None:
        """Release all currently held keys and mouse buttons."""
        ...

    def now(self) -> float:
        """Return a monotonic timestamp in seconds."""
        ...

    def records(self) -> list[InputSample]:
        """Return an ordered copy of recorded input samples."""
        ...


def make_driver(name: str, *, lab_mode: bool) -> InputDriver:
    """Factory function to instantiate input drivers by name."""
    if name == "null":
        from wow_bot.actuation.drivers.null import NullDriver

        return NullDriver()
    elif name == "pynput":
        from wow_bot.actuation.drivers.pynput_backend import PynputBackend

        return PynputBackend()
    elif name == "interception":
        from wow_bot.actuation.drivers.interception_backend import InterceptionBackend

        return InterceptionBackend()
    else:
        raise DriverError(f"Unknown input driver name: '{name}'")
