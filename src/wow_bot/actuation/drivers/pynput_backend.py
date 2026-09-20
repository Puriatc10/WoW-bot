"""Pynput-backed OS input driver implementation."""

import time
from typing import Any

from wow_bot.actuation.driver import DriverError, InputDriver, InputSample, MouseButton


class PynputBackend(InputDriver):
    """OS input driver using pynput for desktop automation in LAB_MODE."""

    def __init__(self, recording_enabled: bool = True) -> None:
        self.recording_enabled = recording_enabled
        self._records: list[InputSample] = []
        self._held_keys: set[str] = set()
        self._held_buttons: set[MouseButton] = set()

        try:
            import pynput  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError as err:
            raise DriverError("pynput is required for PynputBackend but is not installed.") from err

        self._pynput = pynput
        self._keyboard_controller = pynput.keyboard.Controller()
        self._mouse_controller = pynput.mouse.Controller()

    def now(self) -> float:
        """Return monotonic timestamp."""
        return time.monotonic()

    def records(self) -> list[InputSample]:
        """Return an immutable snapshot copy of recorded samples (or [] if disabled)."""
        if not self.recording_enabled:
            return []
        return list(self._records)

    def _resolve_key(self, key: str) -> Any:
        k_lower = key.lower()
        if len(key) == 1:
            return key
        special_keys = {
            "f1": self._pynput.keyboard.Key.f1,
            "f2": self._pynput.keyboard.Key.f2,
            "f3": self._pynput.keyboard.Key.f3,
            "f4": self._pynput.keyboard.Key.f4,
            "f5": self._pynput.keyboard.Key.f5,
            "f6": self._pynput.keyboard.Key.f6,
            "f7": self._pynput.keyboard.Key.f7,
            "f8": self._pynput.keyboard.Key.f8,
            "f9": self._pynput.keyboard.Key.f9,
            "f10": self._pynput.keyboard.Key.f10,
            "f11": self._pynput.keyboard.Key.f11,
            "f12": self._pynput.keyboard.Key.f12,
            "esc": self._pynput.keyboard.Key.esc,
            "space": self._pynput.keyboard.Key.space,
            "tab": self._pynput.keyboard.Key.tab,
            "enter": self._pynput.keyboard.Key.enter,
        }
        if k_lower in special_keys:
            return special_keys[k_lower]
        raise DriverError(f"Unsupported key name for pynput backend: '{key}'")

    def _resolve_button(self, button: MouseButton) -> Any:
        if button == MouseButton.LEFT:
            return self._pynput.mouse.Button.left
        elif button == MouseButton.RIGHT:
            return self._pynput.mouse.Button.right
        elif button == MouseButton.MIDDLE:
            return self._pynput.mouse.Button.middle
        else:
            raise DriverError(f"Unsupported mouse button: {button}")

    def key_down(self, key: str) -> None:
        """Press and hold a key."""
        resolved = self._resolve_key(key)
        self._keyboard_controller.press(resolved)
        self._held_keys.add(key)
        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=key,
                    button=None,
                    x=None,
                    y=None,
                    action="key_down",
                    ts=self.now(),
                )
            )

    def key_up(self, key: str) -> None:
        """Release a held key."""
        resolved = self._resolve_key(key)
        self._keyboard_controller.release(resolved)
        self._held_keys.discard(key)
        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=key,
                    button=None,
                    x=None,
                    y=None,
                    action="key_up",
                    ts=self.now(),
                )
            )

    def mouse_move(self, x: int, y: int) -> None:
        """Move cursor to absolute coordinates (x, y)."""
        self._mouse_controller.position = (x, y)
        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=None,
                    button=None,
                    x=x,
                    y=y,
                    action="move",
                    ts=self.now(),
                )
            )

    def mouse_down(self, button: MouseButton) -> None:
        """Press and hold a mouse button."""
        resolved = self._resolve_button(button)
        self._mouse_controller.press(resolved)
        self._held_buttons.add(button)
        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=None,
                    button=button,
                    x=None,
                    y=None,
                    action="mouse_down",
                    ts=self.now(),
                )
            )

    def mouse_up(self, button: MouseButton) -> None:
        """Release a held mouse button."""
        resolved = self._resolve_button(button)
        self._mouse_controller.release(resolved)
        self._held_buttons.discard(button)
        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=None,
                    button=button,
                    x=None,
                    y=None,
                    action="mouse_up",
                    ts=self.now(),
                )
            )

    def release_all(self) -> None:
        """Release all currently tracked held keys and mouse buttons."""
        for key in list(self._held_keys):
            try:
                resolved_key = self._resolve_key(key)
                self._keyboard_controller.release(resolved_key)
            except Exception:  # noqa: BLE001, S110
                pass
        self._held_keys.clear()

        for button in list(self._held_buttons):
            try:
                resolved_btn = self._resolve_button(button)
                self._mouse_controller.release(resolved_btn)
            except Exception:  # noqa: BLE001, S110
                pass
        self._held_buttons.clear()

        if self.recording_enabled:
            self._records.append(
                InputSample(
                    key=None,
                    button=None,
                    x=None,
                    y=None,
                    action="release_all",
                    ts=self.now(),
                )
            )
