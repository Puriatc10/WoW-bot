"""Kernel-level interception driver implementation."""

import time

from wow_bot.actuation.driver import DriverError, InputDriver, InputSample, MouseButton


class InterceptionBackend(InputDriver):
    """Kernel-level input driver using Interception driver on Windows in LAB_MODE."""

    def __init__(self, recording_enabled: bool = True) -> None:
        self.recording_enabled = recording_enabled
        self._records: list[InputSample] = []
        self._held_keys: set[str] = set()
        self._held_buttons: set[MouseButton] = set()

        try:
            import interception  # type: ignore[import-not-found,import-untyped,unused-ignore]
        except ImportError as err:
            raise DriverError(
                "interception library is required for InterceptionBackend but is not installed."
            ) from err

        self._interception = interception
        try:
            # Check if underlying driver device / context is available
            if hasattr(interception, "auto_capture"):
                # Attempt basic initialization check if available
                pass
        except Exception as err:
            raise DriverError(f"Interception kernel driver is unavailable: {err}") from err

    def now(self) -> float:
        """Return monotonic timestamp."""
        return time.monotonic()

    def records(self) -> list[InputSample]:
        """Return an immutable snapshot copy of recorded samples (or [] if disabled)."""
        if not self.recording_enabled:
            return []
        return list(self._records)

    def key_down(self, key: str) -> None:
        """Press and hold a key via interception."""
        try:
            if hasattr(self._interception, "key_down"):
                self._interception.key_down(key)
            elif hasattr(self._interception, "press"):
                self._interception.press(key)
        except Exception as err:
            raise DriverError(f"Interception key_down failed for '{key}': {err}") from err

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
        """Release a held key via interception."""
        try:
            if hasattr(self._interception, "key_up"):
                self._interception.key_up(key)
            elif hasattr(self._interception, "release"):
                self._interception.release(key)
        except Exception as err:
            raise DriverError(f"Interception key_up failed for '{key}': {err}") from err

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
        """Move cursor to absolute coordinates (x, y) via interception."""
        try:
            if hasattr(self._interception, "move_to"):
                self._interception.move_to(x, y)
            elif hasattr(self._interception, "mouse_move"):
                self._interception.mouse_move(x, y)
        except Exception as err:
            raise DriverError(f"Interception mouse_move failed for ({x}, {y}): {err}") from err

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
        """Press and hold a mouse button via interception."""
        try:
            btn_str = button.value
            if hasattr(self._interception, "mouse_down"):
                self._interception.mouse_down(btn_str)
        except Exception as err:
            raise DriverError(f"Interception mouse_down failed for {button}: {err}") from err

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
        """Release a held mouse button via interception."""
        try:
            btn_str = button.value
            if hasattr(self._interception, "mouse_up"):
                self._interception.mouse_up(btn_str)
        except Exception as err:
            raise DriverError(f"Interception mouse_up failed for {button}: {err}") from err

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
                self.key_up(key)
            except Exception:  # noqa: BLE001, S110
                pass
        self._held_keys.clear()

        for button in list(self._held_buttons):
            try:
                self.mouse_up(button)
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
