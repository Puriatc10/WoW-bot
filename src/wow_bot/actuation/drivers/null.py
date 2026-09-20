"""Null (synthetic/mock) input driver implementation."""

import time

from wow_bot.actuation.driver import InputDriver, InputSample, MouseButton


class NullDriver(InputDriver):
    """In-memory synthetic input driver that records operations without touching the OS."""

    def __init__(self, recording_enabled: bool = True) -> None:
        self.recording_enabled = recording_enabled
        self._records: list[InputSample] = []
        self._held_keys: set[str] = set()
        self._held_buttons: set[MouseButton] = set()

    def now(self) -> float:
        """Return monotonic timestamp."""
        return time.monotonic()

    def records(self) -> list[InputSample]:
        """Return an immutable snapshot copy of recorded samples (or [] if disabled)."""
        if not self.recording_enabled:
            return []
        return list(self._records)

    def held_keys(self) -> frozenset[str]:
        """Return the set of currently held keys."""
        return frozenset(self._held_keys)

    def held_buttons(self) -> frozenset[MouseButton]:
        """Return the set of currently held mouse buttons."""
        return frozenset(self._held_buttons)

    def key_down(self, key: str) -> None:
        """Record key_down and update held keys set."""
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
        """Record key_up and update held keys set."""
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
        """Record absolute mouse movement."""
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
        """Record mouse_down and update held mouse buttons set."""
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
        """Record mouse_up and update held mouse buttons set."""
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
        """Clear held keys and buttons and record a synthetic release_all sample."""
        self._held_keys.clear()
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
