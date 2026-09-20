"""Kill switch implementation for WoW-bot lab execution mode."""

from __future__ import annotations

import threading
from collections.abc import Callable
from typing import Any, Protocol

__all__ = [
    "KillSwitch",
    "KillSwitchBackend",
    "KillSwitchError",
    "NullBackend",
    "PynputBackend",
]

_SUPPORTED_SPECIAL_KEYS = {f"f{i}" for i in range(1, 13)} | {"esc", "space", "tab", "enter"}


class KillSwitchError(Exception):
    """Raised when kill switch configuration, initialization, or backend fails."""


class KillSwitchBackend(Protocol):
    """Protocol defining interface for kill switch backends."""

    def start(self, on_trigger: Callable[[], None]) -> None:
        """Start listening for kill switch key trigger."""
        ...

    def stop(self) -> None:
        """Stop listening for kill switch key trigger."""
        ...


class NullBackend:
    """Null kill switch backend for MOCK_MODE and unit tests."""

    def __init__(self) -> None:
        self._callback: Callable[[], None] | None = None

    def start(self, on_trigger: Callable[[], None]) -> None:
        """Store the callback but never invoke it."""
        self._callback = on_trigger

    def stop(self) -> None:
        """No-op stop."""


class PynputBackend:
    """Pynput-based kill switch backend watching keyboard events."""

    def __init__(self, key: str) -> None:
        try:
            import pynput.keyboard  # type: ignore[import-untyped]
        except Exception as exc:
            raise KillSwitchError("pynput is required for PynputBackend") from exc

        self._pynput_kb = pynput.keyboard
        self._target_key = self._validate_and_parse_key(key)
        self._listener: Any = None
        self._callback: Callable[[], None] | None = None

    def _validate_and_parse_key(self, key: str) -> Any:
        key_clean = key.strip().lower()
        if key_clean in _SUPPORTED_SPECIAL_KEYS:
            target = getattr(self._pynput_kb.Key, key_clean, None)
            if target is None:
                raise KillSwitchError(
                    f"Special key '{key}' not found on pynput.keyboard.Key"
                )
            return target
        elif len(key) == 1 and key.isprintable() and not key.isspace():
            try:
                return self._pynput_kb.KeyCode.from_char(key)
            except Exception as exc:
                raise KillSwitchError(
                    f"Failed to create KeyCode for char '{key}': {exc}"
                ) from exc
        else:
            raise KillSwitchError(f"Unsupported key string format for PynputBackend: {key!r}")

    def start(self, on_trigger: Callable[[], None]) -> None:
        """Start listening for keyboard events matching target key."""
        if self._listener is not None:
            return
        self._callback = on_trigger

        def _on_press(pressed_key: Any) -> None:
            if self._matches(pressed_key) and self._callback is not None:
                self._callback()

        self._listener = self._pynput_kb.Listener(on_press=_on_press)
        self._listener.start()

    def stop(self) -> None:
        """Stop key listener if running."""
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001, S110
                pass
            self._listener = None

    def _matches(self, pressed_key: Any) -> bool:
        if pressed_key == self._target_key:
            return True
        if (
            hasattr(self._target_key, "name")
            and hasattr(pressed_key, "name")
            and pressed_key.name == self._target_key.name
        ):
            return True
        return bool(
            hasattr(self._target_key, "char")
            and hasattr(pressed_key, "char")
            and pressed_key.char == self._target_key.char
        )


class KillSwitch:
    """Physical kill switch observer."""

    def __init__(
        self,
        key: str,
        on_trigger: Callable[[], None],
        *,
        lab_mode: bool = False,
        backend: KillSwitchBackend | None = None,
    ) -> None:
        if not isinstance(key, str) or not key.strip():
            raise KillSwitchError("Key must be a non-empty, non-whitespace string.")

        self._key = key
        self._user_callback = on_trigger

        if backend is not None:
            self._backend = backend
        elif lab_mode:
            self._backend = PynputBackend(key)
        else:
            self._backend = NullBackend()

        self._triggered_event = threading.Event()
        self._started = False
        self._stopped = False
        self._last_error: BaseException | None = None
        self._lock = threading.Lock()

    def _wrapped_callback(self) -> None:
        with self._lock:
            if self._triggered_event.is_set():
                return
            self._triggered_event.set()

        try:
            self._user_callback()
        except BaseException as exc:  # noqa: BLE001
            self._last_error = exc

    def start(self) -> None:
        """Start the kill switch backend. Idempotent."""
        with self._lock:
            if self._started:
                return
            self._started = True

        self._backend.start(self._wrapped_callback)

    def stop(self) -> None:
        """Stop the kill switch backend. Idempotent."""
        with self._lock:
            if self._stopped:
                return
            self._stopped = True

        self._backend.stop()

    def is_triggered(self) -> bool:
        """Return True if kill switch has been triggered."""
        return self._triggered_event.is_set()

    def last_error(self) -> BaseException | None:
        """Return exception raised by on_trigger callback if any."""
        return self._last_error
