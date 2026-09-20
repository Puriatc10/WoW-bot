"""Focus manager and window focus tracking for actuation."""

import threading
import time
from collections.abc import Callable
from types import TracebackType
from typing import Protocol, Self, runtime_checkable

from wow_bot.actuation.backends.focus_null import NullFocusBackend


class FocusLost(Exception):
    """Raised when an operation requires window focus but focus is lost."""


class FocusBackendError(Exception):
    """Raised when a focus backend fails to initialize or operate."""


@runtime_checkable
class FocusBackend(Protocol):
    """Protocol for window focus tracking backends."""

    def find_window(self, title_substring: str) -> object | None: ...

    def get_foreground_handle(self) -> object | None: ...

    def same_window(self, a: object, b: object) -> bool: ...

    def close(self) -> None: ...


class FocusManager:
    """Monitors game window focus state and notifies listeners on transitions."""

    def __init__(
        self,
        window_title: str,
        *,
        poll_hz: float = 20.0,
        backend: FocusBackend | None = None,
        on_focus_lost: Callable[[], None] | None = None,
        on_focus_gained: Callable[[], None] | None = None,
    ) -> None:
        if not window_title or not window_title.strip():
            raise ValueError("window_title cannot be empty or whitespace")

        if poll_hz <= 0.0 or poll_hz > 100.0:
            raise ValueError("poll_hz must be > 0 and <= 100")

        self._window_title = window_title
        self._poll_hz = poll_hz
        self._backend: FocusBackend = backend if backend is not None else NullFocusBackend()

        target = self._backend.find_window(window_title)
        if target is None:
            raise FocusBackendError(f"Window with title '{window_title}' not found")
        self._target_handle: object = target

        self._listeners: list[Callable[[bool], None]] = []
        if on_focus_lost is not None:
            lost_cb = on_focus_lost

            def _lost_adapter(focused: bool) -> None:
                if not focused:
                    lost_cb()

            self._listeners.append(_lost_adapter)

        if on_focus_gained is not None:
            gained_cb = on_focus_gained

            def _gained_adapter(focused: bool) -> None:
                if focused:
                    gained_cb()

            self._listeners.append(_gained_adapter)

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Determine initial cached focus state
        fg_handle = self._backend.get_foreground_handle()
        initial_focused = self._backend.same_window(fg_handle, self._target_handle)
        self._is_focused = initial_focused

        self._last_error: BaseException | None = None
        self._poll_count = 0

    def target_handle(self) -> object:
        """Returns the resolved target window handle."""
        return self._target_handle

    def is_focused(self) -> bool:
        """Returns the cached last-known focus state.

        Non-blocking and cheap. Does not query backend.
        """
        with self._lock:
            return self._is_focused

    def assert_focused(self) -> None:
        """Raises FocusLost if currently not focused."""
        if not self.is_focused():
            raise FocusLost(f"Window '{self._window_title}' is not focused")

    def add_listener(self, callback: Callable[[bool], None]) -> None:
        """Registers a listener that is called with True (gained) or False (lost)."""
        with self._lock:
            self._listeners.append(callback)

    def last_error(self) -> BaseException | None:
        """Returns the last exception caught inside the polling thread, or None."""
        with self._lock:
            return self._last_error

    def poll_count(self) -> int:
        """Returns the number of successful polls since start."""
        with self._lock:
            return self._poll_count

    def start(self) -> None:
        """Starts the background polling thread. Idempotent."""
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run_polling_loop,
                name=f"FocusManager-{self._window_title}",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        """Stops background polling thread and closes backend. Idempotent.

        Safe to call from inside polling thread itself. Safe before start.
        """
        self._stop_event.set()

        thread_to_join: threading.Thread | None = None
        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and threading.current_thread() is not self._thread
            ):
                thread_to_join = self._thread
            self._thread = None

        if thread_to_join is not None:
            thread_to_join.join(timeout=0.05)

        self._backend.close()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    def _run_polling_loop(self) -> None:
        interval = 1.0 / self._poll_hz
        while not self._stop_event.is_set():
            start_time = time.monotonic()
            try:
                fg_handle = self._backend.get_foreground_handle()
                currently_focused = self._backend.same_window(fg_handle, self._target_handle)

                state_changed = False
                new_state = False
                with self._lock:
                    self._poll_count += 1
                    if currently_focused != self._is_focused:
                        self._is_focused = currently_focused
                        state_changed = True
                        new_state = currently_focused

                if state_changed:
                    self._notify_listeners(new_state)

            except BaseException as exc:  # noqa: BLE001
                with self._lock:
                    self._last_error = exc

            elapsed = time.monotonic() - start_time
            sleep_time = interval - elapsed
            if sleep_time > 0:
                self._stop_event.wait(timeout=sleep_time)

    def _notify_listeners(self, focused: bool) -> None:
        with self._lock:
            listeners = list(self._listeners)

        for listener in listeners:
            try:
                listener(focused)
            except Exception:  # noqa: BLE001, S110
                # Swallowed per contract; caller will log if attached to session
                pass
