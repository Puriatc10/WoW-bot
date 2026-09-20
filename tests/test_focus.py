"""Unit tests for FocusManager and focus backends."""

import threading
import time
from typing import Any

import pytest

from wow_bot.actuation.focus import (
    FocusBackendError,
    FocusLost,
    FocusManager,
)


class FakeBackend:
    """Controllable fake focus backend for testing FocusManager."""

    def __init__(self, target: str = "target-handle") -> None:
        self.target: object = target
        self.current_fg: object = target
        self.closed: bool = False
        self.call_count: int = 0

    def find_window(self, title_substring: str) -> object | None:
        self.call_count += 1
        if "nonexistent" in title_substring:
            return None
        return self.target

    def get_foreground_handle(self) -> object | None:
        self.call_count += 1
        return self.current_fg

    def same_window(self, a: object, b: object) -> bool:
        return a == b

    def close(self) -> None:
        self.closed = True


class SlowBackend(FakeBackend):
    """Backend that simulates a blocked/slow poll."""

    def __init__(self, sleep_s: float = 0.5) -> None:
        super().__init__()
        self.sleep_s = sleep_s

    def get_foreground_handle(self) -> object | None:
        time.sleep(self.sleep_s)
        return super().get_foreground_handle()


class ErrorBackend(FakeBackend):
    """Backend that raises an error during polling."""

    def __init__(self) -> None:
        super().__init__()
        self.should_raise: bool = False

    def get_foreground_handle(self) -> object | None:
        if self.should_raise:
            raise RuntimeError("Backend query failed")
        return super().get_foreground_handle()


def wait_until(
    condition: Any,
    timeout: float = 1.0,
    interval: float = 0.01,
) -> bool:
    """Helper to poll for a condition with a bounded wait loop."""
    start = time.monotonic()
    while time.monotonic() - start < timeout:
        if condition():
            return True
        time.sleep(interval)
    return bool(condition())


def test_empty_window_title_raises_value_error() -> None:
    """Constructing FocusManager with empty or whitespace window_title raises ValueError."""
    with pytest.raises(ValueError, match="window_title cannot be empty or whitespace"):
        FocusManager("")

    with pytest.raises(ValueError, match="window_title cannot be empty or whitespace"):
        FocusManager("   ")


def test_invalid_poll_hz_raises_value_error() -> None:
    """Constructing FocusManager with poll_hz <= 0 or > 100 raises ValueError."""
    with pytest.raises(ValueError, match="poll_hz must be > 0 and <= 100"):
        FocusManager("WoW", poll_hz=0.0)

    with pytest.raises(ValueError, match="poll_hz must be > 0 and <= 100"):
        FocusManager("WoW", poll_hz=-5.0)

    with pytest.raises(ValueError, match="poll_hz must be > 0 and <= 100"):
        FocusManager("WoW", poll_hz=100.1)


def test_null_focus_backend_reports_focused() -> None:
    """NullFocusBackend reports focused=True immediately after construction."""
    mgr = FocusManager("World of Warcraft")
    assert mgr.is_focused() is True
    assert mgr.target_handle() is not None
    assert repr(mgr.target_handle()) == "null-window"


def test_fake_backend_reports_focused_when_matching() -> None:
    """FocusManager with FakeBackend reports is_focused() == True when fg == target."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", backend=backend)
    assert mgr.is_focused() is True


def test_focus_flips_to_false_when_fg_changes() -> None:
    """FocusManager flips is_focused() to False within 200 ms when fake foreground changes."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)
    mgr.start()
    try:
        assert mgr.is_focused() is True
        backend.current_fg = "other-window"
        assert wait_until(lambda: not mgr.is_focused(), timeout=0.2)
    finally:
        mgr.stop()


def test_on_focus_lost_fires_once_per_transition() -> None:
    """on_focus_lost callback fires exactly once per transition to lost."""
    lost_calls = 0

    def on_lost() -> None:
        nonlocal lost_calls
        lost_calls += 1

    backend = FakeBackend()
    mgr = FocusManager(
        "World of Warcraft",
        poll_hz=50.0,
        backend=backend,
        on_focus_lost=on_lost,
    )
    mgr.start()
    try:
        backend.current_fg = "other-window"
        assert wait_until(lambda: not mgr.is_focused(), timeout=0.2)
        # Wait extra ticks to ensure it doesn't fire continuously
        time.sleep(0.1)
        assert lost_calls == 1
    finally:
        mgr.stop()


def test_on_focus_gained_fires_once_per_transition() -> None:
    """on_focus_gained callback fires exactly once per transition to gained."""
    gained_calls = 0

    def on_gained() -> None:
        nonlocal gained_calls
        gained_calls += 1

    backend = FakeBackend()
    backend.current_fg = "other-window"  # Start unfocused
    mgr = FocusManager(
        "World of Warcraft",
        poll_hz=50.0,
        backend=backend,
        on_focus_gained=on_gained,
    )
    mgr.start()
    try:
        assert mgr.is_focused() is False
        backend.current_fg = "target-handle"
        assert wait_until(lambda: mgr.is_focused(), timeout=0.2)
        time.sleep(0.1)
        assert gained_calls == 1
    finally:
        mgr.stop()


def test_add_listener_registration_order() -> None:
    """add_listener receives True/False transitions in registration order."""
    events: list[tuple[str, bool]] = []

    def l1(val: bool) -> None:
        events.append(("l1", val))

    def l2(val: bool) -> None:
        events.append(("l2", val))

    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)
    mgr.add_listener(l1)
    mgr.add_listener(l2)
    mgr.start()
    try:
        backend.current_fg = "other-window"
        assert wait_until(lambda: not mgr.is_focused(), timeout=0.2)
        assert events == [("l1", False), ("l2", False)]
    finally:
        mgr.stop()


def test_listener_exception_does_not_prevent_subsequent_listeners() -> None:
    """A listener that raises does not prevent subsequent listeners from being called."""
    calls: list[str] = []

    def raising_listener(val: bool) -> None:
        calls.append("raising")
        raise RuntimeError("Listener error")

    def normal_listener(val: bool) -> None:
        calls.append("normal")

    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)
    mgr.add_listener(raising_listener)
    mgr.add_listener(normal_listener)
    mgr.start()
    try:
        backend.current_fg = "other-window"
        assert wait_until(lambda: not mgr.is_focused(), timeout=0.2)
        assert calls == ["raising", "normal"]
    finally:
        mgr.stop()


def test_assert_focused() -> None:
    """assert_focused raises FocusLost when not focused and returns None when focused."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", backend=backend)
    mgr.assert_focused()  # Focused -> returns None without exception

    backend.current_fg = "other-window"
    # Update state via direct poll or start
    mgr.start()
    try:
        assert wait_until(lambda: not mgr.is_focused(), timeout=0.2)
        with pytest.raises(FocusLost, match="World of Warcraft"):
            mgr.assert_focused()
    finally:
        mgr.stop()


def test_start_is_idempotent() -> None:
    """start() is idempotent: calling twice does not create a second thread."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)

    active_before = threading.active_count()
    mgr.start()
    active_after_first = threading.active_count()
    mgr.start()
    active_after_second = threading.active_count()

    try:
        assert active_after_first == active_before + 1
        assert active_after_second == active_after_first
    finally:
        mgr.stop()


def test_stop_is_idempotent_and_safe_before_start() -> None:
    """stop() is idempotent and safe before start."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", backend=backend)

    # Safe before start
    mgr.stop()
    assert backend.closed is True

    # Idempotent
    mgr.stop()


def test_stop_completes_promptly_with_slow_backend() -> None:
    """stop() completes within 200 ms even if the backend poll is slow."""
    backend = SlowBackend(sleep_s=0.5)
    mgr = FocusManager("World of Warcraft", poll_hz=10.0, backend=backend)
    mgr.start()
    time.sleep(0.05)  # Let thread enter get_foreground_handle sleep

    start_time = time.monotonic()
    mgr.stop()
    elapsed = time.monotonic() - start_time

    assert elapsed < 0.2


def test_backend_exception_stored_in_last_error_and_thread_survives() -> None:
    """Exception inside backend poll stored in last_error(), state unchanged, thread survives."""
    backend = ErrorBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)
    mgr.start()
    try:
        assert mgr.is_focused() is True
        backend.should_raise = True

        assert wait_until(lambda: mgr.last_error() is not None, timeout=0.2)
        assert isinstance(mgr.last_error(), RuntimeError)
        # Focus state left unchanged
        assert mgr.is_focused() is True

        # Thread still running and poll_count increases
        count_before = mgr.poll_count()
        backend.should_raise = False
        assert wait_until(lambda: mgr.poll_count() > count_before, timeout=0.2)
    finally:
        mgr.stop()


def test_poll_count_increments_and_stops() -> None:
    """poll_count increments while running and stops incrementing after stop()."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)

    mgr.start()
    assert wait_until(lambda: mgr.poll_count() >= 3, timeout=0.2)
    mgr.stop()

    count_at_stop = mgr.poll_count()
    time.sleep(0.05)
    assert mgr.poll_count() == count_at_stop


def test_context_manager() -> None:
    """Context manager starts on enter and stops on exit."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", poll_hz=50.0, backend=backend)

    with mgr as m:
        assert m is mgr
        assert wait_until(lambda: mgr.poll_count() > 0, timeout=0.2)

    assert backend.closed is True


def test_target_handle_returns_resolved_handle() -> None:
    """target_handle() returns the resolved handle and raises no exception for NullFocusBackend."""
    null_mgr = FocusManager("World of Warcraft")
    assert null_mgr.target_handle() is not None

    fake_backend = FakeBackend(target="handle-123")
    fake_mgr = FocusManager("World of Warcraft", backend=fake_backend)
    assert fake_mgr.target_handle() == "handle-123"


def test_polling_thread_is_daemon() -> None:
    """Polling thread is a daemon thread."""
    backend = FakeBackend()
    mgr = FocusManager("World of Warcraft", backend=backend)
    mgr.start()
    try:
        assert mgr._thread is not None
        assert mgr._thread.daemon is True
    finally:
        mgr.stop()


def test_nonexistent_window_raises_focus_backend_error() -> None:
    """FocusManager raises FocusBackendError when non-null backend fails to find window."""
    backend = FakeBackend()
    with pytest.raises(FocusBackendError, match="Window with title 'nonexistent' not found"):
        FocusManager("nonexistent", backend=backend)
