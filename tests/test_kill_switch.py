"""Tests for physical kill switch (wow_bot.kill_switch)."""

from __future__ import annotations

import sys
import threading
from collections.abc import Callable
from unittest.mock import MagicMock

import pytest

from wow_bot.kill_switch import (
    KillSwitch,
    KillSwitchError,
    NullBackend,
    PynputBackend,
)


class FakeBackend:
    """Fake kill switch backend for testing."""

    def __init__(self) -> None:
        self.callback: Callable[[], None] | None = None
        self.start_calls = 0
        self.stop_calls = 0

    def start(self, on_trigger: Callable[[], None]) -> None:
        self.start_calls += 1
        self.callback = on_trigger

    def stop(self) -> None:
        self.stop_calls += 1

    def fire(self) -> None:
        if self.callback is not None:
            self.callback()


def test_null_backend_never_invokes_callback() -> None:
    called = False

    def cb() -> None:
        nonlocal called
        called = True

    backend = NullBackend()
    backend.start(cb)
    backend.stop()
    assert not called


def test_kill_switch_empty_key_raises() -> None:
    with pytest.raises(KillSwitchError, match="non-empty"):
        KillSwitch("", lambda: None)


def test_kill_switch_whitespace_key_raises() -> None:
    for whitespace in [" ", "   ", "\t", "\n", " \t\n "]:
        with pytest.raises(KillSwitchError, match="non-empty"):
            KillSwitch(whitespace, lambda: None)


def test_fake_backend_fire_calls_on_trigger_once() -> None:
    calls = 0

    def cb() -> None:
        nonlocal calls
        calls += 1

    backend = FakeBackend()
    ks = KillSwitch("f12", cb, backend=backend)
    ks.start()
    backend.fire()
    assert calls == 1


def test_fake_backend_fire_multiple_times_calls_on_trigger_once() -> None:
    calls = 0

    def cb() -> None:
        nonlocal calls
        calls += 1

    backend = FakeBackend()
    ks = KillSwitch("f12", cb, backend=backend)
    ks.start()
    backend.fire()
    backend.fire()
    backend.fire()
    assert calls == 1


def test_start_is_idempotent() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, backend=backend)
    ks.start()
    ks.start()
    assert backend.start_calls == 1


def test_stop_is_idempotent() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, backend=backend)
    ks.start()
    ks.stop()
    ks.stop()
    assert backend.stop_calls == 1


def test_stop_before_start_does_not_raise() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, backend=backend)
    ks.stop()
    assert backend.stop_calls == 1


def test_stop_after_trigger_does_not_raise() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, backend=backend)
    ks.start()
    backend.fire()
    ks.stop()
    assert backend.stop_calls == 1


def test_is_triggered_lifecycle() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, backend=backend)
    assert not ks.is_triggered()
    ks.start()
    assert not ks.is_triggered()
    backend.fire()
    assert ks.is_triggered()


def test_callback_raising_exception_swallowed_and_stored() -> None:
    exc = RuntimeError("callback failed")

    def cb() -> None:
        raise exc

    backend = FakeBackend()
    ks = KillSwitch("f12", cb, backend=backend)
    ks.start()
    backend.fire()  # Must not raise out to caller of fire()
    assert ks.last_error() is exc


def test_is_triggered_true_after_callback_exception() -> None:
    def cb() -> None:
        raise ValueError("error")

    backend = FakeBackend()
    ks = KillSwitch("f12", cb, backend=backend)
    ks.start()
    backend.fire()
    assert ks.is_triggered()


def test_lab_mode_without_pynput_raises_kill_switch_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Simulation strategy: We simulate pynput being unavailable in lab_mode by setting
    # sys.modules entries for "pynput" and "pynput.keyboard" to None in pytest's monkeypatch.
    # In Python's import machinery, placing None in sys.modules causes any import attempt for
    # that module to raise an ImportError, triggering PynputBackend.__init__'s exception handler.
    monkeypatch.setitem(sys.modules, "pynput", None)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", None)

    with pytest.raises(KillSwitchError, match="pynput"):
        KillSwitch("f12", lambda: None, lab_mode=True)


def test_explicit_backend_ignores_lab_mode() -> None:
    backend = FakeBackend()
    ks = KillSwitch("f12", lambda: None, lab_mode=True, backend=backend)
    ks.start()
    backend.fire()
    assert ks.is_triggered()


def test_pynput_backend_key_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Test PynputBackend key validation logic using a mocked pynput module."""
    mock_pynput = MagicMock()
    mock_pynput.keyboard.Key.f1 = "MOCK_KEY_F1"
    mock_pynput.keyboard.Key.esc = "MOCK_KEY_ESC"
    mock_pynput.keyboard.Key.space = "MOCK_KEY_SPACE"
    mock_pynput.keyboard.KeyCode.from_char = lambda c: f"MOCK_CHAR_{c}"

    monkeypatch.setitem(sys.modules, "pynput", mock_pynput)
    monkeypatch.setitem(sys.modules, "pynput.keyboard", mock_pynput.keyboard)

    # Valid special keys
    for valid_special in ["f1", "F12", "esc", "space", "tab", "enter"]:
        backend = PynputBackend(valid_special)
        assert backend is not None

    # Valid character keys
    for valid_char in ["a", "Z", "1", "$"]:
        backend = PynputBackend(valid_char)
        assert backend is not None

    # Invalid keys
    for invalid_key in ["f13", "ctrl+f12", "invalid_key", "foo"]:
        with pytest.raises(KillSwitchError, match="Unsupported key string format"):
            PynputBackend(invalid_key)


def test_concurrent_triggers_call_callback_at_most_once() -> None:
    calls = 0
    lock = threading.Lock()

    def cb() -> None:
        nonlocal calls
        with lock:
            calls += 1

    backend = FakeBackend()
    ks = KillSwitch("f12", cb, backend=backend)
    ks.start()

    threads = [threading.Thread(target=backend.fire) for _ in range(10)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert calls == 1
    assert ks.is_triggered()
