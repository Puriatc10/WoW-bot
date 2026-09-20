"""Tests for safety layer (wow_bot.safety)."""

import json
import signal
import socket
import threading
from collections.abc import Generator
from pathlib import Path

import pytest

from wow_bot.config import Config
from wow_bot.safety import (
    AllowlistViolation,
    IsolationViolation,
    SafetyError,
    SafetyLayer,
)
from wow_bot.session import Session


@pytest.fixture
def restore_signal_handlers() -> Generator[None, None, None]:
    """Fixture to save and restore SIGTERM/SIGINT handlers around tests."""
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def _make_config(
    tmp_path: Path,
    *,
    server_allowlist: tuple[str, ...] = ("127.0.0.1:8080",),
    isolation_sentinel: str = "127.0.0.1:9999",
) -> Config:
    """Helper to construct a Config instance for tests."""
    return Config(
        lab_mode=True,
        server_allowlist=server_allowlist,
        isolation_sentinel=isolation_sentinel,
        kill_switch_key="f12",
        session_root=tmp_path / "sessions",
        dry_run=True,
        max_session_seconds=60,
        log_level="INFO",
    )


def test_check_allowlist_exact_match(tmp_path: Path) -> None:
    config = _make_config(tmp_path, server_allowlist=("127.0.0.1:8080",))
    safety = SafetyLayer(config)
    safety.check_allowlist("127.0.0.1:8080")


def test_check_allowlist_bare_host(tmp_path: Path) -> None:
    config = _make_config(tmp_path, server_allowlist=("127.0.0.1",))
    safety = SafetyLayer(config)
    safety.check_allowlist("127.0.0.1:8080")
    safety.check_allowlist("127.0.0.1:9000")


def test_check_allowlist_case_insensitive(tmp_path: Path) -> None:
    config = _make_config(tmp_path, server_allowlist=("MyServer.Local:8080",))
    safety = SafetyLayer(config)
    safety.check_allowlist("myserver.local:8080")


def test_check_allowlist_unlisted_host_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path, server_allowlist=("127.0.0.1:8080",))
    safety = SafetyLayer(config)
    with pytest.raises(AllowlistViolation) as exc_info:
        safety.check_allowlist("192.168.1.1:8080")
    assert "192.168.1.1:8080" in str(exc_info.value)


def test_check_allowlist_malformed_addr_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path, server_allowlist=("127.0.0.1:8080",))
    safety = SafetyLayer(config)
    for bad_addr in ["a:b:c", "host:port:8080", "127.0.0.1:invalid", "127.0.0.1:70000", "", "  "]:
        with pytest.raises(SafetyError):
            safety.check_allowlist(bad_addr)


def test_check_isolation_reachable_sentinel_raises(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
        listener.bind(("127.0.0.1", 0))
        listener.listen(1)
        _, port = listener.getsockname()
        config = _make_config(tmp_path, isolation_sentinel=f"127.0.0.1:{port}")
        safety = SafetyLayer(config)
        with pytest.raises(IsolationViolation) as exc_info:
            safety.check_isolation()
        assert f"127.0.0.1:{port}" in str(exc_info.value)


def test_check_isolation_unreachable_sentinel_passes(tmp_path: Path) -> None:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        _, port = sock.getsockname()

    config = _make_config(tmp_path, isolation_sentinel=f"127.0.0.1:{port}")
    safety = SafetyLayer(config)
    safety.check_isolation()


def test_check_isolation_malformed_sentinel_raises(tmp_path: Path) -> None:
    for bad_sentinel in ["127.0.0.1", "127.0.0.1:abc", "a:b:c", ""]:
        config = _make_config(tmp_path, isolation_sentinel=bad_sentinel)
        safety = SafetyLayer(config)
        with pytest.raises(SafetyError):
            safety.check_isolation()


def test_is_aborted_and_abort_reason_initial(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config)
    assert not safety.is_aborted()
    assert safety.abort_reason() is None


def test_abort_with_none_session(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config, session=None)
    safety.abort("test reason")
    assert safety.is_aborted()
    assert safety.abort_reason() == "test reason"


def test_abort_with_session_emits_event(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session=session)

    safety.abort("critical failure")
    assert safety.is_aborted()
    assert safety.abort_reason() == "critical failure"

    events_file = session.path / "events.jsonl"
    with open(events_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    abort_events = [e for e in events if e.get("event") == "safety_abort"]
    assert len(abort_events) == 1
    assert abort_events[0]["payload"]["reason"] == "critical failure"


def test_abort_callback_order_and_exceptions(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session=session)

    called_order: list[int] = []

    def cb1() -> None:
        called_order.append(1)

    def cb2() -> None:
        called_order.append(2)
        raise RuntimeError("cb2 failed")

    def cb3() -> None:
        called_order.append(3)

    safety.register_kill_switch(cb1)
    safety.register_kill_switch(cb2)
    safety.register_kill_switch(cb3)

    safety.abort("trigger callbacks")

    assert called_order == [1, 2, 3]

    events_file = session.path / "events.jsonl"
    with open(events_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    failed_events = [e for e in events if e.get("event") == "safety_callback_failed"]
    assert len(failed_events) == 1
    assert "cb2 failed" in failed_events[0]["payload"]["error"]


def test_abort_idempotency(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session=session)

    count = 0

    def cb() -> None:
        nonlocal count
        count += 1

    safety.register_kill_switch(cb)

    safety.abort("first abort")
    safety.abort("second abort")

    assert count == 1
    assert safety.abort_reason() == "first abort"

    events_file = session.path / "events.jsonl"
    with open(events_file, "r", encoding="utf-8") as f:
        events = [json.loads(line) for line in f]

    abort_events = [e for e in events if e.get("event") == "safety_abort"]
    assert len(abort_events) == 1


def test_register_kill_switch_after_abort(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config)

    safety.abort("manual abort")

    called = False

    def cb() -> None:
        nonlocal called
        called = True

    safety.register_kill_switch(cb)
    assert not called


def test_arm_non_main_thread_raises(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config)

    error_raised: Exception | None = None

    def worker() -> None:
        nonlocal error_raised
        try:
            safety.arm()
        except SafetyError as exc:
            error_raised = exc

    t = threading.Thread(target=worker)
    t.start()
    t.join()

    assert isinstance(error_raised, SafetyError)
    assert "main thread" in str(error_raised)


def test_arm_disarm_lifecycle_and_signal_handler(
    tmp_path: Path, restore_signal_handlers: None
) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config)

    prev_term = signal.getsignal(signal.SIGTERM)
    prev_int = signal.getsignal(signal.SIGINT)

    safety.arm()
    safety.arm()

    handler_term = signal.getsignal(signal.SIGTERM)
    assert handler_term is not prev_term
    assert callable(handler_term)

    handler_term(signal.SIGTERM, None)
    assert safety.is_aborted()
    assert safety.abort_reason() == "signal:SIGTERM"

    safety.disarm()
    assert signal.getsignal(signal.SIGTERM) == prev_term
    assert signal.getsignal(signal.SIGINT) == prev_int

    safety.disarm()


def test_disarm_without_arm_is_noop(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    safety = SafetyLayer(config)
    safety.disarm()
