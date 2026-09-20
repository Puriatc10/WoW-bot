"""Tests for the execution mode gate (wow_bot.mode)."""

import json
import signal
import socket
from collections.abc import Generator
from dataclasses import FrozenInstanceError
from pathlib import Path
from typing import Any

import pytest

import wow_bot.mode as mode_module
from wow_bot.config import Config
from wow_bot.mode import ModeContext, ModeError, enter_mode
from wow_bot.safety import IsolationViolation, SafetyLayer
from wow_bot.session import Session


@pytest.fixture(autouse=True)
def restore_signals() -> Generator[None, None, None]:
    """Save and restore SIGTERM and SIGINT signal handlers around tests."""
    prev_sigterm = signal.getsignal(signal.SIGTERM)
    prev_sigint = signal.getsignal(signal.SIGINT)
    try:
        yield
    finally:
        signal.signal(signal.SIGTERM, prev_sigterm)
        signal.signal(signal.SIGINT, prev_sigint)


def _find_unused_port() -> int:
    """Find an unused TCP port on localhost."""
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind(("127.0.0.1", 0))
    port = int(sock.getsockname()[1])
    sock.close()
    return port


def _create_config(
    tmp_path: Path,
    *,
    lab_mode: bool = False,
    dry_run: bool = True,
    isolation_sentinel: str = "127.0.0.1:65535",
    server_allowlist: tuple[str, ...] = ("192.168.1.50:8085",),
) -> Config:
    """Helper to construct a valid Config using tmp_path as session_root."""
    return Config(
        lab_mode=lab_mode,
        server_allowlist=server_allowlist,
        isolation_sentinel=isolation_sentinel,
        kill_switch_key="F12",
        session_root=tmp_path / "sessions",
        dry_run=dry_run,
        max_session_seconds=3600,
        log_level="INFO",
    )


def _read_events(session: Session) -> list[dict[str, Any]]:
    """Helper to read events written to events.jsonl."""
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def test_mock_mode_success(tmp_path: Path) -> None:
    """Acceptance 1: MOCK mode with dry_run=True returns ModeContext with mode='MOCK', dry_run=True, SafetyLayer."""
    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        ctx = enter_mode(config, session)
        assert isinstance(ctx, ModeContext)
        assert ctx.mode == "MOCK"
        assert ctx.dry_run is True
        assert isinstance(ctx.safety, SafetyLayer)
        assert ctx.config == config
        assert ctx.session == session
    finally:
        session.close("test_finished")


def test_mock_mode_dry_run_false_raises(tmp_path: Path) -> None:
    """Acceptance 2: MOCK mode with dry_run=False raises ModeError."""
    config = _create_config(tmp_path, lab_mode=False, dry_run=False)
    session = Session.start(config)
    try:
        with pytest.raises(ModeError, match="MOCK_MODE requires dry_run=True"):
            enter_mode(config, session)
    finally:
        session.close("test_finished")


def test_mock_mode_does_not_check_isolation(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance 3: MOCK mode does NOT call safety.check_isolation."""
    def _raise_isolation() -> None:
        raise RuntimeError("check_isolation should not be called in MOCK mode")

    monkeypatch.setattr(SafetyLayer, "check_isolation", _raise_isolation)

    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        ctx = enter_mode(config, session)
        assert ctx.mode == "MOCK"
    finally:
        session.close("test_finished")


def test_lab_mode_dry_run_true_raises(tmp_path: Path) -> None:
    """Acceptance 4: LAB mode with dry_run=True raises ModeError."""
    config = _create_config(tmp_path, lab_mode=True, dry_run=True)
    session = Session.start(config)
    try:
        with pytest.raises(ModeError, match="LAB_MODE requires dry_run=False"):
            enter_mode(config, session)
    finally:
        session.close("test_finished")


def test_lab_mode_unreachable_sentinel_success(tmp_path: Path) -> None:
    """Acceptance 5: LAB mode with dry_run=False and unreachable sentinel returns ModeContext."""
    unused_port = _find_unused_port()
    config = _create_config(
        tmp_path,
        lab_mode=True,
        dry_run=False,
        isolation_sentinel=f"127.0.0.1:{unused_port}",
    )
    session = Session.start(config)
    try:
        ctx = enter_mode(config, session)
        assert isinstance(ctx, ModeContext)
        assert ctx.mode == "LAB"
        assert ctx.dry_run is False
        assert isinstance(ctx.safety, SafetyLayer)
    finally:
        session.close("test_finished")


def test_lab_mode_reachable_sentinel_raises_isolation_violation(tmp_path: Path) -> None:
    """Acceptance 6: LAB mode with reachable sentinel raises IsolationViolation and does NOT return ModeContext."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    try:
        config = _create_config(
            tmp_path,
            lab_mode=True,
            dry_run=False,
            isolation_sentinel=f"127.0.0.1:{port}",
        )
        session = Session.start(config)
        try:
            with pytest.raises(IsolationViolation):
                enter_mode(config, session)
        finally:
            session.close("test_finished")
    finally:
        listener.close()


def test_lab_mode_emits_mode_entered_event_with_allowlist(tmp_path: Path) -> None:
    """Acceptance 7: LAB mode emits mode_entered event containing allowlist as a list."""
    unused_port = _find_unused_port()
    allowlist = ("192.168.1.50:8085", "10.0.0.1:3724")
    config = _create_config(
        tmp_path,
        lab_mode=True,
        dry_run=False,
        isolation_sentinel=f"127.0.0.1:{unused_port}",
        server_allowlist=allowlist,
    )
    session = Session.start(config)
    try:
        enter_mode(config, session)
        events = _read_events(session)
        mode_events = [e for e in events if e.get("event") == "mode_entered"]
        assert len(mode_events) == 1
        assert mode_events[0]["mode"] == "LAB"
        assert mode_events[0]["allowlist"] == list(allowlist)
    finally:
        session.close("test_finished")


def test_mock_mode_emits_mode_entered_event_without_allowlist(tmp_path: Path) -> None:
    """Acceptance 8: MOCK mode emits mode_entered event with mode='MOCK' and no allowlist key."""
    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        enter_mode(config, session)
        events = _read_events(session)
        mode_events = [e for e in events if e.get("event") == "mode_entered"]
        assert len(mode_events) == 1
        assert mode_events[0]["mode"] == "MOCK"
        assert "allowlist" not in mode_events[0]
    finally:
        session.close("test_finished")


def test_safety_arm_called_exactly_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Acceptance 9: safety.arm() is called exactly once per enter_mode call."""
    arm_calls = 0
    orig_arm = SafetyLayer.arm

    def _counted_arm(self: SafetyLayer) -> None:
        nonlocal arm_calls
        arm_calls += 1
        orig_arm(self)

    monkeypatch.setattr(SafetyLayer, "arm", _counted_arm)

    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        enter_mode(config, session)
        assert arm_calls == 1
    finally:
        session.close("test_finished")


def test_enter_mode_does_not_close_session_on_success(tmp_path: Path) -> None:
    """Acceptance 10: enter_mode does NOT close session on success (session remains usable)."""
    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        ctx = enter_mode(config, session)
        # Verify session is still open by writing an event
        ctx.session.write_event({"event": "post_enter_test"})
        events = _read_events(session)
        event_names = [e.get("event") for e in events]
        assert "post_enter_test" in event_names
    finally:
        session.close("test_finished")


def test_enter_mode_does_not_close_session_on_failure(tmp_path: Path) -> None:
    """Acceptance 11: enter_mode does NOT close session on failure (IsolationViolation case)."""
    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    port = int(listener.getsockname()[1])

    try:
        config = _create_config(
            tmp_path,
            lab_mode=True,
            dry_run=False,
            isolation_sentinel=f"127.0.0.1:{port}",
        )
        session = Session.start(config)
        try:
            with pytest.raises(IsolationViolation):
                enter_mode(config, session)
            # Verify session can still be written to and closed manually without error
            session.write_event({"event": "failure_handled"})
            events = _read_events(session)
            assert any(e.get("event") == "failure_handled" for e in events)
        finally:
            session.close("manual_cleanup")
    finally:
        listener.close()


def test_setup_logging_called_exactly_once_with_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Acceptance 12: setup_logging is called exactly once with (session, config.log_level)."""
    calls: list[tuple[Any, Any]] = []

    def _mock_setup_logging(session: Session, level: str) -> None:
        calls.append((session, level))

    monkeypatch.setattr(mode_module, "setup_logging", _mock_setup_logging)

    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        enter_mode(config, session)
        assert len(calls) == 1
        assert calls[0] == (session, config.log_level)
    finally:
        session.close("test_finished")


def test_mode_context_is_frozen(tmp_path: Path) -> None:
    """Acceptance 13: ModeContext is frozen: assigning to a field raises."""
    config = _create_config(tmp_path, lab_mode=False, dry_run=True)
    session = Session.start(config)
    try:
        ctx = enter_mode(config, session)
        with pytest.raises(FrozenInstanceError):
            ctx.mode = "LAB"  # type: ignore[misc]
    finally:
        session.close("test_finished")
