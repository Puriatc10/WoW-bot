"""Tests for the session management module.

Validates session creation, metadata serialization, append-only event logging,
traceback capture, idempotency, atomic state updates, context management, and
permission settings against task acceptance criteria.
"""

import json
import os
import re
import stat
import threading
from pathlib import Path

import pytest

from wow_bot.config import Config, ConfigError
from wow_bot.session import Session, SessionClosed


def make_test_config(session_root: Path, lab_mode: bool = False) -> Config:
    """Helper to construct a valid Config instance with a custom session_root."""
    return Config(
        lab_mode=lab_mode,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=session_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def test_distinct_session_ids_same_second(tmp_path: Path) -> None:
    """Acceptance 1: Two sessions started in the same second get distinct session_id values."""
    cfg = make_test_config(tmp_path)
    s1 = Session.start(cfg)
    s2 = Session.start(cfg)
    assert s1.session_id != s2.session_id


def test_session_id_regex_format(tmp_path: Path) -> None:
    """Acceptance 2: session_id matches regex ^\\d{8}-\\d{6}-[0-9a-f]{8}$."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    pattern = r"^\d{8}-\d{6}-[0-9a-f]{8}$"
    assert re.match(pattern, s.session_id) is not None


def test_events_append_in_order(tmp_path: Path) -> None:
    """Acceptance 3: events append in order across multiple write_event calls."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    s.write_event({"event": "first", "seq": 1})
    s.write_event({"event": "second", "seq": 2})
    s.write_event({"event": "third", "seq": 3})

    events_file = s.path / "events.jsonl"
    lines = events_file.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 3

    parsed = [json.loads(line) for line in lines]
    assert [p["seq"] for p in parsed] == [1, 2, 3]


def test_each_line_events_jsonl_valid_json(tmp_path: Path) -> None:
    """Acceptance 4: Each line in events.jsonl is valid JSON."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    s.write_event({"a": 1, "msg": "hello"})
    s.write_event({"b": 2, "msg": "world"})

    events_file = s.path / "events.jsonl"
    for line in events_file.read_text(encoding="utf-8").strip().split("\n"):
        obj = json.loads(line)
        assert isinstance(obj, dict)


def test_write_event_injects_ts_if_missing(tmp_path: Path) -> None:
    """Acceptance 5: write_event injects 'ts' if missing."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    evt = {"name": "test_event"}
    s.write_event(evt)

    events_file = s.path / "events.jsonl"
    line = events_file.read_text(encoding="utf-8").strip()
    data = json.loads(line)
    assert "ts" in data
    assert isinstance(data["ts"], str)


def test_write_event_after_close_raises(tmp_path: Path) -> None:
    """Acceptance 6: write_event after close raises SessionClosed."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    s.close("done")
    with pytest.raises(SessionClosed):
        s.write_event({"event": "should_fail"})


def test_close_is_idempotent(tmp_path: Path) -> None:
    """Acceptance 7: close is idempotent: second call does not raise and does not modify session.json again."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    s.close("first_call")

    session_json = s.path / "session.json"
    content_after_first = session_json.read_text(encoding="utf-8")
    data_first = json.loads(content_after_first)
    assert data_first["stop_reason"] == "first_call"

    # Second call should be a no-op
    s.close("second_call")
    content_after_second = session_json.read_text(encoding="utf-8")
    data_second = json.loads(content_after_second)

    assert data_second["stop_reason"] == "first_call"
    assert content_after_first == content_after_second


def test_write_crash_produces_valid_json(tmp_path: Path) -> None:
    """Acceptance 8: write_crash produces a valid JSON file with the four keys (type, message, traceback, ts)."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    try:
        raise ValueError("Something went wrong")
    except ValueError as exc:
        s.write_crash(exc)

    crash_file = s.path / "crash.json"
    assert crash_file.exists()
    data = json.loads(crash_file.read_text(encoding="utf-8"))

    assert set(data.keys()) == {"type", "message", "traceback", "ts"}
    assert data["type"] == "ValueError"
    assert data["message"] == "Something went wrong"
    assert "ValueError: Something went wrong" in data["traceback"]


def test_write_crash_called_twice_does_not_raise(tmp_path: Path) -> None:
    """Acceptance 9: write_crash called twice does not raise."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    try:
        raise RuntimeError("Crash 1")
    except RuntimeError as exc:
        s.write_crash(exc)

    try:
        raise KeyError("Crash 2")
    except KeyError as exc:
        s.write_crash(exc)

    crash_file = s.path / "crash.json"
    data = json.loads(crash_file.read_text(encoding="utf-8"))
    assert data["type"] == "RuntimeError"


def test_context_manager_clean_exit(tmp_path: Path) -> None:
    """Acceptance 10: context manager on clean exit calls close('clean') and no crash.json exists."""
    cfg = make_test_config(tmp_path)
    with Session.start(cfg) as s:
        s.write_event({"step": "doing_work"})

    assert not (s.path / "crash.json").exists()
    session_json = json.loads((s.path / "session.json").read_text(encoding="utf-8"))
    assert session_json["stop_reason"] == "clean"
    assert "stopped_at" in session_json


def test_context_manager_exception_exit(tmp_path: Path) -> None:
    """Acceptance 11: context manager on exception writes crash.json and session.json has stop_reason='exception'."""
    cfg = make_test_config(tmp_path)
    with pytest.raises(ZeroDivisionError), Session.start(cfg) as s:
        _ = 1 / 0

    assert (s.path / "crash.json").exists()
    crash_data = json.loads((s.path / "crash.json").read_text(encoding="utf-8"))
    assert crash_data["type"] == "ZeroDivisionError"

    session_json = json.loads((s.path / "session.json").read_text(encoding="utf-8"))
    assert session_json["stop_reason"] == "exception"
    assert "stopped_at" in session_json


def test_session_json_contains_stopped_at_after_close(tmp_path: Path) -> None:
    """Acceptance 12: session.json contains stopped_at after close."""
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)
    initial = json.loads((s.path / "session.json").read_text(encoding="utf-8"))
    assert "stopped_at" not in initial

    s.close("normal")
    after_close = json.loads((s.path / "session.json").read_text(encoding="utf-8"))
    assert "stopped_at" in after_close


def test_non_writable_session_root_raises_config_error(tmp_path: Path) -> None:
    """Acceptance 13: session_root that is not writable raises ConfigError."""
    if os.name == "nt":
        pytest.skip("Permission bits check omitted on Windows.")

    unwritable = tmp_path / "read_only"
    unwritable.mkdir(mode=0o555)

    cfg = make_test_config(unwritable)
    with pytest.raises(ConfigError, match="not writable"):
        Session.start(cfg)


def test_concurrent_write_event(tmp_path: Path) -> None:
    """Acceptance 14: Concurrent write_event from two threads produces exactly expected number of well-formed lines.

    Uses threading.Barrier to synchronize start.
    """
    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)

    threads_count = 10
    events_per_thread = 20
    barrier = threading.Barrier(threads_count)

    def worker(thread_idx: int) -> None:
        barrier.wait()
        for i in range(events_per_thread):
            s.write_event({"thread": thread_idx, "i": i})

    threads = [
        threading.Thread(target=worker, args=(t,))
        for t in range(threads_count)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    events_file = s.path / "events.jsonl"
    lines = [line for line in events_file.read_text(encoding="utf-8").split("\n") if line.strip()]
    assert len(lines) == threads_count * events_per_thread

    for line in lines:
        obj = json.loads(line)
        assert "thread" in obj
        assert "i" in obj
        assert "ts" in obj


def test_file_permissions_600(tmp_path: Path) -> None:
    """Acceptance 15: session.json and events.jsonl have mode 0o600 (skip on Windows via os.name check)."""
    if os.name == "nt":
        pytest.skip("Permission mode 0o600 test skipped on Windows.")

    cfg = make_test_config(tmp_path)
    s = Session.start(cfg)

    session_json_st = (s.path / "session.json").stat().st_mode
    events_st = (s.path / "events.jsonl").stat().st_mode

    assert stat.S_IMODE(session_json_st) == 0o600
    assert stat.S_IMODE(events_st) == 0o600
