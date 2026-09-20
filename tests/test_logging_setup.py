"""Unit tests for wow_bot.logging_setup."""

import logging
from collections.abc import Generator
from datetime import datetime
from pathlib import Path

import pytest

from wow_bot.config import Config
from wow_bot.logging_setup import log_event, setup_logging
from wow_bot.session import Session


@pytest.fixture
def restore_root_logger() -> Generator[None, None, None]:
    """Save and restore root logger handlers and level."""
    root = logging.getLogger()
    original_handlers = list(root.handlers)
    original_level = root.level
    try:
        yield
    finally:
        for h in list(root.handlers):
            root.removeHandler(h)
            h.close()
        for h in original_handlers:
            root.addHandler(h)
        root.setLevel(original_level)


def _make_session(tmp_path: Path, mode: str = "MOCK") -> Session:
    lab_mode = mode == "LAB"
    config = Config(
        lab_mode=lab_mode,
        server_allowlist=("127.0.0.1:8080",) if lab_mode else (),
        isolation_sentinel="127.0.0.1:8080",
        kill_switch_key="F12",
        session_root=tmp_path,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(config)


def test_invalid_level_raises_value_error(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    with pytest.raises(ValueError, match="Invalid log level 'VERBOSE'"):
        setup_logging(session, "VERBOSE")


def test_log_event_produces_json_line(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("test_module")

    log_event(logger, logging.INFO, "hello", {"key": "value"})

    app_log = session.path / "app.log"
    assert app_log.exists()
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    import json

    data = json.loads(lines[0])
    assert set(data.keys()) == {"ts", "level", "module", "event", "payload"}
    assert data["level"] == "INFO"
    assert data["module"] == "test_module"
    assert data["event"] == "hello"
    assert data["payload"] == {"key": "value"}


def test_payload_defaults_to_dict(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("test_module")

    log_event(logger, logging.INFO, "no payload event")

    app_log = session.path / "app.log"
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    import json

    data = json.loads(lines[0])
    assert data["payload"] == {}


def test_ts_format_iso8601_utc(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("test_module")

    log_event(logger, logging.INFO, "ts check")

    app_log = session.path / "app.log"
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    import json

    data = json.loads(lines[0])
    ts = data["ts"]
    assert ts.endswith("Z")

    # ISO 8601 parsing without trailing Z
    parsed = datetime.fromisoformat(ts[:-1])
    assert parsed.year >= 2026


def test_unicode_preservation(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("unicode_logger")

    persian_text = "سلام دنیا 🚀"
    log_event(logger, logging.INFO, persian_text, {"greeting": persian_text})

    app_log = session.path / "app.log"
    raw_content = app_log.read_text(encoding="utf-8")
    assert "\\u" not in raw_content
    assert persian_text in raw_content


def test_idempotency_no_duplicate_handlers(
    tmp_path: Path, restore_root_logger: None
) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    root = logging.getLogger()
    assert len(root.handlers) == 2

    setup_logging(session, "INFO")
    assert len(root.handlers) == 2


def test_console_handler_level_mock(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path, mode="MOCK")
    setup_logging(session, "INFO")
    root = logging.getLogger()

    console_handler = next(
        h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console_handler.level == logging.INFO


def test_console_handler_level_lab(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path, mode="LAB")
    setup_logging(session, "INFO")
    root = logging.getLogger()

    console_handler = next(
        h for h in root.handlers if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler)
    )
    assert console_handler.level == logging.WARNING


def test_file_handler_level(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "WARNING")
    root = logging.getLogger()

    file_handler = next(h for h in root.handlers if isinstance(h, logging.FileHandler))
    assert file_handler.level == logging.WARNING


def test_level_filtering(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("filter_test")

    log_event(logger, logging.DEBUG, "debug message")
    log_event(logger, logging.ERROR, "error message")

    app_log = session.path / "app.log"
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1

    import json

    data = json.loads(lines[0])
    assert data["event"] == "error message"
    assert data["level"] == "ERROR"


def test_non_dict_payload_wrapping(tmp_path: Path, restore_root_logger: None) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("non_dict_logger")

    # Pass non-dict via log_event helper
    log_event(logger, logging.INFO, "string payload", "raw_payload_string")  # type: ignore[arg-type]

    # Also test direct logger call with extra payload non-dict
    logger.info("direct call", extra={"payload": 12345})

    app_log = session.path / "app.log"
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2

    import json

    data1 = json.loads(lines[0])
    assert data1["payload"] == {"_raw": "raw_payload_string"}

    data2 = json.loads(lines[1])
    assert data2["payload"] == {"_raw": "12345"}


def test_module_field_matches_logger_name(
    tmp_path: Path, restore_root_logger: None
) -> None:
    session = _make_session(tmp_path)
    setup_logging(session, "INFO")
    logger = logging.getLogger("wow_bot.custom.submodule")

    log_event(logger, logging.INFO, "submodule event")

    app_log = session.path / "app.log"
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    import json

    data = json.loads(lines[0])
    assert data["module"] == "wow_bot.custom.submodule"


def test_path_with_space(tmp_path: Path, restore_root_logger: None) -> None:
    space_dir = tmp_path / "session root with spaces"
    space_dir.mkdir()
    config = Config(
        lab_mode=False,
        server_allowlist=(),
        isolation_sentinel="127.0.0.1:8080",
        kill_switch_key="F12",
        session_root=space_dir,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(config)

    setup_logging(session, "INFO")
    logger = logging.getLogger("space_logger")

    log_event(logger, logging.INFO, "space path event")

    app_log = session.path / "app.log"
    assert app_log.exists()
    lines = app_log.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
