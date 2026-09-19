"""Acceptance tests for Task 0.4 — Shared Logger.

Proves the roadmap criteria:

1. ``get_logger("DYNAMICS")`` + ``log.info("test")`` prints exactly one line
   to the console sink.
2. The same call appends exactly one line to the configured log file.
3. Console output is colored; file output carries millisecond timestamps and
   the per-module context tag.

Temp dirs live under a workspace-local base rather than ``tmp_path``: the
sandboxed system TMP may be unwritable here (mirrors the convention already used
by ``tests/unit/test_config.py``). Each test gets a uniquely named subdir, so no
two runs ever collide on the same log path even though loguru keeps its file
handle open until the sinks are released.
"""

from __future__ import annotations

import re
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from loguru import logger

from wow_bot.shared.config import LoggingConfig, Settings
from wow_bot.shared.logger import configure_logging, get_logger, reset_logging


@pytest.fixture(autouse=True)
def _isolate_sinks() -> Iterator[None]:
    """Keep loguru's global state isolated between tests.

    Uses only the public ``remove()`` API: no private handler state is restored,
    so a sink belonging to one test can never leak into another. Teardown also
    releases our sinks before yielding control back.
    """
    logger.remove()
    reset_logging()
    try:
        yield
    finally:
        logger.complete()
        logger.remove()
        reset_logging()


@pytest.fixture
def local_tmp(request: pytest.FixtureRequest) -> Iterator[Path]:
    """Unique workspace-local temp dir per test.

    Named by the test node id so concurrent/repeat runs never reuse a path while
    a previous loguru file handle might still be open. Cleanup is best-effort and
    ignores locked files, which is safe because uniqueness prevents collisions.
    """
    base = Path.cwd() / ".pytest_local_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"log-{request.node.name}"
    if d.exists():
        # Remove any residue from an earlier crashed run before reusing the name.
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def _settings(tmp: Path, **overrides: Any) -> Settings:
    data: dict[str, Any] = {
        "level": "INFO",
        "log_file": str(tmp / "bot.log"),
        "rotation": "1 day",
        "retention": "7 days",
    }
    data.update(overrides)
    return Settings(logging=LoggingConfig(**data))


_MS_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d{3}")


def test_one_line_to_console_and_one_line_to_file(local_tmp: Path, capsys: Any) -> None:
    # Acceptance criteria 1 & 2 from ROADMAP task 0.4.
    settings = _settings(local_tmp)
    configure_logging(settings)

    log = get_logger("DYNAMICS")
    log.info("test")
    logger.complete()

    captured = capsys.readouterr()
    console_lines = [ln for ln in captured.err.splitlines() if ln.strip()]
    assert len(console_lines) == 1, f"expected 1 console line, got {console_lines}"
    assert "test" in console_lines[0]
    assert "[DYNAMICS]" in console_lines[0]

    file_text = (local_tmp / "bot.log").read_text(encoding="utf-8")
    file_lines = [ln for ln in file_text.splitlines() if ln.strip()]
    assert len(file_lines) == 1, f"expected 1 file line, got {file_lines}"
    assert "test" in file_lines[0]
    assert "[DYNAMICS]" in file_lines[0]


def test_console_sink_is_colorized(local_tmp: Path, capsys: Any) -> None:
    # Requirement: console output with colors. ANSI escapes must appear on the
    # console but never in the file sink.
    settings = _settings(local_tmp)
    configure_logging(settings)

    get_logger("LLM").info("hello")
    logger.complete()

    captured = capsys.readouterr()
    assert "\x1b[" in captured.err, "console output should contain ANSI color codes"

    file_text = (local_tmp / "bot.log").read_text(encoding="utf-8")
    assert "\x1b[" not in file_text, "file output must stay free of ANSI codes"


def test_millisecond_timestamps_in_both_sinks(local_tmp: Path, capsys: Any) -> None:
    # Requirement: millisecond timestamps.
    settings = _settings(local_tmp)
    configure_logging(settings)

    get_logger("EXEC").info("timed")
    logger.complete()

    captured = capsys.readouterr()
    assert _MS_PATTERN.search(captured.err), f"no ms timestamp in {captured.err!r}"

    file_text = (local_tmp / "bot.log").read_text(encoding="utf-8")
    assert _MS_PATTERN.search(file_text), f"no ms timestamp in {file_text!r}"


def test_context_tag_per_module(local_tmp: Path) -> None:
    # Requirement: a distinct context tag per module.
    settings = _settings(local_tmp)
    configure_logging(settings)

    get_logger("DYNAMICS").info("dyn-msg")
    get_logger("STRATEGIST").info("strat-msg")
    logger.complete()

    lines = (local_tmp / "bot.log").read_text(encoding="utf-8").splitlines()
    dyn = [ln for ln in lines if "dyn-msg" in ln]
    strat = [ln for ln in lines if "strat-msg" in ln]
    assert len(dyn) == 1 and "[DYNAMICS]" in dyn[0]
    assert len(strat) == 1 and "[STRATEGIST]" in strat[0]


def test_configure_logging_is_idempotent(local_tmp: Path, capsys: Any) -> None:
    # Repeated init must not duplicate sinks (exactly one line per emit).
    settings = _settings(local_tmp)
    configure_logging(settings)
    configure_logging(settings)

    get_logger("WATCHDOG").info("once")
    logger.complete()

    captured = capsys.readouterr()
    assert len([ln for ln in captured.err.splitlines() if "once" in ln]) == 1
    file_lines = (local_tmp / "bot.log").read_text(encoding="utf-8").splitlines()
    assert len([ln for ln in file_lines if "once" in ln]) == 1


def test_get_logger_configures_lazily(local_tmp: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    # Callers need not invoke configure_logging first.
    settings = _settings(local_tmp)
    monkeypatch.setattr("wow_bot.shared.config.get_settings", lambda: settings)
    reset_logging()

    get_logger("MOCKS").info("lazy")
    logger.complete()

    assert (local_tmp / "bot.log").exists()
    assert "lazy" in (local_tmp / "bot.log").read_text(encoding="utf-8")


def test_level_filtering_from_config(local_tmp: Path) -> None:
    # Tunable comes from config, not a magic number (AGENTS.md rule).
    settings = _settings(local_tmp, level="WARNING")
    configure_logging(settings)

    log = get_logger("DYNAMICS")
    log.debug("dropped-debug")
    log.info("dropped-info")
    log.warning("kept-warning")
    logger.complete()

    text = (local_tmp / "bot.log").read_text(encoding="utf-8")
    assert "dropped-debug" not in text
    assert "dropped-info" not in text
    assert "kept-warning" in text


def test_creates_missing_log_directory(local_tmp: Path) -> None:
    nested = local_tmp / "deep" / "nested" / "bot.log"
    settings = _settings(local_tmp, log_file=str(nested))
    configure_logging(settings)

    get_logger("DYNAMICS").info("nested-ok")
    logger.complete()

    assert nested.exists()
    assert "nested-ok" in nested.read_text(encoding="utf-8")


def test_reset_logging_closes_file_sink(local_tmp: Path) -> None:
    # Lifecycle rule: long-lived resources have explicit cleanup; logs survive
    # shutdown (constraint #7 — no deletion).
    settings = _settings(local_tmp)
    configure_logging(settings)
    get_logger("DYNAMICS").info("pre-shutdown")
    logger.complete()

    reset_logging()

    assert (local_tmp / "bot.log").exists()
    assert "pre-shutdown" in (local_tmp / "bot.log").read_text(encoding="utf-8")


def test_tag_with_brackets_not_double_wrapped(local_tmp: Path) -> None:
    settings = _settings(local_tmp)
    configure_logging(settings)

    get_logger("[ALREADY]").info("bracketed")
    logger.complete()

    line = (local_tmp / "bot.log").read_text(encoding="utf-8")
    assert "[ALREADY]" in line
    assert "[[ALREADY]]" not in line