"""Structured logging configuration and event recording for execution sessions.

Provides setup for stderr console and JSON line file logging, as well as a helper
for structured log event generation.
"""

import json
import logging
import sys
from datetime import UTC, datetime
from typing import Any

from wow_bot.session import Session

VALID_LEVELS: set[str] = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}


class ConsoleFormatter(logging.Formatter):
    """Formatter for human-readable console log output."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        time_str = dt.strftime("%H:%M:%S")

        raw_payload = getattr(record, "payload", {})
        if not isinstance(raw_payload, dict):
            payload = {"_raw": str(raw_payload)}
        else:
            payload = raw_payload

        payload_json = json.dumps(payload, ensure_ascii=False)
        return f"{time_str} {record.levelname} {record.getMessage()} {payload_json}"


class JSONFileFormatter(logging.Formatter):
    """Formatter for single-line JSON log records in app.log."""

    def format(self, record: logging.LogRecord) -> str:
        dt = datetime.fromtimestamp(record.created, tz=UTC)
        ts_str = dt.strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        raw_payload = getattr(record, "payload", {})
        if not isinstance(raw_payload, dict):
            payload = {"_raw": str(raw_payload)}
        else:
            payload = raw_payload

        obj = {
            "ts": ts_str,
            "level": record.levelname,
            "module": record.name,
            "event": record.getMessage(),
            "payload": payload,
        }
        return json.dumps(obj, ensure_ascii=False)


def setup_logging(session: Session, level: str) -> None:
    """Configure root logger with stderr console handler and session file handler.

    Validates level, clears pre-existing handlers on root logger, and adds:
    1. StreamHandler writing to sys.stderr (INFO for MOCK, WARNING for LAB).
    2. FileHandler writing single-line JSON to app.log at the given level.
    """
    if level not in VALID_LEVELS:
        raise ValueError(
            f"Invalid log level '{level}'. Level must be one of: {', '.join(sorted(VALID_LEVELS))}."
        )

    numeric_level = getattr(logging, level)

    root = logging.getLogger()
    for h in list(root.handlers):
        root.removeHandler(h)
        h.close()

    root.setLevel(numeric_level)

    console_handler = logging.StreamHandler(sys.stderr)
    console_level = logging.INFO if session.info.mode == "MOCK" else logging.WARNING
    console_handler.setLevel(console_level)
    console_handler.setFormatter(ConsoleFormatter())
    root.addHandler(console_handler)

    app_log_path = session.path / "app.log"
    file_handler = logging.FileHandler(app_log_path, mode="a", encoding="utf-8")
    file_handler.setLevel(numeric_level)
    file_handler.setFormatter(JSONFileFormatter())
    root.addHandler(file_handler)


def log_event(
    logger: logging.Logger,
    level: int,
    event: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """Log a structured event using the given logger and level.

    Helper function ensuring consistent payload dictionary handling.
    """
    if payload is not None and not isinstance(payload, dict):
        payload_dict: dict[str, Any] = {"_raw": str(payload)}
    else:
        payload_dict = payload if payload is not None else {}

    logger.log(level, event, extra={"payload": payload_dict})
