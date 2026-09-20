"""Session management for research execution runs.

Implements execution session lifecycle handling including session metadata,
append-only event logging, exception traceback capture, atomic session state updates,
and file permission management.
"""

import dataclasses
import json
import os
import secrets
import threading
import traceback
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Any, Literal, Self

from wow_bot.config import Config, ConfigError


class SessionClosed(Exception):
    """Raised when an operation is attempted on a closed Session."""


@dataclasses.dataclass(frozen=True)
class SessionInfo:
    """Immutable snapshot of session metadata."""

    session_id: str
    path: Path
    started_at: str
    mode: str
    config_snapshot: dict[str, Any]


def _utc_now_iso() -> str:
    """Return the current UTC timestamp formatted as ISO 8601 string."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _make_config_snapshot(config: Config) -> dict[str, Any]:
    """Convert Config dataclass to a JSON-serializable dictionary."""
    raw_dict = dataclasses.asdict(config)

    def _convert(obj: Any) -> Any:
        if isinstance(obj, Path):
            return str(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, (tuple, list)):
            return [_convert(item) for item in obj]
        return obj

    converted = _convert(raw_dict)
    assert isinstance(converted, dict)
    return converted


class Session:
    """Execution session managing log files and metadata.

    File permissions are set to 0o600 on session.json and events.jsonl on POSIX systems
    (skipped on Windows via os.name check).
    """

    def __init__(self, info: SessionInfo) -> None:
        self._info = info
        self._path = info.path
        self._events_path = self._path / "events.jsonl"
        self._closed = False
        self._crash_written = False
        self._lock = threading.Lock()

    @classmethod
    def start(cls, config: Config) -> "Session":
        """Start a new session, creating directory structure and initial files."""
        if config.session_root.exists() and not os.access(config.session_root, os.W_OK):
            raise ConfigError(f"Session root directory is not writable: '{config.session_root}'")

        mode = "LAB" if config.lab_mode else "MOCK"
        started_at = _utc_now_iso()
        session_id = f"{datetime.now(UTC).strftime('%Y%m%d-%H%M%S')}-{secrets.token_hex(4)}"
        session_dir = config.session_root / session_id

        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except PermissionError as e:
            raise ConfigError(f"Session root directory is not writable: '{config.session_root}'") from e
        except OSError as e:
            if not os.access(config.session_root, os.W_OK):
                raise ConfigError(f"Session root directory is not writable: '{config.session_root}'") from e
            raise

        config_snapshot = _make_config_snapshot(config)

        events_path = session_dir / "events.jsonl"
        try:
            fd = os.open(events_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            os.close(fd)
        except PermissionError as e:
            raise ConfigError(f"Session root directory is not writable: '{config.session_root}'") from e

        session_json_path = session_dir / "session.json"
        initial_data = {
            "session_id": session_id,
            "started_at": started_at,
            "mode": mode,
            "config_snapshot": config_snapshot,
        }
        try:
            with open(session_json_path, "w", encoding="utf-8") as f:
                json.dump(initial_data, f, indent=2, ensure_ascii=False)
                f.write("\n")
        except PermissionError as e:
            raise ConfigError(f"Session root directory is not writable: '{config.session_root}'") from e

        if os.name != "nt":
            os.chmod(session_json_path, 0o600)
            os.chmod(events_path, 0o600)

        info = SessionInfo(
            session_id=session_id,
            path=session_dir,
            started_at=started_at,
            mode=mode,
            config_snapshot=config_snapshot,
        )
        return cls(info)

    @property
    def info(self) -> SessionInfo:
        """Return session info snapshot."""
        return self._info

    @property
    def path(self) -> Path:
        """Return session directory path."""
        return self._path

    @property
    def session_id(self) -> str:
        """Return session ID string."""
        return self._info.session_id

    def write_event(self, event: dict[str, Any]) -> None:
        """Append an event line to events.jsonl with thread safety.

        Raises SessionClosed if session is closed.
        Injects UTC ISO 8601 timestamp 'ts' if missing.
        """
        with self._lock:
            if self._closed:
                raise SessionClosed("Cannot write event to a closed session.")

            if "ts" not in event:
                event["ts"] = _utc_now_iso()

            line = json.dumps(event, ensure_ascii=False) + "\n"
            fd = os.open(self._events_path, os.O_APPEND | os.O_CREAT | os.O_WRONLY, 0o600)
            try:
                os.write(fd, line.encode("utf-8"))
            finally:
                os.close(fd)

    def write_crash(self, exc: BaseException) -> None:
        """Write crash.json with exception details and traceback.

        Raises SessionClosed if session is closed.
        Idempotent across repeated calls.
        """
        with self._lock:
            if self._closed:
                raise SessionClosed("Cannot write crash to a closed session.")

            if self._crash_written:
                return

            self._crash_written = True
            tb_str = "".join(traceback.format_exception(exc))
            crash_data = {
                "type": type(exc).__name__,
                "message": str(exc),
                "traceback": tb_str,
                "ts": _utc_now_iso(),
            }

            crash_path = self._path / "crash.json"
            temp_path = self._path / f"crash.json.tmp.{secrets.token_hex(4)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(crash_data, f, indent=2, ensure_ascii=False)
                f.write("\n")

            if os.name != "nt":
                os.chmod(temp_path, 0o600)

            os.replace(temp_path, crash_path)

    def close(self, reason: str) -> None:
        """Close the session, appending stopped_at and stop_reason atomically to session.json."""
        with self._lock:
            if self._closed:
                return

            self._closed = True
            stopped_at = _utc_now_iso()

            session_json_path = self._path / "session.json"
            if session_json_path.exists():
                with open(session_json_path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            else:
                data = {
                    "session_id": self._info.session_id,
                    "started_at": self._info.started_at,
                    "mode": self._info.mode,
                    "config_snapshot": self._info.config_snapshot,
                }

            data["stopped_at"] = stopped_at
            data["stop_reason"] = reason

            temp_path = self._path / f"session.json.tmp.{secrets.token_hex(4)}"
            with open(temp_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
                f.write("\n")

            if os.name != "nt":
                os.chmod(temp_path, 0o600)

            os.replace(temp_path, session_json_path)

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        if exc_type is None:
            self.close("clean")
        else:
            if exc_val is not None:
                self.write_crash(exc_val)
            self.close("exception")
        return False
