"""Shared logging setup.

Thin wrapper around :mod:`loguru` providing one console sink with colors and
one rotating file sink, both carrying millisecond-resolution timestamps and a
per-module context tag such as ``[DYNAMICS]`` or ``[LLM]`` (see AGENTS.md,
"Why loguru over stdlib logging?").

Public API:
    - :func:`get_logger` factory returning a bound, tagged logger.
    - :func:`configure_logging` explicit initialization from config.
    - :func:`reset_logging` teardown hook for tests and shutdown.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

from loguru import logger

if TYPE_CHECKING:  # pragma: no cover - typing only, avoids an import cycle
    from wow_bot.shared.config import Settings

# Sink ids so handlers can be removed/replaced without touching user sinks.
_CONSOLE_SINK_ID: int | None = None
_FILE_SINK_ID: int | None = None

# Format carries timestamp (ms), level, context tag and message. The tag lives
# in each record's ``extra`` dict; it is empty until :func:`get_logger` binds it.
_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{extra[tag]}</cyan> | "
    "<level>{message}</level>"
)


def _patch_record(record: dict[str, Any]) -> None:
    """Guarantee ``record["extra"]["tag"]`` exists for every sink.

    Typed as a plain dict because loguru's ``Record`` alias is not exported at a
    stable path across versions; the callable matches ``Callable[[Record], None]``
    structurally.
    """
    record["extra"].setdefault("tag", "")


def configure_logging(settings: Settings | None = None) -> None:
    """Install the console and file sinks described by *settings*.

    Idempotent: previously installed WOW-bot sinks are removed first, so
    repeated calls never duplicate output. Safe to call before any
    :func:`get_logger` call; :func:`get_logger` also lazily configures if this
    has not run yet.

    Args:
        settings: Validated application settings. Defaults to
            :func:`wow_bot.shared.config.get_settings`.
    """
    global _CONSOLE_SINK_ID, _FILE_SINK_ID

    if settings is None:
        # Imported lazily to keep the module free of an import-time dependency
        # on config (which imports pydantic); also avoids a circular import.
        from wow_bot.shared.config import get_settings

        settings = get_settings()

    log_cfg = settings.logging

    reset_logging()

    logger.remove()
    # _patch_record is structurally compatible with loguru's Callable[[Record], None];
    # Record is a TypedDict (dict[str, Any]) but mypy treats them as nominal types.
    logger.configure(patcher=_patch_record)  # type: ignore[arg-type]

    _CONSOLE_SINK_ID = logger.add(
        sys.stderr,
        level=log_cfg.level.upper(),
        format=_FORMAT,
        colorize=True,
        backtrace=False,
        diagnose=False,  # never leak variable contents into logs
        enqueue=False,
    )

    log_path = Path(log_cfg.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    _FILE_SINK_ID = logger.add(
        str(log_path),
        level=log_cfg.level.upper(),
        format=_FORMAT,
        colorize=False,
        rotation=log_cfg.rotation,
        retention=log_cfg.retention,
        backtrace=False,
        diagnose=False,
        enqueue=False,
        encoding="utf-8",
    )


def reset_logging() -> None:
    """Remove the sinks installed by :func:`configure_logging`.

    Used by tests for isolation and by shutdown hooks to flush/close the file
    sink while preserving already-written audit logs (AGENTS.md constraint #7:
    shutdown must preserve logs, never delete them).
    """
    global _CONSOLE_SINK_ID, _FILE_SINK_ID

    for sink_id in (_CONSOLE_SINK_ID, _FILE_SINK_ID):
        if sink_id is not None:
            try:
                logger.remove(sink_id)
            except ValueError:  # pragma: no cover - sink already gone
                pass

    _CONSOLE_SINK_ID = None
    _FILE_SINK_ID = None


def get_logger(tag: str) -> Any:
    """Return a logger whose records carry the context *tag*.

    The tag is rendered as ``[TAG]`` in both sinks, giving per-module context
    (e.g. ``[DYNAMICS]``, ``[LLM]``, ``[EXEC]``). Configuration happens lazily
    on first use so callers need not invoke :func:`configure_logging` manually.

    Args:
        tag: Short uppercase module identifier. Brackets are added here, so
            callers pass ``"DYNAMICS"`` rather than ``"[DYNAMICS]"``.

    Returns:
        A :class:`loguru.Logger` bound to ``{"tag": "[TAG]"}``.
    """
    if _CONSOLE_SINK_ID is None and _FILE_SINK_ID is None:
        configure_logging()

    normalized = tag.strip()
    display = normalized if normalized.startswith("[") else f"[{normalized}]"
    return logger.bind(tag=display)