"""Acceptance tests for Task 0.3 — Config File + Settings.

Proves:
1. ``get_settings().llm.model == "qwen2.5:7b"`` from ``config/config.json``.
2. Invalid config raises a clear Pydantic-backed error (not silent defaults).
"""

from __future__ import annotations

import itertools
import json
import shutil
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from wow_bot.shared.config import (
    DEFAULT_CONFIG_PATH,
    Settings,
    load_settings,
    reset_settings_cache,
)


@pytest.fixture(autouse=True)
def _clear_singleton() -> Any:
    """Keep the singleton cache isolated between tests."""
    reset_settings_cache()
    yield
    reset_settings_cache()


_TMP_COUNTER = itertools.count()


@pytest.fixture
def local_tmp() -> Iterator[Path]:
    """Workspace-local temp dir.

    Avoids depending on the system TMP location, which may be unwritable in
    restricted sandboxes; keeps the canonical test path portable.
    """
    base = Path.cwd() / ".pytest_local_tmp"
    base.mkdir(parents=True, exist_ok=True)
    d = base / f"cfg-{next(_TMP_COUNTER)}"
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)


def test_default_config_file_exists_and_is_valid_json() -> None:
    assert DEFAULT_CONFIG_PATH.is_file(), f"missing {DEFAULT_CONFIG_PATH}"
    data = json.loads(DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict)


def test_get_settings_loads_shipped_config() -> None:
    # Acceptance criterion 1: the shipped config yields the documented model.
    from wow_bot.shared.config import get_settings

    settings = get_settings()
    assert settings.llm.model == "qwen2.5:7b"
    assert settings.llm.base_url == "http://127.0.0.1:11434/v1/"
    assert settings.executor.dry_run is True  # invariant: dry-run default
    assert settings.internal_dynamics.lorenz_dt == 0.001


def test_get_settings_is_singleton() -> None:
    from wow_bot.shared.config import get_settings

    first = get_settings()
    second = get_settings()
    assert first is second


def test_all_sections_have_typed_defaults() -> None:
    # Requirements: all fields typed with sensible defaults → an empty dict
    # must still validate into a complete Settings object.
    settings = Settings.model_validate({})
    assert settings.llm.max_retries == 3
    assert settings.logging.level == "INFO"
    assert settings.internal_dynamics.memory_db_path == "data/memory.db"


def test_invalid_field_raises_clear_error(local_tmp: Path) -> None:
    # Acceptance criterion 2: invalid config fails loudly with a clear message.
    bad = local_tmp / "bad.json"
    bad.write_text(json.dumps({"llm": {"timeout_seconds": -5}}), encoding="utf-8")

    with pytest.raises(ValueError) as excinfo:
        load_settings(bad)

    message = str(excinfo.value)
    assert "Invalid configuration" in message
    assert "timeout_seconds" in message


def test_malformed_json_raises_value_error(local_tmp: Path) -> None:
    broken = local_tmp / "broken.json"
    broken.write_text("{not valid json", encoding="utf-8")

    with pytest.raises(ValueError, match="Invalid JSON"):
        load_settings(broken)


def test_missing_file_raises_filenotfound(local_tmp: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_settings(local_tmp / "does-not-exist.json")


def test_pydantic_validation_directly() -> None:
    # Underlying validator also rejects bad types on its own (belt-and-braces).
    with pytest.raises(ValidationError):
        Settings.model_validate({"executor": {"dry_run": "yes-please-no"}})
