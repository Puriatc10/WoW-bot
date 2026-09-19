"""Application configuration.

Loads and validates ``config/config.json`` through Pydantic models so that
silent config errors fail loudly at startup with a clear message (see
AGENTS.md, "Why Pydantic for Config?").

Public API:
    - :class:`Settings` and its section models.
    - :func:`load_settings` to build a Settings instance from an explicit path.
    - :func:`get_settings` process-wide singleton accessor.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field, ValidationError


class LLMConfig(BaseModel):
    """Local Ollama endpoint settings (OpenAI-compatible)."""

    base_url: str = "http://127.0.0.1:11434/v1/"
    api_key: str = "ollama"
    model: str = "qwen2.5:7b"
    timeout_seconds: float = Field(default=60, gt=0)
    max_retries: int = Field(default=3, ge=0)


class InternalDynamicsConfig(BaseModel):
    """Parameters for drives, oscillators, chaos and memory."""

    update_interval_ms: int = Field(default=100, gt=0)
    lorenz_sigma: float = 10
    lorenz_rho: float = 28
    lorenz_beta: float = 2.667
    lorenz_dt: float = Field(default=0.001, gt=0)
    trigger_threshold_base: float = Field(default=0.3, ge=0)
    memory_db_path: str = "data/memory.db"


class ExecutorConfig(BaseModel):
    """Humanization and controller settings.

    ``dry_run`` defaults to ``True`` per AGENTS.md constraint #6; agents must
    not silently flip it off.
    """

    human_delay_base_ms: int = Field(default=200, gt=0)
    human_delay_sigma_base: float = Field(default=0.4, gt=0)
    dry_run: bool = True


class LoggingConfig(BaseModel):
    """Loguru sink configuration."""

    level: str = "INFO"
    log_file: str = "logs/bot.log"
    rotation: str = "1 day"
    retention: str = "7 days"


class Settings(BaseModel):
    """Top-level validated application settings."""

    llm: LLMConfig = Field(default_factory=LLMConfig)
    internal_dynamics: InternalDynamicsConfig = Field(
        default_factory=InternalDynamicsConfig
    )
    executor: ExecutorConfig = Field(default_factory=ExecutorConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)


DEFAULT_CONFIG_PATH = Path("config") / "config.json"


def load_settings(config_path: str | Path = DEFAULT_CONFIG_PATH) -> Settings:
    """Load and validate settings from a JSON file.

    Raises:
        FileNotFoundError: if ``config_path`` does not exist.
        ValueError: if the file is not valid JSON or fails validation. The
            message includes the underlying Pydantic error detail so invalid
            config surfaces loudly at startup rather than silently.
    """
    path = Path(config_path)
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")

    try:
        raw_text = path.read_text(encoding="utf-8")
    except OSError as exc:  # pragma: no cover - defensive I/O guard
        raise ValueError(f"Could not read config file {path}: {exc}") from exc

    try:
        data: dict[str, Any] = json.loads(raw_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON in config file {path}: {exc}") from exc

    try:
        return Settings.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"Invalid configuration in {path}:\n{exc}") from exc


_settings_singleton: Settings | None = None


def get_settings() -> Settings:
    """Return the process-wide :class:`Settings` singleton.

    On first call, loads from ``config/config.json`` relative to the current
    working directory. Subsequent calls return the cached instance.
    """
    global _settings_singleton
    if _settings_singleton is None:
        _settings_singleton = load_settings(DEFAULT_CONFIG_PATH)
    return _settings_singleton


def reset_settings_cache() -> None:
    """Clear the cached singleton (intended for tests only)."""
    global _settings_singleton
    _settings_singleton = None
