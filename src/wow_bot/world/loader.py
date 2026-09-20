"""World model database loader and validator for WoW-bot system startup."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from wow_bot.session import Session
from wow_bot.world.schema import SCHEMA_VERSION, SchemaError
from wow_bot.world.store import WorldModel, WorldStoreError


class LoaderError(Exception):
    """Exception raised for errors during world database loading."""


@dataclass(frozen=True)
class LoaderConfig:
    """Configuration for World Model database loader."""

    db_path: str
    startup_budget_ms: int = 500
    require_existing: bool = False
    validate_isolation: bool = True

    def __post_init__(self) -> None:
        if not isinstance(self.db_path, str):
            raise TypeError("db_path must be a non-empty string without whitespace")
        path_str = self.db_path.strip()
        if not path_str or any(c.isspace() for c in self.db_path):
            raise ValueError("db_path MUST be a non-empty string without whitespace")
        if not (50 <= self.startup_budget_ms <= 10_000):
            raise ValueError(
                f"startup_budget_ms must be between 50 and 10000, got {self.startup_budget_ms}"
            )


@dataclass(frozen=True)
class LoaderReport:
    """Report summarizing the world loading process and statistics."""

    db_path: str
    schema_version: int
    opened_existing: bool
    duration_ms: float
    node_count: int
    edge_count: int
    entity_count: int

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the report."""
        return {
            "db_path": self.db_path,
            "schema_version": self.schema_version,
            "opened_existing": self.opened_existing,
            "duration_ms": self.duration_ms,
            "node_count": self.node_count,
            "edge_count": self.edge_count,
            "entity_count": self.entity_count,
        }


async def load_world(
    config: LoaderConfig,
    *,
    session: Session | None = None,
) -> tuple[WorldModel, LoaderReport]:
    """Load, validate, and report on a World Model SQLite database.

    Raises LoaderError if any step fails, budget is exceeded, or file validation fails.
    Fails closed: any error after WorldModel.open closes the world before re-raising.
    """
    path = Path(config.db_path)
    if path.is_dir():
        raise LoaderError(f"Database path '{config.db_path}' is a directory.")

    if config.require_existing and not path.exists():
        raise LoaderError(f"Required database file does not exist: '{config.db_path}'")

    start_ns = time.perf_counter_ns()

    opened_existing = path.is_file() and path.stat().st_size > 0

    try:
        world = await WorldModel.open(path)
    except (WorldStoreError, SchemaError) as exc:
        raise LoaderError(f"Failed to open WorldModel database: {exc}") from exc

    try:
        if config.validate_isolation:
            try:
                await world.check_isolation()
            except WorldStoreError as exc:
                raise LoaderError(f"World model database isolation check failed: {exc}") from exc

        try:
            stats = await world.statistics()
        except WorldStoreError as exc:
            raise LoaderError(f"Failed to gather world model statistics: {exc}") from exc

        end_ns = time.perf_counter_ns()
        duration_ms = (end_ns - start_ns) / 1_000_000.0

        if duration_ms > config.startup_budget_ms:
            raise LoaderError(
                f"World loader exceeded startup budget: duration {duration_ms:.2f} ms > "
                f"budget {config.startup_budget_ms} ms"
            )

        report = LoaderReport(
            db_path=str(path),
            schema_version=SCHEMA_VERSION,
            opened_existing=opened_existing,
            duration_ms=duration_ms,
            node_count=stats["node_count"],
            edge_count=stats["edge_count"],
            entity_count=stats["entity_count"],
        )

        if session is not None:
            session.write_event({
                "event": "world_loaded",
                "payload": report.to_json(),
            })

        return world, report
    except BaseException:
        await world.close()
        raise
