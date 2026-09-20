"""Strategist summary layer for reading World Model state into compact representations."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from typing import Any

from wow_bot.world.store import WorldModel, WorldStoreError


class SummaryError(Exception):
    """Exception raised when strategy summary generation fails."""


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


@dataclass(frozen=True)
class NodeSummary:
    id: int
    x: float
    y: float
    z: float
    kind: str
    distance: float
    last_seen_at: str


@dataclass(frozen=True)
class CombatSummary:
    combat_id: int
    target_entity_id: str
    outcome: str
    started_at: str
    ended_at: str


@dataclass(frozen=True)
class WorldSummary:
    around_xy: tuple[float, float]
    radius: float
    generated_at: str
    nearest_vendors: tuple[NodeSummary, ...]
    nearest_trainers: tuple[NodeSummary, ...]
    nearest_nodes: tuple[NodeSummary, ...]
    nearest_mobs: tuple[NodeSummary, ...]
    nearest_waypoints: tuple[NodeSummary, ...]
    recent_combats: tuple[CombatSummary, ...]
    total_nodes: int

    def to_json(self) -> dict[str, Any]:
        """Return a plain dict where tuples are converted to lists and values are JSON-serializable.

        Note on summary_version: summary_version lives in SummaryConfig as a configuration parameter
        governing the summary generation process rather than in WorldSummary output schema itself.
        """
        return {
            "around_xy": list(self.around_xy),
            "radius": self.radius,
            "generated_at": self.generated_at,
            "nearest_vendors": [asdict(n) for n in self.nearest_vendors],
            "nearest_trainers": [asdict(n) for n in self.nearest_trainers],
            "nearest_nodes": [asdict(n) for n in self.nearest_nodes],
            "nearest_mobs": [asdict(n) for n in self.nearest_mobs],
            "nearest_waypoints": [asdict(n) for n in self.nearest_waypoints],
            "recent_combats": [asdict(c) for c in self.recent_combats],
            "total_nodes": self.total_nodes,
        }


@dataclass(frozen=True)
class SummaryConfig:
    radius: float = 100.0
    per_kind_limit: int = 5
    recent_combat_limit: int = 5
    summary_version: int = 1

    def __post_init__(self) -> None:
        if self.radius <= 0:
            raise ValueError(f"radius must be > 0, got {self.radius}")
        if not (1 <= self.per_kind_limit <= 50):
            raise ValueError(
                f"per_kind_limit must be between 1 and 50, got {self.per_kind_limit}"
            )
        if not (0 <= self.recent_combat_limit <= 50):
            raise ValueError(
                f"recent_combat_limit must be between 0 and 50, got {self.recent_combat_limit}"
            )
        if self.summary_version < 1:
            raise ValueError(
                f"summary_version must be >= 1, got {self.summary_version}"
            )


def _validate_around_xy(around_xy: tuple[float, float]) -> None:
    if not isinstance(around_xy, (tuple, list)) or len(around_xy) != 2:
        raise SummaryError(
            f"around_xy must be a tuple of 2 floats, got {around_xy!r}"
        )
    x, y = around_xy
    if not (
        isinstance(x, (int, float))
        and isinstance(y, (int, float))
        and math.isfinite(x)
        and math.isfinite(y)
    ):
        raise SummaryError(
            f"around_xy elements must be finite numbers, got ({x!r}, {y!r})"
        )


async def summarize(
    world: WorldModel,
    around_xy: tuple[float, float],
    *,
    config: SummaryConfig | None = None,
    now: str | None = None,
) -> WorldSummary:
    """Summarize World Model data around a given location into a read-only WorldSummary.

    Pure read aggregation: does not mutate the DB and does not emit session events.
    """
    _validate_around_xy(around_xy)
    cfg = config if config is not None else SummaryConfig()
    ts = now if now is not None else _utc_now_iso()

    try:
        kinds = ("vendor", "trainer", "node", "mob", "waypoint")
        by_kind: dict[str, tuple[NodeSummary, ...]] = {}

        ax, ay = float(around_xy[0]), float(around_xy[1])

        for kind in kinds:
            rows = await world.query_nearest(
                kind=kind,
                from_xy=(ax, ay),
                radius=cfg.radius,
                limit=cfg.per_kind_limit,
            )
            node_summaries = tuple(
                NodeSummary(
                    id=r.id,
                    x=r.x,
                    y=r.y,
                    z=r.z,
                    kind=r.kind,
                    distance=round(math.hypot(r.x - ax, r.y - ay), 3),
                    last_seen_at=r.last_seen_at,
                )
                for r in rows
            )
            by_kind[kind] = node_summaries

        combats = await world.recent_combats(cfg.recent_combat_limit)
        combat_summaries = tuple(
            CombatSummary(
                combat_id=c.combat_id,
                target_entity_id=c.target_entity_id,
                outcome=c.outcome,
                started_at=c.started_at,
                ended_at=c.ended_at,
            )
            for c in combats
        )

        total_nodes = await world.count_nodes()

        return WorldSummary(
            around_xy=(ax, ay),
            radius=cfg.radius,
            generated_at=ts,
            nearest_vendors=by_kind["vendor"],
            nearest_trainers=by_kind["trainer"],
            nearest_nodes=by_kind["node"],
            nearest_mobs=by_kind["mob"],
            nearest_waypoints=by_kind["waypoint"],
            recent_combats=combat_summaries,
            total_nodes=total_nodes,
        )
    except WorldStoreError as err:
        raise SummaryError(f"World model query failed: {err}") from err
