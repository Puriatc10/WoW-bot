"""GameState -> World sync layer for WoW-bot.

Provides periodic observation pass that updates the World Model from
GameState without duplicating player nodes or entity rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Protocol, runtime_checkable

from wow_bot.session import Session
from wow_bot.world.store import VALID_NODE_KINDS, WorldModel


class SyncError(Exception):
    """Exception raised for errors in WorldSync operation."""


@runtime_checkable
class EntityLike(Protocol):
    """Protocol describing an entity observation in GameState."""

    @property
    def entity_id(self) -> str: ...

    @property
    def kind(self) -> str: ...

    @property
    def x(self) -> float: ...

    @property
    def y(self) -> float: ...

    @property
    def z(self) -> float: ...


@runtime_checkable
class GameStateLike(Protocol):
    """Opaque view of GameState consumed by the sync layer.

    Implementations MUST provide these attributes (or properties):
    """

    @property
    def player_x(self) -> float: ...

    @property
    def player_y(self) -> float: ...

    @property
    def player_z(self) -> float: ...

    @property
    def entities(self) -> tuple[EntityLike, ...]: ...

    @property
    def target_entity_id(self) -> str | None: ...


@dataclass(frozen=True)
class SyncConfig:
    """Configuration for world sync behaviour."""

    player_node_kind: str = "waypoint"
    player_node_min_distance_units: float = 5.0
    entity_meta_max_keys: int = 8

    def __post_init__(self) -> None:
        if self.player_node_kind not in VALID_NODE_KINDS:
            raise ValueError(
                f"Invalid player_node_kind '{self.player_node_kind}'. "
                f"Expected one of {sorted(VALID_NODE_KINDS)}"
            )
        if self.player_node_min_distance_units <= 0:
            raise ValueError(
                f"player_node_min_distance_units must be > 0, got {self.player_node_min_distance_units}"
            )
        if self.entity_meta_max_keys < 0:
            raise ValueError(
                f"entity_meta_max_keys must be >= 0, got {self.entity_meta_max_keys}"
            )


@dataclass(frozen=True)
class SyncStats:
    """Statistics collected during a single sync pass."""

    player_node_created: bool
    player_node_updated: bool
    entities_upserted: int
    entities_skipped: int
    target_seen: bool

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation of stats."""
        return {
            "player_node_created": self.player_node_created,
            "player_node_updated": self.player_node_updated,
            "entities_upserted": self.entities_upserted,
            "entities_skipped": self.entities_skipped,
            "target_seen": self.target_seen,
        }


class WorldSync:
    """Synchronizes GameState observations into WorldModel."""

    def __init__(
        self,
        world: WorldModel,
        *,
        config: SyncConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self._world: WorldModel = world
        self._config: SyncConfig = config if config is not None else SyncConfig()
        self._session: Session | None = session

    async def sync_once(
        self,
        state: GameStateLike,
        *,
        now: str | None = None,
    ) -> SyncStats:
        """Perform a single GameState sync pass against the WorldModel."""
        # 1. Player node handling
        player_x = float(state.player_x)
        player_y = float(state.player_y)

        nearest_nodes = await self._world.query_nearest(
            kind=self._config.player_node_kind,
            from_xy=(player_x, player_y),
            radius=self._config.player_node_min_distance_units,
            limit=1,
        )

        if nearest_nodes:
            existing_node = nearest_nodes[0]
            await self._world.update_node_last_seen(existing_node.id, now=now)
            player_node_updated = True
            player_node_created = False
        else:
            await self._world.add_node(
                x=player_x,
                y=player_y,
                z=float(state.player_z),
                kind=self._config.player_node_kind,
                meta={"created_by": "sync", "entity_kind": "player"},
            )
            player_node_created = True
            player_node_updated = False

        # 2. Entity observation
        entities_upserted = 0
        entities_skipped = 0

        for entity in state.entities:
            kind_val = getattr(entity, "kind", None)
            if not isinstance(kind_val, str) or not kind_val:
                entities_skipped += 1
                continue

            # Build bounded meta dict from entity: include only "kind", "x", "y", "z"
            raw_meta = {
                "kind": kind_val,
                "x": entity.x,
                "y": entity.y,
                "z": entity.z,
            }
            if self._config.entity_meta_max_keys < len(raw_meta):
                keys = list(raw_meta.keys())[: self._config.entity_meta_max_keys]
                meta = {k: raw_meta[k] for k in keys}
            else:
                meta = raw_meta

            await self._world.mark_seen(
                entity.entity_id,
                kind=kind_val,
                x=entity.x,
                y=entity.y,
                z=entity.z,
                meta=meta,
                now=now,
            )
            entities_upserted += 1

        # 3. Target observation
        target_seen = False
        if state.target_entity_id is not None:
            target_id = state.target_entity_id
            for entity in state.entities:
                if entity.entity_id == target_id:
                    target_seen = True
                    break

        stats = SyncStats(
            player_node_created=player_node_created,
            player_node_updated=player_node_updated,
            entities_upserted=entities_upserted,
            entities_skipped=entities_skipped,
            target_seen=target_seen,
        )

        # 4. Emit session event if session is attached
        if self._session is not None:
            self._session.write_event({
                "event": "world_sync",
                "stats": stats.to_json(),
            })

        # 5. Return SyncStats
        return stats

    async def run_periodic(
        self,
        state_source: Callable[[], GameStateLike],
        *,
        rate_hz: float = 1.0,
        stop_event: asyncio.Event | None = None,
        max_iterations: int | None = None,
    ) -> list[SyncStats]:
        """Run periodic sync loop."""
        if not (0 < rate_hz <= 10):
            raise ValueError(f"rate_hz must be in range (0, 10], got {rate_hz}")

        if max_iterations is not None and max_iterations < 1:
            raise ValueError(f"max_iterations must be >= 1 if provided, got {max_iterations}")

        if stop_event is None:
            stop_event = asyncio.Event()

        interval = 1.0 / rate_hz
        results: list[SyncStats] = []

        while True:
            if stop_event.is_set():
                break

            stats = await self.sync_once(state_source())
            results.append(stats)

            if max_iterations is not None and len(results) >= max_iterations:
                break

            if stop_event.is_set():
                break

            try:
                await asyncio.wait_for(stop_event.wait(), timeout=interval)
                if stop_event.is_set():
                    break
            except TimeoutError:
                pass

        return results
