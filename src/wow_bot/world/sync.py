"""GameState -> World sync layer for WoW-bot.

Provides periodic observation pass that updates the World Model from
GameState without duplicating player nodes or entity rows.
"""

from __future__ import annotations

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from wow_bot.session import Session
from wow_bot.world.store import PROMOTABLE_NODE_KINDS, VALID_NODE_KINDS, WorldModel

if TYPE_CHECKING:
    from wow_bot.nav.graph import GraphConfig, NavGraph


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
    entity_expiry_seconds: float | None = 300.0
    auto_refresh_graph: bool = False

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
        if self.entity_expiry_seconds is not None and self.entity_expiry_seconds <= 0:
            raise ValueError(
                f"entity_expiry_seconds must be > 0 or None, got {self.entity_expiry_seconds}"
            )


@dataclass(frozen=True)
class SyncStats:
    """Statistics collected during a single sync pass."""

    player_node_created: bool
    player_node_updated: bool
    entities_upserted: int
    entities_skipped: int
    target_seen: bool
    entities_expired: int = 0

    def to_json(self) -> dict[str, Any]:
        """Return JSON-serializable dictionary representation of stats."""
        data: dict[str, Any] = {
            "player_node_created": self.player_node_created,
            "player_node_updated": self.player_node_updated,
            "entities_upserted": self.entities_upserted,
            "entities_skipped": self.entities_skipped,
            "target_seen": self.target_seen,
        }
        if self.entities_expired > 0:
            data["entities_expired"] = self.entities_expired
        return data


class WorldSync:
    """Synchronizes GameState observations into WorldModel."""

    def __init__(
        self,
        world: WorldModel,
        *,
        config: SyncConfig | None = None,
        session: Session | None = None,
        graph: NavGraph | None = None,
    ) -> None:
        self._world: WorldModel = world
        self._config: SyncConfig = config if config is not None else SyncConfig()
        self._session: Session | None = session
        self._graph: NavGraph | None = graph

    @property
    def graph(self) -> NavGraph | None:
        """Current navigation graph, or None if not initialized."""
        return self._graph

    async def refresh_graph(
        self,
        *,
        previous: NavGraph | None = None,
        config: GraphConfig | None = None,
    ) -> NavGraph:
        """Rebuild navigation graph reflecting latest WorldModel state."""
        from wow_bot.nav.graph import build_graph, rebuild_graph

        prev = previous if previous is not None else self._graph
        if prev is not None:
            self._graph = await rebuild_graph(
                prev, self._world, config=config, session=self._session
            )
        else:
            self._graph = await build_graph(
                self._world, config=config, session=self._session
            )
        return self._graph

    async def sync_once(
        self,
        state: GameStateLike,
        *,
        now: str | None = None,
    ) -> SyncStats:
        """Perform a single GameState sync pass against the WorldModel."""
        # 0. Staleness expiry pass
        entities_expired = 0
        if (
            self._config.entity_expiry_seconds is not None
            and self._config.entity_expiry_seconds > 0
        ):
            ts_now = now if now is not None else datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
            cutoff_iso: str | None = None
            try:
                now_dt = datetime.fromisoformat(ts_now)
                cutoff_dt = now_dt - timedelta(seconds=self._config.entity_expiry_seconds)
                cutoff_iso = cutoff_dt.strftime("%Y-%m-%dT%H:%M:%SZ")
            except (ValueError, TypeError):
                cutoff_iso = None

            if cutoff_iso is not None:
                entities_expired = await self._world.expire_stale_entities(
                    older_than_iso=cutoff_iso
                )

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

        # 2. Entity observation & promotion
        entities_upserted = 0
        entities_skipped = 0

        for entity in state.entities:
            entity_id = getattr(entity, "entity_id", None)
            if not isinstance(entity_id, str) or not entity_id:
                entities_skipped += 1
                continue

            kind_val = getattr(entity, "kind", None)
            if not isinstance(kind_val, str) or not kind_val or kind_val == "unknown":
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
                entity_id,
                kind=kind_val,
                x=entity.x,
                y=entity.y,
                z=entity.z,
                meta=meta,
                now=now,
            )

            # Promotion of observed entities into map nodes
            if kind_val in PROMOTABLE_NODE_KINDS:
                await self._world.upsert_entity_node(
                    entity_id,
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
            entities_expired=entities_expired,
        )

        # 4. Optional graph refresh on sync
        if self._config.auto_refresh_graph or self._graph is not None:
            await self.refresh_graph()

        # 5. Emit session event if session is attached
        if self._session is not None:
            self._session.write_event({
                "event": "world_sync",
                "stats": stats.to_json(),
            })

        # 6. Return SyncStats
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
