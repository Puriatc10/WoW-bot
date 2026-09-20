"""Async World Model store implementation for WoW-bot."""

from __future__ import annotations

import asyncio
import json
import math
import sqlite3
import types
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Self

import aiosqlite

from wow_bot.world.schema import SchemaError, apply_migrations, assert_isolated

VALID_NODE_KINDS: set[str] = {
    "vendor",
    "trainer",
    "node",
    "mob",
    "waypoint",
    "unknown",
}

VALID_COMBAT_OUTCOMES: set[str] = {
    "win",
    "loss",
    "flee",
    "timeout",
    "unknown",
}


class WorldStoreError(Exception):
    """Exception raised for errors in WorldModel store operations."""


def _utc_now_iso() -> str:
    """Return current UTC timestamp in ISO 8601 format with a trailing Z."""
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


async def _run_sync(
    conn: aiosqlite.Connection, fn: Callable[..., Any], *args: Any
) -> Any:
    return await conn._execute(fn, *args)  # type: ignore[no-untyped-call]


@dataclass(frozen=True)
class NodeRow:
    id: int
    x: float
    y: float
    z: float
    kind: str
    discovered_at: str
    last_seen_at: str
    meta_json: str


@dataclass(frozen=True)
class EdgeRow:
    from_id: int
    to_id: int
    cost: float
    bidirectional: bool
    discovered_at: str


@dataclass(frozen=True)
class EntityRow:
    entity_id: str
    kind: str
    last_x: float
    last_y: float
    last_z: float
    last_seen_at: str
    meta_json: str


@dataclass(frozen=True)
class RouteRow:
    route_id: int
    from_id: int
    to_id: int
    succeeded: bool
    taken_at: str


@dataclass(frozen=True)
class CombatRow:
    combat_id: int
    target_entity_id: str
    outcome: str
    started_at: str
    ended_at: str


class WorldModel:
    """Async store API for managing the World Model SQLite database."""

    def __init__(self, conn: aiosqlite.Connection) -> None:
        self._conn: aiosqlite.Connection = conn
        self._lock: asyncio.Lock | None = None
        self._closed: bool = False

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    @classmethod
    async def open(cls, path: str | Path) -> WorldModel:
        db_path = Path(path)
        try:
            conn = await aiosqlite.connect(db_path)
        except Exception as err:
            raise WorldStoreError(f"Failed to open database at {db_path}: {err}") from err

        try:
            await conn.execute("PRAGMA foreign_keys = ON;")
            await _run_sync(conn, assert_isolated, conn._connection)
            await _run_sync(conn, apply_migrations, conn._connection)
            await conn.execute("PRAGMA journal_mode = WAL;")
            await _run_sync(conn, assert_isolated, conn._connection)
            await conn.commit()
        except SchemaError as err:
            await conn.rollback()
            await conn.close()
            raise WorldStoreError(f"Schema isolation or migration error: {err}") from err
        except Exception as err:
            await conn.rollback()
            await conn.close()
            raise WorldStoreError(f"Failed to initialize database: {err}") from err

        return cls(conn)

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._conn.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: types.TracebackType | None,
    ) -> None:
        await self.close()

    async def add_node(
        self,
        x: float,
        y: float,
        *,
        z: float = 0.0,
        kind: str,
        meta: dict[str, Any] | None = None,
    ) -> int:
        if kind not in VALID_NODE_KINDS:
            raise WorldStoreError(
                f"Invalid kind '{kind}'. Expected one of {sorted(VALID_NODE_KINDS)}"
            )

        ts = _utc_now_iso()
        meta_str = json.dumps(meta or {}, sort_keys=True, ensure_ascii=False)

        async with self._get_lock():
            try:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO wm_map_nodes (x, y, z, kind, discovered_at, last_seen_at, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    """,
                    (x, y, z, kind, ts, ts, meta_str),
                )
                node_id = cursor.lastrowid
                await self._conn.commit()
                if node_id is None:
                    raise WorldStoreError("Failed to obtain inserted node id")
                return node_id
            except (sqlite3.Error, aiosqlite.Error) as e:
                raise WorldStoreError(f"Database error in add_node: {e}") from e

    async def get_node(self, node_id: int) -> NodeRow | None:
        async with self._conn.execute(
            """
            SELECT id, x, y, z, kind, discovered_at, last_seen_at, meta_json
            FROM wm_map_nodes
            WHERE id = ?
            """,
            (node_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return NodeRow(
                id=int(row[0]),
                x=float(row[1]),
                y=float(row[2]),
                z=float(row[3]),
                kind=str(row[4]),
                discovered_at=str(row[5]),
                last_seen_at=str(row[6]),
                meta_json=str(row[7]),
            )

    async def update_node_last_seen(
        self, node_id: int, *, now: str | None = None
    ) -> bool:
        ts = now if now is not None else _utc_now_iso()
        async with self._get_lock():
            cursor = await self._conn.execute(
                """
                UPDATE wm_map_nodes
                SET last_seen_at = ?
                WHERE id = ?
                """,
                (ts, node_id),
            )
            updated = cursor.rowcount > 0
            await self._conn.commit()
            return updated

    async def add_edge(
        self,
        from_id: int,
        to_id: int,
        *,
        cost: float,
        bidirectional: bool = True,
        now: str | None = None,
    ) -> None:
        if cost < 0:
            raise WorldStoreError(f"Cost must be non-negative, got {cost}")
        if from_id == to_id:
            raise WorldStoreError(
                f"Self-loop edge (from_id == to_id == {from_id}) forbidden"
            )

        ts = now if now is not None else _utc_now_iso()
        b_val = 1 if bidirectional else 0

        async with self._get_lock():
            try:
                await self._conn.execute(
                    """
                    INSERT OR REPLACE INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (from_id, to_id, cost, b_val, ts),
                )
                if bidirectional:
                    await self._conn.execute(
                        """
                        INSERT OR REPLACE INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at)
                        VALUES (?, ?, ?, ?, ?)
                        """,
                        (to_id, from_id, cost, b_val, ts),
                    )
                await self._conn.commit()
            except sqlite3.IntegrityError as e:
                raise WorldStoreError(
                    f"Foreign key constraint or integrity failure in add_edge: {e}"
                ) from e
            except (sqlite3.Error, aiosqlite.Error) as e:
                raise WorldStoreError(f"Database error in add_edge: {e}") from e

    async def all_nodes_and_edges(self) -> tuple[list[NodeRow], list[EdgeRow]]:
        async with self._conn.execute(
            """
            SELECT id, x, y, z, kind, discovered_at, last_seen_at, meta_json
            FROM wm_map_nodes
            ORDER BY id ASC
            """
        ) as cursor:
            node_rows = await cursor.fetchall()

        nodes = [
            NodeRow(
                id=int(r[0]),
                x=float(r[1]),
                y=float(r[2]),
                z=float(r[3]),
                kind=str(r[4]),
                discovered_at=str(r[5]),
                last_seen_at=str(r[6]),
                meta_json=str(r[7]),
            )
            for r in node_rows
        ]

        async with self._conn.execute(
            """
            SELECT from_id, to_id, cost, bidirectional, discovered_at
            FROM wm_map_edges
            ORDER BY from_id ASC, to_id ASC
            """
        ) as cursor:
            edge_rows = await cursor.fetchall()

        edges = [
            EdgeRow(
                from_id=int(r[0]),
                to_id=int(r[1]),
                cost=float(r[2]),
                bidirectional=bool(r[3]),
                discovered_at=str(r[4]),
            )
            for r in edge_rows
        ]

        return nodes, edges

    async def edges_from(self, node_id: int) -> list[EdgeRow]:
        async with self._conn.execute(
            """
            SELECT from_id, to_id, cost, bidirectional, discovered_at
            FROM wm_map_edges
            WHERE from_id = ?
            ORDER BY to_id ASC
            """,
            (node_id,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                EdgeRow(
                    from_id=int(r[0]),
                    to_id=int(r[1]),
                    cost=float(r[2]),
                    bidirectional=bool(r[3]),
                    discovered_at=str(r[4]),
                )
                for r in rows
            ]

    async def mark_seen(
        self,
        entity_id: str,
        *,
        kind: str,
        x: float,
        y: float,
        z: float = 0.0,
        meta: dict[str, Any] | None = None,
        now: str | None = None,
    ) -> None:
        """Upsert entity in wm_entities_seen.

        Updates kind, last_x, last_y, last_z, last_seen_at, and meta_json.

        meta handling on update: MERGE the new meta into the existing
        meta_json (shallow, new keys override). If the existing meta_json is
        malformed JSON, replace it with the new meta.
        """
        ts = now if now is not None else _utc_now_iso()
        new_meta = meta if meta is not None else {}

        async with self._get_lock():
            existing = await self.get_entity(entity_id)
            if existing is not None:
                try:
                    loaded = json.loads(existing.meta_json)
                    if isinstance(loaded, dict):
                        merged = {**loaded, **new_meta}
                    else:
                        merged = dict(new_meta)
                except (json.JSONDecodeError, TypeError, ValueError):
                    merged = dict(new_meta)
            else:
                merged = dict(new_meta)

            meta_str = json.dumps(merged, sort_keys=True, ensure_ascii=False)

            try:
                await self._conn.execute(
                    """
                    INSERT INTO wm_entities_seen (entity_id, kind, last_x, last_y, last_z, last_seen_at, meta_json)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                    ON CONFLICT(entity_id) DO UPDATE SET
                        kind = excluded.kind,
                        last_x = excluded.last_x,
                        last_y = excluded.last_y,
                        last_z = excluded.last_z,
                        last_seen_at = excluded.last_seen_at,
                        meta_json = excluded.meta_json
                    """,
                    (entity_id, kind, x, y, z, ts, meta_str),
                )
                await self._conn.commit()
            except (sqlite3.Error, aiosqlite.Error) as e:
                raise WorldStoreError(f"Database error in mark_seen: {e}") from e

    async def get_entity(self, entity_id: str) -> EntityRow | None:
        async with self._conn.execute(
            """
            SELECT entity_id, kind, last_x, last_y, last_z, last_seen_at, meta_json
            FROM wm_entities_seen
            WHERE entity_id = ?
            """,
            (entity_id,),
        ) as cursor:
            row = await cursor.fetchone()
            if row is None:
                return None
            return EntityRow(
                entity_id=str(row[0]),
                kind=str(row[1]),
                last_x=float(row[2]),
                last_y=float(row[3]),
                last_z=float(row[4]),
                last_seen_at=str(row[5]),
                meta_json=str(row[6]),
            )

    async def record_route(
        self,
        from_id: int,
        to_id: int,
        *,
        succeeded: bool,
        now: str | None = None,
    ) -> int:
        ts = now if now is not None else _utc_now_iso()
        succ_val = 1 if succeeded else 0

        async with self._get_lock():
            try:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO wm_routes_taken (from_id, to_id, succeeded, taken_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (from_id, to_id, succ_val, ts),
                )
                route_id = cursor.lastrowid
                await self._conn.commit()
                if route_id is None:
                    raise WorldStoreError("Failed to obtain inserted route id")
                return route_id
            except sqlite3.IntegrityError as e:
                raise WorldStoreError(
                    f"Foreign key or integrity constraint in record_route: {e}"
                ) from e
            except (sqlite3.Error, aiosqlite.Error) as e:
                raise WorldStoreError(f"Database error in record_route: {e}") from e

    async def record_combat(
        self,
        target_entity_id: str,
        *,
        outcome: str,
        started_at: str,
        ended_at: str,
    ) -> int:
        if outcome not in VALID_COMBAT_OUTCOMES:
            raise WorldStoreError(
                f"Invalid combat outcome '{outcome}'. Expected one of {sorted(VALID_COMBAT_OUTCOMES)}"
            )

        async with self._get_lock():
            try:
                cursor = await self._conn.execute(
                    """
                    INSERT INTO wm_combat_history (target_entity_id, outcome, started_at, ended_at)
                    VALUES (?, ?, ?, ?)
                    """,
                    (target_entity_id, outcome, started_at, ended_at),
                )
                combat_id = cursor.lastrowid
                await self._conn.commit()
                if combat_id is None:
                    raise WorldStoreError("Failed to obtain inserted combat id")
                return combat_id
            except (sqlite3.Error, aiosqlite.Error) as e:
                raise WorldStoreError(f"Database error in record_combat: {e}") from e

    async def recent_combats(self, limit: int) -> list[CombatRow]:
        """Fetch recent combat history rows ordered by ended_at DESC, then combat_id DESC.

        Used by the strategist summary layer (T4.4).
        """
        if limit < 0:
            raise WorldStoreError(f"limit must be >= 0, got {limit}")
        if limit == 0:
            return []

        async with self._conn.execute(
            """
            SELECT combat_id, target_entity_id, outcome, started_at, ended_at
            FROM wm_combat_history
            ORDER BY ended_at DESC, combat_id DESC
            LIMIT ?
            """,
            (limit,),
        ) as cursor:
            rows = await cursor.fetchall()
            return [
                CombatRow(
                    combat_id=int(r[0]),
                    target_entity_id=str(r[1]),
                    outcome=str(r[2]),
                    started_at=str(r[3]),
                    ended_at=str(r[4]),
                )
                for r in rows
            ]

    async def query_nearest(
        self,
        kind: str,
        from_xy: tuple[float, float],
        *,
        radius: float,
        limit: int = 10,
    ) -> list[NodeRow]:
        if radius <= 0:
            raise WorldStoreError(f"radius must be > 0, got {radius}")
        if limit < 1:
            raise WorldStoreError(f"limit must be >= 1, got {limit}")

        min_x = from_xy[0] - radius
        max_x = from_xy[0] + radius
        min_y = from_xy[1] - radius
        max_y = from_xy[1] + radius

        async with self._conn.execute(
            """
            SELECT id, x, y, z, kind, discovered_at, last_seen_at, meta_json
            FROM wm_map_nodes
            WHERE kind = ? AND x >= ? AND x <= ? AND y >= ? AND y <= ?
            """,
            (kind, min_x, max_x, min_y, max_y),
        ) as cursor:
            rows = await cursor.fetchall()

        fx, fy = from_xy
        candidates: list[tuple[float, NodeRow]] = []
        for r in rows:
            node = NodeRow(
                id=int(r[0]),
                x=float(r[1]),
                y=float(r[2]),
                z=float(r[3]),
                kind=str(r[4]),
                discovered_at=str(r[5]),
                last_seen_at=str(r[6]),
                meta_json=str(r[7]),
            )
            dist = math.hypot(node.x - fx, node.y - fy)
            if dist <= radius:
                candidates.append((dist, node))

        candidates.sort(key=lambda item: (item[0], item[1].id))
        return [item[1] for item in candidates[:limit]]

    async def count_nodes(self, kind: str | None = None) -> int:
        if kind is None:
            async with self._conn.execute(
                "SELECT COUNT(*) FROM wm_map_nodes"
            ) as cursor:
                row = await cursor.fetchone()
                return int(row[0]) if row else 0
        else:
            async with self._conn.execute(
                "SELECT COUNT(*) FROM wm_map_nodes WHERE kind = ?",
                (kind,),
            ) as cursor:
                row = await cursor.fetchone()
                return int(row[0]) if row else 0

    async def check_isolation(self) -> None:
        """Verify schema isolation for the underlying database connection.

        Raises WorldStoreError if any non-wm_ table exists.
        """
        try:
            await _run_sync(self._conn, assert_isolated, self._conn._connection)
        except SchemaError as err:
            raise WorldStoreError(f"Database schema isolation error: {err}") from err
        except Exception as err:
            raise WorldStoreError(f"Failed to check database isolation: {err}") from err

    async def statistics(self) -> dict[str, int]:
        """Gather database statistics for nodes, edges, and entities.

        Returns {"node_count": N, "edge_count": E, "entity_count": X}.
        """
        try:
            async with self._conn.execute(
                """
                SELECT
                    (SELECT COUNT(*) FROM wm_map_nodes) AS node_count,
                    (SELECT COUNT(*) FROM wm_map_edges) AS edge_count,
                    (SELECT COUNT(*) FROM wm_entities_seen) AS entity_count
                """
            ) as cursor:
                row = await cursor.fetchone()
                if row is None:
                    return {"node_count": 0, "edge_count": 0, "entity_count": 0}
                return {
                    "node_count": int(row[0]),
                    "edge_count": int(row[1]),
                    "entity_count": int(row[2]),
                }
        except (sqlite3.Error, aiosqlite.Error) as err:
            raise WorldStoreError(f"Failed to fetch statistics: {err}") from err
