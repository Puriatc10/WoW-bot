"""SQLite-backed async persistent memory store (Task 3.4).

Stores events along with the 5-dimensional internal drive state vector present
when each event occurred. Provides Euclidean-distance vector similarity recall,
time-based decay/retention, and async persistence using aiosqlite.

Public API:
    - :class:`MemoryStore`
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import aiosqlite
import numpy as np

from wow_bot.shared.interfaces import META_STATE_DIM, Event
from wow_bot.shared.logger import get_logger

log = get_logger("MEMORY")


class MemoryStore:
    """Async SQLite persistence engine for events and associated drive state vectors."""

    def __init__(self, db_path: str) -> None:
        self._db_path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        """Initialize database connection and schema. Safe to call multiple times."""
        if self._conn is not None:
            return

        # Ensure parent directory exists for file-backed databases.
        if self._db_path != ":memory:" and not self._db_path.startswith("file:"):
            db_file = Path(self._db_path)
            if db_file.parent != Path():
                db_file.parent.mkdir(parents=True, exist_ok=True)

        self._conn = await aiosqlite.connect(self._db_path)

        schema_sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            event_type TEXT NOT NULL,
            timestamp REAL NOT NULL,
            state_vector BLOB NOT NULL,
            data TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_event_type
        ON memories(event_type);

        CREATE INDEX IF NOT EXISTS idx_timestamp
        ON memories(timestamp);
        """
        try:
            await self._conn.executescript(schema_sql)
            await self._conn.commit()
        except BaseException:
            await self.close()
            raise
        log.info(f"Initialized MemoryStore at {self._db_path}")

    def _require_connection(self) -> aiosqlite.Connection:
        """Return active connection or raise RuntimeError if uninitialized/closed."""
        if self._conn is None:
            raise RuntimeError(
                "MemoryStore is not initialized or has been closed. Call init() first."
            )
        return self._conn

    @staticmethod
    def _validate_and_normalize_vector(vector: Any) -> np.ndarray:
        """Validate state vector shape, finiteness, and return float64 array of shape (5,)."""
        try:
            arr = np.asarray(vector, dtype=np.float64)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"State vector must be numeric, got {vector!r}") from exc

        if arr.shape != (META_STATE_DIM,):
            raise ValueError(
                f"State vector must have shape ({META_STATE_DIM},), got {arr.shape}"
            )

        if not np.all(np.isfinite(arr)):
            raise ValueError(f"State vector must not contain NaN or Inf values, got {arr.tolist()}")

        return arr

    async def add(
        self,
        event: Event,
        state_vector: np.ndarray,
    ) -> None:
        """Persist event and associated 5D internal drive state vector."""
        conn = self._require_connection()
        vec_arr = self._validate_and_normalize_vector(state_vector)

        blob = vec_arr.tobytes()
        payload_data = event.data if event.data is not None else {}
        try:
            data_json = json.dumps(payload_data)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Event data must be JSON-serializable: {exc}") from exc

        insert_sql = """
        INSERT INTO memories (event_type, timestamp, state_vector, data)
        VALUES (?, ?, ?, ?);
        """
        await conn.execute(
            insert_sql,
            (event.type, float(event.timestamp), blob, data_json),
        )
        await conn.commit()

    async def recall_similar(
        self,
        state: np.ndarray,
        k: int = 5,
    ) -> list[Event]:
        """Return up to k events whose stored state vectors are closest (Euclidean distance)."""
        if k <= 0:
            raise ValueError(f"k must be positive, got {k}")

        conn = self._require_connection()
        query_vec = self._validate_and_normalize_vector(state)

        select_sql = "SELECT id, event_type, timestamp, state_vector, data FROM memories;"
        async with conn.execute(select_sql) as cursor:
            rows = await cursor.fetchall()

        if not rows:
            return []

        candidates: list[tuple[float, int, str, float, str]] = []
        for row_id, event_type, timestamp, blob, data_json in rows:
            cand_vec = np.frombuffer(blob, dtype=np.float64)
            if cand_vec.shape != (META_STATE_DIM,):
                raise ValueError(
                    f"Corrupted state vector in database row {row_id}: "
                    f"expected shape ({META_STATE_DIM},), got {cand_vec.shape}"
                )

            dist = float(np.linalg.norm(cand_vec - query_vec))
            candidates.append((dist, int(row_id), str(event_type), float(timestamp), str(data_json)))

        # Deterministic ordering: distance ASC, id ASC
        candidates.sort(key=lambda item: (item[0], item[1]))

        reconstructed: list[Event] = []
        for _, _, event_type, timestamp, data_json in candidates[:k]:
            data_dict = json.loads(data_json) if data_json else {}
            reconstructed.append(
                Event(
                    type=event_type,
                    timestamp=timestamp,
                    data=data_dict,
                )
            )

        return reconstructed

    async def decay_old(
        self,
        max_age_hours: float = 72,
    ) -> None:
        """Delete memories older than the retention horizon relative to wall-clock time."""
        if max_age_hours < 0:
            raise ValueError(f"max_age_hours must be non-negative, got {max_age_hours}")

        conn = self._require_connection()
        cutoff = time.time() - max_age_hours * 3600.0

        delete_sql = "DELETE FROM memories WHERE timestamp < ?;"
        await conn.execute(delete_sql, (cutoff,))
        await conn.commit()

    async def close(self) -> None:
        """Close database resources safely and idempotently."""
        if self._conn is not None:
            await self._conn.close()
            self._conn = None
            log.info(f"Closed MemoryStore at {self._db_path}")
