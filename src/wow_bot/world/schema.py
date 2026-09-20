"""World model database schema definition and migration applier."""

from __future__ import annotations

import sqlite3
from pathlib import Path

SCHEMA_VERSION: int = 1

MIGRATIONS_DIR: Path = Path(__file__).resolve().parents[3] / "migrations" / "world_model"


class SchemaError(Exception):
    """Exception raised for world model database schema errors."""


def current_version(conn: sqlite3.Connection) -> int:
    """Return the current schema version of the world model database.

    Returns:
        0 if wm_schema_version table does not exist.
        MAX(version) from wm_schema_version otherwise.

    Raises:
        SchemaError: If wm_schema_version exists but contains no rows.
    """
    cursor = conn.cursor()
    cursor.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='wm_schema_version'"
    )
    if cursor.fetchone() is None:
        return 0

    cursor.execute("SELECT MAX(version) FROM wm_schema_version")
    row = cursor.fetchone()
    if row is None or row[0] is None:
        raise SchemaError("wm_schema_version table exists but contains no rows")

    return int(row[0])


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Apply outstanding migrations from current db_version to SCHEMA_VERSION.

    Note:
        Does NOT commit or rollback. The caller owns the transaction.
        Does NOT open or close the connection. The caller owns the connection.
    """
    db_ver = current_version(conn)

    if db_ver > SCHEMA_VERSION:
        raise SchemaError(
            f"Database version ({db_ver}) is newer than code schema version ({SCHEMA_VERSION})"
        )

    if db_ver == SCHEMA_VERSION:
        return SCHEMA_VERSION

    for ver in range(db_ver + 1, SCHEMA_VERSION + 1):
        filename = f"{ver:03d}_init.sql" if ver == 1 else f"{ver:03d}_migration.sql"
        # Search for exact file matching prefix or convention
        sql_file = MIGRATIONS_DIR / filename
        if not sql_file.is_file():
            # Try any filename starting with f"{ver:03d}_"
            matches = list(MIGRATIONS_DIR.glob(f"{ver:03d}_*.sql"))
            if matches:
                sql_file = matches[0]
            else:
                raise SchemaError(f"Migration script for version {ver} not found in {MIGRATIONS_DIR}")

        sql_script = sql_file.read_text(encoding="utf-8")
        conn.executescript(sql_script)

    return SCHEMA_VERSION


def list_non_wm_tables(conn: sqlite3.Connection) -> list[str]:
    """Query sqlite_master for table names that do NOT start with 'wm_'.

    Excludes 'sqlite_sequence'.
    Returns a sorted list of non-wm table names.
    """
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
    rows = cursor.fetchall()
    non_wm: list[str] = []
    for row in rows:
        tbl_name = row[0]
        if tbl_name == "sqlite_sequence":
            continue
        if not tbl_name.startswith("wm_"):
            non_wm.append(tbl_name)

    non_wm.sort()
    return non_wm


def assert_isolated(conn: sqlite3.Connection) -> None:
    """Assert that all user tables in the database are prefixed with 'wm_'.

    Raises:
        SchemaError: If any non-wm tables are present.
    """
    offending = list_non_wm_tables(conn)
    if offending:
        raise SchemaError(
            f"World model database isolation violated. Non-wm_ tables found: {offending}"
        )
