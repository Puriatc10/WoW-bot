"""Tests for World Model database schema definition, migrations, and isolation."""

from __future__ import annotations

import ast
import sqlite3
from pathlib import Path

import pytest

from wow_bot.world.schema import (
    MIGRATIONS_DIR,
    SchemaError,
    apply_migrations,
    assert_isolated,
    current_version,
    list_non_wm_tables,
)


def _connect_db() -> sqlite3.Connection:
    """Helper to open an in-memory SQLite connection with foreign keys enabled."""
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def test_migration_file_exists() -> None:
    """Acceptance: migrations/world_model/001_init.sql exists at expected path."""
    init_sql = MIGRATIONS_DIR / "001_init.sql"
    assert init_sql.exists()
    assert init_sql.is_file()


def test_apply_migrations_fresh_db() -> None:
    """Acceptance: Applying migrations on a fresh in-memory DB returns version 1."""
    conn = _connect_db()
    try:
        version = apply_migrations(conn)
        assert version == 1
    finally:
        conn.close()


def test_wm_schema_version_row() -> None:
    """Acceptance: After migration, wm_schema_version contains exactly one row with version=1 and non-empty applied_at."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT version, applied_at FROM wm_schema_version")
        rows = cursor.fetchall()
        assert len(rows) == 1
        ver, applied_at = rows[0]
        assert ver == 1
        assert isinstance(applied_at, str)
        assert len(applied_at) > 0
    finally:
        conn.close()


def test_domain_tables_exist() -> None:
    """Acceptance: After migration, exactly the five domain tables exist and all start with wm_."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name != 'sqlite_sequence' AND name != 'wm_schema_version' ORDER BY name"
        )
        tables = [row[0] for row in cursor.fetchall()]
        expected_domain_tables = [
            "wm_combat_history",
            "wm_entities_seen",
            "wm_map_edges",
            "wm_map_nodes",
            "wm_routes_taken",
        ]
        assert tables == expected_domain_tables
        assert all(t.startswith("wm_") for t in tables)
    finally:
        conn.close()


def test_list_non_wm_tables_empty() -> None:
    """Acceptance: list_non_wm_tables returns an empty list on a freshly migrated DB."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        non_wm = list_non_wm_tables(conn)
        assert non_wm == []
    finally:
        conn.close()


def test_list_non_wm_tables_excludes_sqlite_sequence() -> None:
    """Acceptance: list_non_wm_tables does NOT include sqlite_sequence."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        # Force creation of sqlite_sequence by inserting into an AUTOINCREMENT table
        conn.execute(
            "INSERT INTO wm_map_nodes (x, y, kind, discovered_at, last_seen_at) VALUES (1.0, 2.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='sqlite_sequence'")
        assert cursor.fetchone() is not None

        non_wm = list_non_wm_tables(conn)
        assert "sqlite_sequence" not in non_wm
        assert non_wm == []
    finally:
        conn.close()


def test_assert_isolated_passes() -> None:
    """Acceptance: assert_isolated passes on a freshly migrated DB."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        assert_isolated(conn)
    finally:
        conn.close()


def test_assert_isolated_raises() -> None:
    """Acceptance: assert_isolated raises SchemaError when an extra table without wm_ prefix is created manually."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        conn.execute("CREATE TABLE non_wm_extra (id INTEGER PRIMARY KEY);")
        with pytest.raises(SchemaError, match="non_wm_extra"):
            assert_isolated(conn)
    finally:
        conn.close()


def test_apply_migrations_idempotent() -> None:
    """Acceptance: Applying migrations twice is idempotent: second call returns 1 and does not raise."""
    conn = _connect_db()
    try:
        v1 = apply_migrations(conn)
        v2 = apply_migrations(conn)
        assert v1 == 1
        assert v2 == 1
    finally:
        conn.close()


def test_current_version_empty_db() -> None:
    """Acceptance: current_version returns 0 on an empty DB."""
    conn = _connect_db()
    try:
        assert current_version(conn) == 0
    finally:
        conn.close()


def test_current_version_after_migration() -> None:
    """Acceptance: current_version returns 1 after migration."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        assert current_version(conn) == 1
    finally:
        conn.close()


def test_current_version_empty_schema_table() -> None:
    """Acceptance: current_version raises SchemaError if wm_schema_version exists but is empty."""
    conn = _connect_db()
    try:
        conn.execute(
            "CREATE TABLE wm_schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);"
        )
        with pytest.raises(SchemaError, match="contains no rows"):
            current_version(conn)
    finally:
        conn.close()


def test_check_constraint_node_kind() -> None:
    """Acceptance: Inserting a node with an invalid kind raises sqlite3.IntegrityError (CHECK constraint)."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_map_nodes (x, y, kind, discovered_at, last_seen_at) VALUES (0.0, 0.0, 'invalid_kind', '2026-01-01', '2026-01-01')"
            )
    finally:
        conn.close()


def test_foreign_key_edge_from_id() -> None:
    """Acceptance: Inserting an edge with a from_id that does not exist raises sqlite3.IntegrityError (FK)."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at) VALUES (999, 1000, 1.0, 1, '2026-01-01')"
            )
    finally:
        conn.close()


def test_cascade_delete_node() -> None:
    """Acceptance: Deleting a node cascades to its edges (ON DELETE CASCADE)."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (1, 0.0, 0.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (2, 1.0, 1.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at) VALUES (1, 2, 1.5, 1, '2026-01-01')"
        )

        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM wm_map_edges")
        assert cursor.fetchone()[0] == 1

        conn.execute("DELETE FROM wm_map_nodes WHERE id = 1")

        cursor.execute("SELECT COUNT(*) FROM wm_map_edges")
        assert cursor.fetchone()[0] == 0
    finally:
        conn.close()


def test_check_constraint_edge_cost() -> None:
    """Acceptance: Inserting an edge with negative cost raises IntegrityError."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (1, 0.0, 0.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (2, 1.0, 1.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at) VALUES (1, 2, -1.0, 1, '2026-01-01')"
            )
    finally:
        conn.close()


def test_check_constraint_edge_bidirectional() -> None:
    """Acceptance: Inserting an edge with bidirectional=2 raises IntegrityError."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (1, 0.0, 0.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (2, 1.0, 1.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_map_edges (from_id, to_id, cost, bidirectional, discovered_at) VALUES (1, 2, 1.0, 2, '2026-01-01')"
            )
    finally:
        conn.close()


def test_check_constraint_routes_succeeded() -> None:
    """Acceptance: Inserting a route with succeeded=2 raises IntegrityError."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (1, 0.0, 0.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        conn.execute(
            "INSERT INTO wm_map_nodes (id, x, y, kind, discovered_at, last_seen_at) VALUES (2, 1.0, 1.0, 'waypoint', '2026-01-01', '2026-01-01')"
        )
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_routes_taken (from_id, to_id, succeeded, taken_at) VALUES (1, 2, 2, '2026-01-01')"
            )
    finally:
        conn.close()


def test_check_constraint_combat_outcome() -> None:
    """Acceptance: Inserting a combat history with an invalid outcome raises IntegrityError."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute(
                "INSERT INTO wm_combat_history (target_entity_id, outcome, started_at, ended_at) VALUES ('e1', 'invalid_outcome', '2026-01-01', '2026-01-01')"
            )
    finally:
        conn.close()


def test_required_indices_exist() -> None:
    """Acceptance: The seven required indices exist on a migrated DB."""
    conn = _connect_db()
    try:
        apply_migrations(conn)
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index'")
        indices = {row[0] for row in cursor.fetchall()}
        required = {
            "idx_wm_map_nodes_kind",
            "idx_wm_map_nodes_xy",
            "idx_wm_map_edges_from",
            "idx_wm_map_edges_to",
            "idx_wm_entities_seen_kind",
            "idx_wm_routes_taken_from",
            "idx_wm_combat_history_target",
        }
        assert required.issubset(indices)
    finally:
        conn.close()


def test_migration_file_static_table_scan() -> None:
    """Acceptance: The migration file does NOT contain any CREATE TABLE without the wm_ prefix (static text scan)."""
    sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8")
    lines = sql.splitlines()
    for line in lines:
        upper = line.upper().strip()
        if "CREATE TABLE" in upper:
            assert "CREATE TABLE WM_" in upper or "CREATE TABLE IF NOT EXISTS WM_" in upper, (
                f"Forbidden CREATE TABLE without wm_ prefix: {line}"
            )


def test_migration_file_static_prohibited_structures() -> None:
    """Acceptance: The migration file does NOT contain CREATE TRIGGER, CREATE VIEW, or CREATE VIRTUAL TABLE."""
    sql = (MIGRATIONS_DIR / "001_init.sql").read_text(encoding="utf-8").upper()
    assert "CREATE TRIGGER" not in sql
    assert "CREATE VIEW" not in sql
    assert "CREATE VIRTUAL TABLE" not in sql


def test_apply_migrations_does_not_commit() -> None:
    """Acceptance: apply_migrations does NOT call conn.commit(). The caller owns the transaction."""

    class TracedConnection(sqlite3.Connection):
        def __init__(self, *args: object, **kwargs: object) -> None:
            super().__init__(*args, **kwargs)  # type: ignore[arg-type]
            self.commit_called = False

        def commit(self) -> None:
            self.commit_called = True
            super().commit()

    conn = sqlite3.connect(":memory:", factory=TracedConnection)
    conn.execute("PRAGMA foreign_keys = ON;")
    try:
        apply_migrations(conn)
        assert not conn.commit_called, "apply_migrations called conn.commit() unexpectedly"
    finally:
        conn.close()


def test_apply_migrations_newer_db_version_raises() -> None:
    """Acceptance: apply_migrations with a SCHEMA_VERSION lower than the DB version raises SchemaError."""
    conn = _connect_db()
    try:
        conn.execute(
            "CREATE TABLE wm_schema_version (version INTEGER PRIMARY KEY, applied_at TEXT NOT NULL);"
        )
        conn.execute("INSERT INTO wm_schema_version (version, applied_at) VALUES (2, '2026-01-01');")
        with pytest.raises(SchemaError, match="newer than code schema version"):
            apply_migrations(conn)
    finally:
        conn.close()


def test_schema_module_static_ast_checks() -> None:
    """Acceptance: schema.py does not import forbidden modules."""
    schema_path = Path(__file__).resolve().parents[1] / "src" / "wow_bot" / "world" / "schema.py"
    source = schema_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(schema_path))

    forbidden_exact = {
        "aiosqlite",
        "asyncio",
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
    }
    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)

    for mod in imported_modules:
        assert mod not in forbidden_exact, f"Forbidden import found in schema.py: {mod}"
        for sub in forbidden_substrings:
            assert sub not in mod.lower(), f"Forbidden LLM import found in schema.py: {mod}"
