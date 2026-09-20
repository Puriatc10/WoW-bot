"""Tests for the World Model database loader."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path
from typing import Any

import pytest

import wow_bot.world.store as store_module
from wow_bot.config import Config
from wow_bot.session import Session
from wow_bot.world.loader import (
    LoaderConfig,
    LoaderError,
    LoaderReport,
    load_world,
)
from wow_bot.world.schema import SCHEMA_VERSION
from wow_bot.world.store import WorldModel, WorldStoreError


@pytest.mark.asyncio
async def test_loader_config_empty_db_path_raises() -> None:
    with pytest.raises(ValueError, match="db_path"):
        LoaderConfig(db_path="")


@pytest.mark.asyncio
async def test_loader_config_whitespace_db_path_raises() -> None:
    with pytest.raises(ValueError, match="db_path"):
        LoaderConfig(db_path="   \t\n")
    with pytest.raises(ValueError, match="db_path"):
        LoaderConfig(db_path="/path/with space/wm.db")


@pytest.mark.asyncio
async def test_loader_config_budget_out_of_bounds_raises() -> None:
    with pytest.raises(ValueError, match="startup_budget_ms"):
        LoaderConfig(db_path="wm.db", startup_budget_ms=49)
    with pytest.raises(ValueError, match="startup_budget_ms"):
        LoaderConfig(db_path="wm.db", startup_budget_ms=10_001)


@pytest.mark.asyncio
async def test_load_world_require_existing_missing_file_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "nonexistent.db"
    config = LoaderConfig(db_path=str(db_file), require_existing=True)
    with pytest.raises(LoaderError, match="Required database file does not exist"):
        await load_world(config)


@pytest.mark.asyncio
async def test_load_world_is_dir_raises(tmp_path: Path) -> None:
    dir_path = tmp_path / "a_dir"
    dir_path.mkdir()
    config = LoaderConfig(db_path=str(dir_path))
    with pytest.raises(LoaderError, match="directory"):
        await load_world(config)


@pytest.mark.asyncio
async def test_load_world_creates_db_when_not_requiring_existing(tmp_path: Path) -> None:
    db_file = tmp_path / "new_wm.db"
    assert not db_file.exists()

    config = LoaderConfig(db_path=str(db_file), require_existing=False)
    world, report = await load_world(config)
    try:
        assert db_file.exists()
        assert report.opened_existing is False
        assert report.node_count == 0
        assert report.edge_count == 0
        assert report.entity_count == 0
        assert report.duration_ms > 0.0
    finally:
        await world.close()


@pytest.mark.asyncio
async def test_load_world_existing_empty_file_returns_opened_existing_false(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "empty_existing.db"
    db_file.touch()  # creates a 0-byte file
    assert db_file.stat().st_size == 0

    config = LoaderConfig(db_path=str(db_file), require_existing=True)
    world, report = await load_world(config)
    try:
        assert report.opened_existing is False
    finally:
        await world.close()


@pytest.mark.asyncio
async def test_load_world_existing_populated_file_returns_opened_existing_true(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "populated.db"
    # Open and populate initial world
    w_init = await WorldModel.open(db_file)
    await w_init.add_node(1.0, 2.0, kind="vendor")
    await w_init.close()
    assert db_file.stat().st_size > 0

    config = LoaderConfig(db_path=str(db_file), require_existing=True)
    world, report = await load_world(config)
    try:
        assert report.opened_existing is True
        assert report.node_count == 1
    finally:
        await world.close()


@pytest.mark.asyncio
async def test_load_world_validates_isolation_by_default(tmp_path: Path) -> None:
    db_file = tmp_path / "non_wm.db"
    # Create DB with non-wm_ table
    raw = sqlite3.connect(db_file)
    raw.execute("CREATE TABLE non_wm_bad (id INT);")
    raw.commit()
    raw.close()

    config = LoaderConfig(db_path=str(db_file), validate_isolation=True)
    with pytest.raises(LoaderError, match="isolation"):
        await load_world(config)


@pytest.mark.asyncio
async def test_load_world_validate_isolation_disabled_succeeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    db_file = tmp_path / "non_wm_allowed.db"

    # Create DB with non-wm_ table
    raw = sqlite3.connect(db_file)
    raw.execute("CREATE TABLE non_wm_ignored (id INT);")
    raw.commit()
    raw.close()

    # Monkeypatch assert_isolated during WorldModel.open so open succeeds
    def mock_assert_isolated_on_open(conn: Any) -> None:
        pass  # Bypass open-time isolation check

    monkeypatch.setattr(store_module, "assert_isolated", mock_assert_isolated_on_open)

    config = LoaderConfig(db_path=str(db_file), validate_isolation=False)
    world, report = await load_world(config)
    try:
        assert isinstance(world, WorldModel)
        assert report.db_path == str(db_file)
    finally:
        await world.close()


@pytest.mark.asyncio
async def test_loader_report_to_json_and_schema_version(tmp_path: Path) -> None:
    report = LoaderReport(
        db_path="/tmp/test.db",
        schema_version=SCHEMA_VERSION,
        opened_existing=True,
        duration_ms=12.34,
        node_count=10,
        edge_count=5,
        entity_count=2,
    )
    res = report.to_json()
    assert isinstance(res, dict)
    assert res["db_path"] == "/tmp/test.db"
    assert res["schema_version"] == SCHEMA_VERSION
    assert res["opened_existing"] is True
    assert res["duration_ms"] == 12.34
    assert res["node_count"] == 10
    assert res["edge_count"] == 5
    assert res["entity_count"] == 2

    # Verify JSON serializability
    serialized = json.dumps(res)
    assert json.loads(serialized) == res


@pytest.mark.asyncio
async def test_load_world_budget_exceeded_closes_world_and_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # We use monkeypatching on `wow_bot.world.loader.time.perf_counter_ns` to simulate a duration exceeding startup_budget_ms.
    # This strategy is chosen to keep the test fast, deterministic, and isolated.
    db_file = tmp_path / "slow.db"
    config = LoaderConfig(db_path=str(db_file), startup_budget_ms=100)

    # Mock time.perf_counter_ns to advance time by 200 ms (200_000_000 ns)
    ticks = [100_000_000, 300_000_000]

    def mock_perf_counter_ns() -> int:
        if ticks:
            return ticks.pop(0)
        return 300_000_000

    monkeypatch.setattr("wow_bot.world.loader.time.perf_counter_ns", mock_perf_counter_ns)

    with pytest.raises(LoaderError, match="exceeded startup budget"):
        await load_world(config)

    # Verify connection was closed on the failed DB file by opening it again successfully
    world2 = await WorldModel.open(db_file)
    await world2.close()


@pytest.mark.asyncio
async def test_load_world_session_event_emission(tmp_path: Path) -> None:
    db_file = tmp_path / "session_test.db"
    cfg = Config(
        lab_mode=False,
        server_allowlist=(),
        isolation_sentinel="127.0.0.1:8080",
        kill_switch_key="f12",
        session_root=tmp_path / "sessions",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    session = Session.start(cfg)

    config = LoaderConfig(db_path=str(db_file))
    world, _report = await load_world(config, session=session)
    await world.close()

    events_file = session.path / "events.jsonl"
    session.close("test_complete")

    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    loaded_events = [json.loads(line) for line in lines if json.loads(line).get("event") == "world_loaded"]

    assert len(loaded_events) == 1
    evt = loaded_events[0]
    assert evt["event"] == "world_loaded"
    assert "payload" in evt
    assert evt["payload"]["db_path"] == str(db_file)
    assert evt["payload"]["schema_version"] == SCHEMA_VERSION


@pytest.mark.asyncio
async def test_load_world_no_session_attached(tmp_path: Path) -> None:
    db_file = tmp_path / "no_session.db"
    config = LoaderConfig(db_path=str(db_file))
    world, report = await load_world(config, session=None)
    try:
        assert report.db_path == str(db_file)
    finally:
        await world.close()


@pytest.mark.asyncio
async def test_world_model_check_isolation_and_statistics_focused(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "focused.db"
    world = await WorldModel.open(db_file)
    try:
        # check_isolation passes on fresh DB
        await world.check_isolation()

        # statistics returns 0s
        stats = await world.statistics()
        assert stats == {"node_count": 0, "edge_count": 0, "entity_count": 0}

        # Insert nodes, edges, entities
        n1 = await world.add_node(0.0, 0.0, kind="node")
        n2 = await world.add_node(1.0, 1.0, kind="node")
        await world.add_edge(n1, n2, cost=1.0)
        await world.mark_seen("ent1", kind="mob", x=0.0, y=0.0)

        stats2 = await world.statistics()
        assert stats2 == {"node_count": 2, "edge_count": 2, "entity_count": 1}

        # Manually create non-wm_ table
        await world._conn.execute("CREATE TABLE non_wm_bad_table (id INT);")
        await world._conn.commit()

        # check_isolation raises WorldStoreError
        with pytest.raises(WorldStoreError, match="isolation"):
            await world.check_isolation()
    finally:
        await world.close()


def test_loader_module_static_ast_checks() -> None:
    loader_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wow_bot"
        / "world"
        / "loader.py"
    )
    source = loader_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(loader_path))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.internal_dynamics",
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
        assert mod not in forbidden_exact, f"Forbidden import in loader.py: {mod}"
        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM import in loader.py: {mod}"
