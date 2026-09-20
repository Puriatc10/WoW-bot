"""Tests for the Strategist Summary layer (T4.4)."""

from __future__ import annotations

import ast
import json
import sqlite3
from pathlib import Path

import pytest

from wow_bot.world.store import WorldModel
from wow_bot.world.summary import (
    NodeSummary,
    SummaryConfig,
    SummaryError,
    WorldSummary,
    summarize,
)

FIXED_NOW: str = "2026-01-01T12:00:00Z"


def test_summary_config_validation() -> None:
    # Valid defaults
    cfg = SummaryConfig()
    assert cfg.radius == 100.0
    assert cfg.per_kind_limit == 5
    assert cfg.recent_combat_limit == 5
    assert cfg.summary_version == 1

    # radius <= 0 raises ValueError
    with pytest.raises(ValueError, match="radius must be > 0"):
        SummaryConfig(radius=0.0)
    with pytest.raises(ValueError, match="radius must be > 0"):
        SummaryConfig(radius=-10.0)

    # per_kind_limit < 1 or > 50 raises ValueError
    with pytest.raises(ValueError, match="per_kind_limit"):
        SummaryConfig(per_kind_limit=0)
    with pytest.raises(ValueError, match="per_kind_limit"):
        SummaryConfig(per_kind_limit=51)

    # recent_combat_limit < 0 or > 50 raises ValueError
    with pytest.raises(ValueError, match="recent_combat_limit"):
        SummaryConfig(recent_combat_limit=-1)
    with pytest.raises(ValueError, match="recent_combat_limit"):
        SummaryConfig(recent_combat_limit=51)

    # summary_version < 1 raises ValueError
    with pytest.raises(ValueError, match="summary_version"):
        SummaryConfig(summary_version=0)


@pytest.mark.asyncio
async def test_summarize_malformed_around_xy_raises(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        with pytest.raises(SummaryError, match="around_xy"):
            await summarize(world, (1.0,))  # type: ignore[arg-type]

        with pytest.raises(SummaryError, match="around_xy"):
            await summarize(world, (1.0, 2.0, 3.0))  # type: ignore[arg-type]

        with pytest.raises(SummaryError, match="finite numbers"):
            await summarize(world, (float("nan"), 10.0))

        with pytest.raises(SummaryError, match="finite numbers"):
            await summarize(world, (10.0, float("inf")))


@pytest.mark.asyncio
async def test_summarize_empty_db(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        summary = await summarize(world, (10.0, 20.0), now=FIXED_NOW)

        assert isinstance(summary, WorldSummary)
        assert summary.around_xy == (10.0, 20.0)
        assert summary.radius == 100.0
        assert summary.generated_at == FIXED_NOW
        assert summary.nearest_vendors == ()
        assert summary.nearest_trainers == ()
        assert summary.nearest_nodes == ()
        assert summary.nearest_mobs == ()
        assert summary.nearest_waypoints == ()
        assert summary.recent_combats == ()
        assert summary.total_nodes == 0


@pytest.mark.asyncio
async def test_summarize_radius_filtering(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        # Center at (0, 0)
        n_close = await world.add_node(10.0, 0.0, kind="node")  # dist = 10
        n_far = await world.add_node(60.0, 0.0, kind="node")  # dist = 60

        cfg = SummaryConfig(radius=50.0)
        summary = await summarize(world, (0.0, 0.0), config=cfg, now=FIXED_NOW)

        node_ids = [n.id for n in summary.nearest_nodes]
        assert n_close in node_ids
        assert n_far not in node_ids


@pytest.mark.asyncio
async def test_summarize_per_kind_limit(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        for i in range(10):
            await world.add_node(float(i), 0.0, kind="mob")

        cfg = SummaryConfig(per_kind_limit=3)
        summary = await summarize(world, (0.0, 0.0), config=cfg, now=FIXED_NOW)

        assert len(summary.nearest_mobs) == 3


@pytest.mark.asyncio
async def test_summarize_preserves_query_nearest_ordering(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        # Nodes at dists 5.0, 5.0 (tie-breaker id), 1.0
        id1 = await world.add_node(3.0, 4.0, kind="waypoint")  # dist 5, id 1
        id2 = await world.add_node(-3.0, -4.0, kind="waypoint")  # dist 5, id 2
        id3 = await world.add_node(1.0, 0.0, kind="waypoint")  # dist 1, id 3

        summary = await summarize(world, (0.0, 0.0), now=FIXED_NOW)
        wp_ids = [w.id for w in summary.nearest_waypoints]

        # Expect dist 1 (id3) first, then dist 5 tie-broken by id ASC (id1 then id2)
        assert wp_ids == [id3, id1, id2]


@pytest.mark.asyncio
async def test_summarize_kind_separation_and_distance_rounding(
    tmp_path: Path,
) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        # Add vendor at (1.1111, 2.2222) -> dist from (0,0) = sqrt(1.1111^2 + 2.2222^2) = 2.48449...
        v_id = await world.add_node(1.1111, 2.2222, kind="vendor")
        m_id = await world.add_node(1.0, 1.0, kind="mob")

        summary = await summarize(world, (0.0, 0.0), now=FIXED_NOW)

        # Vendor appears in nearest_vendors, not in nearest_mobs
        vendor_ids = [v.id for v in summary.nearest_vendors]
        mob_ids = [m.id for m in summary.nearest_mobs]

        assert v_id in vendor_ids
        assert v_id not in mob_ids
        assert m_id in mob_ids
        assert m_id not in vendor_ids

        # Distance rounded to 3 decimals
        vendor_summary = summary.nearest_vendors[0]
        assert isinstance(vendor_summary, NodeSummary)
        assert vendor_summary.distance == 2.484


@pytest.mark.asyncio
async def test_summarize_recent_combats_ordering_and_limit(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        await world.record_combat(
            "mob_1",
            outcome="win",
            started_at="2026-01-01T10:00:00Z",
            ended_at="2026-01-01T10:01:00Z",
        )
        c2 = await world.record_combat(
            "mob_2",
            outcome="loss",
            started_at="2026-01-01T10:05:00Z",
            ended_at="2026-01-01T10:06:00Z",
        )
        c3 = await world.record_combat(
            "mob_3",
            outcome="flee",
            started_at="2026-01-01T10:05:00Z",
            ended_at="2026-01-01T10:06:00Z",
        )

        cfg_limit_2 = SummaryConfig(recent_combat_limit=2)
        summary = await summarize(
            world, (0.0, 0.0), config=cfg_limit_2, now=FIXED_NOW
        )

        assert len(summary.recent_combats) == 2
        # Order: ended_at DESC, combat_id DESC -> c3, c2
        assert [c.combat_id for c in summary.recent_combats] == [c3, c2]

        cfg_limit_0 = SummaryConfig(recent_combat_limit=0)
        summary_0 = await summarize(
            world, (0.0, 0.0), config=cfg_limit_0, now=FIXED_NOW
        )
        assert summary_0.recent_combats == ()


@pytest.mark.asyncio
async def test_world_summary_to_json(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        await world.add_node(1.0, 2.0, z=3.0, kind="trainer")
        await world.record_combat(
            "mob_x",
            outcome="win",
            started_at="2026-01-01T10:00:00Z",
            ended_at="2026-01-01T10:01:00Z",
        )

        summary = await summarize(world, (10.0, 20.0), now=FIXED_NOW)
        data = summary.to_json()

        assert isinstance(data, dict)
        assert isinstance(data["around_xy"], list)
        assert data["around_xy"] == [10.0, 20.0]
        assert data["radius"] == 100.0
        assert data["generated_at"] == FIXED_NOW
        assert isinstance(data["nearest_trainers"], list)
        assert isinstance(data["nearest_trainers"][0], dict)
        assert isinstance(data["recent_combats"], list)
        assert isinstance(data["recent_combats"][0], dict)

        # Note on summary_version: summary_version is configured via SummaryConfig
        # and is intentionally excluded from WorldSummary output schema.
        assert "summary_version" not in data

        # End-to-end JSON serializability test
        dumped = json.dumps(data)
        assert isinstance(dumped, str)
        reloaded = json.loads(dumped)
        assert reloaded == data


@pytest.mark.asyncio
async def test_summarize_determinism(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        await world.add_node(5.0, 5.0, kind="vendor")
        await world.record_combat(
            "mob_1",
            outcome="win",
            started_at="2026-01-01T10:00:00Z",
            ended_at="2026-01-01T10:01:00Z",
        )

        sum1 = await summarize(world, (0.0, 0.0), now=FIXED_NOW)
        sum2 = await summarize(world, (0.0, 0.0), now=FIXED_NOW)

        assert sum1 == sum2
        assert sum1.to_json() == sum2.to_json()


@pytest.mark.asyncio
async def test_summarize_does_not_modify_db(tmp_path: Path) -> None:
    db_file = tmp_path / "wm.db"
    async with await WorldModel.open(db_file) as world:
        await world.add_node(1.0, 1.0, kind="vendor")
        await world.add_node(2.0, 2.0, kind="mob")
        await world.record_combat(
            "mob_1",
            outcome="win",
            started_at="2026-01-01T10:00:00Z",
            ended_at="2026-01-01T10:01:00Z",
        )

        nodes_before = await world.count_nodes()

        # Capture direct DB row snapshot for wm_map_nodes
        raw = sqlite3.connect(db_file)
        rows_before = raw.execute(
            "SELECT * FROM wm_map_nodes ORDER BY id ASC"
        ).fetchall()
        raw.close()

        # Call summarize multiple times
        await summarize(world, (0.0, 0.0), now=FIXED_NOW)
        await summarize(world, (100.0, 100.0), now=FIXED_NOW)

        nodes_after = await world.count_nodes()

        raw = sqlite3.connect(db_file)
        rows_after = raw.execute(
            "SELECT * FROM wm_map_nodes ORDER BY id ASC"
        ).fetchall()
        raw.close()

        assert nodes_before == nodes_after
        assert rows_before == rows_after


def test_summary_module_static_ast_checks() -> None:
    summary_path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "wow_bot"
        / "world"
        / "summary.py"
    )
    source = summary_path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(summary_path))

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
        assert (
            mod not in forbidden_exact
        ), f"Forbidden import in summary.py: {mod}"
        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM import in summary.py: {mod}"
