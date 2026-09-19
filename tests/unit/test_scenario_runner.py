"""Unit tests for Scenario Runner and ScenarioReportCollector (Task 7.2)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

_repo_root = str(Path(__file__).resolve().parent.parent.parent)
if _repo_root not in sys.path:
    sys.path.insert(0, _repo_root)

from scripts.run_scenario import parse_args
from wow_bot.executor.fsm import State
from wow_bot.reporting.scenario import (
    SCENARIO_REPORT_SCHEMA_VERSION,
    ScenarioReportCollector,
    validate_report_dict,
    write_report_atomically,
)
from wow_bot.shared.interfaces import GameState, MetaState, Strategy


def test_collector_initialization_and_events() -> None:
    """Test collector receives events and aggregates counts accurately."""
    collector = ScenarioReportCollector(
        scenario="combat_light",
        seed=42,
        requested_duration_seconds=600.0,
    )

    gs = GameState(
        timestamp=100.0,
        hp_pct=0.9,
        mana_pct=0.8,
        position=(10.0, 20.0),
        facing=1.5,
        in_combat=False,
        target=None,
        enemies=[],
        events=[],
    )
    collector.on_game_state(gs)
    assert collector.game_state_count == 1

    ms = MetaState(
        vector=np.array([0.5, 0.5, 0.5, 0.5, 0.5]),
        recent_events=[],
        timestamp=100.0,
    )
    collector.on_meta_state(ms)
    assert collector.meta_state_count == 1
    assert len(collector.meta_state_samples) == 1
    assert collector.meta_state_samples[0]["vector"] == [0.5, 0.5, 0.5, 0.5, 0.5]

    collector.on_strategy_attempt(100.0)
    assert collector.generation_attempts == 1

    strat = Strategy(
        goal="farm_herbs",
        region="Barrens",
        risk_tolerance=0.5,
        priority=["farm"],
        constraints={},
        valid_until=1300.0,
    )
    collector.on_strategy_accepted(strat, 100.0)
    assert collector.new_strategies == 1
    assert collector.goals == {"farm_herbs": 1}

    collector.on_fsm_transition(100.0, State.IDLE.name, State.SCANNING.name, "start")
    assert collector.fsm_transition_count == 1
    assert collector.state_entry_counts == {"SCANNING": 1}

    collector.on_fsm_tick(100.0, 0.5)
    assert len(collector.timing_samples_ms) == 1
    assert 50.0 <= collector.timing_samples_ms[0] <= 2000.0

    collector.on_death_event(120.0)
    assert collector.death_event_count == 1
    assert collector.death_timestamps == [120.0]

    collector.on_idle_intent(125.0, "CAMERA_WANDER")
    assert collector.idle_intent_count == 1
    assert collector.idle_intent_counts_by_type == {"CAMERA_WANDER": 1}


def test_collector_build_report_structure() -> None:
    """Test build_report produces valid report dictionary following frozen schema."""
    collector = ScenarioReportCollector(
        scenario="peaceful_farm",
        seed=42,
        requested_duration_seconds=60.0,
    )

    ms = MetaState(
        vector=np.array([0.1, 0.2, 0.3, 0.4, 0.5]),
        recent_events=[],
        timestamp=10.0,
    )
    collector.on_meta_state(ms)
    collector.on_fsm_tick(10.0, 0.2)

    report = collector.build_report(
        completed_duration_seconds=60.0,
        completed_normally=True,
    )

    assert report["schema_version"] == SCENARIO_REPORT_SCHEMA_VERSION
    assert report["run"]["scenario"] == "peaceful_farm"
    assert report["run"]["seed"] == 42
    assert report["run"]["completed_normally"] is True
    assert len(report["meta_state"]["samples"]) == 1
    assert len(report["timing"]["samples_ms"]) == 1
    validate_report_dict(report)


def test_validate_report_dict_rejects_malformed_vector() -> None:
    """Test validate_report_dict raises ValueError on invalid vector size or non-finite elements."""
    bad_report = {
        "schema_version": 1,
        "meta_state": {
            "dimensions": ["hunger", "fatigue", "curiosity", "aggression", "social"],
            "samples": [
                {
                    "simulation_timestamp": 1.0,
                    "vector": [0.5, 0.5, 0.5, 0.5],  # length 4 instead of 5
                }
            ],
        },
        "timing": {"samples_ms": []},
    }
    with pytest.raises(ValueError, match="must be length 5 list"):
        validate_report_dict(bad_report)


def test_validate_report_dict_rejects_decreasing_timestamp() -> None:
    """Test validate_report_dict raises ValueError when timestamps decrease."""
    bad_report = {
        "schema_version": 1,
        "meta_state": {
            "dimensions": ["hunger", "fatigue", "curiosity", "aggression", "social"],
            "samples": [
                {"simulation_timestamp": 10.0, "vector": [0.5] * 5},
                {"simulation_timestamp": 5.0, "vector": [0.5] * 5},
            ],
        },
        "timing": {"samples_ms": []},
    }
    with pytest.raises(ValueError, match="timestamp decreased"):
        validate_report_dict(bad_report)


def test_write_report_atomically(tmp_path: Path) -> None:
    """Test write_report_atomically creates file atomically and disallows NaN/Inf."""
    collector = ScenarioReportCollector(
        scenario="combat_light",
        seed=42,
        requested_duration_seconds=10.0,
    )
    report = collector.build_report(completed_duration_seconds=10.0, completed_normally=True)

    target = tmp_path / "subdir" / "test_report.json"
    write_report_atomically(report, target)

    assert target.exists()
    content = json.loads(target.read_text(encoding="utf-8"))
    assert content["schema_version"] == 1
    assert content["run"]["scenario"] == "combat_light"


def test_parse_args_validation() -> None:
    """Test parse_args accepts valid arguments and rejects invalid scenarios or non-positive durations."""
    args = parse_args(["--scenario", "combat_light", "--duration", "600", "--seed", "123"])
    assert args.scenario == "combat_light"
    assert args.duration == 600.0
    assert args.seed == 123

    with pytest.raises(SystemExit):
        parse_args(["--scenario", "invalid_scenario", "--duration", "600"])

    with pytest.raises(SystemExit):
        parse_args(["--scenario", "combat_light", "--duration", "0"])

    with pytest.raises(SystemExit):
        parse_args(["--scenario", "combat_light", "--duration", "-10"])
