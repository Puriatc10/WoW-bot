"""Acceptance tests for Task 1.2 — MetaState and Strategy.

Proves the two roadmap criteria:
1. ``Strategy`` round-trips through JSON (to/from dict).
2. Pydantic validation enforces ``0 <= risk_tolerance <= 1``.

Plus the architecture invariants AGENTS.md attaches to these contracts:
MetaState carries a finite float64 vector of shape (5,) bounded to [0, 1].
"""

from __future__ import annotations

import json
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from wow_bot.shared.interfaces import (
    DRIVE_NAMES,
    META_STATE_DIM,
    Event,
    MetaState,
    Strategy,
    StrategyModel,
)


def _sample_strategy() -> Strategy:
    return Strategy(
        goal="farm_herbs",
        region="silithus",
        risk_tolerance=0.35,
        priority=["avoid_elites", "sell_when_full"],
        constraints={"max_deaths_per_hour": 2, "min_mana_pct": 0.2},
        valid_until=1_700_001_800.0,
        raw_llm_output='{"goal": "farm_herbs"}',
    )


def _sample_meta_state() -> MetaState:
    return MetaState(
        vector=np.array([0.5, 0.7, 0.3, 0.4, 0.6], dtype=np.float64),
        recent_events=[
            Event(type="pvp_hit", timestamp=1_700_000_000.1, data={"source": "player"}),
            Event(type="death", timestamp=1_700_000_000.2),
        ],
        timestamp=1_700_000_000.5,
    )


# --- Acceptance criterion 1: Strategy round-trips through JSON -------------


def test_strategy_serializes_to_json() -> None:
    payload = json.dumps(_sample_strategy().to_dict())
    assert isinstance(payload, str)
    restored = json.loads(payload)
    assert restored["goal"] == "farm_herbs"
    assert restored["risk_tolerance"] == pytest.approx(0.35)
    assert restored["constraints"]["max_deaths_per_hour"] == 2


def test_strategy_json_round_trip_is_lossless() -> None:
    original = _sample_strategy()
    restored = Strategy.from_dict(json.loads(json.dumps(original.to_dict())))
    assert restored == original
    # Mutable containers are rebuilt as plain lists/dicts, not aliases.
    assert isinstance(restored.priority, list)
    assert isinstance(restored.constraints, dict)


def test_strategy_defaults_round_trip() -> None:
    """A minimal strategy (empty optional fields) still survives the trip."""
    sparse = Strategy(
        goal="explore",
        region="durotar",
        risk_tolerance=0.5,
        priority=[],
        constraints={},
        valid_until=0.0,
    )
    assert Strategy.from_dict(sparse.to_dict()) == sparse


def test_strategy_copies_containers_on_construction() -> None:
    """Callers mutating the input lists must not alias into the contract."""
    priority = ["a", "b"]
    constraints = {"k": 1}
    strategy = Strategy(
        goal="flee",
        region="hinterlands",
        risk_tolerance=0.1,
        priority=priority,
        constraints=constraints,
        valid_until=1.0,
    )
    priority.append("c")
    constraints["k"] = 99
    assert strategy.priority == ["a", "b"]
    assert strategy.constraints == {"k": 1}


# --- Acceptance criterion 2: Pydantic enforces 0 <= risk_tolerance <= 1 ----


@pytest.mark.parametrize("bad_risk", [-0.01, 1.01, 2.5, -3.0])
def test_from_dict_rejects_out_of_range_risk(bad_risk: float) -> None:
    data: dict[str, Any] = _sample_strategy().to_dict()
    data["risk_tolerance"] = bad_risk
    with pytest.raises(ValueError, match="risk_tolerance"):
        Strategy.from_dict(data)


@pytest.mark.parametrize("good_risk", [0.0, 0.5, 1.0])
def test_from_dict_accepts_boundary_risk(good_risk: float) -> None:
    data: dict[str, Any] = _sample_strategy().to_dict()
    data["risk_tolerance"] = good_risk
    assert Strategy.from_dict(data).risk_tolerance == pytest.approx(good_risk)


@pytest.mark.parametrize("bad_risk", [-0.01, 1.01])
def test_direct_construction_rejects_out_of_range_risk(bad_risk: float) -> None:
    """The dataclass mirrors the Pydantic bound so nothing bypasses it."""
    with pytest.raises(ValueError, match="risk_tolerance"):
        Strategy(
            goal="grind_humans",
            region="alterac",
            risk_tolerance=bad_risk,
            priority=[],
            constraints={},
            valid_until=1.0,
        )


def test_pydantic_model_enforces_bounds_directly() -> None:
    """The validator itself rejects out-of-range values (roadmap wording)."""
    with pytest.raises(ValidationError):
        StrategyModel(
            goal="explore",
            region="durotar",
            risk_tolerance=1.5,
            priority=[],
            constraints={},
            valid_until=1.0,
        )
    with pytest.raises(ValidationError):
        StrategyModel(
            goal="explore",
            region="durotar",
            risk_tolerance=-0.2,
            priority=[],
            constraints={},
            valid_until=1.0,
        )


def test_from_dict_rejects_missing_and_wrong_typed_fields() -> None:
    missing: dict[str, Any] = _sample_strategy().to_dict()
    del missing["region"]
    with pytest.raises(ValueError, match="invalid Strategy payload"):
        Strategy.from_dict(missing)

    wrong_type: dict[str, Any] = _sample_strategy().to_dict()
    wrong_type["priority"] = "not-a-list"
    with pytest.raises(ValueError, match="invalid Strategy payload"):
        Strategy.from_dict(wrong_type)


def test_from_dict_rejects_non_numeric_risk() -> None:
    data: dict[str, Any] = _sample_strategy().to_dict()
    data["risk_tolerance"] = "high"
    with pytest.raises(ValueError, match="invalid Strategy payload"):
        Strategy.from_dict(data)


# --- MetaState contract invariants -----------------------------------------


def test_meta_state_vector_normalized_to_float64_shape() -> None:
    state = MetaState(vector=[0.5, 0.5, 0.5, 0.5, 0.5], recent_events=[], timestamp=1.0)
    assert isinstance(state.vector, np.ndarray)
    assert state.vector.dtype == np.float64
    assert state.vector.shape == (META_STATE_DIM,)


def test_meta_state_drives_mapping_matches_drive_names() -> None:
    state = _sample_meta_state()
    drives = state.drives()
    assert tuple(drives.keys()) == DRIVE_NAMES
    assert drives["fatigue"] == pytest.approx(0.7)
    assert drives["social"] == pytest.approx(0.6)


def test_meta_state_json_round_trip_preserves_vector_and_events() -> None:
    original = _sample_meta_state()
    restored = MetaState.from_dict(json.loads(json.dumps(original.to_dict())))
    assert np.array_equal(restored.vector, original.vector)
    assert restored.timestamp == original.timestamp
    assert restored.recent_events == original.recent_events
    assert restored == original


@pytest.mark.parametrize(
    "bad_vector",
    [
        np.array([0.5, 0.5, 0.5, 0.5]),  # too short
        np.array([0.5] * 6),  # too long
        np.array([-0.01, 0.5, 0.5, 0.5, 0.5]),  # below range
        np.array([0.5, 0.5, 0.5, 0.5, 1.01]),  # above range
        np.array([np.nan, 0.5, 0.5, 0.5, 0.5]),  # non-finite
        np.array([np.inf, 0.5, 0.5, 0.5, 0.5]),  # non-finite
    ],
)
def test_meta_state_rejects_invalid_vectors(bad_vector: np.ndarray) -> None:
    with pytest.raises(ValueError):
        MetaState(vector=bad_vector, recent_events=[], timestamp=1.0)


def test_meta_state_from_dict_rejects_wrong_width() -> None:
    with pytest.raises(ValueError, match="vector"):
        MetaState.from_dict({"vector": [0.5, 0.5], "timestamp": 1.0})