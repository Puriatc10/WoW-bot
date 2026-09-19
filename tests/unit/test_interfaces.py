"""Acceptance tests for Task 1.1 — GameState and sub-types.

Proves:
1. A ``GameState`` can be built and serialized to JSON (round-trip fidelity).
2. Contract invariants reject malformed values with clear errors.
3. All type hints pass ``mypy --strict`` (enforced by the project check, not here).
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from wow_bot.shared.interfaces import (
    EnemyInfo,
    Event,
    GameState,
    TargetInfo,
)


def _sample_state() -> GameState:
    return GameState(
        timestamp=1_700_000_000.5,
        hp_pct=0.82,
        mana_pct=0.44,
        position=(1234.5, -678.25),
        facing=1.5708,
        in_combat=True,
        target=TargetInfo(
            name="Defias Pillager",
            hp_pct=0.61,
            reaction="hostile",
            distance_estimate=8.5,
        ),
        enemies=[
            EnemyInfo(bbox=(100, 200, 30, 40), confidence=0.91, distance_estimate=12.0),
            EnemyInfo(bbox=(0, 0, 10, 10), confidence=0.5, distance_estimate=0.0),
        ],
        events=[
            Event(type="pvp_hit", timestamp=1_700_000_000.1, data={"source": "player"}),
            Event(type="death", timestamp=1_700_000_000.2),
        ],
    )


# --- Acceptance criterion 1: build + serialize to JSON ---------------------


def test_game_state_serializes_to_json() -> None:
    state = _sample_state()
    payload = json.dumps(state.to_dict())  # must not raise
    assert isinstance(payload, str)
    restored = json.loads(payload)
    assert restored["hp_pct"] == pytest.approx(0.82)
    assert restored["target"]["name"] == "Defias Pillager"
    assert len(restored["enemies"]) == 2
    assert len(restored["events"]) == 2


def test_game_state_json_round_trip_preserves_types() -> None:
    original = _sample_state()
    restored = GameState.from_dict(original.to_dict())

    assert restored == original
    # Tuple-typed geometry survives the round trip as tuples, not lists.
    assert isinstance(restored.position, tuple)
    assert isinstance(restored.enemies[0].bbox, tuple)
    assert restored.target is not None
    assert restored.target.reaction == "hostile"


def test_none_target_round_trips() -> None:
    state = _sample_state()
    state.target = None
    restored = GameState.from_dict(json.loads(json.dumps(state.to_dict())))
    assert restored.target is None
    assert restored == state


def test_empty_lists_round_trip() -> None:
    state = GameState(
        timestamp=0.0,
        hp_pct=1.0,
        mana_pct=1.0,
        position=(0.0, 0.0),
        facing=0.0,
        in_combat=False,
        target=None,
        enemies=[],
        events=[],
    )
    assert GameState.from_dict(state.to_dict()) == state


# --- Negative paths: contract invariants -----------------------------------


@pytest.mark.parametrize("hp", [-0.01, 1.01])
def test_hp_out_of_range_rejected(hp: float) -> None:
    with pytest.raises(ValueError, match="hp_pct"):
        _sample_state_with(hp_pct=hp)


def test_mana_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="mana_pct"):
        _sample_state_with(mana_pct=1.5)


def test_bad_reaction_rejected() -> None:
    with pytest.raises(ValueError, match="reaction"):
        TargetInfo(name="x", hp_pct=0.5, reaction="friendly-ish", distance_estimate=1.0)


def test_negative_distance_rejected() -> None:
    with pytest.raises(ValueError, match="distance_estimate"):
        TargetInfo(name="x", hp_pct=0.5, reaction="hostile", distance_estimate=-1.0)


def test_confidence_out_of_range_rejected() -> None:
    with pytest.raises(ValueError, match="confidence"):
        EnemyInfo(bbox=(0, 0, 1, 1), confidence=1.2, distance_estimate=5.0)


def test_bbox_wrong_length_rejected() -> None:
    with pytest.raises(ValueError, match="bbox"):
        EnemyInfo(bbox=(0, 0, 1), confidence=0.5, distance_estimate=5.0)  # type: ignore[arg-type]


def test_event_requires_type() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        Event(type="", timestamp=1.0)


def test_from_dict_rejects_short_position() -> None:
    data: dict[str, Any] = _sample_state().to_dict()
    data["position"] = [1.0]
    with pytest.raises(ValueError, match="position"):
        GameState.from_dict(data)


def _sample_state_with(**overrides: Any) -> GameState:
    base: dict[str, Any] = {
        "timestamp": 1.0,
        "hp_pct": 0.5,
        "mana_pct": 0.5,
        "position": (0.0, 0.0),
        "facing": 0.0,
        "in_combat": False,
        "target": None,
        "enemies": [],
        "events": [],
    }
    base.update(overrides)
    return GameState(**base)