"""Acceptance tests for Task 1.3 — Event Types Registry.

Roadmap acceptance criteria:
1. Every event type has an entry in EVENT_EFFECTS.
2. mypy --strict passes (verified externally; structural assertions here).

Additional invariant checks per AGENTS.md:
- Drive keys in every effect row match DRIVE_NAMES from interfaces.
- Effect values are finite floats.
- The table is immutable (MappingProxyType).
- Unknown event lookup raises KeyError with actionable message.
"""

from __future__ import annotations

import math

import pytest

from wow_bot.shared.events import (
    ALL_EVENT_TYPES,
    DEATH,
    EVENT_EFFECTS,
    EVENT_LABELS,
    LEVEL_UP,
    NPC_INTERACT,
    PVP_HIT,
    PVP_KILL,
    QUEST_COMPLETE,
    RARE_LOOT,
    STUCK,
    effects_for,
    is_known_event,
    total_effect_magnitude,
)
from wow_bot.shared.interfaces import DRIVE_NAMES

# ---------------------------------------------------------------------------
# Acceptance criterion 1: every event type has an entry in EVENT_EFFECTS
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "event_type",
    [DEATH, RARE_LOOT, PVP_HIT, PVP_KILL, STUCK, LEVEL_UP, QUEST_COMPLETE, NPC_INTERACT],
)
def test_every_declared_constant_has_an_effects_entry(event_type: str) -> None:
    assert event_type in EVENT_EFFECTS


def test_all_event_types_tuple_covers_every_effects_key() -> None:
    assert set(ALL_EVENT_TYPES) == set(EVENT_EFFECTS.keys())


def test_no_extra_or_missing_keys_between_constants_and_table() -> None:
    constants = {DEATH, RARE_LOOT, PVP_HIT, PVP_KILL, STUCK, LEVEL_UP, QUEST_COMPLETE, NPC_INTERACT}
    assert constants == set(EVENT_EFFECTS.keys())


# ---------------------------------------------------------------------------
# Structural invariants of the effect rows
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_each_row_carries_exactly_the_five_drive_keys(event_type: str) -> None:
    row = EVENT_EFFECTS[event_type]
    assert set(row.keys()) == set(DRIVE_NAMES)
    assert len(row) == len(DRIVE_NAMES)


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_each_delta_is_a_finite_float(event_type: str) -> None:
    row = EVENT_EFFECTS[event_type]
    for drive_name in DRIVE_NAMES:
        value = row[drive_name]
        assert isinstance(value, float), f"{event_type}.{drive_name} is not a float"
        assert math.isfinite(value), f"{event_type}.{drive_name} is not finite"


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_each_row_is_read_only_mapping(event_type: str) -> None:
    row = EVENT_EFFECTS[event_type]
    with pytest.raises(TypeError):
        row["hunger"] = 0.0  # type: ignore[index]


def test_effects_table_itself_is_immutable() -> None:
    with pytest.raises(TypeError):
        EVENT_EFFECTS["bogus"] = {}  # type: ignore[index]


# ---------------------------------------------------------------------------
# Spot-check specific roadmap-specified values (guard against silent edits)
# ---------------------------------------------------------------------------


def test_death_depresses_aggression_and_social_while_raising_fatigue() -> None:
    row = EVENT_EFFECTS[DEATH]
    assert row["aggression"] == pytest.approx(-0.15)
    assert row["social"] == pytest.approx(-0.05)
    assert row["fatigue"] == pytest.approx(+0.10)
    assert row["hunger"] == pytest.approx(0.0)
    assert row["curiosity"] == pytest.approx(0.0)


def test_rare_loot_spikes_hunger_and_curiosity() -> None:
    row = EVENT_EFFECTS[RARE_LOOT]
    assert row["hunger"] == pytest.approx(+0.20)
    assert row["curiosity"] == pytest.approx(+0.15)
    assert row["social"] == pytest.approx(+0.05)
    assert row["fatigue"] == pytest.approx(0.0)
    assert row["aggression"] == pytest.approx(0.0)


def test_stuck_drains_curiosity_and_aggression_but_raises_fatigue() -> None:
    row = EVENT_EFFECTS[STUCK]
    assert row["fatigue"] == pytest.approx(+0.15)
    assert row["curiosity"] == pytest.approx(-0.10)
    assert row["aggression"] == pytest.approx(-0.05)
    assert row["hunger"] == pytest.approx(0.0)
    assert row["social"] == pytest.approx(0.0)


def test_npc_interact_boosts_social() -> None:
    row = EVENT_EFFECTS[NPC_INTERACT]
    assert row["social"] == pytest.approx(+0.15)
    assert row["fatigue"] == pytest.approx(-0.05)
    assert row["hunger"] == pytest.approx(0.0)
    assert row["curiosity"] == pytest.approx(0.0)
    assert row["aggression"] == pytest.approx(0.0)


def test_level_up_reduces_fatigue_and_raises_others() -> None:
    row = EVENT_EFFECTS[LEVEL_UP]
    assert row["fatigue"] == pytest.approx(-0.10)
    assert row["hunger"] == pytest.approx(+0.10)
    assert row["curiosity"] == pytest.approx(+0.10)
    assert row["aggression"] == pytest.approx(+0.05)
    assert row["social"] == pytest.approx(+0.05)


def test_quest_complete_values_match_roadmap() -> None:
    row = EVENT_EFFECTS[QUEST_COMPLETE]
    assert row["hunger"] == pytest.approx(+0.10)
    assert row["fatigue"] == pytest.approx(-0.05)
    assert row["curiosity"] == pytest.approx(+0.10)
    assert row["aggression"] == pytest.approx(0.0)
    assert row["social"] == pytest.approx(+0.05)


def test_pvp_hit_values_match_roadmap() -> None:
    row = EVENT_EFFECTS[PVP_HIT]
    assert row["fatigue"] == pytest.approx(+0.05)
    assert row["curiosity"] == pytest.approx(-0.05)
    assert row["aggression"] == pytest.approx(+0.10)
    assert row["social"] == pytest.approx(-0.10)
    assert row["hunger"] == pytest.approx(0.0)


def test_pvp_kill_values_match_roadmap() -> None:
    row = EVENT_EFFECTS[PVP_KILL]
    assert row["aggression"] == pytest.approx(+0.05)
    assert row["social"] == pytest.approx(+0.05)
    assert row["hunger"] == pytest.approx(0.0)
    assert row["fatigue"] == pytest.approx(0.0)
    assert row["curiosity"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_effects_for_returns_correct_row(event_type: str) -> None:
    row = effects_for(event_type)
    assert row == EVENT_EFFECTS[event_type]


def test_effects_for_unknown_type_raises_keyerror_with_hint() -> None:
    with pytest.raises(KeyError, match="unknown event type"):
        effects_for("totally_bogus")


def test_is_known_event_true_for_registered_types() -> None:
    for event_type in ALL_EVENT_TYPES:
        assert is_known_event(event_type) is True


def test_is_known_event_false_for_unregistered_strings() -> None:
    assert is_known_event("") is False
    assert is_known_event("nonexistent") is False
    assert is_known_event("Death") is False  # case-sensitive


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_total_effect_magnitude_is_nonnegative(event_type: str) -> None:
    mag = total_effect_magnitude(event_type)
    assert mag >= 0.0


def test_total_effect_magnitude_sums_absolute_deltas() -> None:
    row = EVENT_EFFECTS[DEATH]
    expected = sum(abs(v) for v in row.values())
    assert total_effect_magnitude(DEATH) == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Labels mapping
# ---------------------------------------------------------------------------


def test_labels_cover_every_event_type() -> None:
    assert set(EVENT_LABELS.keys()) == set(ALL_EVENT_TYPES)


@pytest.mark.parametrize("event_type", ALL_EVENT_TYPES)
def test_label_is_nonempty_string(event_type: str) -> None:
    label = EVENT_LABELS[event_type]
    assert isinstance(label, str)
    assert len(label) > 0