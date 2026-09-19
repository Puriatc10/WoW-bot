"""Event type registry and drive-effect table (Task 1.3).

The perception layer emits :class:`~wow_bot.shared.interfaces.Event` objects
whose ``type`` is one of the constants defined here. The internal dynamics
layer consumes :data:`EVENT_EFFECTS` to translate an event into an immediate
delta on the five drives, in the canonical order declared by
``interfaces.DRIVE_NAMES``: ``[hunger, fatigue, curiosity, aggression, social]``.

Contract notes
--------------
* Adding a new event type is additive: declare a constant and give it an entry
  in :data:`EVENT_EFFECTS`. Existing keys must not be renamed or removed —
  downstream tests and scenarios depend on them (see AGENTS.md "Can I add a new
  event type?").
* Every effect row carries all five drive components so callers can index by
  drive name without a missing-key branch. Zero means "no direct effect"; it is
  written explicitly rather than omitted so the table stays uniform and mypy
  sees a single homogeneous value type.
* Effect magnitudes are deltas applied on top of the current drive level; the
  dynamics layer is responsible for clamping the result to ``[0, 1]``. This
  module deliberately stores raw signed floats and performs no clamping, so it
  stays free of behaviour that belongs to another layer.
"""

from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Final

# Drive component names, duplicated here only as a documentation anchor. The
# authoritative ordering lives in interfaces.DRIVE_NAMES; importing it here
# would couple this leaf registry to the heavier contracts module, so the rows
# below are validated against the tuple by tests instead.
_DRIVE_KEYS: Final[tuple[str, ...]] = (
    "hunger",
    "fatigue",
    "curiosity",
    "aggression",
    "social",
)

#: A single event's impact on the five drives, keyed by drive name.
DriveEffects = Mapping[str, float]


def _effects(
    *,
    hunger: float,
    fatigue: float,
    curiosity: float,
    aggression: float,
    social: float,
) -> DriveEffects:
    """Build an immutable, fully-specified effect row.

    Keyword-only arguments make omission impossible at the call site, which is
    how the "every row covers every drive" invariant is enforced structurally
    rather than relying on hand-written dict literals staying in sync.
    """
    return MappingProxyType(
        {
            "hunger": hunger,
            "fatigue": fatigue,
            "curiosity": curiosity,
            "aggression": aggression,
            "social": social,
        }
    )


# ---------------------------------------------------------------------------
# Event type constants
# ---------------------------------------------------------------------------

DEATH: Final[str] = "death"
RARE_LOOT: Final[str] = "rare_loot"
PVP_HIT: Final[str] = "pvp_hit"
PVP_KILL: Final[str] = "pvp_kill"
STUCK: Final[str] = "stuck"
LEVEL_UP: Final[str] = "level_up"
QUEST_COMPLETE: Final[str] = "quest_complete"
NPC_INTERACT: Final[str] = "npc_interact"

#: All recognised event types, ordered as declared in ROADMAP.md Task 1.3.
#: Iterating this tuple is the supported way to enumerate known events.
ALL_EVENT_TYPES: Final[tuple[str, ...]] = (
    DEATH,
    RARE_LOOT,
    PVP_HIT,
    PVP_KILL,
    STUCK,
    LEVEL_UP,
    QUEST_COMPLETE,
    NPC_INTERACT,
)

#: Human-readable labels for telemetry/log lines. Keys mirror
#: :data:`ALL_EVENT_TYPES`; values are short display strings, never identifiers.
EVENT_LABELS: Final[Mapping[str, str]] = MappingProxyType(
    {
        DEATH: "Death",
        RARE_LOOT: "Rare loot",
        PVP_HIT: "PvP damage taken",
        PVP_KILL: "PvP kill",
        STUCK: "Stuck / pathing failure",
        LEVEL_UP: "Level up",
        QUEST_COMPLETE: "Quest complete",
        NPC_INTERACT: "NPC interaction",
    }
)


# ---------------------------------------------------------------------------
# Drive-effect table
# ---------------------------------------------------------------------------
#
# Signs follow the roadmap: a death depresses aggression and social while it
# accumulates fatigue; rare loot spikes hunger/greed and curiosity; repeated
# stuck states drain curiosity and aggression. Rows are read-only mappings so a
# caller cannot mutate shared state by accident (AGENTS.md: avoid shared
# mutable global state).

EVENT_EFFECTS: Final[Mapping[str, DriveEffects]] = MappingProxyType(
    {
        DEATH: _effects(
            hunger=0.0,
            fatigue=+0.10,
            curiosity=0.0,
            aggression=-0.15,
            social=-0.05,
        ),
        RARE_LOOT: _effects(
            hunger=+0.20,
            fatigue=0.0,
            curiosity=+0.15,
            aggression=0.0,
            social=+0.05,
        ),
        PVP_HIT: _effects(
            hunger=0.0,
            fatigue=+0.05,
            curiosity=-0.05,
            aggression=+0.10,
            social=-0.10,
        ),
        PVP_KILL: _effects(
            hunger=0.0,
            fatigue=0.0,
            curiosity=0.0,
            aggression=+0.05,
            social=+0.05,
        ),
        STUCK: _effects(
            hunger=0.0,
            fatigue=+0.15,
            curiosity=-0.10,
            aggression=-0.05,
            social=0.0,
        ),
        LEVEL_UP: _effects(
            hunger=+0.10,
            fatigue=-0.10,
            curiosity=+0.10,
            aggression=+0.05,
            social=+0.05,
        ),
        QUEST_COMPLETE: _effects(
            hunger=+0.10,
            fatigue=-0.05,
            curiosity=+0.10,
            aggression=0.0,
            social=+0.05,
        ),
        NPC_INTERACT: _effects(
            hunger=0.0,
            fatigue=-0.05,
            curiosity=0.0,
            aggression=0.0,
            social=+0.15,
        ),
    }
)


def effects_for(event_type: str) -> DriveEffects:
    """Return the drive-effect row for ``event_type``.

    Raises :class:`KeyError` with an actionable message for unknown types so a
    typo surfaces immediately instead of silently applying zero delta. Callers
    that must tolerate unrecognized perception output should check membership
    via :func:`is_known_event` first.
    """
    try:
        return EVENT_EFFECTS[event_type]
    except KeyError:
        raise KeyError(
            f"unknown event type {event_type!r}; known types are {list(ALL_EVENT_TYPES)}"
        ) from None


def is_known_event(event_type: str) -> bool:
    """True when ``event_type`` has an entry in :data:`EVENT_EFFECTS`."""
    return event_type in EVENT_EFFECTS


def total_effect_magnitude(event_type: str) -> float:
    """Sum of absolute drive deltas for one event.

    Useful as a crude salience score: the dynamics layer uses it to decide
    whether an event is worth persisting to memory (roadmap Task 3.5 step 6,
    "memory.add() for important events"). Keeping the helper here avoids
    duplicating the reduction logic across consumers.
    """
    row = effects_for(event_type)
    return sum(abs(delta) for delta in row.values())