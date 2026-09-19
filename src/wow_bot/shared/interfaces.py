"""Layer-boundary data contracts (Tasks 1.1 and 1.2).

These dataclasses are the borders between Perception, Internal Dynamics,
Strategist, and Executor. Per AGENTS.md they must not be renamed or have
fields removed; new fields are added as optional with safe defaults.

Task 1.1 shipped ``TargetInfo``, ``EnemyInfo``, ``Event`` and ``GameState``.
Task 1.2 adds ``MetaState`` (Internal Dynamics → Strategist) and ``Strategy``
(Strategist → Executor), including Pydantic validation of ``risk_tolerance``.

Serialization note
------------------
Plain ``dataclasses.asdict`` is *not* used for nested payloads because it
copies tuples into lists and would silently break the tuple-typed geometry
fields (``position``, ``facing`` container, ``bbox``). The explicit
``to_dict`` / ``from_dict`` pair below preserves those types on a round trip.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pydantic import BaseModel, Field, ValidationError

# Accepted reaction labels for TargetInfo. Kept as a module constant so both
# validation and callers share one definition.
REACTIONS: frozenset[str] = frozenset({"hostile", "neutral", "friendly"})


def _require(condition: bool, message: str) -> None:
    """Raise ``ValueError`` when a contract invariant is violated."""
    if not condition:
        raise ValueError(message)


@dataclass
class TargetInfo:
    """Currently selected hostile/neutral/friendly target."""

    name: str
    hp_pct: float  # 0..1
    reaction: str  # "hostile" | "neutral" | "friendly"
    distance_estimate: float  # yards, approximate

    def __post_init__(self) -> None:
        _require(0.0 <= self.hp_pct <= 1.0, f"hp_pct out of [0,1]: {self.hp_pct}")
        _require(
            self.reaction in REACTIONS,
            f"reaction must be one of {sorted(REACTIONS)}, got {self.reaction!r}",
        )
        _require(
            self.distance_estimate >= 0.0,
            f"distance_estimate must be >= 0, got {self.distance_estimate}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "hp_pct": self.hp_pct,
            "reaction": self.reaction,
            "distance_estimate": self.distance_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TargetInfo:
        return cls(
            name=str(data["name"]),
            hp_pct=float(data["hp_pct"]),
            reaction=str(data["reaction"]),
            distance_estimate=float(data["distance_estimate"]),
        )


@dataclass
class EnemyInfo:
    """One detected enemy bounding box from the perception layer."""

    bbox: tuple[int, int, int, int]  # x, y, w, h
    confidence: float  # 0..1
    distance_estimate: float  # yards, approximate

    def __post_init__(self) -> None:
        _require(len(self.bbox) == 4, f"bbox must have 4 components, got {self.bbox}")
        _require(
            all(isinstance(v, int) for v in self.bbox),
            f"bbox components must be ints, got {self.bbox}",
        )
        _require(0.0 <= self.confidence <= 1.0, f"confidence out of [0,1]: {self.confidence}")
        _require(
            self.distance_estimate >= 0.0,
            f"distance_estimate must be >= 0, got {self.distance_estimate}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "bbox": list(self.bbox),  # JSON has no tuple type
            "confidence": self.confidence,
            "distance_estimate": self.distance_estimate,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> EnemyInfo:
        raw_bbox = data["bbox"]
        _require(
            len(raw_bbox) == 4,
            f"bbox must have 4 components, got {raw_bbox!r}",
        )
        return cls(
            bbox=(int(raw_bbox[0]), int(raw_bbox[1]), int(raw_bbox[2]), int(raw_bbox[3])),
            confidence=float(data["confidence"]),
            distance_estimate=float(data["distance_estimate"]),
        )


@dataclass
class Event:
    """Discrete occurrence reported by perception or injected by tests."""

    type: str
    timestamp: float
    data: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        _require(bool(self.type), "event type must be a non-empty string")

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "timestamp": self.timestamp, "data": dict(self.data)}

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Event:
        payload = data.get("data") or {}
        _require(isinstance(payload, dict), f"event 'data' must be a mapping, got {payload!r}")
        return cls(
            type=str(data["type"]),
            timestamp=float(data["timestamp"]),
            data=dict(payload),
        )


@dataclass
class GameState:
    """Snapshot emitted by the perception layer each frame."""

    timestamp: float
    hp_pct: float
    mana_pct: float
    position: tuple[float, float]
    facing: float
    in_combat: bool
    target: TargetInfo | None
    enemies: list[EnemyInfo]
    events: list[Event]

    def __post_init__(self) -> None:
        _require(0.0 <= self.hp_pct <= 1.0, f"hp_pct out of [0,1]: {self.hp_pct}")
        _require(0.0 <= self.mana_pct <= 1.0, f"mana_pct out of [0,1]: {self.mana_pct}")
        _require(
            len(self.position) == 2,
            f"position must be an (x, y) pair, got {self.position}",
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "hp_pct": self.hp_pct,
            "mana_pct": self.mana_pct,
            "position": [float(self.position[0]), float(self.position[1])],
            "facing": self.facing,
            "in_combat": self.in_combat,
            "target": self.target.to_dict() if self.target is not None else None,
            "enemies": [enemy.to_dict() for enemy in self.enemies],
            "events": [event.to_dict() for event in self.events],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GameState:
        raw_position = data["position"]
        _require(
            len(raw_position) == 2,
            f"position must be an (x, y) pair, got {raw_position!r}",
        )
        raw_target = data.get("target")
        return cls(
            timestamp=float(data["timestamp"]),
            hp_pct=float(data["hp_pct"]),
            mana_pct=float(data["mana_pct"]),
            position=(float(raw_position[0]), float(raw_position[1])),
            facing=float(data["facing"]),
            in_combat=bool(data["in_combat"]),
            target=TargetInfo.from_dict(raw_target) if raw_target is not None else None,
            enemies=[EnemyInfo.from_dict(e) for e in data.get("enemies", [])],
            events=[Event.from_dict(e) for e in data.get("events", [])],
        )


# ---------------------------------------------------------------------------
# Task 1.2 — Internal Dynamics → Strategist → Executor contracts
# ---------------------------------------------------------------------------

#: Canonical ordering of the five drive components inside ``MetaState.vector``.
#: Matches AGENTS.md: [hunger, fatigue, curiosity, aggression, social].
DRIVE_NAMES: tuple[str, ...] = ("hunger", "fatigue", "curiosity", "aggression", "social")

#: Width of a MetaState drive vector.
META_STATE_DIM: int = len(DRIVE_NAMES)

#: Accepted high-level goals produced by the strategist. Kept as a module
#: constant so validation and callers share one definition (same pattern as
#: ``REACTIONS`` above). Extending this set is additive; never remove entries.
STRATEGY_GOALS: frozenset[str] = frozenset({"farm_herbs", "grind_humans", "explore", "flee"})


@dataclass
class MetaState:
    """Aggregate internal state emitted by the dynamics layer each step.

    ``vector`` holds the five drive levels in :data:`DRIVE_NAMES` order and is
    always a float64 array of shape ``(5,)`` with values clamped to ``[0, 1]``.
    """

    # Declared as Any so array-like inputs (lists, tuples, arrays) type-check;
    # __post_init__ always normalizes it to a float64 ndarray of shape (5,).
    vector: Any  # shape (5,): [hunger, fatigue, curiosity, aggression, social]
    recent_events: list[Event]
    timestamp: float

    def __post_init__(self) -> None:
        arr = np.asarray(self.vector, dtype=np.float64)
        _require(
            arr.shape == (META_STATE_DIM,),
            f"MetaState.vector must have shape ({META_STATE_DIM},), got {arr.shape}",
        )
        _require(
            bool(np.all(np.isfinite(arr))),
            f"MetaState.vector must be finite, got {self.vector!r}",
        )
        _require(
            bool(np.all((arr >= 0.0) & (arr <= 1.0))),
            f"MetaState.vector components must lie in [0,1], got {arr.tolist()}",
        )
        self.vector = arr  # normalize caller-supplied lists/ints to float64 ndarray

    # ndarray field makes the generated __eq__ ambiguous, so compare explicitly.
    __hash__ = None  # type: ignore[assignment]

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MetaState):
            return NotImplemented
        return (
            np.array_equal(self.vector, other.vector)
            and self.recent_events == other.recent_events
            and self.timestamp == other.timestamp
        )

    def drives(self) -> dict[str, float]:
        """Return the drive vector keyed by :data:`DRIVE_NAMES`."""
        return {name: float(value) for name, value in zip(DRIVE_NAMES, self.vector)}

    def to_dict(self) -> dict[str, Any]:
        return {
            "vector": [float(v) for v in self.vector],
            "recent_events": [event.to_dict() for event in self.recent_events],
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> MetaState:
        raw_vector = data["vector"]
        _require(
            len(raw_vector) == META_STATE_DIM,
            f"vector must have {META_STATE_DIM} components, got {raw_vector!r}",
        )
        return cls(
            vector=np.array([float(v) for v in raw_vector], dtype=np.float64),
            recent_events=[Event.from_dict(e) for e in data.get("recent_events", [])],
            timestamp=float(data["timestamp"]),
        )


class StrategyModel(BaseModel):
    """Pydantic schema enforcing the numeric bounds of a serialized strategy.

    Only ``risk_tolerance`` carries a hard range constraint (``0..1``); the
    remaining fields mirror the dataclass below so that ``from_dict`` rejects
    malformed payloads loudly rather than silently coercing them.
    """

    goal: str
    region: str
    risk_tolerance: float = Field(ge=0.0, le=1.0)
    priority: list[str]
    constraints: dict[str, Any] = Field(default_factory=dict)
    valid_until: float
    raw_llm_output: str = ""


@dataclass
class Strategy:
    """High-level plan emitted by the strategist, consumed by the executor.

    The LLM produces this roughly every 20-40 minutes; the executor applies it
    locally until ``valid_until`` elapses.
    """

    goal: str  # "farm_herbs" | "grind_humans" | "explore" | "flee"
    region: str
    risk_tolerance: float  # 0..1
    priority: list[str]
    constraints: dict[str, Any]
    valid_until: float
    raw_llm_output: str = ""

    def __post_init__(self) -> None:
        # Mirror the Pydantic bound so direct construction fails as loudly as
        # deserialization does; otherwise an out-of-range risk tolerance could
        # slip past validation whenever a Strategy is built without from_dict.
        _require(
            0.0 <= self.risk_tolerance <= 1.0,
            f"risk_tolerance out of [0,1]: {self.risk_tolerance}",
        )
        _require(bool(self.goal), "goal must be a non-empty string")
        self.constraints = dict(self.constraints)
        self.priority = list(self.priority)

    def to_dict(self) -> dict[str, Any]:
        return {
            "goal": self.goal,
            "region": self.region,
            "risk_tolerance": self.risk_tolerance,
            "priority": list(self.priority),
            "constraints": dict(self.constraints),
            "valid_until": self.valid_until,
            "raw_llm_output": self.raw_llm_output,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Strategy:
        """Rebuild a Strategy from a JSON-decoded mapping.

        Raises :class:`ValueError` when the payload violates the contract, so
        callers see one exception type regardless of how validation failed
        (bounds, unknown keys, wrong types).
        """
        try:
            model = StrategyModel.model_validate(dict(data))
        except ValidationError as exc:
            raise ValueError(f"invalid Strategy payload: {exc}") from exc
        return cls(
            goal=model.goal,
            region=model.region,
            risk_tolerance=model.risk_tolerance,
            priority=list(model.priority),
            constraints=dict(model.constraints),
            valid_until=model.valid_until,
            raw_llm_output=model.raw_llm_output,
        )