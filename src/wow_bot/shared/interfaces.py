"""Layer-boundary data contracts (Task 1.1).

These dataclasses are the borders between Perception, Internal Dynamics,
Strategist, and Executor. Per AGENTS.md they must not be renamed or have
fields removed; new fields are added as optional with safe defaults.

Task 1.1 ships ``TargetInfo``, ``EnemyInfo``, ``Event`` and ``GameState``.
``MetaState`` and ``Strategy`` arrive in Task 1.2 by extending this module.

Serialization note
------------------
Plain ``dataclasses.asdict`` is *not* used for nested payloads because it
copies tuples into lists and would silently break the tuple-typed geometry
fields (``position``, ``facing`` container, ``bbox``). The explicit
``to_dict`` / ``from_dict`` pair below preserves those types on a round trip.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

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