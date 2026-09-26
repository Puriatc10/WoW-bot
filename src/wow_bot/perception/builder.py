"""Real state builder: T-FIX-27 readers -> canonical ``GameState`` (T-FIX-28).

Composes one frame through the readers ported in T-FIX-27 and emits the
canonical :class:`wow_bot.shared.interfaces.GameState`. This is the single
place where every unit and convention mismatch between the ``hamberger``
readers and the master contracts is resolved (plan doc §3.1):

* ``TargetReading.hp_pct`` is an ``int 0..100``; ``TargetInfo.hp_pct`` is a
  fraction in ``[0, 1]`` — divided by ``100.0`` here and nowhere else.
* ``MinimapReading.angle_degrees`` is degrees ``[0, 360)``; the conversion
  to a normalised ``[-pi, pi)`` radian heading lives in
  :func:`wow_bot.perception.minimap.degrees_to_facing_radians`. This module
  only *consumes* :attr:`MinimapReading.facing_radians`, so the conversion
  still exists in exactly one place.
* ``EnemyDetection.bbox`` is a corner pair ``(x1, y1, x2, y2)``;
  ``EnemyInfo.bbox`` is origin+size ``(x, y, w, h)``.
* ``EventCandidate``/event mappings become ``Event(type, timestamp, data)``
  with the frame's monotonic timestamp.

**Fields this task does not observe.** ``position``, ``player_z``,
``target_x``, ``target_y``, ``target_reaction``, and
``target_distance_estimate`` have no channel in T-FIX-27; they are the
T-FIX-30 channels and are supplied by the caller through
:class:`InjectedObservations`. The builder never derives them. Likewise a
per-entity ``distance_estimate`` has no source (plan doc §3.2): an injected
:data:`EntityDistance` supplies it, and a detection whose distance is
unknown is **omitted** rather than given a fabricated value, because
``EnemyInfo.distance_estimate`` is non-Optional.

**Fail-loud.** ``GameState.hp_pct``, ``mana_pct``, ``facing``, and
``in_combat`` are non-Optional on the schema. ``facing`` can genuinely be
unobserved (no minimap arrow match yet); the builder then raises
:class:`BuilderIncompleteError` naming the field instead of defaulting to a
zero heading. ``target`` is Optional: when the reader reports no name, or
the injected ``reaction``/``distance_estimate`` is absent, ``target`` is
``None`` rather than a partial ``TargetInfo`` (whose ``__post_init__``
would raise mid-pipeline).

**Confidence.** ``perception_confidence`` is populated only from the three
measured scores the contract names — YOLO confidence, template-match
scores, and OCR confidence. Channels without a confidence model (the bar
fill ratio, the combat edge ratio) are absent rather than reported at a
fabricated ``1.0``.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any

from wow_bot.perception.bars import BarReader
from wow_bot.perception.capture import FrameLike
from wow_bot.perception.combat import CombatDetector
from wow_bot.perception.enemies import EnemyDetection, EnemyDetector
from wow_bot.perception.events import EventCandidate, EventDetector
from wow_bot.perception.minimap import MinimapTracker
from wow_bot.perception.target import TargetReader
from wow_bot.shared.interfaces import (
    REACTIONS,
    EnemyInfo,
    Event,
    GameState,
    TargetInfo,
)
from wow_bot.world.store import VALID_NODE_KINDS

__all__ = [
    "BuilderIncompleteError",
    "EntityDistance",
    "InjectedObservations",
    "RealStateBuilder",
    "bbox_from_corners",
    "build_target_info",
    "event_from_candidate",
    "map_node_kind",
    "target_hp_fraction",
]

#: Supplies the per-entity ``distance_estimate`` the detector cannot observe.
#: Returning ``None`` means "not observed", and the entity is omitted.
EntityDistance = Callable[[EnemyDetection], "float | None"]


class BuilderIncompleteError(RuntimeError):
    """A non-Optional ``GameState`` field could not be observed this frame."""


@dataclass(frozen=True)
class InjectedObservations:
    """Values T-FIX-28 cannot observe, injected by the caller (T-FIX-30).

    ``position`` is required because ``GameState.position`` is non-Optional
    and no T-FIX-27 reader observes a world pose: the minimap arrow centre
    is screen pixels, not world coordinates (plan doc finding F-1), so
    feeding it in would corrupt the world model. Every other field defaults
    to ``None``, meaning "not observed".

    ``confidence`` carries the measured scores of the injected channels, so
    their confidence semantics reach ``GameState.perception_confidence``
    instead of being discarded at the injection boundary. It is merged with
    :meth:`dict.setdefault`, so a score this builder measured itself for one
    of its own channels always wins over an injected one.
    """

    position: tuple[float, float]
    player_z: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    target_reaction: str | None = None
    target_distance_estimate: float | None = None
    confidence: Mapping[str, float] = field(default_factory=dict)


def target_hp_fraction(hp_pct: int) -> float:
    """Convert the target reader's ``int 0..100`` to a canonical fraction.

    ``TargetReading.hp_pct`` follows the ``hamberger`` convention;
    ``TargetInfo.hp_pct`` is a fraction in ``[0, 1]``. This is the only
    division site for that unit (plan doc §3.1 trap 1).
    """
    return hp_pct / 100.0


def bbox_from_corners(bbox: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """Convert a ``(x1, y1, x2, y2)`` corner pair to ``(x, y, w, h)``.

    YOLO emits corner pairs; ``EnemyInfo.bbox`` is documented as origin +
    size (plan doc §3.1 trap 3). This is the only conversion site.
    """
    x1, y1, x2, y2 = bbox
    return (x1, y1, x2 - x1, y2 - y1)


def map_node_kind(class_name: str, kind_map: Mapping[str, str] | None = None) -> str | None:
    """Map a detector class name onto ``VALID_NODE_KINDS``, or ``None``.

    ``None`` means the entity must be omitted rather than carrying a
    defaulted kind, because ``world.sync`` skips entities whose kind is not
    a valid node kind. A mapping entry overrides the class name; without
    one a class name that is already a valid kind maps to itself.
    """
    mapped = class_name if kind_map is None else kind_map.get(class_name, class_name)
    return mapped if mapped in VALID_NODE_KINDS else None


def build_target_info(
    reading: Any,
    *,
    reaction: str | None,
    distance_estimate: float | None,
) -> TargetInfo | None:
    """Build a ``TargetInfo`` only when all four fields are present.

    The reader supplies ``name`` and ``hp_pct``; ``reaction`` and
    ``distance_estimate`` are T-FIX-30 channels injected by the caller.
    Anything missing yields ``None`` — an honest absence — never a partial
    ``TargetInfo`` whose ``__post_init__`` would raise mid-pipeline. An
    invalid ``reaction`` is a caller error and raises ``ValueError``
    explicitly rather than being silently dropped.
    """
    if reading.name is None or reaction is None or distance_estimate is None:
        return None
    if reaction not in REACTIONS:
        raise ValueError(f"target_reaction must be one of {sorted(REACTIONS)}, got {reaction!r}")
    return TargetInfo(
        name=reading.name,
        hp_pct=target_hp_fraction(reading.hp_pct),
        reaction=reaction,
        distance_estimate=distance_estimate,
    )


def event_from_candidate(candidate: EventCandidate | Mapping[str, Any], timestamp: float) -> Event:
    """Convert a reader event into ``Event(type, timestamp, data)``.

    The reader emits :class:`EventCandidate`; the ``hamberger`` original
    emitted a ``{"type", "text", "source"}`` mapping, which is accepted too
    so the conversion is tested against both shapes (plan doc §3.1 trap 6).
    The free text and source move into ``data``; the timestamp is the
    frame's monotonic timestamp, which the reader deliberately does not
    attach so readers stay deterministic given one frame.
    """
    if isinstance(candidate, Mapping):
        event_type = str(candidate["type"])
        text = str(candidate.get("text", ""))
        source = str(candidate.get("source", ""))
    else:
        event_type = candidate.type
        text = candidate.text
        source = candidate.source
    return Event(
        type=event_type,
        timestamp=timestamp,
        data={"text": text, "source": source},
    )


class RealStateBuilder:
    """Composes one frame through the T-FIX-27 readers into a ``GameState``.

    ``bars``, ``combat``, and ``minimap`` are required because they feed
    non-Optional ``GameState`` fields. ``target``, ``events``, and
    ``enemies`` are optional: an unwired channel leaves its fields absent
    (``target=None``, empty collections) rather than fabricated.
    """

    def __init__(
        self,
        *,
        bars: BarReader,
        combat: CombatDetector,
        minimap: MinimapTracker,
        target: TargetReader | None = None,
        events: EventDetector | None = None,
        enemies: EnemyDetector | None = None,
        clock: Callable[[], float] = time.monotonic,
        tesseract_cmd: str | None = None,
        kind_map: Mapping[str, str] | None = None,
        entity_distance: EntityDistance | None = None,
    ) -> None:
        self._bars = bars
        self._combat = combat
        self._minimap = minimap
        self._target = target
        self._events = events
        self._enemies = enemies
        self._clock = clock
        self._tesseract_cmd = tesseract_cmd
        self._kind_map = kind_map
        self._entity_distance = entity_distance

    def build(
        self,
        frame: FrameLike,
        *,
        injected: InjectedObservations,
        now: float | None = None,
    ) -> GameState:
        """Compose ``frame`` into a canonical ``GameState``.

        ``now`` is the frame's clock reading. It defaults to the injected
        clock and is passed to every throttled reader so one frame has one
        timestamp and the throttle decisions are deterministic in tests.
        """
        timestamp = self._clock() if now is None else float(now)

        bar_reading = self._bars.read_all(frame)
        in_combat, _red_ratio = self._combat.detect(frame, timestamp)

        minimap_reading = self._minimap.update(frame, now=timestamp)
        facing = minimap_reading.facing_radians
        if facing is None:
            raise BuilderIncompleteError(
                "facing is a non-Optional GameState field and the minimap "
                "arrow has not been observed yet; refusing to emit a "
                "fabricated heading"
            )

        confidence: dict[str, float] = {"facing": float(minimap_reading.confidence)}

        target_info: TargetInfo | None = None
        if self._target is not None:
            target_reading = self._target.read(
                frame, now=timestamp, tesseract_cmd=self._tesseract_cmd
            )
            target_info = build_target_info(
                target_reading,
                reaction=injected.target_reaction,
                distance_estimate=injected.target_distance_estimate,
            )
            if target_reading.name is not None:
                confidence["target.hp_pct"] = float(target_reading.frame_confidence)
                confidence["target.name"] = float(
                    target_reading.ocr_confidence
                    if target_reading.ocr_confidence is not None
                    else target_reading.frame_confidence
                )

        entity_infos = self._build_entities(frame, timestamp, confidence)

        events: list[Event] = []
        if self._events is not None:
            candidates = self._events.detect(
                frame, now=timestamp, tesseract_cmd=self._tesseract_cmd
            )
            events = [event_from_candidate(candidate, timestamp) for candidate in candidates]

        # Injected channel scores join the map, but never overwrite a score
        # this builder measured for one of its own channels.
        for key, value in injected.confidence.items():
            confidence.setdefault(key, float(value))

        return GameState(
            timestamp=timestamp,
            hp_pct=bar_reading.hp,
            mana_pct=bar_reading.mana,
            position=injected.position,
            facing=facing,
            in_combat=bool(in_combat),
            target=target_info,
            enemies=list(entity_infos),
            events=events,
            player_z=injected.player_z,
            target_x=injected.target_x,
            target_y=injected.target_y,
            # ADR-002 Decision 3: one entity channel served by the same
            # detections in both the legacy and canonical slots.
            entities=tuple(entity_infos),
            perception_confidence=confidence,
            inventory_count=None,  # channel-pending: "Bag frame" (T-FIX-31)
            inventory_max=None,  # channel-pending: "Bag frame" (T-FIX-31)
            level_or_xp=None,  # channel-pending: "XP bar" (T-FIX-31)
            durability_fraction=None,  # channel-pending: "Character frame"
            target_is_lootable=None,  # channel-pending: "Lootable-corpse"
            incoming_casts=(),  # channel-pending: honest "no casts observed"
        )

    def _build_entities(
        self,
        frame: FrameLike,
        timestamp: float,
        confidence: dict[str, float],
    ) -> list[EnemyInfo]:
        """Project the detections into ``EnemyInfo``, omitting what is unknown.

        An entity is omitted when its kind is not a valid node kind
        (never defaulted) or its distance is unobserved
        (``EnemyInfo.distance_estimate`` is non-Optional and no reader
        observes it). Every other unobserved per-entity field stays
        ``None``.
        """
        if self._enemies is None:
            return []

        detections = self._enemies.detect(frame, now=timestamp)
        entities: list[EnemyInfo] = []
        for detection in detections:
            kind = map_node_kind(detection.name, self._kind_map)
            if kind is None:
                continue
            distance = None if self._entity_distance is None else self._entity_distance(detection)
            if distance is None:
                continue
            index = len(entities)
            confidence[f"entities.{index}.bbox"] = float(detection.confidence)
            confidence[f"entities.{index}.kind"] = float(detection.confidence)
            entities.append(
                EnemyInfo(
                    bbox=bbox_from_corners(detection.bbox),
                    confidence=float(detection.confidence),
                    distance_estimate=float(distance),
                    entity_id=None,  # no stable-id policy yet (ADR-002 Q1)
                    kind=kind,
                    x=None,  # world position is T-FIX-30
                    y=None,
                    z=None,
                    hp_fraction=None,  # no per-enemy HP channel
                    threat=None,  # no threat channel (ADR-002 Q3)
                    is_attackable=None,
                    is_alive=None,
                    is_in_combat_with_self=None,
                )
            )
        return entities
