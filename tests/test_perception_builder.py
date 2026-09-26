"""T-FIX-28 — real state builder: readers -> canonical ``GameState``.

One test per row of the conversion table in
``docs/lab_phase/PRE_REAL_PERCEPTION_FIX_ROADMAP.md`` (T-FIX-28), plus the
acceptance items: a full ``__post_init__``/round-trip check, a no-target
frame that yields ``target=None`` instead of raising, the confidence map,
and the static import isolation.

Every reader is replaced by a small fake, so nothing here needs a live
client, Tesseract, YOLO weights, or OpenCV. The facing test drives the real
:class:`~wow_bot.perception.minimap.MinimapReading`, so the single
degrees->radians conversion site is exercised rather than reimplemented.
"""

from __future__ import annotations

import ast
import math
from pathlib import Path

import numpy as np
import pytest

from wow_bot.perception.bars import BarReading
from wow_bot.perception.builder import (
    BuilderIncompleteError,
    InjectedObservations,
    RealStateBuilder,
    bbox_from_corners,
    build_target_info,
    event_from_candidate,
    map_node_kind,
    target_hp_fraction,
)
from wow_bot.perception.enemies import EnemyDetection
from wow_bot.perception.events import EventCandidate
from wow_bot.perception.minimap import MinimapReading, degrees_to_facing_radians
from wow_bot.perception.target import TargetReading
from wow_bot.shared.interfaces import EnemyInfo, GameState, TargetInfo
from wow_bot.world.store import VALID_NODE_KINDS

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILDER_PATH = REPO_ROOT / "src" / "wow_bot" / "perception" / "builder.py"

POSITION = (100.0, 200.0)


def frame() -> np.ndarray:
    """A frame the fakes ignore; only its presence matters here."""
    return np.zeros((8, 8, 3), dtype=np.uint8)


class FakeBars:
    def __init__(self, hp: float = 0.75, mana: float = 0.5) -> None:
        self.hp = hp
        self.mana = mana

    def read_all(self, _frame: np.ndarray) -> BarReading:
        return BarReading(hp=self.hp, mana=self.mana)


class FakeCombat:
    def __init__(self, in_combat: bool = True) -> None:
        self.in_combat = in_combat

    def detect(self, _frame: np.ndarray, now: float | None = None) -> tuple[bool, float]:
        return self.in_combat, 0.5


class FakeMinimap:
    def __init__(self, angle_degrees: float | None = 90.0, confidence: float = 0.9) -> None:
        self.angle_degrees = angle_degrees
        self.confidence = confidence

    def update(self, _frame: np.ndarray, *, now: float | None = None) -> MinimapReading:
        return MinimapReading(
            position_px=(10.0, 20.0),
            angle_degrees=self.angle_degrees,
            confidence=self.confidence,
        )


class FakeTarget:
    def __init__(self, reading: TargetReading) -> None:
        self.reading = reading

    def read(
        self,
        _frame: np.ndarray,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> TargetReading:
        return self.reading


class FakeEvents:
    def __init__(self, candidates: list[EventCandidate] | None = None) -> None:
        self.candidates = candidates or []

    def detect(
        self,
        _frame: np.ndarray,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> list[EventCandidate]:
        return list(self.candidates)


class FakeEnemies:
    def __init__(self, detections: list[EnemyDetection] | None = None) -> None:
        self.detections = detections or []

    def detect(self, _frame: np.ndarray, *, now: float | None = None) -> list[EnemyDetection]:
        return list(self.detections)


def make_builder(
    *,
    target: FakeTarget | None = None,
    events: FakeEvents | None = None,
    enemies: FakeEnemies | None = None,
    minimap: FakeMinimap | None = None,
    kind_map: dict[str, str] | None = None,
    entity_distance: object = None,
    clock: object = None,
) -> RealStateBuilder:
    kwargs: dict[str, object] = {
        "bars": FakeBars(),
        "combat": FakeCombat(),
        "minimap": minimap or FakeMinimap(),
        "target": target,
        "events": events,
        "enemies": enemies,
        "kind_map": kind_map,
    }
    if entity_distance is not None:
        kwargs["entity_distance"] = entity_distance
    if clock is not None:
        kwargs["clock"] = clock
    return RealStateBuilder(**kwargs)  # type: ignore[arg-type]


def build(
    builder: RealStateBuilder,
    *,
    now: float | None = 5.0,
    reaction: str | None = "hostile",
    distance: float | None = 12.0,
) -> GameState:
    return builder.build(
        frame(),
        now=now,
        injected=InjectedObservations(
            position=POSITION,
            target_reaction=reaction,
            target_distance_estimate=distance,
        ),
    )


# ---------------------------------------------------------------------------
# Row 1 — target HP: int 0..100 -> fraction
# ---------------------------------------------------------------------------
def test_target_hp_percent_is_converted_to_fraction() -> None:
    assert target_hp_fraction(0) == 0.0
    assert target_hp_fraction(55) == pytest.approx(0.55)
    assert target_hp_fraction(100) == 1.0

    reading = TargetReading(name="Defias Thug", hp_pct=55, frame_confidence=0.9)
    info = build_target_info(reading, reaction="hostile", distance_estimate=10.0)
    assert info is not None
    assert info.hp_pct == pytest.approx(0.55)
    # The canonical fraction passes the schema's [0, 1] gate.
    assert 0.0 <= info.hp_pct <= 1.0


def test_target_hp_reaches_game_state_as_a_fraction() -> None:
    target = FakeTarget(TargetReading(name="Defias Thug", hp_pct=42, frame_confidence=0.8))
    state = build(make_builder(target=target))
    assert state.target is not None
    assert state.target.hp_pct == pytest.approx(0.42)


# ---------------------------------------------------------------------------
# Row 2 — facing: degrees -> radians, normalised to [-pi, pi), one site
# ---------------------------------------------------------------------------
def test_facing_is_converted_and_normalised() -> None:
    assert FakeMinimap(angle_degrees=90.0).update(frame()).facing_radians == pytest.approx(
        math.pi / 2
    )
    state = build(make_builder(minimap=FakeMinimap(angle_degrees=90.0)))
    assert state.facing == pytest.approx(math.pi / 2)

    # The half-open range: 180 degrees is -pi, not +pi.
    assert degrees_to_facing_radians(180.0) == pytest.approx(-math.pi)
    assert degrees_to_facing_radians(270.0) == pytest.approx(-math.pi / 2)
    for degrees in (0.0, 45.0, 180.0, 270.0, 359.0):
        facing = build(make_builder(minimap=FakeMinimap(angle_degrees=degrees))).facing
        assert -math.pi <= facing < math.pi


def test_facing_is_not_degrees_and_does_not_use_a_second_conversion_site() -> None:
    """Guards plan doc §3.1 trap 2: degrees must never reach ``facing``."""
    degrees = 90.0
    state = build(make_builder(minimap=FakeMinimap(angle_degrees=degrees)))
    assert state.facing != pytest.approx(degrees)
    assert state.facing == pytest.approx(degrees_to_facing_radians(degrees))


def test_unobserved_facing_raises_rather_than_fabricating_a_heading() -> None:
    builder = make_builder(minimap=FakeMinimap(angle_degrees=None))
    with pytest.raises(BuilderIncompleteError) as excinfo:
        build(builder)
    assert "facing" in str(excinfo.value)


# ---------------------------------------------------------------------------
# Row 3 — bbox: (x1, y1, x2, y2) -> (x, y, w, h)
# ---------------------------------------------------------------------------
def test_bbox_corner_pair_is_converted_to_origin_and_size() -> None:
    assert bbox_from_corners((10, 20, 40, 70)) == (10, 20, 30, 50)


def test_entity_bbox_is_origin_size_not_a_corner_pair() -> None:
    enemies = FakeEnemies([EnemyDetection(name="mob", bbox=(10, 20, 40, 70), confidence=0.8)])
    builder = make_builder(
        enemies=enemies,
        entity_distance=lambda _detection: 7.0,
    )
    state = build(builder)
    assert len(state.entities) == 1
    assert state.entities[0].bbox == (10, 20, 30, 50)


# ---------------------------------------------------------------------------
# Row 4 — events: mapping/candidate -> Event(type, timestamp, data)
# ---------------------------------------------------------------------------
def test_event_candidate_becomes_event_with_frame_timestamp() -> None:
    event = event_from_candidate(
        EventCandidate(type="loot", text="You receive item", source="chat"),
        timestamp=12.5,
    )
    assert event.type == "loot"
    assert event.timestamp == 12.5
    assert event.data == {"text": "You receive item", "source": "chat"}


def test_event_mapping_shape_is_converted_too() -> None:
    event = event_from_candidate(
        {"type": "death", "text": "Target dies", "source": "combat"},
        timestamp=3.0,
    )
    assert event.type == "death"
    assert event.timestamp == 3.0
    assert event.data["source"] == "combat"


def test_builder_attaches_the_monotonic_frame_timestamp_to_events() -> None:
    events = FakeEvents([EventCandidate(type="loot", text="loot line", source="chat")])
    state = build(make_builder(events=events), now=42.0)
    assert [event.timestamp for event in state.events] == [42.0]
    assert state.events[0].data["text"] == "loot line"


# ---------------------------------------------------------------------------
# Row 5 — TargetInfo only when all four fields are present
# ---------------------------------------------------------------------------
def test_target_info_requires_all_four_fields() -> None:
    reading = TargetReading(name="Defias Thug", hp_pct=50, frame_confidence=0.9)

    assert build_target_info(reading, reaction="hostile", distance_estimate=10.0) is not None
    assert build_target_info(reading, reaction=None, distance_estimate=10.0) is None
    assert build_target_info(reading, reaction="hostile", distance_estimate=None) is None

    no_name = TargetReading(name=None, hp_pct=0, frame_confidence=0.0)
    assert build_target_info(no_name, reaction="hostile", distance_estimate=10.0) is None


def test_target_is_none_when_the_reaction_channel_is_absent() -> None:
    """The T-FIX-30 reaction/distance channels are not wired yet."""
    target = FakeTarget(TargetReading(name="Defias Thug", hp_pct=50, frame_confidence=0.9))
    state = build(make_builder(target=target), reaction=None, distance=10.0)
    assert state.target is None


def test_invalid_reaction_raises_instead_of_building_a_partial_target() -> None:
    reading = TargetReading(name="Defias Thug", hp_pct=50, frame_confidence=0.9)
    with pytest.raises(ValueError, match="target_reaction"):
        build_target_info(reading, reaction="purple", distance_estimate=10.0)


def test_target_info_is_never_partial() -> None:
    """A partial TargetInfo would raise in ``__post_init__`` mid-pipeline."""
    reading = TargetReading(name="Defias Thug", hp_pct=50, frame_confidence=0.9)
    info = build_target_info(reading, reaction="neutral", distance_estimate=0.0)
    assert isinstance(info, TargetInfo)
    assert info.reaction == "neutral"
    assert info.distance_estimate == 0.0


# ---------------------------------------------------------------------------
# Row 6 — entities/enemies: one list, same detections (ADR-002 Decision 3)
# ---------------------------------------------------------------------------
def test_entities_and_enemies_are_populated_from_the_same_detections() -> None:
    detections = [
        EnemyDetection(name="mob", bbox=(1, 2, 11, 22), confidence=0.7),
        EnemyDetection(name="mob", bbox=(30, 40, 60, 80), confidence=0.6),
    ]
    builder = make_builder(
        enemies=FakeEnemies(detections),
        entity_distance=lambda _detection: 9.0,
    )
    state = build(builder)

    assert len(state.entities) == 2
    assert list(state.entities) == state.enemies
    assert all(isinstance(entity, EnemyInfo) for entity in state.entities)
    assert [entity.bbox for entity in state.entities] == [(1, 2, 10, 20), (30, 40, 30, 40)]


def test_unwired_detector_leaves_both_entity_channels_empty() -> None:
    state = build(make_builder())
    assert state.entities == ()
    assert state.enemies == []


# ---------------------------------------------------------------------------
# Row 7 — kind mapped to VALID_NODE_KINDS, or the entity is omitted
# ---------------------------------------------------------------------------
def test_kind_maps_to_valid_node_kinds_or_is_omitted() -> None:
    assert map_node_kind("mob") == "mob"
    assert map_node_kind("Defias Thug") is None  # never defaulted to "mob"
    assert map_node_kind("Defias Thug", {"Defias Thug": "mob"}) == "mob"
    assert map_node_kind("mob", {"mob": "not-a-kind"}) is None


def test_unknown_kind_omits_the_entity() -> None:
    detections = [EnemyDetection(name="Defias Thug", bbox=(1, 2, 3, 4), confidence=0.7)]
    builder = make_builder(
        enemies=FakeEnemies(detections),
        entity_distance=lambda _detection: 5.0,
    )
    assert build(builder).entities == ()


def test_kind_map_includes_the_entity_under_its_mapped_kind() -> None:
    detections = [EnemyDetection(name="Defias Thug", bbox=(1, 2, 3, 4), confidence=0.7)]
    builder = make_builder(
        enemies=FakeEnemies(detections),
        kind_map={"Defias Thug": "mob"},
        entity_distance=lambda _detection: 5.0,
    )
    state = build(builder)
    assert len(state.entities) == 1
    assert state.entities[0].kind == "mob"
    assert state.entities[0].kind in VALID_NODE_KINDS


def test_unobserved_distance_omits_the_entity_rather_than_defaulting_to_zero() -> None:
    detections = [EnemyDetection(name="mob", bbox=(1, 2, 3, 4), confidence=0.7)]
    builder = make_builder(enemies=FakeEnemies(detections), entity_distance=lambda _d: None)
    assert build(builder).entities == ()


# ---------------------------------------------------------------------------
# Row 8 — perception_confidence from YOLO conf / template match / OCR conf
# ---------------------------------------------------------------------------
def test_confidence_map_records_every_measured_score() -> None:
    target = FakeTarget(
        TargetReading(
            name="Defias Thug",
            hp_pct=50,
            frame_confidence=0.81,
            ocr_confidence=0.72,
        )
    )
    enemies = FakeEnemies([EnemyDetection(name="mob", bbox=(1, 2, 3, 4), confidence=0.65)])
    builder = make_builder(
        target=target,
        enemies=enemies,
        minimap=FakeMinimap(angle_degrees=90.0, confidence=0.93),
        entity_distance=lambda _detection: 5.0,
    )
    confidence = build(builder).perception_confidence

    assert confidence["facing"] == pytest.approx(0.93)
    assert confidence["target.hp_pct"] == pytest.approx(0.81)
    assert confidence["target.name"] == pytest.approx(0.72)
    assert confidence["entities.0.bbox"] == pytest.approx(0.65)
    assert confidence["entities.0.kind"] == pytest.approx(0.65)
    assert all(0.0 <= value <= 1.0 for value in confidence.values())


def test_confidence_falls_back_to_the_template_score_when_ocr_did_not_run() -> None:
    target = FakeTarget(TargetReading(name="Defias Thug", hp_pct=50, frame_confidence=0.77))
    state = build(make_builder(target=target))
    assert state.perception_confidence["target.name"] == pytest.approx(0.77)


def test_confidence_map_has_no_entry_for_an_unobserved_target() -> None:
    state = build(make_builder())
    assert "target.hp_pct" not in state.perception_confidence
    assert "target.name" not in state.perception_confidence
    assert "facing" in state.perception_confidence  # facing was observed


# ---------------------------------------------------------------------------
# Row 9 — timestamp: the monotonic clock
# ---------------------------------------------------------------------------
def test_timestamp_comes_from_the_injected_clock() -> None:
    state = build(make_builder(clock=lambda: 123.5), now=None)
    assert state.timestamp == 123.5


def test_explicit_now_overrides_the_clock_for_the_whole_frame() -> None:
    events = FakeEvents([EventCandidate(type="loot", text="loot line", source="chat")])
    state = build(make_builder(clock=lambda: 999.0, events=events), now=7.25)
    assert state.timestamp == 7.25
    assert [event.timestamp for event in state.events] == [7.25]


# ---------------------------------------------------------------------------
# Row 10 — unobserved fields stay None/empty, never fabricated
# ---------------------------------------------------------------------------
def test_unobserved_fields_are_none_or_empty() -> None:
    enemies = FakeEnemies([EnemyDetection(name="mob", bbox=(1, 2, 3, 4), confidence=0.7)])
    builder = make_builder(enemies=enemies, entity_distance=lambda _detection: 5.0)
    state = build(builder)

    assert state.player_z is None
    assert state.target_x is None
    assert state.target_y is None
    assert state.inventory_count is None
    assert state.inventory_max is None
    assert state.level_or_xp is None
    assert state.durability_fraction is None
    assert state.target_is_lootable is None
    assert state.incoming_casts == ()

    entity = state.entities[0]
    assert entity.entity_id is None
    assert entity.x is None
    assert entity.y is None
    assert entity.z is None
    assert entity.hp_fraction is None
    assert entity.threat is None
    assert entity.is_attackable is None
    assert entity.is_alive is None
    assert entity.is_in_combat_with_self is None


# ---------------------------------------------------------------------------
# Acceptance — schema validity, round trip, no-target frame, static imports
# ---------------------------------------------------------------------------
def test_emitted_game_state_passes_post_init_and_round_trip() -> None:
    target = FakeTarget(TargetReading(name="Defias Thug", hp_pct=60, frame_confidence=0.9))
    enemies = FakeEnemies([EnemyDetection(name="mob", bbox=(1, 2, 3, 4), confidence=0.7)])
    events = FakeEvents([EventCandidate(type="loot", text="loot line", source="chat")])
    builder = make_builder(
        target=target,
        enemies=enemies,
        events=events,
        entity_distance=lambda _detection: 5.0,
    )
    state = build(builder)  # __post_init__ already ran during construction

    payload = state.to_dict()
    restored = GameState.from_dict(payload)
    assert restored.to_dict() == payload
    assert restored == state


def test_a_frame_with_no_target_yields_none_instead_of_raising() -> None:
    target = FakeTarget(TargetReading(name=None, hp_pct=0, frame_confidence=0.0))
    state = build(make_builder(target=target))
    assert state.target is None
    assert isinstance(state, GameState)


def test_bars_and_combat_values_pass_through_without_unit_conversion() -> None:
    state = build(make_builder())
    assert state.hp_pct == pytest.approx(0.75)
    assert state.mana_pct == pytest.approx(0.5)
    assert state.in_combat is True


def test_builder_imports_nothing_from_lab_or_main() -> None:
    tree = ast.parse(BUILDER_PATH.read_text(encoding="utf-8"))
    imported: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)

    assert not any(
        module == "wow_bot.lab" or module.startswith("wow_bot.lab.") for module in imported
    ), imported
    assert not any(module in {"wow_bot.main", "__main__"} for module in imported), imported
