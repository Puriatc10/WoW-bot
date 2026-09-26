"""T-FIX-30 — channel composition, and the three acceptance items.

The acceptance items of the roadmap task are proven here end to end:

1. ``WorldSyncView`` projects once a pose **and** a ``player_z`` are supplied,
   and ``WorldSync.sync_once`` then writes exactly one player node.
2. ``ReactiveView`` projects once a ``distance_estimate`` exists, because
   ``target_in_range`` derives from it.
3. A pose below its confidence floor leaves ``position`` as ``None``, the
   builder refuses to emit a fabricated pose, and **no world node is
   created**.

The pose channel's OCR engine is stubbed, so no test needs Tesseract. The
positive world-sync case exists so the negative case is not vacuous: the same
helper writes a node when the pose is confident, and none when it is not.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.perception.adapter import AdapterDerivationConfig, GameStateAdapter
from wow_bot.perception.bars import BarReading
from wow_bot.perception.builder import (
    BuilderIncompleteError,
    InjectedObservations,
    RealStateBuilder,
)
from wow_bot.perception.minimap import MinimapReading
from wow_bot.perception.observations import ChannelObservations, observe_channels
from wow_bot.perception.pose import WorldPoseReader
from wow_bot.perception.proximity import TargetDistanceEstimator, TargetDistanceModel
from wow_bot.perception.reaction import TargetReactionReader
from wow_bot.perception.target import TargetReading
from wow_bot.world.store import WorldModel
from wow_bot.world.sync import SyncStats, WorldSync

POSE_ROI = (0, 0, 60, 16)
REACTION_ROI = (60, 0, 40, 12)
GREEN = (0, 200, 0)
ENGAGE_DISTANCE_UNITS = 30.0


class FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = float(start)

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> float:
        self.now += float(seconds)
        return self.now


class FakeTesseractData:
    def __init__(self, text: str = "57.3, 42.1", confidence: float = 88.0) -> None:
        self.text = text
        self.confidence = confidence
        self.pytesseract = self
        self.calls = 0

    def image_to_data(
        self, _image: Any, config: str = "", output_type: str = ""
    ) -> dict[str, list[Any]]:
        self.calls += 1
        tokens = self.text.split()
        return {"text": tokens, "conf": [self.confidence] * len(tokens)}


class FakeBars:
    def read_all(self, _frame: np.ndarray) -> BarReading:
        return BarReading(hp=0.8, mana=0.5)


class FakeCombat:
    def detect(self, _frame: np.ndarray, now: float | None = None) -> tuple[bool, float]:
        return True, 0.5


class FakeMinimap:
    def update(self, _frame: np.ndarray, *, now: float | None = None) -> MinimapReading:
        return MinimapReading(position_px=(10.0, 20.0), angle_degrees=90.0, confidence=0.93)


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


def frame() -> np.ndarray:
    """A frame carrying a green nameplate-text region for the reaction reader."""
    canvas = np.zeros((40, 120, 3), dtype=np.uint8)
    canvas[:, :] = (20, 20, 20)
    x, y, width, height = REACTION_ROI
    canvas[y : y + height, x : x + width] = GREEN
    return canvas


def pose_reader(
    *,
    min_confidence: float = 0.6,
    clock: FakeClock | None = None,
) -> WorldPoseReader:
    return WorldPoseReader(
        POSE_ROI,
        min_confidence=min_confidence,
        sampling_hz=float("inf"),
        clock=clock or FakeClock(),
    )


def target_reading(bar_width_px: int | None = 120) -> TargetReading:
    return TargetReading(
        name="Defias Thug",
        hp_pct=50,
        frame_confidence=0.9,
        bar_width_px=bar_width_px,
    )


def builder() -> RealStateBuilder:
    """Builder wired with the same target reading the distance channel uses."""
    return RealStateBuilder(
        bars=FakeBars(),
        combat=FakeCombat(),
        minimap=FakeMinimap(),
        target=FakeTarget(target_reading()),
    )


def distance_estimator() -> TargetDistanceEstimator:
    return TargetDistanceEstimator(
        TargetDistanceModel(reference_width_px=120.0, reference_distance_yd=10.0)
    )


def observe(*, pose: WorldPoseReader) -> ChannelObservations:
    return observe_channels(
        frame(),
        pose=pose,
        reaction=TargetReactionReader(REACTION_ROI, min_pixels=12, dominance_thresh=0.6),
        distance=distance_estimator(),
        target_reading=target_reading(),
        now=1.0,
    )


async def run_pipeline(
    observed: ChannelObservations,
    world: WorldModel,
    *,
    player_z: float | None = None,
) -> SyncStats:
    """Drive channels -> injection -> builder -> adapter -> world sync."""
    state = builder().build(frame(), injected=observed.to_injected(player_z=player_z), now=1.0)
    adapter = GameStateAdapter(
        snapshot=state,
        config=AdapterDerivationConfig(engage_distance_units=ENGAGE_DISTANCE_UNITS),
    )
    view = adapter.to_world_sync_view()
    return await WorldSync(world).sync_once(view)


# ----------------------------------------------------------------------
# composition
# ----------------------------------------------------------------------
def test_channels_compose_into_the_builder_injection(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe(pose=pose_reader())

    assert observed.position == pytest.approx((57.3, 42.1))
    assert observed.target_reaction == "friendly"
    assert observed.target_distance_estimate == pytest.approx(10.0)

    injected = observed.to_injected()
    assert injected.position == pytest.approx((57.3, 42.1))
    assert injected.target_reaction == "friendly"
    assert injected.target_distance_estimate == pytest.approx(10.0)
    # The coordinate frame observes no height, and a target world position
    # would be a second-order inference (ADR-003 Decisions 2 and 5).
    assert injected.player_z is None
    assert injected.target_x is None
    assert injected.target_y is None


def test_unwired_channels_report_not_attempted(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe_channels(frame(), pose=pose_reader(), now=1.0)
    assert observed.target_reaction is None
    assert observed.target_distance_estimate is None
    assert "target.reaction" not in observed.confidence()
    assert "target.distance_estimate" not in observed.confidence()


def test_channel_scores_land_in_the_state_confidence_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe(pose=pose_reader())
    injected = observed.to_injected()
    state = builder().build(frame(), injected=injected, now=1.0)

    assert state.perception_confidence["position"] == pytest.approx(0.88)
    assert state.perception_confidence["target.reaction"] == pytest.approx(1.0)
    assert state.perception_confidence["target.distance_estimate"] == pytest.approx(0.9)


def test_an_injected_score_never_overwrites_a_measured_one() -> None:
    injected = InjectedObservations(
        position=(1.0, 2.0),
        confidence={"facing": 0.01, "position": 0.5},
    )
    state = builder().build(frame(), injected=injected, now=1.0)
    assert state.perception_confidence["facing"] == pytest.approx(0.93)
    assert state.perception_confidence["position"] == pytest.approx(0.5)


# ----------------------------------------------------------------------
# acceptance 1: WorldSyncView projects, and writes a player node
# ----------------------------------------------------------------------
async def test_world_sync_view_projects_once_pose_and_player_z_are_supplied(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe(pose=pose_reader())

    # player_z is supplied from outside the channel: the pose reader has none
    # (ADR-003 Decision 2). Without it the view must still raise.
    injected_no_z = observed.to_injected()
    state_no_z = builder().build(frame(), injected=injected_no_z, now=1.0)
    with pytest.raises(Exception, match="player_z"):
        GameStateAdapter(snapshot=state_no_z).to_world_sync_view()

    async with await WorldModel.open(tmp_path / "world.db") as world:
        stats = await run_pipeline(observed, world, player_z=33.0)
        assert stats.player_node_created is True
        assert await world.count_nodes() == 1

        nodes, _edges = await world.all_nodes_and_edges()
        assert (nodes[0].x, nodes[0].y, nodes[0].z) == pytest.approx((57.3, 42.1, 33.0))


# ----------------------------------------------------------------------
# acceptance 2: ReactiveView projects once a distance exists
# ----------------------------------------------------------------------
def test_reactive_view_projects_once_distance_estimate_exists(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe(pose=pose_reader())
    state = builder().build(frame(), injected=observed.to_injected(), now=1.0)

    assert state.target is not None
    assert state.target.distance_estimate == pytest.approx(10.0)
    assert state.target.reaction == "friendly"

    view = GameStateAdapter(
        snapshot=state,
        config=AdapterDerivationConfig(engage_distance_units=ENGAGE_DISTANCE_UNITS),
    ).to_reactive_view()

    assert view.self_x == pytest.approx(57.3)
    assert view.self_y == pytest.approx(42.1)
    assert view.target_in_range is True, "derived from the observed distance"
    assert view.self_in_combat is True


def test_reactive_view_stays_blocked_without_a_distance(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unobserved distance leaves the target absent, not guessed."""
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData())
    observed = observe_channels(
        frame(),
        pose=pose_reader(),
        reaction=TargetReactionReader(REACTION_ROI, min_pixels=12),
        distance=distance_estimator(),
        target_reading=target_reading(bar_width_px=None),
        now=1.0,
    )
    state = builder().build(frame(), injected=observed.to_injected(), now=1.0)
    assert state.target is None

    with pytest.raises(Exception, match="target_in_range"):
        GameStateAdapter(
            snapshot=state,
            config=AdapterDerivationConfig(engage_distance_units=ENGAGE_DISTANCE_UNITS),
        ).to_reactive_view()


# ----------------------------------------------------------------------
# acceptance 3: low confidence -> position None -> no world node
# ----------------------------------------------------------------------
async def test_low_confidence_pose_creates_no_world_node(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData(confidence=40.0))
    observed = observe(pose=pose_reader(min_confidence=0.6))

    assert observed.position is None
    assert observed.confidence() == {} or "position" not in observed.confidence()

    async with await WorldModel.open(tmp_path / "world.db") as world:
        with pytest.raises(BuilderIncompleteError, match="position"):
            await run_pipeline(observed, world, player_z=33.0)
        assert await world.count_nodes() == 0, "a refused pose writes no node"


async def test_low_confidence_pose_emits_no_game_state_at_all(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The pipeline never reaches ``GameState`` construction without a pose."""
    monkeypatch.setitem(sys.modules, "pytesseract", FakeTesseractData(confidence=40.0))
    observed = observe(pose=pose_reader(min_confidence=0.6))
    with pytest.raises(BuilderIncompleteError):
        observed.to_injected()
    assert isinstance(observed, ChannelObservations)
    assert not isinstance(observed.position, tuple)
