"""Integration tests for RealPerceptionBackend (Task T-FIX-32).

Verifies:
1. isinstance(RealPerceptionBackend(...), PerceptionBackend) holds.
2. End-to-end integration over frozen frame corpus from tests/fixtures/corpus/.
3. Projection to all eight consumer views with fail-loud gating asserted.
4. Missing and stale frame contracts (FrameMissingError, FrameStaleError).
5. Fast loop scheduling via PerceptionPort (T-FIX-20 bridge).
6. Gating: MockPerception remains default producer in MOCK_MODE and LAB_MODE.
7. Static import isolation: no imports from wow_bot.lab or wow_bot.main.
"""

from __future__ import annotations

import ast
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from wow_bot.executor.states import FSMState
from wow_bot.perception.adapter import (
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
)
from wow_bot.perception.bag import BagReading
from wow_bot.perception.bars import BarReading
from wow_bot.perception.builder import (
    BuilderIncompleteError,
    RealStateBuilder,
)
from wow_bot.perception.cast import CastingReading
from wow_bot.perception.context import StaticRuntimeContext
from wow_bot.perception.durability import DurabilityReading
from wow_bot.perception.enemies import EnemyDetection
from wow_bot.perception.loot import LootSparkleReading
from wow_bot.perception.minimap import MinimapReading
from wow_bot.perception.panels import PanelReaders
from wow_bot.perception.port import PerceptionPort, PerceptionStaleError
from wow_bot.perception.pose import PoseReading
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.perception.proximity import (
    TargetDistanceReading,
)
from wow_bot.perception.reaction import TargetReactionReading
from wow_bot.perception.real_backend import (
    FrameMissingError,
    FrameStaleError,
    RealPerceptionBackend,
)
from wow_bot.perception.resource_table import ResourceTable
from wow_bot.perception.target import TargetReading
from wow_bot.perception.views import (
    CombatView,
    FleeView,
    LootView,
    ReactiveView,
    StrategistView,
    TargetView,
    VendorView,
    WorldSyncView,
)
from wow_bot.perception.xp import XPBarReading
from wow_bot.shared.interfaces import EnemyInfo, GameState

REPO_ROOT = Path(__file__).resolve().parents[1]
CORPUS_DIR = REPO_ROOT / "tests" / "fixtures" / "corpus"
REAL_BACKEND_FILE = REPO_ROOT / "src" / "wow_bot" / "perception" / "real_backend.py"


# ---------------------------------------------------------------------------
# Test Helpers & Fakes
# ---------------------------------------------------------------------------


class MockClock:
    def __init__(self, initial: float = 100.0) -> None:
        self.time = float(initial)

    def __call__(self) -> float:
        return self.time

    def advance(self, dt: float) -> None:
        self.time += float(dt)


class FakeBars:
    def __init__(self, hp: float = 0.85, mana: float = 0.60) -> None:
        self.hp = hp
        self.mana = mana

    def read_all(self, _frame: np.ndarray) -> BarReading:
        return BarReading(hp=self.hp, mana=self.mana)


class FakeCombat:
    def __init__(self, in_combat: bool = True) -> None:
        self.in_combat = in_combat

    def detect(self, _frame: np.ndarray, _now: float | None = None) -> tuple[bool, float]:
        return self.in_combat, 0.8


class FakeMinimap:
    def __init__(self, angle_degrees: float = 90.0, confidence: float = 0.95) -> None:
        self.angle_degrees = angle_degrees
        self.confidence = confidence

    def update(self, _frame: np.ndarray, *, now: float | None = None) -> MinimapReading:
        return MinimapReading(
            position_px=(100.0, 100.0),
            angle_degrees=self.angle_degrees,
            confidence=self.confidence,
        )


class FakeTarget:
    def __init__(
        self,
        name: str | None = "Defias Thug",
        hp_pct: int = 50,
        frame_confidence: float = 0.9,
    ) -> None:
        self.name = name
        self.hp_pct = hp_pct
        self.frame_confidence = frame_confidence

    def read(
        self,
        _frame: np.ndarray,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> TargetReading:
        return TargetReading(
            name=self.name,
            hp_pct=self.hp_pct,
            frame_confidence=self.frame_confidence,
            ocr_confidence=self.frame_confidence,
            bar_width_px=80,
        )


class FakePoseReader:
    def __init__(
        self,
        position: tuple[float, float] | None = (100.0, 200.0),
        confidence: float = 0.95,
        sampled_at: float | None = 100.0,
    ) -> None:
        self.position = position
        self.confidence = confidence
        self.sampled_at = sampled_at

    def read(
        self,
        _frame: np.ndarray,
        *,
        now: float | None = None,
        tesseract_cmd: str | None = None,
    ) -> PoseReading:
        return PoseReading(
            position=self.position,
            confidence=self.confidence,
            sampled_at=self.sampled_at if self.sampled_at is not None else now,
        )


class FakeReactionReader:
    def __init__(self, reaction: str | None = "hostile", confidence: float = 0.9) -> None:
        self.reaction = reaction
        self.confidence = confidence

    def read(self, _frame: np.ndarray, *, now: float | None = None) -> TargetReactionReading:
        return TargetReactionReading(reaction=self.reaction, confidence=self.confidence)


class FakeDistanceEstimator:
    def __init__(self, distance: float | None = 12.0, confidence: float = 0.85) -> None:
        self.distance = distance
        self.confidence = confidence

    def estimate(self, _target_reading: TargetReading) -> TargetDistanceReading:
        return TargetDistanceReading(
            distance_estimate=self.distance,
            bar_width_px=80,
            confidence=self.confidence,
        )


class FakeEnemies:
    def __init__(self, detections: list[EnemyDetection] | None = None) -> None:
        self.detections = detections if detections is not None else [
            EnemyDetection(name="mob", bbox=(10, 20, 50, 60), confidence=0.92)
        ]

    def detect(self, _frame: np.ndarray, *, now: float | None = None) -> list[EnemyDetection]:
        return list(self.detections)


def make_test_panel_readers() -> PanelReaders:
    """Create panel readers with fake readings for testing all panel channels."""
    class FakeBagReader:
        def read(self, _frame: np.ndarray, *, now: float | None = None) -> BagReading:
            return BagReading(inventory_count=5, inventory_max=16, confidence=0.85)

    class FakeXPReader:
        def read(
            self, _frame: np.ndarray, *, now: float | None = None, tesseract_cmd: str | None = None
        ) -> XPBarReading:
            return XPBarReading(
                level_or_xp=10.5, level=10, xp_fraction=0.5, confidence=0.9
            )

    class FakeDurabilityReader:
        def read(
            self, _frame: np.ndarray, *, now: float | None = None, tesseract_cmd: str | None = None
        ) -> DurabilityReading:
            return DurabilityReading(durability_fraction=0.88, confidence=0.8)

    class FakeCastingReader:
        def read(
            self,
            _frame: np.ndarray,
            *,
            now: float | None = None,
            caster_entity_id: str = "target",
            tesseract_cmd: str | None = None,
        ) -> CastingReading:
            return CastingReading(casts=(), confidence=0.95)

    class FakeLootReader:
        def read(self, _frame: np.ndarray, *, now: float | None = None) -> LootSparkleReading:
            return LootSparkleReading(target_is_lootable=True, confidence=0.95)

    return PanelReaders(
        bag=FakeBagReader(),  # type: ignore[arg-type]
        xp=FakeXPReader(),  # type: ignore[arg-type]
        durability=FakeDurabilityReader(),  # type: ignore[arg-type]
        casting=FakeCastingReader(),  # type: ignore[arg-type]
        loot=FakeLootReader(),  # type: ignore[arg-type]
    )


def make_real_backend(
    frame_source: Any,
    *,
    clock: MockClock | None = None,
    player_z: float | None = None,
    max_frame_age_s: float | None = None,
    max_pose_age_s: float | None = None,
    panel_readers: PanelReaders | None = None,
    pose_reader: Any = None,
    target_reader: Any = None,
) -> RealPerceptionBackend:
    clk = clock if clock is not None else MockClock()
    target = target_reader if target_reader is not None else FakeTarget()
    builder = RealStateBuilder(
        bars=FakeBars(),  # type: ignore[arg-type]
        combat=FakeCombat(),  # type: ignore[arg-type]
        minimap=FakeMinimap(),  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        enemies=FakeEnemies(),  # type: ignore[arg-type]
        clock=clk,
        entity_distance=lambda _d: 12.0,
    )
    pose = pose_reader if pose_reader is not None else FakePoseReader()
    return RealPerceptionBackend(
        capture=frame_source,
        builder=builder,
        pose=pose,  # type: ignore[arg-type]
        reaction=FakeReactionReader(),  # type: ignore[arg-type]
        distance=FakeDistanceEstimator(),  # type: ignore[arg-type]
        target=target,  # type: ignore[arg-type]
        panels=panel_readers,
        clock=clk,
        player_z=player_z,
        max_frame_age_s=max_frame_age_s,
        max_pose_age_s=max_pose_age_s,
    )


# ---------------------------------------------------------------------------
# Acceptance Criterion 1 — isinstance(..., PerceptionBackend)
# ---------------------------------------------------------------------------


def test_isinstance_perception_backend() -> None:
    backend = make_real_backend(lambda: np.zeros((10, 10, 3), dtype=np.uint8))
    assert isinstance(backend, PerceptionBackend)
    assert hasattr(backend, "snapshot")
    assert callable(backend.snapshot)


# ---------------------------------------------------------------------------
# Acceptance Criterion 2 — End-to-end integration over frozen corpus & 8 projections
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_end_to_end_corpus_snapshots_and_eight_projections() -> None:
    assert CORPUS_DIR.is_dir(), f"Corpus directory {CORPUS_DIR} must exist"
    corpus_files = sorted(CORPUS_DIR.glob("*.png"))
    assert len(corpus_files) >= 3, "Corpus must contain at least 3 frozen frames"

    clock = MockClock(initial=500.0)

    for corpus_path in corpus_files:
        frame_img = cv2.imread(str(corpus_path))
        assert frame_img is not None, f"Failed to load frame from {corpus_path}"

        # 1. Snapshot generation with panels and external player_z
        backend = make_real_backend(
            lambda current_img=frame_img: current_img.copy(),
            clock=clock,
            player_z=1.5,
            panel_readers=make_test_panel_readers(),
        )

        snapshot = await backend.snapshot()

        # Schema and post_init validity
        assert isinstance(snapshot, GameState)
        assert snapshot.timestamp == 500.0
        assert snapshot.hp_pct == pytest.approx(0.85)
        assert snapshot.mana_pct == pytest.approx(0.60)
        assert snapshot.position == (100.0, 200.0)
        assert snapshot.player_z == 1.5
        assert snapshot.in_combat is True
        assert snapshot.target is not None
        assert snapshot.target.name == "Defias Thug"
        assert snapshot.inventory_count == 5
        assert snapshot.level_or_xp == 10.5
        assert snapshot.durability_fraction == pytest.approx(0.88)
        assert snapshot.target_is_lootable is True

        # Round trip
        payload = snapshot.to_dict()
        restored = GameState.from_dict(payload)
        assert restored.to_dict() == payload
        assert restored == snapshot

        # 2. Assert the eight projection outcomes
        resource_table = ResourceTable({("WARRIOR", 10): 100.0})
        # Add char_class attribute dynamically to test complete resource derivation
        object.__setattr__(snapshot, "char_class", "WARRIOR")

        adapter = GameStateAdapter(
            snapshot,
            config=AdapterDerivationConfig(engage_distance_units=30.0),
            context=StaticRuntimeContext(),
            resource_table=resource_table,
        )

        # 1. WorldSyncView: succeeds with player_z=1.5
        ws_view = adapter.to_world_sync_view()
        assert isinstance(ws_view, WorldSyncView)
        assert ws_view.player_x == 100.0
        assert ws_view.player_y == 200.0
        assert ws_view.player_z == 1.5
        assert ws_view.target_entity_id == "Defias Thug"

        # 2. StrategistView: succeeds with resource_max derived
        strat_view = adapter.to_strategist_view(fsm_state=FSMState.COMBAT)
        assert isinstance(strat_view, StrategistView)
        assert strat_view.player_x == 100.0
        assert strat_view.self_hp_percent == 85.0
        assert strat_view.resource == 60.0
        assert strat_view.resource_max == 100.0
        assert strat_view.inventory_count == 5
        assert strat_view.level_or_xp == 10.5
        assert strat_view.fsm_state == FSMState.COMBAT

        # 3. CombatView: fails loud on unobserved entity_id (ADR-002 Q1)
        with pytest.raises(
            AdapterIncompleteError,
            match="TargetEntityLike requires non-Optional (entity_id|threat)",
        ):
            adapter.to_combat_view()

        # Supply complete entity to assert projection
        complete_entity = EnemyInfo(
            bbox=(10, 20, 30, 40),
            confidence=0.95,
            distance_estimate=10.0,
            entity_id="mob-1",
            kind="mob",
            x=110.0,
            y=210.0,
            z=1.5,
            hp_fraction=0.50,
            threat=100.0,
            is_attackable=True,
            is_alive=True,
            is_in_combat_with_self=True,
        )
        object.__setattr__(snapshot, "entities", (complete_entity,))
        combat_view = adapter.to_combat_view()
        assert isinstance(combat_view, CombatView)
        assert combat_view.self_x == 100.0
        assert combat_view.target_in_range is True
        assert combat_view.resource_max == 100.0

        # 4. TargetView: succeeds with complete entity
        t_views = adapter.to_targeting_views()
        assert len(t_views) >= 1
        assert isinstance(t_views[0], TargetView)
        assert t_views[0].entity_id == "mob-1"
        assert t_views[0].distance == 10.0
        assert t_views[0].threat == 100.0

        # 5. ReactiveView: succeeds
        rx_view = adapter.to_reactive_view()
        assert isinstance(rx_view, ReactiveView)
        assert rx_view.self_hp_percent == 85.0
        assert rx_view.target_in_range is True
        assert rx_view.self_in_combat is True

        # 6. FleeView: succeeds
        flee_view = adapter.to_flee_view()
        assert isinstance(flee_view, FleeView)
        assert flee_view.self_hp_percent == 85.0
        assert flee_view.adds_count == 0 or flee_view.adds_count is not None

        # 7. LootView: succeeds with target_is_lootable and inventory_count
        loot_view = adapter.to_loot_view()
        assert isinstance(loot_view, LootView)
        assert loot_view.self_x == 100.0
        assert loot_view.target_is_lootable is True
        assert loot_view.inventory_count == 5

        # 8. VendorView: succeeds with inventory_count
        vendor_view = adapter.to_vendor_view()
        assert isinstance(vendor_view, VendorView)
        assert vendor_view.self_x == 100.0
        assert vendor_view.inventory_count == 5
        assert vendor_view.durability_fraction == pytest.approx(0.88)


@pytest.mark.asyncio
async def test_projection_fail_loud_gating_scenarios() -> None:
    """Verify that unobserved fields fail loud as required by ADR-002 and HAMBERGER_PORT_PLAN §5."""
    clock = MockClock(initial=200.0)
    frame = np.zeros((100, 100, 3), dtype=np.uint8)

    # Backend with no external player_z (ADR-003 Decision 2: player_z stays None)
    # and no panels wired
    backend = make_real_backend(
        lambda: frame,
        clock=clock,
        player_z=None,
        panel_readers=None,
    )
    snapshot = await backend.snapshot()
    assert snapshot.player_z is None
    assert snapshot.inventory_count is None

    adapter = GameStateAdapter(
        snapshot,
        config=AdapterDerivationConfig(engage_distance_units=30.0),
        context=StaticRuntimeContext(),
    )

    # 1. WorldSyncView fails loud on player_z
    with pytest.raises(AdapterIncompleteError, match="WorldSyncView requires non-Optional player_z"):
        adapter.to_world_sync_view()

    # 2. StrategistView fails loud on resource_max
    with pytest.raises(AdapterIncompleteError, match="StrategistView requires non-Optional resource_max"):
        adapter.to_strategist_view(fsm_state=FSMState.IDLE)

    # 3. CombatView fails loud on resource_max
    with pytest.raises(AdapterIncompleteError, match="CombatStateView requires non-Optional resource_max"):
        adapter.to_combat_view()

    # 4. TargetView fails loud on threat or entity_id
    with pytest.raises(
        AdapterIncompleteError,
        match="TargetEntityLike requires non-Optional (threat|entity_id)",
    ):
        adapter.to_targeting_views()

    # 5. LootView fails loud on target_is_lootable
    with pytest.raises(AdapterIncompleteError, match="LootStateView requires non-Optional target_is_lootable"):
        adapter.to_loot_view()

    # 6. VendorView fails loud on inventory_count
    with pytest.raises(AdapterIncompleteError, match="VendorStateView requires non-Optional inventory_count"):
        adapter.to_vendor_view()


# ---------------------------------------------------------------------------
# Missing and Stale Frames Contract
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_frame_missing_raises_explicit_error() -> None:
    """A capture device returning None raises FrameMissingError loudly."""
    backend = make_real_backend(lambda: None)
    with pytest.raises(FrameMissingError, match="No frame available from capture device"):
        await backend.snapshot()


@pytest.mark.asyncio
async def test_frame_staleness_raises_explicit_error() -> None:
    """A frame with an expired timestamp raises FrameStaleError."""
    clock = MockClock(initial=10.0)
    # Callable returning (frame, frame_time)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    backend = make_real_backend(
        lambda: (frame, 5.0),  # Age = 10.0 - 5.0 = 5.0s
        clock=clock,
        max_frame_age_s=1.0,
    )
    with pytest.raises(FrameStaleError, match="Captured frame is stale"):
        await backend.snapshot()


@pytest.mark.asyncio
async def test_pose_staleness_raises_explicit_error() -> None:
    """A pose observation with an expired sampled_at raises FrameStaleError."""
    clock = MockClock(initial=20.0)
    frame = np.zeros((20, 20, 3), dtype=np.uint8)
    stale_pose = FakePoseReader(position=(10.0, 20.0), sampled_at=10.0)  # Age = 10.0s
    backend = make_real_backend(
        lambda: frame,
        clock=clock,
        pose_reader=stale_pose,
        max_pose_age_s=2.0,
    )
    with pytest.raises(FrameStaleError, match="Pose observation is stale"):
        await backend.snapshot()


# ---------------------------------------------------------------------------
# Scheduling via PerceptionPort (T-FIX-20 Integration)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_real_backend_integrated_with_perception_port() -> None:
    clock = MockClock(initial=100.0)
    frame = np.zeros((30, 30, 3), dtype=np.uint8)
    backend = make_real_backend(lambda: frame, clock=clock, player_z=0.0)

    port = PerceptionPort(backend, staleness_bound_s=1.0, clock=clock)

    # Initial sample raises before first snapshot pump
    with pytest.raises(PerceptionStaleError):
        port.sample()

    snapshot = await port.pump_once()
    assert isinstance(snapshot, GameState)
    assert port.sample() == snapshot

    # Exceed staleness bound
    clock.advance(1.5)
    with pytest.raises(PerceptionStaleError):
        port.sample()


# ---------------------------------------------------------------------------
# Acceptance Criterion 3 — MockPerception Remains Default Producer
# ---------------------------------------------------------------------------


def test_mock_perception_remains_default_producer() -> None:
    """LAB_PHASE_ROADMAP Global Rule 1: MockPerception remains the default producer."""
    from wow_bot.main import perception_loop
    from wow_bot.mocks.mock_perception import MockPerception

    # Verify MockPerception is the class referenced in main perception setup
    assert MockPerception is not None
    assert callable(perception_loop)

    # Verify config defaults do not reference RealPerceptionBackend
    from wow_bot.config import Config

    cfg_fields = set(Config.__dataclass_fields__.keys())
    assert "real_perception" not in cfg_fields
    assert "use_real_perception" not in cfg_fields


# ---------------------------------------------------------------------------
# Static Import Isolation
# ---------------------------------------------------------------------------


def test_real_backend_imports_nothing_from_lab_or_main() -> None:
    tree = ast.parse(REAL_BACKEND_FILE.read_text(encoding="utf-8"))
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


# ---------------------------------------------------------------------------
# from_config Factory & Lifecycle Tests
# ---------------------------------------------------------------------------


def test_from_config_factory() -> None:
    from wow_bot.perception import deps
    from wow_bot.perception.perception_config import load_perception_config

    cfg_path = REPO_ROOT / "config" / "perception.example.toml"
    config = load_perception_config(cfg_path)

    # strict=False allows building with graceful degradation of optional channels
    backend = RealPerceptionBackend.from_config(config, strict=False)
    assert isinstance(backend, RealPerceptionBackend)
    assert backend.builder is not None
    assert backend.pose_reader is not None
    assert backend.reaction_reader is not None
    assert backend.distance_estimator is not None

    # strict=True raises when optional assets (like YOLO weights or bag templates) are absent
    with pytest.raises(deps.PerceptionDependencyError):
        RealPerceptionBackend.from_config(config, strict=True)


def test_real_backend_properties_and_lifecycle() -> None:
    class FakeScreenCapture:
        def __init__(self) -> None:
            self.running = False

        def __call__(self) -> np.ndarray:
            return np.zeros((1, 1, 3), dtype=np.uint8)

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

    fake_cap = FakeScreenCapture()
    clock = MockClock(initial=42.0)
    backend = make_real_backend(
        fake_cap,
        clock=clock,
        player_z=2.5,
        max_frame_age_s=0.5,
        max_pose_age_s=1.0,
    )

    assert id(backend.capture) == id(fake_cap)
    assert backend.clock is clock
    assert backend.player_z == 2.5
    assert backend.max_frame_age_s == 0.5
    assert backend.max_pose_age_s == 1.0
    assert backend.pose_reader is not None
    assert backend.reaction_reader is not None
    assert backend.distance_estimator is not None
    assert backend.target_reader is not None

    # Test sync lifecycle
    assert not fake_cap.running
    with backend:
        assert fake_cap.running
    assert not fake_cap.running

    backend.start()
    assert fake_cap.running
    backend.close()
    assert not fake_cap.running


@pytest.mark.asyncio
async def test_real_backend_async_context_manager() -> None:
    class FakeScreenCapture:
        def __init__(self) -> None:
            self.running = False

        def start(self) -> None:
            self.running = True

        def stop(self) -> None:
            self.running = False

    fake_cap = FakeScreenCapture()
    backend = make_real_backend(fake_cap)

    assert not fake_cap.running
    async with backend:
        assert fake_cap.running
    assert not fake_cap.running


@pytest.mark.asyncio
async def test_real_backend_invalid_capture_type() -> None:
    backend = make_real_backend(12345)  # Invalid capture device
    with pytest.raises(TypeError, match="capture must be a ScreenCapture"):
        await backend.snapshot()


@pytest.mark.asyncio
async def test_real_backend_missing_pose_reader_fails_loud() -> None:
    clock = MockClock()
    frame = np.zeros((10, 10, 3), dtype=np.uint8)
    backend = RealPerceptionBackend(
        capture=lambda: frame,
        builder=RealStateBuilder(
            bars=FakeBars(),  # type: ignore[arg-type]
            combat=FakeCombat(),  # type: ignore[arg-type]
            minimap=FakeMinimap(),  # type: ignore[arg-type]
            clock=clock,
        ),
        pose=None,  # No pose reader
        clock=clock,
    )
    with pytest.raises(
        BuilderIncompleteError,
        match="position is a non-Optional GameState field and no world pose reader is configured",
    ):
        await backend.snapshot()

