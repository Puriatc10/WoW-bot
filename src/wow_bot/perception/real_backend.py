"""Real perception backend (Task T-FIX-32).

Composes capture -> readers -> builder into an async PerceptionBackend:
1. Capture: ScreenCapture or frame device grabbing video frames.
2. Readers:
   - T-FIX-27: BarReader, CombatDetector, MinimapTracker, TargetReader,
     EventDetector, EnemyDetector.
   - T-FIX-30: WorldPoseReader, TargetReactionReader, TargetDistanceEstimator.
   - T-FIX-31: BagFrameReader, XPBarReader, DurabilityReader, CastingReader,
     LootSparkleReader.
3. Builder: RealStateBuilder resolving units/conventions and assembling GameState.

Gated: never selected as the default producer in MOCK_MODE or LAB_MODE.
Stale or missing frames are explicit, never silently reused.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from wow_bot.perception import deps
from wow_bot.perception.bars import BarReader
from wow_bot.perception.builder import (
    BuilderIncompleteError,
    EntityDistance,
    InjectedObservations,
    RealStateBuilder,
)
from wow_bot.perception.capture import FrameLike, ScreenCapture
from wow_bot.perception.combat import CombatDetector
from wow_bot.perception.enemies import EnemyDetector
from wow_bot.perception.events import EventDetector
from wow_bot.perception.minimap import MinimapTracker
from wow_bot.perception.observations import observe_channels
from wow_bot.perception.panels import (
    PanelObservations,
    PanelReaders,
    observe_panels,
    readers_from_config,
)
from wow_bot.perception.perception_config import PerceptionConfig
from wow_bot.perception.pose import WorldPoseReader
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.perception.proximity import (
    TargetDistanceEstimator,
    TargetDistanceModel,
)
from wow_bot.perception.reaction import TargetReactionReader
from wow_bot.perception.target import TargetReader, TargetReading
from wow_bot.shared.interfaces import GameState

__all__ = [
    "FrameMissingError",
    "FrameStaleError",
    "RealPerceptionBackend",
    "RealPerceptionError",
]


class RealPerceptionError(Exception):
    """Base error for real perception backend failures."""


class FrameMissingError(RealPerceptionError):
    """Raised when no frame could be acquired (captured frame is None or missing)."""


class FrameStaleError(RealPerceptionError):
    """Raised when an acquired frame or observation exceeds the staleness bound."""


class RealPerceptionBackend(PerceptionBackend):
    """Async PerceptionBackend composing capture, readers, and RealStateBuilder.

    Contract:
    - Never the configured default producer in MOCK_MODE or LAB_MODE.
    - Stale or missing frames fail loud; never silently reused or substituted.
    - Emits canonical GameState snapshots via async def snapshot().
    """

    def __init__(
        self,
        *,
        capture: ScreenCapture | Callable[[], Any],
        builder: RealStateBuilder,
        pose: WorldPoseReader | None = None,
        reaction: TargetReactionReader | None = None,
        distance: TargetDistanceEstimator | None = None,
        target: TargetReader | None = None,
        panels: PanelReaders | None = None,
        clock: Callable[[], float] = time.monotonic,
        tesseract_cmd: str | None = None,
        player_z: float | None = None,
        max_frame_age_s: float | None = None,
        max_pose_age_s: float | None = None,
    ) -> None:
        self._capture = capture
        self._builder = builder
        self._pose = pose
        self._reaction = reaction
        self._distance = distance
        self._target = target
        self._panels = panels
        self._clock = clock
        self._tesseract_cmd = tesseract_cmd
        self._player_z = player_z
        self._max_frame_age_s = float(max_frame_age_s) if max_frame_age_s is not None else None
        self._max_pose_age_s = float(max_pose_age_s) if max_pose_age_s is not None else None

    @property
    def capture(self) -> ScreenCapture | Callable[[], Any]:
        """Underlying capture device or frame source."""
        return self._capture

    @property
    def builder(self) -> RealStateBuilder:
        """Underlying RealStateBuilder."""
        return self._builder

    @property
    def pose_reader(self) -> WorldPoseReader | None:
        """Configured WorldPoseReader, if any."""
        return self._pose

    @property
    def reaction_reader(self) -> TargetReactionReader | None:
        """Configured TargetReactionReader, if any."""
        return self._reaction

    @property
    def distance_estimator(self) -> TargetDistanceEstimator | None:
        """Configured TargetDistanceEstimator, if any."""
        return self._distance

    @property
    def target_reader(self) -> TargetReader | None:
        """Configured TargetReader, if any."""
        return self._target

    @property
    def panel_readers(self) -> PanelReaders | None:
        """Configured PanelReaders, if any."""
        return self._panels

    @property
    def clock(self) -> Callable[[], float]:
        """Monotonic clock provider."""
        return self._clock

    @property
    def player_z(self) -> float | None:
        """Configured permanent or external player height."""
        return self._player_z

    @property
    def max_frame_age_s(self) -> float | None:
        """Maximum allowable frame staleness bound in seconds."""
        return self._max_frame_age_s

    @property
    def max_pose_age_s(self) -> float | None:
        """Maximum allowable pose sample staleness bound in seconds."""
        return self._max_pose_age_s

    def _acquire_frame(self, now: float) -> FrameLike:
        """Acquire one frame from the capture device, checking freshness."""
        frame: FrameLike | None = None
        frame_timestamp: float | None = None

        if hasattr(self._capture, "get_frame"):
            frame = self._capture.get_frame()
            if frame is None and hasattr(self._capture, "grab"):
                frame = self._capture.grab()
        elif hasattr(self._capture, "grab"):
            frame = self._capture.grab()
        elif callable(self._capture):
            result = self._capture()
            if isinstance(result, tuple) and len(result) == 2:
                frame, frame_timestamp = result
            else:
                frame = result
        else:
            raise TypeError(
                f"capture must be a ScreenCapture, frame device, or callable, got {type(self._capture).__name__}"
            )

        if frame is None:
            raise FrameMissingError("No frame available from capture device (captured frame is None)")

        if frame_timestamp is None and hasattr(self._capture, "latest_frame_time"):
            frame_timestamp = self._capture.latest_frame_time

        if (
            self._max_frame_age_s is not None
            and frame_timestamp is not None
            and (now - frame_timestamp) > self._max_frame_age_s
        ):
            raise FrameStaleError(
                f"Captured frame is stale: age {now - frame_timestamp:.3f}s exceeds "
                f"max_frame_age_s {self._max_frame_age_s:.3f}s"
            )

        return frame

    async def snapshot(self) -> GameState:
        """Capture and return a single authoritative GameState snapshot."""
        now = float(self._clock())
        frame = self._acquire_frame(now)

        # 1. Target reading (cached/shared between distance estimator and builder)
        target_reading: TargetReading | None = None
        target_reader = self._target or getattr(self._builder, "_target", None)
        if target_reader is not None:
            target_reading = target_reader.read(
                frame, now=now, tesseract_cmd=self._tesseract_cmd
            )

        # 2. T-FIX-30 Channel observations (pose, reaction, distance)
        injected: InjectedObservations
        if self._pose is not None:
            channel_obs = observe_channels(
                frame,
                pose=self._pose,
                reaction=self._reaction,
                distance=self._distance,
                target_reading=target_reading,
                now=now,
                tesseract_cmd=self._tesseract_cmd,
            )
            if (
                self._max_pose_age_s is not None
                and channel_obs.pose.sampled_at is not None
                and (now - channel_obs.pose.sampled_at) > self._max_pose_age_s
            ):
                raise FrameStaleError(
                    f"Pose observation is stale: age {now - channel_obs.pose.sampled_at:.3f}s "
                    f"exceeds max_pose_age_s {self._max_pose_age_s:.3f}s"
                )
            injected = channel_obs.to_injected(player_z=self._player_z)
        else:
            raise BuilderIncompleteError(
                "position is a non-Optional GameState field and no world pose reader is configured"
            )

        # 3. T-FIX-31 UI Panel observations
        panel_obs: PanelObservations | None = None
        panel_readers = self._panels or getattr(self._builder, "_panel_readers", None)
        if panel_readers is not None:
            caster_id = (
                target_reading.name
                if (target_reading is not None and target_reading.name is not None)
                else "target"
            )
            panel_obs = observe_panels(
                frame,
                readers=panel_readers,
                now=now,
                tesseract_cmd=self._tesseract_cmd,
                caster_entity_id=caster_id,
            )

        # 4. Compose canonical GameState via RealStateBuilder
        return self._builder.build(
            frame,
            injected=injected,
            panels=panel_obs,
            now=now,
        )

    def start(self) -> None:
        """Start the underlying capture device if it supports start()."""
        if hasattr(self._capture, "start") and not getattr(self._capture, "running", False):
            self._capture.start()

    def stop(self) -> None:
        """Stop the underlying capture device if it supports stop()."""
        if hasattr(self._capture, "stop") and getattr(self._capture, "running", True):
            self._capture.stop()

    def close(self) -> None:
        """Alias for stop()."""
        self.stop()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    async def __aenter__(self) -> Self:
        self.start()
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.stop()

    @classmethod
    def from_config(
        cls,
        config: PerceptionConfig,
        *,
        capture: ScreenCapture | Callable[[], Any] | None = None,
        base_dir: Path | None = None,
        clock: Callable[[], float] = time.monotonic,
        player_z: float | None = None,
        max_frame_age_s: float | None = None,
        max_pose_age_s: float | None = None,
        kind_map: Mapping[str, str] | None = None,
        entity_distance: EntityDistance | None = None,
        strict: bool = False,
    ) -> RealPerceptionBackend:
        """Construct RealPerceptionBackend from validated PerceptionConfig.

        When strict=False, optional readers whose assets are not present
        (YOLO weights, bag templates) degrade gracefully to None rather than
        failing construction.
        """
        bars = BarReader(
            hp_roi=config.hp_roi,
            mana_roi=config.mana_roi,
            sampling_hz=config.bars_sampling_hz,
            clock=clock,
        )
        combat = CombatDetector(
            edge_size=config.combat_edge_size,
            red_ratio_thresh=config.combat_red_ratio_thresh,
            cooldown_s=config.combat_cooldown_s,
            sampling_hz=config.combat_sampling_hz,
            clock=clock,
        )
        minimap = MinimapTracker(
            config.minimap_roi,
            arrow_template=config.minimap_arrow_template,
            smoothing_frames=config.minimap_smoothing_frames,
            sampling_hz=config.minimap_sampling_hz,
            match_thresh=config.minimap_match_thresh,
            base_dir=base_dir,
            clock=clock,
        )

        target_reader: TargetReader | None = None
        try:
            target_reader = TargetReader(
                name_template=config.target_name_template,
                frame_template=config.target_frame_template,
                known_enemies=config.known_enemies,
                match_thresh=config.target_match_thresh,
                confirm_thresh=config.target_confirm_thresh,
                ocr_thresh=config.target_ocr_thresh,
                ocr_confirm_thresh=config.target_ocr_confirm_thresh,
                sampling_hz=config.target_sampling_hz,
                name_refresh_frames=config.target_name_refresh_frames,
                base_dir=base_dir,
                clock=clock,
            )
        except (deps.PerceptionDependencyError, FileNotFoundError):
            if strict:
                raise

        events = EventDetector(
            combat_region=config.combat_region,
            chat_region=config.chat_region,
            sampling_hz=config.events_sampling_hz,
            clock=clock,
        )

        enemies_detector: EnemyDetector | None = None
        try:
            enemies_detector = EnemyDetector(
                yolo_weights=config.yolo_weights,
                confidence=config.yolo_confidence,
                sampling_hz=config.enemies_sampling_hz,
                base_dir=base_dir,
                clock=clock,
            )
        except (deps.PerceptionDependencyError, FileNotFoundError):
            if strict:
                raise

        pose_reader = WorldPoseReader(
            coordinate_roi=config.pose_coordinate_roi,
            min_confidence=config.pose_min_confidence,
            sampling_hz=config.pose_sampling_hz,
            clock=clock,
        )
        reaction_reader = TargetReactionReader(
            nameplate_roi=config.reaction_nameplate_roi,
            min_pixels=config.reaction_min_pixels,
            dominance_thresh=config.reaction_dominance_thresh,
            sampling_hz=config.reaction_sampling_hz,
            clock=clock,
        )
        proximity_estimator = TargetDistanceEstimator(
            TargetDistanceModel(
                reference_width_px=config.proximity_reference_width_px,
                reference_distance_yd=config.proximity_reference_distance_yd,
            ),
            min_confidence=config.proximity_min_confidence,
        )

        panel_readers: PanelReaders | None = None
        try:
            panel_readers = readers_from_config(config, base_dir=base_dir, clock=clock)
        except (deps.PerceptionDependencyError, FileNotFoundError):
            if strict:
                raise
            # Degrade gracefully to template-free panels (xp, durability, cast, loot)
            from wow_bot.perception.cast import CastingReader
            from wow_bot.perception.durability import DurabilityReader
            from wow_bot.perception.loot import LootSparkleReader
            from wow_bot.perception.xp import XPBarReader

            panel_readers = PanelReaders(
                bag=None,
                xp=XPBarReader(
                    config.xp_bar_roi,
                    level_roi=config.xp_level_roi,
                    min_confidence=config.xp_min_confidence,
                    sampling_hz=config.xp_sampling_hz,
                    clock=clock,
                ),
                durability=DurabilityReader(
                    config.durability_slot_rois,
                    min_score=config.durability_min_score,
                    quorum_fraction=config.durability_quorum_fraction,
                    min_confidence=config.durability_min_confidence,
                    sampling_hz=config.durability_sampling_hz,
                    clock=clock,
                ),
                casting=CastingReader(
                    config.cast_roi,
                    spell_ids=config.cast_spell_ids,
                    border_roi=config.cast_border_roi,
                    border_interruptible=config.cast_border_interruptible,
                    min_confidence=config.cast_min_confidence,
                    max_remaining_s=config.cast_max_remaining_s,
                    sampling_hz=config.cast_sampling_hz,
                    clock=clock,
                ),
                loot=LootSparkleReader(
                    config.loot_sparkle_roi,
                    min_pixels=config.loot_min_pixels,
                    dominance_thresh=config.loot_dominance_thresh,
                    min_confidence=config.loot_min_confidence,
                    sampling_hz=config.loot_sampling_hz,
                    clock=clock,
                ),
            )

        builder = RealStateBuilder(
            bars=bars,
            combat=combat,
            minimap=minimap,
            target=target_reader,
            events=events,
            enemies=enemies_detector,
            panel_readers=panel_readers,
            clock=clock,
            tesseract_cmd=config.tesseract_cmd,
            kind_map=kind_map,
            entity_distance=entity_distance,
        )

        capture_device = (
            capture
            if capture is not None
            else ScreenCapture(idle_fps=config.idle_fps, combat_fps=config.combat_fps)
        )

        return cls(
            capture=capture_device,
            builder=builder,
            pose=pose_reader,
            reaction=reaction_reader,
            distance=proximity_estimator,
            target=target_reader,
            panels=panel_readers,
            clock=clock,
            tesseract_cmd=config.tesseract_cmd,
            player_z=player_z,
            max_frame_age_s=max_frame_age_s,
            max_pose_age_s=max_pose_age_s,
        )
