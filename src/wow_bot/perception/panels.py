"""Compose the T-FIX-31 UI panel channels into one frame observation.

The five panel readers (bag, XP bar, character frame, enemy cast bar,
lootable corpse) each own one ``docs/PERCEPTION.md`` §2.1 channel. This
module is the single place that runs whichever of them are wired over one
frame and projects the results into the shape the ``GameState`` builder
consumes, so no caller has to re-implement the "which values are
trustworthy" policy.

**The policy this module owns:**

* every field is projected only from a reading that actually emitted a value;
  an unobserved or withheld reading leaves the field ``None`` / ``()``;
* ``incoming_casts`` is the honest empty tuple when no cast was observed;
* ``target_is_lootable`` stays ``None`` while
  :data:`wow_bot.perception.loot.LOOT_CHANNEL_MEASURED` is ``False`` — the
  measured sparkle score is still carried in :meth:`PanelObservations.confidence`
  so the channel is observable, but the assertion is withheld until §6's
  precision/recall is measurable;
* a channel that is not wired at all is "not attempted": no reader, no entry
  in the confidence map, no inferred value.

**Confidence.** Each emitted panel field contributes its measured score under
the same dotted field-path convention the T-FIX-28 builder and the T-FIX-30
observation seam use (``inventory_count``, ``level_or_xp``,
``durability_fraction``, ``incoming_casts``), so
``GameState.perception_confidence`` stays a single flat map.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path

from wow_bot.perception.bag import BagFrameReader, BagReading
from wow_bot.perception.capture import FrameLike
from wow_bot.perception.cast import CastingReader, CastingReading
from wow_bot.perception.durability import DurabilityReader, DurabilityReading
from wow_bot.perception.loot import LootSparkleReader, LootSparkleReading
from wow_bot.perception.perception_config import PerceptionConfig
from wow_bot.perception.xp import XPBarReader, XPBarReading
from wow_bot.shared.interfaces import IncomingCast

__all__ = [
    "PanelObservations",
    "PanelReaders",
    "observe_panels",
    "readers_from_config",
]

#: Reading used when a channel is not wired: "not attempted", confidence 0.
_UNOBSERVED_BAG = BagReading(inventory_count=None, inventory_max=None, confidence=0.0)
_UNOBSERVED_XP = XPBarReading(
    level_or_xp=None, level=None, xp_fraction=None, confidence=0.0
)
_UNOBSERVED_DURABILITY = DurabilityReading(durability_fraction=None, confidence=0.0)
_UNOBSERVED_CASTING = CastingReading()
_UNOBSERVED_LOOT = LootSparkleReading(target_is_lootable=None, confidence=0.0)


@dataclass(frozen=True)
class PanelReaders:
    """The panel readers that are actually wired for a session.

    ``None`` for a channel means "not wired", which is different from a wired
    reader that measured nothing: the former never appears in the confidence
    map, the latter appears with its measured score.
    """

    bag: BagFrameReader | None = None
    xp: XPBarReader | None = None
    durability: DurabilityReader | None = None
    casting: CastingReader | None = None
    loot: LootSparkleReader | None = None


@dataclass(frozen=True)
class PanelObservations:
    """What the T-FIX-31 panel channels observed on one frame."""

    bag: BagReading = _UNOBSERVED_BAG
    xp: XPBarReading = _UNOBSERVED_XP
    durability: DurabilityReading = _UNOBSERVED_DURABILITY
    casting: CastingReading = _UNOBSERVED_CASTING
    loot: LootSparkleReading = _UNOBSERVED_LOOT
    #: Field path -> measured score, for the values this frame emitted.
    confidence: dict[str, float] = field(default_factory=dict)

    @property
    def inventory_count(self) -> int | None:
        """The observed occupied-slot count, or ``None`` when withheld."""
        return self.bag.inventory_count

    @property
    def inventory_max(self) -> int | None:
        """The observed grid capacity, or ``None`` when the grid was unread."""
        return self.bag.inventory_max

    @property
    def level_or_xp(self) -> float | None:
        """The observed ``level + xp_fraction`` scalar, or ``None``."""
        return self.xp.level_or_xp

    @property
    def durability_fraction(self) -> float | None:
        """The observed durability mean, or ``None`` when below quorum."""
        return self.durability.durability_fraction

    @property
    def target_is_lootable(self) -> bool | None:
        """The observed lootability, or ``None`` while the channel is gated."""
        return self.loot.target_is_lootable

    @property
    def incoming_casts(self) -> tuple[IncomingCast, ...]:
        """The observed casts; ``()`` is the honest "none observed"."""
        return self.casting.casts


def _panel_confidence(
    *,
    bag: BagReading,
    xp: XPBarReading,
    durability: DurabilityReading,
    casting: CastingReading,
    loot: LootSparkleReading,
) -> dict[str, float]:
    """Measured scores for the panel values this frame actually emitted."""
    scores: dict[str, float] = {}
    if bag.inventory_count is not None:
        scores["inventory_count"] = float(bag.confidence)
    if xp.level_or_xp is not None:
        scores["level_or_xp"] = float(xp.confidence)
    if durability.durability_fraction is not None:
        scores["durability_fraction"] = float(durability.confidence)
    # A cast list carries one cast, so one score answers for the field.
    if casting.casts:
        scores["incoming_casts"] = float(casting.confidence)
    if loot.target_is_lootable is not None:
        scores["target_is_lootable"] = float(loot.confidence)
    return scores


def observe_panels(
    frame: FrameLike,
    *,
    readers: PanelReaders,
    now: float | None = None,
    tesseract_cmd: str | None = None,
    caster_entity_id: str = "target",
) -> PanelObservations:
    """Run the wired T-FIX-31 channels over one frame.

    ``caster_entity_id`` names the entity the single observed cast bar
    belongs to; it is the target the state builder already resolved, never an
    id invented by the reader. Every channel is throttled independently, so
    passing the same frame twice does not re-run an OCR channel ahead of its
    budget.
    """
    bag = (
        _UNOBSERVED_BAG
        if readers.bag is None
        else readers.bag.read(frame, now=now)
    )
    xp = (
        _UNOBSERVED_XP
        if readers.xp is None
        else readers.xp.read(frame, now=now, tesseract_cmd=tesseract_cmd)
    )
    durability = (
        _UNOBSERVED_DURABILITY
        if readers.durability is None
        else readers.durability.read(frame, now=now, tesseract_cmd=tesseract_cmd)
    )
    casting = (
        _UNOBSERVED_CASTING
        if readers.casting is None
        else readers.casting.read(
            frame,
            now=now,
            caster_entity_id=caster_entity_id,
            tesseract_cmd=tesseract_cmd,
        )
    )
    loot = (
        _UNOBSERVED_LOOT
        if readers.loot is None
        else readers.loot.read(frame, now=now)
    )

    observations = PanelObservations(
        bag=bag, xp=xp, durability=durability, casting=casting, loot=loot
    )
    return replace(
        observations,
        confidence=_panel_confidence(
            bag=bag, xp=xp, durability=durability, casting=casting, loot=loot
        ),
    )


def readers_from_config(
    config: PerceptionConfig,
    *,
    base_dir: Path | None = None,
    clock: Callable[[], float] | None = None,
) -> PanelReaders:
    """Build the five panel readers from a validated ``PerceptionConfig``.

    ``base_dir`` resolves the two relative template paths, exactly as
    :class:`~wow_bot.perception.target.TargetReader` does; a relative path
    with no ``base_dir`` resolves against the process working directory.

    **Eager and fail-closed.** ``BagFrameReader`` loads its templates in its
    constructor, so a missing template raises
    :class:`~wow_bot.perception.deps.PerceptionDependencyError` at wiring
    time rather than on the first frame. A caller that only wants the four
    template-free channels can construct ``PanelReaders`` directly.
    """
    clock = time.monotonic if clock is None else clock
    return PanelReaders(
        bag=BagFrameReader(
            config.bag_grid_origin,
            config.bag_columns,
            config.bag_rows,
            config.bag_slot_size,
            empty_slot_template=config.bag_empty_slot_template,
            occupied_slot_template=config.bag_occupied_slot_template,
            gap=config.bag_gap,
            min_confidence=config.bag_min_confidence,
            sampling_hz=config.bag_sampling_hz,
            base_dir=base_dir,
            clock=clock,
        ),
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
