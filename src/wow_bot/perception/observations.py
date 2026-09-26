"""Compose the T-FIX-30 channels into the builder's injected observations.

T-FIX-28's :class:`~wow_bot.perception.builder.RealStateBuilder` deliberately
does not observe ``position``, ``target_reaction``, or
``target_distance_estimate``: they arrive through
:class:`~wow_bot.perception.builder.InjectedObservations`. This module is the
single place where the three T-FIX-30 readers
(:mod:`~wow_bot.perception.pose`, :mod:`~wow_bot.perception.reaction`,
:mod:`~wow_bot.perception.proximity`) are turned into that injection, so no
caller has to re-implement the "which values are trustworthy" policy.

**The policy this module owns:**

* ``position`` is emitted only from a pose sample that parsed **and** cleared
  its confidence floor. When it did not, :meth:`ChannelObservations.to_injected`
  raises :class:`~wow_bot.perception.builder.BuilderIncompleteError` instead of
  emitting a fabricated pose, because a wrong pose writes a wrong node into the
  world model. The pipeline therefore produces no ``GameState`` at all for that
  frame, and the world sync layer is never reached.
* ``player_z`` is **always** ``None``: the coordinate frame observes two
  dimensions and the readers never invent a third (ADR-003 Decision 2).
* ``target_x`` / ``target_y`` are **always** ``None``: a world-space target
  position would have to be derived from pose + bearing + distance, a
  second-order inference with no validated error model (ADR-003 Decision 5).
* the reaction and the distance are passed through only when their own readers
  emitted them; a missing one leaves its field ``None``, and the builder then
  omits the target rather than building a partial ``TargetInfo``.

**Confidence.** The measured scores of the three channels ride along in
``confidence`` so the builder can place them in
``GameState.perception_confidence``. Only emitted values get an entry: an
unobserved channel is absent from the map (ADR-002 Decision 4: "absence means
not attempted").
"""

from __future__ import annotations

from dataclasses import dataclass

from wow_bot.perception.builder import BuilderIncompleteError, InjectedObservations
from wow_bot.perception.capture import FrameLike
from wow_bot.perception.pose import PoseReading, WorldPoseReader
from wow_bot.perception.proximity import (
    TargetDistanceEstimator,
    TargetDistanceReading,
)
from wow_bot.perception.reaction import TargetReactionReader, TargetReactionReading
from wow_bot.perception.target import TargetReading

__all__ = [
    "ChannelObservations",
    "observe_channels",
]

#: Reading used when a channel is not wired: "not attempted", confidence 0.
_UNOBSERVED_REACTION = TargetReactionReading(reaction=None, confidence=0.0)
_UNOBSERVED_DISTANCE = TargetDistanceReading(
    distance_estimate=None, bar_width_px=None, confidence=0.0
)


@dataclass(frozen=True)
class ChannelObservations:
    """What the T-FIX-30 channels observed on one frame."""

    pose: PoseReading
    reaction: TargetReactionReading = _UNOBSERVED_REACTION
    distance: TargetDistanceReading = _UNOBSERVED_DISTANCE

    @property
    def position(self) -> tuple[float, float] | None:
        """The observed world pose, or ``None`` when it was not confident."""
        return self.pose.position

    @property
    def target_reaction(self) -> str | None:
        """The observed reaction label, or ``None`` when it was not emitted."""
        return self.reaction.reaction

    @property
    def target_distance_estimate(self) -> float | None:
        """The observed distance in yards, or ``None`` when it was not emitted."""
        return self.distance.distance_estimate

    def confidence(self) -> dict[str, float]:
        """Measured scores for the values this frame actually emitted."""
        scores: dict[str, float] = {}
        if self.pose.position is not None:
            scores["position"] = float(self.pose.confidence)
        if self.reaction.reaction is not None:
            scores["target.reaction"] = float(self.reaction.confidence)
        if self.distance.distance_estimate is not None:
            scores["target.distance_estimate"] = float(self.distance.confidence)
        return scores

    def to_injected(self, *, player_z: float | None = None) -> InjectedObservations:
        """Project into the builder's injection shape, refusing a guessed pose.

        ``player_z`` defaults to ``None`` and this channel never produces one
        (ADR-003 Decision 2). A caller that has a *validated* height channel
        from elsewhere may pass it explicitly; injecting a guessed elevation
        is the caller's error, not a value this module will invent.

        Raises :class:`BuilderIncompleteError` when no confident pose was
        observed. That is the honest failure: ``GameState.position`` is
        non-Optional, and a fabricated pair would create a wrong world node
        that the navigator would later plan through.
        """
        if self.position is None:
            raise BuilderIncompleteError(
                "position is a non-Optional GameState field and the world-pose "
                "channel observed no confident coordinate this frame; refusing "
                "to emit a fabricated pose that would write a wrong world node"
            )
        return InjectedObservations(
            position=self.position,
            # The addon coordinate frame has no height channel; None is the
            # permanent LAB_MODE value (ADR-003 Decision 2) unless a validated
            # height channel supplies one from outside.
            player_z=player_z,
            # target_x/target_y need pose + bearing + distance, an inference
            # with no validated error model (ADR-003 Decision 5).
            target_x=None,
            target_y=None,
            target_reaction=self.target_reaction,
            target_distance_estimate=self.target_distance_estimate,
            confidence=self.confidence(),
        )


def observe_channels(
    frame: FrameLike,
    *,
    pose: WorldPoseReader,
    reaction: TargetReactionReader | None = None,
    distance: TargetDistanceEstimator | None = None,
    target_reading: TargetReading | None = None,
    now: float | None = None,
    tesseract_cmd: str | None = None,
) -> ChannelObservations:
    """Run the wired T-FIX-30 channels over one frame.

    ``target_reading`` is the reading the caller already obtained from
    :class:`~wow_bot.perception.target.TargetReader`; passing the same reading
    the builder will use keeps one frame to one target read (the readers are
    throttled, so a second read would be served from cache anyway, but this
    keeps that explicit). Without it — or without a distance estimator — the
    distance channel reports "not attempted".
    """
    pose_reading = pose.read(frame, now=now, tesseract_cmd=tesseract_cmd)
    reaction_reading = (
        _UNOBSERVED_REACTION if reaction is None else reaction.read(frame, now=now)
    )
    distance_reading = _UNOBSERVED_DISTANCE
    if distance is not None and target_reading is not None:
        distance_reading = distance.estimate(target_reading)
    return ChannelObservations(
        pose=pose_reading,
        reaction=reaction_reading,
        distance=distance_reading,
    )
