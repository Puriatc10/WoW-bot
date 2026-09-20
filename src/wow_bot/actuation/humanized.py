"""Humanized Actuator wrapper injecting human timing, micro-pauses, and imperfection modeling."""

import math
import random
from dataclasses import dataclass, field
from types import TracebackType
from typing import Any, Self

from wow_bot.actuation.actuator import Actuator, NullActuator, RealActuator
from wow_bot.actuation.mapper import ActionResult, DelayProvider, Intent, MoveTo, RealDelay, Turn
from wow_bot.humanize.cursor import CursorConfig, TrajectoryPoint
from wow_bot.humanize.imperfections import ImperfectionConfig, ImperfectionLog, ImperfectionSampler
from wow_bot.humanize.intervals import IntervalConfig, sample_interval
from wow_bot.session import Session


class HumanizedActuatorError(Exception):
    """Raised when humanized actuation configuration or execution fails."""


@dataclass(frozen=True)
class HumanizerConfig:
    """Configuration for HumanizedActuator timing, micro-pauses, and cursor hooks."""

    enabled: bool = True
    interval_config: IntervalConfig = field(default_factory=IntervalConfig)
    imperfection_config: ImperfectionConfig = field(default_factory=ImperfectionConfig)
    cursor_config: CursorConfig = field(default_factory=CursorConfig)
    cursor_enabled: bool = False
    min_delay_s: float = 0.0
    log_to_session: bool = True

    def __post_init__(self) -> None:
        if (
            not isinstance(self.min_delay_s, (int, float))
            or isinstance(self.min_delay_s, bool)
            or not math.isfinite(self.min_delay_s)
            or self.min_delay_s < 0.0
        ):
            raise ValueError(f"min_delay_s must be a finite float >= 0.0, got {self.min_delay_s}")

        if self.cursor_enabled:
            raise ValueError(
                "cursor_enabled=True is not available in this phase because cursor integration "
                "is not supported by the action mapper."
            )


class HumanizedActuator:
    """Humanized Actuator wrapper around RealActuator implementing the Actuator protocol."""

    def __init__(
        self,
        wrapped: Actuator,
        *,
        config: HumanizerConfig | None = None,
        delay: DelayProvider | None = None,
        session: Session | None = None,
        rng: random.Random | None = None,
    ) -> None:
        self._wrapped = wrapped
        self._config = config if config is not None else HumanizerConfig()
        self._delay = delay if delay is not None else RealDelay()
        self._session = session
        self._rng = rng if rng is not None else random.Random(0)

        self._sampler = ImperfectionSampler(self._config.imperfection_config)
        self._imperfection_log = ImperfectionLog()
        self._aborted = False

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        """Execute a single action intent with injected inter-action delays and micro-pauses."""
        if not self._config.enabled:
            return self._wrapped.execute(intent, position=position)

        interval_s = sample_interval(self._rng, self._config.interval_config)
        interval_s = max(interval_s, self._config.min_delay_s)

        pause = self._sampler.maybe_pause(self._rng)

        self._delay.wait(interval_s)
        if pause.occurred:
            self._delay.wait(pause.duration_s)

        trajectory = self._apply_cursor_trajectory_if_applicable(intent, self._rng)
        if trajectory is not None:
            raise HumanizedActuatorError(
                "Cursor trajectory returned non-None value, but no consumer is wired up yet."
            )

        result = self._wrapped.execute(intent, position=position)

        if self._config.log_to_session and self._session is not None:
            self._session.write_event({
                "event": "humanizer_action",
                "intent": repr(intent),
                "interval_s": interval_s,
                "pause_occurred": pause.occurred,
                "pause_duration_s": pause.duration_s,
                "status": result.status.value,
                "latency_ms": result.latency_ms,
            })

        if pause.occurred:
            self._imperfection_log.record_pause(pause)

        return result

    def abort(self, reason: str) -> None:
        """Abort actuation idempotently without delaying execution."""
        if self._aborted:
            return

        self._aborted = True
        self._wrapped.abort(reason)

        if self._config.log_to_session and self._session is not None:
            self._session.write_event({
                "event": "humanizer_abort",
                "reason": reason,
            })

    def is_aborted(self) -> bool:
        """Return True if either wrapper or wrapped actuator is aborted."""
        return self._aborted or self._wrapped.is_aborted()

    def close(self) -> None:
        """Close actuator idempotently, aborting if not already aborted."""
        if not self._aborted:
            self.abort("close")

    def last_imperfection_entries(self) -> tuple[dict[str, Any], ...]:
        """Return stored imperfection log entries."""
        return self._imperfection_log.entries()

    @property
    def imperfection_log_size(self) -> int:
        """Return size of in-memory imperfection log."""
        return self._imperfection_log.size

    def describe_humanizer(self) -> dict[str, Any]:
        """Return JSON-serializable representation of humanizer configuration and RNG state summary."""
        cfg = self._config
        return {
            "enabled": cfg.enabled,
            "min_delay_s": cfg.min_delay_s,
            "log_to_session": cfg.log_to_session,
            "cursor_enabled": cfg.cursor_enabled,
            "interval_config": {
                "mu": cfg.interval_config.mu,
                "sigma": cfg.interval_config.sigma,
                "clip_low": cfg.interval_config.clip_low,
                "clip_high": cfg.interval_config.clip_high,
                "max_rejection_attempts": cfg.interval_config.max_rejection_attempts,
            },
            "imperfection_config": self._sampler.describe_config(),
            "cursor_config": {
                "control_point_jitter_fraction": cfg.cursor_config.control_point_jitter_fraction,
                "overshoot_fraction": cfg.cursor_config.overshoot_fraction,
                "overshoot_probability": cfg.cursor_config.overshoot_probability,
                "duration_mu": cfg.cursor_config.duration_mu,
                "duration_sigma": cfg.cursor_config.duration_sigma,
                "duration_clip_low": cfg.cursor_config.duration_clip_low,
                "duration_clip_high": cfg.cursor_config.duration_clip_high,
                "points_per_100ms": cfg.cursor_config.points_per_100ms,
                "min_points": cfg.cursor_config.min_points,
                "max_points": cfg.cursor_config.max_points,
                "min_distance_units": cfg.cursor_config.min_distance_units,
            },
            "rng_initialized": True,
        }

    def _apply_cursor_trajectory_if_applicable(
        self,
        intent: Intent,
        rng: random.Random,
    ) -> list[TrajectoryPoint] | None:
        """Placeholder for future cursor trajectory generation.

        Returns None for all current intent types (MoveTo, Turn).
        When mouse-based intents are added to the action mapper in a future phase,
        this method will invoke generate_trajectory from wow_bot.humanize.cursor.
        """
        if isinstance(intent, (MoveTo, Turn)):
            return None
        return None

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()


def make_humanized_actuator(
    *,
    mode: str,
    wrapped: Actuator | None = None,
    config: HumanizerConfig | None = None,
    delay: DelayProvider | None = None,
    session: Session | None = None,
    rng: random.Random | None = None,
) -> Actuator:
    """Factory function creating an Actuator instance based on execution mode."""
    if mode == "MOCK":
        return NullActuator()
    elif mode == "LAB":
        if wrapped is None or not isinstance(wrapped, RealActuator):
            raise HumanizedActuatorError(
                "LAB mode requires a valid RealActuator instance for 'wrapped'"
            )
        if delay is None:
            raise HumanizedActuatorError(
                "LAB mode requires a DelayProvider instance for 'delay'"
            )
        return HumanizedActuator(
            wrapped,
            config=config,
            delay=delay,
            session=session,
            rng=rng,
        )
    else:
        raise HumanizedActuatorError(f"Invalid mode: '{mode}'")
