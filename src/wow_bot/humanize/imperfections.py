"""Imperfection layer for humanized input timing and precision.

Provides stochastic micro-pauses and miss-clicks that break machine monotony.
This module is pure, deterministic, and stateless with respect to RNG.
No I/O, no wall-clock time reads, and no global random state mutations.
"""

import math
import random
from collections import deque
from dataclasses import dataclass
from typing import Any


class ImperfectionError(Exception):
    """Raised when an imperfection configuration, decision, or log operation fails."""


@dataclass(frozen=True)
class ImperfectionConfig:
    """Configuration parameters for imperfection sampling."""

    pause_probability: float = 0.05
    pause_duration_mu: float = -0.7
    pause_duration_sigma: float = 0.5
    pause_duration_clip_low: float = 0.1
    pause_duration_clip_high: float = 3.0
    miss_click_probability: float = 0.02
    miss_click_max_offset_units: float = 8.0
    miss_click_sigma_fraction: float = 0.5
    max_pause_redraws: int = 1000

    def __post_init__(self) -> None:
        float_fields = {
            "pause_probability": self.pause_probability,
            "pause_duration_mu": self.pause_duration_mu,
            "pause_duration_sigma": self.pause_duration_sigma,
            "pause_duration_clip_low": self.pause_duration_clip_low,
            "pause_duration_clip_high": self.pause_duration_clip_high,
            "miss_click_probability": self.miss_click_probability,
            "miss_click_max_offset_units": self.miss_click_max_offset_units,
            "miss_click_sigma_fraction": self.miss_click_sigma_fraction,
        }

        for field_name, val in float_fields.items():
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
            ):
                raise ValueError(f"{field_name} must be a finite float, got {val}")

        if isinstance(self.max_pause_redraws, bool) or not isinstance(
            self.max_pause_redraws, int
        ):
            raise ValueError(
                f"max_pause_redraws must be an integer, got {self.max_pause_redraws}"
            )

        if not (0.0 <= self.pause_probability <= 1.0):
            raise ValueError(
                f"pause_probability must be in [0.0, 1.0], got {self.pause_probability}"
            )
        if self.pause_duration_sigma <= 0.0:
            raise ValueError(
                f"pause_duration_sigma must be > 0.0, got {self.pause_duration_sigma}"
            )
        if self.pause_duration_clip_low <= 0.0:
            raise ValueError(
                f"pause_duration_clip_low must be > 0.0, got {self.pause_duration_clip_low}"
            )
        if self.pause_duration_clip_high <= self.pause_duration_clip_low:
            raise ValueError(
                f"pause_duration_clip_high ({self.pause_duration_clip_high}) must be > "
                f"pause_duration_clip_low ({self.pause_duration_clip_low})"
            )
        if not (0.0 <= self.miss_click_probability <= 1.0):
            raise ValueError(
                f"miss_click_probability must be in [0.0, 1.0], got {self.miss_click_probability}"
            )
        if self.miss_click_max_offset_units <= 0.0:
            raise ValueError(
                f"miss_click_max_offset_units must be > 0.0, got {self.miss_click_max_offset_units}"
            )
        if not (0.0 < self.miss_click_sigma_fraction <= 1.0):
            raise ValueError(
                f"miss_click_sigma_fraction must be in (0.0, 1.0], got {self.miss_click_sigma_fraction}"
            )
        if self.max_pause_redraws < 1:
            raise ValueError(
                f"max_pause_redraws must be >= 1, got {self.max_pause_redraws}"
            )


@dataclass(frozen=True)
class PauseDecision:
    """Decision outcome for micro-pause sampling."""

    occurred: bool
    duration_s: float
    reason: str = ""
    clip_low: float = 0.1
    clip_high: float = 3.0

    def __post_init__(self) -> None:
        if not isinstance(self.occurred, bool):
            raise ValueError("occurred must be a boolean")
        if (
            not isinstance(self.duration_s, (int, float))
            or isinstance(self.duration_s, bool)
            or not math.isfinite(self.duration_s)
        ):
            raise ValueError("duration_s must be a finite float")
        if not isinstance(self.reason, str) or len(self.reason.strip()) == 0:
            raise ValueError("reason must be a non-empty string")

        if not self.occurred:
            if self.duration_s != 0.0:
                raise ValueError("occurred=False implies duration_s == 0.0")
        else:
            if self.duration_s <= 0.0:
                raise ValueError("occurred=True implies duration_s > 0.0")
            if not (self.clip_low <= self.duration_s <= self.clip_high):
                raise ValueError(
                    f"duration_s ({self.duration_s}) is outside range [{self.clip_low}, {self.clip_high}]"
                )


@dataclass(frozen=True)
class MissClickDecision:
    """Decision outcome for miss-click sampling."""

    occurred: bool
    offset_x: float
    offset_y: float
    magnitude: float
    reason: str = ""
    max_offset_units: float = 8.0

    def __post_init__(self) -> None:
        if not isinstance(self.occurred, bool):
            raise ValueError("occurred must be a boolean")
        for field_name, val in (
            ("offset_x", self.offset_x),
            ("offset_y", self.offset_y),
            ("magnitude", self.magnitude),
            ("max_offset_units", self.max_offset_units),
        ):
            if (
                not isinstance(val, (int, float))
                or isinstance(val, bool)
                or not math.isfinite(val)
            ):
                raise ValueError(f"{field_name} must be a finite float")

        if not isinstance(self.reason, str) or len(self.reason.strip()) == 0:
            raise ValueError("reason must be a non-empty string")

        if not self.occurred:
            if self.offset_x != 0.0 or self.offset_y != 0.0 or self.magnitude != 0.0:
                raise ValueError(
                    "occurred=False implies offset_x == 0.0, offset_y == 0.0, and magnitude == 0.0"
                )
        else:
            if self.magnitude <= 0.0:
                raise ValueError("occurred=True implies magnitude > 0.0")
            if self.magnitude > self.max_offset_units:
                raise ValueError(
                    f"magnitude ({self.magnitude}) exceeds max_offset_units ({self.max_offset_units})"
                )
            calc_mag = math.sqrt(self.offset_x**2 + self.offset_y**2)
            if abs(self.magnitude - calc_mag) > 1e-6:
                raise ValueError(
                    f"magnitude ({self.magnitude}) inconsistent with offsets ({self.offset_x}, {self.offset_y}) -> sqrt={calc_mag}"
                )


class ImperfectionSampler:
    """Stateless sampler generating stochastic micro-pauses and miss-clicks.

    All random sampling uses the explicit `rng` passed to methods.
    The sampler holds no random state or clock reads.
    """

    def __init__(self, config: ImperfectionConfig | None = None) -> None:
        self._config = config if config is not None else ImperfectionConfig()

    def maybe_pause(self, rng: random.Random) -> PauseDecision:
        """Sample whether a micro-pause occurs and its duration.

        Fallback behavior: if lognormal sampling fails to yield a value within
        [pause_duration_clip_low, pause_duration_clip_high] after max_pause_redraws,
        the last drawn value is clamped to [clip_low, clip_high].
        """
        cfg = self._config

        if rng.random() >= cfg.pause_probability:
            return PauseDecision(
                occurred=False,
                duration_s=0.0,
                reason="not_selected",
                clip_low=cfg.pause_duration_clip_low,
                clip_high=cfg.pause_duration_clip_high,
            )

        raw = math.exp(rng.gauss(cfg.pause_duration_mu, cfg.pause_duration_sigma))
        if not (cfg.pause_duration_clip_low <= raw <= cfg.pause_duration_clip_high):
            attempts = 0
            while (
                not (cfg.pause_duration_clip_low <= raw <= cfg.pause_duration_clip_high)
                and attempts < cfg.max_pause_redraws
            ):
                raw = math.exp(
                    rng.gauss(cfg.pause_duration_mu, cfg.pause_duration_sigma)
                )
                attempts += 1

            if not (cfg.pause_duration_clip_low <= raw <= cfg.pause_duration_clip_high):
                raw = max(
                    cfg.pause_duration_clip_low,
                    min(cfg.pause_duration_clip_high, raw),
                )

        return PauseDecision(
            occurred=True,
            duration_s=raw,
            reason="pause",
            clip_low=cfg.pause_duration_clip_low,
            clip_high=cfg.pause_duration_clip_high,
        )

    def maybe_miss_click(self, rng: random.Random) -> MissClickDecision:
        """Sample whether a miss-click occurs and its offset coordinates.

        Fallback behavior: if magnitude sampling yields exactly 0.0, redrawing
        is attempted up to max_pause_redraws times. On exhaustion, magnitude is set
        to epsilon derived from sigma (sigma * 1e-3).
        """
        cfg = self._config

        if rng.random() >= cfg.miss_click_probability:
            return MissClickDecision(
                occurred=False,
                offset_x=0.0,
                offset_y=0.0,
                magnitude=0.0,
                reason="not_selected",
                max_offset_units=cfg.miss_click_max_offset_units,
            )

        theta = rng.random() * 2.0 * math.pi
        sigma = cfg.miss_click_max_offset_units * cfg.miss_click_sigma_fraction
        raw = abs(rng.gauss(0.0, sigma))
        magnitude = min(raw, cfg.miss_click_max_offset_units)

        if magnitude == 0.0:
            attempts = 0
            while magnitude == 0.0 and attempts < cfg.max_pause_redraws:
                raw = abs(rng.gauss(0.0, sigma))
                magnitude = min(raw, cfg.miss_click_max_offset_units)
                attempts += 1
            if magnitude == 0.0:
                magnitude = sigma * 1e-3

        offset_x = magnitude * math.cos(theta)
        offset_y = magnitude * math.sin(theta)

        return MissClickDecision(
            occurred=True,
            offset_x=offset_x,
            offset_y=offset_y,
            magnitude=magnitude,
            reason="miss_click",
            max_offset_units=cfg.miss_click_max_offset_units,
        )

    def describe_config(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the configuration."""
        cfg = self._config
        return {
            "pause_probability": cfg.pause_probability,
            "pause_duration_mu": cfg.pause_duration_mu,
            "pause_duration_sigma": cfg.pause_duration_sigma,
            "pause_duration_clip_low": cfg.pause_duration_clip_low,
            "pause_duration_clip_high": cfg.pause_duration_clip_high,
            "miss_click_probability": cfg.miss_click_probability,
            "miss_click_max_offset_units": cfg.miss_click_max_offset_units,
            "miss_click_sigma_fraction": cfg.miss_click_sigma_fraction,
            "max_pause_redraws": cfg.max_pause_redraws,
        }


class ImperfectionLog:
    """Bounded in-memory recorder of ImperfectionEvent entries.

    Note: ImperfectionLog is explicitly NOT thread-safe by design. It is intended
    to be used strictly from a single-threaded actuation path.
    """

    def __init__(self, *, max_entries: int = 10_000) -> None:
        if isinstance(max_entries, bool) or max_entries < 1:
            raise ImperfectionError("max_entries must be an integer >= 1")
        self._max_entries: int = max_entries
        self._buffer: deque[dict[str, Any]] = deque()

    def record_pause(self, decision: PauseDecision) -> None:
        """Record a pause decision if it occurred."""
        if not decision.occurred:
            return
        if len(self._buffer) >= self._max_entries:
            self._buffer.popleft()
        self._buffer.append({"kind": "pause", "duration_s": decision.duration_s})

    def record_miss_click(self, decision: MissClickDecision) -> None:
        """Record a miss-click decision if it occurred."""
        if not decision.occurred:
            return
        if len(self._buffer) >= self._max_entries:
            self._buffer.popleft()
        self._buffer.append(
            {
                "kind": "miss_click",
                "offset_x": decision.offset_x,
                "offset_y": decision.offset_y,
                "magnitude": decision.magnitude,
            }
        )

    def entries(self) -> tuple[dict[str, Any], ...]:
        """Return a snapshot copy of stored entries in insertion order."""
        return tuple(dict(entry) for entry in self._buffer)

    def clear(self) -> None:
        """Clear all stored entries."""
        self._buffer.clear()

    @property
    def size(self) -> int:
        """Number of stored entries."""
        return len(self._buffer)
