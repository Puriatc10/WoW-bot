"""Symbolic intent to low-level driver call mapper."""

import math
import time
from dataclasses import dataclass
from enum import Enum
from typing import Protocol, runtime_checkable

from wow_bot.actuation.driver import InputDriver


@dataclass(frozen=True)
class Keymap:
    """Configuration mapping movement directions to input key strings."""

    forward: str = "w"
    back: str = "s"
    left: str = "a"
    right: str = "d"
    turn_left: str = "q"
    turn_right: str = "e"


@dataclass(frozen=True)
class MoveTo:
    """Symbolic movement target in 2D coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class Turn:
    """Symbolic camera/character turning intent in radians."""

    angle_rad: float


Intent = MoveTo | Turn


class ActionStatus(str, Enum):
    """Result status of an executed action step."""

    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass(frozen=True)
class ActionResult:
    """Execution result of an action step."""

    status: ActionStatus
    latency_ms: float
    notes: str = ""


@runtime_checkable
class DelayProvider(Protocol):
    """Protocol for delaying execution between input events."""

    def wait(self, seconds: float) -> None:
        """Wait for the specified duration in seconds."""
        ...


class RealDelay:
    """Delay provider that uses real time sleeping."""

    def wait(self, seconds: float) -> None:
        """Sleep for seconds if > 0."""
        if seconds > 0:
            time.sleep(seconds)


class NullDelay:
    """Delay provider that records wait durations without sleeping."""

    def __init__(self) -> None:
        self._waits: list[float] = []

    def wait(self, seconds: float) -> None:
        """Record wait duration."""
        self._waits.append(seconds)

    def waits(self) -> list[float]:
        """Return a copy of all recorded wait durations."""
        return list(self._waits)


class ActionMapper:
    """Maps symbolic intents (MoveTo, Turn) into low-level driver key presses."""

    def __init__(
        self,
        driver: InputDriver,
        *,
        keymap: Keymap = Keymap(),  # noqa: B008
        arrival_tolerance: float = 0.5,
        move_step_duration_s: float = 0.15,
        turn_step_duration_s: float = 0.10,
        assumed_speed_units_per_s: float = 5.0,
        assumed_turn_rate_rad_per_s: float = 3.14,
        delay: DelayProvider | None = None,
    ) -> None:
        if arrival_tolerance <= 0:
            raise ValueError("arrival_tolerance must be positive")
        if move_step_duration_s < 0:
            raise ValueError("move_step_duration_s must be non-negative")
        if turn_step_duration_s < 0:
            raise ValueError("turn_step_duration_s must be non-negative")
        if assumed_speed_units_per_s <= 0:
            raise ValueError("assumed_speed_units_per_s must be positive")
        if assumed_turn_rate_rad_per_s <= 0:
            raise ValueError("assumed_turn_rate_rad_per_s must be positive")

        self._driver = driver
        self._keymap = keymap
        self._arrival_tolerance = arrival_tolerance
        self._move_step_duration_s = move_step_duration_s
        self._turn_step_duration_s = turn_step_duration_s
        self._assumed_speed_units_per_s = assumed_speed_units_per_s
        self._assumed_turn_rate_rad_per_s = assumed_turn_rate_rad_per_s
        self._delay: DelayProvider = delay if delay is not None else RealDelay()

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        """Execute a single step for the given symbolic intent from the current position."""
        start_time = self._driver.now()

        if isinstance(intent, MoveTo):
            dx = intent.x - position[0]
            dy = intent.y - position[1]
            distance = math.sqrt(dx * dx + dy * dy)

            if distance <= self._arrival_tolerance:
                latency_ms = (self._driver.now() - start_time) * 1000.0
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    latency_ms=latency_ms,
                    notes="arrived",
                )

            keys_to_press: list[str] = []
            if dy > 0:
                keys_to_press.append(self._keymap.forward)
            elif dy < 0:
                keys_to_press.append(self._keymap.back)

            if dx > 0:
                keys_to_press.append(self._keymap.right)
            elif dx < 0:
                keys_to_press.append(self._keymap.left)

            step_s = min(
                self._move_step_duration_s,
                distance / self._assumed_speed_units_per_s,
            )

            pressed_keys: list[str] = []
            exc_in_try = False
            try:
                for key in keys_to_press:
                    self._driver.key_down(key)
                    pressed_keys.append(key)
                self._delay.wait(step_s)
            except Exception:
                exc_in_try = True
                raise
            finally:
                while pressed_keys:
                    key = pressed_keys.pop()
                    try:
                        self._driver.key_up(key)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"step={step_s:.3f}",
            )

        elif isinstance(intent, Turn):
            angle_rad = intent.angle_rad
            if abs(angle_rad) <= 1e-6:
                latency_ms = (self._driver.now() - start_time) * 1000.0
                return ActionResult(
                    status=ActionStatus.SUCCESS,
                    latency_ms=latency_ms,
                    notes="noop",
                )

            key = self._keymap.turn_left if angle_rad > 0 else self._keymap.turn_right
            step_s = min(
                self._turn_step_duration_s,
                abs(angle_rad) / self._assumed_turn_rate_rad_per_s,
            )

            pressed_keys = []
            exc_in_try = False
            try:
                self._driver.key_down(key)
                pressed_keys.append(key)
                self._delay.wait(step_s)
            except Exception:
                exc_in_try = True
                raise
            finally:
                while pressed_keys:
                    key = pressed_keys.pop()
                    try:
                        self._driver.key_up(key)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"step={step_s:.3f}",
            )

        else:
            raise TypeError(f"Unsupported intent type: {type(intent)}")
