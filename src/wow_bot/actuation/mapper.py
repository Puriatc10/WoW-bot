"""Symbolic intent to low-level driver call mapper."""

import math
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Protocol, runtime_checkable

from wow_bot.actuation.driver import InputDriver


@dataclass(frozen=True)
class Keymap:
    """Configuration mapping movement directions and gameplay actions to input key strings."""

    forward: str = "w"
    back: str = "s"
    left: str = "a"
    right: str = "d"
    turn_left: str = "q"
    turn_right: str = "e"
    jump: str = "space"
    interact: str = "f"
    loot: str = "f"
    default_cast_key: str = "1"
    spell_keys: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class MoveTo:
    """Symbolic movement target in 2D coordinates."""

    x: float
    y: float


@dataclass(frozen=True)
class Turn:
    """Symbolic camera/character turning intent in radians."""

    angle_rad: float


@dataclass(frozen=True)
class Cast:
    """Symbolic spell casting intent carrying spell_id and optional target_id."""

    spell_id: str
    target_id: str | None = None


@dataclass(frozen=True)
class Loot:
    """Symbolic corpse looting intent carrying optional target_id."""

    target_id: str | None = None


@dataclass(frozen=True)
class VendorInteract:
    """Symbolic vendor interaction intent carrying vendor_entity and action."""

    vendor_entity: str | int | None = None
    action: str = "interact"


@dataclass(frozen=True)
class Jump:
    """Symbolic character jump intent carrying jump direction."""

    direction: str = "forward"


Intent = MoveTo | Turn | Cast | Loot | VendorInteract | Jump


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
        action_step_duration_s: float = 0.05,
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
        if action_step_duration_s < 0:
            raise ValueError("action_step_duration_s must be non-negative")
        if assumed_speed_units_per_s <= 0:
            raise ValueError("assumed_speed_units_per_s must be positive")
        if assumed_turn_rate_rad_per_s <= 0:
            raise ValueError("assumed_turn_rate_rad_per_s must be positive")

        self._driver = driver
        self._keymap = keymap
        self._arrival_tolerance = arrival_tolerance
        self._move_step_duration_s = move_step_duration_s
        self._turn_step_duration_s = turn_step_duration_s
        self._action_step_duration_s = action_step_duration_s
        self._assumed_speed_units_per_s = assumed_speed_units_per_s
        self._assumed_turn_rate_rad_per_s = assumed_turn_rate_rad_per_s
        self._delay: DelayProvider = delay if delay is not None else RealDelay()

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
        heading: float | None = None,
    ) -> ActionResult:
        """Execute a single step for the given symbolic intent from the current position and heading."""
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

            if heading is not None:
                f_comp = dx * math.cos(heading) + dy * math.sin(heading)
                r_comp = dx * math.sin(heading) - dy * math.cos(heading)
            else:
                f_comp = dy
                r_comp = dx

            keys_to_press: list[str] = []
            eps = 1e-4
            if f_comp > eps:
                keys_to_press.append(self._keymap.forward)
            elif f_comp < -eps:
                keys_to_press.append(self._keymap.back)

            if r_comp > eps:
                keys_to_press.append(self._keymap.right)
            elif r_comp < -eps:
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

        elif isinstance(intent, Cast):
            key = self._keymap.spell_keys.get(intent.spell_id, self._keymap.default_cast_key)
            step_s = self._action_step_duration_s
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
                    k = pressed_keys.pop()
                    try:
                        self._driver.key_up(k)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"cast:{intent.spell_id}",
            )

        elif isinstance(intent, Loot):
            key = self._keymap.loot
            step_s = self._action_step_duration_s
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
                    k = pressed_keys.pop()
                    try:
                        self._driver.key_up(k)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            target_str = intent.target_id or ""
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"loot:{target_str}",
            )

        elif isinstance(intent, VendorInteract):
            key = self._keymap.interact
            step_s = self._action_step_duration_s
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
                    k = pressed_keys.pop()
                    try:
                        self._driver.key_up(k)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"vendor:{intent.action}:{intent.vendor_entity}",
            )

        elif isinstance(intent, Jump):
            keys = [self._keymap.jump]
            if intent.direction == "forward":
                keys.insert(0, self._keymap.forward)
            elif intent.direction == "back":
                keys.insert(0, self._keymap.back)
            elif intent.direction == "left":
                keys.insert(0, self._keymap.left)
            elif intent.direction == "right":
                keys.insert(0, self._keymap.right)
            step_s = self._action_step_duration_s
            pressed_keys = []
            exc_in_try = False
            try:
                for k in keys:
                    self._driver.key_down(k)
                    pressed_keys.append(k)
                self._delay.wait(step_s)
            except Exception:
                exc_in_try = True
                raise
            finally:
                while pressed_keys:
                    k = pressed_keys.pop()
                    try:
                        self._driver.key_up(k)
                    except Exception:
                        if not exc_in_try:
                            raise

            latency_ms = (self._driver.now() - start_time) * 1000.0
            return ActionResult(
                status=ActionStatus.SUCCESS,
                latency_ms=latency_ms,
                notes=f"jump:{intent.direction}",
            )

        else:
            raise TypeError(f"Unsupported intent type: {type(intent)}")
