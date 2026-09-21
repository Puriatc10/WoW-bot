"""Vendor interaction controller and spatial node resolver for WoW-bot farm subsystem.

Provides vendor location resolution from the World Model and synchronous vendor flow
execution (navigation, selling junk, repairing gear).

Design notes:
- Phase 11 Stub Convention:
  The vendor interaction itself (opening the vendor window, clicking the sell button)
  has no dedicated mapper primitive yet. Following the conventions from T6.3 ("stand and cast"
  MoveTo) and T11.2 ("stand and loot" MoveTo), vendor actions are expressed as zero-distance MoveTo
  intents at the vendor's position, executed AFTER a successful navigation. This is deterministic
  for the actuator and will be replaced when the mapper gains a Sell or Repair primitive.
- Sell Chunk Size Note:
  In this phase, `sell_chunk_size` defaults to 1, so the sell loop performs exactly one sell step.
  A future phase with a real Sell primitive will raise the chunk size and use the full
  `max_sell_steps` bound.
- Dual Entry Point Architecture:
  1. `resolve_vendor_node(...)`: Async function querying WorldModel store for nearest vendor.
  2. `VendorController.run(...)`: Sync method executing sale/repair navigation sequence.
"""

from __future__ import annotations

import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from wow_bot.actuation.mapper import ActionResult, ActionStatus, Intent, MoveTo
from wow_bot.nav.navigator import Navigator, NavStatus

if TYPE_CHECKING:
    from wow_bot.session import Session
    from wow_bot.world.store import WorldModel


class VendorError(Exception):
    """Exception raised for vendor configuration, resolution, or execution failures."""


@runtime_checkable
class VendorStateView(Protocol):
    """Protocol defining state fields required for vendor interaction evaluation."""

    self_x: float
    self_y: float
    inventory_count: int
    durability_fraction: float | None


@runtime_checkable
class Actuator(Protocol):
    """Protocol for executing symbolic intents at position."""

    def execute(self, intent: Intent, *, position: tuple[float, float]) -> ActionResult:
        """Execute symbolic intent from current position."""
        ...


@dataclass(frozen=True)
class VendorLocation:
    """Resolved spatial vendor or trainer location."""

    node_id: int
    x: float
    y: float
    kind: str
    name: str

    def __post_init__(self) -> None:
        """Validate location attributes."""
        if isinstance(self.node_id, bool) or not isinstance(self.node_id, int) or self.node_id < 1:
            raise ValueError(f"node_id must be an integer >= 1, got {self.node_id}")
        if (
            isinstance(self.x, bool)
            or not isinstance(self.x, (int, float))
            or not math.isfinite(float(self.x))
        ):
            raise ValueError(f"x must be a finite float, got {self.x}")
        if (
            isinstance(self.y, bool)
            or not isinstance(self.y, (int, float))
            or not math.isfinite(float(self.y))
        ):
            raise ValueError(f"y must be a finite float, got {self.y}")
        if self.kind not in {"vendor", "trainer"}:
            raise ValueError(f"kind must be 'vendor' or 'trainer', got {self.kind!r}")
        if not isinstance(self.name, str):
            raise TypeError(f"name must be a string, got {type(self.name).__name__}")


class VendorStatus(str, Enum):
    """Result status of a vendor controller interaction run."""

    SUCCESS = "success"
    NAVIGATION_FAILED = "navigation_failed"
    NAVIGATION_HARD_FAILURE = "navigation_hard_failure"
    NAVIGATION_TIMEOUT = "navigation_timeout"
    SELL_FAILED = "sell_failed"
    REPAIR_FAILED = "repair_failed"
    NO_VENDOR_RESOLVED = "no_vendor_resolved"
    SKIPPED_NO_ACTION = "skipped_no_action"


@dataclass(frozen=True)
class VendorConfig:
    """Configuration options for vendor step bounds and thresholds."""

    max_sell_steps: int = 20
    max_repair_steps: int = 5
    vendor_reach_units: float = 3.0
    sell_chunk_size: int = 1
    repair_durability_threshold: float = 0.5

    def __post_init__(self) -> None:
        """Validate configuration bounds."""
        if (
            isinstance(self.max_sell_steps, bool)
            or not isinstance(self.max_sell_steps, int)
            or self.max_sell_steps < 1
        ):
            raise ValueError(f"max_sell_steps must be >= 1, got {self.max_sell_steps}")
        if (
            isinstance(self.max_repair_steps, bool)
            or not isinstance(self.max_repair_steps, int)
            or self.max_repair_steps < 0
        ):
            raise ValueError(f"max_repair_steps must be >= 0, got {self.max_repair_steps}")
        if (
            isinstance(self.vendor_reach_units, bool)
            or not isinstance(self.vendor_reach_units, (int, float))
            or not math.isfinite(float(self.vendor_reach_units))
            or float(self.vendor_reach_units) <= 0.0
        ):
            raise ValueError(f"vendor_reach_units must be > 0.0, got {self.vendor_reach_units}")
        if (
            isinstance(self.sell_chunk_size, bool)
            or not isinstance(self.sell_chunk_size, int)
            or self.sell_chunk_size < 1
        ):
            raise ValueError(f"sell_chunk_size must be >= 1, got {self.sell_chunk_size}")
        if (
            isinstance(self.repair_durability_threshold, bool)
            or not isinstance(self.repair_durability_threshold, (int, float))
            or not math.isfinite(float(self.repair_durability_threshold))
            or not (0.0 <= float(self.repair_durability_threshold) <= 1.0)
        ):
            raise ValueError(
                f"repair_durability_threshold must be in [0.0, 1.0], got {self.repair_durability_threshold}"
            )


@dataclass(frozen=True)
class VendorResult:
    """Result snapshot of a vendor interaction run."""

    status: VendorStatus
    vendor: VendorLocation | None
    sell_steps: int
    repair_steps: int
    duration_s: float
    reason: str = ""

    def __post_init__(self) -> None:
        """Validate result invariants."""
        if self.status == VendorStatus.SUCCESS:
            if self.reason != "":
                raise ValueError("status == SUCCESS implies reason == ''")
            if self.vendor is None:
                raise ValueError("status == SUCCESS implies vendor is not None")
        elif self.status == VendorStatus.NO_VENDOR_RESOLVED:
            if self.vendor is not None:
                raise ValueError("status == NO_VENDOR_RESOLVED implies vendor is None")
        elif self.status == VendorStatus.SKIPPED_NO_ACTION:
            if self.vendor is None:
                raise ValueError("status == SKIPPED_NO_ACTION implies vendor is not None")
            if not self.reason:
                raise ValueError("status == SKIPPED_NO_ACTION implies non-empty reason")

        if (
            isinstance(self.sell_steps, bool)
            or not isinstance(self.sell_steps, int)
            or self.sell_steps < 0
        ):
            raise ValueError(f"sell_steps must be >= 0, got {self.sell_steps}")
        if (
            isinstance(self.repair_steps, bool)
            or not isinstance(self.repair_steps, int)
            or self.repair_steps < 0
        ):
            raise ValueError(f"repair_steps must be >= 0, got {self.repair_steps}")
        if (
            isinstance(self.duration_s, bool)
            or not isinstance(self.duration_s, (int, float))
            or not math.isfinite(float(self.duration_s))
            or float(self.duration_s) < 0.0
        ):
            raise ValueError(f"duration_s must be >= 0.0, got {self.duration_s}")

    def to_json(self) -> dict[str, Any]:
        """Convert result into JSON-serializable representation."""
        return {
            "status": self.status.value,
            "vendor": {
                "node_id": self.vendor.node_id,
                "x": self.vendor.x,
                "y": self.vendor.y,
                "kind": self.vendor.kind,
                "name": self.vendor.name,
            }
            if self.vendor is not None
            else None,
            "sell_steps": self.sell_steps,
            "repair_steps": self.repair_steps,
            "duration_s": self.duration_s,
            "reason": self.reason,
        }


async def resolve_vendor_node(
    world: WorldModel,
    from_xy: tuple[float, float],
    *,
    search_radius_units: float,
    kind: str = "vendor",
    limit: int = 8,
) -> VendorLocation | None:
    """Async resolver querying the WorldModel store for nearest vendor or trainer node."""
    if kind not in {"vendor", "trainer"}:
        raise VendorError(f"kind must be 'vendor' or 'trainer', got {kind!r}")
    if (
        isinstance(search_radius_units, bool)
        or not isinstance(search_radius_units, (int, float))
        or not math.isfinite(float(search_radius_units))
        or float(search_radius_units) <= 0.0
    ):
        raise VendorError(f"search_radius_units must be > 0.0, got {search_radius_units}")
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise VendorError(f"limit must be >= 1, got {limit}")

    nodes = await world.query_nearest(
        kind=kind,
        from_xy=from_xy,
        radius=float(search_radius_units),
        limit=limit,
    )
    if not nodes:
        return None

    first = nodes[0]
    name = ""
    if first.meta_json:
        try:
            meta = json.loads(first.meta_json)
            if isinstance(meta, dict) and isinstance(meta.get("name"), str):
                name = meta["name"]
        except (json.JSONDecodeError, TypeError, ValueError):
            name = ""

    return VendorLocation(
        node_id=first.id,
        x=first.x,
        y=first.y,
        kind=first.kind,
        name=name,
    )


class VendorController:
    """Synchronous controller driving navigation to vendor and sale/repair sequences."""

    def __init__(
        self,
        navigator: Navigator,
        actuator: Actuator,
        *,
        config: VendorConfig | None = None,
        session: Session | None = None,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._navigator = navigator
        self._actuator = actuator
        self._config = config if config is not None else VendorConfig()
        self._session = session
        self._clock = clock if clock is not None else time.monotonic
        self._last_result: VendorResult | None = None

    def last_result(self) -> VendorResult | None:
        """Return the most recent VendorResult or None."""
        return self._last_result

    def run(
        self,
        vendor: VendorLocation | None,
        state: VendorStateView,
        *,
        need_repair: bool,
    ) -> VendorResult:
        """Run vendor interaction flow: navigate -> sell junk -> repair gear."""
        start = self._clock()

        if vendor is None:
            if self._session is not None:
                self._session.write_event({"event": "vendor_skipped", "reason": "no_vendor"})
            res = VendorResult(
                status=VendorStatus.NO_VENDOR_RESOLVED,
                vendor=None,
                sell_steps=0,
                repair_steps=0,
                duration_s=0.0,
                reason="no_vendor_provided",
            )
            self._last_result = res
            return res

        if state.inventory_count == 0 and not need_repair:
            if self._session is not None:
                self._session.write_event({"event": "vendor_skipped", "reason": "nothing_to_do"})
            res = VendorResult(
                status=VendorStatus.SKIPPED_NO_ACTION,
                vendor=vendor,
                sell_steps=0,
                repair_steps=0,
                duration_s=0.0,
                reason="nothing_to_do",
            )
            self._last_result = res
            return res

        if self._session is not None:
            self._session.write_event({
                "event": "vendor_started",
                "vendor_node_id": vendor.node_id,
                "vendor_kind": vendor.kind,
                "inventory_count": state.inventory_count,
                "need_repair": need_repair,
            })

        # Navigation phase
        try:
            nav_result = self._navigator.go_to((vendor.x, vendor.y))
        except Exception as exc:  # noqa: BLE001
            elapsed = max(0.0, self._clock() - start)
            res = VendorResult(
                status=VendorStatus.NAVIGATION_FAILED,
                vendor=vendor,
                sell_steps=0,
                repair_steps=0,
                duration_s=elapsed,
                reason=f"navigator_error:{type(exc).__name__}",
            )
            self._last_result = res
            return res

        if nav_result.status == NavStatus.SUCCESS:
            if self._session is not None:
                self._session.write_event({"event": "vendor_navigated"})
        elif nav_result.status == NavStatus.FAILED:
            elapsed = max(0.0, self._clock() - start)
            res = VendorResult(
                status=VendorStatus.NAVIGATION_FAILED,
                vendor=vendor,
                sell_steps=0,
                repair_steps=0,
                duration_s=elapsed,
                reason=nav_result.reason,
            )
            self._last_result = res
            return res
        elif nav_result.status == NavStatus.HARD_FAILURE:
            elapsed = max(0.0, self._clock() - start)
            res = VendorResult(
                status=VendorStatus.NAVIGATION_HARD_FAILURE,
                vendor=vendor,
                sell_steps=0,
                repair_steps=0,
                duration_s=elapsed,
                reason=nav_result.reason,
            )
            self._last_result = res
            return res
        elif nav_result.status == NavStatus.TIMEOUT:
            elapsed = max(0.0, self._clock() - start)
            res = VendorResult(
                status=VendorStatus.NAVIGATION_TIMEOUT,
                vendor=vendor,
                sell_steps=0,
                repair_steps=0,
                duration_s=elapsed,
                reason="timeout",
            )
            self._last_result = res
            return res

        # Sell phase
        sell_steps = 0
        if state.inventory_count > 0:
            while sell_steps < self._config.max_sell_steps:
                intent = MoveTo(x=vendor.x, y=vendor.y)
                try:
                    act_result = self._actuator.execute(
                        intent, position=(state.self_x, state.self_y)
                    )
                except Exception as exc:  # noqa: BLE001
                    elapsed = max(0.0, self._clock() - start)
                    res = VendorResult(
                        status=VendorStatus.SELL_FAILED,
                        vendor=vendor,
                        sell_steps=sell_steps,
                        repair_steps=0,
                        duration_s=elapsed,
                        reason=f"actuator_error:{type(exc).__name__}",
                    )
                    self._last_result = res
                    return res

                if act_result.status == ActionStatus.FAILED:
                    elapsed = max(0.0, self._clock() - start)
                    res = VendorResult(
                        status=VendorStatus.SELL_FAILED,
                        vendor=vendor,
                        sell_steps=sell_steps,
                        repair_steps=0,
                        duration_s=elapsed,
                        reason="sell_action_failed",
                    )
                    self._last_result = res
                    return res

                sell_steps += 1
                if self._session is not None:
                    self._session.write_event({"event": "vendor_sell_step", "step": sell_steps})

                if sell_steps >= self._config.sell_chunk_size:
                    break

        # Repair phase
        repair_steps = 0
        if need_repair:
            if state.durability_fraction is None:
                reason = "durability_unknown"
                if self._session is not None:
                    self._session.write_event({"event": "vendor_repair_skipped", "reason": reason})
            elif state.durability_fraction >= self._config.repair_durability_threshold:
                reason = "durability_above_threshold"
                if self._session is not None:
                    self._session.write_event({"event": "vendor_repair_skipped", "reason": reason})
            else:
                while repair_steps < self._config.max_repair_steps:
                    intent = MoveTo(x=vendor.x, y=vendor.y)
                    try:
                        act_result = self._actuator.execute(
                            intent, position=(state.self_x, state.self_y)
                        )
                    except Exception as exc:  # noqa: BLE001
                        elapsed = max(0.0, self._clock() - start)
                        res = VendorResult(
                            status=VendorStatus.REPAIR_FAILED,
                            vendor=vendor,
                            sell_steps=sell_steps,
                            repair_steps=repair_steps,
                            duration_s=elapsed,
                            reason=f"actuator_error:{type(exc).__name__}",
                        )
                        self._last_result = res
                        return res

                    if act_result.status == ActionStatus.FAILED:
                        elapsed = max(0.0, self._clock() - start)
                        res = VendorResult(
                            status=VendorStatus.REPAIR_FAILED,
                            vendor=vendor,
                            sell_steps=sell_steps,
                            repair_steps=repair_steps,
                            duration_s=elapsed,
                            reason="repair_action_failed",
                        )
                        self._last_result = res
                        return res

                    repair_steps += 1
                    if self._session is not None:
                        self._session.write_event({
                            "event": "vendor_repair_step",
                            "step": repair_steps,
                        })
                    break

        duration = max(0.0, self._clock() - start)
        if self._session is not None:
            self._session.write_event({
                "event": "vendor_completed",
                "status": VendorStatus.SUCCESS.value,
                "sell_steps": sell_steps,
                "repair_steps": repair_steps,
                "duration_s": duration,
            })

        res = VendorResult(
            status=VendorStatus.SUCCESS,
            vendor=vendor,
            sell_steps=sell_steps,
            repair_steps=repair_steps,
            duration_s=duration,
            reason="",
        )
        self._last_result = res
        return res
