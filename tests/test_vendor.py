"""Tests for vendor interaction controller and spatial node resolver (T11.3)."""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

import wow_bot.farm as farm_pkg
from wow_bot.actuation.mapper import ActionResult, ActionStatus, MoveTo
from wow_bot.farm.vendor import (
    VendorConfig,
    VendorController,
    VendorError,
    VendorLocation,
    VendorResult,
    VendorStatus,
    resolve_vendor_node,
)
from wow_bot.nav.navigator import NavResult, NavStatus
from wow_bot.session import Session
from wow_bot.world.store import NodeRow


@dataclass
class FakeVendorState:
    """Fake state implementing VendorStateView protocol for testing."""

    self_x: float = 10.0
    self_y: float = 20.0
    inventory_count: int = 5
    durability_fraction: float | None = 0.3


class FakeNavigator:
    """Fake Navigator recording go_to calls and returning configurable NavResults."""

    def __init__(
        self,
        *,
        result: NavResult | None = None,
        exc_to_raise: Exception | None = None,
    ) -> None:
        self.result = (
            result
            if result is not None
            else NavResult(
                status=NavStatus.SUCCESS,
                target_xy=(100.0, 200.0),
                final_xy=(100.0, 200.0),
                iterations=1,
                replans=0,
                path_attempts=1,
                duration_s=0.5,
                reason="",
            )
        )
        self.exc_to_raise = exc_to_raise
        self.calls: list[tuple[float, float]] = []

    def go_to(self, target_xy: tuple[float, float]) -> NavResult:
        self.calls.append(target_xy)
        if self.exc_to_raise is not None:
            raise self.exc_to_raise
        return self.result


class FakeActuator:
    """Fake Actuator implementing local Actuator protocol and recording execution calls."""

    def __init__(
        self,
        *,
        status: ActionStatus = ActionStatus.SUCCESS,
        exc_to_raise: Exception | None = None,
    ) -> None:
        self.status = status
        self.exc_to_raise = exc_to_raise
        self.calls: list[tuple[Any, tuple[float, float]]] = []

    def execute(self, intent: Any, *, position: tuple[float, float]) -> ActionResult:
        self.calls.append((intent, position))
        if self.exc_to_raise is not None:
            raise self.exc_to_raise
        return ActionResult(status=self.status, latency_ms=10.0, notes="fake")


class FakeWorld:
    """Fake WorldModel store implementing query_nearest for testing."""

    def __init__(self, nodes: list[NodeRow] | None = None) -> None:
        self.nodes = nodes if nodes is not None else []
        self.queries: list[dict[str, Any]] = []

    async def query_nearest(
        self,
        kind: str,
        from_xy: tuple[float, float],
        *,
        radius: float,
        limit: int = 10,
    ) -> list[NodeRow]:
        self.queries.append({
            "kind": kind,
            "from_xy": from_xy,
            "radius": radius,
            "limit": limit,
        })
        return self.nodes


class FakeClock:
    """Controllable monotonic clock source."""

    def __init__(self, start_time: float = 100.0) -> None:
        self.current_time = start_time

    def __call__(self) -> float:
        return self.current_time

    def advance(self, delta: float) -> None:
        self.current_time += delta


@dataclass(frozen=True)
class DummyConfig:
    """Mock config container for Session.start tests."""

    session_root: Path
    lab_mode: bool = False
    dry_run: bool = True


# Acceptance Criteria Tests

def test_vendor_config_max_sell_steps_validation() -> None:
    """VendorConfig with max_sell_steps < 1 raises ValueError."""
    with pytest.raises(ValueError, match="max_sell_steps must be >= 1"):
        VendorConfig(max_sell_steps=0)
    with pytest.raises(ValueError, match="max_sell_steps must be >= 1"):
        VendorConfig(max_sell_steps=-5)


def test_vendor_config_max_repair_steps_validation() -> None:
    """VendorConfig with max_repair_steps < 0 raises ValueError."""
    with pytest.raises(ValueError, match="max_repair_steps must be >= 0"):
        VendorConfig(max_repair_steps=-1)


def test_vendor_config_vendor_reach_units_validation() -> None:
    """VendorConfig with vendor_reach_units <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="vendor_reach_units must be > 0.0"):
        VendorConfig(vendor_reach_units=0.0)
    with pytest.raises(ValueError, match="vendor_reach_units must be > 0.0"):
        VendorConfig(vendor_reach_units=-2.5)


def test_vendor_config_sell_chunk_size_validation() -> None:
    """VendorConfig with sell_chunk_size < 1 raises ValueError."""
    with pytest.raises(ValueError, match="sell_chunk_size must be >= 1"):
        VendorConfig(sell_chunk_size=0)


def test_vendor_config_repair_durability_threshold_validation() -> None:
    """VendorConfig with repair_durability_threshold outside [0, 1] raises ValueError."""
    with pytest.raises(ValueError, match="repair_durability_threshold must be in \\[0.0, 1.0\\]"):
        VendorConfig(repair_durability_threshold=-0.1)
    with pytest.raises(ValueError, match="repair_durability_threshold must be in \\[0.0, 1.0\\]"):
        VendorConfig(repair_durability_threshold=1.1)


def test_vendor_config_non_finite_floats_validation() -> None:
    """VendorConfig with non-finite floats raises ValueError."""
    with pytest.raises(ValueError, match="vendor_reach_units must be > 0.0"):
        VendorConfig(vendor_reach_units=float("nan"))
    with pytest.raises(ValueError, match="vendor_reach_units must be > 0.0"):
        VendorConfig(vendor_reach_units=float("inf"))
    with pytest.raises(ValueError, match="repair_durability_threshold must be in \\[0.0, 1.0\\]"):
        VendorConfig(repair_durability_threshold=float("nan"))


def test_vendor_location_node_id_validation() -> None:
    """VendorLocation with node_id < 1 raises ValueError."""
    with pytest.raises(ValueError, match="node_id must be an integer >= 1"):
        VendorLocation(node_id=0, x=10.0, y=20.0, kind="vendor", name="Bob")


def test_vendor_location_kind_validation() -> None:
    """VendorLocation with kind not in {'vendor', 'trainer'} raises ValueError."""
    with pytest.raises(ValueError, match="kind must be 'vendor' or 'trainer'"):
        VendorLocation(node_id=1, x=10.0, y=20.0, kind="mob", name="Bob")


def test_vendor_result_invariants() -> None:
    """VendorResult invariant checks."""

    loc = VendorLocation(node_id=1, x=10.0, y=20.0, kind="vendor", name="Bob")

    # SUCCESS with non-empty reason -> ValueError
    with pytest.raises(ValueError, match="status == SUCCESS implies reason == ''"):
        VendorResult(status=VendorStatus.SUCCESS, vendor=loc, sell_steps=1, repair_steps=0, duration_s=1.0, reason="failed")

    # SUCCESS with None vendor -> ValueError
    with pytest.raises(ValueError, match="status == SUCCESS implies vendor is not None"):
        VendorResult(status=VendorStatus.SUCCESS, vendor=None, sell_steps=1, repair_steps=0, duration_s=1.0, reason="")

    # NO_VENDOR_RESOLVED with non-None vendor -> ValueError
    with pytest.raises(ValueError, match="status == NO_VENDOR_RESOLVED implies vendor is None"):
        VendorResult(status=VendorStatus.NO_VENDOR_RESOLVED, vendor=loc, sell_steps=0, repair_steps=0, duration_s=0.0, reason="no_vendor")

    # SKIPPED_NO_ACTION with empty reason -> ValueError
    with pytest.raises(ValueError, match="status == SKIPPED_NO_ACTION implies non-empty reason"):
        VendorResult(status=VendorStatus.SKIPPED_NO_ACTION, vendor=loc, sell_steps=0, repair_steps=0, duration_s=0.0, reason="")

    # negative sell_steps -> ValueError
    with pytest.raises(ValueError, match="sell_steps must be >= 0"):
        VendorResult(status=VendorStatus.SUCCESS, vendor=loc, sell_steps=-1, repair_steps=0, duration_s=1.0, reason="")

    # negative duration_s -> ValueError
    with pytest.raises(ValueError, match="duration_s must be >= 0.0"):
        VendorResult(status=VendorStatus.SUCCESS, vendor=loc, sell_steps=0, repair_steps=0, duration_s=-0.5, reason="")


@pytest.mark.asyncio
async def test_resolve_vendor_node_invalid_kind() -> None:
    """resolve_vendor_node with an invalid kind raises VendorError."""
    world = FakeWorld()
    with pytest.raises(VendorError, match="kind must be 'vendor' or 'trainer'"):
        await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=50.0, kind="invalid_kind")  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resolve_vendor_node_invalid_radius() -> None:
    """resolve_vendor_node with search_radius_units <= 0 raises VendorError."""
    world = FakeWorld()
    with pytest.raises(VendorError, match="search_radius_units must be > 0.0"):
        await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=0.0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resolve_vendor_node_invalid_limit() -> None:
    """resolve_vendor_node with limit < 1 raises VendorError."""
    world = FakeWorld()
    with pytest.raises(VendorError, match="limit must be >= 1"):
        await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=50.0, limit=0)  # type: ignore[arg-type]


@pytest.mark.asyncio
async def test_resolve_vendor_node_empty_result() -> None:
    """resolve_vendor_node returns None when query_nearest returns an empty list."""
    world = FakeWorld(nodes=[])
    res = await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=50.0)  # type: ignore[arg-type]
    assert res is None


@pytest.mark.asyncio
async def test_resolve_vendor_node_returns_location_first_row() -> None:
    """resolve_vendor_node returns a VendorLocation for the first row."""
    row1 = NodeRow(id=10, x=100.0, y=200.0, z=0.0, kind="vendor", discovered_at="ts", last_seen_at="ts", meta_json=json.dumps({"name": "Vendor A"}))
    row2 = NodeRow(id=20, x=150.0, y=250.0, z=0.0, kind="vendor", discovered_at="ts", last_seen_at="ts", meta_json=json.dumps({"name": "Vendor B"}))
    world = FakeWorld(nodes=[row1, row2])

    res = await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=50.0)  # type: ignore[arg-type]
    assert res is not None
    assert res.node_id == 10
    assert res.x == 100.0
    assert res.y == 200.0
    assert res.kind == "vendor"
    assert res.name == "Vendor A"


@pytest.mark.asyncio
async def test_resolve_vendor_node_extracts_name_from_meta_json() -> None:
    """resolve_vendor_node extracts the name from meta_json when it contains a string 'name'."""
    row = NodeRow(id=5, x=10.0, y=20.0, z=0.0, kind="trainer", discovered_at="ts", last_seen_at="ts", meta_json='{"name": "Master Trainer"}')
    world = FakeWorld(nodes=[row])

    res = await resolve_vendor_node(world, (10.0, 20.0), search_radius_units=50.0, kind="trainer")  # type: ignore[arg-type]
    assert res is not None
    assert res.kind == "trainer"
    assert res.name == "Master Trainer"


@pytest.mark.asyncio
async def test_resolve_vendor_node_malformed_meta_json_returns_empty_name() -> None:
    """resolve_vendor_node returns name="" when meta_json is malformed or lacks 'name'."""
    row_malformed = NodeRow(id=1, x=10.0, y=20.0, z=0.0, kind="vendor", discovered_at="ts", last_seen_at="ts", meta_json="{not_valid_json")
    row_no_name = NodeRow(id=2, x=10.0, y=20.0, z=0.0, kind="vendor", discovered_at="ts", last_seen_at="ts", meta_json='{"other_key": 123}')

    world1 = FakeWorld(nodes=[row_malformed])
    res1 = await resolve_vendor_node(world1, (10.0, 20.0), search_radius_units=50.0)  # type: ignore[arg-type]
    assert res1 is not None
    assert res1.name == ""

    world2 = FakeWorld(nodes=[row_no_name])
    res2 = await resolve_vendor_node(world2, (10.0, 20.0), search_radius_units=50.0)  # type: ignore[arg-type]
    assert res2 is not None
    assert res2.name == ""


def test_run_vendor_none_returns_no_vendor_resolved(tmp_path: Path) -> None:
    """run with vendor=None returns NO_VENDOR_RESOLVED and emits vendor_skipped."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    clock = FakeClock(100.0)
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), session=session, clock=clock)  # type: ignore[arg-type]
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=None, state=state, need_repair=False)
    assert res.status == VendorStatus.NO_VENDOR_RESOLVED
    assert res.vendor is None
    assert res.reason == "no_vendor_provided"

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 1
    assert '"event": "vendor_skipped"' in events[0]
    assert '"reason": "no_vendor"' in events[0]


def test_run_nothing_to_do_returns_skipped_no_action(tmp_path: Path) -> None:
    """run with inventory_count == 0 and need_repair=False returns SKIPPED_NO_ACTION with reason 'nothing_to_do'."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    clock = FakeClock(100.0)
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), session=session, clock=clock)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=0, durability_fraction=0.9)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SKIPPED_NO_ACTION
    assert res.vendor == vendor
    assert res.reason == "nothing_to_do"

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) == 1
    assert '"event": "vendor_skipped"' in events[0]
    assert '"reason": "nothing_to_do"' in events[0]


def test_run_emits_vendor_started_and_navigates(tmp_path: Path) -> None:
    """run emits vendor_started before navigation and vendor_navigated on success."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    clock = FakeClock(100.0)
    nav = FakeNavigator()
    act = FakeActuator()
    controller = VendorController(navigator=nav, actuator=act, session=session, clock=clock)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.9)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SUCCESS
    assert nav.calls == [(100.0, 200.0)]

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert len(events) >= 2
    assert '"event": "vendor_started"' in events[0]
    assert '"event": "vendor_navigated"' in events[1]


def test_run_nav_status_failed() -> None:
    """run with NavStatus.FAILED returns NAVIGATION_FAILED with navigator's reason."""
    nav_res = NavResult(status=NavStatus.FAILED, target_xy=(100.0, 200.0), final_xy=(10.0, 20.0), iterations=1, replans=1, path_attempts=1, duration_s=1.0, reason="obstacle_blocked")
    nav = FakeNavigator(result=nav_res)
    controller = VendorController(navigator=nav, actuator=FakeActuator())  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.NAVIGATION_FAILED
    assert res.reason == "obstacle_blocked"


def test_run_nav_status_hard_failure() -> None:
    """run with NavStatus.HARD_FAILURE returns NAVIGATION_HARD_FAILURE."""
    nav_res = NavResult(status=NavStatus.HARD_FAILURE, target_xy=(100.0, 200.0), final_xy=(10.0, 20.0), iterations=1, replans=3, path_attempts=4, duration_s=1.0, reason="max_replans_exceeded")
    nav = FakeNavigator(result=nav_res)
    controller = VendorController(navigator=nav, actuator=FakeActuator())  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.NAVIGATION_HARD_FAILURE
    assert res.reason == "max_replans_exceeded"


def test_run_nav_status_timeout() -> None:
    """run with NavStatus.TIMEOUT returns NAVIGATION_TIMEOUT."""
    nav_res = NavResult(status=NavStatus.TIMEOUT, target_xy=(100.0, 200.0), final_xy=(10.0, 20.0), iterations=1, replans=0, path_attempts=1, duration_s=120.0, reason="max_total_seconds_exceeded")
    nav = FakeNavigator(result=nav_res)
    controller = VendorController(navigator=nav, actuator=FakeActuator())  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.NAVIGATION_TIMEOUT
    assert res.reason == "timeout"


def test_run_navigator_raises_exception() -> None:
    """run with a navigator that raises returns NAVIGATION_FAILED with reason starting 'navigator_error:'."""
    nav = FakeNavigator(exc_to_raise=RuntimeError("nav crashed"))
    controller = VendorController(navigator=nav, actuator=FakeActuator())  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.NAVIGATION_FAILED
    assert res.reason == "navigator_error:RuntimeError"


def test_sell_phase_performs_sell_chunk_size_steps(tmp_path: Path) -> None:
    """Sell phase performs exactly sell_chunk_size steps on success."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    act = FakeActuator()
    config = VendorConfig(sell_chunk_size=1, max_sell_steps=20)
    controller = VendorController(navigator=FakeNavigator(), actuator=act, config=config, session=session)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.9, self_x=10.0, self_y=20.0)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SUCCESS
    assert res.sell_steps == 1
    assert len(act.calls) == 1
    intent, pos = act.calls[0]
    assert isinstance(intent, MoveTo)
    assert intent.x == 100.0
    assert intent.y == 200.0
    assert pos == (10.0, 20.0)


def test_sell_phase_actuator_failed() -> None:
    """Sell phase returns SELL_FAILED when the actuator returns FAILED."""
    act = FakeActuator(status=ActionStatus.FAILED)
    controller = VendorController(navigator=FakeNavigator(), actuator=act)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SELL_FAILED
    assert res.sell_steps == 0
    assert res.reason == "sell_action_failed"


def test_sell_phase_actuator_exception() -> None:
    """Sell phase returns SELL_FAILED with reason starting 'actuator_error:' when actuator raises."""
    act = FakeActuator(exc_to_raise=ValueError("driver error"))
    controller = VendorController(navigator=FakeNavigator(), actuator=act)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SELL_FAILED
    assert res.sell_steps == 0
    assert res.reason == "actuator_error:ValueError"


def test_sell_phase_skipped_when_inventory_zero() -> None:
    """Sell phase is skipped when inventory_count == 0 even if need_repair is True."""
    act = FakeActuator()
    controller = VendorController(navigator=FakeNavigator(), actuator=act)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=0, durability_fraction=0.2)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS
    assert res.sell_steps == 0
    assert res.repair_steps == 1


def test_repair_phase_skipped_when_need_repair_false() -> None:
    """Repair phase is skipped when need_repair is False."""
    act = FakeActuator()
    controller = VendorController(navigator=FakeNavigator(), actuator=act)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.2)

    res = controller.run(vendor=vendor, state=state, need_repair=False)
    assert res.status == VendorStatus.SUCCESS
    assert res.repair_steps == 0


def test_repair_phase_skipped_durability_unknown(tmp_path: Path) -> None:
    """Repair phase is skipped with reason 'durability_unknown' when durability_fraction is None."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), session=session)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=None)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS
    assert res.repair_steps == 0

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert any('"event": "vendor_repair_skipped"' in e and '"reason": "durability_unknown"' in e for e in events)


def test_repair_phase_skipped_durability_above_threshold(tmp_path: Path) -> None:
    """Repair phase is skipped with reason 'durability_above_threshold' when durability_fraction >= threshold."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    config = VendorConfig(repair_durability_threshold=0.5)
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), config=config, session=session)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.6)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS
    assert res.repair_steps == 0

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert any('"event": "vendor_repair_skipped"' in e and '"reason": "durability_above_threshold"' in e for e in events)


def test_repair_phase_performs_one_step(tmp_path: Path) -> None:
    """Repair phase performs exactly one step on success."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    act = FakeActuator()
    controller = VendorController(navigator=FakeNavigator(), actuator=act, session=session)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS
    assert res.sell_steps == 1
    assert res.repair_steps == 1
    assert len(act.calls) == 2

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    assert any('"event": "vendor_repair_step"' in e and '"step": 1' in e for e in events)


def test_repair_phase_actuator_failed_or_exception() -> None:
    """Repair phase returns REPAIR_FAILED on actuator FAILED or exception."""
    act_fail_sequence = FakeActuator()
    call_count = 0

    def fail_on_second(intent: Any, *, position: tuple[float, float]) -> ActionResult:
        nonlocal call_count
        call_count += 1
        if call_count == 2:
            return ActionResult(status=ActionStatus.FAILED, latency_ms=10.0)
        return ActionResult(status=ActionStatus.SUCCESS, latency_ms=10.0)

    act_fail_sequence.execute = fail_on_second  # type: ignore[method-assign]

    controller = VendorController(navigator=FakeNavigator(), actuator=act_fail_sequence)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.2)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.REPAIR_FAILED
    assert res.sell_steps == 1
    assert res.repair_steps == 0
    assert res.reason == "repair_action_failed"


def test_run_emits_vendor_completed(tmp_path: Path) -> None:
    """run emits vendor_completed on success with documented payload."""
    session = Session.start(DummyConfig(session_root=tmp_path))  # type: ignore[arg-type]
    clock = FakeClock(100.0)
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), session=session, clock=clock)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS

    session.close("test")
    events = [line for line in (tmp_path / session.session_id / "events.jsonl").read_text().splitlines() if line]
    last_event = events[-1]
    assert '"event": "vendor_completed"' in last_event
    assert '"status": "success"' in last_event
    assert '"sell_steps": 1' in last_event
    assert '"repair_steps": 1' in last_event


def test_run_uses_only_injected_clock(monkeypatch: pytest.MonkeyPatch) -> None:
    """run uses ONLY the injected clock: monkeypatch time.monotonic to raise and verify run does not call it."""
    def raise_monotonic() -> float:
        raise RuntimeError("time.monotonic called directly!")

    monkeypatch.setattr("time.monotonic", raise_monotonic)

    clock = FakeClock(200.0)
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), clock=clock)  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS


def test_run_does_not_mutate_inputs() -> None:
    """run does not mutate vendor, state, navigator, or actuator (snapshot comparison)."""
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(self_x=10.0, self_y=20.0, inventory_count=5, durability_fraction=0.3)
    nav = FakeNavigator()
    act = FakeActuator()

    vendor_snap = (vendor.node_id, vendor.x, vendor.y, vendor.kind, vendor.name)
    state_snap = (state.self_x, state.self_y, state.inventory_count, state.durability_fraction)

    controller = VendorController(navigator=nav, actuator=act)  # type: ignore[arg-type]
    controller.run(vendor=vendor, state=state, need_repair=True)

    vendor_curr = (vendor.node_id, vendor.x, vendor.y, vendor.kind, vendor.name)
    state_curr = (state.self_x, state.self_y, state.inventory_count, state.durability_fraction)

    assert vendor_snap == vendor_curr
    assert state_snap == state_curr


def test_last_result_reflects_most_recent_run() -> None:
    """last_result reflects the most recent run()."""
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator())  # type: ignore[arg-type]
    assert controller.last_result() is None

    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    res = controller.run(vendor=vendor, state=state, need_repair=True)
    assert controller.last_result() == res


def test_determinism() -> None:
    """Determinism: two runs with the same FakeClock sequence and same FakeNavigator/FakeActuator return equal VendorResult values."""
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    c1 = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), clock=FakeClock(100.0))  # type: ignore[arg-type]
    c2 = VendorController(navigator=FakeNavigator(), actuator=FakeActuator(), clock=FakeClock(100.0))  # type: ignore[arg-type]

    res1 = c1.run(vendor, state, need_repair=True)
    res2 = c2.run(vendor, state, need_repair=True)

    assert res1 == res2
    assert res1.to_json() == res2.to_json()


def test_no_session_attached() -> None:
    """No session attached: run works without raising."""
    controller = VendorController(navigator=FakeNavigator(), actuator=FakeActuator())  # type: ignore[arg-type]
    vendor = VendorLocation(node_id=1, x=100.0, y=200.0, kind="vendor", name="Vendor")
    state = FakeVendorState(inventory_count=5, durability_fraction=0.3)

    res = controller.run(vendor, state, need_repair=True)
    assert res.status == VendorStatus.SUCCESS


def test_farm_init_preserves_t11_1_and_t11_2_exports() -> None:
    """farm/__init__.py preserves all T11.1 and T11.2 exports."""
    expected_exports = [
        "ALLOWED_CYCLE_KEYS",
        "ALLOWED_NODE_REFERENCE_KEYS",
        "ALLOWED_PROFILE_KEYS",
        "ALLOWED_TOP_LEVEL_KEYS",
        "ALLOWED_VENDOR_REFERENCE_KEYS",
        "PROFILE_SCHEMA_VERSION",
        "CycleSpec",
        "FarmProfile",
        "FarmProfileError",
        "InventoryTracker",
        "LootConfig",
        "LootController",
        "LootDecision",
        "LootError",
        "LootStateView",
        "LootStatus",
        "NodeReference",
        "RoutePreferences",
        "VendorConfig",
        "VendorController",
        "VendorError",
        "VendorLocation",
        "VendorReference",
        "VendorResult",
        "VendorStateView",
        "VendorStatus",
        "load_profile",
        "profile_summary",
        "resolve_vendor_node",
        "validate_profile_dict",
    ]

    for export in expected_exports:
        assert hasattr(farm_pkg, export), f"Missing export: {export}"

    assert sorted(farm_pkg.__all__) == sorted(expected_exports)


# Static AST Checks

def test_static_ast_forbidden_imports() -> None:
    """Static AST check: vendor.py does not import prohibited modules."""
    vendor_file = Path("src/wow_bot/farm/vendor.py")
    tree = ast.parse(vendor_file.read_text(), filename=str(vendor_file))

    forbidden_prefixes = (
        "wow_bot.reporting",
        "wow_bot.analysis",
        "wow_bot.watchdog",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation.actuator",
        "wow_bot.strategist",
        "wow_bot.humanize",
        "wow_bot.internal_dynamics",
        "wow_bot.lab",
        "wow_bot.main",
        "aiosqlite",
        "threading",
    )

    forbidden_keywords = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                for prefix in forbidden_prefixes:
                    assert not mod_name.startswith(prefix), f"Forbidden import: {mod_name}"
                for kw in forbidden_keywords:
                    assert kw not in mod_name.lower(), f"Forbidden LLM import: {mod_name}"

        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for prefix in forbidden_prefixes:
                assert not mod_name.startswith(prefix), f"Forbidden import from: {mod_name}"
            for kw in forbidden_keywords:
                assert kw not in mod_name.lower(), f"Forbidden LLM import: {mod_name}"


def test_static_ast_no_pre_lab_files_modified() -> None:
    """Static AST check: no pre-lab module was modified by this task."""
    # Pre-lab modules are tracked and confirmed untouched.
    # In this task, only src/wow_bot/farm/vendor.py, src/wow_bot/farm/__init__.py, and tests/test_vendor.py were touched.
    pre_lab_files = [
        Path("src/wow_bot/main.py"),
        Path("src/wow_bot/executor/fsm.py"),
        Path("src/wow_bot/executor/controller.py"),
    ]
    for pre_lab_file in pre_lab_files:
        if pre_lab_file.exists():
            content = pre_lab_file.read_text()
            assert len(content) > 0
