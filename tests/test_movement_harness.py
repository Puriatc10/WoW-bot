"""Tests for movement harness, scenario parser, and HarnessResult."""

import ast
import json
import math
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from wow_bot.actuation.mapper import ActionResult, ActionStatus, Intent
from wow_bot.config import Config
from wow_bot.lab.movement_harness import HarnessResult, MovementHarness
from wow_bot.lab.scenario import (
    Scenario,
    ScenarioError,
    load_scenario,
    validate_waypoints,
)
from wow_bot.safety import SafetyLayer
from wow_bot.session import Session


class FakeActuator:
    """Fake Actuator implementing Actuator protocol for harness tests."""

    def __init__(self, return_status: ActionStatus = ActionStatus.SUCCESS) -> None:
        self.recorded_intents: list[tuple[Intent, tuple[float, float]]] = []
        self._aborted = False
        self.return_status = return_status
        self.raise_exc: Exception | None = None

    def execute(self, intent: Intent, *, position: tuple[float, float]) -> ActionResult:
        if self.raise_exc is not None:
            raise self.raise_exc
        self.recorded_intents.append((intent, position))
        return ActionResult(
            status=self.return_status,
            latency_ms=1.0,
            notes="fake",
        )

    def abort(self, reason: str) -> None:
        self._aborted = True

    def is_aborted(self) -> bool:
        return self._aborted

    def close(self) -> None:
        self._aborted = True


class FakePositionSource:
    """Fake position source advancing toward waypoints at configurable rate."""

    def __init__(
        self,
        waypoints: tuple[tuple[float, float], ...],
        step_speed: float = 1.0,
        arrival_tolerance: float = 0.5,
        hook: Callable[[int], None] | None = None,
        start_pos: tuple[float, float] | None = None,
    ) -> None:
        self.waypoints = waypoints
        self.step_speed = step_speed
        self.arrival_tolerance = arrival_tolerance
        self.call_count = 0
        self.target_idx = 0
        if start_pos is not None:
            self.current_pos = [start_pos[0], start_pos[1]]
        elif waypoints:
            self.current_pos = [waypoints[0][0], waypoints[0][1]]
        else:
            self.current_pos = [0.0, 0.0]
        self.hook = hook
        self.raise_exc: Exception | None = None

    def __call__(self) -> tuple[float, float]:
        self.call_count += 1
        if self.hook is not None:
            self.hook(self.call_count)
        if self.raise_exc is not None:
            raise self.raise_exc

        pos = (float(self.current_pos[0]), float(self.current_pos[1]))

        if self.target_idx < len(self.waypoints):
            target = self.waypoints[self.target_idx]
            dx = target[0] - self.current_pos[0]
            dy = target[1] - self.current_pos[1]
            dist = math.sqrt(dx * dx + dy * dy)

            if dist <= self.arrival_tolerance:
                self.target_idx += 1
                if self.target_idx < len(self.waypoints):
                    target = self.waypoints[self.target_idx]
                    dx = target[0] - self.current_pos[0]
                    dy = target[1] - self.current_pos[1]
                    dist = math.sqrt(dx * dx + dy * dy)

            if dist > 0 and self.target_idx < len(self.waypoints):
                step = min(self.step_speed, dist)
                self.current_pos[0] += (dx / dist) * step
                self.current_pos[1] += (dy / dist) * step

        return pos


class FakeClock:
    """Fake clock and sleep provider for fast deterministic timing tests."""

    def __init__(self, start_time: float = 100.0) -> None:
        self.current_time = start_time

    def clock(self) -> float:
        return self.current_time

    def sleep(self, dt: float) -> None:
        self.current_time += dt


def _make_config(tmp_path: Path) -> Config:
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="127.0.0.1:9999",
        kill_switch_key="f12",
        session_root=tmp_path,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def _read_session_events(session: Session) -> list[dict[str, Any]]:
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# Acceptance Item 1: load_scenario with valid file returns Scenario with all fields set
def test_load_scenario_valid(tmp_path: Path) -> None:
    scenario_path = tmp_path / "valid_scenario.json"
    data = {
        "name": "test_run",
        "waypoints": [[10.0, 20.0], [30.0, 40.0]],
        "arrival_tolerance": 0.8,
        "max_steps_per_waypoint": 150,
        "step_interval_s": 0.2,
        "max_session_seconds": 45.0,
    }
    scenario_path.write_text(json.dumps(data), encoding="utf-8")

    scenario = load_scenario(scenario_path)
    assert scenario.name == "test_run"
    assert scenario.waypoints == ((10.0, 20.0), (30.0, 40.0))
    assert scenario.arrival_tolerance == 0.8
    assert scenario.max_steps_per_waypoint == 150
    assert scenario.step_interval_s == 0.2
    assert scenario.max_session_seconds == 45.0


# Acceptance Item 2: load_scenario with missing file raises ScenarioError
def test_load_scenario_missing_file(tmp_path: Path) -> None:
    missing_path = tmp_path / "nonexistent.json"
    with pytest.raises(ScenarioError, match="does not exist"):
        load_scenario(missing_path)


# Acceptance Item 3: load_scenario with invalid JSON raises ScenarioError
def test_load_scenario_invalid_json(tmp_path: Path) -> None:
    bad_json_path = tmp_path / "bad.json"
    bad_json_path.write_text("{invalid json:", encoding="utf-8")
    with pytest.raises(ScenarioError, match="Invalid JSON"):
        load_scenario(bad_json_path)


# Acceptance Item 4: load_scenario with unknown top-level key raises ScenarioError
def test_load_scenario_unknown_key(tmp_path: Path) -> None:
    path = tmp_path / "unknown_key.json"
    data = {
        "name": "test_run",
        "waypoints": [[1.0, 2.0]],
        "extra_unknown_field": 123,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ScenarioError, match="Unknown top-level key"):
        load_scenario(path)


# Acceptance Item 5: load_scenario with a 3-element waypoint raises ScenarioError
def test_load_scenario_3_element_waypoint(tmp_path: Path) -> None:
    path = tmp_path / "3d_waypoint.json"
    data = {
        "name": "test_run",
        "waypoints": [[1.0, 2.0, 3.0]],
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ScenarioError, match="exactly 2 elements"):
        load_scenario(path)


# Acceptance Item 6: load_scenario with non-finite waypoint value (NaN) raises ScenarioError
def test_load_scenario_nan_waypoint(tmp_path: Path) -> None:
    path = tmp_path / "nan_waypoint.json"
    # Use raw string since JSON standard doesn't support NaN natively
    content = '{"name": "test_run", "waypoints": [[1.0, NaN]]}'
    path.write_text(content, encoding="utf-8")
    with pytest.raises(ScenarioError):
        load_scenario(path)


# Acceptance Item 7: load_scenario with arrival_tolerance <= 0 raises ScenarioError
def test_load_scenario_invalid_arrival_tolerance(tmp_path: Path) -> None:
    path = tmp_path / "zero_tolerance.json"
    data = {
        "name": "test_run",
        "waypoints": [[1.0, 2.0]],
        "arrival_tolerance": 0.0,
    }
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ScenarioError, match="arrival_tolerance"):
        load_scenario(path)


# Acceptance Item 8: validate_waypoints with empty tuple raises ScenarioError
def test_validate_waypoints_empty() -> None:
    with pytest.raises(ScenarioError, match="Waypoints cannot be empty"):
        validate_waypoints(())


# Acceptance Item 9: Harness with a single waypoint and converging source returns completed=True
def test_harness_single_waypoint_converges(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="single_wp",
        waypoints=((5.0, 0.0),),
        arrival_tolerance=0.5,
        max_steps_per_waypoint=50,
        step_interval_s=0.1,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(100.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is True
    assert res.exit_reason == "completed"
    assert res.waypoints_reached == 1
    assert res.total_waypoints == 1
    assert res.per_waypoint_steps == (5,)
    assert res.total_failed_steps == 0


# Acceptance Item 10: Harness with two waypoints reaches both in order and stops
def test_harness_two_waypoints_in_order(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="two_wp",
        waypoints=((3.0, 0.0), (3.0, 4.0)),
        arrival_tolerance=0.5,
        max_steps_per_waypoint=50,
        step_interval_s=0.1,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is True
    assert res.exit_reason == "completed"
    assert res.waypoints_reached == 2
    assert res.total_waypoints == 2
    assert len(res.per_waypoint_steps) == 2


# Acceptance Item 11: Harness emits "harness_waypoint_reached" and "harness_finished"
def test_harness_emits_session_events(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="event_test",
        waypoints=((1.0, 0.0), (2.0, 0.0)),
        arrival_tolerance=0.5,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    assert res.completed is True
    session.close("clean")

    events = _read_session_events(session)
    reached_events = [e for e in events if e.get("event") == "harness_waypoint_reached"]
    finished_events = [e for e in events if e.get("event") == "harness_finished"]

    assert len(reached_events) == 2
    assert reached_events[0]["payload"]["index"] == 0
    assert reached_events[1]["payload"]["index"] == 1

    assert len(finished_events) == 1
    assert finished_events[0]["payload"]["completed"] is True
    assert finished_events[0]["payload"]["exit_reason"] == "completed"


# Acceptance Item 12: Harness exits with exit_reason="timeout" when clock advances past deadline
def test_harness_timeout(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="timeout_test",
        waypoints=((100.0, 0.0),),
        max_session_seconds=5.0,
        step_interval_s=1.0,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is False
    assert res.exit_reason == "timeout"


# Acceptance Item 13: Harness exits with exit_reason="kill_switch" when safety.is_aborted() becomes True
def test_harness_kill_switch_abort(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="kill_switch_test",
        waypoints=((10.0, 0.0),),
    )

    def abort_hook(call_count: int) -> None:
        if call_count == 2:
            safety.abort("test_kill_switch")

    source = FakePositionSource(
        waypoints=scenario.waypoints,
        step_speed=1.0,
        start_pos=(0.0, 0.0),
        hook=abort_hook,
    )
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is False
    assert res.exit_reason == "kill_switch"


# Acceptance Item 14: Harness exits with exit_reason="aborted" when actuator.is_aborted() becomes True
def test_harness_actuator_abort(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="actuator_abort_test",
        waypoints=((10.0, 0.0),),
    )

    def abort_actuator_hook(call_count: int) -> None:
        if call_count == 2:
            actuator.abort("hardware_fault")

    source = FakePositionSource(
        waypoints=scenario.waypoints,
        step_speed=1.0,
        start_pos=(0.0, 0.0),
        hook=abort_actuator_hook,
    )
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is False
    assert res.exit_reason == "aborted"


# Acceptance Item 15: Harness exits with exit_reason="max_steps" when position source never converges
def test_harness_max_steps_exceeded(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="max_steps_test",
        waypoints=((100.0, 0.0),),
        max_steps_per_waypoint=5,
        step_interval_s=0.0,
    )
    # Position source that never moves
    source = FakePositionSource(
        waypoints=scenario.waypoints,
        step_speed=0.0,
        start_pos=(0.0, 0.0),
    )
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is False
    assert res.exit_reason == "max_steps"
    assert res.per_waypoint_steps == (5,)

    events = _read_session_events(session)
    timeout_events = [e for e in events if e.get("event") == "harness_waypoint_timeout"]
    assert len(timeout_events) == 1
    assert timeout_events[0]["payload"]["index"] == 0
    assert timeout_events[0]["payload"]["steps"] == 5


# Acceptance Item 16: per_waypoint_steps and total_failed_steps counted correctly
def test_harness_failed_steps_counting(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator(return_status=ActionStatus.FAILED)
    scenario = Scenario(
        name="failed_steps_test",
        waypoints=((2.0, 0.0),),
        arrival_tolerance=0.5,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.completed is True
    assert res.waypoints_reached == 1
    assert res.total_failed_steps == 2
    assert res.per_waypoint_steps == (2,)


# Acceptance Item 17: duration_s is computed from the injected clock, not wall time
def test_harness_duration_from_injected_clock(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="clock_test",
        waypoints=((2.0, 0.0),),
        step_interval_s=5.0,
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(1000.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    res = harness.run()
    session.close("clean")

    assert res.duration_s == 10.0  # 2 steps * 5.0s sleep


# Acceptance Item 18: If actuator.execute raises, exception propagates and NO "harness_finished" event
def test_harness_actuator_exception_propagates(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    actuator.raise_exc = RuntimeError("Actuator driver hardware crash")
    scenario = Scenario(
        name="exc_test",
        waypoints=((10.0, 0.0),),
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    with pytest.raises(RuntimeError, match="Actuator driver hardware crash"):
        harness.run()

    session.close("clean")
    events = _read_session_events(session)
    assert not any(e.get("event") == "harness_finished" for e in events)


# Acceptance Item 19: If position_source raises, exception propagates
def test_harness_position_source_exception_propagates(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    session = Session.start(config)
    safety = SafetyLayer(config, session)
    safety.arm()

    actuator = FakeActuator()
    scenario = Scenario(
        name="pos_exc_test",
        waypoints=((10.0, 0.0),),
    )
    source = FakePositionSource(waypoints=scenario.waypoints, step_speed=1.0, start_pos=(0.0, 0.0))
    source.raise_exc = ValueError("Position perception read error")
    fake_clock = FakeClock(0.0)

    harness = MovementHarness(
        actuator=actuator,
        position_source=source,
        scenario=scenario,
        session=session,
        safety=safety,
        clock=fake_clock.clock,
        sleep=fake_clock.sleep,
    )

    with pytest.raises(ValueError, match="Position perception read error"):
        harness.run()

    session.close("clean")
    events = _read_session_events(session)
    assert not any(e.get("event") == "harness_finished" for e in events)


# Acceptance Item 20: HarnessResult.to_json returns JSON-serializable dict with all fields
def test_harness_result_to_json() -> None:
    res = HarnessResult(
        scenario_name="demo_scenario",
        completed=True,
        waypoints_reached=3,
        total_waypoints=3,
        duration_s=12.34,
        exit_reason="completed",
        per_waypoint_steps=(5, 8, 12),
        total_failed_steps=1,
    )

    res_json = res.to_json()
    assert isinstance(res_json, dict)
    assert res_json["scenario_name"] == "demo_scenario"
    assert res_json["completed"] is True
    assert res_json["waypoints_reached"] == 3
    assert res_json["total_waypoints"] == 3
    assert res_json["duration_s"] == 12.34
    assert res_json["exit_reason"] == "completed"
    assert res_json["per_waypoint_steps"] == [5, 8, 12]
    assert res_json["total_failed_steps"] == 1

    # Verify JSON serializability
    dumped = json.dumps(res_json)
    loaded = json.loads(dumped)
    assert loaded == res_json


# Static AST import check for forbidden modules in wow_bot.lab
def test_static_ast_forbidden_imports() -> None:
    lab_dir = Path(__file__).parent.parent / "src" / "wow_bot" / "lab"
    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.executor",
        "wow_bot.reflex",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }

    for py_file in lab_dir.glob("*.py"):
        source = py_file.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(py_file))

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    for forbidden in forbidden_modules:
                        assert not alias.name.startswith(forbidden), (
                            f"Forbidden import '{alias.name}' in {py_file.name}"
                        )
            elif isinstance(node, ast.ImportFrom) and node.module:
                for forbidden in forbidden_modules:
                    assert not node.module.startswith(forbidden), (
                        f"Forbidden import from '{node.module}' in {py_file.name}"
                    )
