"""Unit tests for the Navigator component."""

from __future__ import annotations

import ast
import json
import math
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from wow_bot.actuation.mapper import ActionResult, ActionStatus, MoveTo
from wow_bot.config import Config
from wow_bot.nav.graph import GraphEdge, GraphNode, NavGraph, NodeKind
from wow_bot.nav.navigator import (
    NavConfig,
    Navigator,
    NavResult,
    NavStatus,
    SimpleReplanner,
)
from wow_bot.session import Session


def make_graph(
    nodes: list[tuple[int, float, float, float, NodeKind, str]],
    edges: list[tuple[int, int, float]],
) -> NavGraph:
    """Helper builder returning a NavGraph for testing."""
    nodes_dict: dict[int, GraphNode] = {}
    for nid, x, y, z, kind, last_seen in nodes:
        nodes_dict[nid] = GraphNode(
            id=nid,
            x=x,
            y=y,
            z=z,
            kind=kind,
            last_seen_at=last_seen,
        )

    adj: dict[int, list[GraphEdge]] = {nid: [] for nid in nodes_dict}
    for from_id, to_id, cost in edges:
        if from_id in adj:
            adj[from_id].append(GraphEdge(to_id=to_id, cost=cost))

    sorted_nodes = dict(sorted(nodes_dict.items(), key=lambda item: item[0]))
    sorted_adj: dict[int, tuple[GraphEdge, ...]] = {}
    for nid in sorted_nodes:
        edges_list = adj.get(nid, [])
        edges_list.sort(key=lambda e: e.to_id)
        sorted_adj[nid] = tuple(edges_list)

    return NavGraph(
        nodes=MappingProxyType(sorted_nodes),
        adjacency=MappingProxyType(sorted_adj),
    )


class FakeClock:
    """Controllable clock source."""

    def __init__(self, start_time: float = 0.0) -> None:
        self.time = start_time

    def __call__(self) -> float:
        return self.time

    def advance(self, amount: float) -> None:
        self.time += amount


class FakeSleep:
    """Recording sleep callable."""

    def __init__(self) -> None:
        self.calls: list[float] = []

    def __call__(self, duration: float) -> None:
        self.calls.append(duration)


class FakePositionSource:
    """Scripted position source advancing toward movement target or jumping."""

    def __init__(
        self,
        start_pos: tuple[float, float],
        jump_sequence: dict[int, tuple[float, float]] | None = None,
    ) -> None:
        self.current_pos = start_pos
        self.call_count = 0
        self.jump_sequence = jump_sequence or {}

    def advance_towards(self, target_x: float, target_y: float, step_size: float = 5.0) -> None:
        """Advance current position towards target."""
        dx = target_x - self.current_pos[0]
        dy = target_y - self.current_pos[1]
        dist = math.hypot(dx, dy)
        if dist <= step_size:
            self.current_pos = (target_x, target_y)
        else:
            self.current_pos = (
                self.current_pos[0] + (dx / dist) * step_size,
                self.current_pos[1] + (dy / dist) * step_size,
            )

    def __call__(self) -> tuple[float, float]:
        self.call_count += 1
        if self.call_count in self.jump_sequence:
            self.current_pos = self.jump_sequence[self.call_count]
        return self.current_pos


class FakeActuator:
    """Fake Actuator implementing Actuator protocol."""

    def __init__(
        self,
        position_source: FakePositionSource | None = None,
        *,
        step_size: float = 5.0,
        fail_status: bool = False,
        abort_after_steps: int | None = None,
    ) -> None:
        self.position_source = position_source
        self.step_size = step_size
        self.fail_status = fail_status
        self.abort_after_steps = abort_after_steps
        self.step_count = 0
        self._aborted = False
        self.executed_intents: list[tuple[Any, tuple[float, float]]] = []

    def execute(self, intent: Any, *, position: tuple[float, float]) -> ActionResult:
        self.step_count += 1
        self.executed_intents.append((intent, position))

        if self.abort_after_steps is not None and self.step_count >= self.abort_after_steps:
            self._aborted = True

        if isinstance(intent, MoveTo) and self.position_source is not None and self.step_size > 0:
            self.position_source.advance_towards(intent.x, intent.y, self.step_size)

        if self.fail_status:
            return ActionResult(status=ActionStatus.FAILED, latency_ms=1.0, notes="failed")

        return ActionResult(status=ActionStatus.SUCCESS, latency_ms=1.0, notes="ok")

    def abort(self, reason: str) -> None:
        self._aborted = True

    def is_aborted(self) -> bool:
        return self._aborted

    def close(self) -> None:
        pass


def make_session(tmp_path: Path) -> Session:
    """Create a real Session instance in a temp directory for event testing."""
    cfg = Config(
        lab_mode=False,
        server_allowlist=("192.168.1.50:8085",),
        isolation_sentinel="127.0.0.1:65535",
        kill_switch_key="F12",
        session_root=tmp_path / "sessions",
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )
    return Session.start(cfg)


def read_session_events(session: Session) -> list[dict[str, Any]]:
    """Read all logged session events from events.jsonl."""
    events_file = session.path / "events.jsonl"
    if not events_file.exists():
        return []
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    return [json.loads(line) for line in lines if line.strip()]


# --- Acceptance Tests ---

def test_nav_config_validation_non_positive_fields() -> None:
    """NavConfig with any non-positive required field raises ValueError."""
    with pytest.raises(ValueError, match="arrival_tolerance_units must be > 0"):
        NavConfig(arrival_tolerance_units=0.0)

    with pytest.raises(ValueError, match="node_snap_tolerance_units must be > 0"):
        NavConfig(node_snap_tolerance_units=-1.0)

    with pytest.raises(ValueError, match="max_total_seconds must be > 0"):
        NavConfig(max_total_seconds=0.0)


def test_nav_config_validation_step_interval_negative() -> None:
    """NavConfig with step_interval_s < 0 raises ValueError."""
    with pytest.raises(ValueError, match="step_interval_s must be >= 0"):
        NavConfig(step_interval_s=-0.1)


def test_nav_config_validation_max_replans_negative() -> None:
    """NavConfig with max_replans < 0 raises ValueError."""
    with pytest.raises(ValueError, match="max_replans must be >= 0"):
        NavConfig(max_replans=-1)

    with pytest.raises(ValueError, match="deviation_threshold_units must be >= 0"):
        NavConfig(deviation_threshold_units=-0.5)

    with pytest.raises(ValueError, match="segment_max_steps must be >= 1"):
        NavConfig(segment_max_steps=0)


def test_nav_result_success_non_empty_reason_raises() -> None:
    """NavResult with SUCCESS and non-empty reason raises ValueError."""
    with pytest.raises(ValueError, match="status == SUCCESS implies reason == ''"):
        NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(0.0, 0.0),
            final_xy=(0.0, 0.0),
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
            reason="unexpected_reason",
        )


def test_nav_result_non_success_empty_reason_raises() -> None:
    """NavResult with non-SUCCESS and empty reason raises ValueError."""
    with pytest.raises(ValueError, match="status != SUCCESS implies non-empty reason"):
        NavResult(
            status=NavStatus.FAILED,
            target_xy=(0.0, 0.0),
            final_xy=(0.0, 0.0),
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
            reason="",
        )


def test_nav_result_frozen_and_invariant_checks() -> None:
    """NavResult is frozen and validates non-negative bounds."""
    res = NavResult(
        status=NavStatus.SUCCESS,
        target_xy=(0.0, 0.0),
        final_xy=(0.0, 0.0),
        iterations=0,
        replans=0,
        path_attempts=1,
        duration_s=0.5,
    )
    with pytest.raises(FrozenInstanceError):
        res.iterations = 5  # type: ignore[misc]

    with pytest.raises(ValueError, match="iterations must be >= 0"):
        NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(0.0, 0.0),
            final_xy=(0.0, 0.0),
            iterations=-1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
        )

    with pytest.raises(ValueError, match="duration_s must be >= 0.0"):
        NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(0.0, 0.0),
            final_xy=(0.0, 0.0),
            iterations=0,
            replans=0,
            path_attempts=1,
            duration_s=-0.1,
        )


def test_simple_replanner_snaps_both_endpoints() -> None:
    """SimpleReplanner returns a PathResult when both endpoints snap."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    replanner = SimpleReplanner()
    config = NavConfig()

    res = replanner.replan(from_xy=(0.1, 0.1), goal_xy=(9.9, 0.1), graph=graph, config=config)
    assert res is not None
    assert res.found
    assert res.node_ids == (1, 2)


def test_simple_replanner_unsnappable_goal_returns_none() -> None:
    """SimpleReplanner returns None when goal cannot be snapped."""
    graph = make_graph(
        [(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")],
        [],
    )
    replanner = SimpleReplanner()
    config = NavConfig(node_snap_tolerance_units=2.0)

    res = replanner.replan(from_xy=(0.0, 0.0), goal_xy=(100.0, 100.0), graph=graph, config=config)
    assert res is None


def test_go_to_already_at_target_returns_success() -> None:
    """go_to with target already within arrival_tolerance returns SUCCESS with iterations=0."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    pos_source = FakePositionSource((0.1, 0.1))
    actuator = FakeActuator(pos_source)

    nav = Navigator(graph, actuator, pos_source, config=NavConfig(arrival_tolerance_units=0.5))
    result = nav.go_to((0.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert result.iterations == 0
    assert result.replans == 0
    assert result.path_attempts == 0
    assert result.reason == ""


def test_go_to_unsnappable_goal_returns_hard_failure() -> None:
    """go_to with an unsnappable goal returns HARD_FAILURE with reason='goal_not_snapped'."""
    graph = make_graph([(1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source)

    nav = Navigator(graph, actuator, pos_source, config=NavConfig(node_snap_tolerance_units=2.0))
    result = nav.go_to((100.0, 100.0))

    assert result.status == NavStatus.HARD_FAILURE
    assert result.reason == "goal_not_snapped"


def test_go_to_unsnappable_start_returns_hard_failure() -> None:
    """go_to with an unsnappable start returns HARD_FAILURE with reason='start_not_snapped'."""
    graph = make_graph([(1, 10.0, 10.0, 0.0, NodeKind.WAYPOINT, "t")], [])
    pos_source = FakePositionSource((100.0, 100.0))
    actuator = FakeActuator(pos_source)

    nav = Navigator(graph, actuator, pos_source, config=NavConfig(node_snap_tolerance_units=2.0))
    result = nav.go_to((10.0, 10.0))

    assert result.status == NavStatus.HARD_FAILURE
    assert result.reason == "start_not_snapped"


def test_go_to_straight_line_two_nodes() -> None:
    """go_to on a straight-line two-node graph returns SUCCESS after walking intermediate node."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=5.0)

    nav = Navigator(graph, actuator, pos_source)
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert result.iterations == 1
    assert result.replans == 0
    assert result.path_attempts == 1
    assert result.final_xy == (10.0, 0.0)


def test_go_to_walks_multi_node_path_segment_by_segment_and_calls_hooks() -> None:
    """go_to walks multi-node path segment by segment, calling segment hooks."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (3, 20.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0), (2, 3, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=5.0)

    start_calls: list[int] = []
    end_calls: list[tuple[int, bool]] = []

    def on_start(node_id: int) -> None:
        start_calls.append(node_id)

    def on_end(node_id: int, reached: bool) -> None:
        end_calls.append((node_id, reached))

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        on_segment_start=on_start,
        on_segment_end=on_end,
    )
    result = nav.go_to((20.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert result.iterations == 2
    assert start_calls == [2, 3]
    assert end_calls == [(2, True), (3, True)]


def test_go_to_deviation_triggers_replan() -> None:
    """Position source jumping far from expected position after a segment triggers one replan and completes."""
    # Graph: 1(0,0)->2(10,0)->3(20,0) and 4(100,100)->3(20,0)
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (3, 20.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (4, 100.0, 100.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0), (2, 3, 10.0), (4, 3, 100.0)],
    )

    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=5.0)

    deviation_injected = False

    def on_segment_end(node_id: int, reached: bool) -> None:
        nonlocal deviation_injected
        if node_id == 2 and reached and not deviation_injected:
            deviation_injected = True
            pos_source.current_pos = (100.0, 100.0)

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        on_segment_end=on_segment_end,
        config=NavConfig(deviation_threshold_units=3.0, max_replans=3),
    )
    result = nav.go_to((20.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert result.replans == 1
    assert result.path_attempts == 2


def test_go_to_max_replans_exceeded() -> None:
    """Position source deviating on every segment returns HARD_FAILURE after max_replans + 1 attempts."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )

    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)

    def on_segment_end(node_id: int, reached: bool) -> None:
        if node_id == 2 and reached:
            # Shift back to 0,0 so it snaps to node 1 again but is far from node 2
            pos_source.current_pos = (0.0, 0.0)

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        on_segment_end=on_segment_end,
        config=NavConfig(deviation_threshold_units=1.0, max_replans=2),
    )
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.HARD_FAILURE
    assert result.reason == "max_replans_exceeded"
    assert result.replans == 3
    assert result.path_attempts == 3


def test_go_to_timeout_fake_clock() -> None:
    """FakeClock advancing past max_total_seconds causes TIMEOUT."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=1.0)
    clock = FakeClock(start_time=0.0)

    def stepping_clock() -> float:
        clock.advance(50.0)
        return clock.time

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        config=NavConfig(max_total_seconds=100.0),
        clock=stepping_clock,
    )
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.TIMEOUT
    assert result.reason == "max_total_seconds_exceeded"


def test_go_to_segment_failure_actuator_failed() -> None:
    """Actuator returning ActionStatus.FAILED exhausts max steps and go_to returns FAILED."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    # step_size=0.0 means position never advances
    actuator = FakeActuator(pos_source, step_size=0.0, fail_status=True)

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        config=NavConfig(segment_max_steps=5, step_interval_s=0.0),
    )
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.FAILED
    assert result.reason == "segment_failed"


def test_go_to_actuator_aborted_mid_segment() -> None:
    """Actuator becoming aborted mid-segment causes go_to to return FAILED."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=1.0, abort_after_steps=1)

    nav = Navigator(graph, actuator, pos_source, config=NavConfig(step_interval_s=0.0))
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.FAILED
    assert result.reason == "segment_failed"


def test_path_attempts_and_replans_counting() -> None:
    """path_attempts counts initial attempt + replans; replans counts only deviation replans."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)

    nav = Navigator(graph, actuator, pos_source)
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert result.path_attempts == 1
    assert result.replans == 0


def test_session_events_successful_run(tmp_path: Path) -> None:
    """With Session attached, successful go_to logs nav_started, nav_segment, nav_completed."""
    session = make_session(tmp_path)
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)

    nav = Navigator(graph, actuator, pos_source, session=session)
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    events = read_session_events(session)
    event_names = [e.get("event") for e in events]

    assert "nav_started" in event_names
    assert "nav_segment" in event_names
    assert "nav_completed" in event_names


def test_session_events_deviating_run(tmp_path: Path) -> None:
    """A deviating run logs at least one nav_replan event."""
    session = make_session(tmp_path)
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (3, 20.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (4, 100.0, 100.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0), (2, 3, 10.0), (4, 3, 100.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)

    deviation_injected = False

    def on_segment_end(node_id: int, reached: bool) -> None:
        nonlocal deviation_injected
        if node_id == 2 and reached and not deviation_injected:
            deviation_injected = True
            pos_source.current_pos = (100.0, 100.0)

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        on_segment_end=on_segment_end,
        session=session,
        config=NavConfig(deviation_threshold_units=1.0),
    )
    nav.go_to((20.0, 0.0))

    events = read_session_events(session)
    replan_events = [e for e in events if e.get("event") == "nav_replan"]
    assert len(replan_events) >= 1
    assert replan_events[0]["attempt"] == 1


def test_no_session_attached() -> None:
    """go_to works cleanly when session is None."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)

    nav = Navigator(graph, actuator, pos_source, session=None)
    result = nav.go_to((10.0, 0.0))
    assert result.status == NavStatus.SUCCESS


def test_only_injected_clock_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """Navigator uses ONLY the injected clock; monkeypatched time.monotonic raises if called."""
    def raising_monotonic() -> float:
        raise RuntimeError("real time.monotonic was called!")

    monkeypatch.setattr("time.monotonic", raising_monotonic)

    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=10.0)
    fake_clock = FakeClock(10.0)

    nav = Navigator(graph, actuator, pos_source, clock=fake_clock)
    result = nav.go_to((10.0, 0.0))
    assert result.status == NavStatus.SUCCESS


def test_only_injected_sleep_used() -> None:
    """FakeSleep records sleep calls matching the steps that did not reach target."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )
    pos_source = FakePositionSource((0.0, 0.0))
    actuator = FakeActuator(pos_source, step_size=5.0)  # Needs 2 steps to move 10.0 units
    fake_sleep = FakeSleep()

    nav = Navigator(
        graph,
        actuator,
        pos_source,
        config=NavConfig(step_interval_s=0.25),
        sleep=fake_sleep,
    )
    result = nav.go_to((10.0, 0.0))

    assert result.status == NavStatus.SUCCESS
    assert len(fake_sleep.calls) == 2
    assert fake_sleep.calls == [0.25, 0.25]


def test_determinism_identical_inputs() -> None:
    """Two go_to calls with identical scripted inputs return equal NavResult values."""
    graph = make_graph(
        [
            (1, 0.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
            (2, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        ],
        [(1, 2, 10.0)],
    )

    pos_source1 = FakePositionSource((0.0, 0.0))
    actuator1 = FakeActuator(pos_source1, step_size=5.0)
    clock1 = FakeClock(0.0)
    nav1 = Navigator(graph, actuator1, pos_source1, clock=clock1, sleep=lambda _: None)
    res1 = nav1.go_to((10.0, 0.0))

    pos_source2 = FakePositionSource((0.0, 0.0))
    actuator2 = FakeActuator(pos_source2, step_size=5.0)
    clock2 = FakeClock(0.0)
    nav2 = Navigator(graph, actuator2, pos_source2, clock=clock2, sleep=lambda _: None)
    res2 = nav2.go_to((10.0, 0.0))

    assert res1.status == res2.status
    assert res1.iterations == res2.iterations
    assert res1.replans == res2.replans
    assert res1.path_attempts == res2.path_attempts
    assert res1.reason == res2.reason
    assert res1.target_xy == res2.target_xy
    assert res1.final_xy == res2.final_xy


def test_nav_result_to_json() -> None:
    """NavResult.to_json is JSON-serializable and includes all documented fields."""
    res = NavResult(
        status=NavStatus.SUCCESS,
        target_xy=(10.0, 20.0),
        final_xy=(10.0, 20.0),
        iterations=2,
        replans=0,
        path_attempts=1,
        duration_s=1.23,
        reason="",
    )
    json_dict = res.to_json()
    dumped = json.dumps(json_dict)
    loaded = json.loads(dumped)

    assert loaded["status"] == "success"
    assert loaded["target_xy"] == [10.0, 20.0]
    assert loaded["final_xy"] == [10.0, 20.0]
    assert loaded["iterations"] == 2
    assert loaded["replans"] == 0
    assert loaded["path_attempts"] == 1
    assert loaded["duration_s"] == 1.23
    assert loaded["reason"] == ""


def test_static_ast_import_isolation() -> None:
    """navigator.py does not import prohibited modules."""
    filepath = Path("src/wow_bot/nav/navigator.py")
    tree = ast.parse(filepath.read_text(encoding="utf-8"), filename=str(filepath))

    forbidden_exact = {
        "wow_bot.strategist",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.world",
        "wow_bot.executor",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in forbidden_exact, f"Forbidden module imported: {mod}"
                assert not any(
                    kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
                ), f"Forbidden LLM module imported: {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert mod not in forbidden_exact, f"Forbidden module imported: {mod}"
            assert not any(
                kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
            ), f"Forbidden LLM module imported: {mod}"
