"""Unit tests for flee logic in wow_bot.combat.flee."""

import ast
import json
import time
from dataclasses import FrozenInstanceError, dataclass, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any

import pytest

from wow_bot.combat.flee import (
    FleeConfig,
    FleeController,
    FleeDecision,
    FleeResult,
    FleeStatus,
    FleeTrigger,
)
from wow_bot.config import Config
from wow_bot.nav.graph import GraphEdge, GraphNode, NavGraph, NodeKind
from wow_bot.nav.navigator import NavResult, NavStatus
from wow_bot.session import Session


@dataclass
class FakeFleeState:
    """Local test state view implementing FleeStateView."""

    self_hp_percent: float = 100.0
    self_x: float = 0.0
    self_y: float = 0.0
    current_target_id: str | None = None
    adds_count: int = 0
    target_x: float | None = None
    target_y: float | None = None


class FakeWorld:
    """Minimal stub for WorldModel.

    WorldModel is not directly queried by FleeController (decide reads the NavGraph directly).
    """


class FakeNavigator:
    """Local mock navigator recording go_to calls and returning configurable NavResults."""

    def __init__(
        self,
        result: NavResult | Exception | None = None,
        on_go_to: Any = None,
    ) -> None:
        self.result = result
        self.calls: list[tuple[float, float]] = []
        self.on_go_to = on_go_to

    def go_to(self, target_xy: tuple[float, float]) -> NavResult:
        self.calls.append(target_xy)
        if self.on_go_to is not None:
            self.on_go_to(target_xy)
        if isinstance(self.result, Exception):
            raise self.result
        if self.result is not None:
            return self.result
        return NavResult(
            status=NavStatus.SUCCESS,
            target_xy=target_xy,
            final_xy=target_xy,
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
        )


class FakeClock:
    """Controllable monotonic clock for testing time progression."""

    def __init__(self, start: float = 1000.0) -> None:
        self.time = start

    def __call__(self) -> float:
        return self.time

    def advance(self, delta: float) -> None:
        self.time += delta


def make_graph(
    nodes: list[tuple[int, float, float, float, NodeKind, str]],
    edges: list[tuple[int, int, float]] | None = None,
) -> NavGraph:
    """Builder helper constructing a NavGraph directly."""
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
    if edges is not None:
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


# --- FleeConfig tests ---


def test_flee_config_validation_hp_threshold() -> None:
    """FleeConfig with hp_threshold outside [0, 100] raises ValueError."""
    with pytest.raises(ValueError, match="hp_threshold must be between 0.0 and 100.0"):
        FleeConfig(hp_threshold=-0.1)
    with pytest.raises(ValueError, match="hp_threshold must be between 0.0 and 100.0"):
        FleeConfig(hp_threshold=100.1)


def test_flee_config_validation_adds_threshold() -> None:
    """FleeConfig with adds_threshold < 1 raises ValueError."""
    with pytest.raises(ValueError, match="adds_threshold must be >= 1"):
        FleeConfig(adds_threshold=0)


def test_flee_config_validation_search_radius() -> None:
    """FleeConfig with non-positive search_radius_units raises ValueError."""
    with pytest.raises(ValueError, match="search_radius_units must be > 0"):
        FleeConfig(search_radius_units=0.0)
    with pytest.raises(ValueError, match="search_radius_units must be > 0"):
        FleeConfig(search_radius_units=-5.0)


def test_flee_config_validation_min_distance_from_target() -> None:
    """FleeConfig with negative min_distance_from_target_units raises ValueError."""
    with pytest.raises(
        ValueError, match="min_distance_from_target_units must be >= 0"
    ):
        FleeConfig(min_distance_from_target_units=-1.0)


def test_flee_config_validation_candidates_limit() -> None:
    """FleeConfig with candidates_limit < 1 raises ValueError."""
    with pytest.raises(ValueError, match="candidates_limit must be >= 1"):
        FleeConfig(candidates_limit=0)


def test_flee_config_validation_max_flee_seconds() -> None:
    """FleeConfig with non-positive max_flee_seconds raises ValueError."""
    with pytest.raises(ValueError, match="max_flee_seconds must be > 0"):
        FleeConfig(max_flee_seconds=0.0)


def test_flee_config_validation_fallback_direction_units() -> None:
    """FleeConfig with non-positive fallback_direction_units raises ValueError."""
    with pytest.raises(ValueError, match="fallback_direction_units must be > 0"):
        FleeConfig(fallback_direction_units=-10.0)


def test_flee_config_frozen() -> None:
    """FleeConfig is immutable."""
    cfg = FleeConfig()
    with pytest.raises(FrozenInstanceError):
        cfg.hp_threshold = 10.0  # type: ignore[misc]


# --- FleeDecision & FleeResult Invariant Tests ---


def test_flee_decision_invariants() -> None:
    """FleeDecision enforces invariants in __post_init__."""
    # NONE with target raises ValueError
    with pytest.raises(ValueError, match="trigger == NONE implies target_xy is None"):
        FleeDecision(trigger=FleeTrigger.NONE, target_xy=(1.0, 2.0), reason="test")

    # Non-NONE with None target raises ValueError
    with pytest.raises(ValueError, match="trigger != NONE implies target_xy is not None"):
        FleeDecision(trigger=FleeTrigger.LOW_HP, target_xy=None, reason="test")

    # Empty reason raises ValueError
    with pytest.raises(ValueError, match="reason is a non-empty string"):
        FleeDecision(trigger=FleeTrigger.LOW_HP, target_xy=(1.0, 2.0), reason="")


def test_flee_result_invariants() -> None:
    """FleeResult enforces invariants in __post_init__."""
    # NOT_TRIGGERED with non-NONE trigger
    with pytest.raises(ValueError, match="status == NOT_TRIGGERED implies"):
        FleeResult(
            status=FleeStatus.NOT_TRIGGERED,
            trigger=FleeTrigger.LOW_HP,
            target_xy=None,
            navigator_status="",
            duration_s=0.0,
            reason="not_triggered",
        )

    # COMPLETED with None target
    with pytest.raises(ValueError, match="status == STARTED or COMPLETED implies"):
        FleeResult(
            status=FleeStatus.COMPLETED,
            trigger=FleeTrigger.LOW_HP,
            target_xy=None,
            navigator_status="success",
            duration_s=1.0,
            reason="",
        )

    # FAILED with empty reason
    with pytest.raises(ValueError, match="status == FAILED implies reason != ''"):
        FleeResult(
            status=FleeStatus.FAILED,
            trigger=FleeTrigger.LOW_HP,
            target_xy=(1.0, 2.0),
            navigator_status="failed",
            duration_s=1.0,
            reason="",
        )

    # Negative duration
    with pytest.raises(ValueError, match="duration_s must be >= 0.0"):
        FleeResult(
            status=FleeStatus.FAILED,
            trigger=FleeTrigger.LOW_HP,
            target_xy=(1.0, 2.0),
            navigator_status="failed",
            duration_s=-0.5,
            reason="err",
        )


# --- should_flee tests ---


def test_should_flee_low_hp() -> None:
    """should_flee returns LOW_HP when hp is below or equal to threshold."""
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
        config=FleeConfig(hp_threshold=25.0),
    )

    state = FakeFleeState(self_hp_percent=20.0)
    assert controller.should_flee(state) == FleeTrigger.LOW_HP

    state_boundary = FakeFleeState(self_hp_percent=25.0)
    assert controller.should_flee(state_boundary) == FleeTrigger.LOW_HP


def test_should_flee_low_hp_priority_over_adds() -> None:
    """should_flee returns LOW_HP even when adds_count is also above threshold."""
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
        config=FleeConfig(hp_threshold=25.0, adds_threshold=3),
    )

    state = FakeFleeState(self_hp_percent=10.0, adds_count=5)
    assert controller.should_flee(state) == FleeTrigger.LOW_HP


def test_should_flee_too_many_adds() -> None:
    """should_flee returns TOO_MANY_ADDS when hp is fine and adds count >= threshold."""
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
        config=FleeConfig(hp_threshold=25.0, adds_threshold=3),
    )

    state = FakeFleeState(self_hp_percent=80.0, adds_count=3)
    assert controller.should_flee(state) == FleeTrigger.TOO_MANY_ADDS


def test_should_flee_none() -> None:
    """should_flee returns NONE when hp and adds count are within safe limits."""
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
        config=FleeConfig(hp_threshold=25.0, adds_threshold=3),
    )

    state = FakeFleeState(self_hp_percent=80.0, adds_count=1)
    assert controller.should_flee(state) == FleeTrigger.NONE


# --- decide tests ---


def test_decide_none_trigger() -> None:
    """decide returns NONE trigger when should_flee is NONE."""
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
    )
    state = FakeFleeState(self_hp_percent=100.0, adds_count=0)
    decision = controller.decide(state)
    assert decision.trigger == FleeTrigger.NONE
    assert decision.target_xy is None
    assert decision.reason == "not_triggered"


def test_decide_picks_closest_waypoint_kind() -> None:
    """decide picks the closest WAYPOINT node within search_radius_units when require_waypoint_kind is True."""
    nodes = [
        (1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (3, 50.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(search_radius_units=60.0, require_waypoint_kind=True),
    )
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)
    decision = controller.decide(state)

    assert decision.trigger == FleeTrigger.LOW_HP
    assert decision.target_xy == (5.0, 0.0)
    assert decision.reason == "waypoint:2"


def test_decide_ignores_non_waypoint_nodes_when_required() -> None:
    """decide ignores non-waypoint nodes when require_waypoint_kind is True."""
    nodes = [
        (1, 2.0, 0.0, 0.0, NodeKind.VENDOR, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(require_waypoint_kind=True),
    )
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)
    decision = controller.decide(state)

    assert decision.target_xy == (5.0, 0.0)
    assert decision.reason == "waypoint:2"


def test_decide_includes_non_waypoint_nodes_when_require_waypoint_kind_false() -> None:
    """decide includes non-waypoint nodes when require_waypoint_kind is False."""
    nodes = [
        (1, 2.0, 0.0, 0.0, NodeKind.VENDOR, "t"),
        (2, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(require_waypoint_kind=False),
    )
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)
    decision = controller.decide(state)

    assert decision.target_xy == (2.0, 0.0)
    assert decision.reason == "waypoint:1"


def test_decide_excludes_nodes_too_close_to_target() -> None:
    """decide excludes nodes within min_distance_from_target_units of the target."""
    nodes = [
        (1, 5.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),   # Dist to target (10, 0) is 5.0 (< 15.0) -> excluded
        (2, -10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"), # Dist to target (10, 0) is 20.0 (>= 15.0) -> kept
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(min_distance_from_target_units=15.0),
    )
    state = FakeFleeState(
        self_hp_percent=10.0,
        self_x=0.0,
        self_y=0.0,
        target_x=10.0,
        target_y=0.0,
    )
    decision = controller.decide(state)

    assert decision.target_xy == (-10.0, 0.0)
    assert decision.reason == "waypoint:2"


def test_decide_tie_breaks_by_node_id_asc() -> None:
    """decide tie-breaks by node ID ASC when two candidates are equidistant."""
    nodes = [
        (20, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
        (10, -10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t"),
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
    )
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)
    decision = controller.decide(state)

    assert decision.target_xy == (-10.0, 0.0)
    assert decision.reason == "waypoint:10"


def test_decide_candidates_limit() -> None:
    """decide takes at most candidates_limit candidates and chooses the top ranked one."""
    nodes = [
        (i, float(i), 0.0, 0.0, NodeKind.WAYPOINT, "t") for i in range(1, 20)
    ]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(candidates_limit=3),
    )
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)
    decision = controller.decide(state)

    # Closest nodes are 1, 2, 3. Node 1 is chosen.
    assert decision.target_xy == (1.0, 0.0)
    assert decision.reason == "waypoint:1"


def test_decide_fallback_with_known_target() -> None:
    """decide computes fallback point away from target when no waypoint candidate exists."""
    graph = make_graph([])
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(fallback_direction_units=20.0),
    )
    # Self at (0, 0), target at (10, 0). Vector away from target is (-1, 0).
    state = FakeFleeState(
        self_hp_percent=10.0,
        self_x=0.0,
        self_y=0.0,
        target_x=10.0,
        target_y=0.0,
    )
    decision = controller.decide(state)

    assert decision.trigger == FleeTrigger.LOW_HP
    assert decision.target_xy == (-20.0, 0.0)
    assert decision.reason == "fallback_point"


def test_decide_fallback_with_unknown_target() -> None:
    """decide computes fallback point along -x when target coordinates are unknown."""
    graph = make_graph([])
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
        config=FleeConfig(fallback_direction_units=20.0),
    )
    state = FakeFleeState(
        self_hp_percent=10.0,
        self_x=5.0,
        self_y=10.0,
        target_x=None,
        target_y=None,
    )
    decision = controller.decide(state)

    assert decision.trigger == FleeTrigger.LOW_HP
    assert decision.target_xy == (-15.0, 10.0)  # 5.0 - 20.0 = -15.0
    assert decision.reason == "fallback_point"


def test_decide_determinism_across_100_runs() -> None:
    """decide produces identical output across 100 consecutive calls."""
    nodes = [(i, float(i), 0.0, 0.0, NodeKind.WAYPOINT, "t") for i in range(1, 5)]
    graph = make_graph(nodes)
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
    )
    state = FakeFleeState(self_hp_percent=10.0)

    first = controller.decide(state)
    for _ in range(100):
        current = controller.decide(state)
        assert current == first


def test_decide_non_mutability() -> None:
    """decide does not mutate graph or state views."""
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    graph = make_graph(nodes)
    state = FakeFleeState(self_hp_percent=10.0, self_x=0.0, self_y=0.0)

    graph_nodes_before = dict(graph.nodes)
    state_before = replace(state)

    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=graph,
    )
    _ = controller.decide(state)

    assert dict(graph.nodes) == graph_nodes_before
    assert state == state_before


# --- execute tests ---


def test_execute_not_triggered() -> None:
    """execute with should_flee NONE returns NOT_TRIGGERED and does not call go_to."""
    fake_nav = FakeNavigator()
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph([]),
    )
    state = FakeFleeState(self_hp_percent=100.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.NOT_TRIGGERED
    assert res.trigger == FleeTrigger.NONE
    assert res.target_xy is None
    assert res.navigator_status == ""
    assert res.duration_s == 0.0
    assert res.reason == "not_triggered"
    assert len(fake_nav.calls) == 0


def test_execute_successful_navigation() -> None:
    """execute with a successful navigator returns COMPLETED and navigator_status='success'."""
    fake_nav = FakeNavigator(
        result=NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(10.0, 0.0),
            final_xy=(10.0, 0.0),
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=0.5,
        )
    )
    clock = FakeClock(1000.0)
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
        clock=clock,
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.COMPLETED
    assert res.trigger == FleeTrigger.LOW_HP
    assert res.target_xy == (10.0, 0.0)
    assert res.navigator_status == "success"
    assert res.reason == ""
    assert fake_nav.calls == [(10.0, 0.0)]


def test_execute_failed_navigator() -> None:
    """execute with a FAILED navigator returns FAILED and reason from NavResult."""
    fake_nav = FakeNavigator(
        result=NavResult(
            status=NavStatus.FAILED,
            target_xy=(10.0, 0.0),
            final_xy=(2.0, 0.0),
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=0.5,
            reason="segment_failed",
        )
    )
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.FAILED
    assert res.navigator_status == "failed"
    assert res.reason == "segment_failed"


def test_execute_hard_failure_navigator() -> None:
    """execute with a HARD_FAILURE navigator returns FAILED."""
    fake_nav = FakeNavigator(
        result=NavResult(
            status=NavStatus.HARD_FAILURE,
            target_xy=(10.0, 0.0),
            final_xy=(0.0, 0.0),
            iterations=0,
            replans=0,
            path_attempts=1,
            duration_s=0.1,
            reason="goal_not_snapped",
        )
    )
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.FAILED
    assert res.navigator_status == "hard_failure"
    assert res.reason == "goal_not_snapped"


def test_execute_timeout_navigator() -> None:
    """execute with a TIMEOUT navigator returns FAILED."""
    fake_nav = FakeNavigator(
        result=NavResult(
            status=NavStatus.TIMEOUT,
            target_xy=(10.0, 0.0),
            final_xy=(5.0, 0.0),
            iterations=2,
            replans=1,
            path_attempts=2,
            duration_s=30.0,
            reason="max_total_seconds_exceeded",
        )
    )
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.FAILED
    assert res.navigator_status == "timeout"
    assert res.reason == "max_total_seconds_exceeded"


def test_execute_navigator_exception() -> None:
    """execute with a navigator raising an exception returns FAILED with reason starting 'navigator_error:'."""
    fake_nav = FakeNavigator(result=RuntimeError("connection_lost"))
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.FAILED
    assert res.navigator_status == ""
    assert res.reason == "navigator_error:RuntimeError"


def test_execute_max_flee_seconds_exceeded() -> None:
    """execute with duration exceeding max_flee_seconds overrides status to FAILED with reason='flee_timeout'."""
    clock = FakeClock(1000.0)

    def _slow_go_to(_target_xy: tuple[float, float]) -> None:
        clock.advance(35.0)  # Exceeds max_flee_seconds=30.0

    fake_nav = FakeNavigator(
        result=NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(10.0, 0.0),
            final_xy=(10.0, 0.0),
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=35.0,
        ),
        on_go_to=_slow_go_to,
    )
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=fake_nav,  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
        config=FleeConfig(max_flee_seconds=30.0),
        clock=clock,
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.FAILED
    assert res.reason == "flee_timeout"
    assert res.duration_s == 35.0


def make_test_config(session_root: Path) -> Config:
    """Helper constructing a valid Config instance."""
    return Config(
        lab_mode=False,
        server_allowlist=("127.0.0.1:8080",),
        isolation_sentinel="10.0.0.1:80",
        kill_switch_key="F12",
        session_root=session_root,
        dry_run=True,
        max_session_seconds=3600,
        log_level="INFO",
    )


def test_session_events(tmp_path: Path) -> None:
    """Verify session events flee_started, flee_completed, and flee_failed are emitted with correct structure."""
    cfg = make_test_config(tmp_path)

    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    graph = make_graph(nodes)

    # Test flee_started and flee_completed on success
    with Session.start(cfg) as session:
        events_path = session.path / "events.jsonl"

        def _verify_started_event(_target_xy: tuple[float, float]) -> None:
            # Verify flee_started was emitted before go_to completes
            lines = events_path.read_text(encoding="utf-8").strip().splitlines()
            events = [json.loads(line) for line in lines]
            started_events = [e for e in events if e.get("event") == "flee_started"]
            assert len(started_events) == 1
            assert started_events[0]["trigger"] == "low_hp"
            assert started_events[0]["target_xy"] == [10.0, 0.0]

        fake_nav = FakeNavigator(on_go_to=_verify_started_event)
        controller = FleeController(
            navigator=fake_nav,  # type: ignore[arg-type]
            world=FakeWorld(),  # type: ignore[arg-type]
            graph=graph,
            session=session,
        )
        state = FakeFleeState(self_hp_percent=10.0)
        res = controller.execute(state)
        assert res.status == FleeStatus.COMPLETED

        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        completed_events = [e for e in events if e.get("event") == "flee_completed"]
        assert len(completed_events) == 1
        assert completed_events[0]["navigator_status"] == "success"
        assert completed_events[0]["target_xy"] == [10.0, 0.0]

    # Test flee_failed on navigator failure
    with Session.start(cfg) as session_failed:
        events_failed_path = session_failed.path / "events.jsonl"
        fake_nav_fail = FakeNavigator(
            result=NavResult(
                status=NavStatus.FAILED,
                target_xy=(10.0, 0.0),
                final_xy=(0.0, 0.0),
                iterations=0,
                replans=0,
                path_attempts=1,
                duration_s=0.5,
                reason="stuck",
            )
        )
        controller_failed = FleeController(
            navigator=fake_nav_fail,  # type: ignore[arg-type]
            world=FakeWorld(),  # type: ignore[arg-type]
            graph=graph,
            session=session_failed,
        )
        res_fail = controller_failed.execute(state)
        assert res_fail.status == FleeStatus.FAILED

        lines = events_failed_path.read_text(encoding="utf-8").strip().splitlines()
        events = [json.loads(line) for line in lines]
        failed_events = [e for e in events if e.get("event") == "flee_failed"]
        assert len(failed_events) == 1
        assert failed_events[0]["navigator_status"] == "failed"
        assert failed_events[0]["reason"] == "stuck"


def test_no_session_attached() -> None:
    """execute works without raising when session is None."""
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
        session=None,
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)
    assert res.status == FleeStatus.COMPLETED


def test_clock_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    """execute uses ONLY the injected clock; time.monotonic read raises."""
    def _raising_monotonic() -> float:
        raise RuntimeError("time.monotonic call forbidden")

    monkeypatch.setattr(time, "monotonic", _raising_monotonic)

    clock = FakeClock(500.0)
    nodes = [(1, 10.0, 0.0, 0.0, NodeKind.WAYPOINT, "t")]
    controller = FleeController(
        navigator=FakeNavigator(),  # type: ignore[arg-type]
        world=FakeWorld(),  # type: ignore[arg-type]
        graph=make_graph(nodes),
        clock=clock,
    )
    state = FakeFleeState(self_hp_percent=10.0)
    res = controller.execute(state)

    assert res.status == FleeStatus.COMPLETED


def test_flee_result_to_json() -> None:
    """FleeResult.to_json is JSON-serializable and matches expected schema."""
    res = FleeResult(
        status=FleeStatus.COMPLETED,
        trigger=FleeTrigger.LOW_HP,
        target_xy=(10.0, 20.0),
        navigator_status="success",
        duration_s=2.5,
        reason="",
    )
    data = res.to_json()
    dumped = json.dumps(data)
    loaded = json.loads(dumped)

    assert loaded == {
        "status": "completed",
        "trigger": "low_hp",
        "target_xy": [10.0, 20.0],
        "navigator_status": "success",
        "duration_s": 2.5,
        "reason": "",
    }


def test_static_ast_import_isolation() -> None:
    """Static AST check verifying flee.py does not import prohibited modules."""
    with open("src/wow_bot/combat/flee.py", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename="src/wow_bot/combat/flee.py")

    prohibited_exact = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation.actuator",
        "wow_bot.executor",
        "aiosqlite",
        "asyncio",
        "threading",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                assert mod not in prohibited_exact, f"Prohibited module imported: {mod}"
                assert not any(
                    kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
                ), f"Prohibited LLM module imported: {mod}"
        elif isinstance(node, ast.ImportFrom) and node.module:
            mod = node.module
            assert mod not in prohibited_exact, f"Prohibited module imported: {mod}"
            assert not any(
                kw in mod.lower() for kw in ("ollama", "openai", "anthropic", "llm")
            ), f"Prohibited LLM module imported: {mod}"
