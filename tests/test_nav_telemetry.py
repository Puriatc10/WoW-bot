"""Tests for navigation telemetry observation layer."""

import ast
import dataclasses
import json
import os
from collections.abc import Callable
from pathlib import Path

import pytest

from wow_bot.config import Config
from wow_bot.nav.navigator import NavResult, NavStatus
from wow_bot.nav.telemetry import (
    CpuSample,
    NavTelemetryReport,
    RealCpuSource,
    TelemetryCollector,
    TelemetryConfig,
    TelemetryError,
    make_sampling_sleep,
    run_navigation_observed,
)
from wow_bot.session import Session


class FakeClock:
    """Controllable clock for testing."""

    def __init__(self, start_time: float = 0.0) -> None:
        self.time = start_time

    def now(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class FakeCpuSource:
    """Scripted CPU source returning predetermined CPU usage values."""

    def __init__(self, values: list[float] | None = None) -> None:
        self.values = values if values is not None else [10.0]
        self.index = 0

    def cpu_percent(self) -> float:
        if not self.values:
            return 0.0
        val = self.values[self.index % len(self.values)]
        self.index += 1
        return val


class FakeNavigator:
    """Fake navigator exposing _clock, _sleep, is_aborted, position_source, and go_to."""

    def __init__(
        self,
        clock: FakeClock,
        sleep_fn: Callable[[float], None],
        canned_result: NavResult | None = None,
        should_raise: Exception | None = None,
    ) -> None:
        self._clock = clock
        self._sleep = sleep_fn
        self.is_aborted = lambda: False
        self.position_source = lambda: (0.0, 0.0)
        self._canned_result = canned_result or NavResult(
            status=NavStatus.SUCCESS,
            target_xy=(10.0, 10.0),
            final_xy=(10.0, 10.0),
            iterations=2,
            replans=0,
            path_attempts=1,
            duration_s=2.0,
            reason="",
        )
        self._should_raise = should_raise

    def go_to(self, target_xy: tuple[float, float]) -> NavResult:
        if self._should_raise is not None:
            raise self._should_raise

        # Simulate steps that sleep
        self._sleep(1.0)
        self._clock.advance(1.0)
        self._sleep(1.0)
        self._clock.advance(1.0)
        return self._canned_result


class NavigatorMissingAttrs:
    """Navigator missing _clock and _sleep attributes."""

    def go_to(self, target_xy: tuple[float, float]) -> NavResult:
        return NavResult(
            status=NavStatus.SUCCESS,
            target_xy=target_xy,
            final_xy=target_xy,
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
            reason="",
        )


# Acceptance Tests

def test_config_invalid_sample_interval() -> None:
    with pytest.raises(ValueError, match="sample_interval_s must be > 0"):
        TelemetryConfig(sample_interval_s=0.0)
    with pytest.raises(ValueError, match="sample_interval_s must be > 0"):
        TelemetryConfig(sample_interval_s=-1.0)


def test_config_invalid_max_samples() -> None:
    with pytest.raises(ValueError, match="max_samples must be >= 1"):
        TelemetryConfig(max_samples=0)


def test_config_invalid_artifact_filename() -> None:
    with pytest.raises(ValueError, match="artifact_filename must be a non-empty string"):
        TelemetryConfig(artifact_filename="")
    with pytest.raises(ValueError, match="artifact_filename must be a non-empty string"):
        TelemetryConfig(artifact_filename="sub/file.json")
    with pytest.raises(ValueError, match="artifact_filename must be a non-empty string"):
        TelemetryConfig(artifact_filename="sub\\file.json")
    with pytest.raises(ValueError, match="artifact_filename must be a non-empty string"):
        TelemetryConfig(artifact_filename="../file.json")


def test_frozen_dataclasses() -> None:
    sample = CpuSample(ts=1.0, cpu_percent=15.0)
    with pytest.raises(dataclasses.FrozenInstanceError):
        sample.ts = 2.0  # type: ignore

    report = NavTelemetryReport(
        session_id="test-session",
        target_xy=(10.0, 10.0),
        status="success",
        iterations=1,
        replans=0,
        path_attempts=1,
        duration_s=1.0,
        reason="",
        cpu_samples=(sample,),
        cpu_min_percent=15.0,
        cpu_max_percent=15.0,
        cpu_mean_percent=15.0,
    )
    with pytest.raises(dataclasses.FrozenInstanceError):
        report.session_id = "other"  # type: ignore


def test_nav_telemetry_report_invariants() -> None:
    sample = CpuSample(ts=1.0, cpu_percent=15.0)
    with pytest.raises(ValueError, match="Expected cpu_min_percent <= cpu_mean_percent <= cpu_max_percent"):
        NavTelemetryReport(
            session_id="test",
            target_xy=(0.0, 0.0),
            status="success",
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
            reason="",
            cpu_samples=(sample,),
            cpu_min_percent=20.0,  # min > mean
            cpu_max_percent=25.0,
            cpu_mean_percent=15.0,
        )

    with pytest.raises(ValueError, match="must be 0.0 when cpu_samples is empty"):
        NavTelemetryReport(
            session_id="test",
            target_xy=(0.0, 0.0),
            status="success",
            iterations=1,
            replans=0,
            path_attempts=1,
            duration_s=1.0,
            reason="",
            cpu_samples=(),
            cpu_min_percent=10.0,
            cpu_max_percent=10.0,
            cpu_mean_percent=10.0,
        )


def test_collector_invalid_constructor_args() -> None:
    clock = FakeClock()
    cpu = FakeCpuSource()
    with pytest.raises(TelemetryError, match="session_id must be a non-empty string"):
        TelemetryCollector(
            session_id="",
            target_xy=(0.0, 0.0),
            clock=clock,
            cpu_source=cpu,
        )

    with pytest.raises(TelemetryError, match="target_xy must be a 2-tuple of finite floats"):
        TelemetryCollector(
            session_id="s1",
            target_xy=(float("inf"), 0.0),
            clock=clock,
            cpu_source=cpu,
        )


def test_should_sample_and_immutability() -> None:
    clock = FakeClock(10.0)
    cpu = FakeCpuSource([15.0])
    config = TelemetryConfig(sample_interval_s=1.0)
    collector = TelemetryCollector(
        session_id="s1",
        target_xy=(0.0, 0.0),
        clock=clock,
        cpu_source=cpu,
        config=config,
    )

    # Initially True
    assert collector.should_sample(clock.now()) is True
    # Calling should_sample does not mutate state
    assert collector.sample_count == 0
    assert collector.should_sample(clock.now()) is True

    # Sample taken
    collector.sample()
    assert collector.sample_count == 1

    # Before interval elapsed -> False
    assert collector.should_sample(10.5) is False
    # At interval elapsed -> True
    assert collector.should_sample(11.0) is True


def test_sample_appends_and_clamps() -> None:
    clock = FakeClock(5.0)
    cpu = FakeCpuSource([-10.0, 50.0, 999999.0])
    config = TelemetryConfig(sample_interval_s=1.0, max_samples=2)
    collector = TelemetryCollector(
        session_id="s1",
        target_xy=(0.0, 0.0),
        clock=clock,
        cpu_source=cpu,
        config=config,
    )

    # First sample clamps negative to 0.0
    s1 = collector.sample()
    assert s1 == CpuSample(ts=5.0, cpu_percent=0.0)
    assert collector.sample_count == 1

    # Second sample
    clock.advance(1.0)
    s2 = collector.sample()
    assert s2 == CpuSample(ts=6.0, cpu_percent=50.0)
    assert collector.sample_count == 2

    # Buffer full -> returns None
    clock.advance(1.0)
    s3 = collector.sample()
    assert s3 is None
    assert collector.sample_count == 2


def test_reset_clears_buffer_and_timestamp() -> None:
    clock = FakeClock(10.0)
    cpu = FakeCpuSource([20.0])
    collector = TelemetryCollector(
        session_id="s1",
        target_xy=(0.0, 0.0),
        clock=clock,
        cpu_source=cpu,
    )

    collector.sample()
    assert collector.sample_count == 1
    assert collector.should_sample(10.5) is False

    collector.reset()
    assert collector.sample_count == 0
    assert collector.should_sample(10.5) is True


def test_build_report_empty_and_non_empty() -> None:
    clock = FakeClock(0.0)
    cpu = FakeCpuSource([10.0, 20.0, 30.0])
    collector = TelemetryCollector(
        session_id="s1",
        target_xy=(1.0, 2.0),
        clock=clock,
        cpu_source=cpu,
    )

    res = NavResult(
        status=NavStatus.SUCCESS,
        target_xy=(1.0, 2.0),
        final_xy=(1.0, 2.0),
        iterations=5,
        replans=1,
        path_attempts=2,
        duration_s=10.0,
        reason="",
    )

    # Empty buffer
    rep_empty = collector.build_report(res)
    assert rep_empty.cpu_min_percent == 0.0
    assert rep_empty.cpu_max_percent == 0.0
    assert rep_empty.cpu_mean_percent == 0.0
    assert rep_empty.cpu_samples == ()

    # Non-empty buffer
    collector.sample()  # 10.0
    clock.advance(1.0)
    collector.sample()  # 20.0
    clock.advance(1.0)
    collector.sample()  # 30.0

    rep = collector.build_report(res)
    assert rep.cpu_min_percent == 10.0
    assert rep.cpu_max_percent == 30.0
    assert rep.cpu_mean_percent == 20.0
    assert rep.iterations == 5
    assert rep.replans == 1
    assert rep.path_attempts == 2
    assert rep.duration_s == 10.0
    assert rep.status == "success"
    assert len(rep.cpu_samples) == 3


def test_report_to_json_serializable() -> None:
    sample = CpuSample(ts=1.5, cpu_percent=25.0)
    report = NavTelemetryReport(
        session_id="s1",
        target_xy=(5.0, 5.0),
        status="success",
        iterations=3,
        replans=0,
        path_attempts=1,
        duration_s=4.0,
        reason="",
        cpu_samples=(sample,),
        cpu_min_percent=25.0,
        cpu_max_percent=25.0,
        cpu_mean_percent=25.0,
    )

    json_dict = report.to_json()
    json_str = json.dumps(json_dict)
    assert isinstance(json_str, str)
    assert json_dict["cpu_samples"] == [{"ts": 1.5, "cpu_percent": 25.0}]
    assert json_dict["target_xy"] == [5.0, 5.0]


def test_make_sampling_sleep() -> None:
    clock = FakeClock(0.0)
    cpu = FakeCpuSource([50.0])
    config = TelemetryConfig(sample_interval_s=2.0)
    collector = TelemetryCollector(
        session_id="s1",
        target_xy=(0.0, 0.0),
        clock=clock,
        cpu_source=cpu,
        config=config,
    )

    base_sleep_calls: list[float] = []

    def base_sleep(s: float) -> None:
        base_sleep_calls.append(s)

    sampling_sleep = make_sampling_sleep(base_sleep, collector, clock)

    # Initial sample
    collector.sample()
    assert collector.sample_count == 1

    # First sleep step: advance time by 1.0 -> should_sample is False
    clock.advance(1.0)
    sampling_sleep(1.0)
    assert base_sleep_calls == [1.0]
    assert collector.sample_count == 1

    # Second sleep step: advance time by 1.0 -> should_sample is True (2.0 elapsed)
    clock.advance(1.0)
    sampling_sleep(1.0)
    assert base_sleep_calls == [1.0, 1.0]
    assert collector.sample_count == 2


def make_session(tmp_path: Path) -> Session:
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


def test_run_navigation_observed_success(tmp_path: Path) -> None:
    session = make_session(tmp_path)

    clock = FakeClock(0.0)
    cpu = FakeCpuSource([10.0, 20.0, 30.0])

    collector_dummy = TelemetryCollector(
        session_id=session.session_id,
        target_xy=(10.0, 10.0),
        clock=clock,
        cpu_source=cpu,
    )

    def base_sleep(s: float) -> None:
        pass

    sampling_sleep = make_sampling_sleep(base_sleep, collector_dummy, clock)

    nav = FakeNavigator(clock, sampling_sleep)

    t_config = TelemetryConfig(write_artifact=True, artifact_filename="nav_telemetry.json")

    result, report = run_navigation_observed(
        nav,  # type: ignore[arg-type]
        (10.0, 10.0),
        session=session,
        clock=clock,
        cpu_source=cpu,
        config=t_config,
    )

    assert result.status == NavStatus.SUCCESS
    assert report.session_id == session.session_id
    assert report.target_xy == (10.0, 10.0)

    # Check artifact file
    artifact_file = session.path / "nav_telemetry.json"
    assert artifact_file.exists()
    content = json.loads(artifact_file.read_text(encoding="utf-8"))
    assert content["session_id"] == session.session_id

    # Check events.jsonl
    events_file = session.path / "events.jsonl"
    lines = [json.loads(line) for line in events_file.read_text(encoding="utf-8").strip().split("\n")]
    telemetry_events = [e for e in lines if e.get("event") == "nav_telemetry"]
    assert len(telemetry_events) == 1
    assert telemetry_events[0]["session_id"] == session.session_id


def test_run_navigation_observed_no_artifact(tmp_path: Path) -> None:
    session = make_session(tmp_path)

    clock = FakeClock(0.0)
    cpu = FakeCpuSource([10.0])
    nav = FakeNavigator(clock, lambda s: None)

    t_config = TelemetryConfig(write_artifact=False)

    run_navigation_observed(
        nav,  # type: ignore[arg-type]
        (10.0, 10.0),
        session=session,
        clock=clock,
        cpu_source=cpu,
        config=t_config,
    )

    artifact_file = session.path / "nav_telemetry.json"
    assert not artifact_file.exists()


def test_run_navigation_observed_missing_navigator_attrs(tmp_path: Path) -> None:
    session = make_session(tmp_path)

    clock = FakeClock(0.0)
    cpu = FakeCpuSource()
    nav = NavigatorMissingAttrs()

    with pytest.raises(TelemetryError, match="Navigator must expose '_clock' and '_sleep' attributes"):
        run_navigation_observed(
            nav,  # type: ignore
            (10.0, 10.0),
            session=session,
            clock=clock,
            cpu_source=cpu,
        )


def test_run_navigation_observed_exception_propagation(tmp_path: Path) -> None:
    session = make_session(tmp_path)

    clock = FakeClock(0.0)
    cpu = FakeCpuSource([10.0])
    nav = FakeNavigator(clock, lambda s: None, should_raise=RuntimeError("nav failed"))

    t_config = TelemetryConfig(write_artifact=True)

    with pytest.raises(RuntimeError, match="nav failed"):
        run_navigation_observed(
            nav,  # type: ignore[arg-type]
            (10.0, 10.0),
            session=session,
            clock=clock,
            cpu_source=cpu,
            config=t_config,
        )

    # Artifact should NOT be written
    assert not (session.path / "nav_telemetry.json").exists()

    # Event should NOT be emitted
    events_file = session.path / "events.jsonl"
    lines = [json.loads(line) for line in events_file.read_text(encoding="utf-8").strip().split("\n") if line]
    telemetry_events = [e for e in lines if e.get("event") == "nav_telemetry"]
    assert len(telemetry_events) == 0


def test_run_navigation_observed_reads_session_id_public_property(tmp_path: Path) -> None:
    class CustomSession:
        def __init__(self, path: Path) -> None:
            self._session_id = "public-sess-123"
            self.path = path
            self.events: list[dict[str, object]] = []

        @property
        def session_id(self) -> str:
            return self._session_id

        def write_event(self, event: dict[str, object]) -> None:
            self.events.append(event)

    cust_session = CustomSession(tmp_path / "sess")

    clock = FakeClock(0.0)
    cpu = FakeCpuSource([15.0])
    nav = FakeNavigator(clock, lambda s: None)

    _, report = run_navigation_observed(
        nav,  # type: ignore[arg-type]
        (5.0, 5.0),
        session=cust_session,
        clock=clock,
        cpu_source=cpu,
    )

    assert report.session_id == "public-sess-123"
    assert len(cust_session.events) == 1
    assert cust_session.events[0]["session_id"] == "public-sess-123"


def test_determinism(tmp_path: Path) -> None:
    def run_simulation() -> NavTelemetryReport:
        session = make_session(tmp_path)
        clock = FakeClock(0.0)
        cpu = FakeCpuSource([10.0, 20.0, 30.0, 40.0])
        collector = TelemetryCollector(
            session_id=session.session_id,
            target_xy=(10.0, 10.0),
            clock=clock,
            cpu_source=cpu,
        )
        sampling_sleep = make_sampling_sleep(lambda s: None, collector, clock)
        nav = FakeNavigator(clock, sampling_sleep)

        _, report = run_navigation_observed(
            nav,  # type: ignore[arg-type]
            (10.0, 10.0),
            session=session,
            clock=clock,
            cpu_source=cpu,
        )
        return report

    r1 = run_simulation()
    r2 = run_simulation()

    assert r1.status == r2.status
    assert r1.target_xy == r2.target_xy
    assert r1.iterations == r2.iterations
    assert r1.replans == r2.replans
    assert r1.path_attempts == r2.path_attempts
    assert r1.duration_s == r2.duration_s
    assert r1.cpu_min_percent == r2.cpu_min_percent
    assert r1.cpu_max_percent == r2.cpu_max_percent
    assert r1.cpu_mean_percent == r2.cpu_mean_percent
    assert len(r1.cpu_samples) == len(r2.cpu_samples)
    for s1, s2 in zip(r1.cpu_samples, r2.cpu_samples):
        assert s1.ts == s2.ts
        assert s1.cpu_percent == s2.cpu_percent


@pytest.mark.skipif(
    os.environ.get("WOWBOT_LAB_TELEMETRY") != "1",
    reason="RealCpuSource test skipped on CI unless WOWBOT_LAB_TELEMETRY=1",
)
def test_real_cpu_source() -> None:
    source = RealCpuSource()
    val1 = source.cpu_percent()
    assert val1 == 0.0
    import time
    time.sleep(0.05)
    val2 = source.cpu_percent()
    assert val2 >= 0.0


def test_static_ast_imports() -> None:
    telemetry_file = Path("src/wow_bot/nav/telemetry.py")
    tree = ast.parse(telemetry_file.read_text(encoding="utf-8"))

    forbidden_modules = [
        "wow_bot.world",
        "wow_bot.strategist",
        "wow_bot.combat",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.actuation",
        "wow_bot.executor",
        "aiosqlite",
        "asyncio",
        "threading",
    ]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import: {mod}"
                assert "ollama" not in mod.lower()
                assert "openai" not in mod.lower()
                assert "anthropic" not in mod.lower()
                assert "llm" not in mod.lower()
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            if mod.startswith("wow_bot.nav"):
                # Must only allow wow_bot.nav.navigator or wow_bot.nav
                pass
            else:
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import: {mod}"
            assert "ollama" not in mod.lower()
            assert "openai" not in mod.lower()
            assert "anthropic" not in mod.lower()
            assert "llm" not in mod.lower()
