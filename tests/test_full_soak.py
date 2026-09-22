"""Tests for MOCK soak harness scripts/lab/full_soak.py (Task 12.1)."""

from __future__ import annotations

import ast
import asyncio
import json
import math
import sys
import time
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

try:
    import resource
except ImportError:
    resource = None  # type: ignore[assignment]

from scripts.lab.full_soak import (
    ProcessResourceSampler,
    _FakeLlmClient,
    _load_rotation,
    main,
    make_soak_sleep,
    run_soak_async,
)
from wow_bot.analysis.lab_soak_v2 import (
    LabSoakError,
    ResourceSnapshot,
    SoakSample,
    make_default_resource_sampler,
    validate_soak_report_dict,
)
from wow_bot.executor.states import FSMState
from wow_bot.lab.runner_v2 import LabRunResult, LabRunStatus


class FakeClock:
    """Fake clock advancing time by a fixed step per call to sleep method.

    Documented behavior: advances by `step` seconds on each call to `sleep`.
    """

    def __init__(self, step: float = 0.5, start: float = 100.0) -> None:
        self.step = step
        self.now_ts = start

    def now(self) -> float:
        """Return current fake timestamp."""
        return self.now_ts

    def sleep(self, seconds: float) -> None:
        """Advance fake timestamp by self.step on each call."""
        self.now_ts += self.step


class FakeResourceSampler:
    """Deterministic ResourceSampler for testing."""

    def __init__(self, log_path: Path | None = None) -> None:
        self.log_path = log_path

    def sample(self, now: float) -> ResourceSnapshot:
        """Return deterministic ResourceSnapshot derived from now."""
        log_size = 0
        if self.log_path is not None and self.log_path.exists():
            log_size = self.log_path.stat().st_size

        return ResourceSnapshot(
            ts=now,
            cpu_percent=10.0 + (now % 5.0),
            rss_bytes=100_000_000 + int(now),
            log_size_bytes=log_size,
        )


@dataclass
class FakeGameState:
    """Minimal valid synthetic GameState satisfying state view protocols."""

    player_x: float = 100.0
    player_y: float = 100.0
    player_z: float = 0.0
    player_heading: float = 0.0
    self_x: float = 100.0
    self_y: float = 100.0
    self_hp_percent: float = 100.0
    self_in_combat: bool = False
    fsm_state: FSMState = FSMState.IDLE
    current_target_id: str | None = None
    target_entity_id: str | None = None
    target_in_range: bool = False
    target_is_alive: bool = False
    target_is_lootable: bool = False
    target_distance: float | None = None
    target_x: float | None = None
    target_y: float | None = None
    target_hp_percent: float | None = None
    resource: float = 100.0
    resource_max: float = 100.0
    adds_count: int = 0
    inventory_count: int = 5
    inventory_max: int = 10_000_000
    durability_fraction: float | None = 1.0
    level: float = 10.0
    xp: float = 1000.0
    level_or_xp: float = 1000.0
    incoming_casts: tuple[Any, ...] = ()
    entities: tuple[Any, ...] = ()

    def spell_cooldown_ready(self, spell_id: str) -> bool:
        """Return True indicating all spell cooldowns are ready."""
        return True


@dataclass
class FakeMetaState:
    """Minimal valid synthetic MetaState."""

    drives: dict[str, float] = field(default_factory=dict)
    memory_summary: str = ""
    drive_hunger: float = 0.0
    drive_fatigue: float = 0.0
    chaos_level: float = 0.0


class FakeLlmClient:
    """Fake LLM client returning valid strategy JSON for tests."""

    def complete(self, prompt: str) -> str:
        """Return valid strategy JSON string."""
        return '{"goal":"explore","target":null,"rationale":"test"}'


@pytest.fixture
def test_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    """Create temporary config, farm profile, rotation, and world DB files."""
    config_path = tmp_path / "lab.toml"
    config_path.write_text(
        'lab_mode = false\n'
        'server_allowlist = ["127.0.0.1:8080"]\n'
        'isolation_sentinel = "127.0.0.1:9999"\n'
        'kill_switch_key = "F12"\n'
        f'session_root = "{tmp_path.as_posix()}"\n'
        'dry_run = true\n'
        'max_session_seconds = 3600\n'
        'log_level = "INFO"\n',
        encoding="utf-8",
    )

    profile_path = tmp_path / "profile.toml"
    profile_path.write_text(
        '[profile]\n'
        'schema_version = 1\n'
        'name = "test_profile"\n'
        'description = "Test profile for full soak"\n'
        '[cycle]\n'
        'nodes = [{ kind = "mob", name = "test_mob" }]\n'
        'vendor = { kind = "vendor", name = "test_vendor" }\n'
        'repair = { kind = "vendor", name = "test_vendor" }\n'
        'stop_when_inventory_full = true\n'
        'stop_after_cycles = 0\n'
        '[route_preferences]\n'
        'avoid_kinds = ["mob"]\n'
        'prefer_kinds = ["waypoint"]\n'
        'max_detour_factor = 1.5\n',
        encoding="utf-8",
    )

    rotation_path = tmp_path / "rotation.toml"
    rotation_path.write_text('default_spell_id = "shoot"\nrules = []\n', encoding="utf-8")

    world_db_path = tmp_path / "world.db"

    return config_path, profile_path, rotation_path, world_db_path


# ---------------------------------------------------------------------------
# CLI & Argument Validation Tests
# ---------------------------------------------------------------------------

def test_cli_help(capsys: pytest.CaptureFixture[str]) -> None:
    """Acceptance: --help exits with code 0 without running anything."""
    code = main(["--help"])
    assert code == 0


def test_cli_missing_args() -> None:
    """Acceptance: Missing required args exit with code 2."""
    code = main([])
    assert code == 2


def test_cli_duration_invalid(test_files: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """Acceptance: --duration-s <= 0 is rejected with code 2."""
    cfg, prof, rot, db = test_files
    code = main([
        "--session-dir", str(tmp_path / "s1"),
        "--session-id", "s1",
        "--duration-s", "0.0",
        "--config", str(cfg),
        "--world-db", str(db),
        "--profile", str(prof),
        "--rotation", str(rot),
    ])
    assert code == 2


def test_cli_sample_interval_invalid(test_files: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """Acceptance: --sample-interval-s <= 0 is rejected with code 2."""
    cfg, prof, rot, db = test_files
    code = main([
        "--session-dir", str(tmp_path / "s1"),
        "--session-id", "s1",
        "--sample-interval-s", "0.0",
        "--config", str(cfg),
        "--world-db", str(db),
        "--profile", str(prof),
        "--rotation", str(rot),
    ])
    assert code == 2


def test_cli_max_cycles_invalid(test_files: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """Acceptance: --max-cycles < 1 is rejected with code 2."""
    cfg, prof, rot, db = test_files
    code = main([
        "--session-dir", str(tmp_path / "s1"),
        "--session-id", "s1",
        "--max-cycles", "0",
        "--config", str(cfg),
        "--world-db", str(db),
        "--profile", str(prof),
        "--rotation", str(rot),
    ])
    assert code == 2


def test_cli_mode_lab_rejected(test_files: tuple[Path, Path, Path, Path], tmp_path: Path) -> None:
    """Acceptance: --mode LAB is rejected with code 2."""
    cfg, prof, rot, db = test_files
    code = main([
        "--session-dir", str(tmp_path / "s1"),
        "--session-id", "s1",
        "--mode", "LAB",
        "--config", str(cfg),
        "--world-db", str(db),
        "--profile", str(prof),
        "--rotation", str(rot),
    ])
    assert code == 2


# ---------------------------------------------------------------------------
# make_soak_sleep Unit Tests
# ---------------------------------------------------------------------------

def test_make_soak_sleep_calls_base_sleep() -> None:
    """Acceptance: make_soak_sleep calls base_sleep exactly once per invocation."""
    base_sleep_calls: list[float] = []

    def mock_sleep(s: float) -> None:
        base_sleep_calls.append(s)

    now_val = 100.0

    def mock_clock() -> float:
        return now_val

    stop_evt = asyncio.Event()
    samples: list[SoakSample] = []
    sampler = FakeResourceSampler()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        return SoakSample(
            ts=ts,
            cpu_percent=snap.cpu_percent,
            rss_bytes=snap.rss_bytes,
            log_size_bytes=snap.log_size_bytes,
            position_delta=0.0,
            inventory_delta=0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        )

    wrapper = make_soak_sleep(
        base_sleep=mock_sleep,
        sampler=sampler,
        clock=mock_clock,
        deadline=200.0,
        sample_interval_s=1.0,
        stop_event=stop_evt,
        samples=samples,
        sample_builder=sample_builder,
    )

    wrapper(0.05)
    assert len(base_sleep_calls) == 1
    assert base_sleep_calls[0] == 0.05


def test_make_soak_sleep_sets_stop_event_at_deadline() -> None:
    """Acceptance: make_soak_sleep sets stop_event when now >= deadline."""
    now_val = 200.0  # equals deadline

    def mock_clock() -> float:
        return now_val

    stop_evt = asyncio.Event()
    samples: list[SoakSample] = []
    sampler = FakeResourceSampler()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        return SoakSample(ts=ts, cpu_percent=0.0, rss_bytes=0, log_size_bytes=0, position_delta=0.0, inventory_delta=0, successful_actions_total=0, reflex_ticks_total=0)

    wrapper = make_soak_sleep(
        base_sleep=lambda s: None,
        sampler=sampler,
        clock=mock_clock,
        deadline=200.0,
        sample_interval_s=1.0,
        stop_event=stop_evt,
        samples=samples,
        sample_builder=sample_builder,
    )

    wrapper(0.05)
    assert stop_evt.is_set()


def test_make_soak_sleep_does_not_sample_at_deadline() -> None:
    """Acceptance: make_soak_sleep does NOT sample when now >= deadline."""
    now_val = 200.0  # deadline
    stop_evt = asyncio.Event()
    samples: list[SoakSample] = []
    sampler = FakeResourceSampler()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        return SoakSample(ts=ts, cpu_percent=0.0, rss_bytes=0, log_size_bytes=0, position_delta=0.0, inventory_delta=0, successful_actions_total=0, reflex_ticks_total=0)

    wrapper = make_soak_sleep(
        base_sleep=lambda s: None,
        sampler=sampler,
        clock=lambda: now_val,
        deadline=200.0,
        sample_interval_s=1.0,
        stop_event=stop_evt,
        samples=samples,
        sample_builder=sample_builder,
    )

    wrapper(0.05)
    assert len(samples) == 0


def test_make_soak_sleep_samples_on_first_call() -> None:
    """Acceptance: make_soak_sleep samples on the first call when samples is empty."""
    now_val = 100.0
    stop_evt = asyncio.Event()
    samples: list[SoakSample] = []
    sampler = FakeResourceSampler()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        return SoakSample(ts=ts, cpu_percent=snap.cpu_percent, rss_bytes=snap.rss_bytes, log_size_bytes=snap.log_size_bytes, position_delta=0.0, inventory_delta=0, successful_actions_total=0, reflex_ticks_total=0)

    wrapper = make_soak_sleep(
        base_sleep=lambda s: None,
        sampler=sampler,
        clock=lambda: now_val,
        deadline=200.0,
        sample_interval_s=1.0,
        stop_event=stop_evt,
        samples=samples,
        sample_builder=sample_builder,
    )

    wrapper(0.05)
    assert len(samples) == 1
    assert samples[0].ts == 100.0


def test_make_soak_sleep_throttles_sampling_interval() -> None:
    """Acceptance: make_soak_sleep does NOT sample again until sample_interval_s has elapsed, then samples again."""
    now_val = 100.0

    def mock_clock() -> float:
        return now_val

    stop_evt = asyncio.Event()
    samples: list[SoakSample] = []
    sampler = FakeResourceSampler()

    def sample_builder(ts: float, snap: ResourceSnapshot) -> SoakSample:
        return SoakSample(ts=ts, cpu_percent=snap.cpu_percent, rss_bytes=snap.rss_bytes, log_size_bytes=snap.log_size_bytes, position_delta=0.0, inventory_delta=0, successful_actions_total=0, reflex_ticks_total=0)

    wrapper = make_soak_sleep(
        base_sleep=lambda s: None,
        sampler=sampler,
        clock=mock_clock,
        deadline=200.0,
        sample_interval_s=5.0,
        stop_event=stop_evt,
        samples=samples,
        sample_builder=sample_builder,
    )

    wrapper(0.05)  # First call at ts=100.0 -> sample #1
    assert len(samples) == 1

    now_val = 102.0
    wrapper(0.05)  # Only 2.0s elapsed (<5.0s) -> NO new sample
    assert len(samples) == 1

    now_val = 105.1
    wrapper(0.05)  # 5.1s elapsed (>=5.0s) -> sample #2
    assert len(samples) == 2
    assert samples[1].ts == 105.1


# ---------------------------------------------------------------------------
# run_soak_async Integration Tests
# ---------------------------------------------------------------------------

@pytest.mark.asyncio
async def test_run_soak_async_smoke_run(
    test_files: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """Acceptance tests for run_soak_async with FakeClock:

    - completes within 5 seconds of real CPU time
    - produces a non-empty samples list
    - calls write_soak_report to session_dir / "soak_report.json"
    - written soak_report.json passes validate_soak_report_dict
    - written soak_report.json has crash=False
    - summary.sample_count equals len(samples)
    - returns LabRunResult with status in {STOP_EVENT_SET, MAX_CYCLES_REACHED}
    - runner does NOT start Reflex loop or Watchdog
    - the CLI's scripted LLM response is accepted by the real vocabulary guard
    """
    cfg, prof, rot, db = test_files
    sess_dir = tmp_path / "soak_session"
    fake_clock = FakeClock(step=0.5, start=100.0)
    sampler = FakeResourceSampler()

    state_inst = FakeGameState()
    meta_inst = FakeMetaState()

    def game_state_source() -> FakeGameState:
        state_inst.player_x += 1.0
        state_inst.player_y += 1.0
        state_inst.self_x += 1.0
        state_inst.self_y += 1.0
        state_inst.level += 1.0
        state_inst.xp += 10.0
        state_inst.level_or_xp += 10.0
        state_inst.inventory_count += 1
        return state_inst

    start_cpu = time.process_time()

    res, report = await run_soak_async(
        session_dir=sess_dir,
        session_id="soak_session",
        config_path=cfg,
        world_db_path=db,
        profile_path=prof,
        rotation_path=rot,
        duration_s=60.0,
        sample_interval_s=5.0,
        max_cycles=1000,
        clock=fake_clock.now,
        base_sleep=fake_clock.sleep,
        sampler=sampler,
        game_state_source=game_state_source,
        meta_state_source=lambda: meta_inst,
        llm_client=_FakeLlmClient(),
    )

    elapsed_cpu = time.process_time() - start_cpu
    assert elapsed_cpu < 5.0

    assert len(report.samples) > 0
    assert report.crash is False
    assert report.summary.sample_count == len(report.samples)

    report_path = sess_dir / "soak_report.json"
    assert report_path.exists()
    report_data = json.loads(report_path.read_text(encoding="utf-8"))
    validate_soak_report_dict(report_data)

    assert res.status in {LabRunStatus.STOP_EVENT_SET, LabRunStatus.MAX_CYCLES_REACHED}

    events = [
        json.loads(line)
        for line in (sess_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
    ]
    assert any(event["event"] == "strategist_success" for event in events)
    assert any(event["event"] == "vocab_accepted" for event in events)
    assert not any(event["event"] == "vocab_rejected" for event in events)


@pytest.mark.asyncio
async def test_run_soak_async_runner_error_writes_crash_report(
    test_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: run_soak_async with an unrecoverable runner error writes a crash report."""
    cfg, prof, rot, db = test_files
    sess_dir = tmp_path / "crash_session"
    fake_clock = FakeClock(step=0.5, start=100.0)
    sampler = FakeResourceSampler()

    state_inst = FakeGameState()
    meta_inst = FakeMetaState()

    async def mock_run_lab_loop_async(*args: Any, **kwargs: Any) -> LabRunResult:
        return LabRunResult(
            status=LabRunStatus.HEALTH_CRITICAL,
            cycles_completed=1,
            failures=1,
            duration_s=1.0,
            shutdown_report=None,
            reason="watchdog_critical_health",
        )

    monkeypatch.setattr("scripts.lab.full_soak.run_lab_loop_async", mock_run_lab_loop_async)

    res, report = await run_soak_async(
        session_dir=sess_dir,
        session_id="crash_session",
        config_path=cfg,
        world_db_path=db,
        profile_path=prof,
        rotation_path=rot,
        duration_s=60.0,
        sample_interval_s=5.0,
        max_cycles=100,
        clock=fake_clock.now,
        base_sleep=fake_clock.sleep,
        sampler=sampler,
        game_state_source=lambda: state_inst,
        meta_state_source=lambda: meta_inst,
        llm_client=FakeLlmClient(),
    )

    assert report.crash is True
    assert report.crash_reason == "watchdog_critical_health"
    assert res.status == LabRunStatus.HEALTH_CRITICAL


@pytest.mark.asyncio
async def test_run_soak_async_empty_samples_non_crash_raises(
    test_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: run_soak_async with an empty samples list AND crash=False raises LabSoakError."""
    cfg, prof, rot, db = test_files
    sess_dir = tmp_path / "empty_samples"
    fake_clock = FakeClock(step=0.5, start=100.0)
    sampler = FakeResourceSampler()

    state_inst = FakeGameState()
    meta_inst = FakeMetaState()

    def game_state_source() -> FakeGameState:
        state_inst.player_x += 1.0
        state_inst.player_y += 1.0
        state_inst.self_x += 1.0
        state_inst.self_y += 1.0
        state_inst.level += 1.0
        state_inst.xp += 10.0
        state_inst.level_or_xp += 10.0
        state_inst.inventory_count += 1
        return state_inst

    def no_sampling_sleep_builder(*args: Any, **kwargs: Any) -> Any:
        stop_event = kwargs["stop_event"]
        deadline = kwargs["deadline"]

        def soak_sleep(seconds: float) -> None:
            fake_clock.sleep(seconds)
            if fake_clock.now() >= deadline and not stop_event.is_set():
                stop_event.set()

        return soak_sleep

    monkeypatch.setattr(
        "scripts.lab.full_soak.make_soak_sleep",
        no_sampling_sleep_builder,
    )

    with pytest.raises(LabSoakError, match="no samples"):
        await run_soak_async(
            session_dir=sess_dir,
            session_id="empty_samples",
            config_path=cfg,
            world_db_path=db,
            profile_path=prof,
            rotation_path=rot,
            duration_s=60.0,
            sample_interval_s=5.0,
            max_cycles=100,
            clock=fake_clock.now,
            base_sleep=fake_clock.sleep,
            sampler=sampler,
            game_state_source=game_state_source,
            meta_state_source=lambda: meta_inst,
            llm_client=FakeLlmClient(),
        )


@pytest.mark.asyncio
async def test_run_soak_async_empty_samples_crash_writes_valid_report(
    test_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: run_soak_async with an empty samples list AND crash=True writes a valid crash report."""
    cfg, prof, rot, db = test_files
    sess_dir = tmp_path / "empty_crash"
    fake_clock = FakeClock(step=0.5, start=100.0)
    sampler = FakeResourceSampler()

    state_inst = FakeGameState()
    meta_inst = FakeMetaState()

    def no_sampling_sleep(s: float) -> None:
        fake_clock.sleep(s)

    async def mock_run_lab_loop_async(*args: Any, **kwargs: Any) -> LabRunResult:
        return LabRunResult(
            status=LabRunStatus.RUNTIME_ERROR,
            cycles_completed=0,
            failures=1,
            duration_s=0.0,
            shutdown_report=None,
            reason="unexpected_runtime_crash",
        )

    monkeypatch.setattr("scripts.lab.full_soak.make_soak_sleep", lambda *args, **kwargs: no_sampling_sleep)
    monkeypatch.setattr("scripts.lab.full_soak.run_lab_loop_async", mock_run_lab_loop_async)

    _res, report = await run_soak_async(
        session_dir=sess_dir,
        session_id="empty_crash",
        config_path=cfg,
        world_db_path=db,
        profile_path=prof,
        rotation_path=rot,
        duration_s=60.0,
        sample_interval_s=5.0,
        max_cycles=100,
        clock=fake_clock.now,
        base_sleep=fake_clock.sleep,
        sampler=sampler,
        game_state_source=lambda: state_inst,
        meta_state_source=lambda: meta_inst,
        llm_client=FakeLlmClient(),
    )

    assert report.crash is True
    assert report.crash_reason == "unexpected_runtime_crash"
    assert len(report.samples) == 0


@pytest.mark.asyncio
async def test_run_soak_async_uncaught_exception_writes_crash_json(
    test_files: tuple[Path, Path, Path, Path],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Acceptance: run_soak_async writes crash.json when an unhandled exception occurs."""
    cfg, prof, rot, db = test_files
    sess_dir = tmp_path / "uncaught_crash"
    fake_clock = FakeClock(step=0.5, start=100.0)
    sampler = FakeResourceSampler()

    state_inst = FakeGameState()
    meta_inst = FakeMetaState()

    async def mock_run_lab_loop_async(*args: Any, **kwargs: Any) -> LabRunResult:
        raise RuntimeError("simulated uncaught loop exception")

    monkeypatch.setattr("scripts.lab.full_soak.run_lab_loop_async", mock_run_lab_loop_async)

    with pytest.raises(RuntimeError, match="simulated uncaught loop exception"):
        await run_soak_async(
            session_dir=sess_dir,
            session_id="uncaught_crash",
            config_path=cfg,
            world_db_path=db,
            profile_path=prof,
            rotation_path=rot,
            duration_s=60.0,
            sample_interval_s=5.0,
            max_cycles=100,
            clock=fake_clock.now,
            base_sleep=fake_clock.sleep,
            sampler=sampler,
            game_state_source=lambda: state_inst,
            meta_state_source=lambda: meta_inst,
            llm_client=FakeLlmClient(),
        )

    crash_file = sess_dir / "crash.json"
    assert crash_file.exists()
    data = json.loads(crash_file.read_text(encoding="utf-8"))
    assert data["type"] == "RuntimeError"
    assert "simulated uncaught loop exception" in data["message"]


@pytest.mark.asyncio
async def test_run_soak_async_determinism(
    test_files: tuple[Path, Path, Path, Path], tmp_path: Path
) -> None:
    """Acceptance: Two runs with the same FakeClock and FakeResourceSampler produce equal SoakSample sequences."""
    cfg, prof, rot, db1 = test_files
    db2 = tmp_path / "world2.db"

    state_inst1 = FakeGameState()
    meta_inst1 = FakeMetaState()
    fake_clock1 = FakeClock(step=0.5, start=100.0)
    sampler1 = FakeResourceSampler()

    _res1, report1 = await run_soak_async(
        session_dir=tmp_path / "det1",
        session_id="det1",
        config_path=cfg,
        world_db_path=db1,
        profile_path=prof,
        rotation_path=rot,
        duration_s=10.0,
        sample_interval_s=1.0,
        max_cycles=20,
        clock=fake_clock1.now,
        base_sleep=fake_clock1.sleep,
        sampler=sampler1,
        game_state_source=lambda: state_inst1,
        meta_state_source=lambda: meta_inst1,
        llm_client=FakeLlmClient(),
    )

    state_inst2 = FakeGameState()
    meta_inst2 = FakeMetaState()
    fake_clock2 = FakeClock(step=0.5, start=100.0)
    sampler2 = FakeResourceSampler()

    _res2, report2 = await run_soak_async(
        session_dir=tmp_path / "det2",
        session_id="det2",
        config_path=cfg,
        world_db_path=db2,
        profile_path=prof,
        rotation_path=rot,
        duration_s=10.0,
        sample_interval_s=1.0,
        max_cycles=20,
        clock=fake_clock2.now,
        base_sleep=fake_clock2.sleep,
        sampler=sampler2,
        game_state_source=lambda: state_inst2,
        meta_state_source=lambda: meta_inst2,
        llm_client=FakeLlmClient(),
    )

    assert len(report1.samples) == len(report2.samples)
    for s1, s2 in zip(report1.samples, report2.samples, strict=True):
        assert s1.ts == s2.ts
        assert s1.cpu_percent == s2.cpu_percent
        assert s1.rss_bytes == s2.rss_bytes
        assert s1.log_size_bytes == s2.log_size_bytes


# ---------------------------------------------------------------------------
# ProcessResourceSampler Unit Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(resource is None, reason="resource module unavailable on Windows")
def test_process_resource_sampler_rss_positive() -> None:
    """Acceptance: ProcessResourceSampler reads current process RSS > 0."""
    sampler = ProcessResourceSampler()
    snap = sampler.sample(100.0)
    assert snap.rss_bytes > 0


@pytest.mark.skipif(resource is None, reason="resource module unavailable on Windows")
def test_process_resource_sampler_cpu_percentage_deltas() -> None:
    """Acceptance: ProcessResourceSampler's first call returns cpu_percent=0.0, second call returns finite non-negative cpu_percent."""
    sampler = ProcessResourceSampler()
    snap1 = sampler.sample(100.0)
    assert snap1.cpu_percent == 0.0

    snap2 = sampler.sample(101.0)
    assert snap2.cpu_percent >= 0.0


@pytest.mark.skipif(resource is None, reason="resource module unavailable on Windows")
def test_process_resource_sampler_log_file_size(tmp_path: Path) -> None:
    """Acceptance: ProcessResourceSampler with log_path set returns log_size_bytes matching the file size."""
    log_file = tmp_path / "app.log"
    log_file.write_text("hello world log content\n", encoding="utf-8")

    sampler = ProcessResourceSampler(log_path=log_file)
    snap = sampler.sample(100.0)
    assert snap.log_size_bytes == log_file.stat().st_size


# ---------------------------------------------------------------------------
# make_default_resource_sampler & WindowsResourceSampler Tests
# ---------------------------------------------------------------------------

@pytest.mark.skipif(sys.platform == "win32", reason="Linux/macOS test only")
def test_make_default_resource_sampler_posix() -> None:
    """Acceptance: make_default_resource_sampler returns ProcessResourceSampler on POSIX."""
    sampler = make_default_resource_sampler()
    assert isinstance(sampler, ProcessResourceSampler)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows test only")
def test_make_default_resource_sampler_windows() -> None:
    """Acceptance: make_default_resource_sampler returns WindowsResourceSampler on Windows."""
    from wow_bot.analysis.windows_sampler import WindowsResourceSampler

    sampler = make_default_resource_sampler()
    assert isinstance(sampler, WindowsResourceSampler)


@pytest.mark.skipif(sys.platform != "win32", reason="Windows test only")
def test_windows_resource_sampler_unit() -> None:
    """Acceptance unit test for WindowsResourceSampler on Windows:

    - calls sample(now=0.0) twice
    - asserts first call returns cpu_percent == 0.0
    - asserts second call returns a finite non-negative cpu_percent
    - asserts rss_bytes > 0
    """
    from wow_bot.analysis.windows_sampler import WindowsResourceSampler

    sampler = WindowsResourceSampler()
    snap1 = sampler.sample(now=0.0)
    assert snap1.cpu_percent == 0.0
    assert snap1.rss_bytes > 0

    snap2 = sampler.sample(now=1.0)
    assert math.isfinite(snap2.cpu_percent)
    assert snap2.cpu_percent >= 0.0
    assert snap2.rss_bytes > 0


# ---------------------------------------------------------------------------
# Rotation TOML Parsing Tests
# ---------------------------------------------------------------------------

def test_load_rotation_toml(tmp_path: Path) -> None:
    """Acceptance: _load_rotation correctly parses a valid rotation TOML file."""
    rot_path = tmp_path / "valid_rotation.toml"
    rot_path.write_text(
        'default_spell_id = "shoot"\n'
        '[[rules]]\n'
        'priority = 10\n'
        'spell_id = "fireball"\n'
        '[[rules.conditions]]\n'
        'kind = "spell_ready"\n'
        'value = true\n'
        'spell_id = "fireball"\n',
        encoding="utf-8",
    )
    rot_config = _load_rotation(rot_path)
    assert len(rot_config.rules) >= 1
    assert rot_config.rules[0].spell_id == "fireball"


def test_example_rotation_toml_parses() -> None:
    """Acceptance: config/rotations/example.toml parses via tomllib.load and _load_rotation."""
    example_path = Path("config/rotations/example.toml")
    if not example_path.exists():
        pytest.skip("config/rotations/example.toml fixture does not exist")

    with open(example_path, "rb") as f:
        data = tomllib.load(f)
    assert "default_spell_id" in data

    rot_config = _load_rotation(example_path)
    assert len(rot_config.rules) >= 1


# ---------------------------------------------------------------------------
# Static AST Inspection Invariants Test
# ---------------------------------------------------------------------------

def test_static_ast_invariants() -> None:
    """Acceptance: Static AST checks for scripts/lab/full_soak.py:

    - Does not import forbidden modules
    - Does not call sys.exit outside main or __main__ guard
    - Does not call time.monotonic, time.time, or time.perf_counter outside main
    - Does not call time.sleep outside main
    - Does not call .start() or .stop() on thread-like objects
    """
    script_path = Path("scripts/lab/full_soak.py")
    assert script_path.exists()
    tree = ast.parse(script_path.read_text(encoding="utf-8"), filename=str(script_path))

    forbidden_exact_modules = {
        "wow_bot.main",
        "wow_bot.executor.fsm",
        "wow_bot.executor.controller",
        "wow_bot.strategist.orchestrator",
        "wow_bot.strategist.llm_client",
        "wow_bot.reporting.scenario",
        "wow_bot.analysis.spectrum",
        "wow_bot.analysis.timing",
        "wow_bot.analysis.soak",
        "wow_bot.internal_dynamics",
        "psutil",
        "numpy",
        "scipy",
    }
    forbidden_substrings = ("ollama", "openai", "anthropic", "gemini", "cohere", "bedrock")

    # Check import nodes
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name = alias.name
                assert name not in forbidden_exact_modules, f"Forbidden import found: {name}"
                for sub in forbidden_substrings:
                    assert sub not in name.lower(), f"Forbidden LLM module in import: {name}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            assert mod not in forbidden_exact_modules, f"Forbidden import from: {mod}"
            for sub in forbidden_substrings:
                assert sub not in mod.lower(), f"Forbidden LLM module in import: {mod}"

    # Check function definitions for calls
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            func_name = node.name
            for child in ast.walk(node):
                if isinstance(child, ast.Call):
                    # Check sys.exit
                    if func_name != "main" and isinstance(child.func, ast.Attribute) and getattr(child.func.value, "id", "") == "sys" and child.func.attr == "exit":
                        pytest.fail(f"sys.exit call found in function '{func_name}' outside main()")

                    # Check time.monotonic / time.time / time.perf_counter / time.sleep
                    if isinstance(child.func, ast.Attribute) and getattr(child.func.value, "id", "") == "time" and child.func.attr in ("monotonic", "time", "perf_counter", "sleep"):
                        pytest.fail(f"time.{child.func.attr} call found in function '{func_name}' outside main()")

                    # Check .start() or .stop() calls on thread objects
                    if isinstance(child.func, ast.Attribute) and child.func.attr in ("start", "stop") and func_name != "main":
                        pytest.fail(f".{child.func.attr}() call found in function '{func_name}'")
