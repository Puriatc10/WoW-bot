"""Regression evidence for verified frozen-contract review findings."""

from __future__ import annotations

import asyncio
import json
import queue
import threading
from unittest.mock import AsyncMock, MagicMock

import pytest

import wow_bot.main as pipeline
from scripts import run_scenario, run_soak_test
from wow_bot.analysis.soak import FakeResourceSampler, SoakSample, build_soak_report
from wow_bot.analysis.timing import validate_timing_samples
from wow_bot.shared.config import Settings
from wow_bot.shared.interfaces import GameState, MetaState, Strategy
from wow_bot.strategist.parser import StrategyParseError, parse_strategy_response


class FakeWatchdog:
    def __init__(self):
        self.message_queue = queue.Queue()
        self.shutdown_event = threading.Event()
        self.is_alive = False

    def start(self):
        self.is_alive = True

    def close(self):
        self.is_alive = False


@pytest.fixture
async def runtime(monkeypatch):
    monkeypatch.setattr(pipeline, "WatchdogProcess", FakeWatchdog)
    settings = Settings()
    settings.internal_dynamics.memory_db_path = ":memory:"
    settings.internal_dynamics.update_interval_ms = 1
    components = await pipeline.build_runtime(settings, scenario="peaceful_farm", seed=42)
    try:
        yield components
    finally:
        await components.memory.close()
        await components.llm_client.close()
        components.watchdog.close()


def snapshot(ts):
    gs = GameState(ts, 1.0, 1.0, (0.0, 0.0), 0.0, False, None, [], [])
    return pipeline.PipelineSnapshot(gs, MetaState([0.5] * 5, [], ts))


def strategy():
    return Strategy("explore", "synthetic", 0.2, [], {}, 10000.0)


async def test_initial_strategy_failure_is_visible_and_closes_resources(runtime):
    failure = StrategyParseError("invalid completion")
    runtime.strategist.generate_strategy = AsyncMock(side_effect=failure)
    with pytest.raises(StrategyParseError) as raised:
        await asyncio.wait_for(pipeline.run_pipeline(runtime), 3)
    assert raised.value is failure
    assert runtime.controller.commands[-1].action == "stop_all"
    assert not runtime.watchdog.is_alive
    with pytest.raises(RuntimeError, match="closed"):
        await runtime.llm_client.query("", "")
    with pytest.raises(RuntimeError, match="not initialized"):
        await runtime.memory.recall_similar([0.5] * 5)


async def test_duration_cancels_blocked_llm_and_backpressure(runtime):
    entered = asyncio.Event()
    cancelled = asyncio.Event()

    async def blocked(*args):
        entered.set()
        try:
            await asyncio.Event().wait()
        finally:
            cancelled.set()

    runtime.strategist.generate_strategy = blocked
    await asyncio.wait_for(pipeline.run_pipeline(runtime, run_duration_seconds=0.1), 3)
    assert entered.is_set() and cancelled.is_set()
    assert not any(t.get_name() in {"duration_timer", "shutdown_waiter", "strategist_loop"}
                   for t in asyncio.all_tasks() if not t.done())


@pytest.mark.parametrize("requested", [True, False])
async def test_watchdog_shutdown_or_death_is_failure(runtime, requested):
    if requested:
        runtime.watchdog.shutdown_event.set()
    else:
        runtime.watchdog.start = lambda: None
    with pytest.raises(RuntimeError, match="Watchdog"):
        await asyncio.wait_for(pipeline.run_pipeline(runtime), 3)


async def test_watchdog_start_failure_closes_runtime(runtime):
    runtime.watchdog.start = MagicMock(side_effect=OSError("spawn failed"))
    with pytest.raises(OSError, match="spawn failed"):
        await pipeline.run_pipeline(runtime)
    assert runtime.controller.commands[-1].action == "stop_all"
    with pytest.raises(RuntimeError, match="not initialized"):
        await runtime.memory.recall_similar([0.5] * 5)


async def test_build_failure_closes_opened_memory(monkeypatch):
    memory = MagicMock(init=AsyncMock(), close=AsyncMock())
    monkeypatch.setattr(pipeline, "MemoryStore", lambda _: memory)
    monkeypatch.setattr(pipeline, "LLMClient", MagicMock(side_effect=ValueError("endpoint")))
    with pytest.raises(ValueError, match="endpoint"):
        await pipeline.build_runtime(Settings())
    memory.close.assert_awaited_once()


@pytest.mark.parametrize("bad", [-1.0, float("nan"), float("inf")])
async def test_dynamics_rejects_invalid_domain_time(runtime, bad):
    incoming, outgoing, planning = asyncio.Queue(), asyncio.Queue(), asyncio.Queue(maxsize=1)
    await incoming.put(snapshot(0).game_state)
    await incoming.put(snapshot(bad).game_state)
    state = {"session_start": None, "progress_token": 0}
    with pytest.raises(ValueError, match="timestamp"):
        await pipeline.dynamics_loop(runtime, incoming, outgoing, planning, asyncio.Event(), state)
    assert state["progress_token"] == 1


async def test_refresh_cooldown_and_zero_session_start(runtime):
    prior = strategy()
    runtime.strategist = MagicMock(current_strategy=prior)
    runtime.strategist.is_expired.return_value = True
    runtime.strategist.generate_strategy = AsyncMock(return_value=prior)
    incoming, outgoing = asyncio.Queue(maxsize=1), asyncio.Queue(maxsize=1)
    shutdown = asyncio.Event()
    task = asyncio.create_task(pipeline.strategist_loop(
        runtime, incoming, outgoing, shutdown, {"session_start": 0.0}
    ))
    try:
        for ts, expected in [(0, 1), (9.9, 1), (10, 2)]:
            await incoming.put(snapshot(ts))
            await asyncio.wait_for(incoming.join(), 3)
            assert runtime.strategist.generate_strategy.await_count == expected
        assert runtime.strategist.generate_strategy.call_args.args[1].session_start == 0.0
        assert prior.valid_until == 10000.0
        assert outgoing.empty()
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1, True])
async def test_duration_must_be_positive_finite(runtime, bad):
    with pytest.raises(ValueError, match="positive finite"):
        await pipeline.run_pipeline(runtime, run_duration_seconds=bad)


async def test_latest_queue_join_accounts_for_discard():
    q = asyncio.Queue(maxsize=1)
    pipeline._put_latest(q, 1)
    pipeline._put_latest(q, 2)
    assert await q.get() == 2
    q.task_done()
    await asyncio.wait_for(q.join(), 1)


@pytest.mark.parametrize("unit,delay", [("s", 200.0), ("ms", 201.0)])
def test_timing_rejects_inconsistent_dataset(unit, delay):
    with pytest.raises(ValueError):
        validate_timing_samples({"unit": unit, "samples_ms": [200.0], "samples": [{
            "simulation_timestamp": 1.0, "delay_ms": delay,
            "base_ms": 200, "fatigue": 0.5, "chaos_component": 0.0,
        }]})


@pytest.mark.parametrize("suffix", ["extra```", "```junk```"])
def test_parser_rejects_invalid_fence_suffix(suffix):
    raw = json.dumps({"reasoning": "x", "goal": "explore", "region": "x",
                      "risk_tolerance": 0.5, "priority": [], "constraints": {
                          "max_deaths_per_hour": 1, "max_session_minutes": 10,
                          "avoid_pvp": True}})
    with pytest.raises(StrategyParseError):
        parse_strategy_response("```json\n" + raw + "\n" + suffix, 10)


def test_parser_rejects_duplicate_fields():
    with pytest.raises(StrategyParseError, match="Duplicate"):
        parse_strategy_response('{"goal":"flee","goal":"explore"}', 10)


@pytest.mark.parametrize("elapsed,shutdown", [(99.9, False), (100.0, True)])
def test_soak_never_accepts_short_or_emergency_run(elapsed, shutdown):
    report = build_soak_report(
        scenario="peaceful_farm", seed=42, requested_duration_seconds=100,
        sample_interval_seconds=5, completed_duration_seconds=elapsed,
        samples=[SoakSample(0, 0, 100, 0, 0, True, shutdown)],
        completed_normally=True, termination_reason="duration_completed",
    )
    assert report["acceptance"]["zero_crash_target_met"] is False


@pytest.mark.parametrize("runner", ["scenario", "soak"])
async def test_runner_preserves_startup_failure_report(monkeypatch, tmp_path, runner):
    module = run_scenario if runner == "scenario" else run_soak_test
    monkeypatch.setattr(module, "build_runtime", AsyncMock(side_effect=ValueError("startup")))
    path = tmp_path / "failure.json"
    if runner == "scenario":
        code = await module.execute_scenario("peaceful_farm", 1, 42, path)
    else:
        code = await module.run_soak_test_async(
            "peaceful_farm", 1, 0.1, 42, path, resource_sampler=FakeResourceSampler()
        )
    assert code == 1
    report = json.loads(path.read_text())
    status = report["run"] if runner == "scenario" else report["termination"]
    assert status["completed_normally"] is False


async def test_soak_sampler_failure_stops_pipeline(monkeypatch, tmp_path, runtime):
    monkeypatch.setattr(run_soak_test, "build_runtime", AsyncMock(return_value=runtime))
    stopped = asyncio.Event()

    async def blocked(*args, **kwargs):
        try:
            await asyncio.Event().wait()
        finally:
            stopped.set()

    monkeypatch.setattr(run_soak_test, "run_pipeline", blocked)
    sampler = MagicMock(sample=MagicMock(side_effect=OSError("sampling failed")))
    path = tmp_path / "sampling.json"
    code = await asyncio.wait_for(run_soak_test.run_soak_test_async(
        "peaceful_farm", 100, 1, 42, path, resource_sampler=sampler
    ), 3)
    assert code == 1 and stopped.is_set()
    assert not json.loads(path.read_text())["acceptance"]["zero_crash_target_met"]


def test_child_cpu_sampling_reuses_measurement_baseline(monkeypatch):
    from wow_bot.analysis import soak

    class Process:
        def __init__(self, pid):
            self.pid = pid
            self.calls = 0

        def __eq__(self, other):
            return isinstance(other, Process) and self.pid == other.pid

        def cpu_percent(self, interval=None):
            self.calls += 1
            return 0.0 if self.calls == 1 else 25.0

        def children(self, recursive=True):
            return [Process(2)]

        def is_running(self):
            return True

        def memory_info(self):
            return MagicMock(rss=1024 * 1024)

    monkeypatch.setattr(soak.psutil, "Process", Process)
    sampler = soak.ProcessResourceSampler(pid=1)
    assert sampler.sample().cpu_percent == 50.0
    assert sampler.sample().memory_rss_mb == 2.0


async def test_memory_schema_failure_closes_connection(monkeypatch):
    from wow_bot.internal_dynamics import memory

    connection = MagicMock(executescript=AsyncMock(side_effect=OSError("schema")),
                           close=AsyncMock())
    monkeypatch.setattr(memory.aiosqlite, "connect", AsyncMock(return_value=connection))
    store = memory.MemoryStore(":memory:")
    with pytest.raises(OSError, match="schema"):
        await store.init()
    connection.close.assert_awaited_once()
    with pytest.raises(RuntimeError, match="not initialized"):
        await store.recall_similar([0.5] * 5)


async def test_primary_failure_survives_cleanup_failure(runtime):
    failure = ValueError("primary")
    runtime.strategist.generate_strategy = AsyncMock(side_effect=failure)
    runtime.controller.stop_all = AsyncMock(side_effect=OSError("cleanup"))
    with pytest.raises(ValueError) as raised:
        await asyncio.wait_for(pipeline.run_pipeline(runtime), 3)
    assert raised.value is failure
    assert not runtime.watchdog.is_alive


async def test_external_cancellation_cleans_up(runtime):
    entered = asyncio.Event()

    async def blocked(*args):
        entered.set()
        await asyncio.Event().wait()

    runtime.strategist.generate_strategy = blocked
    task = asyncio.create_task(pipeline.run_pipeline(runtime))
    await asyncio.wait_for(entered.wait(), 3)
    task.cancel()
    with pytest.raises(asyncio.CancelledError):
        await task
    assert not runtime.watchdog.is_alive
    assert runtime.controller.commands[-1].action == "stop_all"


async def test_slow_failed_refresh_cooldown_starts_on_return(runtime):
    prior = strategy()
    runtime.strategist = MagicMock(current_strategy=prior)
    runtime.strategist.is_expired.return_value = True
    state = {"session_start": 0.0, "last_sim_timestamp": 0.0}

    async def fallback(*args):
        state["last_sim_timestamp"] = 100.0
        return prior

    runtime.strategist.generate_strategy = AsyncMock(side_effect=fallback)
    incoming, outgoing = asyncio.Queue(maxsize=1), asyncio.Queue(maxsize=1)
    task = asyncio.create_task(pipeline.strategist_loop(
        runtime, incoming, outgoing, asyncio.Event(), state
    ))
    try:
        for ts, expected in [(0, 1), (100, 1), (109.9, 1), (110, 2)]:
            await incoming.put(snapshot(ts))
            await asyncio.wait_for(incoming.join(), 3)
            assert runtime.strategist.generate_strategy.await_count == expected
    finally:
        task.cancel()
        await asyncio.gather(task, return_exceptions=True)


async def test_llm_exception_body_not_logged(monkeypatch):
    import httpx
    import openai

    from wow_bot.shared.config import LLMConfig
    from wow_bot.strategist import llm_client

    log = MagicMock()
    monkeypatch.setattr(llm_client, "logger", log)
    marker = "RAW_RESPONSE_SECRET_MARKER"
    sdk = MagicMock()
    sdk.chat.completions.create = AsyncMock(side_effect=openai.APIConnectionError(
        message=marker, request=httpx.Request("POST", "http://localhost/v1")
    ))
    client = llm_client.LLMClient(LLMConfig(max_retries=0), _async_openai_client=sdk)
    try:
        with pytest.raises(openai.APIConnectionError):
            await client.query("system", "user")
        assert marker not in str(log.mock_calls)
    finally:
        await client.close()
