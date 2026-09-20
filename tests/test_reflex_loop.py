"""Tests for the reflex loop, signals, controls, and AST isolation checks."""

import ast
import random
import threading
import time
from pathlib import Path

import pytest

from wow_bot.config import Config
from wow_bot.reflex import (
    CallbackSink,
    ControlKind,
    ControlSignal,
    ListSignalSource,
    NullClock,
    NullControlSink,
    NullSignalSource,
    ReflexError,
    ReflexLoop,
    Signal,
)
from wow_bot.session import Session


def make_test_config(session_root: Path) -> Config:
    """Helper to construct a valid Config instance."""
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


def test_rate_hz_validation() -> None:
    """ReflexLoop with rate_hz <= 0 or > 100 raises ValueError."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    with pytest.raises(ValueError, match="rate_hz"):
        ReflexLoop(rate_hz=0.0, sources=(), sinks=(), rules=rules)

    with pytest.raises(ValueError, match="rate_hz"):
        ReflexLoop(rate_hz=-5.0, sources=(), sinks=(), rules=rules)

    with pytest.raises(ValueError, match="rate_hz"):
        ReflexLoop(rate_hz=100.1, sources=(), sinks=(), rules=rules)


def test_non_callable_rules_validation() -> None:
    """ReflexLoop with a non-callable rules raises ValueError."""
    with pytest.raises(ValueError, match="rules must be callable"):
        ReflexLoop(
            rate_hz=20.0,
            sources=(),
            sinks=(),
            rules="not_callable",  # type: ignore[arg-type]
        )


def test_null_sources_and_sinks() -> None:
    """ReflexLoop with null sources/sinks and rules returning [] produces zero stats."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(NullSignalSource(),),
        sinks=(NullControlSink(),),
        rules=rules,
        clock=NullClock(),
    )
    stats = loop.tick()
    assert stats.signals_processed == 0
    assert stats.controls_emitted == 0
    assert stats.sink_errors == 0
    assert not stats.overrun


def test_list_signal_source_batch_consumption() -> None:
    """ListSignalSource injected with 3 signals causes signals_processed=3 on tick 1, 0 on tick 2."""
    sig1 = Signal("test1", {"a": 1}, 100.0)
    sig2 = Signal("test2", {"b": 2}, 100.0)
    sig3 = Signal("test3", {"c": 3}, 100.0)

    source = ListSignalSource([sig1, sig2, sig3])
    assert source.remaining() == 3

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(source,),
        sinks=(),
        rules=rules,
        clock=NullClock(),
    )

    stats1 = loop.tick()
    assert stats1.signals_processed == 3
    assert source.remaining() == 0

    stats2 = loop.tick()
    assert stats2.signals_processed == 0


def test_rules_receives_strictly_increasing_tick_index() -> None:
    """rules receives tick_index in strictly increasing order starting at 0."""
    seen_indices: list[int] = []

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        seen_indices.append(tick_index)
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=NullClock(),
    )

    for _ in range(5):
        loop.tick()

    assert seen_indices == [0, 1, 2, 3, 4]


def test_distinct_rng_instance_per_tick() -> None:
    """rules receives a distinct random.Random instance per tick (different identity)."""
    rng_instances: list[random.Random] = []

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        rng_instances.append(rng)
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=NullClock(),
    )

    loop.tick()
    loop.tick()

    assert len(rng_instances) == 2
    assert rng_instances[0] is not rng_instances[1]


def test_determinism_same_seed() -> None:
    """Two loops with same seed and signal sequence produce identical bit sequences over 100 ticks."""
    seq1: list[int] = []
    seq2: list[int] = []

    def rules1(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        seq1.append(rng.getrandbits(32))
        return []

    def rules2(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        seq2.append(rng.getrandbits(32))
        return []

    loop1 = ReflexLoop(rate_hz=20.0, sources=(), sinks=(), rules=rules1, seed=12345, clock=NullClock())
    loop2 = ReflexLoop(rate_hz=20.0, sources=(), sinks=(), rules=rules2, seed=12345, clock=NullClock())

    for _ in range(100):
        loop1.tick()
        loop2.tick()

    assert seq1 == seq2


def test_determinism_different_seeds() -> None:
    """Two loops with different seeds produce different rng-derived values."""
    seq1: list[int] = []
    seq2: list[int] = []

    def rules1(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        seq1.append(rng.getrandbits(32))
        return []

    def rules2(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        seq2.append(rng.getrandbits(32))
        return []

    loop1 = ReflexLoop(rate_hz=20.0, sources=(), sinks=(), rules=rules1, seed=111, clock=NullClock())
    loop2 = ReflexLoop(rate_hz=20.0, sources=(), sinks=(), rules=rules2, seed=222, clock=NullClock())

    for _ in range(10):
        loop1.tick()
        loop2.tick()

    assert seq1 != seq2


def test_control_signal_dispatch_order() -> None:
    """Control signals are dispatched to sinks in registration order."""
    dispatch_log: list[str] = []

    def make_cb(sink_name: str) -> CallbackSink:
        return CallbackSink(lambda sig: dispatch_log.append(f"{sink_name}:{sig.reason}"))

    sink_a = make_cb("sink_a")
    sink_b = make_cb("sink_b")

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return [
            ControlSignal(ControlKind.ABORT_ACTUATION, "r1", 10.0),
            ControlSignal(ControlKind.PAUSE_FSM, "r2", 10.0),
        ]

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(sink_a, sink_b),
        rules=rules,
        clock=NullClock(),
    )

    loop.tick()

    assert dispatch_log == [
        "sink_a:r1",
        "sink_b:r1",
        "sink_a:r2",
        "sink_b:r2",
    ]


def test_sink_raises_handled_gracefully() -> None:
    """If a sink raises, last_sink_error() captures it, remaining sinks are called, tick() succeeds."""
    called: list[str] = []

    def failing_cb(sig: ControlSignal) -> None:
        raise RuntimeError("sink error")

    def succeeding_cb(sig: ControlSignal) -> None:
        called.append("succeeded")

    sink1 = CallbackSink(failing_cb)
    sink2 = CallbackSink(succeeding_cb)

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return [ControlSignal(ControlKind.ABORT_ACTUATION, "reason", 1.0)]

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(sink1, sink2),
        rules=rules,
        clock=NullClock(),
    )

    stats = loop.tick()
    assert stats.sink_errors == 1
    assert isinstance(loop.last_sink_error, RuntimeError)
    assert called == ["succeeded"]


def test_source_raises_handled_gracefully() -> None:
    """If a source raises, last_source_error() captures it, subsequent sources polled, tick() succeeds."""
    called: list[str] = []

    class FailingSource:
        def poll(self, now: float) -> list[Signal]:
            raise ValueError("source error")

    class WorkingSource:
        def poll(self, now: float) -> list[Signal]:
            called.append("polled")
            return [Signal("sig", {}, now)]

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(FailingSource(), WorkingSource()),
        sinks=(),
        rules=rules,
        clock=NullClock(),
    )

    stats = loop.tick()
    assert stats.signals_processed == 1
    assert isinstance(loop.last_source_error, ValueError)
    assert called == ["polled"]


def test_rules_exception_propagates() -> None:
    """If rules raises, the exception propagates out of tick()."""
    def broken_rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        raise KeyError("rules failure")

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=broken_rules,
        clock=NullClock(),
    )

    with pytest.raises(KeyError, match="rules failure"):
        loop.tick()


def test_overrun_detection() -> None:
    """overrun is True when tick_end - tick_start > 1/rate_hz."""
    clock = NullClock()

    def slow_rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        # Advance clock by 0.1s (greater than 1/20 = 0.05s)
        clock.sleep(0.1)
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=slow_rules,
        clock=clock,
    )

    stats = loop.tick()
    assert stats.overrun is True


def test_session_event_emission_on_interesting_tick(tmp_path: Path) -> None:
    """With Session attached and signals_processed>0, exactly 1 'reflex_tick' event is emitted."""
    config = make_test_config(tmp_path / "runs")
    session = Session.start(config)

    sig = Signal("sig", {"val": 1}, 0.0)
    source = ListSignalSource([sig])

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(source,),
        sinks=(),
        rules=rules,
        clock=NullClock(),
        session=session,
    )

    loop.tick()

    events_file = session.path / "events.jsonl"
    lines = events_file.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 1
    assert '"event": "reflex_tick"' in lines[0]


def test_session_event_suppressed_on_quiet_tick(tmp_path: Path) -> None:
    """With Session attached and no signals/controls/errors/overrun, NO event is emitted."""
    config = make_test_config(tmp_path / "runs")
    session = Session.start(config)

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(NullSignalSource(),),
        sinks=(NullControlSink(),),
        rules=rules,
        clock=NullClock(),
        session=session,
    )

    loop.tick()

    events_file = session.path / "events.jsonl"
    lines = [line for line in events_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 0


def test_run_for_duration() -> None:
    """run_for(1.0) with rate_hz=20 and NullClock returns exactly 20 TickStats."""
    clock = NullClock()

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=clock,
    )

    stats_list = loop.run_for(1.0)
    assert len(stats_list) == 20


def test_run_for_runaway_rules_bound() -> None:
    """run_for on runaway rules where clock never advances raises ReflexError after max_ticks."""
    clock = NullClock()

    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=20.0,
        sources=(),
        sinks=(),
        rules=rules,
        clock=clock,
    )

    # Do not sleep in clock, so now() never reaches target_time
    with pytest.raises(ReflexError, match="exceeded maximum tick count"):
        loop.run_for(1.0, max_ticks=10)


def test_start_is_idempotent() -> None:
    """start() is idempotent: calling twice does not spawn a second thread."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=50.0,
        sources=(),
        sinks=(),
        rules=rules,
    )

    threads_before = threading.active_count()
    loop.start()
    assert loop.is_running()
    threads_after = threading.active_count()

    loop.start()  # Second call
    assert threading.active_count() == threads_after
    assert threads_after == threads_before + 1

    loop.stop()
    assert not loop.is_running()


def test_stop_is_idempotent_and_safe_before_start() -> None:
    """stop() is idempotent and safe before start()."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=50.0,
        sources=(),
        sinks=(),
        rules=rules,
    )

    # Call stop before start
    loop.stop()
    assert not loop.is_running()

    loop.start()
    assert loop.is_running()
    loop.stop()
    loop.stop()  # Second stop call
    assert not loop.is_running()


def test_stop_timeout_performance() -> None:
    """stop(timeout_s=0.5) returns within 1 second even while loop is running."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=50.0,
        sources=(),
        sinks=(),
        rules=rules,
    )

    loop.start()
    t0 = time.monotonic()
    loop.stop(timeout_s=0.5)
    t1 = time.monotonic()

    assert (t1 - t0) < 1.0
    assert not loop.is_running()


def test_context_manager_lifecycle() -> None:
    """Context manager starts loop on enter and stops on exit."""
    def rules(signals: list[Signal], tick_index: int, rng: random.Random) -> list[ControlSignal]:
        return []

    loop = ReflexLoop(
        rate_hz=50.0,
        sources=(),
        sinks=(),
        rules=rules,
    )

    with loop:
        assert loop.is_running()

    assert not loop.is_running()


def test_static_ast_check_imports() -> None:
    """ReflexLoop module static AST check: no forbidden architectural layer or LLM imports."""
    loop_file = Path("src/wow_bot/reflex/loop.py")
    tree = ast.parse(loop_file.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.executor",
        "wow_bot.reflex.stuck",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }

    forbidden_llm_substrings = ["ollama", "openai", "anthropic", "llm"]

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod = alias.name
                for forbidden in forbidden_modules:
                    assert not mod.startswith(forbidden), f"Forbidden import found: {mod}"
                for llm_sub in forbidden_llm_substrings:
                    assert llm_sub not in mod.lower(), f"Forbidden LLM import found: {mod}"
        elif isinstance(node, ast.ImportFrom):
            mod = node.module or ""
            for forbidden in forbidden_modules:
                assert not mod.startswith(forbidden), f"Forbidden import found: {mod}"
            for llm_sub in forbidden_llm_substrings:
                assert llm_sub not in mod.lower(), f"Forbidden LLM import found: {mod}"
            for alias in node.names:
                name = alias.name
                for llm_sub in forbidden_llm_substrings:
                    assert llm_sub not in name.lower(), f"Forbidden LLM import found: {name}"
