"""Reflex loop execution and scheduling."""

import collections
import dataclasses
import random
import threading
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from types import TracebackType
from typing import Literal, Protocol, Self

from wow_bot.reflex.controls import ControlSignal, ControlSink
from wow_bot.reflex.signals import Signal, SignalSource
from wow_bot.session import Session


class ReflexError(Exception):
    """Raised when reflex loop operations fail or run bounds are exceeded."""


class Clock(Protocol):
    """Clock interface for time measurement and sleeping."""

    def now(self) -> float:
        """Return current clock time in seconds."""
        ...

    def sleep(self, seconds: float) -> None:
        """Sleep for specified duration in seconds."""
        ...


class RealClock:
    """Clock implementation using standard time.monotonic and time.sleep."""

    def now(self) -> float:
        return time.monotonic()

    def sleep(self, seconds: float) -> None:
        time.sleep(seconds)


class NullClock:
    """Clock implementation for deterministic testing.

    `now()` returns an internal counter advancing by `step_s` on `sleep(step_s)`.
    `sleep(0)` does not advance the counter.
    """

    def __init__(self, initial_time: float = 0.0) -> None:
        self._time = initial_time

    def now(self) -> float:
        return self._time

    def sleep(self, seconds: float) -> None:
        if seconds > 0:
            self._time += seconds


@dataclass(frozen=True)
class TickStats:
    """Statistics recorded for a single tick of the reflex loop."""

    tick_index: int
    started_at: float
    ended_at: float
    signals_processed: int
    controls_emitted: int
    sink_errors: int
    overrun: bool
    jitter_ms: float | None = None

    def duration_ms(self) -> float:
        """Return tick execution duration in milliseconds."""
        return (self.ended_at - self.started_at) * 1000.0


@dataclass(frozen=True)
class ReflexTelemetry:
    """Telemetry snapshot for ReflexLoop execution."""

    tick_index: int
    total_ticks: int
    overruns_total: int
    sink_errors_total: int
    signals_processed_total: int
    controls_emitted_total: int
    jitter_samples_count: int
    mean_jitter_ms: float | None = None
    max_jitter_ms: float | None = None


def compute_jitter_series(
    samples: Sequence[TickStats | float],
    rate_hz: float,
) -> list[float]:
    """Compute the unbiased tick interval jitter series in milliseconds.

    Accepts either a sequence of TickStats or a sequence of tick start timestamps in seconds.
    For N ticks (N >= 2), returns N - 1 interval jitter values:
        jitter[k] = ((t[k] - t[k-1]) - (1.0 / rate_hz)) * 1000.0.
    For N < 2, returns an empty list.

    All ticks in the input sequence are included without filtering, ensuring
    an unbiased jitter series with no systematic omission of quiet ticks.
    """
    if rate_hz <= 0:
        raise ValueError(f"rate_hz must be > 0, got {rate_hz}")
    if len(samples) < 2:
        return []

    nominal_s = 1.0 / rate_hz
    timestamps: list[float] = [
        s.started_at if isinstance(s, TickStats) else float(s)
        for s in samples
    ]

    jitter_series: list[float] = []
    for i in range(1, len(timestamps)):
        dt = timestamps[i] - timestamps[i - 1]
        jitter_ms = (dt - nominal_s) * 1000.0
        jitter_series.append(jitter_ms)

    return jitter_series


class ReflexLoop:
    """Fixed-rate reflex loop executing rules on polled signals and emitting control signals."""

    def __init__(
        self,
        *,
        rate_hz: float,
        sources: tuple[SignalSource, ...],
        sinks: tuple[ControlSink, ...],
        rules: Callable[[list[Signal], int, random.Random], list[ControlSignal]],
        clock: Clock | None = None,
        seed: int = 0,
        session: Session | None = None,
        summary_interval_ticks: int = 100,
        max_jitter_history: int = 10_000,
    ) -> None:
        if not (0 < rate_hz <= 100):
            raise ValueError(f"rate_hz must be in (0, 100], got {rate_hz}")
        if not callable(rules):
            raise ValueError("rules must be callable")  # noqa: TRY004
        if summary_interval_ticks <= 0:
            raise ValueError(f"summary_interval_ticks must be > 0, got {summary_interval_ticks}")
        if max_jitter_history <= 0:
            raise ValueError(f"max_jitter_history must be > 0, got {max_jitter_history}")

        self._rate_hz = rate_hz
        self._sources = tuple(sources)
        self._sinks = tuple(sinks)
        self._rules = rules
        self._clock = clock if clock is not None else RealClock()
        self._seed = seed
        self._session = session
        self._summary_interval_ticks = summary_interval_ticks
        self._max_jitter_history = max_jitter_history

        self._tick_index = 0
        self._rng = random.Random(seed)

        self._prev_tick_start: float | None = None
        self._jitter_series: collections.deque[float] = collections.deque(maxlen=max_jitter_history)
        self._overruns_total: int = 0
        self._sink_errors_total: int = 0
        self._signals_processed_total: int = 0
        self._controls_emitted_total: int = 0
        self._ticks_since_summary: int = 0

        self._last_source_error: BaseException | None = None
        self._last_sink_error: BaseException | None = None

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    @property
    def tick_index(self) -> int:
        with self._lock:
            return self._tick_index

    @property
    def total_ticks(self) -> int:
        with self._lock:
            return self._tick_index

    def get_jitter_series(self) -> list[float]:
        """Return the unbiased list of tick interval jitter samples in milliseconds."""
        with self._lock:
            return list(self._jitter_series)

    def telemetry(self) -> ReflexTelemetry:
        """Return a structured telemetry snapshot of reflex loop execution."""
        with self._lock:
            jitters = list(self._jitter_series)
            mean_j = sum(jitters) / len(jitters) if jitters else None
            max_j = max(jitters) if jitters else None
            return ReflexTelemetry(
                tick_index=self._tick_index,
                total_ticks=self._tick_index,
                overruns_total=self._overruns_total,
                sink_errors_total=self._sink_errors_total,
                signals_processed_total=self._signals_processed_total,
                controls_emitted_total=self._controls_emitted_total,
                jitter_samples_count=len(jitters),
                mean_jitter_ms=mean_j,
                max_jitter_ms=max_j,
            )

    def _emit_summary_event(self) -> None:
        if self._session is None:
            return
        with self._lock:
            if self._ticks_since_summary == 0:
                return
            period_ticks = self._ticks_since_summary
            self._ticks_since_summary = 0
            total_ticks = self._tick_index
            overruns_total = self._overruns_total
            sink_errors_total = self._sink_errors_total
            signals_processed_total = self._signals_processed_total
            controls_emitted_total = self._controls_emitted_total

        self._session.write_event({
            "event": "reflex_summary",
            "payload": {
                "tick_index": total_ticks,
                "total_ticks": total_ticks,
                "period_ticks": period_ticks,
                "overruns_total": overruns_total,
                "sink_errors_total": sink_errors_total,
                "signals_processed_total": signals_processed_total,
                "controls_emitted_total": controls_emitted_total,
            },
        })

    def flush_summary(self) -> None:
        """Flush any pending periodic summary event to the session log."""
        self._emit_summary_event()

    @property
    def last_source_error(self) -> BaseException | None:
        with self._lock:
            return self._last_source_error

    @property
    def last_sink_error(self) -> BaseException | None:
        with self._lock:
            return self._last_sink_error

    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive()

    def tick(self) -> TickStats:
        with self._lock:
            tick_start = self._clock.now()
            tick_idx = self._tick_index
            prev_start = self._prev_tick_start
            self._prev_tick_start = tick_start

            nominal_period = 1.0 / self._rate_hz
            if prev_start is not None:
                interval_s = tick_start - prev_start
                jitter_ms = (interval_s - nominal_period) * 1000.0
                self._jitter_series.append(jitter_ms)
            else:
                jitter_ms = 0.0

            # Derived per-tick RNG
            tick_rng = random.Random(self._rng.getrandbits(64))

        signals: list[Signal] = []
        for source in self._sources:
            try:
                batch = source.poll(tick_start)
                signals.extend(batch)
            except Exception as exc:  # noqa: BLE001
                with self._lock:
                    self._last_source_error = exc

        # Call developer-authored rules outside lock; propagate exceptions directly
        controls = self._rules(signals, tick_idx, tick_rng)

        sink_errors_count = 0
        for control in controls:
            for sink in self._sinks:
                try:
                    # Invoke sink callback OUTSIDE internal lock
                    sink.handle(control)
                except Exception as exc:  # noqa: BLE001
                    sink_errors_count += 1
                    with self._lock:
                        self._last_sink_error = exc

        tick_end = self._clock.now()
        overrun = (tick_end - tick_start) > (1.0 / self._rate_hz)

        stats = TickStats(
            tick_index=tick_idx,
            started_at=tick_start,
            ended_at=tick_end,
            signals_processed=len(signals),
            controls_emitted=len(controls),
            sink_errors=sink_errors_count,
            overrun=overrun,
            jitter_ms=jitter_ms,
        )

        with self._lock:
            self._tick_index += 1
            if overrun:
                self._overruns_total += 1
            self._sink_errors_total += sink_errors_count
            self._signals_processed_total += len(signals)
            self._controls_emitted_total += len(controls)

        # Emit session event if attached and interesting
        if self._session is not None:
            is_interesting = (
                overrun
                or stats.signals_processed > 0
                or stats.controls_emitted > 0
                or stats.sink_errors > 0
            )
            if is_interesting:
                self._session.write_event({
                    "event": "reflex_tick",
                    "payload": dataclasses.asdict(stats),
                })

            with self._lock:
                self._ticks_since_summary += 1
                should_emit = self._ticks_since_summary >= self._summary_interval_ticks

            if should_emit:
                self._emit_summary_event()

        return stats

    def run_for(self, duration_s: float, max_ticks: int = 10_000_000) -> list[TickStats]:
        start_time = self._clock.now()
        target_time = start_time + duration_s
        stats_list: list[TickStats] = []
        tick_period = 1.0 / self._rate_hz

        while self._clock.now() < target_time:
            if len(stats_list) >= max_ticks:
                raise ReflexError(f"run_for exceeded maximum tick count bound of {max_ticks}")

            stats = self.tick()
            stats_list.append(stats)

            # Sleep remaining time until next tick boundary if time remains
            elapsed = stats.ended_at - stats.started_at
            remaining_sleep = tick_period - elapsed
            if remaining_sleep > 0:
                self._clock.sleep(remaining_sleep)

        return stats_list

    def _run_loop(self) -> None:
        tick_period = 1.0 / self._rate_hz
        while not self._stop_event.is_set():
            stats = self.tick()
            elapsed = stats.ended_at - stats.started_at
            remaining_sleep = tick_period - elapsed
            if remaining_sleep > 0:
                self._clock.sleep(remaining_sleep)

    def start(self) -> None:
        with self._lock:
            if threading.current_thread() is self._thread:
                raise ReflexError("start() cannot be called from within the loop thread itself")

            if self._thread is not None and self._thread.is_alive():
                return

            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run_loop, daemon=True)
            self._thread.start()

    def stop(self, timeout_s: float = 1.0) -> None:
        self._stop_event.set()

        thread_to_join: threading.Thread | None = None
        with self._lock:
            if (
                self._thread is not None
                and self._thread.is_alive()
                and threading.current_thread() is not self._thread
            ):
                thread_to_join = self._thread

        if thread_to_join is not None:
            thread_to_join.join(timeout=timeout_s)

        with self._lock:
            has_pending = self._ticks_since_summary > 0
        if has_pending:
            self._emit_summary_event()

    def __enter__(self) -> Self:
        self.start()
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> Literal[False]:
        self.stop()
        return False
