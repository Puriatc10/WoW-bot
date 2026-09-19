"""Independent Watchdog Supervisor implementation (Task 6.1).

The Watchdog process is an independent system-health supervisor that observes
heartbeats, monitors FSM recovery loops, tracks death loops and liveness,
classifies operational health (HEALTHY, DEGRADED, CRITICAL), requests graceful
shutdown via multiprocessing Event, and escalates to forced termination only when
a graceful shutdown timeout is exceeded.

Public API:
    - :class:`HealthState`
    - :class:`HeartbeatMessage`
    - :class:`DeathEventMessage`
    - :type:`WatchdogMessage`
    - :class:`HealthReport`
    - :class:`WatchdogProtocolError`
    - :class:`Clock`
    - :class:`SystemClock`
    - :class:`ResourceProbe`
    - :class:`DummyResourceProbe`
    - :class:`ProcessControl`
    - :class:`WatchdogMonitor`
    - :class:`WatchdogProcess`
    - :func:`watchdog_process_main`
"""

from __future__ import annotations

import math
import multiprocessing
import multiprocessing.synchronize
import queue
import time
from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Final, Protocol

from wow_bot.shared.logger import get_logger

log = get_logger("WATCHDOG")

# Threshold constants frozen per Task 6.1 specifications
WATCHDOG_POLL_INTERVAL_SECONDS: Final[float] = 0.5
STARTUP_GRACE_SECONDS: Final[float] = 30.0
HEARTBEAT_DEGRADED_SECONDS: Final[float] = 10.0
HEARTBEAT_CRITICAL_SECONDS: Final[float] = 30.0
PROGRESS_DEGRADED_SECONDS: Final[float] = 30.0
RECOVERY_WINDOW_SECONDS: Final[float] = 60.0
RECOVERY_DEGRADED_COUNT: Final[int] = 3
RECOVERY_CRITICAL_COUNT: Final[int] = 5
RECOVERY_MAX_DURATION_SECONDS: Final[float] = 20.0
DEATH_LOOP_WINDOW_SECONDS: Final[float] = 600.0  # 10 minutes in simulation time
DEATH_LOOP_DEGRADED_COUNT: Final[int] = 3
DEATH_LOOP_CRITICAL_COUNT: Final[int] = 5
GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS: Final[float] = 10.0


class WatchdogProtocolError(ValueError):
    """Raised when an IPC message or state update violates Watchdog protocol invariants."""


class HealthState(Enum):
    """Explicit health state classification."""

    HEALTHY = auto()
    DEGRADED = auto()
    CRITICAL = auto()

    def severity_rank(self) -> int:
        """Return integer rank for severity comparison (HEALTHY=0 < DEGRADED=1 < CRITICAL=2)."""
        ranks = {
            HealthState.HEALTHY: 0,
            HealthState.DEGRADED: 1,
            HealthState.CRITICAL: 2,
        }
        return ranks[self]


@dataclass(frozen=True)
class HeartbeatMessage:
    """Immutable heartbeat IPC payload emitted by the main pipeline."""

    monotonic_sent_at: float
    simulation_timestamp: float
    fsm_state: str
    progress_token: int
    recent_death_count: int = 0

    def __post_init__(self) -> None:
        if (
            isinstance(self.monotonic_sent_at, bool)
            or not isinstance(self.monotonic_sent_at, (int, float))
            or not math.isfinite(float(self.monotonic_sent_at))
        ):
            raise WatchdogProtocolError(
                f"monotonic_sent_at must be finite float, got {self.monotonic_sent_at!r}"
            )
        if (
            isinstance(self.simulation_timestamp, bool)
            or not isinstance(self.simulation_timestamp, (int, float))
            or not math.isfinite(float(self.simulation_timestamp))
        ):
            raise WatchdogProtocolError(
                f"simulation_timestamp must be finite float, got {self.simulation_timestamp!r}"
            )
        if not isinstance(self.fsm_state, str) or not self.fsm_state.strip():
            raise WatchdogProtocolError(
                f"fsm_state must be a non-empty string, got {self.fsm_state!r}"
            )
        if isinstance(self.progress_token, bool) or not isinstance(self.progress_token, int):
            raise WatchdogProtocolError(
                f"progress_token must be an int, got {type(self.progress_token).__name__} ({self.progress_token!r})"
            )


@dataclass(frozen=True)
class DeathEventMessage:
    """Immutable death event IPC payload emitted when a player death is observed."""

    simulation_timestamp: float

    def __post_init__(self) -> None:
        if (
            isinstance(self.simulation_timestamp, bool)
            or not isinstance(self.simulation_timestamp, (int, float))
            or not math.isfinite(float(self.simulation_timestamp))
        ):
            raise WatchdogProtocolError(
                f"simulation_timestamp must be finite float, got {self.simulation_timestamp!r}"
            )


WatchdogMessage = HeartbeatMessage | DeathEventMessage


@dataclass(frozen=True)
class HealthReport:
    """Immutable health evaluation snapshot produced by WatchdogMonitor."""

    state: HealthState
    reasons: tuple[str, ...]
    heartbeat_age_seconds: float | None = None
    progress_age_seconds: float | None = None
    recovery_count: int = 0
    death_count: int = 0
    cpu_percent: float | None = None
    memory_mb: float | None = None


class Clock(Protocol):
    """Protocol abstraction for monotonic clock source."""

    def monotonic(self) -> float:
        ...


class SystemClock:
    """Production clock implementation delegating to time.monotonic()."""

    def monotonic(self) -> float:
        return time.monotonic()


class ResourceProbe(Protocol):
    """Protocol abstraction for system resource sampling."""

    def cpu_percent(self) -> float:
        ...

    def memory_mb(self) -> float:
        ...


class DummyResourceProbe:
    """Neutral resource probe used when external system monitoring is unavailable."""

    def cpu_percent(self) -> float:
        return 0.0

    def memory_mb(self) -> float:
        return 0.0


class ProcessControl(Protocol):
    """Protocol abstraction for checking process liveness and forcing termination."""

    def is_alive(self) -> bool:
        ...

    def terminate(self) -> None:
        ...


class MultiprocessingProcessControl:
    """ProcessControl adapter wrapping a multiprocessing process instance."""

    def __init__(self, process: Any) -> None:
        self._process = process

    def is_alive(self) -> bool:
        return bool(self._process.is_alive())

    def terminate(self) -> None:
        self._process.terminate()


class WatchdogMonitor:
    """Pure, deterministic core logic for evaluating system health.

    Maintains liveness, FSM recovery entry counters, progress token monotonicity,
    and death history without thread or process dependencies.
    """

    def __init__(
        self,
        clock: Clock | None = None,
        resource_probe: ResourceProbe | None = None,
    ) -> None:
        self._clock: Clock = clock if clock is not None else SystemClock()
        self._resource_probe: ResourceProbe = (
            resource_probe if resource_probe is not None else DummyResourceProbe()
        )
        self._start_monotonic: float = self._clock.monotonic()

        self._last_heartbeat_received_at: float | None = None
        self._last_heartbeat_msg: HeartbeatMessage | None = None
        self._last_progress_token: int | None = None
        self._last_progress_change_at: float | None = None
        self._last_fsm_state: str | None = None

        self._recovery_entry_timestamps: list[float] = []
        self._recovery_started_at: float | None = None

        self._death_sim_timestamps: list[float] = []
        self._last_sim_timestamp: float | None = None

        self._critical_latched: bool = False
        self._latched_reasons: list[str] = []

    def process_message(self, message: WatchdogMessage) -> None:
        """Update monitor state from an incoming IPC message.

        Raises:
            WatchdogProtocolError: if simulation timestamp or progress token regresses,
                or message payload is invalid.
        """
        recv_time = self._clock.monotonic()

        if isinstance(message, HeartbeatMessage):
            if (
                self._last_sim_timestamp is not None
                and message.simulation_timestamp < self._last_sim_timestamp
            ):
                raise WatchdogProtocolError(
                    f"Simulation timestamp moved backward: {message.simulation_timestamp} < {self._last_sim_timestamp}"
                )
            self._last_sim_timestamp = max(
                self._last_sim_timestamp if self._last_sim_timestamp is not None else 0.0,
                message.simulation_timestamp,
            )

            if self._last_progress_token is not None:
                if message.progress_token < self._last_progress_token:
                    raise WatchdogProtocolError(
                        f"Progress token regressed: {message.progress_token} < {self._last_progress_token}"
                    )
                if message.progress_token > self._last_progress_token:
                    self._last_progress_token = message.progress_token
                    self._last_progress_change_at = recv_time
            else:
                self._last_progress_token = message.progress_token
                self._last_progress_change_at = recv_time

            # Handle FSM state transitions
            if self._last_fsm_state is None:
                # First heartbeat: initialize state without fake recovery transition
                self._last_fsm_state = message.fsm_state
                if message.fsm_state == "STUCK_RECOVERY":
                    self._recovery_started_at = recv_time
            else:
                if self._last_fsm_state != "STUCK_RECOVERY" and message.fsm_state == "STUCK_RECOVERY":
                    self._recovery_entry_timestamps.append(recv_time)
                    self._recovery_started_at = recv_time
                elif message.fsm_state != "STUCK_RECOVERY":
                    self._recovery_started_at = None
                self._last_fsm_state = message.fsm_state

            self._last_heartbeat_received_at = recv_time
            self._last_heartbeat_msg = message

        elif isinstance(message, DeathEventMessage):
            if (
                self._last_sim_timestamp is not None
                and message.simulation_timestamp < self._last_sim_timestamp
            ):
                raise WatchdogProtocolError(
                    f"Simulation timestamp moved backward in death event: {message.simulation_timestamp} < {self._last_sim_timestamp}"
                )
            self._last_sim_timestamp = max(
                self._last_sim_timestamp if self._last_sim_timestamp is not None else 0.0,
                message.simulation_timestamp,
            )
            self._death_sim_timestamps.append(message.simulation_timestamp)

        else:
            raise WatchdogProtocolError(f"Unsupported watchdog message type: {type(message)!r}")

    def evaluate(self) -> HealthReport:
        """Evaluate operational health based on current state and timestamps."""
        now_mono = self._clock.monotonic()
        hb_age: float | None = None
        prg_age: float | None = None

        if self._critical_latched:
            if self._last_heartbeat_received_at is not None:
                hb_age = now_mono - self._last_heartbeat_received_at
            if self._last_progress_change_at is not None:
                prg_age = now_mono - self._last_progress_change_at
            return HealthReport(
                state=HealthState.CRITICAL,
                reasons=tuple(self._latched_reasons),
                heartbeat_age_seconds=hb_age,
                progress_age_seconds=prg_age,
                recovery_count=len(self._recovery_entry_timestamps),
                death_count=len(self._death_sim_timestamps),
                cpu_percent=self._resource_probe.cpu_percent(),
                memory_mb=self._resource_probe.memory_mb(),
            )

        reasons: list[str] = []
        highest_severity = HealthState.HEALTHY

        def _escalate(state: HealthState, reason: str) -> None:
            nonlocal highest_severity
            reasons.append(reason)
            if state.severity_rank() > highest_severity.severity_rank():
                highest_severity = state

        # 1. Heartbeat liveness check
        if self._last_heartbeat_received_at is None:
            time_since_start = now_mono - self._start_monotonic
            if time_since_start >= STARTUP_GRACE_SECONDS:
                _escalate(HealthState.CRITICAL, "heartbeat_missing")
        else:
            hb_age = now_mono - self._last_heartbeat_received_at
            if hb_age >= HEARTBEAT_CRITICAL_SECONDS:
                _escalate(HealthState.CRITICAL, "heartbeat_stale")
            elif hb_age >= HEARTBEAT_DEGRADED_SECONDS:
                _escalate(HealthState.DEGRADED, "heartbeat_stale")

        # 2. Progress stall check
        if self._last_progress_change_at is not None:
            prg_age = now_mono - self._last_progress_change_at
            if prg_age >= PROGRESS_DEGRADED_SECONDS:
                _escalate(HealthState.DEGRADED, "progress_stalled")

        # 3. Recovery loop check
        # Prune recovery entries older than RECOVERY_WINDOW_SECONDS
        window_cutoff = now_mono - RECOVERY_WINDOW_SECONDS
        self._recovery_entry_timestamps = [
            t for t in self._recovery_entry_timestamps if t >= window_cutoff
        ]
        rec_count = len(self._recovery_entry_timestamps)
        if rec_count >= RECOVERY_CRITICAL_COUNT:
            _escalate(HealthState.CRITICAL, "recovery_loop")
        elif rec_count >= RECOVERY_DEGRADED_COUNT:
            _escalate(HealthState.DEGRADED, "recovery_loop")

        # Prolonged recovery check
        if self._last_fsm_state == "STUCK_RECOVERY" and self._recovery_started_at is not None:
            rec_duration = now_mono - self._recovery_started_at
            if rec_duration >= RECOVERY_MAX_DURATION_SECONDS:
                _escalate(HealthState.CRITICAL, "recovery_timeout")

        # 4. Death loop check
        if self._last_sim_timestamp is not None:
            sim_cutoff = self._last_sim_timestamp - DEATH_LOOP_WINDOW_SECONDS
            self._death_sim_timestamps = [
                t for t in self._death_sim_timestamps if t >= sim_cutoff
            ]
        death_count = len(self._death_sim_timestamps)
        if death_count >= DEATH_LOOP_CRITICAL_COUNT:
            _escalate(HealthState.CRITICAL, "death_loop")
        elif death_count >= DEATH_LOOP_DEGRADED_COUNT:
            _escalate(HealthState.DEGRADED, "death_loop")

        # 5. Resource probing check
        cpu_val = self._resource_probe.cpu_percent()
        mem_val = self._resource_probe.memory_mb()
        if cpu_val >= 95.0:
            _escalate(HealthState.CRITICAL, "high_cpu")
        elif cpu_val >= 85.0:
            _escalate(HealthState.DEGRADED, "high_cpu")

        if highest_severity == HealthState.CRITICAL:
            self._critical_latched = True
            self._latched_reasons = list(reasons)

        return HealthReport(
            state=highest_severity,
            reasons=tuple(reasons),
            heartbeat_age_seconds=hb_age,
            progress_age_seconds=prg_age,
            recovery_count=rec_count,
            death_count=death_count,
            cpu_percent=cpu_val,
            memory_mb=mem_val,
        )


def watchdog_process_main(
    message_queue: Any,
    shutdown_event: multiprocessing.synchronize.Event,
    stop_event: multiprocessing.synchronize.Event,
    poll_interval: float = WATCHDOG_POLL_INTERVAL_SECONDS,
    process_control: ProcessControl | None = None,
    clock: Clock | None = None,
    resource_probe: ResourceProbe | None = None,
) -> None:
    """Main execution loop for independent Watchdog process.

    Polls message queue, evaluates health, sets shutdown_event when CRITICAL,
    and escalates to forced process termination if graceful shutdown times out.
    """
    log.info("Watchdog supervisor process started.")
    monitor = WatchdogMonitor(clock=clock, resource_probe=resource_probe)
    last_reported_state: HealthState | None = None
    shutdown_requested_at: float | None = None
    proc_clock = clock if clock is not None else SystemClock()

    while True:
        # Drain incoming queue messages
        while True:
            try:
                msg = message_queue.get_nowait()
            except queue.Empty:
                break

            try:
                monitor.process_message(msg)
            except WatchdogProtocolError as exc:
                log.error(f"Watchdog protocol error: {exc}")

        report = monitor.evaluate()

        if report.state != last_reported_state:
            prev_name = last_reported_state.name if last_reported_state else "INITIAL"
            log.info(
                f"Health transition: {prev_name} -> {report.state.name} | "
                f"reasons={list(report.reasons)}"
            )
            last_reported_state = report.state

        if report.state == HealthState.CRITICAL and not shutdown_event.is_set():
            log.error(
                f"CRITICAL health detected! Reasons: {list(report.reasons)} | "
                f"Requesting graceful shutdown via Event."
            )
            shutdown_event.set()

        if shutdown_event.is_set():
            now_mono = proc_clock.monotonic()
            if shutdown_requested_at is None:
                shutdown_requested_at = now_mono

            if process_control is not None and process_control.is_alive():
                elapsed = now_mono - shutdown_requested_at
                if elapsed >= GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS:
                    log.error(
                        f"Graceful shutdown timeout ({GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS}s) exceeded. "
                        f"Escalating: terminating supervised process."
                    )
                    process_control.terminate()

        if stop_event.is_set():
            break

        stop_event.wait(timeout=poll_interval)

    log.info("Watchdog supervisor process exiting cleanly.")


class WatchdogProcess:
    """Process owner interface for managing the independent Watchdog subprocess."""

    def __init__(
        self,
        message_queue: Any | None = None,
        shutdown_event: multiprocessing.synchronize.Event | None = None,
        stop_event: multiprocessing.synchronize.Event | None = None,
        supervised_process: Any | None = None,
        poll_interval: float = WATCHDOG_POLL_INTERVAL_SECONDS,
    ) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._message_queue = message_queue if message_queue is not None else self._ctx.Queue()
        self._shutdown_event = (
            shutdown_event if shutdown_event is not None else self._ctx.Event()
        )
        self._stop_event = stop_event if stop_event is not None else self._ctx.Event()
        self._supervised_process = supervised_process
        self._poll_interval = poll_interval
        self._process: Any = None

    @property
    def message_queue(self) -> Any:
        return self._message_queue

    @property
    def shutdown_event(self) -> multiprocessing.synchronize.Event:
        return self._shutdown_event

    @property
    def stop_event(self) -> multiprocessing.synchronize.Event:
        return self._stop_event

    def start(self) -> None:
        """Start the independent Watchdog supervisor subprocess."""
        if self._process is not None and bool(self._process.is_alive()):
            return

        proc_control = (
            MultiprocessingProcessControl(self._supervised_process)
            if self._supervised_process is not None
            else None
        )
        self._process = self._ctx.Process(
            target=watchdog_process_main,
            kwargs={
                "message_queue": self._message_queue,
                "shutdown_event": self._shutdown_event,
                "stop_event": self._stop_event,
                "poll_interval": self._poll_interval,
                "process_control": proc_control,
            },
            daemon=False,
        )
        self._process.start()

    def stop(self) -> None:
        """Signal the Watchdog supervisor subprocess to stop cleanly."""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the Watchdog supervisor subprocess to exit."""
        if self._process is not None:
            self._process.join(timeout=timeout)

    @property
    def is_alive(self) -> bool:
        """Check if Watchdog supervisor process is running."""
        return bool(self._process.is_alive()) if self._process is not None else False
