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
import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Final, Protocol

from wow_bot.shared.logger import get_logger
from wow_bot.watchdog.health import HealthState
from wow_bot.watchdog.shutdown import ShutdownReason

log = get_logger("WATCHDOG")

#: Explicit mapping from internal Watchdog reason strings to ShutdownReason enum values.
_WATCHDOG_TO_SHUTDOWN_REASON: Final[dict[str, ShutdownReason]] = {
    "heartbeat_missing": ShutdownReason.HEALTH_CRITICAL,
    "heartbeat_stale": ShutdownReason.HEALTH_CRITICAL,
    "progress_stalled": ShutdownReason.HEALTH_CRITICAL,
    "recovery_loop": ShutdownReason.HEALTH_CRITICAL,
    "recovery_timeout": ShutdownReason.HEALTH_CRITICAL,
    "death_loop": ShutdownReason.HEALTH_CRITICAL,
    "high_cpu": ShutdownReason.HEALTH_CRITICAL,
}

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
    reason_queue: Any | None = None,
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
            raw_reason = report.reasons[0] if report.reasons else "unknown"
            mapped_reason = _WATCHDOG_TO_SHUTDOWN_REASON.get(
                raw_reason, ShutdownReason.HEALTH_CRITICAL
            )
            log.error(
                f"CRITICAL health detected! Reasons: {list(report.reasons)} | "
                f"Requesting graceful shutdown via Event."
            )
            if reason_queue is not None:
                try:
                    reason_queue.put_nowait(mapped_reason.value)
                except Exception:  # noqa: BLE001
                    pass
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
    """Process owner interface for managing the independent Watchdog supervisor subprocess.

    Role: DETECTION ONLY.
    The WatchdogProcess monitors process health and heartbeats in an independent
    subprocess. When health reaches CRITICAL, it requests a shutdown by setting
    the multiprocessing shutdown_event and invoking registered on_shutdown_request
    callbacks with a ShutdownReason. It does NOT execute shutdown operations
    directly.

    Contract:
    - Main process subscribes to shutdown requests via on_shutdown_request(callback).
    - When health reaches CRITICAL, all callbacks are invoked once with a ShutdownReason.
    - Callbacks are executed outside internal locks with exception isolation.
    - If the main process does not complete graceful shutdown within
      GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS, WatchdogProcess escalates to forced process
      termination via ProcessControl.
    """

    def __init__(
        self,
        message_queue: Any | None = None,
        shutdown_event: multiprocessing.synchronize.Event | None = None,
        stop_event: multiprocessing.synchronize.Event | None = None,
        supervised_process: Any | None = None,
        poll_interval: float = WATCHDOG_POLL_INTERVAL_SECONDS,
        reason_queue: Any | None = None,
    ) -> None:
        self._ctx = multiprocessing.get_context("spawn")
        self._owns_queue = message_queue is None
        self._message_queue = message_queue if message_queue is not None else self._ctx.Queue(maxsize=256)
        self._shutdown_event = (
            shutdown_event if shutdown_event is not None else self._ctx.Event()
        )
        self._stop_event = stop_event if stop_event is not None else self._ctx.Event()
        self._reason_queue = reason_queue if reason_queue is not None else self._ctx.Queue(maxsize=16)
        self._supervised_process = supervised_process
        self._poll_interval = poll_interval
        self._process: Any = None
        self._callbacks: list[Callable[[ShutdownReason], None]] = []
        self._lock = threading.Lock()
        self._triggered = False
        self._triggered_reason: ShutdownReason | None = None
        self._monitor_thread: threading.Thread | None = None

    @property
    def message_queue(self) -> Any:
        return self._message_queue

    @property
    def shutdown_event(self) -> multiprocessing.synchronize.Event:
        return self._shutdown_event

    @property
    def stop_event(self) -> multiprocessing.synchronize.Event:
        return self._stop_event

    def on_shutdown_request(
        self, callback: Callable[[ShutdownReason], None]
    ) -> None:
        """Register a callback to be invoked when Watchdog requests shutdown.

        Callbacks are invoked exactly once with a ShutdownReason when health
        evaluation reaches CRITICAL. Order of registration is preserved. Callbacks
        are invoked outside internal locks.
        """
        with self._lock:
            self._callbacks.append(callback)
            already_triggered = self._triggered
            reason = self._triggered_reason

        if already_triggered and reason is not None:
            try:
                callback(reason)
            except Exception as exc:  # noqa: BLE001
                log.error(f"Error in late shutdown request callback: {exc}")

    def trigger_shutdown(
        self, reason: ShutdownReason = ShutdownReason.HEALTH_CRITICAL
    ) -> None:
        """Explicitly trigger shutdown callbacks and set shutdown event."""
        self._shutdown_event.set()
        self._trigger_callbacks(reason)

    def _trigger_callbacks(self, reason: ShutdownReason) -> None:
        with self._lock:
            if self._triggered:
                return
            self._triggered = True
            self._triggered_reason = reason
            callbacks_to_call = list(self._callbacks)

        for cb in callbacks_to_call:
            try:
                cb(reason)
            except Exception as exc:  # noqa: BLE001
                log.error(f"Error in shutdown request callback: {exc}")

    def _start_monitor_thread(self) -> None:
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            return
        self._monitor_thread = threading.Thread(
            target=self._monitor_loop,
            name="watchdog_shutdown_monitor",
            daemon=True,
        )
        self._monitor_thread.start()

    def _monitor_loop(self) -> None:
        while not self._stop_event.is_set():
            if self._shutdown_event.wait(timeout=0.1):
                reason_val: str | None = None
                if self._reason_queue is not None:
                    try:
                        reason_val = self._reason_queue.get_nowait()
                    except Exception:  # noqa: BLE001
                        pass
                reason = (
                    ShutdownReason(reason_val)
                    if reason_val in [r.value for r in ShutdownReason]
                    else ShutdownReason.HEALTH_CRITICAL
                )
                self._trigger_callbacks(reason)
                break

    def start(self) -> None:
        """Start the independent Watchdog supervisor subprocess and monitor thread."""
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
                "reason_queue": self._reason_queue,
            },
            daemon=False,
        )
        self._process.start()
        self._start_monitor_thread()

    def stop(self) -> None:
        """Signal the Watchdog supervisor subprocess to stop cleanly."""
        self._stop_event.set()

    def join(self, timeout: float | None = None) -> None:
        """Wait for the Watchdog supervisor subprocess to exit."""
        if self._process is not None:
            self._process.join(timeout=timeout)

    def close(self) -> None:
        """Stop and reap the owned supervisor, escalating only after a grace period."""
        self.stop()
        if self._monitor_thread is not None and self._monitor_thread.is_alive():
            self._monitor_thread.join(timeout=1.0)
        if self._process is not None and self._process.pid is not None:
            self.join(timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)
            if self.is_alive:
                self._process.terminate()
                self.join(timeout=GRACEFUL_SHUTDOWN_TIMEOUT_SECONDS)
            if self.is_alive:
                raise RuntimeError("Watchdog process did not stop")
            self._process.close()
            self._process = None
        if self._owns_queue:
            # No consumer remains; do not block on a feeder flushing to a dead child.
            self._message_queue.cancel_join_thread()
            self._message_queue.close()
            if self._reason_queue is not None:
                self._reason_queue.cancel_join_thread()
                self._reason_queue.close()

    @property
    def is_alive(self) -> bool:
        """Check if Watchdog supervisor process is running."""
        return bool(self._process.is_alive()) if self._process is not None else False
