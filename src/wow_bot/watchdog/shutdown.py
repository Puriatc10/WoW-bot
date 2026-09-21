"""Graceful shutdown sequence manager for research execution runs (Task 8.4).

Provides deterministic, idempotent shutdown orchestration when critical operational
conditions occur, ensuring actuation release, log flushing, snapshot writing, and
orderly cleanup without directly exiting the process.
"""

from __future__ import annotations

import json
import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, runtime_checkable

from wow_bot.watchdog.health import HealthState

if TYPE_CHECKING:
    from wow_bot.actuation.actuator import Actuator
    from wow_bot.safety import SafetyLayer
    from wow_bot.session import Session
    from wow_bot.watchdog.health import HealthStateMachine


class ShutdownError(Exception):
    """Raised when graceful shutdown operations violate constraints."""


class ShutdownReason(str, Enum):
    """Reasons triggering graceful shutdown."""

    HEALTH_CRITICAL = "health_critical"
    LOOP_DETECTED = "loop_detected"
    SESSION_TIMEOUT = "session_timeout"
    KILL_SWITCH = "kill_switch"
    SAFETY_ABORT = "safety_abort"
    OPERATOR_REQUEST = "operator_request"
    UNKNOWN = "unknown"


class ShutdownStep(str, Enum):
    """Steps in the graceful shutdown sequence."""

    MARK_CRITICAL = "mark_critical"
    ABORT_SAFETY = "abort_safety"
    ABORT_ACTUATOR = "abort_actuator"
    WRITE_SNAPSHOT = "write_snapshot"
    FLUSH_SESSION = "flush_session"
    COMPLETE = "complete"


GRACEFUL_EXIT_CODE = 0
FAILED_EXIT_CODE = 1
ALREADY_EXITED_CODE = 2  # Reserved for future use (caller handling process lifecycle)


@dataclass(frozen=True)
class ShutdownStepResult:
    """Immutable execution outcome of a single shutdown step.

    Invariants:
      * succeeded is True implies error == ""
      * succeeded is False implies error != ""
      * duration_ms >= 0.0
    """

    step: ShutdownStep
    succeeded: bool
    duration_ms: float
    error: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.step, ShutdownStep):
            raise ValueError(f"step must be a ShutdownStep, got {type(self.step)}")
        if not isinstance(self.succeeded, bool):
            raise ValueError("succeeded must be a bool")
        if (
            isinstance(self.duration_ms, bool)
            or not isinstance(self.duration_ms, (int, float))
            or not math.isfinite(self.duration_ms)
            or self.duration_ms < 0.0
        ):
            raise ValueError("duration_ms must be a non-negative finite float")
        if not isinstance(self.error, str):
            raise ValueError("error must be a string")
        if self.succeeded and self.error != "":
            raise ValueError("succeeded=True requires error == ''")
        if not self.succeeded and self.error == "":
            raise ValueError("succeeded=False requires error != ''")


@dataclass(frozen=True)
class ShutdownReport:
    """Immutable report summarizing the outcome of the graceful shutdown sequence.

    Invariants:
      * exit_code is 0 or 1
      * len(steps) >= 1
      * finished_at >= started_at
    """

    reason: ShutdownReason
    exit_code: int
    steps: tuple[ShutdownStepResult, ...]
    started_at: float
    finished_at: float
    snapshot_path: str | None

    def __post_init__(self) -> None:
        if not isinstance(self.reason, ShutdownReason):
            raise ValueError(f"reason must be a ShutdownReason, got {type(self.reason)}")
        if isinstance(self.exit_code, bool) or self.exit_code not in (0, 1):
            raise ValueError("exit_code must be 0 or 1")
        if not isinstance(self.steps, tuple) or len(self.steps) < 1:
            raise ValueError("steps must be a non-empty tuple of ShutdownStepResult")
        for s in self.steps:
            if not isinstance(s, ShutdownStepResult):
                raise ValueError(f"All elements in steps must be ShutdownStepResult, got {type(s)}")
        if (
            isinstance(self.started_at, bool)
            or not isinstance(self.started_at, (int, float))
            or not math.isfinite(self.started_at)
            or self.started_at < 0.0
        ):
            raise ValueError("started_at must be a non-negative finite float")
        if (
            isinstance(self.finished_at, bool)
            or not isinstance(self.finished_at, (int, float))
            or not math.isfinite(self.finished_at)
            or self.finished_at < 0.0
        ):
            raise ValueError("finished_at must be a non-negative finite float")
        if self.finished_at < self.started_at:
            raise ValueError("finished_at must be >= started_at")
        if self.snapshot_path is not None and not isinstance(self.snapshot_path, str):
            raise ValueError("snapshot_path must be a str or None")

    def duration_ms(self) -> float:
        """Return total shutdown sequence duration in milliseconds."""
        return (self.finished_at - self.started_at) * 1000.0

    def to_json(self) -> dict[str, Any]:
        """Return a JSON-serializable dictionary representation of the report."""
        return {
            "reason": self.reason.value,
            "exit_code": self.exit_code,
            "steps": [
                {
                    "step": s.step.value,
                    "succeeded": s.succeeded,
                    "duration_ms": float(s.duration_ms),
                    "error": s.error,
                }
                for s in self.steps
            ],
            "started_at": float(self.started_at),
            "finished_at": float(self.finished_at),
            "duration_ms": float(self.duration_ms()),
            "snapshot_path": self.snapshot_path,
        }


@dataclass(frozen=True)
class ShutdownConfig:
    """Configuration for graceful shutdown step and total timeout parameters.

    Validation in __post_init__:
      * per_step_timeout_s > 0.0, else ValueError
      * total_timeout_s >= per_step_timeout_s, else ValueError
      * snapshot_filename non-empty, no "/", no "\\", no "..", else ValueError
      * All floats finite, else ValueError
    """

    per_step_timeout_s: float = 2.0
    total_timeout_s: float = 10.0
    write_snapshot: bool = True
    snapshot_filename: str = "shutdown.json"

    def __post_init__(self) -> None:
        if (
            isinstance(self.per_step_timeout_s, bool)
            or not isinstance(self.per_step_timeout_s, (int, float))
            or not math.isfinite(self.per_step_timeout_s)
            or self.per_step_timeout_s <= 0.0
        ):
            raise ValueError("per_step_timeout_s must be a finite float > 0.0")
        if (
            isinstance(self.total_timeout_s, bool)
            or not isinstance(self.total_timeout_s, (int, float))
            or not math.isfinite(self.total_timeout_s)
            or self.total_timeout_s < self.per_step_timeout_s
        ):
            raise ValueError("total_timeout_s must be a finite float >= per_step_timeout_s")
        if not isinstance(self.write_snapshot, bool):
            raise ValueError("write_snapshot must be a bool")
        if (
            not isinstance(self.snapshot_filename, str)
            or not self.snapshot_filename.strip()
            or "/" in self.snapshot_filename
            or "\\" in self.snapshot_filename
            or ".." in self.snapshot_filename
        ):
            raise ValueError("snapshot_filename must be a non-empty string without '/', '\\', or '..'")


@runtime_checkable
class Clock(Protocol):
    """Protocol for monotonic time sources."""

    def now(self) -> float:
        """Return current monotonic time reading."""
        ...


class RealClock:
    """Monotonic clock implementation wrapping time.monotonic."""

    def now(self) -> float:
        """Return monotonic time using time.monotonic."""
        return time.monotonic()


class NullClock:
    """Deterministic clock advancing time by a fixed step on each reading."""

    def __init__(self, start: float = 0.0, step: float = 0.01) -> None:
        if (
            isinstance(start, bool)
            or not isinstance(start, (int, float))
            or not math.isfinite(start)
            or start < 0.0
        ):
            raise ValueError("start must be a non-negative finite float")
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or step < 0.0
        ):
            raise ValueError("step must be a non-negative finite float")
        self._current = float(start)
        self._step = float(step)

    def set_step(self, step: float) -> None:
        """Adjust the step interval for subsequent now() calls."""
        if (
            isinstance(step, bool)
            or not isinstance(step, (int, float))
            or not math.isfinite(step)
            or step < 0.0
        ):
            raise ValueError("step must be a non-negative finite float")
        self._step = float(step)

    def now(self) -> float:
        """Return current time and advance by step."""
        val = self._current
        self._current += self._step
        return val


class GracefulShutdown:
    """Orchestrates deterministic, bounded, idempotent shutdown sequence."""

    def __init__(
        self,
        *,
        safety: SafetyLayer | Any | None = None,
        actuator: Actuator | Any | None = None,
        session: Session | Any | None = None,
        health: HealthStateMachine | Any | None = None,
        config: ShutdownConfig | None = None,
        clock: Clock | None = None,
    ) -> None:
        self._safety = safety
        self._actuator = actuator
        self._session = session
        self._health = health
        self._config = config if config is not None else ShutdownConfig()
        self._clock = clock if clock is not None else RealClock()

        self._lock = threading.Lock()
        self._cond = threading.Condition(self._lock)
        self._completed = False
        self._report: ShutdownReport | None = None
        self._shutting_down = False
        self._snapshot_path: str | None = None

    def is_completed(self) -> bool:
        """Return True if shutdown sequence has completed, False otherwise."""
        with self._lock:
            return self._completed

    def report(self) -> ShutdownReport | None:
        """Return cached ShutdownReport if completed, None otherwise."""
        with self._lock:
            return self._report

    def reset_for_tests(self) -> None:
        """Reset internal flags and state. For test use only."""
        with self._lock:
            self._completed = False
            self._report = None
            self._shutting_down = False
            self._snapshot_path = None

    def run(self, reason: ShutdownReason) -> ShutdownReport:
        """Execute graceful shutdown sequence idempotently."""
        if not isinstance(reason, ShutdownReason):
            raise ShutdownError(f"Invalid shutdown reason: {reason}")

        with self._cond:
            if self._completed:
                assert self._report is not None
                return self._report
            if self._shutting_down:
                while self._shutting_down:
                    self._cond.wait()
                assert self._report is not None
                return self._report
            self._shutting_down = True

        try:
            report = self._execute_sequence(reason)
        finally:
            with self._cond:
                self._completed = True
                self._report = report
                self._shutting_down = False
                self._cond.notify_all()

        return report

    def _execute_sequence(self, reason: ShutdownReason) -> ShutdownReport:
        started_at = self._clock.now()
        steps: list[ShutdownStepResult] = []

        # STEP 1 — MARK_CRITICAL
        self._run_step_mark_critical(reason, started_at, steps)

        # STEP 2 — ABORT_SAFETY
        self._run_step_abort_safety(reason, started_at, steps)

        # STEP 3 — ABORT_ACTUATOR
        self._run_step_abort_actuator(reason, started_at, steps)

        # STEP 4 — WRITE_SNAPSHOT
        self._run_step_write_snapshot(reason, started_at, steps)

        # STEP 5 — FLUSH_SESSION
        self._run_step_flush_session(reason, started_at, steps)

        # STEP 6 — COMPLETE
        steps.append(
            ShutdownStepResult(
                step=ShutdownStep.COMPLETE,
                succeeded=True,
                duration_ms=0.0,
                error="",
            )
        )

        finished_at = self._clock.now()
        all_succeeded = all(s.succeeded for s in steps)
        exit_code = GRACEFUL_EXIT_CODE if all_succeeded else FAILED_EXIT_CODE

        return ShutdownReport(
            reason=reason,
            exit_code=exit_code,
            steps=tuple(steps),
            started_at=started_at,
            finished_at=finished_at,
            snapshot_path=self._snapshot_path,
        )

    def _is_total_timeout_exceeded(self, started_at: float) -> bool:
        return (self._clock.now() - started_at) >= self._config.total_timeout_s

    def _get_health_state(self) -> Any:
        if self._health is None:
            return None
        st_attr = getattr(self._health, "current_state", None)
        if callable(st_attr):
            return st_attr()
        return st_attr

    def _run_step_mark_critical(
        self,
        reason: ShutdownReason,
        started_at: float,
        steps: list[ShutdownStepResult],
    ) -> None:
        if self._is_total_timeout_exceeded(started_at):
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.MARK_CRITICAL,
                    succeeded=False,
                    duration_ms=0.0,
                    error="total_timeout",
                )
            )
            return

        if self._health is None:
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.MARK_CRITICAL,
                    succeeded=True,
                    duration_ms=0.0,
                    error="",
                )
            )
            return

        step_start = self._clock.now()
        try:
            curr_st = self._get_health_state()
            is_critical = (
                curr_st == HealthState.CRITICAL
                or getattr(curr_st, "value", curr_st) == "critical"
            )
            if not is_critical:
                self._health.force_state(
                    HealthState.CRITICAL,
                    now=self._clock.now(),
                    reason=reason.value,
                )

            dur = (self._clock.now() - step_start) * 1000.0
            if dur > self._config.per_step_timeout_s * 1000.0:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.MARK_CRITICAL,
                        succeeded=False,
                        duration_ms=dur,
                        error="per_step_timeout",
                    )
                )
            else:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.MARK_CRITICAL,
                        succeeded=True,
                        duration_ms=dur,
                        error="",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            dur = (self._clock.now() - step_start) * 1000.0
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.MARK_CRITICAL,
                    succeeded=False,
                    duration_ms=dur,
                    error=str(exc) or repr(exc),
                )
            )

    def _run_step_abort_safety(
        self,
        reason: ShutdownReason,
        started_at: float,
        steps: list[ShutdownStepResult],
    ) -> None:
        if self._is_total_timeout_exceeded(started_at):
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_SAFETY,
                    succeeded=False,
                    duration_ms=0.0,
                    error="total_timeout",
                )
            )
            return

        if self._safety is None:
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_SAFETY,
                    succeeded=True,
                    duration_ms=0.0,
                    error="",
                )
            )
            return

        step_start = self._clock.now()
        try:
            if not self._safety.is_aborted():
                self._safety.abort(reason.value)

            dur = (self._clock.now() - step_start) * 1000.0
            if dur > self._config.per_step_timeout_s * 1000.0:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.ABORT_SAFETY,
                        succeeded=False,
                        duration_ms=dur,
                        error="per_step_timeout",
                    )
                )
            else:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.ABORT_SAFETY,
                        succeeded=True,
                        duration_ms=dur,
                        error="",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            dur = (self._clock.now() - step_start) * 1000.0
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_SAFETY,
                    succeeded=False,
                    duration_ms=dur,
                    error=str(exc) or repr(exc),
                )
            )

    def _run_step_abort_actuator(
        self,
        reason: ShutdownReason,
        started_at: float,
        steps: list[ShutdownStepResult],
    ) -> None:
        if self._is_total_timeout_exceeded(started_at):
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_ACTUATOR,
                    succeeded=False,
                    duration_ms=0.0,
                    error="total_timeout",
                )
            )
            return

        if self._actuator is None:
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_ACTUATOR,
                    succeeded=True,
                    duration_ms=0.0,
                    error="",
                )
            )
            return

        step_start = self._clock.now()
        try:
            self._actuator.abort(reason.value)
            is_aborted = self._actuator.is_aborted()
            dur = (self._clock.now() - step_start) * 1000.0
            if not is_aborted:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.ABORT_ACTUATOR,
                        succeeded=False,
                        duration_ms=dur,
                        error="actuator_not_aborted",
                    )
                )
            elif dur > self._config.per_step_timeout_s * 1000.0:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.ABORT_ACTUATOR,
                        succeeded=False,
                        duration_ms=dur,
                        error="per_step_timeout",
                    )
                )
            else:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.ABORT_ACTUATOR,
                        succeeded=True,
                        duration_ms=dur,
                        error="",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            dur = (self._clock.now() - step_start) * 1000.0
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.ABORT_ACTUATOR,
                    succeeded=False,
                    duration_ms=dur,
                    error=str(exc) or repr(exc),
                )
            )

    def _run_step_write_snapshot(
        self,
        reason: ShutdownReason,
        started_at: float,
        steps: list[ShutdownStepResult],
    ) -> None:
        if self._is_total_timeout_exceeded(started_at):
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.WRITE_SNAPSHOT,
                    succeeded=False,
                    duration_ms=0.0,
                    error="total_timeout",
                )
            )
            return

        if not self._config.write_snapshot or self._session is None:
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.WRITE_SNAPSHOT,
                    succeeded=True,
                    duration_ms=0.0,
                    error="",
                )
            )
            return

        step_start = self._clock.now()
        try:
            health_state_str: str | None = None
            if self._health is not None:
                st = self._get_health_state()
                health_state_str = getattr(st, "value", str(st)) if st is not None else None

            safety_dict: dict[str, Any] | None = None
            if self._safety is not None:
                abort_reason = (
                    self._safety.abort_reason()
                    if callable(getattr(self._safety, "abort_reason", None))
                    else getattr(self._safety, "abort_reason", None)
                )
                safety_dict = {
                    "abort_reason": abort_reason,
                    "is_aborted": bool(self._safety.is_aborted()),
                }

            actuator_dict: dict[str, Any] | None = None
            if self._actuator is not None:
                actuator_dict = {
                    "is_aborted": bool(self._actuator.is_aborted()),
                }

            health_dict: dict[str, Any] | None = None
            if self._health is not None:
                health_dict = {
                    "current_state": health_state_str,
                }

            snapshot_dict: dict[str, Any] = {
                "actuator": actuator_dict,
                "health": health_dict,
                "reason": reason.value,
                "safety": safety_dict,
                "started_at": float(started_at),
                "steps_so_far": [
                    {
                        "duration_ms": float(s.duration_ms),
                        "error": s.error,
                        "step": s.step.value,
                        "succeeded": s.succeeded,
                    }
                    for s in steps
                ],
            }

            session_path = getattr(self._session, "path", Path("."))
            snapshot_file = session_path / self._config.snapshot_filename
            content = json.dumps(snapshot_dict, ensure_ascii=False, sort_keys=True) + "\n"
            with open(snapshot_file, "w", encoding="utf-8") as f:
                f.write(content)

            self._snapshot_path = str(snapshot_file.resolve())

            dur = (self._clock.now() - step_start) * 1000.0
            if dur > self._config.per_step_timeout_s * 1000.0:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.WRITE_SNAPSHOT,
                        succeeded=False,
                        duration_ms=dur,
                        error="per_step_timeout",
                    )
                )
            else:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.WRITE_SNAPSHOT,
                        succeeded=True,
                        duration_ms=dur,
                        error="",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            dur = (self._clock.now() - step_start) * 1000.0
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.WRITE_SNAPSHOT,
                    succeeded=False,
                    duration_ms=dur,
                    error=str(exc) or repr(exc),
                )
            )

    def _run_step_flush_session(
        self,
        reason: ShutdownReason,
        started_at: float,
        steps: list[ShutdownStepResult],
    ) -> None:
        if self._is_total_timeout_exceeded(started_at):
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.FLUSH_SESSION,
                    succeeded=False,
                    duration_ms=0.0,
                    error="total_timeout",
                )
            )
            return

        if self._session is None:
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.FLUSH_SESSION,
                    succeeded=True,
                    duration_ms=0.0,
                    error="",
                )
            )
            return

        step_start = self._clock.now()
        try:
            self._session.write_event({
                "event": "shutdown_complete",
                "reason": reason.value,
                "step_count": len(steps) + 2,
            })
            dur = (self._clock.now() - step_start) * 1000.0
            if dur > self._config.per_step_timeout_s * 1000.0:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.FLUSH_SESSION,
                        succeeded=False,
                        duration_ms=dur,
                        error="per_step_timeout",
                    )
                )
            else:
                steps.append(
                    ShutdownStepResult(
                        step=ShutdownStep.FLUSH_SESSION,
                        succeeded=True,
                        duration_ms=dur,
                        error="",
                    )
                )
        except Exception as exc:  # noqa: BLE001
            dur = (self._clock.now() - step_start) * 1000.0
            steps.append(
                ShutdownStepResult(
                    step=ShutdownStep.FLUSH_SESSION,
                    succeeded=False,
                    duration_ms=dur,
                    error=str(exc) or repr(exc),
                )
            )
