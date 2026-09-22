"""Unit tests for Graceful Shutdown sequence manager (Task 8.4)."""

from __future__ import annotations

import ast
import json
import threading
from pathlib import Path
from typing import Any

import pytest

from wow_bot.watchdog.health import HealthState
from wow_bot.watchdog.shutdown import (
    FAILED_EXIT_CODE,
    GRACEFUL_EXIT_CODE,
    GracefulShutdown,
    NullClock,
    ShutdownConfig,
    ShutdownReason,
    ShutdownReport,
    ShutdownStep,
    ShutdownStepResult,
)


class FakeSafety:
    """Fake SafetyLayer for testing graceful shutdown."""

    def __init__(
        self,
        aborted: bool = False,
        raise_on_abort: Exception | None = None,
        clock: NullClock | None = None,
        advance_clock_s: float = 0.0,
    ) -> None:
        self._aborted = aborted
        self._abort_calls: list[str] = []
        self._raise_on_abort = raise_on_abort
        self._abort_reason: str | None = None
        self._clock = clock
        self._advance_clock_s = advance_clock_s

    def is_aborted(self) -> bool:
        return self._aborted

    def abort_reason(self) -> str | None:
        return self._abort_reason

    def abort(self, reason: str) -> None:
        if self._clock is not None and self._advance_clock_s > 0.0:
            self._clock.set_step(self._advance_clock_s)
            self._clock.now()  # Advance clock during step
        self._abort_calls.append(reason)
        self._abort_reason = reason
        if self._raise_on_abort is not None:
            raise self._raise_on_abort
        self._aborted = True


class FakeActuator:
    """Fake Actuator for testing graceful shutdown."""

    def __init__(self, stay_unaborted: bool = False) -> None:
        self._aborted = False
        self._stay_unaborted = stay_unaborted
        self._abort_calls: list[str] = []

    def is_aborted(self) -> bool:
        if self._stay_unaborted:
            return False
        return self._aborted

    def abort(self, reason: str) -> None:
        self._abort_calls.append(reason)
        if not self._stay_unaborted:
            self._aborted = True


class FakeSession:
    """Fake Session for testing graceful shutdown event writing."""

    def __init__(self, path: Path, raise_on_write: Exception | None = None) -> None:
        self.path = path
        self.events: list[dict[str, Any]] = []
        self._raise_on_write = raise_on_write

    def write_event(self, event: dict[str, Any]) -> None:
        if self._raise_on_write is not None:
            raise self._raise_on_write
        self.events.append(event)


class FakeHealth:
    """Fake HealthStateMachine for testing graceful shutdown state marking."""

    def __init__(
        self,
        state: HealthState = HealthState.HEALTHY,
        raise_on_force: Exception | None = None,
    ) -> None:
        self._state = state
        self._force_calls: list[tuple[HealthState, float, str]] = []
        self._raise_on_force = raise_on_force

    def current_state(self) -> HealthState:
        return self._state

    def force_state(self, state: HealthState, *, now: float, reason: str) -> None:
        self._force_calls.append((state, now, reason))
        if self._raise_on_force is not None:
            raise self._raise_on_force
        self._state = state


# -----------------------------------------------------------------------------
# Configuration Validation Tests
# -----------------------------------------------------------------------------


def test_shutdown_config_invalid_per_step_timeout() -> None:
    """ShutdownConfig with per_step_timeout_s <= 0 raises ValueError."""
    with pytest.raises(ValueError, match="per_step_timeout_s"):
        ShutdownConfig(per_step_timeout_s=0.0)
    with pytest.raises(ValueError, match="per_step_timeout_s"):
        ShutdownConfig(per_step_timeout_s=-1.0)


def test_shutdown_config_invalid_total_timeout() -> None:
    """ShutdownConfig with total_timeout_s < per_step_timeout_s raises ValueError."""
    with pytest.raises(ValueError, match="total_timeout_s"):
        ShutdownConfig(per_step_timeout_s=5.0, total_timeout_s=4.0)


def test_shutdown_config_empty_snapshot_filename() -> None:
    """ShutdownConfig with an empty snapshot_filename raises ValueError."""
    with pytest.raises(ValueError, match="snapshot_filename"):
        ShutdownConfig(snapshot_filename="")
    with pytest.raises(ValueError, match="snapshot_filename"):
        ShutdownConfig(snapshot_filename="   ")


def test_shutdown_config_snapshot_filename_with_slash() -> None:
    """ShutdownConfig with snapshot_filename containing '/' raises ValueError."""
    with pytest.raises(ValueError, match="snapshot_filename"):
        ShutdownConfig(snapshot_filename="sub/shutdown.json")


def test_shutdown_config_snapshot_filename_with_parent_dir() -> None:
    """ShutdownConfig with snapshot_filename containing '..' raises ValueError."""
    with pytest.raises(ValueError, match="snapshot_filename"):
        ShutdownConfig(snapshot_filename="../shutdown.json")


# -----------------------------------------------------------------------------
# Invariant Validation Tests
# -----------------------------------------------------------------------------


def test_shutdown_step_result_invariants() -> None:
    """ShutdownStepResult invariant validation."""
    # Succeeded=True with non-empty error raises ValueError
    with pytest.raises(ValueError, match="succeeded=True requires error == ''"):
        ShutdownStepResult(
            step=ShutdownStep.MARK_CRITICAL,
            succeeded=True,
            duration_ms=1.0,
            error="unexpected error",
        )

    # Succeeded=False with empty error raises ValueError
    with pytest.raises(ValueError, match="succeeded=False requires error != ''"):
        ShutdownStepResult(
            step=ShutdownStep.MARK_CRITICAL,
            succeeded=False,
            duration_ms=1.0,
            error="",
        )

    # Negative duration_ms raises TypeError (type/range check)
    with pytest.raises(TypeError, match="duration_ms"):
        ShutdownStepResult(
            step=ShutdownStep.MARK_CRITICAL,
            succeeded=True,
            duration_ms=-0.5,
            error="",
        )


def test_shutdown_report_invariants() -> None:
    """ShutdownReport invariant validation."""
    step_ok = ShutdownStepResult(
        step=ShutdownStep.COMPLETE, succeeded=True, duration_ms=0.0, error=""
    )

    # exit_code not in {0, 1} raises ValueError
    with pytest.raises(ValueError, match="exit_code"):
        ShutdownReport(
            reason=ShutdownReason.HEALTH_CRITICAL,
            exit_code=2,
            steps=(step_ok,),
            started_at=0.0,
            finished_at=1.0,
            snapshot_path=None,
        )

    # empty steps raises TypeError (type check)
    with pytest.raises(TypeError, match="steps"):
        ShutdownReport(
            reason=ShutdownReason.HEALTH_CRITICAL,
            exit_code=0,
            steps=(),
            started_at=0.0,
            finished_at=1.0,
            snapshot_path=None,
        )

    # finished_at < started_at raises ValueError
    with pytest.raises(ValueError, match="finished_at"):
        ShutdownReport(
            reason=ShutdownReason.HEALTH_CRITICAL,
            exit_code=0,
            steps=(step_ok,),
            started_at=10.0,
            finished_at=5.0,
            snapshot_path=None,
        )


# -----------------------------------------------------------------------------
# Sequence Execution Tests
# -----------------------------------------------------------------------------


def test_run_all_dependencies_healthy(tmp_path: Path) -> None:
    """run with all dependencies healthy returns exit_code=0 and all steps succeeded=True."""
    clock = NullClock()
    safety = FakeSafety()
    actuator = FakeActuator()
    session = FakeSession(path=tmp_path)
    health = FakeHealth()

    sd = GracefulShutdown(
        safety=safety,
        actuator=actuator,
        session=session,
        health=health,
        clock=clock,
    )

    report = sd.run(ShutdownReason.HEALTH_CRITICAL)

    assert report.exit_code == GRACEFUL_EXIT_CODE
    assert report.reason == ShutdownReason.HEALTH_CRITICAL
    assert len(report.steps) == 6

    step_names = [s.step for s in report.steps]
    assert step_names == [
        ShutdownStep.MARK_CRITICAL,
        ShutdownStep.ABORT_SAFETY,
        ShutdownStep.ABORT_ACTUATOR,
        ShutdownStep.WRITE_SNAPSHOT,
        ShutdownStep.FLUSH_SESSION,
        ShutdownStep.COMPLETE,
    ]
    assert all(s.succeeded for s in report.steps)
    assert all(s.error == "" for s in report.steps)


def test_run_skips_none_safety(tmp_path: Path) -> None:
    """run with safety=None skips ABORT_SAFETY with succeeded=True and error=""."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=None,
        actuator=FakeActuator(),
        session=FakeSession(tmp_path),
        health=FakeHealth(),
        clock=clock,
    )
    report = sd.run(ShutdownReason.SAFETY_ABORT)
    step = next(s for s in report.steps if s.step == ShutdownStep.ABORT_SAFETY)
    assert step.succeeded is True
    assert step.error == ""


def test_run_skips_none_actuator(tmp_path: Path) -> None:
    """run with actuator=None skips ABORT_ACTUATOR."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=None,
        session=FakeSession(tmp_path),
        health=FakeHealth(),
        clock=clock,
    )
    report = sd.run(ShutdownReason.OPERATOR_REQUEST)
    step = next(s for s in report.steps if s.step == ShutdownStep.ABORT_ACTUATOR)
    assert step.succeeded is True
    assert step.error == ""


def test_run_skips_none_session() -> None:
    """run with session=None skips WRITE_SNAPSHOT and FLUSH_SESSION."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=FakeActuator(),
        session=None,
        health=FakeHealth(),
        clock=clock,
    )
    report = sd.run(ShutdownReason.KILL_SWITCH)
    step_ws = next(s for s in report.steps if s.step == ShutdownStep.WRITE_SNAPSHOT)
    step_fs = next(s for s in report.steps if s.step == ShutdownStep.FLUSH_SESSION)
    assert step_ws.succeeded is True
    assert step_fs.succeeded is True


def test_run_skips_none_health(tmp_path: Path) -> None:
    """run with health=None skips MARK_CRITICAL."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=FakeActuator(),
        session=FakeSession(tmp_path),
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.SESSION_TIMEOUT)
    step = next(s for s in report.steps if s.step == ShutdownStep.MARK_CRITICAL)
    assert step.succeeded is True
    assert step.error == ""


def test_run_skips_write_snapshot_when_configured_false(tmp_path: Path) -> None:
    """run with config.write_snapshot=False skips WRITE_SNAPSHOT even when session is provided."""
    clock = NullClock()
    config = ShutdownConfig(write_snapshot=False)
    session = FakeSession(tmp_path)
    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=FakeActuator(),
        session=session,
        health=FakeHealth(),
        config=config,
        clock=clock,
    )
    report = sd.run(ShutdownReason.OPERATOR_REQUEST)
    step = next(s for s in report.steps if s.step == ShutdownStep.WRITE_SNAPSHOT)
    assert step.succeeded is True
    assert not (tmp_path / "shutdown.json").exists()


def test_run_calls_safety_abort_once() -> None:
    """run calls safety.abort exactly once with reason.value."""
    clock = NullClock()
    safety = FakeSafety()
    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    sd.run(ShutdownReason.SAFETY_ABORT)
    assert safety._abort_calls == ["safety_abort"]


def test_run_does_not_call_safety_abort_if_already_aborted() -> None:
    """run does NOT call safety.abort if safety.is_aborted() is already True."""
    clock = NullClock()
    safety = FakeSafety(aborted=True)
    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    sd.run(ShutdownReason.KILL_SWITCH)
    assert len(safety._abort_calls) == 0


def test_run_calls_actuator_abort_once() -> None:
    """run calls actuator.abort exactly once with reason.value."""
    clock = NullClock()
    actuator = FakeActuator()
    sd = GracefulShutdown(
        safety=None,
        actuator=actuator,
        session=None,
        health=None,
        clock=clock,
    )
    sd.run(ShutdownReason.LOOP_DETECTED)
    assert actuator._abort_calls == ["loop_detected"]


def test_run_marks_actuator_unaborted_failure() -> None:
    """run marks ABORT_ACTUATOR as succeeded=False with error='actuator_not_aborted' when actuator remains unaborted."""
    clock = NullClock()
    actuator = FakeActuator(stay_unaborted=True)
    sd = GracefulShutdown(
        safety=None,
        actuator=actuator,
        session=None,
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.LOOP_DETECTED)
    assert report.exit_code == FAILED_EXIT_CODE
    step = next(s for s in report.steps if s.step == ShutdownStep.ABORT_ACTUATOR)
    assert step.succeeded is False
    assert step.error == "actuator_not_aborted"


def test_run_marks_safety_abort_failure_and_continues() -> None:
    """run marks ABORT_SAFETY as succeeded=False when safety.abort raises; sequence continues."""
    clock = NullClock()
    safety = FakeSafety(raise_on_abort=RuntimeError("Safety device failure"))
    actuator = FakeActuator()
    sd = GracefulShutdown(
        safety=safety,
        actuator=actuator,
        session=None,
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.SAFETY_ABORT)
    assert report.exit_code == FAILED_EXIT_CODE
    step_safety = next(s for s in report.steps if s.step == ShutdownStep.ABORT_SAFETY)
    assert step_safety.succeeded is False
    assert "Safety device failure" in step_safety.error

    step_act = next(s for s in report.steps if s.step == ShutdownStep.ABORT_ACTUATOR)
    assert step_act.succeeded is True
    assert actuator.is_aborted() is True


def test_run_marks_write_snapshot_failure_and_continues(tmp_path: Path) -> None:
    """run marks WRITE_SNAPSHOT as succeeded=False when writing fails; sequence continues."""
    clock = NullClock()
    # Read-only or invalid session path causes write failure
    invalid_session_dir = tmp_path / "non_existent_dir" / "file_as_dir"
    invalid_session_dir.parent.mkdir(parents=True, exist_ok=True)
    invalid_session_dir.touch()  # Make it a file so child path fails to open

    session = FakeSession(path=invalid_session_dir)
    sd = GracefulShutdown(
        safety=None,
        actuator=None,
        session=session,
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.UNKNOWN)
    assert report.exit_code == FAILED_EXIT_CODE
    step_ws = next(s for s in report.steps if s.step == ShutdownStep.WRITE_SNAPSHOT)
    assert step_ws.succeeded is False
    assert len(step_ws.error) > 0

    step_complete = next(s for s in report.steps if s.step == ShutdownStep.COMPLETE)
    assert step_complete.succeeded is True


def test_run_with_any_failed_step_returns_exit_code_1() -> None:
    """run with any failed step returns exit_code=1."""
    clock = NullClock()
    safety = FakeSafety(raise_on_abort=ValueError("Safety error"))
    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.HEALTH_CRITICAL)
    assert report.exit_code == FAILED_EXIT_CODE


# -----------------------------------------------------------------------------
# Snapshot Verification Tests
# -----------------------------------------------------------------------------


def test_snapshot_file_creation_and_format(tmp_path: Path) -> None:
    """Snapshot file is created at session.path/snapshot_filename with valid UTF-8 JSON, sorted keys."""
    clock = NullClock()
    safety = FakeSafety()
    actuator = FakeActuator()
    session = FakeSession(tmp_path)
    health = FakeHealth()

    sd = GracefulShutdown(
        safety=safety,
        actuator=actuator,
        session=session,
        health=health,
        clock=clock,
    )
    report = sd.run(ShutdownReason.HEALTH_CRITICAL)

    snapshot_file = tmp_path / "shutdown.json"
    assert snapshot_file.exists()
    assert report.snapshot_path == str(snapshot_file.resolve())

    content_raw = snapshot_file.read_text(encoding="utf-8")
    data = json.loads(content_raw)

    # Check key ordering
    top_keys = list(data.keys())
    assert top_keys == sorted(top_keys)

    # Content structure checks
    assert data["reason"] == "health_critical"
    assert data["safety"] == {"abort_reason": "health_critical", "is_aborted": True}
    assert data["actuator"] == {"is_aborted": True}
    assert data["health"] == {"current_state": "critical"}
    assert isinstance(data["steps_so_far"], list)


def test_snapshot_does_not_contain_object_reprs(tmp_path: Path) -> None:
    """Snapshot does NOT contain object reprs or non-JSON values."""
    clock = NullClock()
    session = FakeSession(tmp_path)
    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=FakeActuator(),
        session=session,
        health=FakeHealth(),
        clock=clock,
    )
    sd.run(ShutdownReason.HEALTH_CRITICAL)

    snapshot_file = tmp_path / "shutdown.json"
    content_raw = snapshot_file.read_text(encoding="utf-8")
    data = json.loads(content_raw)

    # Ensure no Python object repr strings (e.g. <FakeSafety object at ...>) exist
    content_str = json.dumps(data)
    assert "<" not in content_str
    assert ">" not in content_str


# -----------------------------------------------------------------------------
# Idempotency & Concurrency Tests
# -----------------------------------------------------------------------------


def test_run_is_idempotent() -> None:
    """run is idempotent: second call returns the cached report and does NOT re-execute steps."""
    clock = NullClock()
    safety = FakeSafety()
    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    r1 = sd.run(ShutdownReason.KILL_SWITCH)
    r2 = sd.run(ShutdownReason.OPERATOR_REQUEST)

    assert r1 is r2
    assert len(safety._abort_calls) == 1


def test_run_concurrent_multithread_idempotency() -> None:
    """run is idempotent when called concurrently from two threads."""
    clock = NullClock()
    safety = FakeSafety()
    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )

    reports: list[ShutdownReport] = []

    def _worker() -> None:
        rep = sd.run(ShutdownReason.HEALTH_CRITICAL)
        reports.append(rep)

    t1 = threading.Thread(target=_worker)
    t2 = threading.Thread(target=_worker)

    t1.start()
    t2.start()
    t1.join()
    t2.join()

    assert len(reports) == 2
    assert reports[0] is reports[1]
    assert len(safety._abort_calls) == 1


# -----------------------------------------------------------------------------
# Timeout Tests
# -----------------------------------------------------------------------------


def test_total_timeout_exceeded() -> None:
    """total_timeout exceeded: NullClock advancing past total_timeout_s marks remaining steps total_timeout."""
    # Clock starts at 0.0, first read returns 0.0. Step 1 executes.
    # On next step read, clock steps by 15.0 s (exceeding total_timeout_s=10.0).
    clock = NullClock(start=0.0, step=15.0)
    config = ShutdownConfig(total_timeout_s=10.0, per_step_timeout_s=2.0)

    sd = GracefulShutdown(
        safety=FakeSafety(),
        actuator=FakeActuator(),
        session=None,
        health=FakeHealth(),
        config=config,
        clock=clock,
    )

    report = sd.run(ShutdownReason.HEALTH_CRITICAL)
    assert report.exit_code == FAILED_EXIT_CODE

    # Step 1 starts at t=0.0 (< 10.0), so it runs.
    # Steps 2-5 start when clock >= 15.0 (> 10.0), so they get error="total_timeout"
    timeout_steps = [s for s in report.steps if s.error == "total_timeout"]
    assert len(timeout_steps) >= 1


def test_per_step_timeout_exceeded() -> None:
    """Per-step timeout: a step taking longer than per_step_timeout_s is recorded succeeded=False."""
    clock = NullClock(start=0.0, step=0.001)
    config = ShutdownConfig(per_step_timeout_s=0.1, total_timeout_s=10.0)

    # FakeSafety advances the clock by 0.5 s during abort() call
    safety = FakeSafety(clock=clock, advance_clock_s=0.5)

    sd = GracefulShutdown(
        safety=safety,
        actuator=None,
        session=None,
        health=None,
        config=config,
        clock=clock,
    )

    report = sd.run(ShutdownReason.SAFETY_ABORT)
    assert report.exit_code == FAILED_EXIT_CODE

    step_safety = next(s for s in report.steps if s.step == ShutdownStep.ABORT_SAFETY)
    assert step_safety.succeeded is False
    assert step_safety.error == "per_step_timeout"


# -----------------------------------------------------------------------------
# Complete & Helper Method Tests
# -----------------------------------------------------------------------------


def test_complete_step_always_present_and_succeeded() -> None:
    """COMPLETE step is always present and always succeeded=True."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=None,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    report = sd.run(ShutdownReason.UNKNOWN)
    last_step = report.steps[-1]
    assert last_step.step == ShutdownStep.COMPLETE
    assert last_step.succeeded is True
    assert last_step.duration_ms == 0.0


def test_is_completed_and_report_properties() -> None:
    """is_completed and report properties before and after run."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=None,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    assert sd.is_completed() is False
    assert sd.report() is None

    rep = sd.run(ShutdownReason.OPERATOR_REQUEST)
    assert sd.is_completed() is True
    assert sd.report() is rep


def test_reset_for_tests() -> None:
    """reset_for_tests clears state and allows a second run."""
    clock = NullClock()
    sd = GracefulShutdown(
        safety=None,
        actuator=None,
        session=None,
        health=None,
        clock=clock,
    )
    r1 = sd.run(ShutdownReason.OPERATOR_REQUEST)
    assert sd.is_completed() is True

    sd.reset_for_tests()
    assert sd.is_completed() is False
    assert sd.report() is None

    r2 = sd.run(ShutdownReason.KILL_SWITCH)
    assert sd.is_completed() is True
    assert r1 is not r2


def test_shutdown_report_to_json_and_duration_ms() -> None:
    """ShutdownReport.to_json and duration_ms methods."""
    step_ok = ShutdownStepResult(
        step=ShutdownStep.COMPLETE, succeeded=True, duration_ms=0.0, error=""
    )
    rep = ShutdownReport(
        reason=ShutdownReason.HEALTH_CRITICAL,
        exit_code=0,
        steps=(step_ok,),
        started_at=10.0,
        finished_at=12.5,
        snapshot_path="/tmp/shutdown.json",
    )

    assert rep.duration_ms() == 2500.0

    js = rep.to_json()
    assert js["reason"] == "health_critical"
    assert js["exit_code"] == 0
    assert js["started_at"] == 10.0
    assert js["finished_at"] == 12.5
    assert js["duration_ms"] == 2500.0
    assert js["snapshot_path"] == "/tmp/shutdown.json"
    assert isinstance(js["steps"], list)
    assert js["steps"][0]["step"] == "complete"


# -----------------------------------------------------------------------------
# Static AST Analysis Tests
# -----------------------------------------------------------------------------


def test_static_ast_forbidden_imports() -> None:
    """Static AST check: shutdown.py does not import forbidden modules or LLM clients."""
    source_path = Path("src/wow_bot/watchdog/shutdown.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    forbidden_modules = {
        "wow_bot.strategist",
        "wow_bot.perception",
        "wow_bot.reflex",
        "wow_bot.executor",
        "wow_bot.world",
        "wow_bot.nav",
        "wow_bot.combat",
        "aiosqlite",
        "asyncio",
    }

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                name_lower = alias.name.lower()
                assert alias.name not in forbidden_modules, f"Forbidden import: {alias.name}"
                for term in ("ollama", "openai", "anthropic", "llm"):
                    assert term not in name_lower, f"Forbidden LLM import: {alias.name}"

        elif isinstance(node, ast.ImportFrom) and node.module:
            mod_lower = node.module.lower()
            assert node.module not in forbidden_modules, f"Forbidden import: {node.module}"
            for term in ("ollama", "openai", "anthropic", "llm"):
                assert term not in mod_lower, f"Forbidden LLM import: {node.module}"


def test_static_ast_no_forbidden_time_or_exit_calls() -> None:
    """Static AST check: shutdown.py does not call sys.exit, os._exit, os.kill, signal.raise_signal, or direct time reads in GracefulShutdown."""
    source_path = Path("src/wow_bot/watchdog/shutdown.py")
    tree = ast.parse(source_path.read_text(encoding="utf-8"))

    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and isinstance(node.func.value, ast.Name)
        ):
            owner = node.func.value.id
            attr = node.func.attr
            assert not (owner == "sys" and attr == "exit"), "sys.exit call forbidden"
            assert not (owner == "os" and attr in ("_exit", "kill")), f"os.{attr} call forbidden"
            assert not (owner == "signal" and attr == "raise_signal"), "signal.raise_signal call forbidden"
            assert not (owner == "time" and attr in ("time", "perf_counter")), f"time.{attr} call forbidden"
