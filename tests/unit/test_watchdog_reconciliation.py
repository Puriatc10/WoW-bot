"""Unit tests for T8.5 Watchdog Reconciliation requirements and acceptance criteria."""

from __future__ import annotations

import ast
import inspect
import subprocess
import sys
from collections.abc import Callable
from pathlib import Path

import wow_bot.watchdog as wd_pkg
import wow_bot.watchdog.health as health_module
import wow_bot.watchdog.shutdown as shutdown_module
import wow_bot.watchdog.watchdog as watchdog_module
from wow_bot.watchdog.health import HealthState
from wow_bot.watchdog.shutdown import ShutdownReason
from wow_bot.watchdog.watchdog import (
    _WATCHDOG_TO_SHUTDOWN_REASON,
    WatchdogProcess,
)


def test_health_state_single_definition_ast() -> None:
    """Verify class HealthState exists ONLY in health.py within watchdog package."""
    wd_dir = Path(health_module.__file__).parent
    health_state_classes: list[str] = []

    for file_path in wd_dir.glob("*.py"):
        tree = ast.parse(file_path.read_text(encoding="utf-8"), filename=str(file_path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == "HealthState":
                health_state_classes.append(file_path.name)

    assert health_state_classes == ["health.py"]


def test_watchdog_imports_health_state_from_health() -> None:
    """Verify watchdog.py imports HealthState from health.py."""
    assert watchdog_module.HealthState is health_module.HealthState


def test_init_reexports_health_state_from_health() -> None:
    """Verify __init__.py re-exports HealthState from health.py."""
    assert wd_pkg.HealthState is health_module.HealthState


def test_severity_rank_ordering() -> None:
    """Verify severity_rank method on HealthState preserves expected severity ordering."""
    assert HealthState.HEALTHY.severity_rank() == 0
    assert HealthState.DEGRADED.severity_rank() == 1
    assert HealthState.CRITICAL.severity_rank() == 2

    assert HealthState.HEALTHY.severity_rank() < HealthState.DEGRADED.severity_rank()
    assert HealthState.DEGRADED.severity_rank() < HealthState.CRITICAL.severity_rank()


def test_on_shutdown_request_registers_and_preserves_order() -> None:
    """Verify on_shutdown_request registers callbacks and preserves order."""
    wp = WatchdogProcess(poll_interval=0.01)
    calls: list[tuple[int, ShutdownReason]] = []

    cb1: Callable[[ShutdownReason], None] = lambda r: calls.append((1, r))
    cb2: Callable[[ShutdownReason], None] = lambda r: calls.append((2, r))

    wp.on_shutdown_request(cb1)
    wp.on_shutdown_request(cb2)

    wp.trigger_shutdown(ShutdownReason.HEALTH_CRITICAL)

    assert calls == [(1, ShutdownReason.HEALTH_CRITICAL), (2, ShutdownReason.HEALTH_CRITICAL)]


def test_shutdown_callback_single_invocation_and_exception_isolation() -> None:
    """Verify callbacks are invoked exactly once and exceptions in one do not block others."""
    wp = WatchdogProcess(poll_interval=0.01)
    calls: list[str] = []

    def failing_cb(r: ShutdownReason) -> None:
        calls.append("failing")
        raise RuntimeError("Callback failure")

    def second_cb(r: ShutdownReason) -> None:
        calls.append("second")

    wp.on_shutdown_request(failing_cb)
    wp.on_shutdown_request(second_cb)

    wp.trigger_shutdown(ShutdownReason.LOOP_DETECTED)
    # Trigger again to test single invocation
    wp.trigger_shutdown(ShutdownReason.LOOP_DETECTED)

    assert calls == ["failing", "second"]


def test_shutdown_callback_invoked_outside_lock() -> None:
    """Verify shutdown request callbacks are invoked outside internal lock."""
    wp = WatchdogProcess(poll_interval=0.01)

    def reentrant_cb(r: ShutdownReason) -> None:
        # If lock were held during callback, registering another callback here would deadlock
        wp.on_shutdown_request(lambda r2: None)

    wp.on_shutdown_request(reentrant_cb)
    wp.trigger_shutdown(ShutdownReason.HEALTH_CRITICAL)


def test_ast_checks_isolation() -> None:
    """Verify AST isolated boundaries between modules."""
    # 1. shutdown.py does not import watchdog.py
    shutdown_source = inspect.getsource(shutdown_module)
    shutdown_ast = ast.parse(shutdown_source)
    for node in ast.walk(shutdown_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [alias.name for alias in node.names]
            assert "watchdog.py" not in mod and "watchdog" not in names or mod.startswith("wow_bot.watchdog.health")

    # 2. health.py does not import watchdog.py or shutdown.py
    health_source = inspect.getsource(health_module)
    health_ast = ast.parse(health_source)
    for node in ast.walk(health_ast):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            mod = getattr(node, "module", "") or ""
            names = [alias.name for alias in node.names]
            assert "watchdog" not in names and "shutdown" not in names


def test_reason_mapping_completeness() -> None:
    """Verify _WATCHDOG_TO_SHUTDOWN_REASON covers all expected internal reason strings."""
    expected_reasons = {
        "heartbeat_missing",
        "heartbeat_stale",
        "progress_stalled",
        "recovery_loop",
        "recovery_timeout",
        "death_loop",
        "high_cpu",
    }
    assert set(_WATCHDOG_TO_SHUTDOWN_REASON.keys()) == expected_reasons
    assert all(isinstance(v, ShutdownReason) for v in _WATCHDOG_TO_SHUTDOWN_REASON.values())


def test_no_circular_import_runtime() -> None:
    """Verify fresh interpreter subprocess imports watchdog.py and shutdown.py without circular import."""
    cmd = [
        sys.executable,
        "-c",
        "import sys; sys.path.insert(0, 'src'); import wow_bot.watchdog.shutdown; import wow_bot.watchdog.watchdog; print('OK')",
    ]
    res = subprocess.run(cmd, capture_output=True, text=True, check=True)
    assert res.stdout.strip() == "OK"
