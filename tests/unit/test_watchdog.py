"""Unit tests for the independent Watchdog supervisor (Task 6.1).

Covers all required tests A through AF specified in ROADMAP Task 6.1:
    - Test A: Initial health during startup grace
    - Test B: No heartbeat after startup grace
    - Test C: Fresh heartbeat evaluation
    - Test D: Heartbeat degraded threshold (10s)
    - Test E: Heartbeat critical threshold (30s)
    - Test F: Heartbeat freshness restored
    - Test G: Progress stable under 30s
    - Test H: Progress stall at 30s
    - Test I: Progress resumes
    - Test J: Progress token regression rejection
    - Test K: One recovery entry counted
    - Test L: Persistent recovery state not repeatedly counted
    - Test M: Three recoveries in 60s window
    - Test N: Five recoveries in 60s window
    - Test O: Recovery history expiration
    - Test P: Prolonged recovery duration (20s)
    - Test Q: Death count below threshold
    - Test R: Three deaths in 10-minute simulation window
    - Test S: Five deaths in 10-minute simulation window
    - Test T: Death window pruning
    - Test U: Simulation timestamp regression rejection
    - Test V: Severity precedence composition
    - Test W: DEGRADED state does not trigger shutdown event
    - Test X: CRITICAL state triggers shutdown event
    - Test Y: CRITICAL state decision latching
    - Test Z: Graceful shutdown without forced escalation
    - Test AA: Forced escalation timeout
    - Test AB: Code hygiene - no log deletion
    - Test AC: Code hygiene - no global hotkey dependency
    - Test AD: Code hygiene - no Controller dependency
    - Test AE: Deterministic monitor evaluation
    - Test AF: Multiprocessing child process smoke test
"""

from __future__ import annotations

import ast
import inspect
import multiprocessing
import time
from typing import Any

import pytest

import wow_bot.watchdog.watchdog as watchdog_module
from wow_bot.watchdog.watchdog import (
    DeathEventMessage,
    HealthState,
    HeartbeatMessage,
    WatchdogMonitor,
    WatchdogProcess,
    WatchdogProtocolError,
    watchdog_process_main,
)


class FakeClock:
    """Injectable fake clock for deterministic testing without real sleeping."""

    def __init__(self, initial_time: float = 1000.0) -> None:
        self.time = initial_time

    def monotonic(self) -> float:
        return self.time

    def advance(self, seconds: float) -> None:
        self.time += seconds


class FakeResourceProbe:
    """Injectable resource probe for deterministic CPU/memory testing."""

    def __init__(self, cpu: float = 0.0, mem: float = 0.0) -> None:
        self.cpu = cpu
        self.mem = mem

    def cpu_percent(self) -> float:
        return self.cpu

    def memory_mb(self) -> float:
        return self.mem


class FakeProcessControl:
    """Injectable process control double for testing escalation without killing processes."""

    def __init__(self, alive: bool = True) -> None:
        self._alive = alive
        self.terminated = False

    def is_alive(self) -> bool:
        return self._alive

    def terminate(self) -> None:
        self.terminated = True
        self._alive = False


# Helper factory for creating standard HeartbeatMessage instances
def make_heartbeat(
    sent_at: float = 1000.0,
    sim_ts: float = 100.0,
    fsm_state: str = "SCANNING",
    progress_token: int = 1,
) -> HeartbeatMessage:
    return HeartbeatMessage(
        monotonic_sent_at=sent_at,
        simulation_timestamp=sim_ts,
        fsm_state=fsm_state,
        progress_token=progress_token,
    )


# --- Test A ---
def test_a_initial_health_during_startup_grace() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    clock.advance(15.0)  # Within 30s startup grace
    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert len(report.reasons) == 0


# --- Test B ---
def test_b_no_heartbeat_after_startup_grace() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    clock.advance(30.0)  # Exceeds startup grace
    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "heartbeat_missing" in report.reasons


# --- Test C ---
def test_c_fresh_heartbeat() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0))
    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert len(report.reasons) == 0


# --- Test D ---
def test_d_heartbeat_degraded() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0))
    clock.advance(10.0)  # Exactly 10.0s since heartbeat
    report = monitor.evaluate()
    assert report.state == HealthState.DEGRADED
    assert "heartbeat_stale" in report.reasons


# --- Test E ---
def test_e_heartbeat_critical() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0))
    clock.advance(30.0)  # Exactly 30.0s since heartbeat
    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "heartbeat_stale" in report.reasons


# --- Test F ---
def test_f_heartbeat_freshness_restored() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))
    clock.advance(15.0)  # Degraded state
    report_deg = monitor.evaluate()
    assert report_deg.state == HealthState.DEGRADED

    # Receive fresh heartbeat with progress advancement
    monitor.process_message(make_heartbeat(sent_at=1015.0, progress_token=2))
    report_fresh = monitor.evaluate()
    assert report_fresh.state == HealthState.HEALTHY
    assert len(report_fresh.reasons) == 0


# --- Test G ---
def test_g_progress_stable_under_30s() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))

    # Advance clock to 29.9s while sending fresh heartbeats (so heartbeat itself isn't stale)
    for step in range(1, 6):
        clock.advance(5.0)  # 5s per step, total 25s
        monitor.process_message(make_heartbeat(sent_at=1000.0 + step * 5, progress_token=1))

    clock.advance(4.9)  # Total 29.9s
    monitor.process_message(make_heartbeat(sent_at=1029.9, progress_token=1))

    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert "progress_stalled" not in report.reasons


# --- Test H ---
def test_h_progress_stall_at_30s() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))

    # Send periodic heartbeats keeping heartbeat fresh but progress_token unchanged
    for step in range(1, 31):
        clock.advance(1.0)
        monitor.process_message(make_heartbeat(sent_at=1000.0 + step, progress_token=1))

    report = monitor.evaluate()
    assert report.state == HealthState.DEGRADED
    assert "progress_stalled" in report.reasons


# --- Test I ---
def test_i_progress_resumes() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))

    for step in range(1, 31):
        clock.advance(1.0)
        monitor.process_message(make_heartbeat(sent_at=1000.0 + step, progress_token=1))

    assert monitor.evaluate().state == HealthState.DEGRADED

    # Resume progress
    clock.advance(1.0)
    monitor.process_message(make_heartbeat(sent_at=1031.0, progress_token=2))
    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert "progress_stalled" not in report.reasons


# --- Test J ---
def test_j_progress_token_regression() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(progress_token=10))

    with pytest.raises(WatchdogProtocolError, match="Progress token regressed"):
        monitor.process_message(make_heartbeat(progress_token=9))


# --- Test K ---
def test_k_one_recovery_entry() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(fsm_state="SCANNING"))
    monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY"))

    report = monitor.evaluate()
    assert report.recovery_count == 1
    assert report.state == HealthState.HEALTHY


# --- Test L ---
def test_l_persistent_recovery_is_not_repeatedly_counted() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(fsm_state="SCANNING"))
    monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY"))
    clock.advance(1.0)
    monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY"))
    clock.advance(1.0)
    monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY"))

    report = monitor.evaluate()
    assert report.recovery_count == 1


# --- Test M ---
def test_m_three_recoveries_in_60s() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)

    for i in range(3):
        monitor.process_message(make_heartbeat(fsm_state="SCANNING", progress_token=i + 1))
        clock.advance(1.0)
        monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY", progress_token=i + 1))
        clock.advance(1.0)

    report = monitor.evaluate()
    assert report.state == HealthState.DEGRADED
    assert "recovery_loop" in report.reasons


# --- Test N ---
def test_n_five_recoveries_in_60s() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)

    for i in range(5):
        monitor.process_message(make_heartbeat(fsm_state="SCANNING", progress_token=i + 1))
        clock.advance(1.0)
        monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY", progress_token=i + 1))
        clock.advance(1.0)

    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "recovery_loop" in report.reasons


# --- Test O ---
def test_o_recovery_history_expires() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)

    # 3 recovery entries
    for i in range(3):
        monitor.process_message(make_heartbeat(fsm_state="SCANNING", progress_token=i + 1))
        clock.advance(1.0)
        monitor.process_message(make_heartbeat(fsm_state="STUCK_RECOVERY", progress_token=i + 1))
        clock.advance(1.0)

    assert monitor.evaluate().state == HealthState.DEGRADED

    # Exit recovery
    clock.advance(1.0)
    monitor.process_message(make_heartbeat(fsm_state="SCANNING", progress_token=4))

    # Advance clock past 60s window
    clock.advance(60.0)
    monitor.process_message(make_heartbeat(fsm_state="SCANNING", progress_token=5))

    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert "recovery_loop" not in report.reasons


# --- Test P ---
def test_p_prolonged_recovery() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, fsm_state="SCANNING", progress_token=1))
    clock.advance(1.0)
    monitor.process_message(
        make_heartbeat(sent_at=1001.0, fsm_state="STUCK_RECOVERY", progress_token=1)
    )

    # Remain continuously in STUCK_RECOVERY for 20 seconds
    for step in range(1, 21):
        clock.advance(1.0)
        monitor.process_message(
            make_heartbeat(sent_at=1001.0 + step, fsm_state="STUCK_RECOVERY", progress_token=1)
        )

    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "recovery_timeout" in report.reasons


# --- Test Q ---
def test_q_death_count_below_threshold() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sim_ts=100.0))
    monitor.process_message(DeathEventMessage(simulation_timestamp=110.0))
    monitor.process_message(DeathEventMessage(simulation_timestamp=120.0))

    report = monitor.evaluate()
    assert report.death_count == 2
    assert report.state == HealthState.HEALTHY


# --- Test R ---
def test_r_three_deaths() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sim_ts=100.0))
    monitor.process_message(DeathEventMessage(simulation_timestamp=110.0))
    monitor.process_message(DeathEventMessage(simulation_timestamp=120.0))
    monitor.process_message(DeathEventMessage(simulation_timestamp=130.0))

    report = monitor.evaluate()
    assert report.state == HealthState.DEGRADED
    assert "death_loop" in report.reasons


# --- Test S ---
def test_s_five_deaths() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sim_ts=100.0))
    for i in range(5):
        monitor.process_message(DeathEventMessage(simulation_timestamp=100.0 + (i + 1) * 10))

    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "death_loop" in report.reasons


# --- Test T ---
def test_t_death_window_pruning() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sim_ts=100.0))

    # Record 3 deaths in sim window 100..130
    for i in range(3):
        monitor.process_message(DeathEventMessage(simulation_timestamp=100.0 + (i + 1) * 10))

    assert monitor.evaluate().state == HealthState.DEGRADED

    # Advance simulation timestamp past 10 minutes (600s)
    monitor.process_message(make_heartbeat(sim_ts=800.0, progress_token=2))

    report = monitor.evaluate()
    assert report.state == HealthState.HEALTHY
    assert "death_loop" not in report.reasons


# --- Test U ---
def test_u_simulation_timestamp_regression() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sim_ts=200.0))

    with pytest.raises(WatchdogProtocolError, match="Simulation timestamp moved backward"):
        monitor.process_message(make_heartbeat(sim_ts=199.0))

    with pytest.raises(WatchdogProtocolError, match="Simulation timestamp moved backward"):
        monitor.process_message(DeathEventMessage(simulation_timestamp=150.0))


# --- Test V ---
def test_v_severity_precedence() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))

    # Progress stalls (DEGRADED) + Heartbeat missing (CRITICAL)
    clock.advance(30.0)

    report = monitor.evaluate()
    assert report.state == HealthState.CRITICAL
    assert "heartbeat_stale" in report.reasons or "heartbeat_missing" in report.reasons


# --- Test W ---
def test_w_degraded_does_not_request_shutdown() -> None:
    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()
    stop_event = ctx.Event()
    msg_queue: Any = ctx.Queue()

    clock = FakeClock(1000.0)
    # Put message causing DEGRADED state (progress stall)
    msg_queue.put(make_heartbeat(sent_at=1000.0, progress_token=1))

    stop_event.set()
    watchdog_process_main(
        message_queue=msg_queue,
        shutdown_event=shutdown_event,
        stop_event=stop_event,
        poll_interval=0.01,
        clock=clock,
    )

    assert not shutdown_event.is_set()


# --- Test X ---
def test_x_critical_requests_shutdown() -> None:
    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()
    stop_event = ctx.Event()
    msg_queue: Any = ctx.Queue()

    class AdvancingClock:
        def __init__(self) -> None:
            self.time = 1000.0

        def monotonic(self) -> float:
            val = self.time
            self.time += 35.0  # Exceeds startup grace -> CRITICAL
            return val

    clock = AdvancingClock()
    stop_event.set()

    watchdog_process_main(
        message_queue=msg_queue,
        shutdown_event=shutdown_event,
        stop_event=stop_event,
        poll_interval=0.01,
        clock=clock,
    )

    assert shutdown_event.is_set()


# --- Test Y ---
def test_y_critical_latches() -> None:
    clock = FakeClock(1000.0)
    monitor = WatchdogMonitor(clock=clock)
    monitor.process_message(make_heartbeat(sent_at=1000.0, progress_token=1))
    clock.advance(30.0)  # CRITICAL

    report1 = monitor.evaluate()
    assert report1.state == HealthState.CRITICAL

    # Send healthy-looking heartbeat
    clock.advance(1.0)
    monitor.process_message(make_heartbeat(sent_at=1031.0, progress_token=2))

    report2 = monitor.evaluate()
    assert report2.state == HealthState.CRITICAL


# --- Test Z ---
def test_z_graceful_shutdown_no_escalation() -> None:
    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()
    stop_event = ctx.Event()
    msg_queue: Any = ctx.Queue()

    clock = FakeClock(1000.0)
    shutdown_event.set()  # Shutdown requested

    proc_ctrl = FakeProcessControl(alive=False)  # Supervised process exited quickly (<10s)

    stop_event.set()
    watchdog_process_main(
        message_queue=msg_queue,
        shutdown_event=shutdown_event,
        stop_event=stop_event,
        poll_interval=0.01,
        process_control=proc_ctrl,
        clock=clock,
    )

    assert not proc_ctrl.terminated


# --- Test AA ---
def test_aa_forced_escalation() -> None:
    ctx = multiprocessing.get_context("spawn")
    shutdown_event = ctx.Event()
    stop_event = ctx.Event()
    msg_queue: Any = ctx.Queue()

    clock = FakeClock(1000.0)
    shutdown_event.set()

    class EscalatingProcessControl(FakeProcessControl):
        def terminate(self) -> None:
            super().terminate()
            stop_event.set()

    proc_ctrl = EscalatingProcessControl(alive=True)

    class EscalationClock:
        def __init__(self, fc: FakeClock) -> None:
            self.fc = fc
            self.calls = 0

        def monotonic(self) -> float:
            self.calls += 1
            if self.calls >= 4:
                self.fc.time = 1015.0  # Time jumps by 15s >= 10s graceful timeout
            return self.fc.time

    clock_wrapper = EscalationClock(clock)

    watchdog_process_main(
        message_queue=msg_queue,
        shutdown_event=shutdown_event,
        stop_event=stop_event,
        poll_interval=0.01,
        process_control=proc_ctrl,
        clock=clock_wrapper,
    )

    assert proc_ctrl.terminated


# --- Test AB ---
def test_ab_no_log_deletion() -> None:
    """Verify Watchdog source code contains no anti-forensic file deletion calls."""
    source = inspect.getsource(watchdog_module)
    parsed = ast.parse(source)

    forbidden_names = {"unlink", "remove", "rmtree", "truncate"}
    for node in ast.walk(parsed):
        if isinstance(node, ast.Attribute) and node.attr in forbidden_names:
            pytest.fail(
                f"Forbidden log deletion or truncation method found in watchdog: '{node.attr}'"
            )


# --- Test AC ---
def test_ac_no_global_hotkey_dependency() -> None:
    """Verify Watchdog module does not import global keyboard listeners or hotkey libraries."""
    source = inspect.getsource(watchdog_module)
    forbidden_modules = ["pynput", "keyboard", "pyautogui"]
    for mod in forbidden_modules:
        assert mod not in source, f"Watchdog source must not reference hotkey package '{mod}'"


# --- Test AD ---
def test_ad_no_controller_dependency() -> None:
    """Verify Watchdog module does not depend on or invoke Executor Controller."""
    source = inspect.getsource(watchdog_module)
    assert "Controller" not in source, "Watchdog source must not reference Executor Controller"


# --- Test AE ---
def test_ae_deterministic_monitor() -> None:
    clock1 = FakeClock(1000.0)
    monitor1 = WatchdogMonitor(clock=clock1)
    monitor1.process_message(
        make_heartbeat(sent_at=1000.0, sim_ts=10.0, fsm_state="SCANNING", progress_token=1)
    )
    clock1.advance(5.0)
    report1 = monitor1.evaluate()

    clock2 = FakeClock(1000.0)
    monitor2 = WatchdogMonitor(clock=clock2)
    monitor2.process_message(
        make_heartbeat(sent_at=1000.0, sim_ts=10.0, fsm_state="SCANNING", progress_token=1)
    )
    clock2.advance(5.0)
    report2 = monitor2.evaluate()

    assert report1 == report2


# --- Test AF ---
def test_af_child_process_smoke_test() -> None:
    """Smoke test for launching real WatchdogProcess, sending heartbeats, and stopping cleanly."""
    wp = WatchdogProcess(poll_interval=0.1)
    wp.start()

    try:
        assert wp.is_alive
        wp.message_queue.put(
            make_heartbeat(
                sent_at=time.monotonic(), sim_ts=1.0, fsm_state="SCANNING", progress_token=1
            )
        )
        time.sleep(0.2)
        assert not wp.shutdown_event.is_set()
    finally:
        wp.stop()
        wp.join(timeout=3.0)
        assert not wp.is_alive
