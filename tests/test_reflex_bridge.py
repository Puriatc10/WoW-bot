"""Tests for FSMReflexBridge adapter and tick_reflex_to_fsm helper."""

import ast
import copy
import inspect
import json
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from wow_bot.config import Config
from wow_bot.executor.fsm_v2 import FSM
from wow_bot.executor.reflex_bridge import FSMReflexBridge
from wow_bot.executor.states import FSMState
from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.session import Session


def make_test_config(session_root: Path) -> Config:
    """Helper to construct a valid Config instance with a custom session_root."""
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


@runtime_checkable
class RuntimeFSMController(Protocol):
    """Runtime checkable FSMController protocol for test verification."""

    def pause(self, reason: str) -> None: ...
    def resume(self, reason: str) -> None: ...
    def enter_recovery(self, reason: str) -> None: ...


class FakeClock:
    """Controllable clock for deterministic timestamp injection."""

    def __init__(self, start: float = 100.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, delta: float) -> None:
        self.now += delta


def _read_session_events(session_path: Path) -> list[dict[str, Any]]:
    events_path = session_path / "events.jsonl"
    if not events_path.exists():
        return []
    events = []
    with open(events_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def test_protocol_implementation() -> None:
    """Verify FSMReflexBridge implements FSMController protocol signatures."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    assert isinstance(bridge, RuntimeFSMController)

    for method_name in ("pause", "resume", "enter_recovery"):
        sig = inspect.signature(getattr(bridge, method_name))
        params = list(sig.parameters.values())
        assert len(params) == 1
        assert params[0].name == "reason"


def test_pause_running_fsm_emits_event(tmp_path: Path) -> None:
    """Verify pause() on running FSM transitions to PAUSED and emits bridge_pause event."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    fsm = FSM(session=session)
    clock = FakeClock(200.0)
    bridge = FSMReflexBridge(fsm, clock=clock, session=session)

    assert fsm.current_state == FSMState.IDLE
    bridge.pause("safety_alert")

    assert fsm.current_state == FSMState.PAUSED
    assert bridge.call_count == 1

    events = _read_session_events(session.path)
    bridge_events = [e for e in events if e.get("event") == "bridge_pause"]
    assert len(bridge_events) == 1
    assert bridge_events[0]["reason"] == "safety_alert"


def test_pause_already_paused_fsm_noop(tmp_path: Path) -> None:
    """Verify pause() on an already-paused FSM is a no-op."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    fsm = FSM(session=session)
    bridge = FSMReflexBridge(fsm, session=session)

    bridge.pause("first_pause")
    assert bridge.call_count == 1

    events_first = _read_session_events(session.path)
    bridge_pause_count_first = sum(1 for e in events_first if e.get("event") == "bridge_pause")
    assert bridge_pause_count_first == 1

    # Second pause call when already paused
    bridge.pause("second_pause")
    assert bridge.call_count == 1

    events_second = _read_session_events(session.path)
    bridge_pause_count_second = sum(1 for e in events_second if e.get("event") == "bridge_pause")
    assert bridge_pause_count_second == 1


def test_resume_from_paused_emits_event(tmp_path: Path) -> None:
    """Verify resume() from PAUSED returns to previous state and emits bridge_resume."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    fsm = FSM(session=session)
    bridge = FSMReflexBridge(fsm, session=session)

    bridge.pause("pause_before_resume")
    assert fsm.current_state == FSMState.PAUSED

    bridge.resume("user_resumed")
    assert fsm.current_state == FSMState.IDLE
    assert bridge.call_count == 2

    events = _read_session_events(session.path)
    resume_events = [e for e in events if e.get("event") == "bridge_resume"]
    assert len(resume_events) == 1
    assert resume_events[0]["reason"] == "user_resumed"


def test_resume_when_not_paused_noop(tmp_path: Path) -> None:
    """Verify resume() when FSM is not paused is a no-op."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    fsm = FSM(session=session)
    bridge = FSMReflexBridge(fsm, session=session)

    assert fsm.current_state == FSMState.IDLE
    bridge.resume("spurious_resume")

    assert fsm.current_state == FSMState.IDLE
    assert bridge.call_count == 0

    events = _read_session_events(session.path)
    resume_events = [e for e in events if e.get("event") == "bridge_resume"]
    assert len(resume_events) == 0


def test_enter_recovery_legal_state(tmp_path: Path) -> None:
    """Verify enter_recovery() from a legal state transitions to STUCK_RECOVERY without bridge event."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    # IDLE -> SCANNING -> MOVING_TO_TARGET -> enter_recovery is legal
    fsm = FSM(session=session)
    fsm._state = FSMState.MOVING_TO_TARGET
    bridge = FSMReflexBridge(fsm, session=session)

    bridge.enter_recovery("position_stuck")
    assert fsm.current_state == FSMState.STUCK_RECOVERY
    assert bridge.call_count == 1

    events = _read_session_events(session.path)
    # The FSM itself emits fsm_recovery_entered; bridge does not emit a bridge event on success.
    bridge_events = [e for e in events if e.get("event", "").startswith("bridge_")]
    assert len(bridge_events) == 0

    fsm_recovery_events = [e for e in events if e.get("event") == "fsm_recovery_entered"]
    assert len(fsm_recovery_events) == 1


def test_enter_recovery_illegal_state_rejected(tmp_path: Path) -> None:
    """Verify enter_recovery() from an illegal state is caught and emits bridge_recovery_rejected."""
    config = make_test_config(tmp_path)
    session = Session.start(config)
    fsm = FSM(session=session)
    # IDLE -> STUCK_RECOVERY is illegal
    assert fsm.current_state == FSMState.IDLE
    bridge = FSMReflexBridge(fsm, session=session)

    # Should catch FSMError without raising
    bridge.enter_recovery("illegal_stuck_request")

    assert fsm.current_state == FSMState.IDLE
    assert bridge.call_count == 0
    assert bridge.last_dispatch_error is not None
    assert "illegal transition" in str(bridge.last_dispatch_error)

    events = _read_session_events(session.path)
    rejected_events = [e for e in events if e.get("event") == "bridge_recovery_rejected"]
    assert len(rejected_events) == 1
    assert rejected_events[0]["reason"] == "illegal_stuck_request"
    assert "illegal transition" in rejected_events[0]["error"]


def test_enter_recovery_already_in_recovery_noop() -> None:
    """Verify enter_recovery() when already in STUCK_RECOVERY is a no-op."""
    fsm = FSM()
    fsm._state = FSMState.STUCK_RECOVERY
    bridge = FSMReflexBridge(fsm)

    bridge.enter_recovery("repeated_stuck")
    assert bridge.call_count == 0
    assert fsm.current_state == FSMState.STUCK_RECOVERY


def test_tick_reflex_to_fsm_empty_list() -> None:
    """Verify tick_reflex_to_fsm with empty list returns 0 and makes no calls."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    res = bridge.tick_reflex_to_fsm([])
    assert res == 0
    assert bridge.call_count == 0


def test_tick_reflex_to_fsm_single_pause() -> None:
    """Verify tick_reflex_to_fsm with single PAUSE_FSM returns 1 and pauses FSM."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)
    signal = ControlSignal(kind=ControlKind.PAUSE_FSM, reason="test_pause", ts=1.0)

    res = bridge.tick_reflex_to_fsm([signal])
    assert res == 1
    assert fsm.is_paused
    assert bridge.call_count == 1


def test_tick_reflex_to_fsm_single_resume() -> None:
    """Verify tick_reflex_to_fsm with single RESUME_FSM returns 1 when paused."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)
    fsm.pause("initial_pause", now=0.0)
    assert fsm.is_paused

    signal = ControlSignal(kind=ControlKind.RESUME_FSM, reason="test_resume", ts=2.0)
    res = bridge.tick_reflex_to_fsm([signal])
    assert res == 1
    assert not fsm.is_paused


def test_tick_reflex_to_fsm_single_enter_recovery() -> None:
    """Verify tick_reflex_to_fsm with single ENTER_RECOVERY returns 1 from legal state."""
    fsm = FSM()
    fsm._state = FSMState.MOVING_TO_TARGET
    bridge = FSMReflexBridge(fsm)

    signal = ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="test_recovery", ts=3.0)
    res = bridge.tick_reflex_to_fsm([signal])
    assert res == 1
    assert fsm.current_state == FSMState.STUCK_RECOVERY


def test_tick_reflex_to_fsm_single_abort_actuation() -> None:
    """Verify tick_reflex_to_fsm with single ABORT_ACTUATION returns 0 and ignores signal."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    signal = ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="test_abort", ts=4.0)
    res = bridge.tick_reflex_to_fsm([signal])
    assert res == 0
    assert bridge.call_count == 0
    assert fsm.current_state == FSMState.IDLE


def test_tick_reflex_to_fsm_pause_then_resume() -> None:
    """Verify [PAUSE_FSM, RESUME_FSM] sequence leaves FSM unpaused and returns 2."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    signals = [
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="pause", ts=1.0),
        ControlSignal(kind=ControlKind.RESUME_FSM, reason="resume", ts=2.0),
    ]
    res = bridge.tick_reflex_to_fsm(signals)
    assert res == 2
    assert not fsm.is_paused
    assert bridge.call_count == 2


def test_tick_reflex_to_fsm_resume_then_pause() -> None:
    """Verify [RESUME_FSM, PAUSE_FSM] sequence leaves FSM paused and returns 2 when starting paused."""
    fsm = FSM()
    fsm.pause("pre_pause", now=0.0)
    bridge = FSMReflexBridge(fsm)

    signals = [
        ControlSignal(kind=ControlKind.RESUME_FSM, reason="resume", ts=1.0),
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="pause", ts=2.0),
    ]
    res = bridge.tick_reflex_to_fsm(signals)
    assert res == 2
    assert fsm.is_paused


def test_tick_reflex_to_fsm_enter_recovery_then_pause() -> None:
    """Verify [ENTER_RECOVERY, PAUSE_FSM] from MOVING_TO_TARGET ends in PAUSED."""
    fsm = FSM()
    fsm._state = FSMState.MOVING_TO_TARGET
    bridge = FSMReflexBridge(fsm)

    signals = [
        ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="stuck", ts=1.0),
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="pause_after_stuck", ts=2.0),
    ]
    res = bridge.tick_reflex_to_fsm(signals)
    assert res == 2
    assert fsm.is_paused
    assert fsm.previous_state == FSMState.STUCK_RECOVERY


def test_tick_reflex_to_fsm_input_immutability() -> None:
    """Verify tick_reflex_to_fsm does not mutate the input list."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    signals = [
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="r1", ts=1.0),
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="r2", ts=2.0),
        ControlSignal(kind=ControlKind.RESUME_FSM, reason="r3", ts=3.0),
    ]
    snapshot = copy.deepcopy(signals)

    bridge.tick_reflex_to_fsm(signals)
    assert signals == snapshot


def test_last_dispatch_error_property() -> None:
    """Verify last_dispatch_error reflects the last caught FSMError or None."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    assert bridge.last_dispatch_error is None

    # Trigger illegal transition error
    bridge.enter_recovery("illegal_recovery")
    assert bridge.last_dispatch_error is not None
    assert isinstance(bridge.last_dispatch_error, Exception)


def test_call_count_property() -> None:
    """Verify call_count increments for each FSM invocation and excludes ABORT_ACTUATION."""
    fsm = FSM()
    fsm._state = FSMState.MOVING_TO_TARGET
    bridge = FSMReflexBridge(fsm)

    assert bridge.call_count == 0

    bridge.tick_reflex_to_fsm([
        ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="a1", ts=1.0),
        ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="r1", ts=2.0),
        ControlSignal(kind=ControlKind.PAUSE_FSM, reason="p1", ts=3.0),
    ])

    assert bridge.call_count == 2


def test_clock_is_only_time_source() -> None:
    """Verify injected clock is used as the timestamp passed to fsm.pause."""
    clock = FakeClock(555.5)

    class SpyFSM(FSM):
        def __init__(self) -> None:
            super().__init__()
            self.last_pause_now: float | None = None

        def pause(self, reason: str, *, now: float) -> None:
            self.last_pause_now = now
            super().pause(reason, now=now)

    spy_fsm = SpyFSM()
    bridge = FSMReflexBridge(spy_fsm, clock=clock)

    bridge.pause("check_clock")
    assert spy_fsm.last_pause_now == 555.5


def test_no_session_attached() -> None:
    """Verify all bridge operations work gracefully without a session."""
    fsm = FSM()
    fsm._state = FSMState.MOVING_TO_TARGET
    bridge = FSMReflexBridge(fsm, session=None)

    bridge.pause("p")
    bridge.resume("r")
    bridge.enter_recovery("rec")
    bridge.enter_recovery("illegal_rec")  # should catch without session event crash

    assert bridge.call_count == 3


def test_context_manager_not_required() -> None:
    """Verify FSMReflexBridge does not require context manager usage."""
    fsm = FSM()
    bridge = FSMReflexBridge(fsm)

    bridge.pause("cm_test")
    assert fsm.is_paused


def test_static_ast_import_boundaries() -> None:
    """Static AST check verifying reflex_bridge.py does not import forbidden modules."""
    bridge_path = Path("src/wow_bot/executor/reflex_bridge.py")
    assert bridge_path.exists()

    with open(bridge_path, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=str(bridge_path))

    forbidden_exact = {
        "wow_bot.reflex.rules",
        "wow_bot.reflex.loop",
        "wow_bot.reflex.sources",
        "wow_bot.actuation.actuator",
        "wow_bot.strategist",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }

    forbidden_substrings = ["ollama", "openai", "anthropic", "llm"]

    imported_modules: set[str] = set()

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported_modules.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_modules.add(node.module)
            for alias in node.names:
                imported_modules.add(f"{node.module}.{alias.name}")

    for mod in imported_modules:
        for forbidden in forbidden_exact:
            assert not mod.startswith(
                forbidden
            ), f"Forbidden import detected in reflex_bridge.py: {mod}"

        for sub in forbidden_substrings:
            assert (
                sub not in mod.lower()
            ), f"Forbidden LLM-related import detected in reflex_bridge.py: {mod}"
