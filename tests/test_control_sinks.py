"""Unit tests for reflex control sinks."""

import ast
import json
from pathlib import Path

import pytest

from wow_bot.actuation.actuator import Actuator
from wow_bot.actuation.mapper import ActionResult, ActionStatus, Intent
from wow_bot.config import Config
from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.reflex.sinks import (
    ActuatorAbortSink,
    FSMController,
    FSMSink,
    NullFSMController,
    RecoverySink,
)
from wow_bot.session import Session


def _make_test_config(session_root: Path) -> Config:
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


class FakeActuator(Actuator):
    """Local FakeActuator implementing Actuator protocol for sink testing."""

    def __init__(self) -> None:
        self.abort_calls: list[str] = []

    def execute(
        self,
        intent: Intent,
        *,
        position: tuple[float, float],
    ) -> ActionResult:
        return ActionResult(status=ActionStatus.SUCCESS, latency_ms=0.0, notes="fake")

    def abort(self, reason: str) -> None:
        self.abort_calls.append(reason)

    def is_aborted(self) -> bool:
        return len(self.abort_calls) > 0

    def close(self) -> None:
        pass


class RaisingActuator(FakeActuator):
    """Local Actuator that raises an exception when abort is called."""

    def abort(self, reason: str) -> None:
        raise RuntimeError(f"Actuator abort failed: {reason}")


class RaisingFSMController(FSMController):
    """Local FSMController raising exceptions on all method calls."""

    def pause(self, reason: str) -> None:
        raise RuntimeError(f"Pause failed: {reason}")

    def resume(self, reason: str) -> None:
        raise RuntimeError(f"Resume failed: {reason}")

    def enter_recovery(self, reason: str) -> None:
        raise RuntimeError(f"Enter recovery failed: {reason}")


def test_actuator_abort_sink_calls_abort_once_on_abort_actuation() -> None:
    actuator = FakeActuator()
    sink = ActuatorAbortSink(actuator)
    signal = ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="test_reason", ts=1.0)

    sink.handle(signal)

    assert actuator.abort_calls == ["test_reason"]


def test_actuator_abort_sink_does_not_call_abort_on_other_kinds() -> None:
    actuator = FakeActuator()
    sink = ActuatorAbortSink(actuator)

    for kind in (ControlKind.PAUSE_FSM, ControlKind.RESUME_FSM, ControlKind.ENTER_RECOVERY):
        sink.handle(ControlSignal(kind=kind, reason="ignore", ts=1.0))

    assert actuator.abort_calls == []


def test_actuator_abort_sink_passes_reason_unchanged() -> None:
    actuator = FakeActuator()
    sink = ActuatorAbortSink(actuator)
    reason = "custom_abort_reason_123"

    sink.handle(ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason=reason, ts=1.0))

    assert actuator.abort_calls == [reason]


def test_actuator_abort_sink_propagates_exceptions() -> None:
    actuator = RaisingActuator()
    sink = ActuatorAbortSink(actuator)

    with pytest.raises(RuntimeError, match="Actuator abort failed: test_error"):
        sink.handle(ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="test_error", ts=1.0))


def test_fsm_sink_pause_passes_reason_unchanged() -> None:
    fsm = NullFSMController()
    sink = FSMSink(fsm)

    sink.handle(ControlSignal(kind=ControlKind.PAUSE_FSM, reason="pause_reason", ts=1.0))

    assert fsm.calls() == [("pause", "pause_reason")]


def test_fsm_sink_resume_passes_reason_unchanged() -> None:
    fsm = NullFSMController()
    sink = FSMSink(fsm)

    sink.handle(ControlSignal(kind=ControlKind.RESUME_FSM, reason="resume_reason", ts=1.0))

    assert fsm.calls() == [("resume", "resume_reason")]


def test_fsm_sink_enter_recovery_passes_reason_unchanged() -> None:
    fsm = NullFSMController()
    sink = FSMSink(fsm)

    sink.handle(ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="recovery_reason", ts=1.0))

    assert fsm.calls() == [("enter_recovery", "recovery_reason")]


def test_fsm_sink_does_not_call_fsm_on_abort_actuation() -> None:
    fsm = NullFSMController()
    sink = FSMSink(fsm)

    sink.handle(ControlSignal(kind=ControlKind.ABORT_ACTUATION, reason="abort", ts=1.0))

    assert fsm.calls() == []


def test_fsm_sink_propagates_exceptions() -> None:
    fsm = RaisingFSMController()
    sink = FSMSink(fsm)

    with pytest.raises(RuntimeError, match="Pause failed: fail"):
        sink.handle(ControlSignal(kind=ControlKind.PAUSE_FSM, reason="fail", ts=1.0))

    with pytest.raises(RuntimeError, match="Resume failed: fail"):
        sink.handle(ControlSignal(kind=ControlKind.RESUME_FSM, reason="fail", ts=1.0))

    with pytest.raises(RuntimeError, match="Enter recovery failed: fail"):
        sink.handle(ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="fail", ts=1.0))


def test_recovery_sink_calls_enter_recovery() -> None:
    fsm = NullFSMController()
    sink = RecoverySink(fsm)

    sink.handle(ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="stuck", ts=1.0))

    assert fsm.calls() == [("enter_recovery", "stuck")]


def test_recovery_sink_emits_event_when_session_provided(tmp_path: Path) -> None:
    fsm = NullFSMController()
    cfg = _make_test_config(tmp_path)
    session = Session.start(cfg)
    sink = RecoverySink(fsm, session=session)

    sink.handle(ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="path_blocked", ts=1.0))

    assert fsm.calls() == [("enter_recovery", "path_blocked")]
    events_path = session.path / "events.jsonl"
    lines = events_path.read_text(encoding="utf-8").strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert data["event"] == "recovery_entered"
    assert data["reason"] == "path_blocked"


def test_recovery_sink_does_not_emit_event_when_session_none() -> None:
    fsm = NullFSMController()
    sink = RecoverySink(fsm, session=None)

    sink.handle(ControlSignal(kind=ControlKind.ENTER_RECOVERY, reason="path_blocked", ts=1.0))

    assert fsm.calls() == [("enter_recovery", "path_blocked")]


def test_recovery_sink_does_not_call_fsm_on_other_kinds(tmp_path: Path) -> None:
    fsm = NullFSMController()
    cfg = _make_test_config(tmp_path)
    session = Session.start(cfg)
    sink = RecoverySink(fsm, session=session)

    for kind in (ControlKind.ABORT_ACTUATION, ControlKind.PAUSE_FSM, ControlKind.RESUME_FSM):
        sink.handle(ControlSignal(kind=kind, reason="other", ts=1.0))

    assert fsm.calls() == []
    events_path = session.path / "events.jsonl"
    assert events_path.read_text(encoding="utf-8") == ""


def test_null_fsm_controller_records_calls_in_order() -> None:
    fsm = NullFSMController()

    fsm.pause("reason1")
    fsm.resume("reason2")
    fsm.enter_recovery("reason3")

    calls = fsm.calls()
    assert calls == [
        ("pause", "reason1"),
        ("resume", "reason2"),
        ("enter_recovery", "reason3"),
    ]

    # Verify calls() returns a copy
    calls.clear()
    assert len(fsm.calls()) == 3


def test_static_ast_sinks_import_isolation() -> None:
    sinks_file = Path("src/wow_bot/reflex/sinks.py")
    tree = ast.parse(sinks_file.read_text("utf-8"))

    forbidden_modules = {
        "wow_bot.executor",
        "wow_bot.strategist",
        "wow_bot.reflex.stuck",
        "wow_bot.navigation",
        "wow_bot.combat",
        "wow_bot.world",
        "wow_bot.perception",
    }
    forbidden_substrings = ("ollama", "openai", "anthropic", "llm")

    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                mod_name = alias.name
                assert mod_name not in forbidden_modules, f"Forbidden import: {mod_name}"
                for sub in forbidden_substrings:
                    assert sub not in mod_name.lower(), f"Forbidden LLM module import: {mod_name}"
        elif isinstance(node, ast.ImportFrom):
            mod_name = node.module or ""
            for forbidden in forbidden_modules:
                assert not (mod_name == forbidden or mod_name.startswith(f"{forbidden}.")), (
                    f"Forbidden import from {mod_name}"
                )
            for sub in forbidden_substrings:
                assert sub not in mod_name.lower(), f"Forbidden LLM module import: {mod_name}"
