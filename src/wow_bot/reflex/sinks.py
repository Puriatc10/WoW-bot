"""Reflex control sinks for driving actuator and state machine actions."""

from typing import Protocol

from wow_bot.actuation.actuator import Actuator
from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.session import Session


class FSMController(Protocol):
    """Protocol for finite state machine controllers handling reflex control actions."""

    def pause(self, reason: str) -> None:
        """Pause state machine execution for the specified reason."""
        ...

    def resume(self, reason: str) -> None:
        """Resume state machine execution for the specified reason."""
        ...

    def enter_recovery(self, reason: str) -> None:
        """Transition state machine into recovery state for the specified reason."""
        ...


class NullFSMController:
    """Mock FSMController recording call sequences in memory for tests and MOCK mode."""

    def __init__(self) -> None:
        self._calls: list[tuple[str, str]] = []

    def pause(self, reason: str) -> None:
        """Record pause invocation."""
        self._calls.append(("pause", reason))

    def resume(self, reason: str) -> None:
        """Record resume invocation."""
        self._calls.append(("resume", reason))

    def enter_recovery(self, reason: str) -> None:
        """Record enter_recovery invocation."""
        self._calls.append(("enter_recovery", reason))

    def calls(self) -> list[tuple[str, str]]:
        """Return a copy of all recorded calls as (method_name, reason) tuples."""
        return list(self._calls)


class ActuatorAbortSink:
    """Reflex control sink that triggers actuator abort on ABORT_ACTUATION signals."""

    def __init__(self, actuator: Actuator) -> None:
        self._actuator = actuator

    def handle(self, signal: ControlSignal) -> None:
        """Handle control signal, triggering actuator abort when kind is ABORT_ACTUATION."""
        if signal.kind == ControlKind.ABORT_ACTUATION:
            self._actuator.abort(signal.reason)


class FSMSink:
    """Reflex control sink that translates control signals into FSM state transitions."""

    def __init__(self, fsm: FSMController) -> None:
        self._fsm = fsm

    def handle(self, signal: ControlSignal) -> None:
        """Dispatch control signal to corresponding FSM controller action."""
        if signal.kind == ControlKind.PAUSE_FSM:
            self._fsm.pause(signal.reason)
        elif signal.kind == ControlKind.RESUME_FSM:
            self._fsm.resume(signal.reason)
        elif signal.kind == ControlKind.ENTER_RECOVERY:
            self._fsm.enter_recovery(signal.reason)


class RecoverySink:
    """Reflex control sink for handling recovery entry and session logging."""

    def __init__(
        self,
        fsm: FSMController,
        session: Session | None = None,
    ) -> None:
        self._fsm = fsm
        self._session = session

    def handle(self, signal: ControlSignal) -> None:
        """Dispatch ENTER_RECOVERY to FSM and write recovery event to session if provided."""
        if signal.kind == ControlKind.ENTER_RECOVERY:
            self._fsm.enter_recovery(signal.reason)
            if self._session is not None:
                self._session.write_event({
                    "event": "recovery_entered",
                    "reason": signal.reason,
                })
