"""Control signals and sinks for the reflex loop."""

from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class ControlKind(str, Enum):
    """Enumeration of valid reflex control actions."""

    ABORT_ACTUATION = "abort_actuation"
    PAUSE_FSM = "pause_fsm"
    RESUME_FSM = "resume_fsm"
    ENTER_RECOVERY = "enter_recovery"


@dataclass(frozen=True)
class ControlSignal:
    """Represents an outbound control action emitted by the reflex loop."""

    kind: ControlKind
    reason: str
    ts: float


class ControlSink(Protocol):
    """Protocol for sinks that handle control signals emitted by the reflex loop."""

    def handle(self, signal: ControlSignal) -> None:
        """Handle an emitted control signal."""
        ...


class NullControlSink:
    """A ControlSink that records all received control signals in memory."""

    def __init__(self) -> None:
        self._received: list[ControlSignal] = []

    def handle(self, signal: ControlSignal) -> None:
        self._received.append(signal)

    def received(self) -> list[ControlSignal]:
        return list(self._received)

    def received_kinds(self) -> list[ControlKind]:
        return [signal.kind for signal in self._received]


class CallbackSink:
    """A ControlSink that delegates handling to a callable."""

    def __init__(self, callback: Callable[[ControlSignal], None]) -> None:
        self._callback = callback

    def handle(self, signal: ControlSignal) -> None:
        self._callback(signal)
