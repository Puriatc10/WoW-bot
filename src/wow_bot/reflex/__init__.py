"""Reflex loop package for fixed-rate control signal processing."""

from wow_bot.reflex.controls import (
    CallbackSink,
    ControlKind,
    ControlSignal,
    ControlSink,
    NullControlSink,
)
from wow_bot.reflex.loop import (
    Clock,
    NullClock,
    RealClock,
    ReflexError,
    ReflexLoop,
    ReflexTelemetry,
    TickStats,
    compute_jitter_series,
)
from wow_bot.reflex.rules import (
    RULES_PRIORITY_ORDER,
    default_rules,
)
from wow_bot.reflex.signals import (
    ListSignalSource,
    NullSignalSource,
    Signal,
    SignalError,
    SignalSource,
)
from wow_bot.reflex.sinks import (
    ActuatorAbortSink,
    FSMController,
    FSMSink,
    NullFSMController,
    RecoverySink,
)
from wow_bot.reflex.sources import (
    FocusSignalSource,
    KillSwitchSignalSource,
    SafetySignalSource,
    SessionTimeoutSignalSource,
)

__all__ = [
    "RULES_PRIORITY_ORDER",
    "ActuatorAbortSink",
    "CallbackSink",
    "Clock",
    "ControlKind",
    "ControlSignal",
    "ControlSink",
    "FSMController",
    "FSMSink",
    "FocusSignalSource",
    "KillSwitchSignalSource",
    "ListSignalSource",
    "NullClock",
    "NullControlSink",
    "NullFSMController",
    "NullSignalSource",
    "RealClock",
    "RecoverySink",
    "ReflexError",
    "ReflexLoop",
    "ReflexTelemetry",
    "SafetySignalSource",
    "SessionTimeoutSignalSource",
    "Signal",
    "SignalError",
    "SignalSource",
    "TickStats",
    "compute_jitter_series",
    "default_rules",
]
