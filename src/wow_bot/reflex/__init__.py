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
    TickStats,
)
from wow_bot.reflex.signals import (
    ListSignalSource,
    NullSignalSource,
    Signal,
    SignalError,
    SignalSource,
)
from wow_bot.reflex.sources import (
    FocusSignalSource,
    KillSwitchSignalSource,
    SafetySignalSource,
    SessionTimeoutSignalSource,
)

__all__ = [
    "CallbackSink",
    "Clock",
    "ControlKind",
    "ControlSignal",
    "ControlSink",
    "FocusSignalSource",
    "KillSwitchSignalSource",
    "ListSignalSource",
    "NullClock",
    "NullControlSink",
    "NullSignalSource",
    "RealClock",
    "ReflexError",
    "ReflexLoop",
    "SafetySignalSource",
    "SessionTimeoutSignalSource",
    "Signal",
    "SignalError",
    "SignalSource",
    "TickStats",
]
