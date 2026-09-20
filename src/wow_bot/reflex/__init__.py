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

__all__ = [
    "CallbackSink",
    "Clock",
    "ControlKind",
    "ControlSignal",
    "ControlSink",
    "ListSignalSource",
    "NullClock",
    "NullControlSink",
    "NullSignalSource",
    "RealClock",
    "ReflexError",
    "ReflexLoop",
    "Signal",
    "SignalError",
    "SignalSource",
    "TickStats",
]
