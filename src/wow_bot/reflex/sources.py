"""Concrete poll-based signal sources for the reflex loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wow_bot.reflex.signals import Signal, SignalSource

if TYPE_CHECKING:
    from wow_bot.actuation.focus import FocusManager
    from wow_bot.kill_switch import KillSwitch
    from wow_bot.safety import SafetyLayer


class FocusSignalSource:
    """Poll-based signal source tracking FocusManager window focus state transitions."""

    NAME_LOST = "focus_lost"
    NAME_GAINED = "focus_gained"
    PRODUCED_SIGNAL_TYPES: frozenset[str] = frozenset({NAME_LOST, NAME_GAINED})

    def __init__(
        self,
        focus: FocusManager,
        *,
        emit_initial: bool = False,
    ) -> None:
        self._focus = focus
        self._emit_initial = emit_initial
        self._initialized = False
        self._last_state = False

    def poll(self, now: float) -> list[Signal]:
        is_focused = self._focus.is_focused()
        if not self._initialized:
            self._initialized = True
            self._last_state = is_focused
            if self._emit_initial and not is_focused:
                return [Signal(name=self.NAME_LOST, payload={}, ts=now)]
            return []

        signals: list[Signal] = []
        if self._last_state and not is_focused:
            signals.append(Signal(name=self.NAME_LOST, payload={}, ts=now))
        elif not self._last_state and is_focused:
            signals.append(Signal(name=self.NAME_GAINED, payload={}, ts=now))

        self._last_state = is_focused
        return signals


class SessionTimeoutSignalSource:
    """Poll-based signal source tracking monotonic session deadline expiration."""

    NAME = "session_timeout"
    PRODUCED_SIGNAL_TYPES: frozenset[str] = frozenset({NAME})

    def __init__(
        self,
        deadline_s: float,
        *,
        start_time_s: float,
    ) -> None:
        if deadline_s <= start_time_s:
            raise ValueError(
                f"deadline_s ({deadline_s}) must be greater than start_time_s ({start_time_s})"
            )
        self._deadline_s = deadline_s
        self._start_time_s = start_time_s
        self._emitted = False

    def poll(self, now: float) -> list[Signal]:
        if not self._emitted and now >= self._deadline_s:
            self._emitted = True
            return [
                Signal(
                    name=self.NAME,
                    payload={
                        "elapsed_s": now - self._start_time_s,
                        "max_s": self._deadline_s - self._start_time_s,
                    },
                    ts=now,
                )
            ]
        return []


class SafetySignalSource:
    """Poll-based signal source tracking SafetyLayer abort transitions."""

    NAME = "safety_abort"
    PRODUCED_SIGNAL_TYPES: frozenset[str] = frozenset({NAME})

    def __init__(self, safety: SafetyLayer) -> None:
        self._safety = safety
        self._emitted = False
        self._last_state = safety.is_aborted()

    def poll(self, now: float) -> list[Signal]:
        if self._emitted:
            return []

        is_aborted = self._safety.is_aborted()
        if self._last_state or is_aborted:
            self._emitted = True
            self._last_state = True
            return [
                Signal(
                    name=self.NAME,
                    payload={"reason": self._safety.abort_reason()},
                    ts=now,
                )
            ]

        self._last_state = is_aborted
        return []


class KillSwitchSignalSource:
    """Poll-based signal source tracking KillSwitch trigger transitions."""

    NAME = "kill_switch"
    PRODUCED_SIGNAL_TYPES: frozenset[str] = frozenset({NAME})

    def __init__(self, kill_switch: KillSwitch) -> None:
        self._kill_switch = kill_switch
        self._emitted = False
        self._last_state = kill_switch.is_triggered()

    def poll(self, now: float) -> list[Signal]:
        if self._emitted:
            return []

        is_triggered = self._kill_switch.is_triggered()
        if self._last_state or is_triggered:
            self._emitted = True
            self._last_state = True
            err = self._kill_switch.last_error()
            err_repr = repr(err) if err is not None else None
            return [
                Signal(
                    name=self.NAME,
                    payload={"error": err_repr},
                    ts=now,
                )
            ]

        self._last_state = is_triggered
        return []


class SourceError(Exception):
    """Raised when an underlying signal source poll raises an unhandled exception and re_raise is enabled."""


class SafeSignalSource:
    """Wraps a SignalSource to provide fail-closed error handling and escalation."""

    NAME_SOURCE_ERROR = "source_error"
    PRODUCED_SIGNAL_TYPES: frozenset[str] = frozenset({NAME_SOURCE_ERROR})

    def __init__(
        self,
        delegate: SignalSource,
        *,
        re_raise: bool = False,
        source_name: str | None = None,
    ) -> None:
        self._delegate = delegate
        self._re_raise = re_raise
        self._source_name = source_name or type(delegate).__name__
        self._error_count = 0
        self._last_error: Exception | None = None

    @property
    def delegate(self) -> SignalSource:
        """Return the underlying wrapped signal source."""
        return self._delegate

    @property
    def error_count(self) -> int:
        """Return cumulative count of errors captured by this wrapper."""
        return self._error_count

    @property
    def last_error(self) -> Exception | None:
        """Return the most recent exception raised by the delegate, if any."""
        return self._last_error

    def poll(self, now: float) -> list[Signal]:
        """Poll the delegate, catching exceptions to provide fail-closed escalation."""
        try:
            return self._delegate.poll(now)
        except Exception as exc:
            self._error_count += 1
            self._last_error = exc
            if self._re_raise:
                raise SourceError(
                    f"Signal source {self._source_name} failed: {exc}"
                ) from exc
            return [
                Signal(
                    name=self.NAME_SOURCE_ERROR,
                    payload={
                        "source": self._source_name,
                        "error": repr(exc),
                        "error_count": self._error_count,
                    },
                    ts=now,
                )
            ]


REGISTERED_SIGNAL_SOURCE_CLASSES: tuple[type, ...] = (
    FocusSignalSource,
    SessionTimeoutSignalSource,
    SafetySignalSource,
    KillSwitchSignalSource,
    SafeSignalSource,
)


def enumerate_registered_signal_types() -> set[str]:
    """Enumerate all signal types produced by registered sources across the system."""
    types: set[str] = {
        "position_stuck",
        "position_clear",
        "combat_interrupt",
        "combat_defensive",
        "combat_retreat",
    }
    for cls in REGISTERED_SIGNAL_SOURCE_CLASSES:
        if hasattr(cls, "PRODUCED_SIGNAL_TYPES"):
            types.update(cls.PRODUCED_SIGNAL_TYPES)
    return types

