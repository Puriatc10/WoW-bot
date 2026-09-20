"""Concrete poll-based signal sources for the reflex loop."""

from __future__ import annotations

from typing import TYPE_CHECKING

from wow_bot.reflex.signals import Signal

if TYPE_CHECKING:
    from wow_bot.actuation.focus import FocusManager
    from wow_bot.kill_switch import KillSwitch
    from wow_bot.safety import SafetyLayer


class FocusSignalSource:
    """Poll-based signal source tracking FocusManager window focus state transitions."""

    NAME_LOST = "focus_lost"
    NAME_GAINED = "focus_gained"

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
