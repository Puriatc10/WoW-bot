"""Reference reflex rules mapping input signals to outbound control signals."""

import random
from collections.abc import Callable
from enum import Enum

from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.reflex.signals import Signal

RULES_PRIORITY_ORDER: tuple[str, ...] = (
    "kill_switch",
    "safety_abort",
    "session_timeout",
    "focus_lost",
    "focus_gained",
)


class SignalDisposition(str, Enum):
    """Disposition of an incoming signal in the reflex rule engine."""

    HANDLED = "handled"
    IGNORED = "ignored"


class UnknownSignalError(Exception):
    """Raised when an unknown signal type has no declared rule disposition."""


SIGNAL_DISPOSITIONS: dict[str, SignalDisposition] = {
    "kill_switch": SignalDisposition.HANDLED,
    "safety_abort": SignalDisposition.HANDLED,
    "source_error": SignalDisposition.HANDLED,
    "session_timeout": SignalDisposition.HANDLED,
    "focus_lost": SignalDisposition.HANDLED,
    "focus_gained": SignalDisposition.HANDLED,
    "position_stuck": SignalDisposition.HANDLED,
    "position_clear": SignalDisposition.IGNORED,
    "combat_interrupt": SignalDisposition.IGNORED,
    "combat_defensive": SignalDisposition.IGNORED,
    "combat_retreat": SignalDisposition.IGNORED,
}


def default_rules(
    signals: list[Signal],
    tick_index: int,
    rng: random.Random,
    *,
    catch_all: Callable[[Signal], ControlSignal | None] | None = None,
) -> list[ControlSignal]:
    """Evaluate input signals against priority rules to produce outbound control signals.

    Note: parameters `tick_index` and `rng` are accepted for signature compatibility
    with future rule functions, but are intentionally unused in this default reference rule.
    """
    # 0. Check all signals against declarative dispositions
    for sig in signals:
        if sig.name not in SIGNAL_DISPOSITIONS:
            if catch_all is not None:
                ca_res = catch_all(sig)
                if ca_res is not None:
                    return [ca_res]
                continue
            raise UnknownSignalError(f"Unknown signal type: {sig.name!r}")

    # Priority 1 — kill_switch
    kill_switch_signals = [s for s in signals if s.name == "kill_switch"]
    if kill_switch_signals:
        # Default timestamp if no triggering signals found
        ts = max((s.ts for s in kill_switch_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.ABORT_ACTUATION,
                reason="kill_switch",
                ts=ts,
            )
        ]

    # Priority 2 — safety_abort
    safety_abort_signals = [s for s in signals if s.name == "safety_abort"]
    if safety_abort_signals:
        # Default timestamp if no triggering signals found
        ts = max((s.ts for s in safety_abort_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.ABORT_ACTUATION,
                reason="safety_abort",
                ts=ts,
            )
        ]

    # Priority 3 — source_error
    source_error_signals = [s for s in signals if s.name == "source_error"]
    if source_error_signals:
        ts = max((s.ts for s in source_error_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.ABORT_ACTUATION,
                reason="source_error",
                ts=ts,
            )
        ]

    # Priority 4 — session_timeout
    session_timeout_signals = [s for s in signals if s.name == "session_timeout"]
    if session_timeout_signals:
        # Default timestamp if no triggering signals found
        ts = max((s.ts for s in session_timeout_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.PAUSE_FSM,
                reason="session_timeout",
                ts=ts,
            )
        ]

    # Priority 5 — focus_lost
    focus_lost_signals = [s for s in signals if s.name == "focus_lost"]
    if focus_lost_signals:
        # Default timestamp if no triggering signals found
        ts = max((s.ts for s in focus_lost_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.ABORT_ACTUATION,
                reason="focus_lost",
                ts=ts,
            ),
            ControlSignal(
                kind=ControlKind.PAUSE_FSM,
                reason="focus_lost",
                ts=ts,
            ),
        ]

    # Priority 6 — focus_gained
    focus_gained_signals = [s for s in signals if s.name == "focus_gained"]
    if focus_gained_signals:
        # Default timestamp if no triggering signals found
        ts = max((s.ts for s in focus_gained_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.RESUME_FSM,
                reason="focus_gained",
                ts=ts,
            )
        ]

    # Priority 7 — position_stuck (enter recovery)
    stuck_signals = [s for s in signals if s.name == "position_stuck"]
    if stuck_signals:
        ts = max((s.ts for s in stuck_signals), default=0.0)
        return [
            ControlSignal(
                kind=ControlKind.ENTER_RECOVERY,
                reason="position_stuck",
                ts=ts,
            )
        ]

    return []

