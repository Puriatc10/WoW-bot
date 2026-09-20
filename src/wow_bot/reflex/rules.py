"""Reference reflex rules mapping input signals to outbound control signals."""

import random

from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.reflex.signals import Signal

RULES_PRIORITY_ORDER: tuple[str, ...] = (
    "kill_switch",
    "safety_abort",
    "session_timeout",
    "focus_lost",
    "focus_gained",
)


def default_rules(
    signals: list[Signal],
    tick_index: int,
    rng: random.Random,
) -> list[ControlSignal]:
    """Evaluate input signals against priority rules to produce outbound control signals.

    Note: parameters `tick_index` and `rng` are accepted for signature compatibility
    with future rule functions, but are intentionally unused in this default reference rule.
    """
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

    # Priority 3 — session_timeout
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

    # Priority 4 — focus_lost
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

    # Priority 5 — focus_gained
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

    return []
