"""Reflex-to-FSM bridge adapter.

Provides an adapter implementing FSMController protocol to allow the Reflex
loop to drive FSM state transitions without direct module coupling.
"""

import threading
import time
from collections.abc import Callable

from wow_bot.executor.fsm_v2 import FSM, FSMError
from wow_bot.executor.states import FSMState
from wow_bot.reflex.controls import ControlKind, ControlSignal
from wow_bot.session import Session


class ReflexBridgeError(Exception):
    """Base exception for reflex bridge errors."""


class FSMReflexBridge:
    """Adapter implementing FSMController to drive FSM transitions from Reflex loop signals."""

    def __init__(
        self,
        fsm: FSM,
        *,
        clock: Callable[[], float] | None = None,
        session: Session | None = None,
    ) -> None:
        self._fsm = fsm
        self._clock: Callable[[], float] = clock if clock is not None else time.monotonic
        self._session = session

        self._lock = threading.Lock()
        self._call_count: int = 0
        self._last_dispatch_error: BaseException | None = None

    @property
    def call_count(self) -> int:
        """Return total number of bridge-level dispatch calls that resulted in an FSM invocation."""
        with self._lock:
            return self._call_count

    @property
    def last_dispatch_error(self) -> BaseException | None:
        """Return the last exception caught during tick_reflex_to_fsm or enter_recovery."""
        with self._lock:
            return self._last_dispatch_error

    def pause(self, reason: str) -> None:
        """Pause state machine execution if not already paused."""
        if self._fsm.is_paused:
            return

        now = self._clock()

        with self._lock:
            self._call_count += 1

        self._fsm.pause(reason, now=now)

        if self._session is not None:
            self._session.write_event(
                {
                    "event": "bridge_pause",
                    "reason": reason,
                }
            )

    def resume(self, reason: str) -> None:
        """Resume state machine execution if currently paused."""
        if not self._fsm.is_paused:
            return

        now = self._clock()

        with self._lock:
            self._call_count += 1

        self._fsm.resume(reason, now=now)

        if self._session is not None:
            self._session.write_event(
                {
                    "event": "bridge_resume",
                    "reason": reason,
                }
            )

    def enter_recovery(self, reason: str) -> None:
        """Transition state machine into recovery state if legal."""
        if self._fsm.current_state == FSMState.STUCK_RECOVERY:
            return

        now = self._clock()

        try:
            self._fsm.enter_recovery(reason, now=now)
        except FSMError as exc:
            with self._lock:
                self._last_dispatch_error = exc

            if self._session is not None:
                self._session.write_event(
                    {
                        "event": "bridge_recovery_rejected",
                        "reason": reason,
                        "error": str(exc),
                    }
                )
            return

        with self._lock:
            self._call_count += 1

    def tick_reflex_to_fsm(
        self,
        signals: list[ControlSignal],
    ) -> int:
        """Process a list of ControlSignals in order and dispatch to FSM.

        Returns the number of FSM calls actually made (state change or legal call).
        Rejected recovery attempts count as 0.
        """
        calls_before = self.call_count

        for signal in signals:
            if signal.kind == ControlKind.ABORT_ACTUATION:
                continue
            elif signal.kind == ControlKind.PAUSE_FSM:
                self.pause(signal.reason)
            elif signal.kind == ControlKind.RESUME_FSM:
                self.resume(signal.reason)
            elif signal.kind == ControlKind.ENTER_RECOVERY:
                self.enter_recovery(signal.reason)

        return self.call_count - calls_before
