"""Dynamic cooldown gate for WoW-bot strategist slow loop (Task 9.2).

Provides pure, deterministic rate-limiting, state-filtering, and timing-check authorization
before invoking the LLM strategist. Emits session events on manual overrides or allowed invocations.
Isolated from time reads, I/O, threads, and LLM modules.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING

from wow_bot.executor.states import FSMState

if TYPE_CHECKING:
    from wow_bot.session import Session


class CooldownError(Exception):
    """Raised when an error or invalid input occurs in cooldown gate operations."""


class CooldownDecision(str, Enum):
    """Possible outcomes of a strategist cooldown gate check."""

    ALLOWED = "allowed"
    BLOCKED_STATE = "blocked_state"
    BLOCKED_COOLDOWN = "blocked_cooldown"
    BLOCKED_RATE_LIMIT = "blocked_rate_limit"
    BLOCKED_MANUAL = "blocked_manual"
    FORCED_ALLOWED = "forced_allowed"


@dataclass(frozen=True)
class CooldownConfig:
    """Configuration for CooldownGate parameters and FSM state classifications."""

    min_interval_s: float = 15.0
    max_calls_per_minute: int = 4
    allow_while_busy: bool = False
    blocked_states: frozenset[FSMState] = frozenset({
        FSMState.COMBAT,
        FSMState.FLEEING,
        FSMState.STUCK_RECOVERY,
        FSMState.PAUSED,
    })
    preferred_states: frozenset[FSMState] = frozenset({
        FSMState.IDLE,
        FSMState.SCANNING,
    })

    def __post_init__(self) -> None:
        if (
            isinstance(self.min_interval_s, bool)
            or not isinstance(self.min_interval_s, (int, float))
            or not math.isfinite(self.min_interval_s)
        ):
            raise ValueError("min_interval_s must be a finite float")
        if float(self.min_interval_s) < 0.0:
            raise ValueError(f"min_interval_s must be >= 0.0, got {self.min_interval_s}")

        if (
            isinstance(self.max_calls_per_minute, bool)
            or not isinstance(self.max_calls_per_minute, int)
        ):
            raise ValueError("max_calls_per_minute must be an integer")  # noqa: TRY004
        if self.max_calls_per_minute < 1:
            raise ValueError(f"max_calls_per_minute must be >= 1, got {self.max_calls_per_minute}")

        if not isinstance(self.allow_while_busy, bool):
            raise ValueError("allow_while_busy must be a bool")  # noqa: TRY004

        if type(self.blocked_states) is not frozenset:
            raise ValueError("blocked_states must be a frozenset")

        if type(self.preferred_states) is not frozenset:
            raise ValueError("preferred_states must be a frozenset")

        for state in self.blocked_states:
            if not isinstance(state, FSMState):
                raise ValueError(f"blocked_states member {state!r} is not an FSMState")  # noqa: TRY004

        for state in self.preferred_states:
            if not isinstance(state, FSMState):
                raise ValueError(f"preferred_states member {state!r} is not an FSMState")  # noqa: TRY004

        if bool(self.blocked_states & self.preferred_states):
            raise ValueError(
                "blocked_states and preferred_states MUST be disjoint, got overlap: "
                f"{sorted(s.value for s in (self.blocked_states & self.preferred_states))}"
            )


@dataclass(frozen=True)
class CooldownCheck:
    """Outcome and diagnostics of a single CooldownGate check."""

    decision: CooldownDecision
    reason: str
    seconds_until_next_allowed: float
    recent_calls: int

    def __post_init__(self) -> None:
        if not isinstance(self.decision, CooldownDecision):
            raise ValueError(f"decision must be a CooldownDecision, got {self.decision!r}")  # noqa: TRY004

        if not isinstance(self.reason, str) or len(self.reason) == 0:
            raise ValueError("reason must be a non-empty string")

        if (
            isinstance(self.recent_calls, bool)
            or not isinstance(self.recent_calls, int)
            or self.recent_calls < 0
        ):
            raise ValueError(f"recent_calls must be an integer >= 0, got {self.recent_calls!r}")

        if (
            isinstance(self.seconds_until_next_allowed, bool)
            or not isinstance(self.seconds_until_next_allowed, (int, float))
            or math.isnan(self.seconds_until_next_allowed)
        ):
            raise ValueError("seconds_until_next_allowed must be a float")

        wait = float(self.seconds_until_next_allowed)

        if self.decision in (CooldownDecision.ALLOWED, CooldownDecision.FORCED_ALLOWED) and wait != 0.0:
            raise ValueError(
                f"decision == {self.decision.value} implies seconds_until_next_allowed == 0.0, got {wait}"
            )

        if self.decision in (CooldownDecision.BLOCKED_STATE, CooldownDecision.BLOCKED_MANUAL) and wait != float("inf"):
            raise ValueError(
                f"decision == {self.decision.value} implies seconds_until_next_allowed == inf, got {wait}"
            )

        if self.decision in (CooldownDecision.BLOCKED_COOLDOWN, CooldownDecision.BLOCKED_RATE_LIMIT) and (
            not math.isfinite(wait) or wait < 0.0
        ):
            raise ValueError(
                f"decision == {self.decision.value} implies finite seconds_until_next_allowed >= 0.0, got {wait}"
            )


class CooldownGate:
    """Pure, deterministic gate controlling authorization for LLM strategist calls.

    Note: The gate is NOT thread-safe. It is designed to be invoked from a single scheduler thread.
    """

    def __init__(
        self,
        *,
        config: CooldownConfig | None = None,
        session: Session | None = None,
    ) -> None:
        self._config: CooldownConfig = config if config is not None else CooldownConfig()
        self._session: Session | None = session
        self._deque: deque[float] = deque(maxlen=self._config.max_calls_per_minute + 1)
        self._manual_block_until: float | None = None
        self._force_open_once: bool = False
        self._last_decision: CooldownCheck | None = None

    @property
    def recent_calls(self) -> int:
        """Return the number of allowed calls currently within the sliding 60 s window."""
        return len(self._deque)

    @property
    def last_decision(self) -> CooldownCheck | None:
        """Return the most recent CooldownCheck decision, or None if check() has not been called."""
        return self._last_decision

    @property
    def is_manual_blocked(self) -> bool:
        """Return True if a manual operator block is currently active."""
        return self._manual_block_until is not None

    def force_open_once(self) -> None:
        """Force the gate open on the next check() call regardless of state or rate limits.

        Idempotent across repeated calls before check(). Does not emit session events immediately;
        the event is emitted on the consuming check() call.
        """
        self._force_open_once = True

    def force_closed_until(self, now: float) -> None:
        """Indefinitely block the gate until clear_manual_block() is called.

        Emits 'cooldown_manual_block' event if a Session is attached.
        """
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or float(now) < 0.0
        ):
            raise CooldownError(f"now must be a finite float >= 0.0, got {now!r}")

        now_val = float(now)
        self._manual_block_until = float("inf")
        if self._session is not None:
            self._session.write_event({
                "event": "cooldown_manual_block",
                "ts": now_val,
            })

    def clear_manual_block(self) -> None:
        """Clear active manual block and emit 'cooldown_manual_clear' event if a Session is attached."""
        self._manual_block_until = None
        if self._session is not None:
            self._session.write_event({
                "event": "cooldown_manual_clear",
            })

    def reset(self) -> None:
        """Clear invocation history, force-open flag, and manual block without emitting events."""
        self._deque.clear()
        self._force_open_once = False
        self._manual_block_until = None

    def check(
        self,
        state: FSMState,
        *,
        now: float,
    ) -> CooldownCheck:
        """Evaluate gate authorization for the given state and current time timestamp.

        Semantics evaluated in strict order:
          1. Validate `now` is finite and >= 0.0.
          2. Purge entries from recent history deque with ts < now - 60.0.
          3. If force-open flag is set, consume flag, append now, emit event, return FORCED_ALLOWED.
          4. If manual block is active (now < manual_block_until), return BLOCKED_MANUAL.
          5. If state in config.blocked_states, return BLOCKED_STATE.
          6. If state not in config.preferred_states and not config.allow_while_busy, return BLOCKED_STATE.
          7. If min_interval_s has not elapsed since last call, return BLOCKED_COOLDOWN.
          8. If rate limit max_calls_per_minute reached, return BLOCKED_RATE_LIMIT.
          9. All checks passed: append now, emit event, return ALLOWED.
        """
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or float(now) < 0.0
        ):
            raise CooldownError(f"now must be a finite float >= 0.0, got {now!r}")

        if not isinstance(state, FSMState):
            raise CooldownError(f"state must be an FSMState member, got {state!r}")

        now_val = float(now)

        # 2. Purge entries older than 60.0 seconds
        cutoff = now_val - 60.0
        while self._deque and self._deque[0] < cutoff:
            self._deque.popleft()

        # 3. Force-open flag check
        if self._force_open_once:
            self._force_open_once = False
            self._deque.append(now_val)
            if self._session is not None:
                self._session.write_event({
                    "event": "cooldown_forced_open",
                    "ts": now_val,
                    "state": state.value,
                })
            result = CooldownCheck(
                decision=CooldownDecision.FORCED_ALLOWED,
                reason="manual_force_open",
                seconds_until_next_allowed=0.0,
                recent_calls=len(self._deque),
            )
            self._last_decision = result
            return result

        # 4. Manual block check
        if self._manual_block_until is not None:
            if now_val < self._manual_block_until:
                result = CooldownCheck(
                    decision=CooldownDecision.BLOCKED_MANUAL,
                    reason="manual_block_active",
                    seconds_until_next_allowed=float("inf"),
                    recent_calls=len(self._deque),
                )
                self._last_decision = result
                return result
            else:
                self._manual_block_until = None

        # 5. Blocked states check
        if state in self._config.blocked_states:
            result = CooldownCheck(
                decision=CooldownDecision.BLOCKED_STATE,
                reason=f"blocked_state:{state.value}",
                seconds_until_next_allowed=float("inf"),
                recent_calls=len(self._deque),
            )
            self._last_decision = result
            return result

        # 6. Busy states check (when state not preferred and allow_while_busy is False)
        if state not in self._config.preferred_states and not self._config.allow_while_busy:
            result = CooldownCheck(
                decision=CooldownDecision.BLOCKED_STATE,
                reason=f"busy_state:{state.value}",
                seconds_until_next_allowed=float("inf"),
                recent_calls=len(self._deque),
            )
            self._last_decision = result
            return result

        # 7. Min interval cooldown check
        if self._deque:
            last = max(self._deque)
            elapsed = now_val - last
            if elapsed < self._config.min_interval_s:
                wait = self._config.min_interval_s - elapsed
                result = CooldownCheck(
                    decision=CooldownDecision.BLOCKED_COOLDOWN,
                    reason="min_interval",
                    seconds_until_next_allowed=wait,
                    recent_calls=len(self._deque),
                )
                self._last_decision = result
                return result

        # 8. Sliding per-minute rate limit check
        if len(self._deque) >= self._config.max_calls_per_minute:
            oldest = min(self._deque)
            wait = (oldest + 60.0) - now_val
            wait = max(wait, 0.0)
            result = CooldownCheck(
                decision=CooldownDecision.BLOCKED_RATE_LIMIT,
                reason="rate_limit",
                seconds_until_next_allowed=wait,
                recent_calls=len(self._deque),
            )
            self._last_decision = result
            return result

        # 9. All checks passed
        self._deque.append(now_val)
        if self._session is not None:
            self._session.write_event({
                "event": "cooldown_allowed",
                "ts": now_val,
                "state": state.value,
            })
        result = CooldownCheck(
            decision=CooldownDecision.ALLOWED,
            reason="ok",
            seconds_until_next_allowed=0.0,
            recent_calls=len(self._deque),
        )
        self._last_decision = result
        return result
