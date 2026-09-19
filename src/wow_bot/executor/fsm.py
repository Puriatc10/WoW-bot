"""Deterministic Finite State Machine (FSM) core for local execution control (Task 5.3).

Translates (GameState, Strategy) -> deterministic local state transitions
and simulated execution intent without physical keyboard/mouse actions, real time
reading, or process interaction.

Public API:
    - :class:`State`
    - :class:`ExecutorFSM`
    - :data:`STUCK_THRESHOLD_SECONDS`
    - :data:`FLEE_HP_MAX_THRESHOLD`
    - :data:`FLEE_HP_RISK_RANGE`
"""

from __future__ import annotations

import math
from collections.abc import Callable
from enum import Enum, auto
from typing import Any, Final

from wow_bot.executor.controller import Controller
from wow_bot.shared.config import ExecutorConfig, Settings
from wow_bot.shared.interfaces import GameState, Strategy
from wow_bot.shared.logger import get_logger

log = get_logger("EXEC_FSM")

#: Repeated movement duration required to trigger stuck recovery (in simulated seconds).
STUCK_THRESHOLD_SECONDS: Final[float] = 5.0

#: Base FLEE threshold at zero risk tolerance (flee at <= 60% HP).
FLEE_HP_MAX_THRESHOLD: Final[float] = 0.60

#: FLEE threshold risk multiplier range (flee threshold = 0.60 - 0.40 * risk_tolerance).
FLEE_HP_RISK_RANGE: Final[float] = 0.40


class State(Enum):
    """Supported FSM states strictly frozen for Task 5.3."""

    IDLE = auto()
    SCANNING = auto()
    MOVING_TO_TARGET = auto()
    COMBAT = auto()
    LOOTING = auto()
    FLEEING = auto()
    STUCK_RECOVERY = auto()


class ExecutorFSM:
    """Deterministic state machine translating GameState + Strategy into local execution intent."""

    def __init__(
        self,
        config: ExecutorConfig | Settings | dict[str, Any] | None,
        controller: Controller,
        strategy: Strategy,
        *,
        on_transition: Callable[[float, State, State, str], None] | None = None,
    ) -> None:
        """Initialize ExecutorFSM in IDLE state without executing controller commands or reading clock."""
        if not isinstance(controller, Controller):
            raise ValueError(  # noqa: TRY004
                f"controller must be a Controller instance, got {type(controller).__name__}"
            )
        if not isinstance(strategy, Strategy):
            raise ValueError(  # noqa: TRY004
                f"strategy must be a Strategy instance, got {type(strategy).__name__}"
            )

        self._config = config
        self._controller = controller
        self._strategy = strategy
        self._state: State = State.IDLE
        self._on_transition = on_transition
        self._last_timestamp: float | None = None
        self._last_action_signature: str | None = None
        self._action_started_at: float | None = None

    @property
    def state(self) -> State:
        """Return current FSM state."""
        return self._state

    def set_strategy(self, strategy: Strategy) -> None:
        """Update current strategy immediately for future ticks while preserving local FSM state."""
        if not isinstance(strategy, Strategy):
            raise ValueError(  # noqa: TRY004
                f"strategy must be a Strategy instance, got {type(strategy).__name__}"
            )
        self._strategy = strategy

    def _flee_hp_threshold(self) -> float:
        """Calculate deterministic flee threshold based on risk tolerance."""
        raw = FLEE_HP_MAX_THRESHOLD - FLEE_HP_RISK_RANGE * self._strategy.risk_tolerance
        return round(raw, 6)

    @staticmethod
    def _has_target(game_state: GameState) -> bool:
        """Check if game state contains a valid target."""
        return game_state.target is not None

    @staticmethod
    def _movement_signature(game_state: GameState) -> str:
        """Generate a stable private action signature for movement intent."""
        if game_state.target is not None and game_state.target.name:
            return f"move_to_target:{game_state.target.name}"
        return "move_to_target"

    def _clear_stuck_tracker(self) -> None:
        """Reset internal stuck action tracking state."""
        self._last_action_signature = None
        self._action_started_at = None

    async def _transition_to(
        self,
        new_state: State,
        *,
        timestamp: float,
        reason: str,
    ) -> None:
        """Perform a state transition, log changes, and execute safety triggers."""
        if new_state == self._state:
            return

        old_state = self._state
        self._state = new_state
        log.info(
            f"FSM transition: {old_state.name} -> {new_state.name} | "
            f"timestamp={timestamp} | reason='{reason}' | "
            f"risk_tolerance={self._strategy.risk_tolerance}"
        )

        if new_state in (State.FLEEING, State.STUCK_RECOVERY):
            await self._controller.stop_all()

        if self._on_transition is not None:
            try:
                self._on_transition(timestamp, old_state, new_state, reason)
            except Exception as exc:  # noqa: BLE001
                log.error(f"Error in FSM transition callback: {exc}")

    def _validate_timestamp(self, raw_ts: object) -> float:
        """Validate tick timestamp parameter and return it as a finite float."""
        if isinstance(raw_ts, bool) or not isinstance(raw_ts, (int, float)):
            raise ValueError(  # noqa: TRY004
                f"Timestamp must be a finite numeric value, got {type(raw_ts).__name__} ({raw_ts!r})"
            )
        float_ts = float(raw_ts)
        if not math.isfinite(float_ts):
            raise ValueError(f"Timestamp must be finite, got {raw_ts!r}")
        if self._last_timestamp is not None and float_ts < self._last_timestamp:
            raise ValueError(
                f"Decreasing timestamp rejected: new timestamp {float_ts} < previous timestamp {self._last_timestamp}"
            )
        return float_ts

    async def tick(self, game_state: GameState) -> None:
        """Advance the state machine by processing one GameState snapshot."""
        if not isinstance(game_state, GameState):
            raise ValueError(  # noqa: TRY004
                f"game_state must be a GameState instance, got {type(game_state).__name__}"
            )

        ts = self._validate_timestamp(game_state.timestamp)

        flee_threshold = self._flee_hp_threshold()
        low_hp_in_combat = game_state.in_combat and (game_state.hp_pct <= flee_threshold)

        # 1. Handle FLEEING state
        if self._state == State.FLEEING:
            if not game_state.in_combat and game_state.hp_pct > flee_threshold:
                self._clear_stuck_tracker()
                await self._transition_to(
                    State.SCANNING,
                    timestamp=ts,
                    reason="fleeing_safe_exit",
                )
            self._last_timestamp = ts
            return

        # 2. Handle low-HP combat safety
        if low_hp_in_combat:
            self._clear_stuck_tracker()
            await self._transition_to(
                State.FLEEING,
                timestamp=ts,
                reason=f"low_hp_combat (hp_pct={game_state.hp_pct:.2f} <= threshold={flee_threshold:.2f})",
            )
            self._last_timestamp = ts
            return

        # 3. Handle STUCK_RECOVERY state
        if self._state == State.STUCK_RECOVERY:
            self._clear_stuck_tracker()
            await self._transition_to(
                State.SCANNING,
                timestamp=ts,
                reason="stuck_recovery_complete",
            )
            self._last_timestamp = ts
            return

        # 4. Handle LOOTING state
        if self._state == State.LOOTING:
            self._clear_stuck_tracker()
            if game_state.in_combat:
                await self._transition_to(
                    State.COMBAT,
                    timestamp=ts,
                    reason="combat_started_during_looting",
                )
            else:
                await self._transition_to(
                    State.SCANNING,
                    timestamp=ts,
                    reason="looting_complete",
                )
            self._last_timestamp = ts
            return

        # 5. Determine normal candidate state
        if game_state.in_combat:
            target_state = State.COMBAT
            reason = "combat_active"
        elif self._state == State.COMBAT:
            target_state = State.LOOTING
            reason = "combat_ended"
        elif self._has_target(game_state):
            target_state = State.MOVING_TO_TARGET
            reason = "target_present"
        else:
            target_state = State.SCANNING
            reason = "no_target_no_combat"

        # 6. Stuck detection for movement-like intent
        if target_state == State.MOVING_TO_TARGET:
            sig = self._movement_signature(game_state)
            if self._last_action_signature != sig:
                self._last_action_signature = sig
                self._action_started_at = ts
            else:
                assert self._action_started_at is not None
                elapsed = ts - self._action_started_at
                if elapsed >= STUCK_THRESHOLD_SECONDS:
                    target_state = State.STUCK_RECOVERY
                    reason = f"stuck_detected_after_{elapsed:.1f}s"
                    self._clear_stuck_tracker()
        else:
            self._clear_stuck_tracker()

        # 7. Perform state transition if needed
        await self._transition_to(target_state, timestamp=ts, reason=reason)
        self._last_timestamp = ts
