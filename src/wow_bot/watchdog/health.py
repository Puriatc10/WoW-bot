"""Health states state machine for progress metric evaluation (Task 8.2).

Pure, hysteresis-aware state machine mapping ProgressSnapshot values
to HEALTHY / DEGRADED / CRITICAL health states and emitting transition events.
Deterministic and isolated from OS, I/O, LLMs, and wall-clock time.
"""

import math
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any

from wow_bot.watchdog.metrics import ProgressSnapshot

if TYPE_CHECKING:
    from wow_bot.session import Session


class HealthError(Exception):
    """Raised when health state operations or validations violate health constraints."""


class HealthState(str, Enum):
    """Health classification levels."""

    HEALTHY = "healthy"
    DEGRADED = "degraded"
    CRITICAL = "critical"


class MetricName(str, Enum):
    """Monitored progress metrics."""

    POSITION_DELTA = "position_delta"
    INVENTORY_DELTA = "inventory_delta"
    LEVEL_OR_XP_DELTA = "level_or_xp_delta"
    SUCCESSFUL_ACTIONS_PER_MINUTE = "successful_actions_per_minute"
    REFLEX_TICK_RATE_HZ = "reflex_tick_rate_hz"


@dataclass(frozen=True)
class MetricThresholds:
    """Configured degraded and critical threshold bounds with recovery margin.

    Invariants:
      * critical_below <= degraded_below
      * recovery_margin >= 0.0
      * All floats must be finite
    """

    degraded_below: float
    critical_below: float
    recovery_margin: float = 0.1

    def __post_init__(self) -> None:
        if (
            isinstance(self.degraded_below, bool)
            or not isinstance(self.degraded_below, (int, float))
            or not math.isfinite(self.degraded_below)
        ):
            raise ValueError("degraded_below must be a finite float")
        if (
            isinstance(self.critical_below, bool)
            or not isinstance(self.critical_below, (int, float))
            or not math.isfinite(self.critical_below)
        ):
            raise ValueError("critical_below must be a finite float")
        if (
            isinstance(self.recovery_margin, bool)
            or not isinstance(self.recovery_margin, (int, float))
            or not math.isfinite(self.recovery_margin)
        ):
            raise ValueError("recovery_margin must be a finite float")

        if self.critical_below > self.degraded_below:
            raise ValueError(
                f"critical_below ({self.critical_below}) must be <= degraded_below ({self.degraded_below})"
            )
        if self.recovery_margin < 0.0:
            raise ValueError(
                f"recovery_margin must be >= 0.0, got {self.recovery_margin}"
            )


@dataclass(frozen=True)
class HealthConfig:
    """Configuration mapping metrics to threshold specifications and hold windows.

    Invariants:
      * thresholds MUST cover every MetricName member
      * hold_s >= 0.0
      * min_observations_for_transition >= 1
      * rate_metrics MUST be a subset of MetricName members
    """

    thresholds: Mapping[MetricName, MetricThresholds]
    hold_s: float = 5.0
    min_observations_for_transition: int = 3
    rate_metrics: frozenset[MetricName] = frozenset({
        MetricName.SUCCESSFUL_ACTIONS_PER_MINUTE,
        MetricName.REFLEX_TICK_RATE_HZ,
    })

    def __post_init__(self) -> None:
        if not isinstance(self.thresholds, Mapping):
            raise TypeError("thresholds must be a Mapping")
        if set(self.thresholds.keys()) != set(MetricName):
            raise ValueError(
                f"thresholds MUST cover every MetricName member, got keys: {set(self.thresholds.keys())}"
            )
        for k, v in self.thresholds.items():
            if not isinstance(v, MetricThresholds):
                raise TypeError(
                    f"threshold value for {k} must be MetricThresholds instance, got {type(v)}"
                )

        if (
            isinstance(self.hold_s, bool)
            or not isinstance(self.hold_s, (int, float))
            or not math.isfinite(self.hold_s)
            or self.hold_s < 0.0
        ):
            raise ValueError(f"hold_s must be a finite float >= 0.0, got {self.hold_s}")

        if (
            isinstance(self.min_observations_for_transition, bool)
            or not isinstance(self.min_observations_for_transition, int)
            or self.min_observations_for_transition < 1
        ):
            raise ValueError(
                f"min_observations_for_transition must be an integer >= 1, got {self.min_observations_for_transition}"
            )

        if not isinstance(self.rate_metrics, (set, frozenset)) or not self.rate_metrics.issubset(
            set(MetricName)
        ):
            raise ValueError(
                f"rate_metrics MUST be a subset of MetricName members, got {self.rate_metrics}"
            )

    @classmethod
    def default(cls) -> "HealthConfig":
        """Return a config with default starting thresholds for a 60s window.

        Starting points to be tuned in LAB_MODE.
        """
        margin = 0.2
        return cls(
            thresholds={
                MetricName.POSITION_DELTA: MetricThresholds(
                    degraded_below=0.5, critical_below=0.1, recovery_margin=margin
                ),
                MetricName.INVENTORY_DELTA: MetricThresholds(
                    degraded_below=0.0, critical_below=0.0, recovery_margin=margin
                ),
                MetricName.LEVEL_OR_XP_DELTA: MetricThresholds(
                    degraded_below=0.0, critical_below=0.0, recovery_margin=margin
                ),
                MetricName.SUCCESSFUL_ACTIONS_PER_MINUTE: MetricThresholds(
                    degraded_below=2.0, critical_below=0.5, recovery_margin=margin
                ),
                MetricName.REFLEX_TICK_RATE_HZ: MetricThresholds(
                    degraded_below=2.0, critical_below=0.5, recovery_margin=margin
                ),
            },
            hold_s=5.0,
            min_observations_for_transition=3,
            rate_metrics=frozenset({
                MetricName.SUCCESSFUL_ACTIONS_PER_MINUTE,
                MetricName.REFLEX_TICK_RATE_HZ,
            }),
        )


@dataclass(frozen=True)
class HealthTransition:
    """Immutable transition event record emitted on state change.

    Invariants:
      * from_state != to_state
      * trigger_metrics is a non-empty tuple
      * reason is a non-empty string
      * ts is finite and >= 0.0
    """

    from_state: HealthState
    to_state: HealthState
    ts: float
    trigger_metrics: tuple[MetricName, ...]
    reason: str

    def __post_init__(self) -> None:
        if not isinstance(self.from_state, HealthState):
            raise TypeError(f"from_state must be a HealthState, got {self.from_state}")
        if not isinstance(self.to_state, HealthState):
            raise TypeError(f"to_state must be a HealthState, got {self.to_state}")
        if self.from_state == self.to_state:
            raise ValueError("from_state and to_state cannot be equal")

        if not (
            isinstance(self.trigger_metrics, tuple)
            and len(self.trigger_metrics) > 0
            and all(isinstance(m, MetricName) for m in self.trigger_metrics)
        ):
            raise ValueError("trigger_metrics must be a non-empty tuple of MetricName")

        if not isinstance(self.reason, str) or not self.reason.strip():
            raise ValueError("reason must be a non-empty string")

        if (
            isinstance(self.ts, bool)
            or not isinstance(self.ts, (int, float))
            or not math.isfinite(self.ts)
            or self.ts < 0.0
        ):
            raise ValueError(f"ts must be a finite float >= 0.0, got {self.ts}")

    def to_dict(self) -> dict[str, Any]:
        """Return event dictionary representation suitable for session logging."""
        return {
            "event": "watchdog_transition",
            "from_state": self.from_state.value,
            "to_state": self.to_state.value,
            "ts": float(self.ts),
            "trigger_metrics": [m.value for m in self.trigger_metrics],
            "reason": self.reason,
        }


class HealthStateMachine:
    """Hysteresis-aware health state machine mapping ProgressSnapshots to HealthStates."""

    def __init__(
        self,
        *,
        config: HealthConfig | None = None,
        session: "Session | None" = None,
    ) -> None:
        self._config: HealthConfig = config if config is not None else HealthConfig.default()
        self._session: Any | None = session
        self._current_state: HealthState = HealthState.HEALTHY
        self._per_metric_states: dict[MetricName, HealthState] = {
            m: HealthState.HEALTHY for m in MetricName
        }
        self._pending_candidate: HealthState | None = None
        self._pending_first_ts: float | None = None
        self._pending_count: int = 0

    @property
    def current_state(self) -> HealthState:
        return self._current_state

    @property
    def pending_state(self) -> HealthState | None:
        return self._pending_candidate

    @property
    def pending_count(self) -> int:
        return self._pending_count

    def reset(self) -> None:
        """Reset state machine to initial HEALTHY state with no pending candidate. Idempotent."""
        self._current_state = HealthState.HEALTHY
        self._per_metric_states = {m: HealthState.HEALTHY for m in MetricName}
        self._pending_candidate = None
        self._pending_first_ts = None
        self._pending_count = 0

    def force_state(
        self,
        state: HealthState,
        *,
        now: float,
        reason: str,
    ) -> HealthTransition:
        """Immediately transition to specified state regardless of thresholds or hysteresis."""
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or now < 0.0
        ):
            raise HealthError(f"now must be a finite float >= 0.0, got {now}")

        if not isinstance(reason, str) or not reason.strip():
            raise HealthError("reason must be a non-empty string")

        if not isinstance(state, HealthState):
            raise HealthError(f"state must be a HealthState, got {state}")

        if state == self._current_state:
            raise HealthError(f"Cannot force state to current state ({state.value})")

        non_healthy = [
            m for m, s in self._per_metric_states.items() if s != HealthState.HEALTHY
        ]
        if state in (HealthState.DEGRADED, HealthState.CRITICAL):
            for m in MetricName:
                self._per_metric_states[m] = state
            trigger_metrics = tuple(sorted(MetricName, key=lambda m: m.value))
        else:
            if non_healthy:
                trigger_metrics = tuple(sorted(non_healthy, key=lambda m: m.value))
            else:
                trigger_metrics = tuple(sorted(MetricName, key=lambda m: m.value))
            self._per_metric_states = {m: HealthState.HEALTHY for m in MetricName}

        transition = HealthTransition(
            from_state=self._current_state,
            to_state=state,
            ts=float(now),
            trigger_metrics=trigger_metrics,
            reason=reason,
        )

        self._current_state = state
        self._pending_candidate = None
        self._pending_first_ts = None
        self._pending_count = 0

        if self._session is not None:
            self._session.write_event(transition.to_dict())

        return transition

    def _get_metric_value(self, snapshot: ProgressSnapshot, metric: MetricName) -> float:
        if metric == MetricName.POSITION_DELTA:
            return float(snapshot.position_delta)
        elif metric == MetricName.INVENTORY_DELTA:
            return float(snapshot.inventory_delta)
        elif metric == MetricName.LEVEL_OR_XP_DELTA:
            return float(snapshot.level_or_xp_delta)
        elif metric == MetricName.SUCCESSFUL_ACTIONS_PER_MINUTE:
            return float(snapshot.successful_actions_per_minute)
        elif metric == MetricName.REFLEX_TICK_RATE_HZ:
            return float(snapshot.reflex_tick_rate_hz)
        else:
            raise ValueError(f"Unknown metric name: {metric}")

    def observe(
        self,
        snapshot: ProgressSnapshot,
        *,
        now: float,
    ) -> HealthTransition | None:
        """Observe progress snapshot and evaluate state machine transitions."""
        if (
            isinstance(now, bool)
            or not isinstance(now, (int, float))
            or not math.isfinite(now)
            or now < 0.0
        ):
            raise HealthError(f"now must be a finite float >= 0.0, got {now}")

        prev_non_healthy = [
            m for m, st in self._per_metric_states.items() if st != HealthState.HEALTHY
        ]

        evaluated_values: dict[MetricName, float] = {}
        for metric in MetricName:
            if metric in self._config.rate_metrics and not snapshot.is_rate_reliable:
                continue

            val = self._get_metric_value(snapshot, metric)
            evaluated_values[metric] = val

            thresholds = self._config.thresholds[metric]
            crit_below = thresholds.critical_below
            deg_below = thresholds.degraded_below
            margin = thresholds.recovery_margin
            last_st = self._per_metric_states[metric]

            if last_st == HealthState.HEALTHY:
                if val <= crit_below:
                    cand_m = HealthState.CRITICAL
                elif val <= deg_below:
                    cand_m = HealthState.DEGRADED
                else:
                    cand_m = HealthState.HEALTHY
            elif last_st == HealthState.DEGRADED:
                if val <= crit_below:
                    cand_m = HealthState.CRITICAL
                elif val >= deg_below + margin:
                    cand_m = HealthState.HEALTHY
                else:
                    cand_m = HealthState.DEGRADED
            elif last_st == HealthState.CRITICAL:
                if val >= crit_below + margin:
                    if val >= deg_below + margin:
                        cand_m = HealthState.HEALTHY
                    else:
                        cand_m = HealthState.DEGRADED
                else:
                    cand_m = HealthState.CRITICAL
            else:
                cand_m = HealthState.HEALTHY

            self._per_metric_states[metric] = cand_m

        worst_rank = -1
        overall_candidate = HealthState.HEALTHY
        rank_map = {
            HealthState.HEALTHY: 0,
            HealthState.DEGRADED: 1,
            HealthState.CRITICAL: 2,
        }

        for m in MetricName:
            st = self._per_metric_states[m]
            if rank_map[st] > worst_rank:
                worst_rank = rank_map[st]
                overall_candidate = st

        non_healthy_metrics = [
            m for m, st in self._per_metric_states.items() if st != HealthState.HEALTHY
        ]

        if overall_candidate == self._current_state:
            self._pending_candidate = None
            self._pending_first_ts = None
            self._pending_count = 0
            return None

        if self._pending_candidate != overall_candidate:
            self._pending_candidate = overall_candidate
            self._pending_first_ts = float(now)
            self._pending_count = 1
        else:
            self._pending_count += 1

        assert self._pending_first_ts is not None
        time_held = float(now) - self._pending_first_ts
        if (
            time_held < self._config.hold_s
            or self._pending_count < self._config.min_observations_for_transition
        ):
            return None

        if non_healthy_metrics:
            trigger_metrics = tuple(sorted(non_healthy_metrics, key=lambda m: m.value))
        elif prev_non_healthy:
            trigger_metrics = tuple(sorted(prev_non_healthy, key=lambda m: m.value))
        else:
            trigger_metrics = tuple(sorted(MetricName, key=lambda m: m.value))

        worst_metrics = [
            m for m in trigger_metrics if self._per_metric_states[m] == overall_candidate
        ]
        if worst_metrics:
            worst_m = worst_metrics[0]
        elif trigger_metrics:
            worst_m = trigger_metrics[0]
        else:
            worst_m = MetricName.POSITION_DELTA

        worst_val = evaluated_values.get(
            worst_m, self._get_metric_value(snapshot, worst_m)
        )
        reason = f"{worst_m.value}={worst_val}"

        transition = HealthTransition(
            from_state=self._current_state,
            to_state=overall_candidate,
            ts=float(now),
            trigger_metrics=trigger_metrics,
            reason=reason,
        )

        self._current_state = overall_candidate
        self._pending_candidate = None
        self._pending_first_ts = None
        self._pending_count = 0

        if self._session is not None:
            self._session.write_event(transition.to_dict())

        return transition
