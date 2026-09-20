"""
Pure, deterministic combat target selector for WoW-bot.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Protocol


class TargetingError(Exception):
    """Raised when targeting configuration or evaluation fails."""


class TargetEntityLike(Protocol):
    """Protocol for target entities evaluated by TargetSelector."""

    entity_id: str
    distance: float  # >= 0.0
    threat: float  # >= 0.0, higher = more threat
    hp_percent: float  # 0.0 .. 100.0
    is_attackable: bool
    is_alive: bool
    is_in_combat_with_self: bool


class TargetingStateLike(Protocol):
    """Protocol for targeting state supplied to TargetSelector."""

    entities: tuple[TargetEntityLike, ...]
    current_target_id: str | None


class PriorityMetric(str, Enum):
    """Metrics for ranking target candidate priority."""

    THREAT = "threat"
    DISTANCE = "distance"
    HP = "hp"


@dataclass(frozen=True)
class TargetConfig:
    """Configuration options for target selection."""

    max_distance: float = 40.0
    min_hp_percent: float = 0.1
    require_in_combat: bool = True
    require_attackable: bool = True
    priority: tuple[PriorityMetric, ...] = (
        PriorityMetric.THREAT,
        PriorityMetric.DISTANCE,
    )
    switch_hysteresis: float = 0.0

    def __post_init__(self) -> None:
        if (
            isinstance(self.max_distance, bool)
            or not isinstance(self.max_distance, (int, float))
            or float(self.max_distance) <= 0.0
        ):
            raise ValueError("max_distance must be > 0.0")

        if (
            isinstance(self.min_hp_percent, bool)
            or not isinstance(self.min_hp_percent, (int, float))
            or not (0.0 <= float(self.min_hp_percent) <= 100.0)
        ):
            raise ValueError("min_hp_percent must be in range [0.0, 100.0]")

        if not isinstance(self.priority, tuple) or len(self.priority) < 1:
            raise ValueError("priority must be a non-empty tuple")

        if not all(isinstance(p, PriorityMetric) for p in self.priority):
            raise ValueError("All elements in priority must be PriorityMetric instances")

        if len(set(self.priority)) != len(self.priority):
            raise ValueError("priority cannot contain duplicate metrics")

        if (
            isinstance(self.switch_hysteresis, bool)
            or not isinstance(self.switch_hysteresis, (int, float))
            or float(self.switch_hysteresis) < 0.0
        ):
            raise ValueError("switch_hysteresis must be >= 0.0")


@dataclass(frozen=True)
class TargetDecision:
    """Outcome of target selection evaluation."""

    entity_id: str | None
    reason: str
    metric_values: tuple[tuple[str, float], ...] = ()


def _get_metric_value(entity: TargetEntityLike, metric: PriorityMetric) -> float:
    if metric == PriorityMetric.THREAT:
        return float(entity.threat)
    elif metric == PriorityMetric.DISTANCE:
        return float(entity.distance)
    elif metric == PriorityMetric.HP:
        return float(entity.hp_percent)
    else:
        raise ValueError(f"Unknown priority metric: {metric}")


def _extract_metric_values(
    entity: TargetEntityLike, priority: tuple[PriorityMetric, ...]
) -> tuple[tuple[str, float], ...]:
    return tuple((m.value, _get_metric_value(entity, m)) for m in priority)


class TargetSelector:
    """Pure, deterministic target selection component."""

    def __init__(self, config: TargetConfig | None = None) -> None:
        if config is None:
            config = TargetConfig()
        elif not isinstance(config, TargetConfig):
            raise TypeError("config must be a TargetConfig instance or None")
        self._config = config

    @property
    def config(self) -> TargetConfig:
        return self._config

    def select(self, state: TargetingStateLike) -> TargetDecision:
        cfg = self._config

        candidates: list[TargetEntityLike] = []
        for entity in state.entities:
            if not entity.is_alive:
                continue
            if cfg.require_attackable and not entity.is_attackable:
                continue
            if cfg.require_in_combat and not entity.is_in_combat_with_self:
                continue
            if entity.distance > cfg.max_distance:
                continue
            if entity.hp_percent < cfg.min_hp_percent:
                continue
            candidates.append(entity)

        if not candidates:
            return TargetDecision(entity_id=None, reason="no_candidates")

        def sort_key(entity: TargetEntityLike) -> tuple[float | str, ...]:
            key_parts: list[float | str] = []
            for metric in cfg.priority:
                val = _get_metric_value(entity, metric)
                if metric == PriorityMetric.THREAT:
                    key_parts.append(-val)
                else:
                    key_parts.append(val)
            key_parts.append(str(entity.entity_id))
            return tuple(key_parts)

        candidates.sort(key=sort_key)
        best = candidates[0]
        chosen = best

        if cfg.switch_hysteresis > 0.0 and state.current_target_id is not None:
            current = next(
                (c for c in candidates if c.entity_id == state.current_target_id),
                None,
            )
            if current is not None:
                top_metric = cfg.priority[0]
                best_val = _get_metric_value(best, top_metric)
                current_val = _get_metric_value(current, top_metric)
                if abs(best_val - current_val) <= cfg.switch_hysteresis:
                    chosen = current

        return TargetDecision(
            entity_id=chosen.entity_id,
            reason="selected",
            metric_values=_extract_metric_values(chosen, cfg.priority),
        )

    def should_switch(self, state: TargetingStateLike) -> bool:
        decision = self.select(state)
        return decision.entity_id != state.current_target_id
