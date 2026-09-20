"""
Combat subsystem for WoW-bot.
"""

from wow_bot.combat.rotation import (
    CombatStateLike,
    Condition,
    RotationConfig,
    RotationError,
    RotationRule,
    RotationTable,
    load_rotation_from_dict,
)
from wow_bot.combat.targeting import (
    PriorityMetric,
    TargetConfig,
    TargetDecision,
    TargetEntityLike,
    TargetingError,
    TargetingStateLike,
    TargetSelector,
)

__all__ = [
    "CombatStateLike",
    "Condition",
    "PriorityMetric",
    "RotationConfig",
    "RotationError",
    "RotationRule",
    "RotationTable",
    "TargetConfig",
    "TargetDecision",
    "TargetEntityLike",
    "TargetSelector",
    "TargetingError",
    "TargetingStateLike",
    "load_rotation_from_dict",
]
