"""
Combat subsystem for WoW-bot.
"""

from wow_bot.combat.loop import (
    CombatLoop,
    CombatLoopConfig,
    CombatLoopError,
    CombatMetaView,
    CombatStateView,
)
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
    "CombatLoop",
    "CombatLoopConfig",
    "CombatLoopError",
    "CombatMetaView",
    "CombatStateLike",
    "CombatStateView",
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
