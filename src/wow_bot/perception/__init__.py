"""Perception contract layer: backend interface, adapter, and consumer views."""

from __future__ import annotations

from wow_bot.perception.adapter import (
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
    fraction_to_percent,
)
from wow_bot.perception.context import RuntimeContext, StaticRuntimeContext
from wow_bot.perception.protocol import PerceptionBackend
from wow_bot.perception.resource_table import EMPTY_RESOURCE_TABLE, ResourceTable
from wow_bot.perception.views import (
    CombatView,
    EnemyCastView,
    FleeView,
    FSMState,
    LootView,
    ReactiveView,
    StrategistView,
    TargetView,
    VendorView,
    WorldSyncEntityView,
    WorldSyncView,
)

__all__ = [
    "EMPTY_RESOURCE_TABLE",
    "AdapterDerivationConfig",
    "AdapterIncompleteError",
    "CombatView",
    "EnemyCastView",
    "FSMState",
    "FleeView",
    "GameStateAdapter",
    "LootView",
    "PerceptionBackend",
    "ReactiveView",
    "ResourceTable",
    "RuntimeContext",
    "StaticRuntimeContext",
    "StrategistView",
    "TargetView",
    "VendorView",
    "WorldSyncEntityView",
    "WorldSyncView",
    "fraction_to_percent",
]