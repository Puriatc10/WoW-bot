"""Perception contract layer: backend interface, adapter, and consumer views."""

from __future__ import annotations

from wow_bot.perception.adapter import AdapterIncompleteError, GameStateAdapter, fraction_to_percent
from wow_bot.perception.protocol import PerceptionBackend
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
    "AdapterIncompleteError",
    "CombatView",
    "EnemyCastView",
    "FSMState",
    "FleeView",
    "GameStateAdapter",
    "LootView",
    "PerceptionBackend",
    "ReactiveView",
    "StrategistView",
    "TargetView",
    "VendorView",
    "WorldSyncEntityView",
    "WorldSyncView",
    "fraction_to_percent",
]