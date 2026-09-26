"""Perception contract layer: backend interface, adapter, and consumer views."""

from __future__ import annotations

from wow_bot.perception.adapter import (
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
    fraction_to_percent,
)
from wow_bot.perception.context import RuntimeContext, StaticRuntimeContext
from wow_bot.perception.deps import (
    YOLO_EXTRA_NAME,
    YOLO_OFFLINE_ENV,
    PerceptionDependencyError,
    apply_yolo_offline_env,
    has_cv2,
    has_mss,
    has_pytesseract,
    has_tesseract_binary,
    has_ultralytics,
    require_cv2,
    require_mss,
    require_pytesseract,
    require_tesseract_binary,
    require_ultralytics,
    verify_yolo_offline,
)
from wow_bot.perception.perception_config import (
    PERCEPTION_SCHEMA_VERSION,
    PerceptionConfig,
    PerceptionConfigError,
    load_perception_config,
    load_perception_config_from_dict,
)
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
    "PERCEPTION_SCHEMA_VERSION",
    "YOLO_EXTRA_NAME",
    "YOLO_OFFLINE_ENV",
    "AdapterDerivationConfig",
    "AdapterIncompleteError",
    "CombatView",
    "EnemyCastView",
    "FSMState",
    "FleeView",
    "GameStateAdapter",
    "LootView",
    "PerceptionBackend",
    "PerceptionConfig",
    "PerceptionConfigError",
    "PerceptionDependencyError",
    "ReactiveView",
    "ResourceTable",
    "RuntimeContext",
    "StaticRuntimeContext",
    "StrategistView",
    "TargetView",
    "VendorView",
    "WorldSyncEntityView",
    "WorldSyncView",
    "apply_yolo_offline_env",
    "fraction_to_percent",
    "has_cv2",
    "has_mss",
    "has_pytesseract",
    "has_tesseract_binary",
    "has_ultralytics",
    "load_perception_config",
    "load_perception_config_from_dict",
    "require_cv2",
    "require_mss",
    "require_pytesseract",
    "require_tesseract_binary",
    "require_ultralytics",
    "verify_yolo_offline",
]
