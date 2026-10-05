"""Perception contract layer: backend interface, adapter, and consumer views."""

from __future__ import annotations

from wow_bot.perception.adapter import (
    AdapterConfidenceConfig,
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
    fraction_to_percent,
)
from wow_bot.perception.bag import BagFrameReader, BagReading, slot_rectangles
from wow_bot.perception.cast import CastingReader, CastingReading, normalise_spell_name
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
from wow_bot.perception.durability import (
    DurabilityReader,
    DurabilityReading,
    parse_percent_text,
)
from wow_bot.perception.loot import (
    LOOT_CHANNEL_MEASURED,
    LootSparkleReader,
    LootSparkleReading,
    classify_loot_sparkle,
)
from wow_bot.perception.panels import (
    PanelObservations,
    PanelReaders,
    observe_panels,
    readers_from_config,
)
from wow_bot.perception.perception_config import (
    PERCEPTION_SCHEMA_VERSION,
    PerceptionConfig,
    PerceptionConfigError,
    load_perception_config,
    load_perception_config_from_dict,
)
from wow_bot.perception.port import (
    PerceptionBackendError,
    PerceptionPort,
    PerceptionPortError,
    PerceptionStaleError,
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
from wow_bot.perception.xp import XPBarReader, XPBarReading, parse_level_text

__all__ = [
    "EMPTY_RESOURCE_TABLE",
    "LOOT_CHANNEL_MEASURED",
    "PERCEPTION_SCHEMA_VERSION",
    "YOLO_EXTRA_NAME",
    "YOLO_OFFLINE_ENV",
    "AdapterConfidenceConfig",
    "AdapterDerivationConfig",
    "AdapterIncompleteError",
    "BagFrameReader",
    "BagReading",
    "CastingReader",
    "CastingReading",
    "CombatView",
    "DurabilityReader",
    "DurabilityReading",
    "EnemyCastView",
    "FSMState",
    "FleeView",
    "GameStateAdapter",
    "LootSparkleReader",
    "LootSparkleReading",
    "LootView",
    "PanelObservations",
    "PanelReaders",
    "PerceptionBackend",
    "PerceptionBackendError",
    "PerceptionConfig",
    "PerceptionConfigError",
    "PerceptionDependencyError",
    "PerceptionPort",
    "PerceptionPortError",
    "PerceptionStaleError",
    "ReactiveView",
    "ResourceTable",
    "RuntimeContext",
    "StaticRuntimeContext",
    "StrategistView",
    "TargetView",
    "VendorView",
    "WorldSyncEntityView",
    "WorldSyncView",
    "XPBarReader",
    "XPBarReading",
    "apply_yolo_offline_env",
    "classify_loot_sparkle",
    "fraction_to_percent",
    "has_cv2",
    "has_mss",
    "has_pytesseract",
    "has_tesseract_binary",
    "has_ultralytics",
    "load_perception_config",
    "load_perception_config_from_dict",
    "normalise_spell_name",
    "observe_panels",
    "parse_level_text",
    "parse_percent_text",
    "readers_from_config",
    "require_cv2",
    "require_mss",
    "require_pytesseract",
    "require_tesseract_binary",
    "require_ultralytics",
    "slot_rectangles",
    "verify_yolo_offline",
]
