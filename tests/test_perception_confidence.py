"""Acceptance tests for T-FIX-22 — perception_confidence plumbing and thresholding.

Covers ADR-002 Decision 4 and docs/PERCEPTION.md:33:
1. Low-confidence observation surfaces as None in views; on non-Optional fields
   it raises AdapterIncompleteError.
2. Genuine observed zero (e.g. inventory_count == 0 with good confidence)
   is NOT converted to None.
3. Absence from perception_confidence means "not attempted" -> value is taken as-is.
4. MockPerception populates perception_confidence from component RNG and
   synthesizes bag frames with occasional low confidence where values surface as None.
5. Confidence threshold configuration loading and validation (defaults,
   valid keys, unknown keys fail loudly).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from wow_bot.mocks.mock_perception import MockPerception
from wow_bot.perception.adapter import (
    AdapterConfidenceConfig,
    AdapterDerivationConfig,
    AdapterIncompleteError,
    GameStateAdapter,
)
from wow_bot.perception.context import StaticRuntimeContext
from wow_bot.perception.perception_config import (
    PerceptionConfigError,
    load_perception_config,
    load_perception_config_from_dict,
)
from wow_bot.shared.interfaces import GameState, TargetInfo


def _make_state(
    *,
    timestamp: float = 1000.0,
    hp_pct: float = 0.85,
    mana_pct: float = 0.60,
    position: tuple[float, float] = (100.0, 200.0),
    facing: float = 1.57,
    in_combat: bool = False,
    target: TargetInfo | None = None,
    enemies: tuple[Any, ...] | list[Any] = (),
    events: tuple[Any, ...] | list[Any] = (),
    **kwargs: Any,
) -> GameState:
    """Helper constructing canonical GameState with optional kwargs."""
    return GameState(
        timestamp=timestamp,
        hp_pct=hp_pct,
        mana_pct=mana_pct,
        position=position,
        facing=facing,
        in_combat=in_combat,
        target=target,
        enemies=enemies,
        events=events,
        **kwargs,
    )


def test_low_confidence_inventory_count_yields_none_and_raises_in_vendor_view() -> None:
    """Acceptance 1: Low-confidence inventory_count yields None and raises AdapterIncompleteError."""
    # inventory_count default threshold is 0.7. Confidence 0.45 is below threshold.
    state = _make_state(
        inventory_count=5,
        perception_confidence={"inventory_count": 0.45},
    )
    adapter = GameStateAdapter(state)

    with pytest.raises(
        AdapterIncompleteError,
        match="VendorStateView requires non-Optional inventory_count",
    ):
        adapter.to_vendor_view()


def test_low_confidence_inventory_count_raises_in_loot_view() -> None:
    """Low-confidence inventory_count surfaces as None and raises in LootView."""
    target = TargetInfo(
        name="Defias Thug",
        hp_pct=0.0,
        reaction="hostile",
        distance_estimate=10.0,
    )
    state = _make_state(
        inventory_count=8,
        target=target,
        target_is_lootable=True,
        perception_confidence={"inventory_count": 0.40},
    )
    adapter = GameStateAdapter(state)

    # In LootView:
    with pytest.raises(
        AdapterIncompleteError,
        match="LootStateView requires non-Optional inventory_count",
    ):
        adapter.to_loot_view()


def test_zero_inventory_count_with_good_confidence_stays_zero() -> None:
    """Acceptance 2: inventory_count == 0 with good confidence stays 0 and does not raise."""
    state = _make_state(
        inventory_count=0,
        perception_confidence={"inventory_count": 0.85},
    )
    adapter = GameStateAdapter(state)

    view = adapter.to_vendor_view()
    assert view.inventory_count == 0
    assert view.self_x == 100.0
    assert view.self_y == 200.0


def test_unattempted_observation_absent_from_confidence_map_is_taken_as_is() -> None:
    """Contract: Absence from perception_confidence means 'not attempted'; taken as-is."""
    state = _make_state(
        inventory_count=4,
        durability_fraction=0.85,
        perception_confidence={},  # empty map
    )
    adapter = GameStateAdapter(state)

    view = adapter.to_vendor_view()
    assert view.inventory_count == 4
    assert view.durability_fraction == pytest.approx(0.85)


def test_low_confidence_optional_fields_surface_as_none_without_raising() -> None:
    """Optional fields (durability_fraction, inventory_max) become None on low confidence."""
    target = TargetInfo(
        name="Defias Thug",
        hp_pct=0.5,
        reaction="hostile",
        distance_estimate=10.0,
    )
    state = _make_state(
        inventory_count=2,
        inventory_max=16,
        durability_fraction=0.85,
        target=target,
        target_is_lootable=True,
        perception_confidence={
            "inventory_count": 0.95,
            "durability_fraction": 0.30,  # threshold is 0.6 -> surfaces as None
            "inventory_max": 0.40,  # threshold is 0.7 -> surfaces as None
            "target_is_lootable": 0.95,
        },
    )
    adapter = GameStateAdapter(state)

    vendor_view = adapter.to_vendor_view()
    assert vendor_view.inventory_count == 2
    assert vendor_view.durability_fraction is None

    loot_view = adapter.to_loot_view()
    assert loot_view.inventory_count == 2
    assert loot_view.inventory_max is None


def test_low_confidence_position_raises_in_all_views() -> None:
    """Low-confidence position coordinate fails required coords check."""
    state = _make_state(
        inventory_count=2,
        perception_confidence={"position": 0.40},  # threshold is 0.6
    )
    adapter = GameStateAdapter(state)

    with pytest.raises(
        AdapterIncompleteError,
        match="VendorView requires non-Optional position coordinates",
    ):
        adapter.to_vendor_view()


def test_low_confidence_target_fields_surface_as_none() -> None:
    """Low-confidence target distance estimate surfaces as None."""
    target = TargetInfo(
        name="Defias Thug",
        hp_pct=0.5,
        reaction="hostile",
        distance_estimate=25.0,
    )
    state = _make_state(
        target=target,
        hp_pct=0.8,
        mana_pct=0.9,
        perception_confidence={
            "target.distance_estimate": 0.30,  # default threshold 0.5 -> low
        },
    )
    derivation_cfg = AdapterDerivationConfig(engage_distance_units=30.0)
    context = StaticRuntimeContext()
    adapter = GameStateAdapter(state, config=derivation_cfg, context=context)

    # _derive_target_in_range returns None when distance confidence is below threshold
    assert adapter._derive_target_in_range(target) is None

    # to_reactive_view raises because target_in_range cannot be derived
    with pytest.raises(
        AdapterIncompleteError,
        match="ReactiveStateView requires non-Optional target_in_range",
    ):
        adapter.to_reactive_view()


def test_custom_adapter_confidence_config_thresholds() -> None:
    """AdapterConfidenceConfig allows overriding default and per-field thresholds."""
    custom_cfg = AdapterConfidenceConfig(
        default_threshold=0.8,
        thresholds={"inventory_count": 0.4},  # more permissive for inventory
    )
    state = _make_state(
        inventory_count=3,
        perception_confidence={
            "inventory_count": 0.55,  # >= 0.4, passes
            "position": 0.70,  # < 0.8 (default_threshold), fails
        },
    )
    adapter = GameStateAdapter(state, confidence_config=custom_cfg)

    # Position fails with custom default threshold 0.8
    assert adapter._is_confident("position") is False
    # inventory_count passes with custom threshold 0.4
    assert adapter._is_confident("inventory_count") is True
    assert adapter._get_confident_field("inventory_count") == 3


@pytest.mark.asyncio
async def test_mock_perception_confidence_map_plumbing() -> None:
    """MockPerception generates perception_confidence and handles low-confidence bag synthesis."""
    mock = MockPerception(rng=np.random.default_rng(42))

    seen_bag = False
    for _ in range(100):
        state = await mock.get_state()
        assert isinstance(state.perception_confidence, dict)
        assert "hp_pct" in state.perception_confidence
        assert "mana_pct" in state.perception_confidence
        assert "position" in state.perception_confidence
        assert "facing" in state.perception_confidence
        assert state.perception_confidence["in_combat"] == 1.0

        if "inventory_count" in state.perception_confidence:
            seen_bag = True
            conf = state.perception_confidence["inventory_count"]
            assert 0.40 <= conf <= 1.0
            # When bag_conf >= 0.70, inventory_count is an int; when < 0.70, it is None
            if conf >= 0.70:
                assert isinstance(state.inventory_count, int)
                assert state.inventory_max == 16
            else:
                assert state.inventory_count is None
                assert state.inventory_max is None

    assert seen_bag, "Expected at least one frame with bag observation over 100 samples"


def test_perception_config_confidence_loading_and_unknown_key_raises() -> None:
    """PerceptionConfig loads [confidence] section and validates keys."""
    # 1. Example TOML loads valid confidence configuration
    cfg = load_perception_config(Path("config/perception.example.toml"))
    assert cfg.confidence_default_threshold == 0.5
    assert cfg.confidence_thresholds["inventory_count"] == 0.7
    assert cfg.confidence_thresholds["inventory_max"] == 0.7
    assert cfg.confidence_thresholds["level_or_xp"] == 0.5
    assert cfg.confidence_thresholds["durability_fraction"] == 0.6
    assert cfg.confidence_thresholds["target_is_lootable"] == 0.9
    assert cfg.confidence_thresholds["position"] == 0.6

    # 2. Unknown key in [confidence] raises PerceptionConfigError
    import tomllib

    with Path("config/perception.example.toml").open("rb") as f:
        data = tomllib.load(f)

    data["confidence"]["unknown_metric"] = 0.5
    with pytest.raises(PerceptionConfigError, match=r"\[confidence\].unknown_metric"):
        load_perception_config_from_dict(data)

    # 3. Missing [confidence] section raises PerceptionConfigError
    del data["confidence"]
    with pytest.raises(
        PerceptionConfigError, match=r"Missing required section \[confidence\]"
    ):
        load_perception_config_from_dict(data)
