"""Unit and acceptance tests for ProgressSample contract alignment (T-FIX-11).

Validates:
  1. The sampler reads real values from a producer that supplies them (no silent zero).
  2. A cumulative producer and an incremental consumer cannot silently disagree
     (mismatch raises ContractMismatchError).
  3. Summary position and inventory deltas use the same incremental aggregation rule.
  4. ProgressDeltaAdapter behaves correctly on initial sample, consecutive steps, and reset.
"""

from __future__ import annotations

import pytest

from wow_bot.analysis.lab_soak_v2 import (
    ResourceSnapshot,
    SoakSample,
    build_soak_summary,
)
from wow_bot.watchdog.metrics import (
    ContractMismatchError,
    ProgressDeltaAdapter,
    ProgressSample,
    extract_incremental_deltas,
)


def test_progress_sample_invariants_and_optional_delta_fields() -> None:
    """Requirement: ProgressSample preserves existing fields and accepts additive optional delta fields."""
    # Standard absolute sample without deltas
    sample1 = ProgressSample(
        ts=1.0,
        position=(10.0, 20.0),
        inventory_count=5,
        level_or_xp=1.0,
        successful_actions_total=10,
        reflex_ticks_total=20,
    )
    assert sample1.position_delta is None
    assert sample1.inventory_delta is None

    # Sample with explicitly populated deltas
    sample2 = ProgressSample(
        ts=2.0,
        position=(13.0, 24.0),
        inventory_count=7,
        level_or_xp=1.0,
        successful_actions_total=12,
        reflex_ticks_total=40,
        position_delta=5.0,
        inventory_delta=2,
    )
    assert sample2.position_delta == 5.0
    assert sample2.inventory_delta == 2

    # Invariants on delta fields
    with pytest.raises(ValueError, match="position_delta"):
        ProgressSample(
            ts=3.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
            position_delta=-1.0,
        )

    with pytest.raises(ValueError, match="position_delta"):
        ProgressSample(
            ts=3.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
            position_delta=float("nan"),
        )

    with pytest.raises(ValueError, match="inventory_delta"):
        ProgressSample(
            ts=3.0,
            position=(0.0, 0.0),
            inventory_count=0,
            level_or_xp=0.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
            inventory_delta=-5,
        )


def test_progress_delta_adapter_computes_accurate_deltas() -> None:
    """ProgressDeltaAdapter computes correct incremental step deltas from cumulative stream."""
    adapter = ProgressDeltaAdapter()

    # Step 1: initial baseline
    s1 = ProgressSample(
        ts=10.0,
        position=(100.0, 100.0),
        inventory_count=20,
        level_or_xp=5.0,
        successful_actions_total=0,
        reflex_ticks_total=0,
    )
    adapted1 = adapter.adapt(s1)
    assert adapted1.position_delta == 0.0
    assert adapted1.inventory_delta == 0

    # Step 2: moved (dx=3, dy=4 -> dist=5), gained 2 items
    s2 = ProgressSample(
        ts=11.0,
        position=(103.0, 104.0),
        inventory_count=22,
        level_or_xp=5.0,
        successful_actions_total=2,
        reflex_ticks_total=20,
    )
    adapted2 = adapter.adapt(s2)
    assert adapted2.position_delta == pytest.approx(5.0)
    assert adapted2.inventory_delta == 2

    # Step 3: stationary, items dropped/sold (inventory drops to 15 -> clamped to 0 delta)
    s3 = ProgressSample(
        ts=12.0,
        position=(103.0, 104.0),
        inventory_count=15,
        level_or_xp=5.0,
        successful_actions_total=4,
        reflex_ticks_total=40,
    )
    adapted3 = adapter.adapt(s3)
    assert adapted3.position_delta == 0.0
    assert adapted3.inventory_delta == 0

    # Reset behavior
    adapter.reset()
    adapted_reset = adapter.adapt(s3)
    assert adapted_reset.position_delta == 0.0
    assert adapted_reset.inventory_delta == 0


def test_sampler_reads_real_values_from_producer_no_silent_zero() -> None:
    """Acceptance: Sampler reads real non-zero values from a producer that supplies them."""
    adapter = ProgressDeltaAdapter()

    raw_samples = [
        ProgressSample(
            ts=1.0,
            position=(0.0, 0.0),
            inventory_count=10,
            level_or_xp=1.0,
            successful_actions_total=0,
            reflex_ticks_total=0,
        ),
        ProgressSample(
            ts=2.0,
            position=(3.0, 4.0),
            inventory_count=13,
            level_or_xp=1.0,
            successful_actions_total=2,
            reflex_ticks_total=20,
        ),
        ProgressSample(
            ts=3.0,
            position=(6.0, 8.0),
            inventory_count=17,
            level_or_xp=1.0,
            successful_actions_total=5,
            reflex_ticks_total=40,
        ),
    ]

    soak_samples: list[SoakSample] = []
    snap = ResourceSnapshot(ts=1.0, cpu_percent=10.0, rss_bytes=1000, log_size_bytes=500)

    for raw in raw_samples:
        adapted = adapter.adapt(raw)
        s = SoakSample(
            ts=raw.ts,
            cpu_percent=snap.cpu_percent,
            rss_bytes=snap.rss_bytes,
            log_size_bytes=snap.log_size_bytes,
            position_delta=adapted.position_delta if adapted.position_delta is not None else 0.0,
            inventory_delta=adapted.inventory_delta if adapted.inventory_delta is not None else 0,
            successful_actions_total=raw.successful_actions_total,
            reflex_ticks_total=raw.reflex_ticks_total,
        )
        soak_samples.append(s)

    # First sample is baseline
    assert soak_samples[0].position_delta == 0.0
    assert soak_samples[0].inventory_delta == 0

    # Subsequent samples read REAL non-zero values (no silent zero)
    assert soak_samples[1].position_delta == pytest.approx(5.0)
    assert soak_samples[1].inventory_delta == 3

    assert soak_samples[2].position_delta == pytest.approx(5.0)
    assert soak_samples[2].inventory_delta == 4


def test_cumulative_producer_and_incremental_consumer_cannot_silently_disagree() -> None:
    """Acceptance: A cumulative producer and an incremental consumer cannot silently disagree.

    Mismatch must be impossible by type or raise an explicit ContractMismatchError.
    """
    cumulative_sample = ProgressSample(
        ts=1.0,
        position=(100.0, 200.0),
        inventory_count=42,
        level_or_xp=10.0,
        successful_actions_total=5,
        reflex_ticks_total=100,
    )

    # When an incremental consumer attempts to extract deltas directly from a cumulative sample,
    # it must raise ContractMismatchError rather than silently evaluating to 0.0 / 0.
    with pytest.raises(ContractMismatchError, match="carries absolute metrics"):
        extract_incremental_deltas(cumulative_sample)

    # When correctly bridged via ProgressDeltaAdapter, extraction succeeds
    adapter = ProgressDeltaAdapter()
    adapted = adapter.adapt(cumulative_sample)
    pos_d, inv_d = extract_incremental_deltas(adapted)
    assert pos_d == 0.0
    assert inv_d == 0

    # Next step
    cumulative_step2 = ProgressSample(
        ts=2.0,
        position=(103.0, 204.0),
        inventory_count=45,
        level_or_xp=10.0,
        successful_actions_total=6,
        reflex_ticks_total=120,
    )
    adapted2 = adapter.adapt(cumulative_step2)
    pos_d2, inv_d2 = extract_incremental_deltas(adapted2)
    assert pos_d2 == pytest.approx(5.0)
    assert inv_d2 == 3


def test_summary_position_and_inventory_deltas_use_same_rule() -> None:
    """Acceptance: Summary position and inventory deltas use the same incremental sum aggregation rule."""
    samples = [
        SoakSample(
            ts=10.0,
            cpu_percent=5.0,
            rss_bytes=1000,
            log_size_bytes=100,
            position_delta=1.5,
            inventory_delta=2,
            successful_actions_total=1,
            reflex_ticks_total=10,
        ),
        SoakSample(
            ts=11.0,
            cpu_percent=6.0,
            rss_bytes=1100,
            log_size_bytes=200,
            position_delta=3.5,
            inventory_delta=4,
            successful_actions_total=2,
            reflex_ticks_total=20,
        ),
        SoakSample(
            ts=12.0,
            cpu_percent=7.0,
            rss_bytes=1200,
            log_size_bytes=300,
            position_delta=2.0,
            inventory_delta=1,
            successful_actions_total=3,
            reflex_ticks_total=30,
        ),
    ]

    summary = build_soak_summary(samples)

    # Both position_delta_total and inventory_delta_total must sum all per-sample deltas
    expected_pos_total = 1.5 + 3.5 + 2.0  # 7.0
    expected_inv_total = 2 + 4 + 1        # 7

    assert summary.position_delta_total == pytest.approx(expected_pos_total)
    assert summary.inventory_delta_total == expected_inv_total
