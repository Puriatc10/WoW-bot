"""Unit tests for OscillatorBank (Task 3.2)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from wow_bot.internal_dynamics.oscillators import OscillatorBank


def test_canonical_frequencies() -> None:
    """Test A: Verify canonical frequencies count and exact roadmap values."""
    bank = OscillatorBank(seed=42)
    freqs = bank.frequencies
    assert len(freqs) == 5
    expected = [1.0 / 18000.0, 1.0 / 5400.0, 1.0 / 1200.0, 1.0 / 300.0, 1.0 / 60.0]
    for actual_f, expected_f in zip(freqs, expected, strict=True):
        assert actual_f == pytest.approx(expected_f, rel=1e-9)


def test_seeded_reproducibility() -> None:
    """Test C: Two banks with the same seed produce identical outputs."""
    a = OscillatorBank(seed=123)
    b = OscillatorBank(seed=123)

    assert a.phases == pytest.approx(b.phases)

    for dt in [0.1, 0.2, 1.0, 5.0, 60.0]:
        out_a = a.step(dt)
        out_b = b.step(dt)
        assert out_a == pytest.approx(out_b, abs=1e-12)
        assert a.phases == pytest.approx(b.phases, abs=1e-12)


def test_different_seeds() -> None:
    """Test D: Different seeds produce different initial phases."""
    a = OscillatorBank(seed=101)
    b = OscillatorBank(seed=202)
    assert not np.allclose(a.phases, b.phases)


def test_zero_dt() -> None:
    """Test E: dt=0 does not advance phases and returns aggregate state output."""
    bank = OscillatorBank(seed=42)
    initial_phases = bank.phases
    _ = bank.step(0.0)
    assert bank.phases == pytest.approx(initial_phases)

    # Perform a normal step to advance state
    bank.step(1.0)
    phases_after_step = bank.phases

    out2 = bank.step(0.0)
    assert bank.phases == pytest.approx(phases_after_step)
    # Consecutive step(0) gives exact same aggregate output
    out3 = bank.step(0.0)
    assert out2 == pytest.approx(out3)


def test_negative_dt() -> None:
    """Test F: Negative dt raises ValueError."""
    bank = OscillatorBank(seed=42)
    with pytest.raises(ValueError, match="dt must be non-negative"):
        bank.step(-0.1)


def test_bounded_output_and_long_run() -> None:
    """Test G & Test H: Bounded output (abs(output) <= 0.35) and phase wrapping over long run."""
    bank = OscillatorBank(seed=999)
    # Simulate 10,000 steps with dt=1.0 s
    for _ in range(10000):
        out = bank.step(1.0)
        assert math.isfinite(out)
        assert abs(out) <= 0.35
        for p in bank.phases:
            assert math.isfinite(p)
            assert 0.0 <= p < 2.0 * math.pi


def test_property_isolation() -> None:
    """Test I: Mutating returned frequencies or phases list does not alter bank state."""
    bank = OscillatorBank(seed=42)

    freqs = bank.frequencies
    original_freqs = list(freqs)
    freqs[0] = 999.0
    assert bank.frequencies == pytest.approx(original_freqs)

    phases = bank.phases
    original_phases = list(phases)
    phases[0] = 999.0
    assert bank.phases == pytest.approx(original_phases)


def test_deterministic_evolution_independent_of_wall_clock() -> None:
    """Test J: Output sequence is identical regardless of execution timing."""
    dts = [0.1, 0.5, 1.0, 10.0, 100.0]

    bank1 = OscillatorBank(seed=777)
    outputs1 = [bank1.step(dt) for dt in dts]

    bank2 = OscillatorBank(seed=777)
    outputs2 = [bank2.step(dt) for dt in dts]

    assert outputs1 == pytest.approx(outputs2)


def test_relative_oscillator_speeds() -> None:
    """Test K: High-frequency oscillator accumulates more phase than low-frequency oscillator."""
    # Initialize bank with all zero initial phases using configured initial phases
    config = {"initial_phases": [0.0, 0.0, 0.0, 0.0, 0.0]}
    bank = OscillatorBank(config=config)

    dt = 10.0  # 10 seconds simulation time
    bank.step(dt)

    phases = bank.phases
    # Fastest oscillator is 1/60 Hz (index 4), slowest is 1/18000 Hz (index 0)
    slowest_delta = phases[0]
    fastest_delta = phases[4]

    assert fastest_delta > slowest_delta
    expected_fastest = (2.0 * math.pi * (1.0 / 60.0) * dt) % (2.0 * math.pi)
    expected_slowest = (2.0 * math.pi * (1.0 / 18000.0) * dt) % (2.0 * math.pi)

    assert fastest_delta == pytest.approx(expected_fastest)
    assert slowest_delta == pytest.approx(expected_slowest)


def test_configured_initial_phases() -> None:
    """Test priority 1: Configured initial phases."""
    custom_phases = [0.1, 0.2, 0.3, 0.4, 0.5]
    config = {"initial_phases": custom_phases}
    bank = OscillatorBank(config=config, seed=42)  # Config takes priority over seed
    assert bank.phases == pytest.approx(custom_phases)
