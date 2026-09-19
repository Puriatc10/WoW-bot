"""Unit tests for Drives core (Task 3.1)."""

from __future__ import annotations

import numpy as np
import pytest

from wow_bot.internal_dynamics.drives import Drives
from wow_bot.shared.events import DEATH, EVENT_EFFECTS
from wow_bot.shared.interfaces import Event


def test_drives_initial_state() -> None:
    """Test A — Initial state is baseline (0.5) across all 5 drives."""
    drives = Drives(config=None)
    vec = drives.vector
    assert isinstance(vec, np.ndarray)
    assert vec.shape == (5,)
    assert vec.dtype == np.float64
    assert np.allclose(vec, 0.5)
    assert np.all((vec >= 0.0) & (vec <= 1.0))


def test_drives_boundedness_1000_steps() -> None:
    """Test B — 1000 steps with step and decay keep all drives bounded in [0, 1]."""
    drives = Drives(config=None)
    for i in range(1000):
        # Deterministic varying chaos component
        chaos_val = np.sin(i * 0.1)
        drives.step(dt=0.1, chaos_component=chaos_val)
        drives.decay(dt=0.1)

        vec = drives.vector
        assert vec.shape == (5,)
        assert np.all(np.isfinite(vec))
        assert np.all(vec >= 0.0)
        assert np.all(vec <= 1.0)


def test_death_event_effect() -> None:
    """Test C — Death event immediately depresses aggression according to EVENT_EFFECTS."""
    drives = Drives(config=None)
    death_event = Event(type=DEATH, timestamp=1000.0)

    initial_aggression = drives.get_drive("aggression")
    expected_delta = EVENT_EFFECTS[DEATH]["aggression"]

    drives.apply_event(death_event)
    new_aggression = drives.get_drive("aggression")

    assert np.isclose(new_aggression, initial_aggression + expected_delta)


def test_recovery_after_event() -> None:
    """Test D — Recovery (decay) after event moves aggression back toward baseline."""
    drives = Drives(config=None)
    death_event = Event(type=DEATH, timestamp=1000.0)
    drives.apply_event(death_event)

    depressed_aggression = drives.get_drive("aggression")  # 0.35
    baseline = 0.5
    assert depressed_aggression < baseline

    # Decay over time
    drives.decay(dt=100.0)
    recovered_aggression = drives.get_drive("aggression")

    # Should move closer to baseline without overshooting
    assert recovered_aggression > depressed_aggression
    assert recovered_aggression <= baseline


def test_clamping() -> None:
    """Test E — Clamping at upper (1.0) and lower (0.0) bounds."""
    drives = Drives(config=None, initial_drives={"hunger": 0.95, "fatigue": 0.05})

    # Event pushing hunger above 1.0 and fatigue below 0.0
    # Let's create custom delta or apply event multiple times
    fake_rare_loot = Event(type="rare_loot", timestamp=1.0)  # hunger delta +0.20
    drives.apply_event(fake_rare_loot)
    assert drives.get_drive("hunger") == 1.0

    fake_death = Event(type=DEATH, timestamp=2.0)  # aggression delta -0.15
    for _ in range(10):
        drives.apply_event(fake_death)
    assert drives.get_drive("aggression") == 0.0


def test_unknown_event() -> None:
    """Test F — Unknown event causes no exception and leaves vector unchanged."""
    drives = Drives(config=None)
    vec_before = drives.vector.copy()

    unknown_evt = Event(type="unknown_alien_event", timestamp=10.0)
    drives.apply_event(unknown_evt)

    vec_after = drives.vector
    assert np.array_equal(vec_before, vec_after)


def test_invalid_dt() -> None:
    """Test G — Negative dt raises ValueError in step and decay."""
    drives = Drives(config=None)

    with pytest.raises(ValueError):
        drives.step(dt=-1.0, chaos_component=0.0)

    with pytest.raises(ValueError):
        drives.decay(dt=-1.0)


def test_zero_dt() -> None:
    """Test H — Zero dt causes no continuous state evolution."""
    drives = Drives(config=None)
    vec_before = drives.vector.copy()

    drives.step(dt=0.0, chaos_component=1.0)
    drives.decay(dt=0.0)

    vec_after = drives.vector
    assert np.array_equal(vec_before, vec_after)


def test_vector_isolation() -> None:
    """Test I — Mutating returned vector does not alter internal state."""
    drives = Drives(config=None)
    vec = drives.vector
    vec[0] = 999.0

    new_vec = drives.vector
    assert new_vec[0] == 0.5
    assert drives.get_drive("hunger") == 0.5


def test_rate_differences() -> None:
    """Test that fatigue dynamics > social dynamics as required by roadmap."""
    drives = Drives(config=None)
    initial_fatigue = drives.get_drive("fatigue")
    initial_social = drives.get_drive("social")

    drives.step(dt=10.0, chaos_component=0.0)

    fatigue_change = abs(drives.get_drive("fatigue") - initial_fatigue)
    social_change = abs(drives.get_drive("social") - initial_social)

    assert fatigue_change > social_change
