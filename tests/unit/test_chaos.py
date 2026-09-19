"""Unit tests for LorenzAttractor (Task 3.3).

Tests cover:
    - Test A: Default construction / initial state validation.
    - Test B: RK4 reference step accuracy against independent RK4 formula.
    - Test C: Determinism across independent instances.
    - Test D: Finite state guarantee (no NaN/Inf) over >= 1,000 steps.
    - Test E: Bounded phase-space behavior (sanity bounds).
    - Test F: Sensitive dependence on initial conditions (divergence).
    - Test G: Normalized signal range [-1.0, +1.0].
    - Test H: Normalized purity (side-effect-free, deterministic).
    - Test I & J: Parameter & initial state input validation.
    - Test K: Returned array mutation isolation.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from wow_bot.internal_dynamics.chaos import LorenzAttractor


def test_default_construction() -> None:
    """Test A: Verify default constructor parameters and initial state accessor."""
    attractor = LorenzAttractor()
    assert attractor.sigma == 10.0
    assert attractor.rho == 28.0
    assert attractor.beta == 2.667
    assert attractor.dt == 0.001
    np.testing.assert_array_equal(attractor.state, np.array([1.0, 1.0, 1.0]))


def test_rk4_reference_step() -> None:
    """Test B: Verify one RK4 integration step against independent reference calculation."""
    sigma = 10.0
    rho = 28.0
    beta = 8.0 / 3.0
    dt = 0.001
    s0 = np.array([1.0, 1.0, 1.0], dtype=np.float64)

    def ref_f(s: np.ndarray) -> np.ndarray:
        x, y, z = s[0], s[1], s[2]
        return np.array([
            sigma * (y - x),
            x * (rho - z) - y,
            x * y - beta * z,
        ], dtype=np.float64)

    k1 = ref_f(s0)
    k2 = ref_f(s0 + 0.5 * dt * k1)
    k3 = ref_f(s0 + 0.5 * dt * k2)
    k4 = ref_f(s0 + dt * k3)
    expected_s1 = s0 + (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)

    attractor = LorenzAttractor(sigma=sigma, rho=rho, beta=beta, dt=dt, initial_state=(1.0, 1.0, 1.0))
    actual_s1 = attractor.step()

    np.testing.assert_allclose(actual_s1, expected_s1, rtol=1e-12, atol=1e-12)


def test_determinism() -> None:
    """Test C: Verify two attractors with identical parameters yield identical trajectories."""
    a = LorenzAttractor(initial_state=(1.5, -2.0, 10.0))
    b = LorenzAttractor(initial_state=(1.5, -2.0, 10.0))

    for _ in range(500):
        sa = a.step()
        sb = b.step()
        np.testing.assert_array_equal(sa, sb)
        assert a.normalized() == b.normalized()


def test_no_nan_or_inf() -> None:
    """Test D: Verify state remains finite over 1,000 steps (roadmap requirement)."""
    attractor = LorenzAttractor()
    for step_idx in range(1000):
        state = attractor.step()
        assert np.all(np.isfinite(state)), f"Non-finite state at step {step_idx}: {state}"
        assert math.isfinite(attractor.normalized())


def test_bounded_phase_space() -> None:
    """Test E: Verify phase space components remain within loose sanity bound (< 200)."""
    attractor = LorenzAttractor()
    max_bound = 200.0

    for _ in range(10000):
        state = attractor.step()
        abs_state = np.abs(state)
        assert np.all(abs_state < max_bound), f"Trajectory exploded beyond sanity bound: {state}"


def test_sensitivity_to_initial_conditions() -> None:
    """Test F: Verify initial perturbation diverges significantly over simulated time."""
    init_a = (1.0, 1.0, 1.0)
    init_b = (1.001, 1.0, 1.0)

    a = LorenzAttractor(initial_state=init_a)
    b = LorenzAttractor(initial_state=init_b)

    initial_diff = np.linalg.norm(np.array(init_a) - np.array(init_b))
    assert math.isclose(initial_diff, 0.001)

    # 20,000 steps at dt=0.001 represents 20 simulated seconds
    num_steps = 20000
    for _ in range(num_steps):
        state_a = a.step()
        state_b = b.step()

    final_diff = float(np.linalg.norm(state_a - state_b))

    # Expect substantial divergence (divergence factor > 1000x initial separation)
    assert final_diff > 1.0, f"Final separation {final_diff} did not demonstrate chaotic divergence"
    assert final_diff / initial_diff > 1000.0


def test_normalized_bounds() -> None:
    """Test G: Verify normalized() output remains strictly in [-1.0, +1.0]."""
    attractor = LorenzAttractor()
    for _ in range(1000):
        attractor.step()
        norm_val = attractor.normalized()
        assert -1.0 <= norm_val <= 1.0


def test_normalized_purity() -> None:
    """Test H: Verify normalized() is side-effect-free and pure."""
    attractor = LorenzAttractor()
    for _ in range(100):
        attractor.step()

    state_before = attractor.state
    n1 = attractor.normalized()
    n2 = attractor.normalized()
    state_after = attractor.state

    assert n1 == n2
    np.testing.assert_array_equal(state_before, state_after)


def test_invalid_dt() -> None:
    """Test I: Verify non-positive or non-finite dt raises ValueError."""
    with pytest.raises(ValueError, match="dt must be positive and finite"):
        LorenzAttractor(dt=0.0)

    with pytest.raises(ValueError, match="dt must be positive and finite"):
        LorenzAttractor(dt=-0.001)

    with pytest.raises(ValueError, match="dt must be positive and finite"):
        LorenzAttractor(dt=float("nan"))


def test_invalid_parameters_and_initial_state() -> None:
    """Test J: Verify invalid sigma, beta, rho or initial_state components raise ValueError."""
    with pytest.raises(ValueError, match="sigma must be positive and finite"):
        LorenzAttractor(sigma=0.0)

    with pytest.raises(ValueError, match="beta must be positive and finite"):
        LorenzAttractor(beta=-1.0)

    with pytest.raises(ValueError, match="rho must be finite"):
        LorenzAttractor(rho=float("inf"))

    with pytest.raises(ValueError, match="initial_state components must all be finite"):
        LorenzAttractor(initial_state=(1.0, float("nan"), 1.0))

    with pytest.raises(ValueError, match="initial_state components must all be finite"):
        LorenzAttractor(initial_state=(float("inf"), 1.0, 1.0))

    with pytest.raises(ValueError, match="initial_state must be a 3-tuple"):
        LorenzAttractor(initial_state=(1.0, 1.0))  # type: ignore[arg-type]


def test_returned_state_isolation() -> None:
    """Test K: Verify modifying returned array from step() or state property does not mutate internal attractor state."""
    attractor = LorenzAttractor()

    step_state = attractor.step()
    original_val = step_state[0]
    step_state[0] = 9999.9

    prop_state = attractor.state
    prop_state[1] = -8888.8

    current_internal = attractor.state
    assert current_internal[0] == original_val
    assert current_internal[0] != 9999.9
    assert current_internal[1] != -8888.8
