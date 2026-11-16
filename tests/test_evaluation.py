"""Unit tests for performance metrics and objective function evaluation.

Tests cover:
- Rise time calculation (including linear interpolation and unreached thresholds).
- Settling time calculation (tolerance band, multi-crossing, and unsettled responses).
- Percentage overshoot calculation (underdamped, overdamped, and negative steps).
- Steady-state error calculation (zero error, non-zero error, and tail averaging).
- IAE integration (analytic test cases).
- calculate_all_metrics container unpacker and edge-case handling.
- calculate_scalar_fitness penalization and weighting.
- build_objective end-to-end integration with plant models and controllers.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from smart_control.evaluation.metrics import (
    calculate_all_metrics,
    calculate_iae,
    calculate_overshoot,
    calculate_rise_time,
    calculate_settling_time,
    calculate_steady_state_error,
)
from smart_control.evaluation.objective import (
    DEFAULT_PENALTY,
    build_objective,
    calculate_scalar_fitness,
)
from smart_control.types import PerformanceMetrics, SimulationResult


# ---------------------------------------------------------------------------
# 1. Rise Time Tests
# ---------------------------------------------------------------------------
def test_rise_time_linear_ramp():
    """A linear ramp from 0 to 1 between t=1.0 and t=3.0s has Tr = 1.6s."""
    t = np.linspace(0, 5, 501)
    # y stays 0 until t=1, ramps to 1 at t=3, then stays 1
    y = np.clip((t - 1.0) / 2.0, 0.0, 1.0)
    
    tr = calculate_rise_time(t, y, reference=1.0, low_percent=10.0, high_percent=90.0)
    assert tr is not None
    # 10% is at t=1.2, 90% is at t=2.8 => tr = 1.6
    assert pytest.approx(tr, abs=1e-3) == 1.6


def test_rise_time_unreached_threshold():
    """If the output stalls at 0.5 (never reaches 90%), rise time is None."""
    t = np.linspace(0, 10, 100)
    y = np.full_like(t, 0.5)
    tr = calculate_rise_time(t, y, reference=1.0)
    assert tr is None


def test_rise_time_negative_step():
    """Step from 0 down to -2.0."""
    t = np.linspace(0, 4, 401)
    # Ramps from 0 to -2 between t=0 and t=2
    y = np.clip(-t, -2.0, 0.0)
    tr = calculate_rise_time(t, y, reference=-2.0)
    assert tr is not None
    # 10% of -2 is -0.2 (t=0.2), 90% is -1.8 (t=1.8) => Tr = 1.6s
    assert pytest.approx(tr, abs=1e-2) == 1.6


# ---------------------------------------------------------------------------
# 2. Settling Time Tests
# ---------------------------------------------------------------------------
def test_settling_time_ideal_exponential():
    """First-order response y(t) = 1 - exp(-t/tau). Settles to 2% at t = 3.912 * tau."""
    tau = 1.0
    t = np.linspace(0, 10, 1001)
    y = 1.0 - np.exp(-t / tau)
    
    ts = calculate_settling_time(t, y, reference=1.0, tolerance=0.02)
    assert ts is not None
    expected_ts = -tau * math.log(0.02)  # ~3.912s
    assert pytest.approx(ts, abs=0.02) == expected_ts


def test_settling_time_with_damped_oscillations():
    """Response that leaves and re-enters the 2% band multiple times."""
    t = np.linspace(0, 20, 2001)
    # Damped sinusoid: y = 1 - exp(-0.5t) * cos(3t)
    y = 1.0 - np.exp(-0.5 * t) * np.cos(3.0 * t)
    
    ts = calculate_settling_time(t, y, reference=1.0, tolerance=0.02)
    assert ts is not None
    # Verify that for all t >= ts, |y - 1| <= 0.02
    settled_indices = np.where(t >= ts)[0]
    assert np.all(np.abs(y[settled_indices] - 1.0) <= 0.02 + 1e-4)


def test_settling_time_unsettled():
    """Undamped oscillation never settles => settling time is None."""
    t = np.linspace(0, 10, 1001)
    y = 1.0 + 0.1 * np.sin(5.0 * t)  # Oscillates +/-10%, well outside 2%
    
    ts = calculate_settling_time(t, y, reference=1.0, tolerance=0.02)
    assert ts is None


def test_settling_time_already_settled():
    """Response starting at reference stays inside band => Ts = 0.0."""
    t = np.linspace(0, 5, 101)
    y = np.ones_like(t)
    ts = calculate_settling_time(t, y, reference=1.0, tolerance=0.02)
    assert ts == 0.0


# ---------------------------------------------------------------------------
# 3. Overshoot Tests
# ---------------------------------------------------------------------------
def test_overshoot_underdamped():
    """Peak output of 1.25 on reference 1.0 has 25% overshoot."""
    y = np.array([0.0, 0.5, 1.0, 1.25, 1.10, 1.02, 1.0])
    os_val = calculate_overshoot(y, reference=1.0)
    assert pytest.approx(os_val, abs=1e-4) == 25.0


def test_overshoot_overdamped():
    """Response that approaches 1.0 monotonically has 0% overshoot."""
    y = np.array([0.0, 0.4, 0.7, 0.9, 0.98, 1.0])
    os_val = calculate_overshoot(y, reference=1.0)
    assert os_val == 0.0


def test_overshoot_negative_step():
    """Negative step to -1.0 with dip to -1.3 has 30% overshoot."""
    y = np.array([0.0, -0.5, -1.0, -1.3, -1.05, -1.0])
    os_val = calculate_overshoot(y, reference=-1.0)
    assert pytest.approx(os_val, abs=1e-4) == 30.0


# ---------------------------------------------------------------------------
# 4. Steady-State Error Tests
# ---------------------------------------------------------------------------
def test_steady_state_error_zero():
    """Response perfectly settling to reference has 0 steady-state error."""
    y = np.linspace(0, 1, 100)
    y[-20:] = 1.0
    ess = calculate_steady_state_error(y, reference=1.0)
    assert pytest.approx(ess, abs=1e-7) == 0.0


def test_steady_state_error_nonzero():
    """Response settling to 0.85 on reference 1.0 has Ess = 0.15."""
    y = np.full(100, 0.85)
    ess = calculate_steady_state_error(y, reference=1.0)
    assert pytest.approx(ess, abs=1e-5) == 0.15


# ---------------------------------------------------------------------------
# 5. IAE Tests
# ---------------------------------------------------------------------------
def test_iae_constant_error():
    """Error of constant 0.5 over 4 seconds => IAE = 2.0."""
    t = np.linspace(0, 4, 401)
    y = np.full_like(t, 0.5)
    iae_val = calculate_iae(t, y, reference=1.0)
    assert pytest.approx(iae_val, abs=1e-3) == 2.0


def test_iae_zero_error():
    """Zero tracking error => IAE = 0.0."""
    t = np.linspace(0, 5, 101)
    y = np.ones_like(t)
    iae_val = calculate_iae(t, y, reference=1.0)
    assert pytest.approx(iae_val, abs=1e-7) == 0.0


# ---------------------------------------------------------------------------
# 6. Container & Edge-Case Tests (calculate_all_metrics)
# ---------------------------------------------------------------------------
def test_calculate_all_metrics_simulation_result():
    """Check calculate_all_metrics on SimulationResult container."""
    t = np.linspace(0, 10, 501)
    y = 1.0 - np.exp(-t)
    sim = SimulationResult(time=t, reference=1.0, output=y)

    metrics = calculate_all_metrics(sim)
    assert isinstance(metrics, PerformanceMetrics)
    assert metrics.settling_time is not None
    assert metrics.rise_time is not None
    assert metrics.overshoot == 0.0
    assert metrics.steady_state_error < 0.01
    assert metrics.iae > 0.0

    # Test dictionary conversion and key indexing
    d = metrics.to_dict()
    assert "iae" in d
    assert metrics["overshoot"] == 0.0


def test_calculate_all_metrics_unstable_divergence():
    """Divergent or NaN output produces NaN/None metrics safely."""
    t = np.linspace(0, 5, 51)
    y_diverged = np.array([0.0, 10.0, 1000.0, 1e8, float("nan")] + [0.0] * 46)
    sim = SimulationResult(time=t, reference=1.0, output=y_diverged)

    metrics = calculate_all_metrics(sim)
    assert metrics.settling_time is None
    assert metrics.rise_time is None
    assert math.isnan(metrics.overshoot)


# ---------------------------------------------------------------------------
# 7. Scalar Fitness & Objective Tests
# ---------------------------------------------------------------------------
def test_scalar_fitness_weighting():
    """Test scalar cost computation with weights."""
    metrics = PerformanceMetrics(
        settling_time=2.0,
        rise_time=1.0,
        overshoot=10.0,
        steady_state_error=0.05,
        iae=1.5,
    )
    weights = {
        "iae": 1.0,
        "overshoot": 0.5,
        "settling_time": 0.3,
        "steady_state_error": 2.0,
        "rise_time": 0.1,
    }
    # Expected: 1.0*1.5 + 0.5*10.0 + 0.3*2.0 + 2.0*0.05 + 0.1*1.0 = 1.5 + 5.0 + 0.6 + 0.1 + 0.1 = 7.3
    cost = calculate_scalar_fitness(metrics, weights=weights)
    assert pytest.approx(cost, abs=1e-4) == 7.3


def test_scalar_fitness_unsettled_penalty():
    """Unsettled response (settling_time=None) receives a heavy penalty."""
    m_settled = PerformanceMetrics(settling_time=2.0, rise_time=1.0, overshoot=5.0, steady_state_error=0.0, iae=1.0)
    m_unsettled = PerformanceMetrics(settling_time=None, rise_time=1.0, overshoot=5.0, steady_state_error=0.0, iae=1.0)

    cost_settled = calculate_scalar_fitness(m_settled)
    cost_unsettled = calculate_scalar_fitness(m_unsettled)
    assert cost_unsettled > cost_settled + 20.0


def test_build_objective_integration():
    """End-to-end integration: build_objective with NeuralNetworkController and second_order plant."""
    from smart_control.controllers.neural_network import NeuralNetworkController
    from smart_control.plants.models import get_second_order_plant

    plant = get_second_order_plant()
    controller = NeuralNetworkController()

    objective_fn = build_objective(controller, plant)
    assert callable(objective_fn)

    # Evaluate zero parameter vector
    params = [0.0] * controller.n_params
    cost = objective_fn(params)
    assert math.isfinite(cost)
    assert cost > 0.0

    # NaN parameters should immediately yield penalty without exception
    nan_params = [float("nan")] * controller.n_params
    bad_cost = objective_fn(nan_params)
    assert bad_cost == DEFAULT_PENALTY

    # Wrong parameter dimensions should yield penalty without exception
    wrong_dim_cost = objective_fn([1.0, 2.0])
    assert wrong_dim_cost == DEFAULT_PENALTY
