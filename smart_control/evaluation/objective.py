"""Objective-function adapter used by optimizers.

Translates candidate parameter vectors into a scalar cost to be minimized.
Combines the five required performance criteria (IAE, Overshoot, Settling Time,
Rise Time, and Steady-State Error) with configurable weights and instability penalties.

Architecture:
-------------
- `calculate_scalar_fitness`: Pure function combining `PerformanceMetrics` into a float.
- `build_objective`: Factory creating an `ObjectiveFunction` callable for any controller
  and plant pair, conforming to `optimizers.base.BaseOptimizer`.
"""
from __future__ import annotations

import math
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

import smart_control.config as cfg
from smart_control.evaluation.metrics import calculate_all_metrics
from smart_control.types import PerformanceMetrics, SimulationResult

# Finite penalty used for unstable or invalid parameter evaluations
DEFAULT_PENALTY: float = 1e4

# Default weights if not configured in config.py
DEFAULT_WEIGHTS: Dict[str, float] = {
    "iae": 1.0,
    "overshoot": 0.5,
    "settling_time": 0.3,
    "steady_state_error": 1.0,
    "rise_time": 0.0,
}


def _get_active_weights(custom_weights: Optional[Dict[str, float]] = None) -> Dict[str, float]:
    """Merge custom weights with config.OBJECTIVE_WEIGHTS and built-in defaults."""
    weights = dict(DEFAULT_WEIGHTS)
    cfg_weights = getattr(cfg, "OBJECTIVE_WEIGHTS", {})
    if isinstance(cfg_weights, dict):
        weights.update(cfg_weights)
    if custom_weights:
        weights.update(custom_weights)
    return weights


def calculate_scalar_fitness(
    metrics: PerformanceMetrics,
    weights: Optional[Dict[str, float]] = None,
    penalty_value: float = DEFAULT_PENALTY,
    max_duration: float = 20.0,
) -> float:
    """Combine PerformanceMetrics into a single scalar fitness value to minimize.

    Handles unsettled responses and NaN/Inf values by applying appropriate penalties.

    Parameters
    ----------
    metrics : PerformanceMetrics
        Calculated performance metrics for a simulation run.
    weights : dict, optional
        Weights dictionary for 'iae', 'overshoot', 'settling_time', 'steady_state_error', 'rise_time'.
    penalty_value : float, optional
        Value returned when response diverged or produced NaN/Inf (default: 1e4).
    max_duration : float, optional
        Simulation duration used for penalizing unsettled responses (default: 20.0s).

    Returns
    -------
    float
        Scalar cost value to be minimized.
    """
    w = _get_active_weights(weights)

    # 1. Check for invalid or divergent metrics
    for val in (metrics.overshoot, metrics.steady_state_error, metrics.iae):
        if val is None or not math.isfinite(val):
            return float(penalty_value)

    # 2. Extract values with penalization for unreached thresholds
    iae = float(metrics.iae)
    overshoot = float(metrics.overshoot)
    sse = float(metrics.steady_state_error)

    # If settling time is None, response did not settle in the simulation horizon
    if metrics.settling_time is None or not math.isfinite(metrics.settling_time):
        ts = max_duration * 1.5 + 50.0  # Heavy settling penalty
    else:
        ts = float(metrics.settling_time)

    # If rise time is None, response never reached 90% of the setpoint
    if metrics.rise_time is None or not math.isfinite(metrics.rise_time):
        tr = max_duration * 1.5 + 20.0  # Rise penalty
    else:
        tr = float(metrics.rise_time)

    # 3. Weighted sum of criteria
    cost = (
        w.get("iae", 1.0) * iae
        + w.get("overshoot", 0.5) * overshoot
        + w.get("settling_time", 0.3) * ts
        + w.get("steady_state_error", 0.0) * sse
        + w.get("rise_time", 0.0) * tr
    )

    return float(cost) if math.isfinite(cost) else float(penalty_value)


def build_objective(
    controller: Any,
    plant: Any,
    simulation_config: Optional[Dict[str, Any]] = None,
    metric_config: Optional[Dict[str, Any]] = None,
    objective_config: Optional[Dict[str, Any]] = None,
    simulator: Optional[Callable[[Any], Optional[SimulationResult]]] = None,
) -> Callable[[Sequence[float]], float]:
    """Create and return a callable objective function for an optimizer.

    The returned function:
    1. Receives candidate parameter vector x.
    2. Updates controller parameters via `controller.set_parameters(x)`.
    3. Simulates closed-loop response.
    4. Computes performance metrics.
    5. Returns a scalar cost value (minimized).

    Parameters
    ----------
    controller : BaseController
        Controller instance to optimize.
    plant : TransferFunction or StateSpace
        Plant model to control.
    simulation_config : dict, optional
        Simulation parameters ('time_step', 'time_end', 'reference_value').
    metric_config : dict, optional
        Metric thresholds and tolerances.
    objective_config : dict, optional
        Objective weights and penalties.
    simulator : callable, optional
        Custom simulation function f(controller) -> SimulationResult.

    Returns
    -------
    Callable[[Sequence[float]], float]
        Objective function f(x) -> scalar cost.
    """
    sim_cfg = dict(getattr(cfg, "SIMULATION", {}))
    if simulation_config:
        sim_cfg.update(simulation_config)

    t_end = float(sim_cfg.get("time_end", getattr(cfg, "T_END", 20.0)))

    # Use the project's single closed-loop simulation implementation. Do not
    # probe it by running an extra simulation here: the optimizer will call the
    # objective for its first candidate, and the runner supplies this callable
    # explicitly when it needs to retain the final best response.
    active_simulator = simulator
    if active_simulator is None:
        from smart_control.plants.simulation import simulate_closed_loop
        active_simulator = lambda ctrl: simulate_closed_loop(plant, ctrl, sim_cfg)

    weights = _get_active_weights(objective_config)
    penalty_val = float(weights.get("penalty", DEFAULT_PENALTY))

    def objective_fn(parameters: Sequence[float]) -> float:
        try:
            params_arr = np.asarray(parameters, dtype=float)
            if not np.all(np.isfinite(params_arr)):
                return penalty_val
            controller.set_parameters(parameters)
        except Exception:
            return penalty_val

        try:
            sim_result = active_simulator(controller)
            if sim_result is None:
                return penalty_val
        except Exception:
            return penalty_val

        # Compute metrics
        try:
            metrics = calculate_all_metrics(sim_result, metric_config)
            cost = calculate_scalar_fitness(
                metrics,
                weights=weights,
                penalty_value=penalty_val,
                max_duration=t_end,
            )
            return cost
        except Exception:
            return penalty_val

    return objective_fn
