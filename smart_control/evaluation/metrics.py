"""Performance metrics calculation for closed-loop control responses.

Implements the five core control-engineering performance criteria:
1. Rise Time (Tr): Time taken to rise from 10% to 90% of the step change.
2. Settling Time (Ts): Time after which the response remains within the +/-2%
   (or configured) tolerance band around the reference value.
3. Percentage Overshoot (%OS): Peak output exceeding the reference value as a
   percentage of the step magnitude.
4. Steady-State Error (Ess): Absolute difference between the reference and the
   settled output (averaged over the final tail to filter minor oscillation).
5. Integral of Absolute Error (IAE): Integral of |reference - output| over time.

All functions accept Python lists or NumPy arrays and handle edge cases
(unsettled responses, divergence, negative steps, etc.) safely.
"""
from __future__ import annotations

import math
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np

from smart_control.types import PerformanceMetrics, SimulationResult

# Numerical integration: prefer numpy.trapezoid (NumPy 2.0+), fall back to trapz
_trapezoid = getattr(np, "trapezoid", np.trapz)


def _to_1d_array(arr: Any, name: str = "array") -> np.ndarray:
    """Safely convert input to a 1D float NumPy array."""
    if arr is None:
        raise ValueError(f"{name} cannot be None")
    res = np.asarray(arr, dtype=float).ravel()
    if res.size == 0:
        raise ValueError(f"{name} cannot be empty")
    return res


def calculate_rise_time(
    time: Sequence[float],
    output: Sequence[float],
    reference: Union[float, Sequence[float]],
    low_percent: float = 10.0,
    high_percent: float = 90.0,
) -> Optional[float]:
    """Calculate rise time (Tr) using sub-sample linear interpolation.

    Standard definition: Time for the response to transition from low_percent
    (default 10%) to high_percent (default 90%) of the total step change.

    Parameters
    ----------
    time : Sequence[float]
        Monotonically increasing time vector.
    output : Sequence[float]
        System output response values y(t).
    reference : float or Sequence[float]
        Setpoint reference value r(t) or constant scalar r.
    low_percent : float, optional
        Lower threshold percentage (default: 10.0).
    high_percent : float, optional
        Upper threshold percentage (default: 90.0).

    Returns
    -------
    Optional[float]
        Rise time in seconds, or None if thresholds are never crossed or data is invalid.
    """
    t = _to_1d_array(time, "time")
    y = _to_1d_array(output, "output")

    if len(t) != len(y):
        raise ValueError(f"Length mismatch: time ({len(t)}) vs output ({len(y)})")
    if len(t) < 2:
        return 0.0

    # Determine reference target (final value if sequence)
    if isinstance(reference, (int, float, np.number)):
        r_target = float(reference)
    else:
        ref_arr = _to_1d_array(reference, "reference")
        r_target = float(ref_arr[-1])

    y0 = float(y[0])
    delta = r_target - y0

    if abs(delta) < 1e-12:
        return 0.0  # Already at reference

    # Threshold values in output units
    y_low = y0 + (low_percent / 100.0) * delta
    y_high = y0 + (high_percent / 100.0) * delta

    def find_crossing(y_target: float) -> Optional[float]:
        """Find exact interpolated crossing time for y_target."""
        for i in range(len(y) - 1):
            y_curr, y_next = y[i], y[i + 1]
            if (y_curr <= y_target <= y_next) or (y_curr >= y_target >= y_next):
                dy = y_next - y_curr
                if abs(dy) < 1e-15:
                    return float(t[i])
                frac = (y_target - y_curr) / dy
                return float(t[i] + frac * (t[i + 1] - t[i]))
        return None

    # For positive step: low threshold crossed first, then high
    # For negative step: low threshold is closer to y0, high is closer to r_target
    t_low = find_crossing(y_low)
    t_high = find_crossing(y_high)

    if t_low is None or t_high is None or not (math.isfinite(t_low) and math.isfinite(t_high)):
        return None

    tr = t_high - t_low
    return max(0.0, float(tr)) if tr >= 0 else None


def calculate_settling_time(
    time: Sequence[float],
    output: Sequence[float],
    reference: Union[float, Sequence[float]],
    tolerance: float = 0.02,
) -> Optional[float]:
    """Calculate settling time (Ts) within a relative tolerance band (+/-2% standard).

    Standard definition: The earliest time after which the output remains within
    [reference * (1 - tolerance), reference * (1 + tolerance)] for all subsequent time.

    Parameters
    ----------
    time : Sequence[float]
        Monotonically increasing time vector.
    output : Sequence[float]
        System output response values y(t).
    reference : float or Sequence[float]
        Setpoint reference value.
    tolerance : float, optional
        Tolerance fraction (default: 0.02 for +/-2%).

    Returns
    -------
    Optional[float]
        Settling time in seconds, or None if response does not settle within the simulation.
    """
    t = _to_1d_array(time, "time")
    y = _to_1d_array(output, "output")

    if len(t) != len(y):
        raise ValueError(f"Length mismatch: time ({len(t)}) vs output ({len(y)})")
    if len(t) < 2:
        return 0.0

    if isinstance(reference, (int, float, np.number)):
        r = np.full_like(y, float(reference))
        r_scale = abs(float(reference))
    else:
        r = _to_1d_array(reference, "reference")
        r_scale = abs(float(r[-1]))

    # Tolerance band width: proportional to reference scale, or absolute tolerance if scale is 0
    band = tolerance * r_scale if r_scale > 1e-12 else tolerance

    # Points outside the tolerance band
    outside_indices = np.where(np.abs(y - r) > band)[0]

    # If it was never outside the band, settled immediately at t=0
    if len(outside_indices) == 0:
        return float(t[0])

    last_outside_idx = int(outside_indices[-1])

    # If the response is still outside the band at the very end of simulation, it never settled
    if last_outside_idx == len(y) - 1:
        return None

    # Settled starting at the sample right after the last violation
    settled_idx = last_outside_idx + 1
    return float(t[settled_idx])


def calculate_overshoot(
    output: Sequence[float],
    reference: Union[float, Sequence[float]],
) -> float:
    """Calculate percentage overshoot (%OS).

    Standard definition:
    For positive step (r > y0): max(0, (max(y) - r) / |r - y0| * 100)
    For negative step (r < y0): max(0, (r - min(y)) / |r - y0| * 100)

    Parameters
    ----------
    output : Sequence[float]
        System output response values y(t).
    reference : float or Sequence[float]
        Setpoint reference value.

    Returns
    -------
    float
        Percentage overshoot in % (>= 0.0).
    """
    y = _to_1d_array(output, "output")
    if not np.all(np.isfinite(y)):
        return float("nan")

    if isinstance(reference, (int, float, np.number)):
        r_target = float(reference)
    else:
        ref_arr = _to_1d_array(reference, "reference")
        r_target = float(ref_arr[-1])

    y0 = float(y[0])
    delta = r_target - y0

    if abs(delta) < 1e-12:
        # Step magnitude is zero: reference is constant at y0
        peak_diff = np.max(np.abs(y - r_target))
        return 0.0 if peak_diff < 1e-12 else float(peak_diff)

    if delta > 0:
        # Rising step: overshoot is peak above reference
        y_max = float(np.max(y))
        overshoot = (y_max - r_target) / abs(delta) * 100.0
    else:
        # Falling step: overshoot is dip below reference
        y_min = float(np.min(y))
        overshoot = (r_target - y_min) / abs(delta) * 100.0

    return max(0.0, float(overshoot))


def calculate_steady_state_error(
    output: Sequence[float],
    reference: Union[float, Sequence[float]],
    tail_fraction: float = 0.05,
) -> float:
    """Calculate final absolute steady-state error (Ess).

    Averages the output over the final tail_fraction of simulation samples to
    suppress minor high-frequency oscillation, then computes |reference - y_ss|.

    Parameters
    ----------
    output : Sequence[float]
        System output response values y(t).
    reference : float or Sequence[float]
        Setpoint reference value.
    tail_fraction : float, optional
        Fraction of samples at the end of the simulation to average (default: 0.05 = 5%).

    Returns
    -------
    float
        Absolute steady-state error (>= 0.0).
    """
    y = _to_1d_array(output, "output")
    if not np.all(np.isfinite(y)):
        return float("nan")

    n_samples = len(y)
    n_tail = max(1, int(tail_fraction * n_samples))
    y_ss = float(np.mean(y[-n_tail:]))

    if isinstance(reference, (int, float, np.number)):
        r_ss = float(reference)
    else:
        ref_arr = _to_1d_array(reference, "reference")
        r_ss = float(np.mean(ref_arr[-n_tail:]))

    return abs(r_ss - y_ss)


def calculate_iae(
    time: Sequence[float],
    output: Sequence[float],
    reference: Union[float, Sequence[float]],
) -> float:
    """Calculate Integral of Absolute Error (IAE).

    IAE = integral from 0 to T of |reference(t) - output(t)| dt
    Evaluated using trapezoidal numerical integration.

    Parameters
    ----------
    time : Sequence[float]
        Time vector.
    output : Sequence[float]
        System output response values y(t).
    reference : float or Sequence[float]
        Setpoint reference value.

    Returns
    -------
    float
        IAE value (>= 0.0).
    """
    t = _to_1d_array(time, "time")
    y = _to_1d_array(output, "output")

    if len(t) != len(y):
        raise ValueError(f"Length mismatch: time ({len(t)}) vs output ({len(y)})")
    if len(t) < 2:
        return 0.0

    if isinstance(reference, (int, float, np.number)):
        r = np.full_like(y, float(reference))
    else:
        r = _to_1d_array(reference, "reference")
        if len(r) != len(y):
            raise ValueError(f"Length mismatch: reference ({len(r)}) vs output ({len(y)})")

    error_abs = np.abs(r - y)
    if not np.all(np.isfinite(error_abs)):
        return float("nan")

    iae_val = float(_trapezoid(error_abs, t))
    return max(0.0, iae_val)


def calculate_all_metrics(
    simulation_result: Union[SimulationResult, Dict[str, Any], Any],
    metric_config: Optional[Dict[str, Any]] = None,
) -> PerformanceMetrics:
    """Compute all five performance metrics from a simulation result.

    Parameters
    ----------
    simulation_result : SimulationResult, dict, or object
        Container holding 'time', 'output', and 'reference' data.
    metric_config : dict, optional
        Optional overrides for calculation settings:
        - 'settling_tolerance': tolerance fraction (default: 0.02)
        - 'rise_low_percent': lower rise threshold (default: 10.0)
        - 'rise_high_percent': upper rise threshold (default: 90.0)
        - 'tail_fraction': steady-state averaging window fraction (default: 0.05)

    Returns
    -------
    PerformanceMetrics
        Dataclass containing:
        - settling_time (Optional[float])
        - rise_time (Optional[float])
        - overshoot (Optional[float])
        - steady_state_error (Optional[float])
        - iae (Optional[float])
    """
    # 1. Unpack simulation result safely
    if isinstance(simulation_result, dict):
        time_data = simulation_result["time"]
        out_data = simulation_result["output"]
        ref_data = simulation_result["reference"]
    elif hasattr(simulation_result, "time") and hasattr(simulation_result, "output"):
        time_data = simulation_result.time
        out_data = simulation_result.output
        ref_data = getattr(simulation_result, "reference", 1.0)
    elif isinstance(simulation_result, (tuple, list)) and len(simulation_result) >= 3:
        time_data, ref_data, out_data = simulation_result[0], simulation_result[1], simulation_result[2]
    else:
        raise TypeError(f"Unsupported simulation_result type: {type(simulation_result)}")

    # 2. Extract configuration options
    cfg = metric_config or {}
    settling_tol = float(cfg.get("settling_tolerance", 0.02))
    rise_low = float(cfg.get("rise_low_percent", 10.0))
    rise_high = float(cfg.get("rise_high_percent", 90.0))
    tail_frac = float(cfg.get("tail_fraction", 0.05))

    # 3. Handle unstable/diverged/NaN outputs gracefully
    y_arr = np.asarray(out_data, dtype=float)
    if y_arr.size == 0 or not np.all(np.isfinite(y_arr)) or np.max(np.abs(y_arr)) > 1e4:
        return PerformanceMetrics(
            settling_time=None,
            rise_time=None,
            overshoot=float("nan"),
            steady_state_error=float("nan"),
            iae=float("nan"),
        )

    # 4. Compute all 5 metrics
    ts = calculate_settling_time(time_data, out_data, ref_data, tolerance=settling_tol)
    tr = calculate_rise_time(time_data, out_data, ref_data, low_percent=rise_low, high_percent=rise_high)
    os = calculate_overshoot(out_data, ref_data)
    ess = calculate_steady_state_error(out_data, ref_data, tail_fraction=tail_frac)
    iae = calculate_iae(time_data, out_data, ref_data)

    return PerformanceMetrics(
        settling_time=ts,
        rise_time=tr,
        overshoot=os,
        steady_state_error=ess,
        iae=iae,
    )
