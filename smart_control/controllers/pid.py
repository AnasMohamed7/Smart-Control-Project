"""PID Controller – Khaira's implementation.

Candidate parameter vector: [Kp, Ki, Kd]

Design decisions (from design_decisions.md / team agreement):
  - Derivative is computed from finite difference: (e[k] - e[k-1]) / dt.
    On the very first call after reset(), de/dt = 0 (no previous error yet).
  - Integral accumulation: integral += e * dt each step.
  - Anti-windup: the integral is clamped to ± INTEGRAL_LIMIT so the
    controller cannot wind up during large transients.
  - Output saturation: control signal clipped to ± U_LIMIT (matches the
    simulator's own clamp in plants/simulation.py, so no double clipping).
  - Derivative filtering: a first-order low-pass filter with time constant
    TAU_F smooths the numerical derivative. TAU_F = 0 → no filter
    (pure backward difference). Default 0.0 → compatible with all plants.
  - All these defaults can be overridden at construction time via design_config.
"""
from __future__ import annotations

import math
from typing import Sequence

from smart_control.controllers.base import BaseController

# Default limits (can be overridden in __init__ via design_config)
_DEFAULT_INTEGRAL_LIMIT = 50.0   # anti-windup clamp  |∫e dt| ≤ this value
_DEFAULT_U_LIMIT        = 50.0   # output saturation   |u|      ≤ this value
_DEFAULT_TAU_F          = 0.0    # derivative filter time constant (s), 0 = off


class PIDController(BaseController):
    """Proportional-Integral-Derivative controller.

    Parameters
    ----------
    design_config : dict, optional
        Construction-time overrides for:
          Kp, Ki, Kd            - initial gains (defaults: 1.0, 0.0, 0.0)
          integral_limit        - anti-windup bound   (default: 50.0)
          u_limit               - output clamp        (default: 50.0)
          tau_f                 - derivative filter t (default: 0.0, off)
    """

    def __init__(self, design_config=None):
        cfg = dict(design_config or {})

        # Tunable gains - set here as defaults; overwritten by set_parameters()
        self.Kp = float(cfg.get("Kp", 1.0))
        self.Ki = float(cfg.get("Ki", 0.0))
        self.Kd = float(cfg.get("Kd", 0.0))

        # Fixed design parameters
        self._integral_limit = float(cfg.get("integral_limit", _DEFAULT_INTEGRAL_LIMIT))
        self._u_limit        = float(cfg.get("u_limit",        _DEFAULT_U_LIMIT))
        self._tau_f          = float(cfg.get("tau_f",          _DEFAULT_TAU_F))

        # Dynamic state (initialised properly by reset())
        self.reset()

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self):
        """Clear integral accumulator, previous error, and filtered derivative.

        Called by the simulator at the start of every new simulation run so
        the controller starts from a clean state. Never clears the gains.
        """
        self._integral   = 0.0
        self._prev_error = 0.0
        self._deriv_filt = 0.0   # low-pass filtered derivative value
        self._has_prev   = False  # True after the first compute() call

    def set_parameters(self, parameters):
        """Load [Kp, Ki, Kd] from an optimizer candidate vector.

        Parameters
        ----------
        parameters : sequence of float, length 3
            [Kp, Ki, Kd]  - must all be finite.

        Raises
        ------
        ValueError
            If the vector is not length-3 or contains non-finite values.
        """
        params = [float(v) for v in parameters]
        if len(params) != 3:
            raise ValueError(
                "PIDController expects 3 parameters [Kp, Ki, Kd], "
                f"got {len(params)}"
            )
        if not all(math.isfinite(v) for v in params):
            raise ValueError("PIDController parameters must all be finite")

        self.Kp, self.Ki, self.Kd = params

    def compute(self, error, dt, **signals):
        """Compute the PID control signal for the current time step.

        u(t) = Kp*e(t)  +  Ki*integral(e dt)  +  Kd*(de/dt)

        Parameters
        ----------
        error : float
            Tracking error  e = r - y  at the current step.
        dt : float
            Time step in seconds.  Must be finite and > 0.
        **signals : dict
            Optional extra signals. Recognised keys:
              ``derivative`` - pre-computed de/dt; skips finite-difference
                               and filter logic when provided.

        Returns
        -------
        float
            Saturated control signal u, clipped to +/-u_limit.
        """
        error = float(error)
        dt    = float(dt)

        if not math.isfinite(error):
            return 0.0
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be a finite positive number")

        # --- Integral (with anti-windup clamp) ---
        self._integral += error * dt
        self._integral = max(-self._integral_limit,
                             min(self._integral_limit, self._integral))

        # --- Derivative ---
        if "derivative" in signals:
            # Caller pre-computed de/dt (e.g. from a sensor)
            derivative = float(signals["derivative"])
        elif not self._has_prev:
            # Very first call: no previous sample, so derivative = 0
            derivative = 0.0
        else:
            # Backward finite difference: (e[k] - e[k-1]) / dt
            raw_deriv = (error - self._prev_error) / dt

            if self._tau_f > 0.0:
                # First-order low-pass filter (Tustin/bilinear discretisation):
                #   d[k] = alpha * d[k-1] + (1-alpha) * raw_deriv
                #   where alpha = tau_f / (tau_f + dt)
                # This smooths high-frequency noise in the derivative channel.
                alpha = self._tau_f / (self._tau_f + dt)
                self._deriv_filt = (alpha * self._deriv_filt
                                    + (1.0 - alpha) * raw_deriv)
                derivative = self._deriv_filt
            else:
                derivative = raw_deriv

        # Store current error for next step and mark initialised
        self._prev_error = error
        self._has_prev   = True

        # --- PID sum ---
        u = self.Kp * error + self.Ki * self._integral + self.Kd * derivative

        # --- Output saturation ---
        return max(-self._u_limit, min(self._u_limit, u))

    def parameter_bounds(self):
        """Return optimizer search bounds for [Kp, Ki, Kd].

        Bounds are intentionally generous so all three plant orders can be
        tuned from the same search space.  An optimizer that finds gains
        outside these ranges would already produce output > u_limit anyway.

        Returns
        -------
        list of (lower, upper) pairs, length 3.
        """
        return [
            (0.0, 50.0),   # Kp - proportional gain
            (0.0, 20.0),   # Ki - integral gain
            (0.0, 10.0),   # Kd - derivative gain
        ]
