"""Mamdani fuzzy-logic controller for a single-input single-output plant.

Inputs are normalized tracking error and error rate.  Five triangular
linguistic sets (NB, NS, ZE, PS, PB) are used for each input and output,
with a 25-rule PD-like rule base and centroid defuzzification.  The optimizer
adjusts three positive scaling gains: [K_e, K_de, K_u].
"""
from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from smart_control.controllers.base import BaseController


class FuzzyController(BaseController):
    """Two-input Mamdani controller with tunable input/output scaling gains."""

    _CENTERS = np.asarray([-1.0, -0.5, 0.0, 0.5, 1.0], dtype=float)
    _LABELS = ("NB", "NS", "ZE", "PS", "PB")
    # Rows correspond to error; columns correspond to error rate.
    _RULES = np.asarray([
        [0, 0, 0, 1, 2],
        [0, 0, 1, 2, 3],
        [0, 1, 2, 3, 4],
        [1, 2, 3, 4, 4],
        [2, 3, 4, 4, 4],
    ], dtype=int)

    def __init__(self, design_config=None):
        cfg = dict(design_config or {})
        self.K_e = float(cfg.get("K_e", 2.0))
        self.K_de = float(cfg.get("K_de", 0.1))
        self.K_u = float(cfg.get("K_u", 10.0))
        self.error_rate_limit = float(cfg.get("error_rate_limit", 100.0))
        self._output_grid = np.linspace(-1.0, 1.0, 201)
        self._output_memberships = self._membership_matrix(self._output_grid)
        self.reset()

    @classmethod
    def _membership_matrix(cls, values):
        """Return memberships for NB/NS/ZE/PS/PB triangular sets."""
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        centers = cls._CENTERS.reshape(1, -1)
        # Adjacent centers are 0.5 apart; shoulder sets reach full membership
        # at the universe endpoints.
        return np.clip(1.0 - np.abs(values - centers) / 0.5, 0.0, 1.0)

    @classmethod
    def _fuzzify(cls, value):
        value = float(np.clip(value, -1.0, 1.0))
        return cls._membership_matrix([value])[0]

    def reset(self):
        """Clear dynamic state without changing the optimized gains."""
        self.previous_error = 0.0
        self._has_previous_error = False

    def set_parameters(self, parameters: Sequence[float]):
        values = np.asarray(parameters, dtype=float).reshape(-1)
        if values.size != 3:
            raise ValueError("FuzzyController expects [K_e, K_de, K_u]")
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0):
            raise ValueError("Fuzzy scaling gains must be finite and positive")
        self.K_e, self.K_de, self.K_u = map(float, values)

    def compute(self, error, dt, **signals):
        error = float(error)
        dt = float(dt)
        if not math.isfinite(error):
            return 0.0
        if not math.isfinite(dt) or dt <= 0.0:
            raise ValueError("dt must be finite and positive")

        if "error_rate" in signals:
            error_rate = float(signals["error_rate"])
        elif "derivative" in signals:
            error_rate = float(signals["derivative"])
        elif self._has_previous_error:
            error_rate = (error - self.previous_error) / dt
        else:
            error_rate = 0.0
        self.previous_error = error
        self._has_previous_error = True
        if not math.isfinite(error_rate):
            error_rate = 0.0

        e_norm = float(np.clip(self.K_e * error, -1.0, 1.0))
        de_norm = float(np.clip(self.K_de * error_rate,
                                -self.error_rate_limit, self.error_rate_limit))
        de_norm = float(np.clip(de_norm, -1.0, 1.0))
        e_membership = self._fuzzify(e_norm)
        de_membership = self._fuzzify(de_norm)

        # Mamdani min implication and max aggregation across all 25 rules.
        aggregated = np.zeros_like(self._output_grid)
        for i in range(5):
            for j in range(5):
                firing_strength = min(e_membership[i], de_membership[j])
                if firing_strength <= 0.0:
                    continue
                consequent = self._output_memberships[:, self._RULES[i, j]]
                aggregated = np.maximum(aggregated,
                                         np.minimum(firing_strength, consequent))

        integrate = getattr(np, "trapezoid", None)
        if integrate is None:  # NumPy < 2.0 compatibility
            integrate = np.trapz
        area = float(integrate(aggregated, self._output_grid))
        if area <= np.finfo(float).eps:
            normalized_output = 0.0
        else:
            moment = float(integrate(self._output_grid * aggregated,
                                     self._output_grid))
            normalized_output = moment / area
        return float(self.K_u * normalized_output)

    def parameter_bounds(self):
        """Bounds for the three gains optimized by GA/DE/PSO/SA."""
        return [(0.1, 20.0), (0.001, 5.0), (0.1, 50.0)]
