"""Simulated Annealing optimizer implemented with NumPy only."""
from __future__ import annotations

import math
from typing import Callable, Sequence

import numpy as np

from smart_control.optimizers.base import BaseOptimizer, CountedObjective
import smart_control.config as config


class SimulatedAnnealing(BaseOptimizer):
    """Single-solution stochastic optimizer for bounded continuous parameters.

    The optimizer follows the common project contract and returns an
    ``OptimizationResult``.  Its history contains the best-so-far fitness after
    each iteration, so it is directly comparable with the other optimizers.
    """

    name = "SimulatedAnnealing"

    def default_settings(self) -> dict:
        settings = dict(getattr(config, "SA_SETTINGS", {}))
        settings.setdefault("max_iterations", 300)
        settings.setdefault("random_seed", 42)
        settings.setdefault("initial_temperature", 1.0)
        settings.setdefault("final_temperature", 1e-3)
        settings.setdefault("cooling_rate", 0.98)
        settings.setdefault("step_size", 0.10)
        settings.setdefault("reheat_interval", 0)
        return settings

    def __init__(self, settings=None) -> None:
        super().__init__(settings=settings)

    @staticmethod
    def _reflect_into_bounds(candidate: np.ndarray, lower: np.ndarray,
                             upper: np.ndarray) -> np.ndarray:
        """Reflect a candidate at bounds rather than accumulating edge clamps."""
        span = upper - lower
        result = candidate.copy()
        fixed = span == 0.0
        result[fixed] = lower[fixed]
        free = ~fixed
        if np.any(free):
            period = 2.0 * span[free]
            offset = np.mod(result[free] - lower[free], period)
            result[free] = lower[free] + np.where(
                offset <= span[free], offset, period - offset
            )
        return result

    def optimize(self, objective_function: Callable[[Sequence[float]], float],
                 bounds, settings=None):
        lower_list, upper_list = self.validate_bounds(bounds)
        lower = np.asarray(lower_list, dtype=float)
        upper = np.asarray(upper_list, dtype=float)
        span = upper - lower

        cfg = self.resolve_settings(settings)
        iterations = int(cfg.get("max_iterations", 300))
        temperature = float(cfg.get("initial_temperature", 1.0))
        final_temperature = float(cfg.get("final_temperature", 1e-3))
        cooling_rate = float(cfg.get("cooling_rate", 0.98))
        step_size = float(cfg.get("step_size", 0.10))
        seed = cfg.get("random_seed", 42)
        reheat_interval = int(cfg.get("reheat_interval", 0))

        if iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not (math.isfinite(temperature) and temperature > 0.0):
            raise ValueError("initial_temperature must be finite and positive")
        if not (math.isfinite(final_temperature) and 0.0 < final_temperature <= temperature):
            raise ValueError("final_temperature must be positive and <= initial_temperature")
        if not (math.isfinite(cooling_rate) and 0.0 < cooling_rate < 1.0):
            raise ValueError("cooling_rate must be between 0 and 1")
        if not (math.isfinite(step_size) and step_size > 0.0):
            raise ValueError("step_size must be finite and positive")
        if reheat_interval < 0:
            raise ValueError("reheat_interval must be non-negative")

        rng = np.random.default_rng(seed)
        objective = CountedObjective(objective_function)

        current = lower + rng.random(len(lower)) * span
        current_fitness = objective(current.tolist())
        best = current.copy()
        best_fitness = current_fitness
        history = []
        temperature_now = temperature
        iterations_run = 0

        for iteration in range(iterations):
            # A Gaussian move scales naturally with each parameter's range.
            proposal = current + rng.normal(0.0, step_size, size=len(current)) * span
            proposal = self._reflect_into_bounds(proposal, lower, upper)
            proposal_fitness = objective(proposal.tolist())

            delta = proposal_fitness - current_fitness
            accept = delta <= 0.0
            if not accept and math.isfinite(delta):
                exponent = -delta / max(temperature_now, np.finfo(float).tiny)
                # Avoid needless underflow while preserving the Metropolis rule.
                accept = rng.random() < (math.exp(exponent) if exponent > -745.0 else 0.0)

            if accept:
                current = proposal
                current_fitness = proposal_fitness

            if current_fitness < best_fitness:
                best = current.copy()
                best_fitness = current_fitness

            history.append(float(best_fitness))
            iterations_run = iteration + 1
            temperature_now *= cooling_rate
            if reheat_interval and iterations_run % reheat_interval == 0:
                temperature_now = max(temperature_now, temperature * 0.5)
            if temperature_now < final_temperature:
                break

        return self.make_result(
            best_solution=best.tolist(),
            best_fitness=best_fitness,
            convergence_history=history,
            algorithm=self.name,
            iterations_run=iterations_run,
            n_evaluations=objective.n_calls,
            random_seed=seed,
        )
