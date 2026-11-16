"""Differential Evolution (DE), written from scratch.

Default strategy is DE/rand/1/bin:
    mutant  = x_r1 + F * (x_r2 - x_r3)
    trial   = binomial crossover of target and mutant (rate CR)
    keep trial if its fitness <= target fitness (greedy selection)
Bounds are handled by reflecting values back inside the box.
"""
import numpy as np

from smart_control import config
from smart_control.optimizers.base import BaseOptimizer, CountedObjective


class DifferentialEvolution(BaseOptimizer):
    name = "DE"

    def __init__(self, settings=None):
        super().__init__(settings)

    def default_settings(self):
        s = {
            "population_size": 30,
            "max_iterations": 100,
            "random_seed": 42,
            "mutation_factor": 0.6,
            "crossover_rate": 0.9,
            "strategy": "rand1",
            "tolerance": None,
        }
        s.update(getattr(config, "DE_SETTINGS", {}))
        return s

    def _reflect(self, x, lower, upper, rng):
        span = upper - lower
        x = np.where(x < lower, lower + (lower - x), x)
        x = np.where(x > upper, upper - (x - upper), x)
        bad = (x < lower) | (x > upper)
        if np.any(bad):
            x = np.where(bad, rng.uniform(lower, upper), x)
        return np.clip(x, lower, upper)

    def optimize(self, objective_function, bounds, settings=None):
        s = self.resolve_settings(settings)
        lo, hi = self.validate_bounds(bounds)
        lower = np.array(lo)
        upper = np.array(hi)
        dim = len(lower)

        pop_size = max(int(s["population_size"]), 4)
        max_iter = int(s["max_iterations"])
        F = float(s["mutation_factor"])
        CR = float(s["crossover_rate"])
        strategy = s["strategy"]
        tol = s["tolerance"]
        seed = s["random_seed"]

        if strategy not in ("rand1", "best1"):
            raise ValueError("strategy must be 'rand1' or 'best1'")

        rng = np.random.default_rng(seed)
        f = CountedObjective(objective_function)

        pop = rng.uniform(lower, upper, size=(pop_size, dim))
        fit = np.array([f(p) for p in pop])

        best_idx = int(np.argmin(fit))
        best_x = pop[best_idx].copy()
        best_f = float(fit[best_idx])

        history = []
        iterations_run = 0

        for it in range(max_iter):
            for i in range(pop_size):
                others = [j for j in range(pop_size) if j != i]
                if strategy == "best1":
                    r1, r2 = rng.choice(others, 2, replace=False)
                    base = pop[best_idx]
                    diff = pop[r1] - pop[r2]
                else:
                    r1, r2, r3 = rng.choice(others, 3, replace=False)
                    base = pop[r1]
                    diff = pop[r2] - pop[r3]

                mutant = self._reflect(base + F * diff, lower, upper, rng)

                cross = rng.random(dim) < CR
                cross[rng.integers(dim)] = True
                trial = np.where(cross, mutant, pop[i])

                trial_f = f(trial)
                if trial_f <= fit[i]:
                    pop[i] = trial
                    fit[i] = trial_f

            best_idx = int(np.argmin(fit))
            if fit[best_idx] < best_f:
                best_f = float(fit[best_idx])
                best_x = pop[best_idx].copy()

            history.append(best_f)
            iterations_run = it + 1

            if tol is not None and np.std(fit) <= float(tol) * (abs(np.mean(fit)) + 1e-12):
                break

        return self.make_result(
            best_x, best_f, history,
            algorithm="DE",
            iterations_run=iterations_run,
            n_evaluations=f.n_calls,
            random_seed=seed,
        )
