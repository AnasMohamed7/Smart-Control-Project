"""Genetic Algorithm (GA) – Khaira's implementation.

Implements a real-valued GA from scratch. Works with ANY controller that
exposes the standard BaseController interface (PID, NN, Fuzzy, GP).

Algorithm summary
-----------------
1. Population initialisation – uniform random in parameter bounds.
2. Fitness evaluation        – CountedObjective wraps the objective.
3. Selection                 – Binary tournament (two random individuals,
                               the fitter one survives).
4. Crossover                 – Simulated Binary Crossover (SBX) with
                               distribution index eta_c (default 20).
                               SBX is used in NSGA-II and works well for
                               real-valued parameter optimisation.
5. Mutation                  – Polynomial mutation with distribution index
                               eta_m (default 20). Applied gene-by-gene
                               with probability mutation_rate.
6. Elitism                   – The best elite_count individuals are
                               always copied to the next generation
                               (elite_count = 1 by default).
7. Stopping conditions       – max_iterations (generations) exceeded, OR
                               population fitness std-dev falls below tol
                               (tol=None means run to max_iterations).

All randomness is produced by a *local* numpy Generator seeded from
``random_seed``, never from np.random global state.

References
----------
Deb, K. et al. (2002). A fast and elitist multi-objective genetic
algorithm: NSGA-II. IEEE TEC.
"""
from __future__ import annotations

import math

import numpy as np

from smart_control import config
from smart_control.optimizers.base import BaseOptimizer, CountedObjective


class GeneticAlgorithm(BaseOptimizer):
    """Real-valued Genetic Algorithm using SBX crossover and polynomial mutation."""

    name = "GA"

    def __init__(self, settings=None):
        super().__init__(settings)

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------

    def default_settings(self) -> dict:
        """GA defaults.  Values from config.GA_SETTINGS override these."""
        s = {
            # Population / iteration budget (shared names, same meaning as DE/PSO)
            "population_size": 30,
            "max_iterations":  100,
            "random_seed":     42,

            # Selection: binary tournament is parameter-free and effective.
            # Could be extended to "roulette" or "rank" in the future.
            "selection":       "tournament",
            "tournament_size": 2,          # k = 2 → classic binary tournament

            # Crossover: Simulated Binary Crossover (SBX)
            "crossover_rate":  0.9,        # probability that two parents produce offspring
            "eta_c":           20.0,       # SBX distribution index (higher → closer to parents)

            # Mutation: Polynomial mutation, applied gene-by-gene
            "mutation_rate":   None,       # None → 1/n_params (standard adaptive rate)
            "eta_m":           20.0,       # polynomial mutation distribution index

            # Elitism: how many best individuals are carried over unchanged
            "elite_count":     1,

            # Early stopping: stop if std(fitness) / (|mean(fitness)| + eps) < tol
            # Set to None to always run all generations.
            "tolerance":       None,
        }
        # Allow config.GA_SETTINGS to override defaults without requiring
        # the dict to exist (backward compatible with scaffolded projects).
        s.update(getattr(config, "GA_SETTINGS", {}))
        return s

    # ------------------------------------------------------------------
    # Genetic operators
    # ------------------------------------------------------------------

    @staticmethod
    def _tournament_select(population, fitness, k, rng):
        """Return the index of the best individual from k random contestants.

        Parameters
        ----------
        population : ndarray, shape (pop_size, dim)
        fitness    : ndarray, shape (pop_size,)
        k          : int, tournament size
        rng        : numpy Generator

        Returns
        -------
        int – index of winner in population
        """
        # Choose k distinct random indices
        candidates = rng.choice(len(population), size=k, replace=False)
        # Winner is the one with the lowest (minimised) fitness
        winner = candidates[int(np.argmin(fitness[candidates]))]
        return int(winner)

    @staticmethod
    def _sbx_crossover(parent1, parent2, lower, upper, eta_c, rng):
        """Simulated Binary Crossover (SBX) for real-valued vectors.

        SBX mimics single-point crossover on binary strings but operates
        directly on real values, preserving the spread of solutions.

        Parameters
        ----------
        parent1, parent2 : ndarray, shape (dim,)
        lower, upper     : ndarray, shape (dim,)  – parameter bounds
        eta_c            : float – distribution index (larger = closer to parents)
        rng              : numpy Generator

        Returns
        -------
        child1, child2 : ndarray, shape (dim,) each
        """
        u = rng.random(len(parent1))
        # Beta factor for each gene
        beta = np.where(
            u <= 0.5,
            (2.0 * u) ** (1.0 / (eta_c + 1.0)),
            (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta_c + 1.0)),
        )
        child1 = 0.5 * ((1.0 + beta) * parent1 + (1.0 - beta) * parent2)
        child2 = 0.5 * ((1.0 - beta) * parent1 + (1.0 + beta) * parent2)

        # Clip to bounds
        child1 = np.clip(child1, lower, upper)
        child2 = np.clip(child2, lower, upper)
        return child1, child2

    @staticmethod
    def _polynomial_mutation(individual, lower, upper, eta_m, mutation_rate, rng):
        """Polynomial mutation applied gene-by-gene.

        Each gene is mutated independently with probability mutation_rate.
        The perturbation follows a polynomial probability distribution so
        small changes are more likely than large ones.

        Parameters
        ----------
        individual    : ndarray, shape (dim,)
        lower, upper  : ndarray, shape (dim,)
        eta_m         : float – distribution index (larger = smaller perturbation)
        mutation_rate : float – probability of mutating each gene
        rng           : numpy Generator

        Returns
        -------
        ndarray, shape (dim,) – mutated individual
        """
        child = individual.copy()
        dim   = len(child)
        span  = upper - lower

        for i in range(dim):
            if rng.random() > mutation_rate:
                continue   # gene is not mutated this time

            # Normalised distance to nearer bound in [0, 1]
            delta1 = (child[i] - lower[i]) / max(span[i], 1e-12)
            delta2 = (upper[i] - child[i]) / max(span[i], 1e-12)

            u = rng.random()
            if u <= 0.5:
                # Perturbation towards lower bound
                val = 2.0 * u + (1.0 - 2.0 * u) * (1.0 - delta1) ** (eta_m + 1.0)
                delta_q = val ** (1.0 / (eta_m + 1.0)) - 1.0
            else:
                # Perturbation towards upper bound
                val = 2.0 * (1.0 - u) + (2.0 * u - 1.0) * (1.0 - delta2) ** (eta_m + 1.0)
                delta_q = 1.0 - val ** (1.0 / (eta_m + 1.0))

            child[i] = np.clip(child[i] + delta_q * span[i], lower[i], upper[i])

        return child

    # ------------------------------------------------------------------
    # Main optimise loop
    # ------------------------------------------------------------------

    def optimize(self, objective_function, bounds, settings=None):
        """Run the GA and return an OptimizationResult.

        Parameters
        ----------
        objective_function : callable (x) -> float
            Scalar to minimise; x is a flat list/array of floats.
        bounds : sequence of (lower, upper)
            One pair per parameter.  Taken from controller.parameter_bounds().
        settings : dict, optional
            Run-time overrides for any setting key.

        Returns
        -------
        OptimizationResult
        """
        # --- Merge settings (defaults < constructor < call-time) ---
        s           = self.resolve_settings(settings)
        lo, hi      = self.validate_bounds(bounds)
        lower       = np.array(lo, dtype=float)
        upper       = np.array(hi, dtype=float)
        dim         = len(lower)

        pop_size    = max(int(s["population_size"]), 4)
        max_iter    = int(s["max_iterations"])
        seed        = s["random_seed"]
        eta_c       = float(s["eta_c"])
        eta_m       = float(s["eta_m"])
        cx_rate     = float(s["crossover_rate"])
        elite_count = max(0, int(s["elite_count"]))
        k_tournament= max(2, int(s["tournament_size"]))
        tol         = s["tolerance"]

        # Mutation rate defaults to 1/dim if not specified
        mut_rate = s["mutation_rate"]
        if mut_rate is None:
            mut_rate = 1.0 / dim
        mut_rate = float(mut_rate)

        # Local RNG – no global side effects
        rng = np.random.default_rng(seed)

        # Wrap objective to count calls and handle NaN/inf safely
        f = CountedObjective(objective_function)

        # -----------------------------------------------------------------
        # 1. Population initialisation – uniform random in bounds
        # -----------------------------------------------------------------
        pop = rng.uniform(lower, upper, size=(pop_size, dim))
        fit = np.array([f(ind) for ind in pop], dtype=float)

        # Track overall best-so-far
        best_idx = int(np.argmin(fit))
        best_x   = pop[best_idx].copy()
        best_f   = float(fit[best_idx])

        history        = []
        iterations_run = 0

        # -----------------------------------------------------------------
        # 2. Main generational loop
        # -----------------------------------------------------------------
        for gen in range(max_iter):

            # --- Elitism: identify the elite_count best individuals ---
            # Sort indices by fitness (ascending = best first)
            elite_indices = np.argsort(fit)[:elite_count]
            elites        = pop[elite_indices].copy()
            elite_fits    = fit[elite_indices].copy()

            # --- Build next generation ---
            new_pop  = np.empty_like(pop)
            new_fit  = np.empty_like(fit)
            n_filled = 0

            # Copy elites first
            for j in range(elite_count):
                if n_filled < pop_size:
                    new_pop[n_filled] = elites[j]
                    new_fit[n_filled] = elite_fits[j]
                    n_filled += 1

            # --- Fill remaining slots with selected+varied offspring ---
            while n_filled < pop_size:
                # Selection: pick two parents via tournament
                p1_idx = self._tournament_select(pop, fit, k_tournament, rng)
                p2_idx = self._tournament_select(pop, fit, k_tournament, rng)
                # (Allow p1 == p2 – the crossover will then produce a mutated clone)
                parent1 = pop[p1_idx].copy()
                parent2 = pop[p2_idx].copy()

                # Crossover: apply SBX with probability cx_rate
                if rng.random() < cx_rate:
                    child1, child2 = self._sbx_crossover(
                        parent1, parent2, lower, upper, eta_c, rng
                    )
                else:
                    # No crossover: children are copies of the parents
                    child1, child2 = parent1.copy(), parent2.copy()

                # Mutation: apply polynomial mutation to each child
                child1 = self._polynomial_mutation(child1, lower, upper, eta_m, mut_rate, rng)
                child2 = self._polynomial_mutation(child2, lower, upper, eta_m, mut_rate, rng)

                # Insert first child
                if n_filled < pop_size:
                    new_pop[n_filled] = child1
                    new_fit[n_filled] = f(child1)
                    n_filled += 1

                # Insert second child (if there is still room)
                if n_filled < pop_size:
                    new_pop[n_filled] = child2
                    new_fit[n_filled] = f(child2)
                    n_filled += 1

            # Advance generation
            pop = new_pop
            fit = new_fit

            # Update global best
            gen_best_idx = int(np.argmin(fit))
            if fit[gen_best_idx] < best_f:
                best_f = float(fit[gen_best_idx])
                best_x = pop[gen_best_idx].copy()

            history.append(best_f)
            iterations_run = gen + 1

            # Early stopping: if population has converged, no need to continue
            if tol is not None:
                std_fit  = float(np.std(fit))
                mean_fit = float(np.mean(fit))
                relative = std_fit / (abs(mean_fit) + 1e-12)
                if relative <= float(tol):
                    break

        # -----------------------------------------------------------------
        # 3. Return result in the shared OptimizationResult format
        # -----------------------------------------------------------------
        return self.make_result(
            best_x, best_f, history,
            algorithm="GA",
            iterations_run=iterations_run,
            n_evaluations=f.n_calls,
            random_seed=seed,
            population_size=pop_size,
        )
