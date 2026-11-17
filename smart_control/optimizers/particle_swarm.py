"""Particle Swarm Optimization (PSO), implemented from scratch.

Uses the standard velocity/position update with a linearly decreasing inertia
weight, cognitive and social terms, and velocity clamping. It follows the same
BaseOptimizer contract as GA, DE, SA, and GP.
"""
from __future__ import annotations

import math

import numpy as np

from smart_control.optimizers.base import BaseOptimizer, CountedObjective
from smart_control.types import OptimizationResult


_DEFAULT_PSO_SETTINGS = {
    "population_size": 30,
    "max_iterations": 100,
    "inertia_weight": 0.7,
    "cognitive_coeff": 1.5,
    "social_coeff": 1.5,
    "v_max_ratio": 0.2,
    "random_seed": 42,
}


def _load_cfg_defaults() -> dict:
    try:
        import smart_control.config as cfg
        return dict(getattr(cfg, "PSO_SETTINGS", _DEFAULT_PSO_SETTINGS))
    except Exception:  # pragma: no cover - config is part of the project
        return dict(_DEFAULT_PSO_SETTINGS)


class ParticleSwarmOptimization(BaseOptimizer):
    """Minimize a bounded scalar objective using particle swarm optimization."""

    name = "PSO"

    def __init__(self, settings: dict | None = None) -> None:
        super().__init__(settings=settings)

    def default_settings(self) -> dict:
        settings = dict(_DEFAULT_PSO_SETTINGS)
        settings.update(_load_cfg_defaults())
        return settings

    def optimize(self, objective_function, bounds, settings: dict | None = None) -> OptimizationResult:
        merged = self.resolve_settings(settings)
        lower_list, upper_list = self.validate_bounds(bounds)
        lower = np.asarray(lower_list, dtype=float)
        upper = np.asarray(upper_list, dtype=float)
        span = upper - lower

        n_particles = int(merged["population_size"])
        max_iterations = int(merged["max_iterations"])
        w_max = float(merged["inertia_weight"])
        c1 = float(merged["cognitive_coeff"])
        c2 = float(merged["social_coeff"])
        v_max_ratio = float(merged["v_max_ratio"])
        seed = merged.get("random_seed", 42)

        if n_particles < 2:
            raise ValueError("population_size must be at least 2")
        if max_iterations < 0:
            raise ValueError("max_iterations must be non-negative")
        if not all(math.isfinite(v) for v in (w_max, c1, c2, v_max_ratio)):
            raise ValueError("PSO coefficients must be finite")
        if v_max_ratio < 0:
            raise ValueError("v_max_ratio must be non-negative")

        # A local generator avoids changing NumPy's global random state.
        rng = np.random.default_rng(seed)
        objective = CountedObjective(objective_function)
        dimensions = len(lower)
        v_max = v_max_ratio * span

        positions = lower + rng.random((n_particles, dimensions)) * span
        velocities = rng.uniform(-1.0, 1.0, size=(n_particles, dimensions)) * v_max
        # Fixed dimensions (lower == upper) remain fixed with zero velocity.
        velocities[:, span == 0.0] = 0.0

        fitness = np.asarray([objective(row) for row in positions], dtype=float)
        personal_best_positions = positions.copy()
        personal_best_fitness = fitness.copy()
        best_index = int(np.argmin(personal_best_fitness))
        global_best_position = personal_best_positions[best_index].copy()
        global_best_fitness = float(personal_best_fitness[best_index])

        history: list[float] = []
        for iteration in range(max_iterations):
            # Decay from w_max to 0.4 over the configured iterations.
            fraction = iteration / max(1, max_iterations - 1)
            inertia = w_max + (0.4 - w_max) * fraction

            r1 = rng.random((n_particles, dimensions))
            r2 = rng.random((n_particles, dimensions))
            cognitive = c1 * r1 * (personal_best_positions - positions)
            social = c2 * r2 * (global_best_position - positions)
            velocities = inertia * velocities + cognitive + social
            velocities = np.clip(velocities, -v_max, v_max)
            positions = np.clip(positions + velocities, lower, upper)
            positions[:, span == 0.0] = lower[span == 0.0]

            fitness = np.asarray([objective(row) for row in positions], dtype=float)
            improved = fitness < personal_best_fitness
            personal_best_positions[improved] = positions[improved]
            personal_best_fitness[improved] = fitness[improved]

            best_index = int(np.argmin(personal_best_fitness))
            if personal_best_fitness[best_index] < global_best_fitness:
                global_best_fitness = float(personal_best_fitness[best_index])
                global_best_position = personal_best_positions[best_index].copy()
            history.append(global_best_fitness)

        return self.make_result(
            global_best_position,
            global_best_fitness,
            history,
            algorithm=self.name,
            iterations_run=len(history),
            n_evaluations=objective.n_calls,
            population_size=n_particles,
            random_seed=seed,
        )


if __name__ == "__main__":
    def sphere(x):
        return float(np.sum(np.asarray(x, dtype=float) ** 2))

    result = ParticleSwarmOptimization().optimize(sphere, [(-5, 5)] * 5)
    print("Best solution:", result.best_solution)
    print("Best fitness:", result.best_fitness)
    print("Evaluations:", result.metadata["n_evaluations"])
