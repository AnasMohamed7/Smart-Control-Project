"""Particle Swarm Optimization (PSO) implementation.

Implements the standard velocity-position PSO with:
- Linearly decaying inertia weight (w_max → 0.4) for better convergence
- Cognitive (personal best) and social (global best) acceleration terms
- Velocity clamping and position clipping to enforce bounds

Reference:
    Kennedy, J., & Eberhart, R. (1995). Particle swarm optimization.
    Proceedings of ICNN'95 - International Conference on Neural Networks.
"""

import numpy as np

from smart_control.optimizers.base import BaseOptimizer
from smart_control.types import OptimizationResult

# ---------------------------------------------------------------------------
# Default PSO hyper-parameters (used when cfg.PSO_SETTINGS is unavailable)
# ---------------------------------------------------------------------------
_DEFAULT_PSO_SETTINGS = {
    "population_size": 30,
    "max_iterations": 100,
    "inertia_weight": 0.7,   # w_max; decays linearly to 0.4
    "cognitive_coeff": 1.5,  # c1
    "social_coeff": 1.5,     # c2
    "v_max_ratio": 0.2,      # v_max = v_max_ratio * (upper - lower)
    "random_seed": 42,
}


def _load_cfg_defaults() -> dict:
    """Return PSO_SETTINGS from config, falling back to built-in defaults."""
    try:
        import smart_control.config as cfg  # noqa: PLC0415
        return dict(getattr(cfg, "PSO_SETTINGS", _DEFAULT_PSO_SETTINGS))
    except Exception:  # pragma: no cover
        return dict(_DEFAULT_PSO_SETTINGS)


class ParticleSwarmOptimization(BaseOptimizer):
    """Particle Swarm Optimization minimiser.

    Parameters
    ----------
    settings : dict, optional
        Override specific PSO hyper-parameters. Any key present here takes
        precedence over the project-wide ``cfg.PSO_SETTINGS`` defaults.
        Recognised keys: population_size, max_iterations, inertia_weight,
        cognitive_coeff, social_coeff, v_max_ratio, random_seed.
    """

    def __init__(self, settings: dict | None = None) -> None:
        self._instance_settings: dict = dict(settings) if settings else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def optimize(
        self,
        objective_function,
        bounds,
        settings: dict | None = None,
    ) -> OptimizationResult:
        """Run PSO and return the best solution found.

        Parameters
        ----------
        objective_function : callable
            Scalar function to *minimise*; signature ``f(x) -> float``
            where *x* is a sequence of floats.
        bounds : Sequence[Tuple[float, float]]
            Per-dimension ``(lower, upper)`` box constraints.
        settings : dict, optional
            Run-time overrides. Merged on top of instance-level settings,
            which themselves override ``cfg.PSO_SETTINGS`` defaults.

        Returns
        -------
        OptimizationResult
        """
        # ----------------------------------------------------------------
        # 1. Merge settings: cfg defaults ← instance settings ← call settings
        # ----------------------------------------------------------------
        merged = _load_cfg_defaults()
        merged.update(self._instance_settings)
        if settings:
            merged.update(settings)

        population_size  = int(merged["population_size"])
        max_iterations   = int(merged["max_iterations"])
        w_max            = float(merged["inertia_weight"])
        c1               = float(merged["cognitive_coeff"])
        c2               = float(merged["social_coeff"])
        v_max_ratio      = float(merged["v_max_ratio"])
        random_seed      = merged["random_seed"]

        # ----------------------------------------------------------------
        # 2. Initialise RNG and problem geometry
        # ----------------------------------------------------------------
        np.random.seed(random_seed)

        n_particles = population_size
        n_dims      = len(bounds)

        lower = np.array([b[0] for b in bounds], dtype=float)
        upper = np.array([b[1] for b in bounds], dtype=float)

        v_max = v_max_ratio * (upper - lower)   # shape (n_dims,)

        # ----------------------------------------------------------------
        # 3. Initialise swarm
        # ----------------------------------------------------------------
        positions  = np.random.uniform(lower, upper, (n_particles, n_dims))
        velocities = np.random.uniform(-v_max, v_max, (n_particles, n_dims))

        # Evaluate initial fitness
        fitness = np.array(
            [float(objective_function(positions[i])) for i in range(n_particles)]
        )

        pbest_positions = positions.copy()
        pbest_fitness   = fitness.copy()

        best_idx       = int(np.argmin(pbest_fitness))
        gbest_position = pbest_positions[best_idx].copy()
        gbest_fitness  = float(pbest_fitness[best_idx])

        convergence_history: list[float] = []

        # ----------------------------------------------------------------
        # 4. Main PSO loop
        # ----------------------------------------------------------------
        for iteration in range(max_iterations):
            # Linearly decaying inertia: w_max → 0.4
            w = w_max - (w_max - 0.4) * iteration / max_iterations

            # Sample stochastic coefficients once per iteration (vectorised)
            r1 = np.random.uniform(0.0, 1.0, (n_particles, n_dims))
            r2 = np.random.uniform(0.0, 1.0, (n_particles, n_dims))

            # --- Velocity update (PSO equation) ---
            cognitive = c1 * r1 * (pbest_positions - positions)
            social    = c2 * r2 * (gbest_position  - positions)
            velocities = w * velocities + cognitive + social

            # Clamp velocity
            velocities = np.clip(velocities, -v_max, v_max)

            # --- Position update ---
            positions = positions + velocities

            # Clip positions to bounds
            positions = np.clip(positions, lower, upper)

            # --- Evaluate fitness ---
            fitness = np.array(
                [float(objective_function(positions[i])) for i in range(n_particles)]
            )

            # --- Update personal bests ---
            improved = fitness < pbest_fitness
            pbest_positions[improved] = positions[improved]
            pbest_fitness[improved]   = fitness[improved]

            # --- Update global best ---
            best_idx = int(np.argmin(pbest_fitness))
            if pbest_fitness[best_idx] < gbest_fitness:
                gbest_fitness  = float(pbest_fitness[best_idx])
                gbest_position = pbest_positions[best_idx].copy()

            convergence_history.append(gbest_fitness)

        # ----------------------------------------------------------------
        # 5. Return result
        # ----------------------------------------------------------------
        return OptimizationResult(
            best_solution=gbest_position.tolist(),
            best_fitness=float(gbest_fitness),
            convergence_history=convergence_history,
            metadata={
                "iterations_run": max_iterations,
                "n_particles": n_particles,
                "algorithm": "PSO",
            },
        )


# ---------------------------------------------------------------------------
# Standalone smoke-test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    def sphere(x):
        """Sphere function: global minimum 0 at origin."""
        return sum(xi ** 2 for xi in x)

    bounds = [(-5, 5)] * 5  # 5-dimensional
    pso = ParticleSwarmOptimization()
    result = pso.optimize(sphere, bounds)
    print(f"Best solution: {result.best_solution}")
    print(f"Best fitness: {result.best_fitness:.6f}")   # should be close to 0
    print(f"Convergence history length: {len(result.convergence_history)}")
