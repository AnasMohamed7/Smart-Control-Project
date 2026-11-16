"""Common optimizer contract.

The optimizer receives an objective function and parameter bounds, then returns
an OptimizationResult. Each algorithm must be implemented from scratch.

Contract (every optimizer: GA, DE, PSO, SA, GP)
-----------------------------------------------
    optimizer = SomeOptimizer(settings=None)          # settings: dict, optional
    result = optimizer.optimize(objective_function, bounds, settings=None)

* ``objective_function(x) -> float``  -- *minimized*; ``x`` is a flat sequence of
  floats, one value per bound.  The optimizer never needs to know which
  controller or plant is behind it.
* ``bounds`` -- sequence of ``(lower, upper)``, one per dimension.  Always
  taken from ``controller.parameter_bounds()``.
* ``settings`` -- dict of run-time overrides.  Precedence (low -> high):
  ``default_settings()`` / ``config.<ALGO>_SETTINGS``  <  constructor settings
  <  ``optimize(..., settings=...)``.
* Shared setting names (same meaning for every optimizer, so budgets are
  comparable): ``population_size``, ``max_iterations``, ``random_seed``.
  Single-solution methods (SA) ignore ``population_size``.
* Returns :class:`smart_control.types.OptimizationResult`:
    - ``best_solution``       list[float], same order/length as ``bounds``
    - ``best_fitness``        float, ``objective_function(best_solution)``
    - ``convergence_history`` list[float], best-so-far fitness after each
      iteration (non-increasing; ``max_iterations`` entries unless the run
      stopped early)
    - ``metadata``            dict; recommended keys: ``algorithm``,
      ``iterations_run``, ``n_evaluations``, ``random_seed``.
* Randomness should come from a *local* generator seeded with ``random_seed``
  (no global ``np.random.seed`` side effects) so runs are reproducible and
  one optimizer cannot disturb another.
* Invalid (NaN/inf) objective values must not break the search; use
  :meth:`BaseOptimizer.safe_fitness` / :class:`CountedObjective`.

Optional helpers below (``validate_bounds``, ``safe_fitness``, ``make_result``,
``CountedObjective``) are available to every optimizer owner; nothing forces
existing implementations (e.g. PSO) to change.
"""
from __future__ import annotations

import math
from typing import Any, Callable, List, Optional, Sequence, Tuple

from smart_control.types import OptimizationResult

# Finite stand-in for NaN / +-inf objective values (keeps comparisons and means valid).
PENALTY_FITNESS = 1e10


class BaseOptimizer:
    """Base class: settings handling + small helpers shared by all optimizers."""

    name = "BaseOptimizer"

    def __init__(self, settings: Optional[dict] = None) -> None:
        self._instance_settings: dict = dict(settings) if settings else {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def optimize(self, objective_function, bounds, settings=None) -> OptimizationResult:
        """Search for a candidate minimizing objective_function.

        Expected return: OptimizationResult.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Settings
    # ------------------------------------------------------------------
    def default_settings(self) -> dict:
        """Algorithm defaults. Subclasses override (e.g. read ``config.GA_SETTINGS``)."""
        return {}

    def resolve_settings(self, call_settings: Optional[dict] = None) -> dict:
        """defaults < constructor settings < call settings."""
        merged = dict(self.default_settings())
        merged.update(getattr(self, "_instance_settings", None) or {})
        if call_settings:
            merged.update(call_settings)
        return merged

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def validate_bounds(bounds) -> Tuple[List[float], List[float]]:
        """Return (lower, upper) lists; raise ValueError for empty/invalid bounds."""
        if bounds is None or len(bounds) == 0:
            raise ValueError("bounds must contain at least one (lower, upper) pair")
        lower: List[float] = []
        upper: List[float] = []
        for i, pair in enumerate(bounds):
            if len(pair) != 2:
                raise ValueError(f"bounds[{i}] must be a (lower, upper) pair")
            lo, hi = float(pair[0]), float(pair[1])
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"bounds[{i}] must be finite")
            if lo > hi:
                raise ValueError(f"bounds[{i}] has lower > upper: ({lo}, {hi})")
            lower.append(lo)
            upper.append(hi)
        return lower, upper

    @staticmethod
    def safe_fitness(value: Any, penalty: float = PENALTY_FITNESS) -> float:
        """float(value), or ``penalty`` if it is NaN/inf/not convertible."""
        try:
            value = float(value)
        except (TypeError, ValueError):
            return penalty
        return value if math.isfinite(value) else penalty

    @staticmethod
    def make_result(best_solution: Sequence[float], best_fitness: float,
                    convergence_history: Sequence[float], **metadata: Any) -> OptimizationResult:
        """Build an OptimizationResult with plain-Python types."""
        return OptimizationResult(
            best_solution=[float(v) for v in best_solution],
            best_fitness=float(best_fitness),
            convergence_history=[float(v) for v in convergence_history],
            metadata=dict(metadata),
        )


class CountedObjective:
    """Wrap an objective: counts calls and maps invalid values to a finite penalty."""

    def __init__(self, objective_function: Callable[[Sequence[float]], float],
                 penalty: float = PENALTY_FITNESS) -> None:
        self.objective_function = objective_function
        self.penalty = penalty
        self.n_calls = 0

    def __call__(self, x: Sequence[float]) -> float:
        self.n_calls += 1
        return BaseOptimizer.safe_fitness(self.objective_function(x), self.penalty)
