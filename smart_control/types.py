"""Shared type aliases and lightweight data containers.

These structures define the expected data exchanged by project modules.
They intentionally contain no control or optimization implementation.
"""
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

Vector = Sequence[float]
Bounds = Sequence[Tuple[float, float]]
ObjectiveFunction = Callable[[Vector], float]


@dataclass
class SimulationResult:
    """Output from one closed-loop simulation."""
    time: Any
    reference: Any
    output: Any
    


@dataclass
class PerformanceMetrics:
    """Required performance measures for one simulation."""
    settling_time: Optional[float]
    rise_time: Optional[float]
    overshoot: Optional[float]
    steady_state_error: Optional[float]
    iae: Optional[float]


@dataclass
class OptimizationResult:
    """Common return format for every optimizer."""
    best_solution: Vector
    best_fitness: float
    convergence_history: List[float]
    metadata: Dict[str, Any]
