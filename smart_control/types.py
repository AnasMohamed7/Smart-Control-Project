"""Shared type aliases and lightweight data containers.

These structures define the expected data exchanged by project modules.
They intentionally contain no control or optimization implementation.
"""
from __future__ import annotations

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
    control_signal: Any = None


@dataclass
class PerformanceMetrics:
    """Required performance measures for one simulation."""
    settling_time: Optional[float]
    rise_time: Optional[float]
    overshoot: Optional[float]
    steady_state_error: Optional[float]
    iae: Optional[float]

    def to_dict(self) -> Dict[str, Optional[float]]:
        """Convert metrics to a standard dictionary."""
        return {
            "settling_time": self.settling_time,
            "rise_time": self.rise_time,
            "overshoot": self.overshoot,
            "steady_state_error": self.steady_state_error,
            "iae": self.iae,
        }

    def __getitem__(self, key: str) -> Optional[float]:
        """Support dict-like indexing (e.g. metrics['iae'])."""
        if hasattr(self, key):
            return getattr(self, key)
        raise KeyError(f"Metric '{key}' not found in PerformanceMetrics")


@dataclass
class OptimizationResult:
    """Common return format for every optimizer."""
    best_solution: Vector
    best_fitness: float
    convergence_history: List[float]
    metadata: Dict[str, Any]
