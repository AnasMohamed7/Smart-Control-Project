"""Focused tests for Fares' SA, fuzzy-controller, and plotting modules."""
import numpy as np
import pytest

from smart_control.controllers.fuzzy import FuzzyController
from smart_control.optimizers.simulated_annealing import SimulatedAnnealing
from smart_control.types import OptimizationResult, PerformanceMetrics, SimulationResult
from smart_control.visualization.plots import (
    plot_convergence_curves,
    plot_metric_comparison,
    plot_step_responses,
)


def test_sa_minimizes_bounded_quadratic_reproducibly():
    objective = lambda x: float(np.sum(np.asarray(x, dtype=float) ** 2))
    settings = {"max_iterations": 100, "random_seed": 17}
    first = SimulatedAnnealing(settings).optimize(objective, [(-5, 5), (-2, 2)])
    second = SimulatedAnnealing(settings).optimize(objective, [(-5, 5), (-2, 2)])
    assert isinstance(first, OptimizationResult)
    assert first.best_fitness < 0.25
    assert first.best_solution == second.best_solution
    assert first.best_fitness == second.best_fitness
    assert len(first.convergence_history) == first.metadata["iterations_run"]
    assert first.metadata["n_evaluations"] == first.metadata["iterations_run"] + 1
    assert all(a >= b for a, b in zip(first.convergence_history,
                                     first.convergence_history[1:]))


def test_sa_rejects_invalid_bounds_and_settings():
    optimizer = SimulatedAnnealing()
    with pytest.raises(ValueError):
        optimizer.optimize(lambda x: 0.0, [])
    with pytest.raises(ValueError):
        optimizer.optimize(lambda x: 0.0, [(1, -1)])
    with pytest.raises(ValueError):
        optimizer.optimize(lambda x: 0.0, [(0, 1)], {"cooling_rate": 1.0})


def test_fuzzy_controller_contract_and_state_reset():
    controller = FuzzyController()
    assert len(controller.parameter_bounds()) == 3
    assert np.isfinite(controller.compute(0.25, 0.01))
    controller.set_parameters([3.0, 0.2, 12.0])
    controller.reset()
    assert controller.compute(0.0, 0.01) == pytest.approx(0.0, abs=1e-8)
    assert np.isfinite(controller.compute(-0.4, 0.01))
    with pytest.raises(ValueError):
        controller.set_parameters([1.0, 2.0])


def test_plot_functions_accept_project_data_types():
    import matplotlib
    matplotlib.use("Agg", force=True)
    import matplotlib.pyplot as plt

    simulation = SimulationResult(
        time=np.linspace(0.0, 1.0, 20), reference=1.0,
        output=np.linspace(0.0, 1.0, 20),
    )
    optimization = OptimizationResult(
        best_solution=[0.1], best_fitness=0.01,
        convergence_history=[1.0, 0.5, 0.1], metadata={"algorithm": "SA"},
    )
    experiment = {
        "optimizer_name": "SA", "controller_name": "Fuzzy",
        "plant_name": "Second order",
        "metrics": PerformanceMetrics(1.0, 0.2, 5.0, 0.01, 0.5),
    }

    figures = [
        plot_step_responses([simulation])[0],
        plot_convergence_curves([optimization])[0],
        plot_metric_comparison([experiment], "iae")[0],
    ]
    assert all(fig.axes for fig in figures)
    for figure in figures:
        plt.close(figure)


def test_plot_functions_report_missing_data_clearly():
    with pytest.raises(ValueError):
        plot_step_responses([])
    with pytest.raises(ValueError):
        plot_convergence_curves([{"label": "missing history"}])
    with pytest.raises(ValueError):
        plot_metric_comparison([{"metrics": {"iae": None}}], "iae")
