"""Top-level entry point for the Smart Control course project.

Run from the repository root:
    python -m smart_control.main

For a short integration smoke run:
    python -m smart_control.main --quick

The optional GP optimizer run is restricted to GPController because the current
GP optimizer evolves expression-tree encodings rather than arbitrary numeric
controller parameter vectors.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import smart_control.config as config
from smart_control.controllers.fuzzy import FuzzyController
from smart_control.controllers.gp_controller import GPController
from smart_control.controllers.neural_network import NeuralNetworkController
from smart_control.controllers.pid import PIDController
from smart_control.evaluation.objective import DEFAULT_WEIGHTS
from smart_control.experiments.results import print_results_table, save_results_csv, successful_records
from smart_control.experiments.runner import run_experiment_matrix
from smart_control.optimizers.differential_evolution import DifferentialEvolution
from smart_control.optimizers.genetic_algorithm import GeneticAlgorithm
from smart_control.optimizers.particle_swarm import ParticleSwarmOptimization
from smart_control.optimizers.simulated_annealing import SimulatedAnnealing
from smart_control.visualization.plots import plot_convergence_curves, plot_step_responses


def _build_optimizers(include_gp: bool = False):
    optimizers = {
        "DE": DifferentialEvolution,
        "GA": GeneticAlgorithm,
        "PSO": ParticleSwarmOptimization,
        "SA": SimulatedAnnealing,
    }
    if include_gp:
        from smart_control.optimizers.genetic_programming import GeneticProgramming
        optimizers["GP"] = GeneticProgramming
    return optimizers


def _build_controllers():
    return {
        "PID": PIDController,
        "Neural Network": NeuralNetworkController,
        "Fuzzy": FuzzyController,
        "GP Controller": GPController,
    }


def _save_plots(records, output_dir: Path):
    """Save one step-response and one convergence plot per plant."""
    successful = successful_records(records)
    plant_names = sorted({record["plant_name"] for record in successful})
    for plant_name in plant_names:
        plant_records = [r for r in successful if r["plant_name"] == plant_name]
        if not plant_records:
            continue

        fig, _ = plot_step_responses(plant_records)
        fig.savefig(output_dir / f"step_responses_{plant_name}.png", dpi=140, bbox_inches="tight")
        import matplotlib.pyplot as plt
        plt.close(fig)

        convergence_records = [r for r in plant_records if r.get("convergence_history")]
        if convergence_records:
            fig, _ = plot_convergence_curves(convergence_records)
            fig.savefig(output_dir / f"convergence_{plant_name}.png", dpi=140, bbox_inches="tight")
            plt.close(fig)


def main(settings=None, output_dir=None, include_gp=False, quick=False):
    """Run the configured experiment matrix and save the deliverables.

    The default uses about 18 objective evaluations per optimizer/controller/
    plant combination to keep the five-minute target plausible. ``settings``
    can override simulation, metrics, objective, and optimizer settings.
    ``quick=True`` is only for integration testing, not final results.
    """
    from smart_control.plants.models import get_plant_models

    project_root = Path(__file__).resolve().parents[1]
    destination = Path(output_dir) if output_dir is not None else project_root / "results"
    destination.mkdir(parents=True, exist_ok=True)

    run_settings = {
        "simulation_config": dict(config.SIMULATION),
        "metric_config": {"settling_tolerance": config.SETTLING_TOLERANCE},
        "objective_config": dict(config.OBJECTIVE_WEIGHTS),
        # A bounded, shared evaluation budget is used by the entry point to
        # respect the five-minute target. Population methods: 10 initial
        # candidates + 2 generations x 6 = 18 evaluations. SA: one initial
        # candidate + 17 moves = 18 evaluations. This is a low-budget runtime
        # compromise; increase it only after benchmarking the full matrix.
        "optimizer_settings": {
            "DE": {"population_size": 6, "max_iterations": 2},
            "GA": {"population_size": 6, "max_iterations": 2},
            "PSO": {"population_size": 6, "max_iterations": 2},
            "SA": {"max_iterations": 17},
            "GP": {"population_size": 6, "max_iterations": 2},
        },
    }
    if settings:
        for key, value in settings.items():
            if isinstance(value, dict) and isinstance(run_settings.get(key), dict):
                run_settings[key].update(value)
            else:
                run_settings[key] = value

    if quick:
        run_settings["optimizer_settings"] = {
            "DE": {"population_size": 4, "max_iterations": 1},
            "GA": {"population_size": 4, "max_iterations": 1},
            "PSO": {"population_size": 4, "max_iterations": 1},
            "SA": {"max_iterations": 7},
            "GP": {"population_size": 4, "max_iterations": 1},
        }
        # Quick mode evaluates approximately 8 candidates per run.

    plants = get_plant_models()
    optimizers = _build_optimizers(include_gp=include_gp)
    controllers = _build_controllers()

    print("Smart Control Project")
    print("---------------------")
    print(f"Plants      : {', '.join(plants)}")
    print(f"Controllers : {', '.join(controllers)}")
    print(f"Optimizers  : {', '.join(optimizers)}")
    print(f"Simulation  : dt={run_settings['simulation_config'].get('time_step')} s, "
          f"T={run_settings['simulation_config'].get('time_end')} s")
    effective_weights = dict(DEFAULT_WEIGHTS)
    effective_weights.update(run_settings["objective_config"])
    print(f"Objective weights: {effective_weights}")
    print("Note: GP optimizer, when enabled, runs only with the GP controller.")

    started = time.perf_counter()
    records = run_experiment_matrix(optimizers, controllers, plants, run_settings)
    run_elapsed = time.perf_counter() - started

    csv_path = save_results_csv(records, destination / "experiment_results.csv")
    _save_plots(records, destination)
    total_elapsed = time.perf_counter() - started

    print("\nExperiment summary")
    print_results_table(records)
    failed = [r for r in records if r.get("status") != "success"]
    print(f"\nCompleted: {len(records) - len(failed)} / {len(records)}")
    print(f"Failed   : {len(failed)}")
    print(f"Run time : {run_elapsed:.2f} s")
    print(f"Total    : {total_elapsed:.2f} s (including CSV/plots)")
    print(f"CSV      : {csv_path}")
    print(f"Outputs  : {destination}")

    if total_elapsed > 300.0:
        print("WARNING: Total execution exceeded the 5-minute project target.")
    if failed:
        print("\nFailure details:")
        for record in failed:
            print(f"- {record['label']}: {record['error']}")

    return records


def _cli():
    parser = argparse.ArgumentParser(description="Run Smart Control experiments")
    parser.add_argument("--quick", action="store_true", help="small smoke-test run; not final results")
    parser.add_argument("--include-gp", action="store_true", help="also run GP optimizer on GP controller only")
    parser.add_argument("--output-dir", default=None, help="directory for CSV and plots")
    args = parser.parse_args()
    main(output_dir=args.output_dir, include_gp=args.include_gp, quick=args.quick)


if __name__ == "__main__":
    _cli()
