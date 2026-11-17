"""Experiment orchestration for the Smart Control project.

The runner deliberately depends on the common controller/optimizer contracts:
controllers expose ``parameter_bounds``, ``set_parameters``, ``reset`` and
``compute``; optimizers expose ``optimize(objective, bounds, settings)``.

A fresh controller and optimizer are created for every experiment so mutable
controller state and optimizer state cannot leak between runs.
"""
from __future__ import annotations

import copy
import re
import time
from collections.abc import Mapping
from typing import Any

import smart_control.config as config
from smart_control.evaluation.metrics import calculate_all_metrics
from smart_control.evaluation.objective import build_objective, calculate_scalar_fitness


def _component_name(component: Any, fallback: str | None = None) -> str:
    """Return a readable component name for classes and instances."""
    if fallback:
        return str(fallback)
    name = getattr(component, "name", None)
    if name:
        return str(name)
    cls = component if isinstance(component, type) else type(component)
    return cls.__name__


def _make_component(spec: Any) -> Any:
    """Instantiate a class/factory or clone an existing mutable instance."""
    if isinstance(spec, type):
        return spec()
    if callable(spec) and not hasattr(spec, "optimize") and not hasattr(spec, "compute"):
        return spec()
    return copy.deepcopy(spec)


def _normalise_settings(settings: Mapping[str, Any] | None) -> dict[str, Any]:
    raw = dict(settings or {})
    return {
        "simulation_config": dict(raw.get("simulation_config", raw.get("simulation", config.SIMULATION))),
        "metric_config": dict(raw.get("metric_config", raw.get("metrics", {}))),
        "objective_config": dict(raw.get("objective_config", raw.get("objective", config.OBJECTIVE_WEIGHTS))),
        "optimizer_settings": dict(raw.get("optimizer_settings", raw.get("optimizer", {}))),
        "simulator": raw.get("simulator"),
        "plant_name": raw.get("plant_name"),
        "controller_name": raw.get("controller_name"),
        "optimizer_name": raw.get("optimizer_name"),
    }


def _default_simulator(plant: Any, simulation_config: dict[str, Any]):
    """Build the project's standard closed-loop simulator lazily."""
    from smart_control.plants.simulation import simulate_closed_loop

    return lambda controller: simulate_closed_loop(plant, controller, simulation_config)


def _plant_order(plant: Any, plant_name: str) -> int | None:
    """Infer the order from the canonical plant key or transfer denominator."""
    match = re.search(r"(second|third|fourth)_order", plant_name.lower())
    if match:
        return {"second": 2, "third": 3, "fourth": 4}[match.group(1)]
    try:
        denominator = plant.den[0][0]
        return max(0, len(denominator) - 1)
    except Exception:
        return None


def _is_gp_optimizer(optimizer: Any, optimizer_name: str) -> bool:
    return "geneticprogramming" in type(optimizer).__name__.replace("_", "").lower() or optimizer_name.upper() == "GP"


def _is_gp_controller(controller: Any, controller_name: str) -> bool:
    return type(controller).__name__.lower() == "gpcontroller" or controller_name.lower() in {
        "gp", "gp_controller", "genetic_programming_controller", "genetic programming controller"
    }


def run_single_experiment(optimizer, controller, plant, settings=None):
    """Run one optimizer/controller/plant combination and return a record.

    Parameters may be classes, zero-argument factories, or instances. Instances
    are deep-copied before use. ``settings`` accepts:

    - ``simulation_config``: keys for ``simulate_closed_loop``
    - ``metric_config``: metric thresholds
    - ``objective_config``: weights and objective penalties
    - ``optimizer_settings``: settings passed to the optimizer
    - ``simulator``: optional callable ``simulator(controller) -> SimulationResult``
    - optional display names: ``optimizer_name``, ``controller_name``, ``plant_name``
    """
    cfg = _normalise_settings(settings)
    optimizer_instance = _make_component(optimizer)
    controller_instance = _make_component(controller)

    optimizer_name = cfg["optimizer_name"] or _component_name(optimizer_instance)
    controller_name = cfg["controller_name"] or _component_name(controller_instance)
    plant_name = cfg["plant_name"] or getattr(plant, "name", None) or type(plant).__name__

    if _is_gp_optimizer(optimizer_instance, optimizer_name) and not _is_gp_controller(
        controller_instance, controller_name
    ):
        raise ValueError(
            "The current GeneticProgramming optimizer evolves GP expression-tree vectors "
            "and is compatible with GPController only. It is not a drop-in optimizer "
            "for PID, Neural Network, or Fuzzy parameter vectors."
        )

    bounds = controller_instance.parameter_bounds()
    if not bounds:
        raise ValueError(f"{controller_name} returned no parameter bounds")

    simulator = cfg["simulator"] or _default_simulator(plant, cfg["simulation_config"])
    objective = build_objective(
        controller=controller_instance,
        plant=plant,
        simulation_config=cfg["simulation_config"],
        metric_config=cfg["metric_config"],
        objective_config=cfg["objective_config"],
        # Supplying the simulator avoids build_objective's fallback probing path.
        simulator=simulator,
    )

    started = time.perf_counter()
    optimization_result = optimizer_instance.optimize(
        objective,
        bounds,
        settings=cfg["optimizer_settings"] or None,
    )

    best_solution = list(optimization_result.best_solution)
    if len(best_solution) != len(bounds):
        raise ValueError(
            f"{optimizer_name} returned {len(best_solution)} parameters, "
            f"but {controller_name} requires {len(bounds)}"
        )

    # Re-run the best candidate once to retain its actual response and metrics.
    controller_instance.set_parameters(best_solution)
    simulation_result = simulator(controller_instance)
    if simulation_result is None:
        raise RuntimeError("The best candidate produced an invalid/unstable simulation")

    metrics = calculate_all_metrics(simulation_result, cfg["metric_config"])
    import numpy as np
    response_output = (
        simulation_result.get("output")
        if isinstance(simulation_result, Mapping)
        else simulation_result.output
    )
    response_values = np.asarray(response_output, dtype=float).reshape(-1)
    simulation_finite = bool(response_values.size and np.all(np.isfinite(response_values)))
    best_cost_from_response = calculate_scalar_fitness(
        metrics,
        weights=cfg["objective_config"],
        max_duration=float(cfg["simulation_config"].get("time_end", config.T_END)),
    )
    elapsed = time.perf_counter() - started

    metadata = dict(getattr(optimization_result, "metadata", {}) or {})
    return {
        "status": "success",
        "optimizer_name": optimizer_name,
        "controller_name": controller_name,
        "plant_name": str(plant_name),
        "plant_order": _plant_order(plant, str(plant_name)),
        "best_solution": best_solution,
        "best_fitness": float(optimization_result.best_fitness),
        "fitness_recomputed": float(best_cost_from_response),
        "metrics": metrics,
        # Execution success does not imply a well-controlled/settled response.
        "simulation_finite": simulation_finite,
        "settled": metrics.settling_time is not None,
        "convergence_history": list(optimization_result.convergence_history),
        "execution_time": float(elapsed),
        "n_evaluations": metadata.get("n_evaluations"),
        "iterations_run": metadata.get("iterations_run", len(optimization_result.convergence_history)),
        "optimization_result": optimization_result,
        "simulation_result": simulation_result,
        "label": f"{optimizer_name} / {controller_name} / {plant_name}",
        "error": None,
    }


def _named_items(items, kind: str):
    """Accept mappings (preferred) or iterables of component specifications."""
    if isinstance(items, Mapping):
        return list(items.items())
    named = []
    for item in items:
        name = _component_name(item)
        named.append((name, item))
    if not named:
        raise ValueError(f"At least one {kind} must be supplied")
    return named


def run_experiment_matrix(optimizers, controllers, plants, settings=None):
    """Run the experiment matrix and return one record per attempted run.

    The four numeric optimizers (GA, DE, PSO, SA) are run against all four
    controller types and all plants. If a GeneticProgramming optimizer is
    included, it is run only against GPController because its current search
    representation is specifically a GP tree encoding.

    Failed experiments are retained as records with ``status='failed'`` so one
    failure does not erase the results from all other combinations. Set
    ``settings={'raise_errors': True}`` to fail fast.
    """
    raw_settings = dict(settings or {})
    raise_errors = bool(raw_settings.pop("raise_errors", False))
    optimizer_items = _named_items(optimizers, "optimizer")
    controller_items = _named_items(controllers, "controller")
    plant_items = _named_items(plants, "plant")

    records = []
    for optimizer_name, optimizer_spec in optimizer_items:
        for controller_name, controller_spec in controller_items:
            # The current tree-based GP optimizer is not compatible with numeric
            # parameter vectors belonging to PID/NN/Fuzzy controllers.
            opt_probe = _make_component(optimizer_spec)
            ctrl_probe = _make_component(controller_spec)
            if _is_gp_optimizer(opt_probe, str(optimizer_name)) and not _is_gp_controller(
                ctrl_probe, str(controller_name)
            ):
                continue

            for plant_name, plant in plant_items:
                run_settings = dict(raw_settings)
                # Allow per-algorithm budgets while keeping the public runner
                # convenient for a single shared settings dictionary.
                optimizer_overrides = raw_settings.get("optimizer_settings", {})
                if isinstance(optimizer_overrides, Mapping) and optimizer_name in optimizer_overrides:
                    selected = optimizer_overrides[optimizer_name]
                    if isinstance(selected, Mapping):
                        run_settings["optimizer_settings"] = dict(selected)
                run_settings.update({
                    "optimizer_name": str(optimizer_name),
                    "controller_name": str(controller_name),
                    "plant_name": str(plant_name),
                })
                try:
                    record = run_single_experiment(
                        optimizer_spec, controller_spec, plant, run_settings
                    )
                except Exception as exc:
                    if raise_errors:
                        raise
                    record = {
                        "status": "failed",
                        "optimizer_name": str(optimizer_name),
                        "controller_name": str(controller_name),
                        "plant_name": str(plant_name),
                        "plant_order": _plant_order(plant, str(plant_name)),
                        "best_solution": None,
                        "best_fitness": None,
                        "fitness_recomputed": None,
                        "metrics": None,
                        "simulation_finite": False,
                        "settled": False,
                        "convergence_history": [],
                        "execution_time": None,
                        "n_evaluations": None,
                        "iterations_run": None,
                        "optimization_result": None,
                        "simulation_result": None,
                        "label": f"{optimizer_name} / {controller_name} / {plant_name}",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                records.append(record)
    return records
