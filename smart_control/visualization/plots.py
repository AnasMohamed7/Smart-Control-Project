"""Matplotlib visualizations for simulation, convergence, and metric records."""
from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import matplotlib.pyplot as plt


def _get(obj, *names, default=None):
    """Read a field from either a mapping or an object."""
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _records(value):
    if value is None:
        return []
    if isinstance(value, Mapping):
        return [value]
    return list(value)


def _simulation_payload(record):
    """Accept a SimulationResult or a wrapper record containing one."""
    result = _get(record, "simulation_result", "simulation", "result", default=record)
    time = _get(result, "time", "t")
    output = _get(result, "output", "y", "response")
    reference = _get(result, "reference", "setpoint", "r", default=None)
    if time is None or output is None:
        raise ValueError("Each simulation record needs time and output arrays")
    time = np.asarray(time, dtype=float).reshape(-1)
    output = np.asarray(output, dtype=float).reshape(-1)
    if time.size != output.size:
        raise ValueError("Simulation time and output arrays must have equal length")
    if reference is None:
        reference = np.zeros_like(time)
    else:
        reference = np.asarray(reference, dtype=float)
        if reference.ndim == 0:
            reference = np.full(time.shape, float(reference))
        else:
            reference = reference.reshape(-1)
        if reference.size != time.size:
            raise ValueError("Reference must be scalar or match the time array length")
    label = _get(record, "label", "name", default=None)
    if label is None:
        optimizer = _get(record, "optimizer_name", "optimizer", default="")
        controller = _get(record, "controller_name", "controller", default="")
        plant = _get(record, "plant_name", "plant", default="")
        label = " / ".join(str(x) for x in (optimizer, controller, plant) if x)
    return time, output, reference, str(label or "Simulation")


def plot_step_responses(simulation_records):
    """Plot one or more closed-loop outputs with their reference signals.

    Returns ``(figure, axes)`` so callers can save or further customize it.
    """
    records = _records(simulation_records)
    if not records:
        raise ValueError("simulation_records must contain at least one record")
    payloads = [_simulation_payload(record) for record in records]

    fig, ax = plt.subplots(figsize=(10, 5))
    for time, output, reference, label in payloads:
        ax.plot(time, output, label=label)
        ax.plot(time, reference, linestyle="--", alpha=0.45,
                label=f"{label} reference")
    ax.set(title="Closed-Loop Step Responses", xlabel="Time (s)",
           ylabel="Output")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_convergence_curves(optimization_records):
    """Plot best-so-far fitness against iteration for optimizer records."""
    records = _records(optimization_records)
    if not records:
        raise ValueError("optimization_records must contain at least one record")
    payloads = []
    for record in records:
        result = _get(record, "optimization_result", "optimization", "result",
                      default=record)
        history = _get(result, "convergence_history", "history")
        if history is None:
            raise ValueError("Each optimization record needs convergence_history")
        values = np.asarray(history, dtype=float).reshape(-1)
        if values.size == 0:
            raise ValueError("convergence_history cannot be empty")
        label = _get(record, "label", "name", "optimizer_name", "optimizer",
                     default=_get(result, "algorithm", default="Optimizer"))
        payloads.append((values, str(label)))

    fig, ax = plt.subplots(figsize=(10, 5))
    for values, label in payloads:
        ax.plot(np.arange(1, values.size + 1), values, label=label)
    ax.set(title="Optimizer Convergence", xlabel="Iteration",
           ylabel="Best Fitness (lower is better)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    return fig, ax


def plot_metric_comparison(experiment_records, metric_name):
    """Compare a named performance metric across experiment combinations.

    Records may store metrics in a ``metrics`` mapping/dataclass or expose the
    metric directly. Returns ``(figure, axes)``.
    """
    records = _records(experiment_records)
    if not records:
        raise ValueError("experiment_records must contain at least one record")
    values = []
    labels = []
    for record in records:
        metrics = _get(record, "metrics", default=record)
        value = _get(metrics, metric_name, default=None)
        if value is None:
            continue
        try:
            numeric = float(value)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(numeric):
            continue
        label = _get(record, "label", "name", default=None)
        if label is None:
            optimizer = _get(record, "optimizer_name", "optimizer", default="")
            controller = _get(record, "controller_name", "controller", default="")
            plant = _get(record, "plant_name", "plant", default="")
            label = " / ".join(str(x) for x in (optimizer, controller, plant) if x)
        labels.append(str(label or f"Run {len(labels) + 1}"))
        values.append(numeric)
    if not values:
        raise ValueError(f"No finite values found for metric '{metric_name}'")

    fig_width = max(8.0, min(18.0, 0.55 * len(labels)))
    fig, ax = plt.subplots(figsize=(fig_width, 5))
    positions = np.arange(len(values))
    ax.bar(positions, values)
    ax.set_xticks(positions)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set(title=f"Experiment Comparison: {metric_name.replace('_', ' ').title()}",
           xlabel="Experiment", ylabel=metric_name.replace("_", " ").title())
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    return fig, ax
