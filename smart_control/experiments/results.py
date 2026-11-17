"""Result conversion and export helpers for experiment records."""
from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Iterable


METRIC_COLUMNS = (
    "settling_time",
    "rise_time",
    "overshoot",
    "steady_state_error",
    "iae",
)

CSV_COLUMNS = (
    "status",
    "optimizer_name",
    "controller_name",
    "plant_name",
    "plant_order",
    "simulation_finite",
    "settled",
    "best_fitness",
    "fitness_recomputed",
    *METRIC_COLUMNS,
    "execution_time",
    "n_evaluations",
    "iterations_run",
    "best_solution",
    "convergence_history",
    "error",
)


def _read(record: Any, key: str, default=None):
    if isinstance(record, Mapping):
        return record.get(key, default)
    return getattr(record, key, default)


def _metrics_dict(metrics: Any) -> dict[str, Any]:
    if metrics is None:
        return {}
    if isinstance(metrics, Mapping):
        return dict(metrics)
    if hasattr(metrics, "to_dict"):
        return metrics.to_dict()
    return {name: getattr(metrics, name, None) for name in METRIC_COLUMNS}


def records_to_rows(experiment_records: Iterable[Any]) -> list[dict[str, Any]]:
    """Flatten experiment records into CSV/table-friendly dictionaries."""
    rows = []
    for record in experiment_records:
        metrics = _metrics_dict(_read(record, "metrics"))
        row = {column: _read(record, column) for column in CSV_COLUMNS}
        for metric in METRIC_COLUMNS:
            row[metric] = metrics.get(metric)
        parameters = row.get("best_solution")
        history = row.get("convergence_history")
        row["best_solution"] = json.dumps(parameters, separators=(",", ":")) if parameters is not None else ""
        row["convergence_history"] = json.dumps(history, separators=(",", ":")) if history else ""
        rows.append(row)
    return rows


def save_results_csv(experiment_records: Iterable[Any], output_path: str | Path) -> Path:
    """Save flattened records as CSV using only Python's standard library.

    Arrays such as the best parameter vector and convergence history are stored
    as JSON strings in their respective CSV cells. The full step response stays
    in memory for plotting and is intentionally not expanded into CSV columns.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = records_to_rows(experiment_records)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return path


def successful_records(experiment_records: Iterable[Any]) -> list[Any]:
    """Return only successful runs, preserving original record objects."""
    return [record for record in experiment_records if _read(record, "status", "success") == "success"]


def print_results_table(experiment_records: Iterable[Any]) -> None:
    """Print a compact dependency-free summary table to the console."""
    rows = records_to_rows(experiment_records)
    if not rows:
        print("No experiment records.")
        return
    headers = ["Run", "Settled", "Optimizer", "Controller", "Plant", "Fitness", "Ts", "Tr", "OS %", "Ess", "IAE", "Time (s)"]
    keys = ["status", "settled", "optimizer_name", "controller_name", "plant_name", "best_fitness",
            "settling_time", "rise_time", "overshoot", "steady_state_error", "iae", "execution_time"]

    def fmt(value):
        if value is None or value == "":
            return "-"
        if isinstance(value, float):
            return f"{value:.4g}"
        return str(value)

    display_rows = [[fmt(row.get(key)) for key in keys] for row in rows]
    widths = [max(len(headers[i]), *(len(row[i]) for row in display_rows)) for i in range(len(headers))]
    print(" | ".join(headers[i].ljust(widths[i]) for i in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in display_rows:
        print(" | ".join(row[i].ljust(widths[i]) for i in range(len(headers))))
