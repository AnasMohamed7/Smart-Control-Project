"""Experiment execution and result-set storage for the GUI.

* Experiments are executed with the project's own
  ``smart_control.experiments.runner.run_single_experiment`` — the GUI adds no
  alternative simulation or scoring logic.
* Jobs run in a process pool (one job = optimizer × controller × plant × seed),
  longest first, so a full matrix uses every core.
* A finished batch is a *result set*: a pickle in ``results/gui/`` holding
  compact records (metrics, convergence, best parameters and the best
  step response).  Result sets are what every analysis page reads, so the
  presentation never has to wait for a run.
"""
from __future__ import annotations

import math
import os
import pickle
import re
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from concurrent.futures.process import BrokenProcessPool
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

import smart_control.config as config

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = PROJECT_ROOT / "results" / "gui"

# Equal-evaluation budgets. Population methods evaluate P*(1+I) candidates;
# SA gets the same count as 1 initial + (P*(1+I) - 1) moves.
BUDGETS = {
    "Course default (≈18 evals)": (6, 2),
    "Extended (≈60 evals)": (10, 5),
    "Thorough (≈150 evals)": (15, 9),
    "Smoke test (≈8 evals)": (4, 1),
}


def optimizer_settings_for(pop: int, iters: int) -> dict:
    evals = pop * (1 + iters)
    shared = {"population_size": pop, "max_iterations": iters}
    return {"DE": dict(shared), "GA": dict(shared), "PSO": dict(shared), "GP": dict(shared),
            "SA": {"max_iterations": max(1, evals - 1)}}


def default_run_config() -> dict:
    return {
        "simulation_config": dict(config.SIMULATION),
        "metric_config": {"settling_tolerance": config.SETTLING_TOLERANCE},
        "objective_config": dict(config.OBJECTIVE_WEIGHTS),
    }


# ---------------------------------------------------------------------------
# Jobs
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class Job:
    optimizer: str
    controller: str
    plant: str
    seed: int
    optimizer_settings: tuple   # frozen dict items
    run_config: tuple           # pickled-friendly frozen config

    @property
    def cost_hint(self) -> float:
        # Fuzzy inference is ~20x slower per simulation step than the others.
        weight = 20.0 if self.controller == "Fuzzy" else 1.0
        settings = dict(self.optimizer_settings)
        evals = settings.get("population_size", 1) * (1 + settings.get("max_iterations", 1))
        if self.optimizer == "SA":
            evals = settings.get("max_iterations", 1) + 1
        return weight * evals


def build_jobs(optimizers: Iterable[str], controllers: Iterable[str], plants: Iterable[str],
               seeds: Iterable[int], pop: int, iters: int, run_config: dict | None = None) -> list[Job]:
    from gui.core.catalog import is_compatible

    run_config = run_config or default_run_config()
    frozen_cfg = tuple(sorted((k, tuple(sorted(v.items()))) for k, v in run_config.items()))
    per_opt = optimizer_settings_for(pop, iters)
    jobs = []
    for opt in optimizers:
        for ctrl in controllers:
            if not is_compatible(opt, ctrl):
                continue
            for plant in plants:
                for seed in seeds:
                    settings = dict(per_opt[opt], random_seed=int(seed))
                    jobs.append(Job(opt, ctrl, plant, int(seed), tuple(sorted(settings.items())), frozen_cfg))
    return jobs


def _thaw_config(frozen: tuple) -> dict:
    return {k: dict(v) for k, v in frozen}


def _clean(value):
    if value is None:
        return None
    value = float(value)
    return value if math.isfinite(value) else None


def run_job(job: Job) -> dict:
    """Execute one job in the current process and return a compact record."""
    from gui.core.catalog import CONTROLLERS, OPTIMIZERS, plants
    from smart_control.experiments.runner import run_single_experiment

    cfg = _thaw_config(job.run_config)
    plant_tf = plants()[job.plant]
    settings = {
        "simulation_config": cfg.get("simulation_config", {}),
        "metric_config": cfg.get("metric_config", {}),
        "objective_config": cfg.get("objective_config", {}),
        "optimizer_settings": dict(job.optimizer_settings),
        "optimizer_name": job.optimizer,
        "controller_name": job.controller,
        "plant_name": job.plant,
    }
    base = {"optimizer": job.optimizer, "controller": job.controller, "plant": job.plant,
            "seed": job.seed}
    started = time.perf_counter()
    try:
        rec = run_single_experiment(OPTIMIZERS[job.optimizer], CONTROLLERS[job.controller],
                                    plant_tf, settings)
    except Exception as exc:  # keep the batch alive; surface the error in the UI
        return {**base, "status": "failed", "error": f"{type(exc).__name__}: {exc}",
                "execution_time": time.perf_counter() - started}

    metrics = rec["metrics"].to_dict() if hasattr(rec["metrics"], "to_dict") else dict(rec["metrics"])
    sim = rec["simulation_result"]
    expression = None
    if job.controller == "GP Controller":
        try:
            from smart_control.gp_tree import decode_vector, tree_to_string
            expression = tree_to_string(decode_vector(rec["best_solution"]), precision=3)
        except Exception:
            expression = None
    return {
        **base,
        "status": "success",
        "error": None,
        "fitness": _clean(rec["best_fitness"]),
        "fitness_recomputed": _clean(rec["fitness_recomputed"]),
        "metrics": {k: _clean(v) for k, v in metrics.items()},
        "settled": bool(rec["settled"]),
        "finite": bool(rec["simulation_finite"]),
        "best_solution": [float(x) for x in rec["best_solution"]],
        "convergence": [float(x) for x in rec["convergence_history"]],
        "n_evaluations": rec.get("n_evaluations"),
        "iterations_run": rec.get("iterations_run"),
        "execution_time": float(rec["execution_time"]),
        "time": np.asarray(sim.time, dtype=np.float32),
        "output": np.asarray(sim.output, dtype=np.float32),
        "control": np.asarray(sim.control_signal, dtype=np.float32),
        "reference": float(sim.reference),
        "expression": expression,
        "optimizer_settings": dict(job.optimizer_settings),
    }


def run_jobs(jobs: list[Job], workers: int | None = None,
             on_progress: Callable[[int, int, dict], None] | None = None) -> list[dict]:
    """Run jobs in parallel (longest first). ``on_progress(done, total, record)``."""
    jobs = sorted(jobs, key=lambda j: -j.cost_hint)
    total = len(jobs)
    workers = max(1, min(workers or (os.cpu_count() or 2), total or 1))
    records: list[dict] = []

    def finish(rec: dict) -> None:
        records.append(rec)
        if on_progress:
            on_progress(len(records), total, rec)

    if workers > 1:
        # Worker processes (spawn/forkserver) must be able to import ``gui`` and
        # ``smart_control`` regardless of how the parent was launched.
        paths = os.environ.get("PYTHONPATH", "").split(os.pathsep)
        if str(PROJECT_ROOT) not in paths:
            os.environ["PYTHONPATH"] = os.pathsep.join([str(PROJECT_ROOT), *[p for p in paths if p]])
        try:
            with ProcessPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(run_job, job): job for job in jobs}
                for fut in as_completed(futures):
                    finish(fut.result())
            return records
        except (BrokenProcessPool, OSError, EOFError) as exc:
            # Never lose a live demo to a worker-start problem: finish in-process.
            warnings.warn(f"Process pool failed ({type(exc).__name__}: {exc}); "
                          f"running the remaining jobs sequentially.", RuntimeWarning)
            done = {(r["optimizer"], r["controller"], r["plant"], r["seed"]) for r in records}
            jobs = [j for j in jobs if (j.optimizer, j.controller, j.plant, j.seed) not in done]
    for job in jobs:
        finish(run_job(job))
    return records


# ---------------------------------------------------------------------------
# Result sets
# ---------------------------------------------------------------------------
@dataclass
class ResultSet:
    name: str
    records: list[dict]
    meta: dict = field(default_factory=dict)

    @property
    def slug(self) -> str:
        return slugify(self.name)


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "results"


def save_result_set(result_set: ResultSet) -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    path = RESULTS_DIR / f"{result_set.slug}.pkl"
    with path.open("wb") as fh:
        pickle.dump({"name": result_set.name, "meta": result_set.meta,
                     "records": result_set.records}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return path


def list_result_sets() -> list[Path]:
    if not RESULTS_DIR.exists():
        return []
    return sorted(RESULTS_DIR.glob("*.pkl"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_result_set(path: Path) -> ResultSet:
    # Result sets are written locally by this app (trusted input).
    with Path(path).open("rb") as fh:
        data = pickle.load(fh)
    return ResultSet(name=data["name"], records=data["records"], meta=data.get("meta", {}))


def delete_result_set(path: Path) -> None:
    Path(path).unlink(missing_ok=True)


def execute_batch(name: str, optimizers, controllers, plants_, seeds, pop: int, iters: int,
                  run_config: dict | None = None, workers: int | None = None,
                  on_progress=None, budget_label: str = "") -> ResultSet:
    run_config = run_config or default_run_config()
    jobs = build_jobs(optimizers, controllers, plants_, seeds, pop, iters, run_config)
    started = time.perf_counter()
    records = run_jobs(jobs, workers=workers, on_progress=on_progress)
    wall = time.perf_counter() - started
    meta = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "budget_label": budget_label,
        "population": pop, "iterations": iters, "evaluations": pop * (1 + iters),
        "seeds": list(seeds),
        "optimizers": list(optimizers), "controllers": list(controllers), "plants": list(plants_),
        "run_config": run_config,
        "wall_time": wall,
        "cpu_time": float(sum(r.get("execution_time") or 0.0 for r in records)),
        "workers": workers or os.cpu_count(),
        "n_runs": len(records),
    }
    return ResultSet(name=name, records=records, meta=meta)


# ---------------------------------------------------------------------------
# Re-simulation (playground / robustness)
# ---------------------------------------------------------------------------
def simulate(controller_name: str, parameters, plant_tf, simulation_config: dict | None = None):
    """Simulate a controller with given parameters on any plant; returns (sim, metrics)."""
    from gui.core.catalog import make_controller
    from smart_control.evaluation.metrics import calculate_all_metrics
    from smart_control.plants.simulation import simulate_closed_loop

    cfg = dict(config.SIMULATION)
    if simulation_config:
        cfg.update(simulation_config)
    ctrl = make_controller(controller_name, parameters)
    sim = simulate_closed_loop(plant_tf, ctrl, cfg)
    metrics = calculate_all_metrics(sim, {"settling_tolerance": config.SETTLING_TOLERANCE})
    return sim, metrics
