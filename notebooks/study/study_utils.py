"""Shared helpers for the optimiser study notebooks.

Only the course-permitted libraries are used (numpy, matplotlib, control).
Every experiment goes through the project's own
``smart_control.experiments.runner.run_single_experiment`` so the notebooks
measure exactly what the final system measures.
"""
from __future__ import annotations

import math
import sys
import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ---------------------------------------------------------------------------
# Project import
# ---------------------------------------------------------------------------
def find_project_root(start: Path | None = None) -> Path:
    here = (start or Path.cwd()).resolve()
    for candidate in [here, *here.parents]:
        if (candidate / "smart_control" / "main.py").exists():
            return candidate
    raise FileNotFoundError("Run the notebook from inside the project folder (smart_control/ not found).")


ROOT = find_project_root(Path(__file__).parent)
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import smart_control.config as config  # noqa: E402
from smart_control.controllers.fuzzy import FuzzyController  # noqa: E402
from smart_control.controllers.gp_controller import GPController  # noqa: E402
from smart_control.controllers.neural_network import NeuralNetworkController  # noqa: E402
from smart_control.controllers.pid import PIDController  # noqa: E402
from smart_control.experiments.runner import run_single_experiment  # noqa: E402
from smart_control.plants.models import get_plant_models  # noqa: E402

PLANTS = get_plant_models()
PLANT_NAMES = list(PLANTS)
PLANT_LABELS = {"second_order": "2nd order", "third_order": "3rd order", "fourth_order": "4th order"}
CONTROLLERS = {"PID": PIDController, "Neural Network": NeuralNetworkController,
               "Fuzzy": FuzzyController, "GP Controller": GPController}

# Same entity colours as the presentation app.
OPT_COLORS = {"DE": "#2a78d6", "GA": "#eb6834", "PSO": "#1baf7a", "SA": "#eda100", "GP": "#6250d6"}
CTRL_COLORS = {"PID": "#e87ba4", "Neural Network": "#008300", "Fuzzy": "#6250d6", "GP Controller": "#e34948"}
PLANT_COLORS = {"second_order": "#86b6ef", "third_order": "#2a78d6", "fourth_order": "#104281"}

plt.rcParams.update({
    "figure.dpi": 110, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.alpha": 0.25, "axes.titleweight": "bold", "font.size": 10,
})

RUN_SETTINGS = {
    "simulation_config": dict(config.SIMULATION),
    "metric_config": {"settling_tolerance": config.SETTLING_TOLERANCE},
    "objective_config": dict(config.OBJECTIVE_WEIGHTS),
}
METRICS = ["iae", "settling_time", "rise_time", "overshoot", "steady_state_error"]


# ---------------------------------------------------------------------------
# Budgets and runs
# ---------------------------------------------------------------------------
def budget(optimizer: str, pop: int, iters: int, **extra) -> dict:
    """Equal-evaluation settings: population methods use P*(1+I) evals, SA the same count of moves."""
    if optimizer == "SA":
        return {"max_iterations": max(1, pop * (1 + iters) - 1), **extra}
    return {"population_size": pop, "max_iterations": iters, **extra}


def run(optimizer_cls, controller: str, plant: str, settings: dict, seed: int = 42) -> dict:
    """One optimiser × controller × plant run through the project's runner."""
    cfg = dict(RUN_SETTINGS)
    cfg["optimizer_settings"] = {**settings, "random_seed": seed}
    cfg.update(optimizer_name=getattr(optimizer_cls, "name", optimizer_cls.__name__),
               controller_name=controller, plant_name=plant)
    rec = run_single_experiment(optimizer_cls, CONTROLLERS[controller], PLANTS[plant], cfg)
    rec["seed"] = seed
    return rec


def run_matrix(optimizer_cls, controllers, plants, settings, seeds, verbose=True) -> list[dict]:
    records, started = [], time.perf_counter()
    for ctrl in controllers:
        for plant in plants:
            for seed in seeds:
                rec = run(optimizer_cls, ctrl, plant, settings, seed)
                records.append(rec)
                if verbose:
                    m = rec["metrics"]
                    print(f"{ctrl:15s} {PLANT_LABELS[plant]:9s} seed {seed}  J = {rec['best_fitness']:9.4g}  "
                          f"IAE = {fmt(m.iae):>7s}  settled = {'yes' if rec['settled'] else 'no ':3s}  "
                          f"({rec['execution_time']:.1f} s)")
    if verbose:
        print(f"\n{len(records)} runs in {time.perf_counter() - started:.0f} s")
    return records


def fmt(v, digits=3) -> str:
    if v is None or (isinstance(v, float) and not math.isfinite(v)):
        return "—"
    return f"{v:.{digits}g}"


def best_by_cell(records: list[dict]) -> dict:
    """{(controller, plant): best-seed record}."""
    best = {}
    for r in records:
        key = (r["controller_name"], r["plant_name"])
        if key not in best or r["best_fitness"] < best[key]["best_fitness"]:
            best[key] = r
    return best


def summary_table(records: list[dict]) -> None:
    """Mean ± std of J across seeds plus the best seed's metrics, one row per cell."""
    cells = {}
    for r in records:
        cells.setdefault((r["controller_name"], r["plant_name"]), []).append(r)
    head = f"{'Controller':15s} {'Plant':9s} {'J mean':>9s} {'± std':>8s} {'settled':>8s} " \
           f"{'IAE':>7s} {'Ts (s)':>7s} {'Tr (s)':>7s} {'OS %':>7s} {'Ess':>8s}"
    print(head)
    print("-" * len(head))
    for (ctrl, plant), rs in cells.items():
        js = np.array([r["best_fitness"] for r in rs])
        b = min(rs, key=lambda r: r["best_fitness"])["metrics"]
        settled = sum(r["settled"] for r in rs)
        print(f"{ctrl:15s} {PLANT_LABELS[plant]:9s} {js.mean():9.4g} {js.std():8.3g} {settled:>5d}/{len(rs)} "
              f"{fmt(b.iae):>7s} {fmt(b.settling_time):>7s} {fmt(b.rise_time):>7s} "
              f"{fmt(b.overshoot):>7s} {fmt(b.steady_state_error):>8s}")


# ---------------------------------------------------------------------------
# Benchmarks
# ---------------------------------------------------------------------------
def sphere(x):
    x = np.asarray(x, float)
    return float(np.sum(x ** 2))


def rastrigin(x):
    x = np.asarray(x, float)
    return float(10 * x.size + np.sum(x ** 2 - 10 * np.cos(2 * np.pi * x)))


def rosenbrock(x):
    x = np.asarray(x, float)
    return float(np.sum(100 * (x[1:] - x[:-1] ** 2) ** 2 + (1 - x[:-1]) ** 2))


BENCHMARKS = {"Sphere": (sphere, 5.0), "Rastrigin": (rastrigin, 5.12), "Rosenbrock": (rosenbrock, 2.0)}


class Tracked:
    """Objective wrapper that records every evaluated point (to see *where* an optimiser searches)."""

    def __init__(self, fn):
        self.fn, self.points, self.values = fn, [], []

    def __call__(self, x):
        v = self.fn(x)
        self.points.append(np.asarray(x, float).copy())
        self.values.append(v)
        return v

    def best_so_far(self):
        return np.minimum.accumulate(self.values)


def benchmark_runs(optimizer_cls, settings: dict, dims=(2, 5, 10), seeds=(0, 1, 2)) -> dict:
    out = {}
    for name, (fn, half) in BENCHMARKS.items():
        for d in dims:
            finals = []
            for seed in seeds:
                obj = Tracked(fn)
                optimizer_cls().optimize(obj, [(-half, half)] * d, settings={**settings, "random_seed": seed})
                finals.append(min(obj.values))
            out[(name, d)] = finals
    return out


def benchmark_table(results: dict) -> None:
    print(f"{'Function':11s} {'dim':>4s} {'best f (mean of seeds)':>23s} {'worst seed':>11s}")
    for (name, d), finals in results.items():
        print(f"{name:11s} {d:4d} {np.mean(finals):23.4g} {np.max(finals):11.4g}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------
def plot_search(optimizer_cls, settings: dict, color: str, seed: int = 0, title: str = ""):
    """Where does the optimiser sample on 2-D Rastrigin, and how fast does it improve?"""
    obj = Tracked(rastrigin)
    optimizer_cls().optimize(obj, [(-5.12, 5.12)] * 2, settings={**settings, "random_seed": seed})
    pts = np.array(obj.points)
    xs = np.linspace(-5.12, 5.12, 200)
    X, Y = np.meshgrid(xs, xs)
    Z = 20 + X ** 2 - 10 * np.cos(2 * np.pi * X) + Y ** 2 - 10 * np.cos(2 * np.pi * Y)
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4.3), gridspec_kw={"width_ratios": [1, 1.3]})
    a1.contourf(X, Y, np.log1p(Z), levels=20, cmap="Greys", alpha=0.55)
    a1.scatter(pts[:, 0], pts[:, 1], c=np.arange(len(pts)), cmap="viridis", s=14, edgecolors="none")
    k = int(np.argmin(obj.values))
    a1.plot(*pts[k], marker="*", ms=16, color=color, mec="black", label=f"best f = {obj.values[k]:.3g}")
    a1.plot(0, 0, marker="+", ms=14, color="black", mew=2, label="global optimum")
    a1.set(title="Evaluated points (dark = early, yellow = late)", xlabel="x₁", ylabel="x₂", aspect="equal")
    a1.legend(loc="lower left", fontsize=8)
    a2.plot(np.arange(1, len(obj.values) + 1), obj.best_so_far(), color=color, lw=2)
    a2.set(yscale="log", title="Best-so-far value", xlabel="objective evaluations", ylabel="f (log)")
    fig.suptitle(title or "Search behaviour on 2-D Rastrigin", fontweight="bold")
    fig.tight_layout()
    plt.show()


def sweep_heatmap(grid: np.ndarray, row_labels, col_labels, row_name: str, col_name: str, title: str,
                  std: np.ndarray | None = None):
    """Annotated heatmap of mean J (lower is better, best cell outlined)."""
    fig, ax = plt.subplots(figsize=(1.6 * len(col_labels) + 2.5, 1.0 * len(row_labels) + 1.6))
    im = ax.imshow(np.log10(grid), cmap="Blues_r", aspect="auto")
    best = np.unravel_index(np.nanargmin(grid), grid.shape)
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            txt = f"{grid[i, j]:.3g}" + (f"\n±{std[i, j]:.2g}" if std is not None else "")
            ax.text(j, i, txt, ha="center", va="center", fontsize=9,
                    color="white" if np.log10(grid[i, j]) < np.nanmean(np.log10(grid)) else "black",
                    fontweight="bold" if (i, j) == best else "normal")
    ax.add_patch(plt.Rectangle((best[1] - 0.5, best[0] - 0.5), 1, 1, fill=False, ec="#eb6834", lw=3))
    ax.set_xticks(range(len(col_labels)), [str(c) for c in col_labels])
    ax.set_yticks(range(len(row_labels)), [str(r) for r in row_labels])
    ax.set(xlabel=col_name, ylabel=row_name, title=title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="log₁₀ mean J")
    fig.tight_layout()
    plt.show()
    return best


def plot_budget_split(splits, means, stds, color, title):
    labels = [f"P={p}\nG={g}" if g else f"{p}\nevals" for p, g in splits]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    ax.bar(labels, means, yerr=stds, color=color, alpha=0.85, capsize=4)
    k = int(np.argmin(means))
    ax.bar(labels[k], means[k], color=color, edgecolor="black", lw=2)
    ax.set(ylabel="mean J over seeds", title=title, yscale="log")
    fig.tight_layout()
    plt.show()


def plot_matrix_heatmap(records, title):
    ctrls = [c for c in CONTROLLERS if any(r["controller_name"] == c for r in records)]
    grid = np.full((len(ctrls), len(PLANT_NAMES)), np.nan)
    settled = np.zeros_like(grid)
    for i, c in enumerate(ctrls):
        for j, p in enumerate(PLANT_NAMES):
            rs = [r for r in records if r["controller_name"] == c and r["plant_name"] == p]
            if rs:
                grid[i, j] = np.mean([r["best_fitness"] for r in rs])
                settled[i, j] = np.mean([r["settled"] for r in rs])
    fig, ax = plt.subplots(figsize=(7, 0.9 * len(ctrls) + 1.6))
    im = ax.imshow(np.log10(grid), cmap="Blues_r", aspect="auto")
    for i in range(grid.shape[0]):
        for j in range(grid.shape[1]):
            if np.isnan(grid[i, j]):
                continue
            mark = "" if settled[i, j] == 1 else (" ⚠" if settled[i, j] > 0 else " ✖")
            ax.text(j, i, f"{grid[i, j]:.3g}{mark}\nsettled {settled[i, j]:.0%}", ha="center", va="center",
                    fontsize=9, color="white" if np.log10(grid[i, j]) < np.nanmean(np.log10(grid)) else "black")
    ax.set_xticks(range(len(PLANT_NAMES)), [PLANT_LABELS[p] for p in PLANT_NAMES])
    ax.set_yticks(range(len(ctrls)), ctrls)
    ax.set_title(title)
    ax.grid(False)
    fig.colorbar(im, ax=ax, label="log₁₀ mean J")
    fig.tight_layout()
    plt.show()


def plot_step_grid(records, title, ref=1.0, tol=0.02):
    best = best_by_cell(records)
    ctrls = [c for c in CONTROLLERS if any(k[0] == c for k in best)]
    fig, axes = plt.subplots(1, len(PLANT_NAMES), figsize=(14, 3.9), sharey=True)
    for ax, plant in zip(np.atleast_1d(axes), PLANT_NAMES):
        ax.axhspan(ref * (1 - tol), ref * (1 + tol), color="#0ca30c", alpha=0.12)
        ax.axhline(ref, color="grey", lw=0.8)
        for c in ctrls:
            r = best.get((c, plant))
            if r is None:
                continue
            sim = r["simulation_result"]
            ax.plot(sim.time, sim.output, color=CTRL_COLORS[c], lw=1.8,
                    label=f"{c} (IAE {fmt(r['metrics'].iae)})")
        ax.set(title=PLANT_LABELS[plant], xlabel="time (s)", ylim=(-0.2, 2.0))
        ax.legend(fontsize=7.5, loc="lower right")
    np.atleast_1d(axes)[0].set_ylabel("y(t)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


def plot_convergence_grid(records, color, title):
    ctrls = [c for c in CONTROLLERS if any(r["controller_name"] == c for r in records)]
    fig, axes = plt.subplots(1, len(ctrls), figsize=(3.6 * len(ctrls), 3.3), squeeze=False)
    for ax, c in zip(axes[0], ctrls):
        for plant, ls in zip(PLANT_NAMES, ["-", "--", ":"]):
            for k, r in enumerate([r for r in records if r["controller_name"] == c and r["plant_name"] == plant]):
                h = np.asarray(r["convergence_history"], float)
                n = r.get("n_evaluations") or len(h)
                x = np.linspace(n / len(h), n, len(h))
                ax.plot(x, h, ls=ls, color=PLANT_COLORS[plant], lw=1.5, alpha=0.9,
                        label=PLANT_LABELS[plant] if k == 0 else None, drawstyle="steps-post")
        ax.set(title=c, yscale="log", xlabel="evaluations")
        ax.legend(fontsize=7.5)
    axes[0][0].set_ylabel("best J so far (log)")
    fig.suptitle(title, fontweight="bold")
    fig.tight_layout()
    plt.show()


def plot_control_effort(records, plant, title):
    best = best_by_cell(records)
    fig, ax = plt.subplots(figsize=(10, 3.2))
    for c in CONTROLLERS:
        r = best.get((c, plant))
        if r is not None:
            sim = r["simulation_result"]
            ax.plot(sim.time, sim.control_signal, color=CTRL_COLORS[c], lw=1.4, label=c)
    ax.set(title=title, xlabel="time (s)", ylabel="u(t)", xlim=(0, 10))
    ax.legend(fontsize=8, ncol=4)
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Hyper-parameter sweeps
# ---------------------------------------------------------------------------
def _apply(settings: dict, param, value) -> dict:
    out = dict(settings)
    for name in (param if isinstance(param, (tuple, list)) else (param,)):
        out[name] = value
    return out


def sweep2d(optimizer_cls, base: dict, row_param, rows, col_param, cols,
            controller="PID", plant="third_order", seeds=(42, 43, 44)):
    """Mean/std of the best J over seeds for every (row, col) setting."""
    mean = np.zeros((len(rows), len(cols)))
    std = np.zeros_like(mean)
    started = time.perf_counter()
    for i, rv in enumerate(rows):
        for j, cv in enumerate(cols):
            s = _apply(_apply(base, row_param, rv), col_param, cv)
            js = [run(optimizer_cls, controller, plant, s, seed)["best_fitness"] for seed in seeds]
            mean[i, j], std[i, j] = np.mean(js), np.std(js)
    print(f"{len(rows) * len(cols) * len(seeds)} runs in {time.perf_counter() - started:.0f} s")
    return mean, std


def sweep1d(optimizer_cls, base: dict, param, values, controller="PID", plant="third_order", seeds=(42, 43, 44)):
    means, stds = [], []
    for v in values:
        js = [run(optimizer_cls, controller, plant, _apply(base, param, v), seed)["best_fitness"] for seed in seeds]
        means.append(np.mean(js))
        stds.append(np.std(js))
    return np.array(means), np.array(stds)


def plot_sweep1d(values, means, stds, name, color, title):
    fig, ax = plt.subplots(figsize=(6.5, 3.3))
    labels = [str(v) for v in values]
    ax.bar(labels, means, yerr=stds, color=color, alpha=0.85, capsize=4)
    k = int(np.argmin(means))
    ax.bar(labels[k], means[k], color=color, edgecolor="black", lw=2)
    ax.set(xlabel=name, ylabel="mean J over seeds", title=title, yscale="log")
    fig.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# GP trees
# ---------------------------------------------------------------------------
def draw_tree(tree, ax=None, title=""):
    """Draw a nested-tuple GP expression tree with matplotlib."""
    symbols = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "tanh": "tanh"}
    nodes, edges, counter = [], [], [0]

    def place(node, depth):
        if isinstance(node, tuple):
            kids = [place(c, depth + 1) for c in node[1:]]
            x = sum(k[0] for k in kids) / len(kids)
            label, kind = symbols.get(node[0], node[0]), "fn"
        else:
            x, kids = counter[0], []
            counter[0] += 1
            if isinstance(node, str):
                label, kind = {"e": "e", "de": "ė", "ie": "∫e"}.get(node, node), "var"
            else:
                label, kind = f"{float(node):.3g}", "const"
        me = (x, -depth)
        edges.extend((me, k) for k in kids)
        nodes.append((x, -depth, label, kind))
        return me

    place(tree, 0)
    if ax is None:
        _, ax = plt.subplots(figsize=(max(4, counter[0] * 0.9), 3.2))
    for (x0, y0), (x1, y1) in edges:
        ax.plot([x0, x1], [y0, y1], color="#999", lw=1.2, zorder=1)
    colors = {"fn": "#6250d6", "var": "#1baf7a", "const": "#eda100"}
    for x, y, label, kind in nodes:
        ax.text(x, y, label, ha="center", va="center", fontsize=9, zorder=2,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec=colors[kind], lw=1.8))
    ax.set(title=title)
    ax.axis("off")
    ax.margins(0.15)
    return ax
