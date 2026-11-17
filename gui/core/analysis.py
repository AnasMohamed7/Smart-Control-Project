"""Turn result-set records into tidy tables, aggregates and rankings."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

from gui.core.catalog import METRIC_KEYS, PLANT_NAMES
from smart_control.evaluation.objective import DEFAULT_WEIGHTS, calculate_scalar_fitness
from smart_control.types import PerformanceMetrics

OPT_ORDER = ["DE", "GA", "PSO", "SA", "GP"]
CTRL_ORDER = ["PID", "Neural Network", "Fuzzy", "GP Controller"]


def records_frame(records: list[dict]) -> pd.DataFrame:
    rows = []
    for i, r in enumerate(records):
        row = {
            "idx": i, "optimizer": r["optimizer"], "controller": r["controller"],
            "plant": r["plant"], "seed": r["seed"], "status": r.get("status"),
            "fitness": r.get("fitness"), "settled": bool(r.get("settled", False)),
            "execution_time": r.get("execution_time"), "n_evaluations": r.get("n_evaluations"),
            "expression": r.get("expression"), "error": r.get("error"),
        }
        metrics = r.get("metrics") or {}
        for key in METRIC_KEYS:
            row[key] = metrics.get(key)
        rows.append(row)
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    for col in ["fitness", "execution_time", *METRIC_KEYS]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["optimizer"] = pd.Categorical(df["optimizer"], [o for o in OPT_ORDER if o in set(df["optimizer"])])
    df["controller"] = pd.Categorical(df["controller"], [c for c in CTRL_ORDER if c in set(df["controller"])])
    df["plant"] = pd.Categorical(df["plant"], [p for p in PLANT_NAMES if p in set(df["plant"])])
    df["combo"] = df["optimizer"].astype(str) + " · " + df["controller"].astype(str)
    return df


def effective_weights(weights: dict | None = None) -> dict:
    w = dict(DEFAULT_WEIGHTS)
    import smart_control.config as config
    w.update(config.OBJECTIVE_WEIGHTS)
    if weights:
        w.update(weights)
    return w


def rescore(df: pd.DataFrame, weights: dict, max_duration: float = 20.0) -> pd.Series:
    """Recompute the objective for every run from its stored metrics."""
    def score(row):
        m = PerformanceMetrics(
            settling_time=_none(row["settling_time"]), rise_time=_none(row["rise_time"]),
            overshoot=_none(row["overshoot"]), steady_state_error=_none(row["steady_state_error"]),
            iae=_none(row["iae"]),
        )
        return calculate_scalar_fitness(m, weights=weights, max_duration=max_duration)
    return df.apply(score, axis=1)


def _none(v):
    return None if v is None or (isinstance(v, float) and not math.isfinite(v)) else float(v)


def aggregate(df: pd.DataFrame, score_col: str = "fitness") -> pd.DataFrame:
    """One row per optimizer × controller × plant (aggregated over seeds)."""
    if df.empty:
        return df
    ok = df[df["status"] == "success"]
    grouped = ok.groupby(["optimizer", "controller", "plant"], observed=True)
    agg = grouped.agg(
        score_mean=(score_col, "mean"), score_std=(score_col, "std"), score_best=(score_col, "min"),
        settled_rate=("settled", "mean"), runs=("seed", "count"),
        runtime=("execution_time", "mean"),
        **{f"{k}": (k, "mean") for k in METRIC_KEYS},
    ).reset_index()
    best_idx = ok.loc[grouped[score_col].idxmin().values, ["optimizer", "controller", "plant", "idx"]]
    agg = agg.merge(best_idx.rename(columns={"idx": "best_idx"}), on=["optimizer", "controller", "plant"])
    agg["score_std"] = agg["score_std"].fillna(0.0)
    agg["combo"] = agg["optimizer"].astype(str) + " · " + agg["controller"].astype(str)
    return agg


def optimizer_ranks(agg: pd.DataFrame) -> pd.DataFrame:
    """Rank optimizers within each controller × plant cell (1 = best)."""
    numeric = agg[agg["optimizer"].astype(str) != "GP"].copy()
    numeric["rank"] = numeric.groupby(["controller", "plant"], observed=True)["score_mean"].rank(method="min")
    return numeric


def controller_ranks(agg: pd.DataFrame) -> pd.DataFrame:
    df = agg.copy()
    df["rank"] = df.groupby(["optimizer", "plant"], observed=True)["score_mean"].rank(method="min")
    return df


def win_matrix(agg: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Per controller × plant comparison of optimizer a vs b."""
    pa = agg[agg["optimizer"].astype(str) == a].set_index(["controller", "plant"])["score_mean"]
    pb = agg[agg["optimizer"].astype(str) == b].set_index(["controller", "plant"])["score_mean"]
    both = pd.concat([pa.rename(a), pb.rename(b)], axis=1).dropna().reset_index()
    both["winner"] = np.where(both[a] < both[b], a, np.where(both[b] < both[a], b, "tie"))
    both["ratio"] = both[b] / both[a]
    return both


def fmt(value, digits: int = 3, unit: str = "") -> str:
    if value is None or (isinstance(value, float) and not math.isfinite(value)):
        return "—"
    v = float(value)
    if abs(v) >= 1000:
        s = f"{v:,.0f}"
    elif abs(v) >= 100:
        s = f"{v:.1f}"
    else:
        s = f"{v:.{digits}g}"
    return f"{s}{unit}"
