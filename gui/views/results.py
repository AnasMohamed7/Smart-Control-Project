"""Results dashboard: the full matrix at a glance, with re-weighting and drill-downs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gui.core import analysis, charts, state
from gui.core import theme as T
from gui.core.catalog import METRIC_KEYS, PLANT_LABELS, PLANT_NAMES, metric_label

VALUE_CHOICES = ["fitness", *METRIC_KEYS, "execution_time"]
CTRLS = ["PID", "Neural Network", "Fuzzy", "GP Controller"]
OPTS = ["DE", "GA", "PSO", "SA", "GP"]


def _weights_panel() -> tuple[dict, bool]:
    base = analysis.effective_weights()
    with st.expander("⚖️  What if we weighted the objective differently?  (re-ranks every run instantly)"):
        st.caption("J is recomputed from each run's stored metrics. Note: the controllers were *tuned* with the "
                   "original weights — this re-scores the same solutions, it does not re-optimise them.")
        cols = st.columns(5)
        w = {}
        for col, key in zip(cols, ["iae", "overshoot", "settling_time", "steady_state_error", "rise_time"]):
            w[key] = col.slider(metric_label(key, False), 0.0, 3.0, float(base[key]), 0.05, key=f"w_{key}")
        def _reset():
            for key in w:
                st.session_state[f"w_{key}"] = float(base[key])
        st.button("Reset to project weights", key="w_reset", type="tertiary", on_click=_reset)
    changed = any(abs(w[k] - base[k]) > 1e-9 for k in w)
    return w, changed


def render() -> None:
    T.page_header("Results", "The full experiment matrix",
                  "Every optimiser × controller × plant combination, scored on the same objective. "
                  "Cells show the mean over seeds; ★ marks the best cell in each plant.")
    rs = state.require_results()
    if rs is None:
        return
    df = state.frame().copy()
    st.caption(f"**{rs.name}** — {state.describe(rs)}")
    plants_avail = state.available(df, "plant", PLANT_NAMES)
    if not plants_avail:
        state.no_data("any successful run", "Every run in this result set failed — see the Data table in the Lab.")
        return

    weights, reweighted = _weights_panel()
    score_col = "fitness"
    if reweighted:
        df["J_custom"] = analysis.rescore(df, weights)
        score_col = "J_custom"
        T.callout("Showing the <b>re-weighted objective</b>. Rankings below use your weights.", "warn")

    c1, c2, c3, c4 = st.columns([1.3, 1.1, 1.9, 0.9])
    value = c1.selectbox("Colour cells by", VALUE_CHOICES, format_func=lambda k: metric_label(k), key="res_value")
    stat = c2.radio("Across seeds", ["Mean", "Best seed"], horizontal=True, key="res_stat",
                    disabled=df["seed"].nunique() < 2)
    # Only plants that exist in this result set are offered (key per set so stale picks never linger).
    plants_sel = c3.multiselect("Plants", plants_avail, default=plants_avail, format_func=PLANT_LABELS.get,
                                key=f"res_plants_{rs.slug}")
    log = c4.toggle("Log colour scale", value=value in ("fitness", "overshoot", "iae", "steady_state_error"), key="res_log")
    if not plants_sel:
        st.info("Select at least one plant.")
        return

    agg = analysis.aggregate(df, score_col)
    agg = agg[agg["plant"].astype(str).isin(plants_sel)]
    df_sel = df[df["plant"].astype(str).isin(plants_sel)]
    col = {"fitness": "score_mean"}.get(value, value)
    if value == "fitness" and stat == "Best seed":
        col = "score_best"
    if value == "execution_time":
        col = "runtime"
    if stat == "Best seed" and value in METRIC_KEYS:
        best_rows = df.loc[agg["best_idx"].values].reset_index(drop=True)
        agg = agg.copy()
        agg[value] = best_rows[value].values
    label = "J (re-weighted)" if (value == "fitness" and reweighted) else metric_label(value, False)

    # KPI tiles
    ok = df_sel[df_sel["status"] == "success"]
    o_rank = analysis.optimizer_ranks(agg)
    c_rank = analysis.controller_ranks(agg)
    best_opt = o_rank.groupby("optimizer", observed=True)["rank"].mean().idxmin() if not o_rank.empty else "—"
    best_ctrl = c_rank.groupby("controller", observed=True)["rank"].mean().idxmin() if not c_rank.empty else "—"
    overall = agg.loc[agg["score_mean"].idxmin()]
    T.stat_tiles([
        ("Runs", f"{len(df_sel)}", f"{len(ok)} completed · {df['seed'].nunique()} seed(s)"),
        ("Settled", f"{ok['settled'].mean():.0%}", "of all responses"),
        ("Top optimiser", str(best_opt), "lowest average rank"),
        ("Top controller", str(best_ctrl), "lowest average rank"),
        ("Lowest J overall", analysis.fmt(overall["score_mean"]),
         f"{overall['optimizer']} · {overall['controller']} · {PLANT_LABELS[str(overall['plant'])]}"),
    ], min_width=170)

    missing = [p for p in PLANT_NAMES if p not in plants_avail]
    if missing:
        T.callout(f"This result set has no runs on the <b>{', '.join(PLANT_LABELS[p] for p in missing)}</b> "
                  f"plant{'s' if len(missing) > 1 else ''}, so {'they are' if len(missing) > 1 else 'it is'} not offered here.")
    fig = charts.heatmap_grid(agg, col, plants_sel, log=log, label=label)
    T.show(fig, height=330 if len(plants_sel) == 3 else 360, key="res_heat")
    st.caption("⚠ some seeds did not settle · ✖ no seed settled · — metric undefined (never settled / never reached 90 %) · "
               "n/a combination not run (GP optimiser drives only the GP controller). Lower is better.")

    c1, c2 = st.columns(2)
    with c1:
        T.show(charts.rank_bars(o_rank, "optimizer", T.OPTIMIZER_COLORS,
                                "Optimisers — average rank within each controller × plant"), key="res_orank")
    with c2:
        T.show(charts.rank_bars(c_rank, "controller", T.CONTROLLER_COLORS,
                                "Controllers — average rank within each optimiser × plant"), key="res_crank")

    tabs = st.tabs(["Step responses", "Convergence", "Seed spread", "Runtime", "Data table"])
    with tabs[0]:
        plant = st.selectbox("Plant", plants_sel, format_func=PLANT_LABELS.get, key="res_step_plant")
        panels = []
        for ctrl in state.available(df, "controller", CTRLS, plant=plant):
            traces = []
            sub = agg[(agg["plant"].astype(str) == plant) & (agg["controller"].astype(str) == ctrl)]
            for _, row in sub.sort_values("optimizer").iterrows():
                rec = rs.records[int(row["best_idx"])]
                opt = str(row["optimizer"])
                traces.append({"t": rec["time"], "y": rec["output"], "name": opt, "group": opt,
                               "color": T.OPTIMIZER_COLORS[opt]})
            if traces:
                panels.append({"title": ctrl, "traces": traces})
        if panels:
            T.show(charts.step_grid(panels), key="res_grid")
            st.caption("Best seed of each combination. Green band = ±2 % settling tolerance.")
        else:
            state.no_data(f"the {PLANT_LABELS[plant]} plant")
    with tabs[1]:
        c1, c2, c3 = st.columns([1, 1, 1])
        plant = c1.selectbox("Plant", plants_sel, format_func=PLANT_LABELS.get, key="res_conv_plant")
        ctrl = c2.selectbox("Controller", state.available(df, "controller", CTRLS, plant=plant) or CTRLS,
                            key="res_conv_ctrl")
        all_seeds = c3.toggle("Show every seed", value=False, key="res_conv_all")
        sub = df[(df["plant"].astype(str) == plant) & (df["controller"].astype(str) == ctrl) & (df["status"] == "success")]
        traces = []
        for opt in ["DE", "GA", "PSO", "SA", "GP"]:
            s = sub[sub["optimizer"].astype(str) == opt].sort_values("fitness")
            rows = s if all_seeds else s.head(1)
            for k, (_, row) in enumerate(rows.iterrows()):
                rec = rs.records[int(row["idx"])]
                traces.append({"history": rec["convergence"], "n_evals": rec.get("n_evaluations"),
                               "name": opt if k == 0 else f"{opt} (seed {rec['seed']})", "group": opt,
                               "color": T.OPTIMIZER_COLORS[opt], "dash": "solid" if k == 0 else "dot"})
        if not traces:
            state.no_data(f"{ctrl} on the {PLANT_LABELS[plant]} plant")
        else:
            T.show(charts.convergence_figure(traces, title=f"Best-so-far objective — {ctrl} on {PLANT_LABELS[plant]}"),
                   key="res_conv")
        st.caption("x is mapped to objective evaluations so population methods (one point per generation) and SA "
                   "(one point per move) share an axis.")
    with tabs[2]:
        if df["seed"].nunique() < 2:
            st.info("This result set has a single seed. Build a multi-seed set in the Experiment Lab to see the spread.")
        else:
            c1, c2 = st.columns(2)
            plant = c2.selectbox("Plant", plants_sel, format_func=PLANT_LABELS.get, key="res_seed_plant")
            ctrl = c1.selectbox("Controller", state.available(df, "controller", CTRLS, plant=plant) or CTRLS,
                                key="res_seed_ctrl")
            if state.available(df, "optimizer", OPTS, plant=plant, controller=ctrl):
                T.show(charts.seed_strip(df, ctrl, plant, value=score_col if value == "fitness" else value,
                                         log=value in ("fitness", "overshoot", "iae", "steady_state_error")), key="res_seed")
            else:
                state.no_data(f"{ctrl} on the {PLANT_LABELS[plant]} plant")
            st.caption("Each dot is one independent run (different random seed). A tall spread means the result "
                       "depends on luck as much as on the algorithm.")
    with tabs[3]:
        T.show(charts.runtime_bars(agg), key="res_rt")
        st.caption("Wall time per run in the worker process. Fuzzy inference (25 rules, 201-point centroid per "
                   "step) dominates the cost of the matrix.")
    with tabs[4]:
        table = agg.copy()  # already limited to the selected plants
        table["plant"] = table["plant"].astype(str).map(PLANT_LABELS)
        table = table.rename(columns={"score_mean": "J mean", "score_std": "J std", "score_best": "J best",
                                      "settled_rate": "settled", "runtime": "runtime (s)"})
        show_cols = ["optimizer", "controller", "plant", "J mean", "J std", "J best", "settled", *METRIC_KEYS,
                     "runtime (s)", "runs"]
        st.dataframe(table[show_cols].sort_values(["plant", "J mean"]), hide_index=True, width="stretch",
                     column_config={"settled": st.column_config.ProgressColumn("settled", min_value=0, max_value=1,
                                                                                 format="percent")})
        raw = df.drop(columns=["idx"]).copy()
        st.download_button("Download all runs (CSV)", raw.to_csv(index=False).encode(), file_name=f"{rs.slug}.csv",
                           mime="text/csv")
