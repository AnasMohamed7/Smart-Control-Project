"""How well do the tuned controllers cope when the real plant differs from the model?"""
from __future__ import annotations

import math

import control
import numpy as np
import pandas as pd
import streamlit as st

from gui.core import analysis, charts, engine, state
from gui.core import theme as T
from gui.core.catalog import METRIC_KEYS, PLANT_LABELS, PLANT_NAMES, metric_label, plants

PERTURBATIONS = {
    "Plant gain": "G'(s) = k · G(s) — e.g. actuator or sensor gain error",
    "Plant speed": "G'(s) = G(s / k) — every time constant divided by k (k > 1 = faster plant)",
}


def perturbed(plant: str, kind: str, k: float):
    tf = plants()[plant]
    num = np.asarray(tf.num[0][0], float)
    den = np.asarray(tf.den[0][0], float)
    if kind == "Plant gain":
        return control.TransferFunction(num * k, den)
    n = len(den) - 1
    den2 = np.array([a * k ** i for i, a in enumerate(den)])
    m = len(num) - 1
    num2 = np.array([b * k ** (n - m + i) for i, b in enumerate(num)])
    return control.TransferFunction(num2, den2)


@st.cache_data(show_spinner=False, max_entries=64)
def sweep(controller: str, params: tuple, plant: str, kind: str, factors: tuple):
    out = []
    for k in factors:
        sim, m = engine.simulate(controller, list(params), perturbed(plant, kind, k))
        d = m.to_dict()
        d["fitness"] = analysis.rescore(pd.DataFrame([d]), analysis.effective_weights()).iloc[0]
        d["t"], d["y"] = sim.time, sim.output
        out.append(d)
    return out


def render() -> None:
    T.page_header("Robustness", "What if the real plant is not the model?",
                  "Controllers were tuned on a nominal model. Here the tuned parameters are kept fixed and the "
                  "plant is perturbed — a controller that only works on the exact model is fragile.")
    rs = state.require_results()
    if rs is None:
        return
    df = state.frame()
    agg = analysis.aggregate(df)

    c1, c2, c3 = st.columns([1, 1, 1.2])
    plants_ = state.available(df, "plant", PLANT_NAMES)
    if not plants_:
        state.no_data("any tuned controller")
        return
    plant = c1.selectbox("Nominal plant", plants_, index=plants_.index("third_order") if "third_order" in plants_ else 0,
                         format_func=PLANT_LABELS.get, key="rob_plant")
    kind = c2.selectbox("Perturbation", list(PERTURBATIONS), key="rob_kind")
    lo, hi = c3.slider("Range of k", 0.3, 3.0, (0.5, 2.0), 0.1, key="rob_range")
    st.caption(PERTURBATIONS[kind])

    sub = agg[agg["plant"].astype(str) == plant].sort_values("score_mean")
    options = [f"{r.optimizer} · {r.controller}" for r in sub.itertuples()]
    default = []
    for ctrl in ["PID", "Neural Network", "Fuzzy", "GP Controller"]:
        first = sub[sub["controller"].astype(str) == ctrl].head(1)
        if not first.empty:
            default.append(f"{first['optimizer'].iloc[0]} · {ctrl}")
    picks = st.multiselect("Tuned controllers (best seed of each)", options, default=default[:4], max_selections=4,
                           key=f"rob_picks_{plant}")
    metric = st.segmented_control("Measure", ["fitness", "iae", "overshoot", "settling_time"], default="iae",
                                  format_func=lambda k: metric_label(k, False), key="rob_metric") or "iae"
    if not picks:
        st.info("Pick at least one controller.")
        return

    factors = tuple(np.round(np.linspace(lo, hi, 9), 3))
    if not any(abs(f - 1.0) < 1e-9 for f in factors):
        factors = tuple(sorted(factors + (1.0,)))
    series, extremes = [], []
    with st.spinner("Re-simulating on perturbed plants…"):
        for i, pick in enumerate(picks):
            opt, ctrl = pick.split(" · ", 1)
            row = sub[(sub["optimizer"].astype(str) == opt) & (sub["controller"].astype(str) == ctrl)].iloc[0]
            rec = rs.records[int(row["best_idx"])]
            res = sweep(ctrl, tuple(rec["best_solution"]), plant, kind, factors)
            color = T.SLOT_COLORS[i]
            ys = [r[metric] if r[metric] is not None and math.isfinite(r[metric]) else None for r in res]
            series.append({"x": list(factors), "y": ys, "name": pick, "color": color, "dash": T.SLOT_DASHES[i]})
            settled = sum(1 for r in res if r["settling_time"] is not None)
            worst = max((v for v in ys if v is not None), default=None)
            nominal = ys[list(factors).index(1.0)]
            extremes.append({"pick": pick, "color": color, "dash": T.SLOT_DASHES[i], "res": res, "settled": settled,
                             "worst": worst, "nominal": nominal})

    fig = charts.robustness_figure(series, metric)
    if metric in ("fitness", "iae", "overshoot"):
        fig.update_yaxes(type="log", dtick="D2")
    T.show(fig, key="rob_fig")

    T.stat_tiles([(e["pick"], f"{e['settled']}/{len(factors)}",
                   f"settled · worst {metric_label(metric, False)} {analysis.fmt(e['worst'])} "
                   f"(nominal {analysis.fmt(e['nominal'])})") for e in extremes], min_width=220)

    st.markdown("<h3>Responses at the edges of the range</h3>", unsafe_allow_html=True)
    panels = []
    for label, idx in [(f"k = {factors[0]:g}", 0), ("nominal (k = 1)", list(factors).index(1.0)),
                       (f"k = {factors[-1]:g}", len(factors) - 1)]:
        panels.append({"title": label, "traces": [
            {"t": e["res"][idx]["t"], "y": e["res"][idx]["y"], "name": e["pick"], "group": e["pick"],
             "color": e["color"], "dash": e["dash"]} for e in extremes]})
    T.show(charts.step_grid(panels, cols=3, height=380), key="rob_edges")
    T.callout("A flat curve means the controller tolerates model error. Controllers tuned to the edge of "
              "stability on the nominal model (high gain, little phase margin) degrade fastest — exactly the "
              "trade-off an objective that only rewards speed will push the optimiser into.")
