"""Head-to-head comparisons: custom picks (across result sets) and optimiser duels."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from gui.core import analysis, charts, state
from gui.core import theme as T
from gui.core.catalog import CONTROLLERS, METRIC_KEYS, PLANT_LABELS, PLANT_NAMES, is_compatible, metric_label

SLOTS = 4
CTRL_SHORT = {"PID": "PID", "Neural Network": "NN", "Fuzzy": "Fuzzy", "GP Controller": "GP-ctrl"}


def _pick_slot(i: int, sets: list[Path], default: tuple[str, str, str]) -> dict | None:
    color = T.SLOT_COLORS[i]
    st.markdown(f"<div class='sc-eyebrow' style='color:{color}'>Pick {i + 1}</div>", unsafe_allow_html=True)
    on = st.toggle("Include", value=True, key=f"cmp_on_{i}")
    if not on:
        return None
    paths = [str(p) for p in sets]
    default_set = str(state.active_path())
    rs_path = st.selectbox("Result set", paths, index=paths.index(default_set) if default_set in paths else 0,
                           format_func=lambda p: state.set_label(Path(p)), key=f"cmp_rs_{i}",
                           label_visibility="collapsed")
    df = state.frame(Path(rs_path))
    rs = state.load(Path(rs_path))
    # Cascade: only offer what this result set actually contains.
    ctrls = state.available(df, "controller", list(CONTROLLERS))
    if not ctrls:
        state.no_data("any controller")
        return None
    ctrl = st.selectbox("Controller", ctrls, index=ctrls.index(default[1]) if default[1] in ctrls else 0,
                        key=f"cmp_ctrl_{i}")
    opts = state.available(df, "optimizer", ["DE", "GA", "PSO", "SA", "GP"], controller=ctrl)
    opt = st.selectbox("Optimiser", opts, index=opts.index(default[0]) if default[0] in opts else 0, key=f"cmp_opt_{i}")
    plants_ = state.available(df, "plant", PLANT_NAMES, controller=ctrl, optimizer=opt)
    plant = st.selectbox("Plant", plants_, index=plants_.index(default[2]) if default[2] in plants_ else 0,
                         format_func=PLANT_LABELS.get, key=f"cmp_plant_{i}")
    sub = df[(df["optimizer"].astype(str) == opt) & (df["controller"].astype(str) == ctrl)
             & (df["plant"].astype(str) == plant) & (df["status"] == "success")]
    if sub.empty:
        state.no_data(f"{opt} · {ctrl} · {PLANT_LABELS.get(plant, plant)}")
        return None
    row = sub.sort_values("fitness").iloc[0]
    rec = rs.records[int(row["idx"])]
    short_set = "" if rs_path == default_set else f" [{rs.meta.get('budget_label', rs.name).split(' (')[0]}]"
    return {"rec": rec, "color": color, "dash": T.SLOT_DASHES[i], "slot": i,
            "name": f"{opt} · {ctrl} · {PLANT_LABELS[plant]}{short_set}", "short": f"#{i + 1}",
            "legend": f"#{i + 1} {opt}·{CTRL_SHORT[ctrl]}{short_set}",
            "seeds": len(sub), "mean_J": float(sub["fitness"].mean())}


def _picks_mode() -> None:
    sets = state.available_sets()
    defaults = [("PSO", "PID", "third_order"), ("GA", "Fuzzy", "third_order"),
                ("PSO", "Neural Network", "third_order"), ("GP", "GP Controller", "third_order")]
    cols = st.columns(SLOTS)
    picks = []
    for i, col in enumerate(cols):
        with col:
            with st.container(border=True):
                p = _pick_slot(i, sets, defaults[i])
                if p:
                    picks.append(p)
    if not picks:
        st.info("Include at least one pick.")
        return
    st.caption("Tip: pick the same combination from two result sets to see what a larger evaluation budget buys.")

    traces = [{"t": p["rec"]["time"], "y": p["rec"]["output"], "u": p["rec"]["control"], "name": p["legend"],
               "short": p["short"], "color": p["color"], "dash": p["dash"], "group": p["name"]} for p in picks]
    c1, c2 = st.columns([3, 2])
    with c1:
        T.show(charts.step_figure(traces, title="Step response (best seed)", height=440), key="cmp_step")
    with c2:
        T.show(charts.control_figure(traces, height=440), key="cmp_u")

    # Scorecard
    rows, keys = [], ["fitness", *METRIC_KEYS, "execution_time"]
    for p in picks:
        rec = p["rec"]
        vals = {"fitness": rec["fitness"], "execution_time": rec["execution_time"], **rec["metrics"]}
        rows.append({"name": p["name"], "short": p["legend"], "color": p["color"], "values": vals})
    best = {}
    for k in keys:
        col = [r["values"].get(k) for r in rows if r["values"].get(k) is not None and math.isfinite(r["values"].get(k))]
        best[k] = min(col) if col else None
    head = "".join(f"<th style='text-align:right;padding:.5rem .8rem;color:{T.MUTED};font-weight:600'>"
                   f"{T.esc(metric_label(k))}</th>" for k in keys)
    body = ""
    for r in rows:
        cells = ""
        for k in keys:
            v = r["values"].get(k)
            is_best = v is not None and best[k] is not None and math.isfinite(v) and v == best[k] and len(rows) > 1
            style = f"color:{T.TEXT};font-weight:800" if is_best else f"color:{T.TEXT_2}"
            mark = " ★" if is_best else ""
            cells += f"<td style='text-align:right;padding:.5rem .8rem;{style};font-variant-numeric:tabular-nums'>{analysis.fmt(v)}{mark}</td>"
        body += (f"<tr style='border-top:1px solid {T.BORDER}'><td style='padding:.5rem .8rem;color:{T.TEXT};white-space:nowrap'>"
                 f"<span class='sc-dot' style='background:{r['color']};margin-right:.5rem'></span><b>{T.esc(r['short'].split(' ')[0])}</b>&nbsp; {T.esc(r['name'])}</td>{cells}</tr>")
    st.markdown(f"<div style='overflow-x:auto;background:{T.SURFACE};border:1px solid {T.BORDER};border-radius:14px;"
                f"margin:.6rem 0 1rem 0'><table style='width:100%;border-collapse:collapse;font-size:.92rem'>"
                f"<thead><tr><th style='text-align:left;padding:.5rem .8rem;color:{T.MUTED};font-weight:600'>Pick</th>{head}</tr></thead>"
                f"<tbody>{body}</tbody></table></div>", unsafe_allow_html=True)
    st.caption("★ best on that column (lower is better). '—' means the metric is undefined (e.g. never settled).")

    c1, c2 = st.columns(2)
    with c1:
        T.show(charts.metric_bars(rows, ["fitness", *METRIC_KEYS]), key="cmp_bars")
    with c2:
        conv = [{"history": p["rec"]["convergence"], "n_evals": p["rec"].get("n_evaluations"), "name": p["legend"],
                 "color": p["color"], "dash": p["dash"]} for p in picks]
        T.show(charts.convergence_figure(conv, title="Convergence", height=340), key="cmp_conv")


def _duel_mode() -> None:
    rs = state.require_results()
    if rs is None:
        return
    agg = analysis.aggregate(state.frame())
    c1, c2, _ = st.columns([1, 1, 2])
    a = c1.selectbox("Optimiser A", ["DE", "GA", "PSO", "SA"], index=2, key="duel_a")
    b = c2.selectbox("Optimiser B", [o for o in ["DE", "GA", "PSO", "SA"] if o != a], index=0, key="duel_b")
    wm = analysis.win_matrix(agg, a, b)
    if wm.empty:
        st.info("No overlapping cells.")
        return
    wins_a, wins_b = int((wm["winner"] == a).sum()), int((wm["winner"] == b).sum())
    ties = int((wm["winner"] == "tie").sum())
    T.stat_tiles([(f"{a} wins", f"{wins_a}", f"of {len(wm)} controller × plant cells"),
                  (f"{b} wins", f"{wins_b}", f"of {len(wm)} cells"),
                  ("Ties", f"{ties}", "identical mean J"),
                  ("Median J ratio", f"{np.median(wm['ratio']):.2f}×", f"{b} ÷ {a} (above 1 favours {a})")])

    wm = wm.copy()
    wm["cell"] = wm["controller"].astype(str) + " · " + wm["plant"].astype(str).map(PLANT_LABELS)
    fig = go.Figure()
    for _, r in wm.iterrows():
        fig.add_trace(go.Scatter(x=[r[a], r[b]], y=[r["cell"], r["cell"]], mode="lines",
                                 line=dict(color=T.AXIS, width=3), hoverinfo="skip", showlegend=False))
    for name in (a, b):
        fig.add_trace(go.Scatter(x=wm[name], y=wm["cell"], mode="markers", name=name,
                                 marker=dict(size=13, color=T.OPTIMIZER_COLORS[name], line=dict(color=T.SURFACE, width=2)),
                                 hovertemplate=f"<b>{name}</b> · %{{y}}<br>mean J = %{{x:.4g}}<extra></extra>"))
    fig.update_layout(height=max(360, 34 * len(wm) + 80), title=f"Mean objective per cell — {a} vs {b} (left is better)",
                      xaxis_title="Mean objective J (log)")
    fig.update_xaxes(type="log", dtick="D2")
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    T.show(fig, key="duel_fig")


def render() -> None:
    T.page_header("Comparison", "Head-to-head",
                  "Put any runs side by side — even from different result sets — or let two optimisers duel "
                  "across every controller and plant.")
    if not state.available_sets():
        state.require_results()
        return
    mode = st.segmented_control("Mode", ["Custom picks", "Optimiser duel"], default="Custom picks",
                                key="cmp_mode", label_visibility="collapsed")
    if mode == "Optimiser duel":
        _duel_mode()
    else:
        _picks_mode()
