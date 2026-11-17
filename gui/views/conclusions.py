"""Findings computed from the active result set, plus limitations and future work."""
from __future__ import annotations

import numpy as np
import plotly.graph_objects as go
import streamlit as st

from gui import content
from gui.core import analysis, charts, state
from gui.core import theme as T
from gui.core.catalog import PLANT_LABELS, PLANT_NAMES

CTRLS = ["PID", "Neural Network", "Fuzzy", "GP Controller"]


def _rank_finding(summary, one: str, many: str, cells_label: str) -> tuple[str, str]:
    """Finding text from a rank summary: names ties, and flags averages built on fewer cells."""
    top = analysis.leaders(summary)
    full = int(summary["cells"].max())
    title = (f"{top[0]} is {one}" if len(top) == 1 else f"{' and '.join(top)} are tied as {many}")

    def item(name):
        n = int(summary.loc[name, "cells"])
        return f"{name} {summary.loc[name, 'mean']:.2f}" + (f" ({n} of {full} cells)" if n < full else "")

    rest = [i for i in summary.index if i not in top]
    body = (f"Average rank {summary.loc[top[0], 'mean']:.2f} across the {cells_label} where it competed"
            + (f" — then {', '.join(item(i) for i in rest)}." if rest else "."))
    partial = [t for t in top if int(summary.loc[t, "cells"]) < full]
    if partial:
        body += (f" Note: {', '.join(partial)} only competed in {int(summary.loc[partial[0], 'cells'])} of {full} "
                 "cells (the GP optimiser only drives the GP controller).")
    return title, body


def _budget_effect():
    """Median mean-J per controller for every saved result set, against its evaluation budget."""
    points = {}
    for path in state.available_sets():
        rs = state.load(path)
        ev = rs.meta.get("evaluations")
        if not ev:
            continue
        df = state.frame(path)
        agg = analysis.aggregate(df)
        # Only compare like with like: a set must cover every plant and controller.
        if (agg.empty or set(state.available(df, "plant", PLANT_NAMES)) != set(PLANT_NAMES)
                or len(state.available(df, "controller", CTRLS)) < len(CTRLS)):
            continue
        med = agg.groupby("controller", observed=True)["score_mean"].median()
        settled = agg.groupby("controller", observed=True)["settled_rate"].mean()
        # keep the most recent set per budget (list is newest first)
        points.setdefault(ev, (med, settled, rs.name))
    return dict(sorted(points.items()))


def render() -> None:
    T.page_header("Wrap-up", "What we learned",
                  "Findings below are computed from the active result set — switch the set in the sidebar and "
                  "they update.")
    rs = state.require_results()
    if rs is not None:
        df = state.frame()
        agg = analysis.aggregate(df)
        o_sum = analysis.rank_summary(analysis.optimizer_ranks(agg), "optimizer")
        c_sum = analysis.rank_summary(analysis.controller_ranks(agg), "controller")
        ok = df[df["status"] == "success"]
        settled_by_plant = ok.groupby("plant", observed=True)["settled"].mean()
        settled_by_ctrl = ok.groupby("controller", observed=True)["settled"].mean()

        cards = []
        for plant in PLANT_NAMES:
            sub = agg[agg["plant"].astype(str) == plant]
            if sub.empty:
                continue
            r = sub.loc[sub["score_mean"].idxmin()]
            cards.append(T.card(
                f"Best on {PLANT_LABELS[plant]}",
                f"<p style='font-size:1.15rem;color:{T.TEXT};font-weight:700'>{r['optimizer']} · {r['controller']}</p>"
                + T.kv([("J", analysis.fmt(r["score_mean"])), ("IAE", analysis.fmt(r["iae"])),
                        ("Ts", analysis.fmt(r["settling_time"], unit=" s")), ("OS", analysis.fmt(r["overshoot"], unit=" %"))]),
                accent=T.PLANT_COLORS[plant]))
        T.card_grid(cards, min_width=240)

        findings = []
        if not o_sum.empty:
            findings.append(_rank_finding(o_sum, "the most reliable tuner", "the most reliable tuners",
                                          "controller × plant cells"))
        if not c_sum.empty:
            title, body = _rank_finding(c_sum, "the strongest controller structure",
                                        "the strongest controller structures", "optimiser × plant cells")
            top = analysis.leaders(c_sum)
            body += " Settled runs: " + ", ".join(f"{t} {settled_by_ctrl.get(t, np.nan):.0%}" for t in top) + "."
            findings.append((title, body))
        if len(settled_by_plant) > 1:
            hardest = settled_by_plant.idxmin()
            findings.append((f"The {PLANT_LABELS[str(hardest)]} plant is the hardest",
                             f"Only {settled_by_plant.min():.0%} of its runs settle, versus "
                             f"{settled_by_plant.max():.0%} on the {PLANT_LABELS[str(settled_by_plant.idxmax())]} plant — "
                             "more phase lag leaves less room for aggressive gains."))
        if all(c in settled_by_ctrl for c in ("Neural Network", "PID", "Fuzzy")):
            findings.append(("Parameter count matters under a small budget",
                             f"The 21-weight neural network settles in {settled_by_ctrl['Neural Network']:.0%} of runs; "
                             f"3-parameter PID and Fuzzy settle in {settled_by_ctrl.get('PID', np.nan):.0%} and "
                             f"{settled_by_ctrl.get('Fuzzy', np.nan):.0%}."))
        if df["seed"].nunique() > 1:
            cv = (agg.assign(cv=agg["score_std"] / agg["score_mean"])
                  .groupby("optimizer", observed=True)["cv"].median().dropna().sort_values())
            findings.append(("Results depend on the random seed",
                             f"Median coefficient of variation of J across seeds: "
                             + ", ".join(f"{k} {v:.0%}" for k, v in cv.items())
                             + ". Single-seed rankings should be read with caution."))
        st.markdown("<h2>Key findings</h2>", unsafe_allow_html=True)
        st.markdown(T.card("", T.steps(findings)), unsafe_allow_html=True)

        effect = _budget_effect()
        if len(effect) >= 2:
            st.markdown("<h2>Does a bigger budget help?</h2>", unsafe_allow_html=True)
            fig = go.Figure()
            evs = list(effect)
            for ctrl in ["PID", "Neural Network", "Fuzzy", "GP Controller"]:
                ys = [effect[e][0].get(ctrl, np.nan) for e in evs]
                fig.add_trace(go.Scatter(x=evs, y=ys, name=ctrl, mode="lines+markers",
                                         line=dict(color=T.CONTROLLER_COLORS[ctrl], width=2),
                                         marker=dict(size=9, line=dict(color=T.SURFACE, width=1.5)),
                                         hovertemplate=f"<b>{ctrl}</b><br>%{{x}} evaluations<br>median J %{{y:.4g}}<extra></extra>"))
            fig.update_layout(height=380, xaxis_title="Objective evaluations per run (full-matrix result sets only)",
                              yaxis_title="Median J across optimisers & plants",
                              title="Median objective per controller vs evaluation budget (one point per saved result set)")
            charts.log_axis(fig, "y")
            fig.update_xaxes(type="log", tickvals=evs, ticktext=[str(e) for e in evs])
            T.show(fig, key="budget_effect")

    c1, c2 = st.columns(2)
    with c1:
        st.markdown(T.card("Limitations", "<ul>" + "".join(f"<li>{T.esc(x)}</li>" for x in content.LIMITATIONS) + "</ul>",
                           accent=T.WARN), unsafe_allow_html=True)
    with c2:
        st.markdown(T.card("Future work", "<ul>" + "".join(f"<li>{T.esc(x)}</li>" for x in content.FUTURE_WORK) + "</ul>",
                           accent=T.ACCENT), unsafe_allow_html=True)

    st.markdown(
        f"<div class='sc-hero' style='text-align:center;margin-top:1.6rem;padding:2.4rem'>"
        f"<h1 style='margin:0'><span class='grad'>Thank you</span></h1>"
        f"<p style='margin:.6rem auto 0 auto'>Questions? Everything you saw is live — ask for any combination.</p></div>",
        unsafe_allow_html=True)
