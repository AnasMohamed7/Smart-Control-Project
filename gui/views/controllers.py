"""The four controller structures, each with a visual built from real tuned values."""
from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st

from gui.core import analysis, charts, engine, state
from gui.core import theme as T
from gui.core.catalog import CONTROLLER_INFO, PLANT_LABELS, PLANT_NAMES, parameter_names, plants
from smart_control.controllers.fuzzy import FuzzyController


@st.cache_data(show_spinner=False)
def fuzzy_surface_data(n: int = 41):
    ctrl = FuzzyController({"K_e": 1.0, "K_de": 1.0, "K_u": 1.0})
    e = np.linspace(-1, 1, n)
    de = np.linspace(-1, 1, n)
    U = np.zeros((n, n))
    for i, dv in enumerate(de):
        for j, ev in enumerate(e):
            ctrl.reset()
            U[i, j] = ctrl.compute(ev, 0.01, error_rate=dv)
    return e, de, U


def _intro(name: str) -> None:
    info = CONTROLLER_INFO[name]
    c1, c2 = st.columns([1.35, 1])
    with c1:
        st.markdown(f"<p style='font-size:1.05rem;color:{T.TEXT}'>{T.esc(info['summary'])}</p>", unsafe_allow_html=True)
        st.latex(info["latex"])
    with c2:
        body = (f"<p class='tag'>{T.esc(info['params'])}</p><ul>"
                + "".join(f"<li>{T.esc(d)}</li>" for d in info["details"]) + "</ul>")
        st.markdown(T.card("Design", body, accent=T.CONTROLLER_COLORS[name]), unsafe_allow_html=True)


def _best_record(df: pd.DataFrame, rs, controller: str, plant: str, optimizer: str | None = None):
    sub = df[(df["controller"].astype(str) == controller) & (df["plant"].astype(str) == plant)
             & (df["status"] == "success")]
    if optimizer:
        sub = sub[sub["optimizer"].astype(str) == optimizer]
    if sub.empty:
        return None
    return rs.records[int(sub.loc[sub["fitness"].idxmin(), "idx"])]


def _has_runs(df: pd.DataFrame, controller: str) -> bool:
    """True if the active set has runs for this controller; otherwise shows a full-width empty state."""
    if state.available(df, "plant", PLANT_NAMES, controller=controller):
        return True
    state.no_data(f"the {controller} controller")
    return False


def _plant_picker(df: pd.DataFrame, controller: str, key: str) -> str:
    """Plant selectbox limited to plants this controller was run on (call after ``_has_runs``)."""
    options = state.available(df, "plant", PLANT_NAMES, controller=controller)
    return st.selectbox("Plant", options, format_func=PLANT_LABELS.get, key=key)


def _param_table(df: pd.DataFrame, rs, controller: str) -> None:
    rows = []
    for plant in PLANT_NAMES:
        for opt in ["DE", "GA", "PSO", "SA", "GP"]:
            rec = _best_record(df, rs, controller, plant, opt)
            if rec is None:
                continue
            names = parameter_names(controller, len(rec["best_solution"]))
            row = {"Plant": PLANT_LABELS[plant], "Optimiser": opt, "J": round(rec["fitness"], 4) if rec["fitness"] else None}
            row.update({n: round(v, 4) for n, v in zip(names, rec["best_solution"])})
            row["Settled"] = "✔" if rec["settled"] else "✖"
            rows.append(row)
    if rows:
        st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    else:
        st.caption(f"No {controller} runs in the active result set.")


def _pid_tab(df, rs) -> None:
    _intro("PID")
    if rs is None:
        return
    st.markdown("<h3>What each term contributes</h3>", unsafe_allow_html=True)
    if not _has_runs(df, "PID"):
        return
    c1, c2 = st.columns([1, 3])
    with c1:
        plant = _plant_picker(df, "PID", "pid_plant")
        rec = _best_record(df, rs, "PID", plant) if plant else None
        if rec is None:
            return
        kp, ki, kd = rec["best_solution"]
        st.markdown(T.kv([("source", f"best run ({rec['optimizer']})"), ("Kp", f"{kp:.3f}"),
                          ("Ki", f"{ki:.3f}"), ("Kd", f"{kd:.3f}")]), unsafe_allow_html=True)
        st.caption("Removing terms from the tuned controller shows why all three are there.")
    with c2:
        variants = [("P only", [kp, 0, 0], T.SLOT_COLORS[0], "dot"), ("PI", [kp, ki, 0], T.SLOT_COLORS[1], "dash"),
                    ("PD", [kp, 0, kd], T.SLOT_COLORS[2], "dashdot"), ("PID (tuned)", [kp, ki, kd], T.CONTROLLER_COLORS["PID"], "solid")]
        traces = []
        for label, params, color, dash in variants:
            sim, m = engine.simulate("PID", params, plants()[plant])
            traces.append({"t": sim.time, "y": sim.output, "name": f"{label} · IAE {analysis.fmt(m.iae)}",
                           "short": label, "color": color, "dash": dash, "width": 2.6 if label.startswith("PID") else 2})
        T.show(charts.step_figure(traces, title=f"Ablation on {PLANT_LABELS[plant]}"), key="pid_ablation")
    with st.expander("Tuned PID gains for every optimiser and plant"):
        _param_table(df, rs, "PID")


def _nn_tab(df, rs) -> None:
    _intro("Neural Network")
    params = None
    if rs is not None and _has_runs(df, "Neural Network"):
        c1, c2 = st.columns([1, 3])
        with c1:
            plant = _plant_picker(df, "Neural Network", "nn_plant")
            rec = None
            if plant:
                opts = state.available(df, "optimizer", ["DE", "GA", "PSO", "SA"], controller="Neural Network", plant=plant)
                opt = st.selectbox("Optimiser", ["best", *opts], key="nn_opt")
                rec = _best_record(df, rs, "Neural Network", plant, None if opt == "best" else opt)
            if rec is not None:
                params = rec["best_solution"]
                st.markdown(T.kv([("source", rec["optimizer"]), ("J", analysis.fmt(rec["fitness"])),
                                  ("settled", "yes" if rec["settled"] else "no")]), unsafe_allow_html=True)
                st.caption("Blue edges are negative weights, red positive; thicker means larger |w|.")
        with c2:
            T.show(charts.nn_diagram(params), key="nn_diag")
    else:
        T.show(charts.nn_diagram(None), key="nn_diag_empty")
    T.callout("21 free parameters versus 3 for PID: a much larger search space for the same evaluation "
              "budget. Expect the NN to need many more evaluations before it competes.", "warn")


def _fuzzy_tab(df, rs) -> None:
    _intro("Fuzzy")
    c1, c2 = st.columns([1, 1])
    with c1:
        T.show(charts.fuzzy_memberships(), key="fz_mf")
        T.show(charts.fuzzy_rule_table(FuzzyController._RULES), key="fz_rules")
    with c2:
        e, de, U = fuzzy_surface_data()
        T.show(charts.fuzzy_surface(e, de, U, height=680), key="fz_surface")
    T.callout("The rule base acts like a smooth, saturating PD law: large error or fast error growth push u "
              "to its extremes, and the ZE diagonal holds it near zero. The optimiser only stretches the axes "
              "(K_e, K_de) and the output (K_u).")
    if rs is not None:
        with st.expander("Tuned fuzzy scaling gains for every optimiser and plant"):
            _param_table(df, rs, "Fuzzy")


def _gp_tab(df, rs) -> None:
    _intro("GP Controller")
    if rs is None or not _has_runs(df, "GP Controller"):
        return
    c1, c2 = st.columns([1, 3])
    with c1:
        plant = _plant_picker(df, "GP Controller", "gp_plant")
        opts = state.available(df, "optimizer", ["GP", "DE", "GA", "PSO", "SA"], controller="GP Controller",
                               plant=plant) if plant else []
        opt = st.selectbox("Evolved by", opts, key="gp_opt") if opts else None
        rec = _best_record(df, rs, "GP Controller", plant, opt) if opt else None
    if rec is None:
        return
    from smart_control.gp_tree import decode_vector, tree_depth, tree_size, uses_variable
    tree = decode_vector(rec["best_solution"])
    with c1:
        st.markdown(T.kv([("J", analysis.fmt(rec["fitness"])), ("depth", str(tree_depth(tree))),
                          ("nodes", str(tree_size(tree))), ("uses e", "yes" if uses_variable(tree, "e") else "no"),
                          ("settled", "yes" if rec["settled"] else "no")]), unsafe_allow_html=True)
    with c2:
        T.show(charts.gp_tree_figure(tree, title=f"u = {rec['expression']}"), key="gp_tree")
    rows = []
    for p in PLANT_NAMES:
        for o in ["GP", "DE", "GA", "PSO", "SA"]:
            r = _best_record(df, rs, "GP Controller", p, o)
            if r is not None:
                rows.append({"Plant": PLANT_LABELS[p], "Optimiser": o, "Evolved law u =": r["expression"],
                             "J": round(r["fitness"], 3), "Settled": "✔" if r["settled"] else "✖"})
    st.markdown("<h3>Every evolved control law</h3>", unsafe_allow_html=True)
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("ie is the clamped integral of the error, de its derivative. The numeric optimisers search the "
               "same 30-gene tree encoding as the GP optimiser, but without tree-aware crossover.")


def render() -> None:
    T.page_header("Controllers", "Four ways to compute u(t) from the error",
                  "All four implement the same interface — reset · set_parameters(θ) · compute(e, dt) · "
                  "parameter_bounds — so any optimiser can tune any controller without special cases.")
    rs = state.load()
    df = state.frame() if rs is not None else pd.DataFrame()
    T.card_grid([T.card(name, f"<p>{T.esc(info['params'])}</p>", accent=T.CONTROLLER_COLORS[name])
                 for name, info in CONTROLLER_INFO.items()], min_width=220)
    tabs = st.tabs(["PID", "Neural Network", "Fuzzy Logic", "Genetic Programming"])
    with tabs[0]:
        _pid_tab(df, rs)
    with tabs[1]:
        _nn_tab(df, rs)
    with tabs[2]:
        _fuzzy_tab(df, rs)
    with tabs[3]:
        _gp_tab(df, rs)
