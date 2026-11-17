"""The five performance criteria and how they become one objective."""
from __future__ import annotations

import streamlit as st

from gui.core import analysis, charts, engine, state
from gui.core import theme as T
from gui.core.catalog import CONTROLLERS, METRICS, PLANT_LABELS, PLANT_NAMES, plants


def render() -> None:
    T.page_header("Evaluation", "Five metrics, one number to minimise",
                  "Each candidate θ is simulated once; the response is measured on five classic criteria, "
                  "which are folded into a single objective J for the optimiser.")

    defs = {
        "rise_time": ("#86b6ef", "Time from 10 % to 90 % of the step (linear interpolation between samples)."),
        "settling_time": (T.GOOD, "First time after which y(t) stays within ±2 % of r for the rest of the run."),
        "overshoot": ("#e87ba4", "How far the peak exceeds r, as a percentage of the step."),
        "steady_state_error": (T.WARN, "|r − y| averaged over the final 5 % of samples."),
        "iae": (T.ACCENT, "∫|e(t)| dt by the trapezoid rule — speed and accuracy in one number."),
    }
    T.card_grid([T.card(METRICS[k][0], f"<p>{T.esc(d)}</p>", accent=c) for k, (c, d) in defs.items()], min_width=165)

    st.markdown("<h2>See them on a real response</h2>", unsafe_allow_html=True)
    rs = state.load()
    c1, c2 = st.columns([1, 3])
    with c1:
        source = st.radio("Response", ["Tuned result", "Manual PID"], horizontal=True, key="met_src",
                          disabled=rs is None, index=1)
        tuned = None
        if source == "Tuned result" and rs is not None:
            df = state.frame()
            plants_ = state.available(df, "plant", PLANT_NAMES)
            if plants_:
                plant = st.selectbox("Plant", plants_, format_func=PLANT_LABELS.get, key="met_plant_tuned")
                ctrl = st.selectbox("Controller", state.available(df, "controller", list(CONTROLLERS), plant=plant),
                                    key="met_ctrl")
                opt = st.selectbox("Optimiser", state.available(df, "optimizer", ["DE", "GA", "PSO", "SA", "GP"],
                                                                plant=plant, controller=ctrl), key="met_opt")
                sub = df[(df["plant"].astype(str) == plant) & (df["controller"].astype(str) == ctrl)
                         & (df["optimizer"].astype(str) == opt) & (df["status"] == "success")]
                if not sub.empty:
                    tuned = rs.records[int(sub.sort_values("fitness").iloc[0]["idx"])]
            if tuned is None:
                state.no_data("a tuned response", "Switch to “Manual PID” to explore the metrics anyway.")
        if tuned is not None:
            t, y, m = tuned["time"], tuned["output"], tuned["metrics"]
        elif source == "Tuned result":
            t = None
        else:
            plant = st.selectbox("Plant", PLANT_NAMES, format_func=PLANT_LABELS.get, key="met_plant")
            kp = st.slider("Kp", 0.0, 20.0, 1.5, 0.1, key="met_kp")
            ki = st.slider("Ki", 0.0, 5.0, 0.9, 0.05, key="met_ki")
            kd = st.slider("Kd", 0.0, 5.0, 0.3, 0.05, key="met_kd")
            sim, pm = engine.simulate("PID", [kp, ki, kd], plants()[plant])
            t, y, m = sim.time, sim.output, pm.to_dict()
        show = {}
        st.markdown("<div style='height:.4rem'></div>", unsafe_allow_html=True)
        for key, label in [("rise", "Rise time"), ("settle", "Settling band"), ("overshoot", "Overshoot"),
                           ("sse", "Steady-state error"), ("iae", "IAE area")]:
            show[key] = st.checkbox(label, True, key=f"met_show_{key}")
    with c2:
        if t is not None:
            T.show(charts.annotated_metrics(t, y, m, show=show), key="met_fig")

    st.markdown("<h2>The objective function</h2>", unsafe_allow_html=True)
    w = analysis.effective_weights()
    c1, c2 = st.columns([1.3, 1])
    with c1:
        st.latex(r"J(\theta) = w_{IAE}\,\mathrm{IAE} + w_{OS}\,\%OS + w_{T_s}\,T_s + w_{e}\,E_{ss} + w_{T_r}\,T_r")
        st.latex(rf"w = \big({w['iae']:g},\ {w['overshoot']:g},\ {w['settling_time']:g},\ "
                 rf"{w['steady_state_error']:g},\ {w['rise_time']:g}\big)")
        st.markdown(T.steps([
            ("Never settled?", f"T<sub>s</sub> is replaced by 1.5·T + 50 = <b>{1.5 * 20 + 50:g} s</b> — a heavy, finite penalty."),
            ("Never reached 90 %?", f"T<sub>r</sub> is replaced by 1.5·T + 20 = <b>{1.5 * 20 + 20:g} s</b>."),
            ("Diverged / NaN?", "J = 10⁴. The optimiser always receives a finite number and simply moves on."),
        ]), unsafe_allow_html=True)
    with c2:
        T.callout("<b>A design choice worth noting:</b> the terms are not normalised. Overshoot is in percent "
                  "and settling time in seconds, so with these weights a 10 % overshoot costs as much as "
                  "16.7 s of settling time. The <b>Results Dashboard</b> lets you change the weights and "
                  "re-rank every run instantly.", "warn")
        T.callout("Every optimiser minimises the <i>same</i> J with the <i>same</i> simulator and budget — "
                  "the comparison is fair by construction.", "good")
