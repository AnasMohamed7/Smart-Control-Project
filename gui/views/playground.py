"""Hand-tune a PID or fuzzy controller live and race the optimiser's best result."""
from __future__ import annotations

import streamlit as st

from gui.core import analysis, charts, engine, state
from gui.core import theme as T
from gui.core.catalog import PLANT_LABELS, PLANT_NAMES, plants

SPECS = {
    "PID": [("Kp", 0.0, 50.0, 2.0, 0.1), ("Ki", 0.0, 20.0, 0.8, 0.05), ("Kd", 0.0, 10.0, 0.4, 0.05)],
    "Fuzzy": [("K_e", 0.1, 20.0, 2.0, 0.1), ("K_de", 0.001, 5.0, 0.1, 0.01), ("K_u", 0.1, 50.0, 10.0, 0.1)],
}


def _best(controller: str, plant: str):
    rs = state.load()
    if rs is None:
        return None
    df = state.frame()
    sub = df[(df["controller"].astype(str) == controller) & (df["plant"].astype(str) == plant) & (df["status"] == "success")]
    if sub.empty:
        return None
    return rs.records[int(sub.loc[sub["fitness"].idxmin(), "idx"])]


def _load_values(controller: str, values) -> None:
    for (name, lo, hi, _, _), v in zip(SPECS[controller], values):
        st.session_state[f"pg_{controller}_{name}"] = float(min(max(v, lo), hi))


def render() -> None:
    T.page_header("Live", "Tuning playground — can you beat the optimiser?",
                  "Move the sliders and the loop is re-simulated instantly with the same simulator and metrics "
                  "the optimisers used. The dashed trace is the best tuned result from the active result set.")
    c1, c2 = st.columns([1, 3])
    with c1:
        controller = st.segmented_control("Controller", list(SPECS), default="PID", key="pg_ctrl") or "PID"
        plant = st.selectbox("Plant", PLANT_NAMES, format_func=PLANT_LABELS.get, key="pg_plant")
        best = _best(controller, plant)
        for name, lo, hi, default, step in SPECS[controller]:
            key = f"pg_{controller}_{name}"
            if key not in st.session_state:
                st.session_state[key] = default
            st.slider(name, lo, hi, step=step, key=key)
        if best is None:
            st.caption(f"No tuned {controller} run on this plant in the active result set — "
                       "you can still tune by hand, there is just no optimiser trace to race.")
        b1, b2 = st.columns(2)
        b1.button("Load best", disabled=best is None, width="stretch", key="pg_load", help="Load the optimiser's best gains for this controller and plant",
                  on_click=_load_values, args=(controller, best["best_solution"] if best else []))
        b2.button("Reset", width="stretch", key="pg_reset", type="tertiary",
                  on_click=_load_values, args=(controller, [s[3] for s in SPECS[controller]]))
    params = [st.session_state[f"pg_{controller}_{s[0]}"] for s in SPECS[controller]]
    sim, m = engine.simulate(controller, params, plants()[plant])
    md = m.to_dict()
    import pandas as pd
    j = float(analysis.rescore(pd.DataFrame([md]), analysis.effective_weights()).iloc[0])

    with c2:
        traces = [{"t": sim.time, "y": sim.output, "u": sim.control_signal, "name": "your tuning", "short": "you",
                   "color": T.SLOT_COLORS[0], "width": 2.6}]
        if best is not None:
            traces.append({"t": best["time"], "y": best["output"], "u": best["control"],
                           "name": f"optimiser ({best['optimizer']})", "short": best["optimizer"],
                           "color": T.SLOT_COLORS[1], "dash": "dash"})
        T.show(charts.step_figure(traces, title=f"{controller} on {PLANT_LABELS[plant]}", height=400), key="pg_step")
        T.show(charts.control_figure(traces, height=230), key="pg_u")

    def tile(label, mine, theirs, unit=""):
        if theirs is None or mine is None:
            note = "optimiser: " + analysis.fmt(theirs, unit=unit)
        else:
            better = mine <= theirs
            color = T.GOOD if better else T.BAD
            arrow = "▼" if mine < theirs else ("▲" if mine > theirs else "=")
            note = (f"<span style='color:{color};font-weight:700'>{arrow} {'better' if better else 'worse'}</span> "
                    f"· optimiser {analysis.fmt(theirs, unit=unit)}")
        return (label, analysis.fmt(mine, unit=unit), note)

    bm = best["metrics"] if best else {}
    T.stat_tiles([
        tile("Objective J", j, best["fitness"] if best else None),
        tile("IAE", md["iae"], bm.get("iae")),
        tile("Settling time", md["settling_time"], bm.get("settling_time"), " s"),
        tile("Overshoot", md["overshoot"], bm.get("overshoot"), " %"),
        tile("Rise time", md["rise_time"], bm.get("rise_time"), " s"),
    ], min_width=170)
    if best is not None and j < best["fitness"]:
        T.callout("🎉 <b>You beat the optimiser on this budget.</b> With only a handful of evaluations, the "
                  "metaheuristics haven't converged — a strong argument for a larger budget.", "good")
