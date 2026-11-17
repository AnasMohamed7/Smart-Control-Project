"""Run new experiments live and manage saved result sets."""
from __future__ import annotations

import os
import time
from datetime import datetime

import pandas as pd
import streamlit as st

from gui.core import analysis, engine, state
from gui.core import theme as T
from gui.core.catalog import CONTROLLERS, OPTIMIZERS, PLANT_LABELS, PLANT_NAMES, is_compatible, metric_label

# Measured cost of one 20 s closed-loop simulation (2 000 steps) per worker,
# with all cores busy (from the precomputed result sets).
SIM_COST = {"Fuzzy": 0.72}
DEFAULT_SIM_COST = 0.052


def _estimate(jobs: list[engine.Job], workers: int) -> tuple[float, float]:
    costs = []
    for j in jobs:
        s = dict(j.optimizer_settings)
        evals = (s.get("max_iterations", 1) + 1) if j.optimizer == "SA" else s["population_size"] * (1 + s["max_iterations"])
        costs.append((evals + 1) * SIM_COST.get(j.controller, DEFAULT_SIM_COST))
    cpu = sum(costs)
    wall = max(cpu / max(workers, 1), max(costs, default=0))
    return cpu, wall


def _builder() -> None:
    flash = st.session_state.pop("lab_flash", None)
    if flash:
        st.success(flash)
    st.markdown("<h2>Build a result set</h2>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    opts = c1.multiselect("Optimisers", list(OPTIMIZERS), default=list(OPTIMIZERS), key="lab_opts")
    ctrls = c2.multiselect("Controllers", list(CONTROLLERS), default=list(CONTROLLERS), key="lab_ctrls")
    plants_ = c3.multiselect("Plants", PLANT_NAMES, default=PLANT_NAMES, format_func=PLANT_LABELS.get, key="lab_plants")

    c1, c2, c3, c4 = st.columns([1.6, 1, 1, 1])
    preset = c1.selectbox("Evaluation budget", [*engine.BUDGETS, "Custom"], index=1, key="lab_budget")
    if preset == "Custom":
        pop = c2.number_input("Population", 2, 100, 10, key="lab_pop")
        iters = c3.number_input("Generations", 1, 200, 5, key="lab_iters")
    else:
        pop, iters = engine.BUDGETS[preset]
        c2.metric("Population", pop)
        c3.metric("Generations", iters)
    c4.metric("Evaluations / run", pop * (1 + iters))

    c1, c2, c3 = st.columns([1, 1, 2])
    n_seeds = c1.number_input("Independent seeds", 1, 20, 1, key="lab_nseeds")
    seed0 = c2.number_input("First seed", 0, 10_000, 42, key="lab_seed0")
    workers = c3.slider("Parallel workers", 1, os.cpu_count() or 1, os.cpu_count() or 1, key="lab_workers")
    seeds = list(range(int(seed0), int(seed0) + int(n_seeds)))

    base_w = analysis.effective_weights()
    with st.expander("Objective weights used for tuning"):
        cols = st.columns(5)
        weights = {k: cols[i].number_input(metric_label(k, False), 0.0, 10.0, float(base_w[k]), 0.05, key=f"lab_w_{k}")
                   for i, k in enumerate(["iae", "overshoot", "settling_time", "steady_state_error", "rise_time"])}

    run_config = engine.default_run_config()
    run_config["objective_config"] = {**run_config["objective_config"], **weights}
    jobs = engine.build_jobs(opts, ctrls, plants_, seeds, int(pop), int(iters), run_config)
    cpu, wall = _estimate(jobs, workers)
    default_name = f"{preset.split(' (')[0] if preset != 'Custom' else f'P{pop}×G{iters}'} · {len(seeds)} seed{'s' if len(seeds) > 1 else ''} · {datetime.now():%H:%M}"
    name = st.text_input("Name", value=default_name, key="lab_name")

    skipped = sum(1 for o in opts for c in ctrls if not is_compatible(o, c)) * len(plants_)
    T.stat_tiles([("Runs", f"{len(jobs)}", f"{skipped // max(len(plants_), 1) * len(plants_)} GP × non-GP skipped" if skipped else "all combinations"),
                  ("Estimated wall time", f"{wall:.0f} s", f"≈{cpu:.0f} s CPU across {workers} workers"),
                  ("Fuzzy share", f"{sum(1 for j in jobs if j.controller == 'Fuzzy')}", "runs · the slow controller")],
                 min_width=200)
    if wall > 300:
        T.callout(f"This will take about <b>{wall / 60:.1f} minutes</b>. During a live talk prefer a small "
                  f"budget or precompute with <code>python -m gui.precompute</code>.", "warn")

    if st.button("▶  Run experiments", type="primary", disabled=not jobs, key="lab_run"):
        bar = st.progress(0.0, text="Starting workers…")
        live = st.empty()
        done_rows = []
        started = time.perf_counter()

        def on_progress(done, total, rec):
            elapsed = time.perf_counter() - started
            eta = elapsed / done * (total - done)
            bar.progress(done / total, text=f"{done}/{total} runs · {elapsed:.0f} s elapsed · ~{eta:.0f} s left")
            done_rows.insert(0, {"optimizer": rec["optimizer"], "controller": rec["controller"],
                                 "plant": PLANT_LABELS[rec["plant"]], "seed": rec["seed"],
                                 "J": rec.get("fitness"), "settled": "✔" if rec.get("settled") else "✖",
                                 "time (s)": round(rec.get("execution_time") or 0, 2),
                                 "error": rec.get("error") or ""})
            live.dataframe(pd.DataFrame(done_rows[:12]), hide_index=True, width="stretch")

        rs = engine.execute_batch(name, opts, ctrls, plants_, seeds, int(pop), int(iters), run_config=run_config,
                                  workers=workers, on_progress=on_progress, budget_label=preset)
        path = engine.save_result_set(rs)
        state.clear_caches()
        state.set_active(path)
        failed = sum(1 for r in rs.records if r.get("status") != "success")
        st.session_state["lab_flash"] = (f"Saved **{rs.name}** — {len(rs.records)} runs ({failed} failed) in "
                                         f"{rs.meta['wall_time']:.1f} s. It is now the active result set.")
        st.toast("Result set ready — open the Results Dashboard", icon="✅")
        st.rerun()


def _manager() -> None:
    st.markdown("<h2>Saved result sets</h2>", unsafe_allow_html=True)
    sets = state.available_sets()
    if not sets:
        st.info("Nothing saved yet.")
        return
    for path in sets:
        rs = state.load(path)
        m = rs.meta
        active = str(path) == st.session_state.get("rs_path")
        with st.container(border=True):
            c1, c2, c3 = st.columns([4, 1, 1])
            badge = f" {T.pill('active', T.ACCENT)}" if active else ""
            c1.markdown(f"<b style='color:{T.TEXT}'>{T.esc(rs.name)}</b>{badge}<br>"
                        f"<span style='color:{T.MUTED};font-size:.88rem'>{T.esc(state.describe(rs))} · "
                        f"wall {m.get('wall_time', 0):.0f} s · <code>{path.name}</code></span>", unsafe_allow_html=True)
            if c2.button("Use", key=f"use_{path.stem}", disabled=active, width="stretch"):
                state.set_active(path)
                st.rerun()
            if c3.button("Delete", key=f"del_{path.stem}", width="stretch", type="tertiary"):
                st.session_state["confirm_del"] = str(path)
            if st.session_state.get("confirm_del") == str(path):
                st.warning(f"Delete **{rs.name}** permanently?")
                d1, d2, _ = st.columns([1, 1, 4])
                if d1.button("Yes, delete", key=f"yes_{path.stem}", type="primary"):
                    engine.delete_result_set(path)
                    state.clear_caches()
                    st.session_state.pop("confirm_del", None)
                    if active:
                        state.set_active(None)
                    st.rerun()
                if d2.button("Cancel", key=f"no_{path.stem}"):
                    st.session_state.pop("confirm_del", None)
                    st.rerun()


def render() -> None:
    T.page_header("Live", "Experiment Lab",
                  "Run the matrix — or any slice of it — right now, with any budget, seeds and objective weights. "
                  "Runs execute in parallel worker processes using the project's own experiment runner.")
    tab1, tab2 = st.tabs(["New run", "Manage result sets"])
    with tab1:
        _builder()
    with tab2:
        _manager()
