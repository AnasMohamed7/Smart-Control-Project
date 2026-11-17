"""The five optimisation algorithms, plus a live 2-D benchmark showing how each searches."""
from __future__ import annotations

import numpy as np
import streamlit as st

from gui.core import charts, engine
from gui.core import theme as T
from gui.core.catalog import NUMERIC_OPTIMIZERS, OPTIMIZER_INFO, OPTIMIZERS

BENCHMARKS = {
    "Rastrigin (many local minima)": (
        lambda x, y: 20 + x ** 2 - 10 * np.cos(2 * np.pi * x) + y ** 2 - 10 * np.cos(2 * np.pi * y),
        (-5.12, 5.12), (0.0, 0.0)),
    "Ackley (flat with a deep hole)": (
        lambda x, y: (-20 * np.exp(-0.2 * np.sqrt(0.5 * (x ** 2 + y ** 2)))
                      - np.exp(0.5 * (np.cos(2 * np.pi * x) + np.cos(2 * np.pi * y))) + np.e + 20),
        (-5.0, 5.0), (0.0, 0.0)),
    "Rosenbrock (curved valley)": (
        lambda x, y: (1 - x) ** 2 + 100 * (y - x ** 2) ** 2, (-2.0, 2.0), (1.0, 1.0)),
    "Sphere (easy bowl)": (lambda x, y: x ** 2 + y ** 2, (-5.0, 5.0), (0.0, 0.0)),
}


@st.cache_data(show_spinner=False)
def run_benchmark(bench: str, pop: int, iters: int, seed: int):
    fn, (lo, hi), _ = BENCHMARKS[bench]
    out = {}
    for name in NUMERIC_OPTIMIZERS:
        points, values = [], []

        def objective(v):
            x, y = float(v[0]), float(v[1])
            val = float(fn(x, y))
            points.append((x, y))
            values.append(val)
            return val

        settings = engine.optimizer_settings_for(pop, iters)[name]
        settings["random_seed"] = seed
        res = OPTIMIZERS[name]().optimize(objective, [(lo, hi), (lo, hi)], settings=settings)
        out[name] = {"points": np.asarray(points), "best": np.asarray(res.best_solution),
                     "fitness": float(res.best_fitness), "history": list(res.convergence_history),
                     "n_evals": len(points)}
    return out


@st.cache_data(show_spinner=False)
def landscape(bench: str, n: int = 160):
    fn, (lo, hi), _ = BENCHMARKS[bench]
    xs = np.linspace(lo, hi, n)
    X, Y = np.meshgrid(xs, xs)
    return xs, xs, fn(X, Y)


def render() -> None:
    T.page_header("Optimisers", "Five ways to search for θ",
                  "Every optimiser receives the same black-box objective J(θ) and the controller's bounds, "
                  "and is given the same number of objective evaluations.")

    cards = []
    for key, info in OPTIMIZER_INFO.items():
        steps = "".join(f"<li><b style='color:{T.TEXT}'>{T.esc(t)}</b> — {T.esc(d)}</li>" for t, d in info["steps"])
        cards.append(T.card(f"{key} · {info['name']}", f"<p>{T.esc(info['idea'])}</p><ul>{steps}</ul>"
                            f"<p class='tag' style='margin-top:.5rem'>{T.esc(info['strengths'])}</p>",
                            accent=T.OPTIMIZER_COLORS[key]))
    T.card_grid(cards, min_width=250)

    with st.expander("Update rules", expanded=False):
        for key, info in OPTIMIZER_INFO.items():
            c1, c2 = st.columns([1, 4])
            c1.markdown(f"<b style='color:{T.OPTIMIZER_COLORS[key]}'>{key}</b>", unsafe_allow_html=True)
            with c2:
                st.latex(info["latex"])

    T.callout("<b>GP is different.</b> DE, GA, PSO and SA search a fixed-length vector of numbers. GP "
              "searches a space of <i>programs</i> — it recombines sub-trees, so it can only drive the GP "
              "controller. That is why the matrix has 4 × 4 × 3 = 48 numeric runs plus 3 GP runs.")

    st.markdown("<h2>Watch them search</h2>", unsafe_allow_html=True)
    st.caption("A 2-D benchmark makes the search visible. Each dot is one objective evaluation — faint dots are "
               "early, bright dots late; ★ is the best point found, ✛ the true optimum.")
    c1, c2, c3, c4 = st.columns([2, 1, 1, 1])
    bench = c1.selectbox("Landscape", list(BENCHMARKS), key="bench_fn")
    pop = c2.slider("Population", 4, 40, 12, key="bench_pop")
    iters = c3.slider("Generations", 1, 40, 10, key="bench_iters")
    seed = c4.number_input("Seed", 0, 9999, 42, key="bench_seed")
    st.caption(f"Budget: {pop * (1 + iters)} evaluations per optimiser (SA: 1 start + {pop * (1 + iters) - 1} moves).")

    with st.spinner("Running the optimisers…"):
        result = run_benchmark(bench, pop, iters, int(seed))
    X, Y, Z = landscape(bench)
    _, _, optimum = BENCHMARKS[bench]
    cols = st.columns(2)
    for i, name in enumerate(NUMERIC_OPTIMIZERS):
        r = result[name]
        with cols[i % 2]:
            fig = charts.landscape_figure(X, Y, Z, r["points"], r["best"], T.OPTIMIZER_COLORS[name],
                                          f"{name} — best f = {r['fitness']:.4g}  ({r['n_evals']} evals)",
                                          optimum=optimum)
            T.show(fig, key=f"bench_{name}")
    traces = [{"history": result[n]["history"], "n_evals": result[n]["n_evals"], "name": n,
               "color": T.OPTIMIZER_COLORS[n]} for n in NUMERIC_OPTIMIZERS]
    fig = charts.convergence_figure(traces, title="Best value found vs evaluations", log=True)
    T.show(fig, key="bench_conv")
    T.callout("Population methods spread their samples (exploration) and then concentrate; SA follows a single "
              "trajectory. On multimodal landscapes the difference between exploring and exploiting decides "
              "who escapes the local minima.")
