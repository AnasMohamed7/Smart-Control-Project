"""Title slide: what we built, who built it, and the headline numbers."""
from __future__ import annotations

import numpy as np
import streamlit as st

from gui import content
from gui.core import analysis, state
from gui.core import theme as T
from gui.core.catalog import CONTROLLERS, OPTIMIZERS, PLANT_LABELS, PLANT_NAMES


def _wave_svg() -> str:
    """Decorative step responses with different damping (drawn, not data)."""
    t = np.linspace(0, 12, 160)
    paths = []
    for zeta, color, opacity in [(0.25, "#9085e9", .55), (0.5, "#3987e5", .8), (0.9, "#86b6ef", .6)]:
        wd = np.sqrt(1 - zeta ** 2)
        y = 1 - np.exp(-zeta * t) * (np.cos(wd * t) + zeta / wd * np.sin(wd * t))
        pts = " ".join(f"{20 + x * 42:.1f},{230 - v * 120:.1f}" for x, v in zip(t, y))
        paths.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2.5" '
                     f'stroke-opacity="{opacity}" stroke-linecap="round"/>')
    return (f'<svg class="wave" width="560" height="260" viewBox="0 0 560 260" aria-hidden="true">'
            f'<line x1="20" y1="110" x2="540" y2="110" stroke="#8a90a0" stroke-opacity=".35" stroke-width="1"/>'
            + "".join(paths) + "</svg>")


def render() -> None:
    pills = "".join([
        T.pill(content.COURSE), T.pill(content.TERM),
        T.pill(f"{len(PLANT_NAMES)} plants", "#86b6ef"),
        T.pill(f"{len(CONTROLLERS)} controllers", T.CONTROLLER_COLORS["PID"]),
        T.pill(f"{len(OPTIMIZERS)} optimisers", T.OPTIMIZER_COLORS["DE"]),
    ])
    st.markdown(
        f'<div class="sc-hero">{_wave_svg()}'
        f'<div class="sc-eyebrow">Course project presentation</div>'
        f'<h1>{T.esc(content.TITLE)}<br><span class="grad">{T.esc(content.TITLE_ACCENT)}</span></h1>'
        f'<p>{T.esc(content.TAGLINE)}</p><div class="meta">{pills}</div></div>',
        unsafe_allow_html=True,
    )

    rs = state.load()
    if rs is not None:
        df = state.frame()
        ok = df[df["status"] == "success"]
        agg = analysis.aggregate(df)
        best_per_plant = []
        for plant in PLANT_NAMES:
            sub = agg[agg["plant"].astype(str) == plant]
            if not sub.empty:
                row = sub.loc[sub["score_mean"].idxmin()]
                best_per_plant.append((f"Best on {PLANT_LABELS[plant]}", f"{row['optimizer']} · {row['controller']}",
                                       f"J = {analysis.fmt(row['score_mean'])} · IAE {analysis.fmt(row['iae'])}"))
        tiles = [
            ("Experiments", f"{len(df)}", f"{len(ok)} completed · {state.set_label(state.active_path())}"),
            ("Settled responses", f"{ok['settled'].mean():.0%}", "inside ±2 % band before t = 20 s"),
            ("Compute (wall)", f"{rs.meta.get('wall_time', 0):.0f} s",
             f"{rs.meta.get('cpu_time', 0):.0f} s CPU on {rs.meta.get('workers', '?')} workers"),
        ]
        st.markdown("<h2>At a glance</h2>", unsafe_allow_html=True)
        T.stat_tiles(tiles)
        if best_per_plant:
            T.stat_tiles(best_per_plant, min_width=260)

    st.markdown("<h2>What we set out to do</h2>", unsafe_allow_html=True)
    accents = ["#86b6ef", T.CONTROLLER_COLORS["PID"], T.OPTIMIZER_COLORS["PSO"], T.OPTIMIZER_COLORS["SA"]]
    T.card_grid([T.card(t, f"<p>{T.esc(d)}</p>", accent=a) for (t, d), a in zip(content.OBJECTIVES, accents)])

    st.markdown("<h2>The pipeline</h2>", unsafe_allow_html=True)
    c1, c2 = st.columns([1.1, 1])
    with c1:
        st.markdown(T.steps([
            ("Pick a plant", "2nd, 3rd or 4th order transfer function, discretised once (ZOH, Δt = 0.01 s)."),
            ("Pick a controller", "PID, Neural Network, Fuzzy or GP — each exposes a bounded parameter vector θ."),
            ("Optimise θ", "DE, GA, PSO or SA propose candidates; GP evolves the control law itself."),
            ("Simulate & score", "20 s unit-step, closed loop, |u| ≤ 50 → five metrics → one objective J(θ)."),
            ("Compare", "Same budget, same seeds, same metrics for every combination."),
        ]), unsafe_allow_html=True)
    with c2:
        team = "".join(
            f"<div class='sc-step'><div class='n'>{T.esc(name.split()[0][:1] + name.split()[-1][:1])}</div>"
            f"<div><div class='t'>{T.esc(name)}</div>"
            f"<div class='d'>{T.esc(role)}</div></div></div>" for name, role in content.TEAM)
        shared = getattr(content, "TEAM_SHARED", "")
        shared_html = (f"<div class='sc-step'><div class='n'>✦</div><div><div class='t'>Everyone</div>"
                       f"<div class='d'>{T.esc(shared)}</div></div></div>") if shared else ""
        sup = f"<p style='margin-top:.6rem'>{T.esc(content.SUPERVISOR)}</p>" if content.SUPERVISOR else ""
        st.markdown(T.card("The team", team + shared_html + sup), unsafe_allow_html=True)
