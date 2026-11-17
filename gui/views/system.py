"""Closed-loop architecture and the three plants."""
from __future__ import annotations

import control
import numpy as np
import streamlit as st

from gui.core import charts, state
from gui.core import theme as T
from gui.core.catalog import PLANT_LABELS, PLANT_LATEX, PLANT_NAMES, PLANT_NOTES, plants


def block_diagram_svg() -> str:
    box = lambda x, y, w, h, title, sub, color: (  # noqa: E731
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" fill="{T.SURFACE_2}" stroke="{color}" stroke-width="2"/>'
        f'<text x="{x + w / 2}" y="{y + h / 2 - 2}" text-anchor="middle" fill="{T.TEXT}" font-size="16" font-weight="700">{title}</text>'
        f'<text x="{x + w / 2}" y="{y + h / 2 + 17}" text-anchor="middle" fill="{T.MUTED}" font-size="12">{sub}</text>')
    label = lambda x, y, text, color=T.TEXT_2, anchor="middle": (  # noqa: E731
        f'<text x="{x}" y="{y}" text-anchor="{anchor}" fill="{color}" font-size="13" font-style="italic">{text}</text>')
    line = lambda pts, color=T.TEXT_2, marker=True, dash="": (  # noqa: E731
        f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="2" '
        f'{"stroke-dasharray=" + chr(34) + dash + chr(34) if dash else ""} '
        f'{"marker-end=" + chr(34) + "url(#arr)" + chr(34) if marker else ""}/>')
    opt = "#9085e9"
    return f"""
<svg class="sc-svg" viewBox="0 0 1000 330" role="img" aria-label="Closed-loop block diagram with optimisation loop">
  <defs>
    <marker id="arr" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{T.TEXT_2}"/></marker>
    <marker id="arrv" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto-start-reverse">
      <path d="M0,0 L10,5 L0,10 z" fill="{opt}"/></marker>
  </defs>
  <rect x="190" y="20" width="580" height="170" rx="16" fill="none" stroke="{T.BORDER}" stroke-width="1.5"/>
  <text x="200" y="40" fill="{T.MUTED}" font-size="12" font-weight="700" letter-spacing="1.5">CLOSED LOOP · simulated 20 s per candidate</text>
  {label(55, 82, "r(t) = 1")}
  {line("30,95 128,95")}
  <circle cx="145" cy="95" r="16" fill="{T.SURFACE_2}" stroke="{T.TEXT_2}" stroke-width="2"/>
  <text x="145" y="100" text-anchor="middle" fill="{T.TEXT}" font-size="16" font-weight="700">Σ</text>
  <text x="122" y="84" fill="{T.GOOD}" font-size="14" font-weight="700">+</text>
  <text x="150" y="128" fill="{T.BAD}" font-size="16" font-weight="700">−</text>
  {line("161,95 222,95")}
  {label(192, 84, "e(t)")}
  {box(224, 65, 150, 60, "Controller", "PID · NN · Fuzzy · GP", T.CONTROLLER_COLORS["PID"])}
  {line("374,95 432,95")}
  {label(402, 84, "u(t)")}
  {box(434, 72, 96, 46, "Saturation", "|u| ≤ 50", T.AXIS)}
  {line("530,95 588,95")}
  {box(590, 65, 150, 60, "Plant G(s)", "2nd · 3rd · 4th order", "#3987e5")}
  {line("740,95 948,95")}
  {label(958, 100, "y(t)", T.TEXT, "start")}
  {line("800,95 800,160 145,160 145,113")}
  <circle cx="800" cy="95" r="4" fill="{T.TEXT_2}"/>
  {label(470, 178, "unity negative feedback", T.MUTED)}
  {line("890,95 890,265 862,265")}
  <circle cx="890" cy="95" r="4" fill="{T.TEXT_2}"/>
  {box(700, 238, 160, 56, "Metrics", "Tr · Ts · OS · Ess · IAE", T.GOOD)}
  {line("700,265 642,265")}
  {box(480, 238, 160, 56, "Objective J(θ)", "weighted sum, penalties", T.WARN)}
  {line("480,265 402,265")}
  {box(240, 238, 160, 56, "Optimiser", "DE · GA · PSO · SA · GP", opt)}
  <polyline points="300,238 300,127" fill="none" stroke="{opt}" stroke-width="2.5" marker-end="url(#arrv)"/>
  <text x="310" y="215" fill="{opt}" font-size="14" font-weight="700" font-style="italic">θ (new candidate)</text>
  <text x="20" y="320" fill="{T.MUTED}" font-size="12">The optimiser only sees J(θ) — it never looks inside the controller or the plant (black-box optimisation).</text>
</svg>"""


@st.cache_data(show_spinner=False)
def plant_properties() -> list[dict]:
    rows = []
    for name, tf in plants().items():
        poles = control.poles(tf)
        dc = float(control.dcgain(tf))
        dominant = poles[np.argmax(poles.real)]
        rows.append({
            "Plant": PLANT_LABELS[name], "Order": len(poles),
            "Poles": ", ".join(f"{p.real:.2f}{p.imag:+.2f}j" if abs(p.imag) > 1e-9 else f"{p.real:.2f}" for p in poles),
            "DC gain": round(dc, 4),
            "Dominant pole": f"{dominant.real:.2f}{dominant.imag:+.2f}j" if abs(dominant.imag) > 1e-9 else f"{dominant.real:.2f}",
            "Slowest τ (s)": round(-1 / float(np.max(poles.real)), 2),
            "Open-loop SS error": round(abs(1 - dc), 3),
        })
    return rows


def render() -> None:
    T.page_header("System", "One loop, many combinations",
                  "Every experiment is the same closed loop. Only three things change: the plant, the controller "
                  "structure, and the algorithm that tunes the controller's parameters θ.")
    st.markdown(block_diagram_svg(), unsafe_allow_html=True)

    st.markdown("<h2>The plants</h2>", unsafe_allow_html=True)
    tfs = plants()
    cols = st.columns(3)
    for col, name in zip(cols, PLANT_NAMES):
        with col:
            st.markdown(f"<div class='sc-eyebrow' style='color:{T.PLANT_COLORS[name]}'>{PLANT_LABELS[name]}</div>",
                        unsafe_allow_html=True)
            st.latex(PLANT_LATEX[name])
            st.caption(PLANT_NOTES[name])

    tab1, tab2, tab3, tab4 = st.tabs(["Open-loop step", "Pole map", "Bode diagram", "Properties"])
    with tab1:
        T.show(charts.open_loop_steps(tfs), key="ol_step")
        T.callout("Left alone, none of the plants tracks r = 1: the 3rd- and 4th-order plants settle at "
                  "their DC gains (1.67 and 1.56) and the 2nd-order plant overshoots by ~9.5 %. "
                  "<b>The controller has to remove the error and shape the transient.</b>")
    with tab2:
        T.show(charts.pole_zero_figure(tfs), key="pz")
        T.callout("All poles sit in the left half-plane, so every plant is open-loop stable. The 4th-order plant "
                  "has the slowest poles (closest to the imaginary axis) and the most phase lag — "
                  "the hardest to close a fast loop around.")
    with tab3:
        T.show(charts.bode_figure(tfs), key="bode")
        T.callout("Higher order ⇒ phase falls faster. The 3rd- and 4th-order plants cross −180° "
                  "(red line), so too much loop gain makes the closed loop unstable.", "warn")
    with tab4:
        st.dataframe(plant_properties(), hide_index=True, width="stretch")
        st.caption("Simulation: discrete state space via zero-order hold, Δt = 0.01 s, horizon 20 s, unit step "
                   "reference, control signal saturated at ±50.")
