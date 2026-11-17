"""Plotly figure builders. Pages compose these; no page styles a chart itself."""
from __future__ import annotations

import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from gui.core import theme as T
from gui.core.catalog import PLANT_LABELS, metric_label

Y_CLIP = (-0.6, 2.6)   # keep diverged responses from flattening every other trace


def log_axis(fig: go.Figure, axis: str = "y", values=None) -> go.Figure:
    """Log axis with readable 1-2-5 ticks (…, 10, 20, 50, 100, …) and full labels.

    Plotly's compact log ticks print minor ticks as bare digits ("2" for 20), which
    reads wrongly on slides; explicit tick values avoid that.
    """
    if values is None:
        values = [v for tr in fig.data for v in (getattr(tr, axis) if getattr(tr, axis) is not None else [])]
    vals = np.asarray([v for v in values if v is not None], dtype=float)
    vals = vals[np.isfinite(vals) & (vals > 0)]
    update = fig.update_yaxes if axis == "y" else fig.update_xaxes
    if not vals.size:
        update(type="log")
        return fig
    lo, hi = vals.min(), vals.max()
    decades = range(int(math.floor(math.log10(lo))) - 1, int(math.ceil(math.log10(hi))) + 1)

    def nice(mantissas, pad):
        return [m * 10.0 ** k for k in decades for m in mantissas if lo / pad <= m * 10.0 ** k <= hi * pad]

    ticks = nice((1, 2, 5), 1.6)
    if len(ticks) > 9:                                   # wide ranges: decades only
        ticks = [t for t in ticks if abs(math.log10(t) - round(math.log10(t))) < 1e-9]
    for finer in ((1, 1.5, 2, 3, 5, 7), (1, 1.2, 1.5, 2, 2.5, 3, 4, 5, 6, 7, 8)):
        if sum(lo <= t <= hi for t in ticks) >= 3:
            break
        ticks = nice(finer, 1.0)                         # narrow ranges: finer nice steps, in range only
    if sum(lo <= t <= hi for t in ticks) < 2:            # very narrow: evenly spaced rounded ticks
        ticks = sorted({float(f"{t:.2g}") for t in np.geomspace(lo, hi, 4)})
    update(type="log", tickvals=ticks, ticktext=[_short_tick(t) for t in ticks])
    return fig


def _short_tick(v: float) -> str:
    if v >= 1000:
        return f"{v / 1000:g}k"
    return f"{v:g}"


# ---------------------------------------------------------------------------
# Time-domain responses
# ---------------------------------------------------------------------------
def _clip_note(fig, clipped: bool):
    if clipped:
        fig.add_annotation(text="axis clipped — some responses leave the frame (unstable)",
                           xref="paper", yref="paper", x=1, y=1.0, xanchor="right", yanchor="bottom",
                           showarrow=False, font=dict(color=T.WARN, size=11))


def step_figure(traces: list[dict], reference: float = 1.0, tolerance: float = 0.02,
                title: str = "", height: int = 420, show_band: bool = True,
                end_labels: bool = True) -> go.Figure:
    """traces: dicts with t, y, name, color, (dash, width, opacity, group)."""
    fig = go.Figure()
    if show_band:
        fig.add_hrect(y0=reference * (1 - tolerance), y1=reference * (1 + tolerance),
                      fillcolor=T.hex_to_rgba(T.GOOD, 0.10), line_width=0, layer="below")
    t_max = max((float(tr["t"][-1]) for tr in traces if len(tr["t"])), default=20.0)
    fig.add_shape(type="line", x0=0, x1=t_max, y0=reference, y1=reference,
                  line=dict(color=T.MUTED, width=1), layer="below")
    clipped = False
    for tr in traces:
        y = np.asarray(tr["y"], dtype=float)
        finite = y[np.isfinite(y)]
        if finite.size and (finite.max() > Y_CLIP[1] or finite.min() < Y_CLIP[0]) or not np.all(np.isfinite(y)):
            clipped = True
        fig.add_trace(go.Scatter(
            x=tr["t"], y=y, name=tr["name"], mode="lines",
            line=dict(color=tr["color"], width=tr.get("width", 2), dash=tr.get("dash", "solid")),
            opacity=tr.get("opacity", 1.0), legendgroup=tr.get("group"),
            hovertemplate=f"<b>{tr['name']}</b><br>t = %{{x:.2f}} s<br>y = %{{y:.4f}}<extra></extra>",
        ))
    if end_labels and len(traces) <= 4:
        # Direct labels at the line ends, skipping any that would collide.
        ends = []
        for tr in traces:
            y = np.asarray(tr["y"], dtype=float)
            if len(y) and math.isfinite(y[-1]) and Y_CLIP[0] < y[-1] < Y_CLIP[1]:
                ends.append((float(y[-1]), tr))
        placed = []
        span = max((abs(e[0]) for e in ends), default=1.0) or 1.0
        for yv, tr in sorted(ends, key=lambda e: e[0]):
            if any(abs(yv - q) < 0.07 * span for q in placed):
                continue
            placed.append(yv)
            fig.add_annotation(x=tr["t"][-1], y=yv, text=tr.get("short", tr["name"]),
                               showarrow=False, xanchor="left", xshift=6,
                               font=dict(color=T.TEXT_2, size=11))
    fig.update_layout(title=title, height=height, hovermode="x unified",
                      xaxis_title="Time (s)", yaxis_title="Output y(t)",
                      margin=dict(l=22, r=86 if end_labels and len(traces) <= 4 else 22, t=60, b=48))
    if clipped:
        fig.update_yaxes(range=list(Y_CLIP))
    _clip_note(fig, clipped)
    return fig


def step_grid(panels: list[dict], reference: float = 1.0, tolerance: float = 0.02,
              height: int = 620, cols: int = 2) -> go.Figure:
    """Small multiples: panels = [{title, traces:[{t,y,name,color,group}]}]; legend shared by group."""
    rows = math.ceil(len(panels) / cols)
    fig = make_subplots(rows=rows, cols=cols, subplot_titles=[p["title"] for p in panels],
                        shared_xaxes=True, shared_yaxes=True, vertical_spacing=0.1, horizontal_spacing=0.05)
    seen, clipped = set(), False
    for i, panel in enumerate(panels):
        r, c = i // cols + 1, i % cols + 1
        fig.add_hrect(y0=reference * (1 - tolerance), y1=reference * (1 + tolerance), row=r, col=c,
                      fillcolor=T.hex_to_rgba(T.GOOD, 0.10), line_width=0, layer="below")
        fig.add_hline(y=reference, row=r, col=c, line=dict(color=T.MUTED, width=1))
        for tr in panel["traces"]:
            y = np.asarray(tr["y"], dtype=float)
            fin = y[np.isfinite(y)]
            if (fin.size and (fin.max() > Y_CLIP[1] or fin.min() < Y_CLIP[0])) or not np.all(np.isfinite(y)):
                clipped = True
            group = tr.get("group", tr["name"])
            fig.add_trace(go.Scatter(
                x=tr["t"], y=y, name=tr["name"], legendgroup=group, showlegend=group not in seen,
                mode="lines", line=dict(color=tr["color"], width=1.8, dash=tr.get("dash", "solid")),
                hovertemplate=f"<b>{tr['name']}</b> · {panel['title']}<br>t = %{{x:.2f}} s<br>y = %{{y:.4f}}<extra></extra>",
            ), row=r, col=c)
            seen.add(group)
    fig.update_yaxes(range=list(Y_CLIP) if clipped else None)
    for c in range(1, cols + 1):
        fig.update_xaxes(title_text="Time (s)", row=rows, col=c)
    fig.update_layout(height=height, margin=dict(l=22, r=22, t=64, b=48))
    for ann in fig.layout.annotations:
        ann.font = dict(color=T.TEXT, size=13)
    _clip_note(fig, clipped)
    return fig


def control_figure(traces: list[dict], title: str = "Control effort u(t)", height: int = 300) -> go.Figure:
    fig = go.Figure()
    for tr in traces:
        fig.add_trace(go.Scatter(
            x=tr["t"], y=tr["u"], name=tr["name"], mode="lines",
            line=dict(color=tr["color"], width=1.6, dash=tr.get("dash", "solid")),
            legendgroup=tr.get("group"),
            hovertemplate=f"<b>{tr['name']}</b><br>t = %{{x:.2f}} s<br>u = %{{y:.3f}}<extra></extra>",
        ))
    fig.update_layout(title=title, height=height, hovermode="x unified",
                      xaxis_title="Time (s)", yaxis_title="u(t)")
    return fig


def convergence_figure(traces: list[dict], title: str = "Convergence", height: int = 360,
                       log: bool = True) -> go.Figure:
    """traces: dicts with history, n_evals, name, color. x is mapped to evaluations so
    population methods (one point per generation) and SA (one per move) align."""
    fig = go.Figure()
    for tr in traces:
        h = np.asarray(tr["history"], dtype=float)
        if not h.size:
            continue
        n_evals = tr.get("n_evals") or len(h)
        x = np.linspace(n_evals / len(h), n_evals, len(h))
        fig.add_trace(go.Scatter(
            x=x, y=h, name=tr["name"], mode="lines+markers" if len(h) < 30 else "lines",
            line=dict(color=tr["color"], width=2, shape="hv", dash=tr.get("dash", "solid")),
            marker=dict(size=6, line=dict(color=T.SURFACE, width=1.5)),
            legendgroup=tr.get("group"),
            hovertemplate=f"<b>{tr['name']}</b><br>≈ %{{x:.0f}} evaluations<br>best J = %{{y:.4g}}<extra></extra>",
        ))
    fig.update_layout(title=title, height=height, xaxis_title="Objective evaluations",
                      yaxis_title="Best objective J (lower is better)")
    if log:
        log_axis(fig, "y")
    return fig


# ---------------------------------------------------------------------------
# Matrix views
# ---------------------------------------------------------------------------
def heatmap_grid(agg: pd.DataFrame, value: str, plants: list[str], log: bool = False,
                 label: str = "", height: int = 360, reverse_good: bool = True) -> go.Figure:
    """Controllers × optimizers heatmap, one panel per plant, shared color scale."""
    opts = [o for o in ["DE", "GA", "PSO", "SA", "GP"] if o in set(agg["optimizer"].astype(str))]
    ctrls = [c for c in ["PID", "Neural Network", "Fuzzy", "GP Controller"] if c in set(agg["controller"].astype(str))]
    fig = make_subplots(rows=1, cols=len(plants), subplot_titles=[PLANT_LABELS[p] for p in plants],
                        horizontal_spacing=0.04, shared_yaxes=True)
    vals = agg[value].replace([np.inf, -np.inf], np.nan).dropna()
    vals = vals[vals > 0] if log else vals
    if vals.empty:
        zmin, zmax = 0, 1
    else:
        zmin = math.log10(vals.min()) if log else vals.min()
        zmax = math.log10(vals.max()) if log else vals.max()
        if zmax == zmin:
            zmax = zmin + 1
    for c, plant in enumerate(plants, 1):
        sub = agg[agg["plant"].astype(str) == plant]
        z, text, custom = [], [], []
        best = sub[value].min() if reverse_good else sub[value].max()
        for ctrl in ctrls:
            zr, tr, cr = [], [], []
            for opt in opts:
                cell = sub[(sub["controller"].astype(str) == ctrl) & (sub["optimizer"].astype(str) == opt)]
                if cell.empty or pd.isna(cell[value].iloc[0]):
                    zr.append(None)
                    tr.append("n/a" if cell.empty else "—")
                    cr.append(["", ""])
                    continue
                v = float(cell[value].iloc[0])
                settled = float(cell["settled_rate"].iloc[0]) if "settled_rate" in cell else 1.0
                zr.append(math.log10(v) if (log and v > 0) else (None if log else v))
                star = " ★" if v == best else ""
                warn = "" if settled >= 0.999 else (" ⚠" if settled > 0 else " ✖")
                tr.append(f"{_short(v)}{star}{warn}")
                cr.append([_short(v), f"{settled:.0%}"])
            z.append(zr)
            text.append(tr)
            custom.append(cr)
        fig.add_trace(go.Heatmap(
            z=z, x=opts, y=ctrls, text=text, texttemplate="%{text}", customdata=custom,
            textfont=dict(size=12, color=T.TEXT), xgap=3, ygap=3,
            colorscale=T.SEQUENTIAL, zmin=zmin, zmax=zmax, showscale=(c == len(plants)),
            colorbar=dict(title=dict(text="log₁₀" if log else "", side="top",
                                     font=dict(color=T.MUTED, size=11)), thickness=10, len=0.85),
            hovertemplate="<b>%{x} · %{y}</b><br>" + label + ": %{customdata[0]}<br>settled runs: %{customdata[1]}<extra></extra>",
        ), row=1, col=c)
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    fig.update_xaxes(showline=False, ticks="", side="bottom")
    fig.update_layout(height=height, margin=dict(l=22, r=22, t=44, b=22))
    for ann in fig.layout.annotations:
        ann.font = dict(color=T.TEXT, size=13)
    return fig


def _short(v: float) -> str:
    if v is None or not math.isfinite(v):
        return "—"
    if abs(v) >= 1000:
        return f"{v/1000:.1f}k"
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.1f}"
    return f"{v:.2f}"


def rank_bars(summary: pd.DataFrame, colors: dict, title: str, height: int | None = None) -> go.Figure:
    """Horizontal bars of mean rank (from ``analysis.rank_summary``); labels show how many
    cells each average is based on, so a partial participant (GP) is read correctly."""
    names = [str(i) for i in summary.index]
    full = int(summary["cells"].max()) if len(summary) else 0
    labels = [f"{m:.2f}" + (f"  ·  {int(c)} of {full} cells" if int(c) < full else "")
              for m, c in zip(summary["mean"], summary["cells"])]
    fig = go.Figure(go.Bar(
        x=summary["mean"], y=names, orientation="h",
        marker=dict(color=[colors.get(n, T.MUTED) for n in names], cornerradius=4),
        text=labels, textposition="outside", textfont=dict(color=T.TEXT_2), cliponaxis=False,
        customdata=summary["cells"].astype(int),
        hovertemplate="<b>%{y}</b><br>average rank %{x:.2f}<br>over %{customdata} contested cells<extra></extra>",
    ))
    fig.update_layout(title=title, height=height or (110 + 46 * max(len(names), 1)),
                      xaxis_title="Average rank (1 = best)", showlegend=False, bargap=0.35,
                      margin=dict(l=22, r=22, t=56, b=52))
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    top = max(float(summary["mean"].max()) if len(summary) else 1.0, 1.0)
    has_note = any("cells" in l for l in labels)
    fig.update_xaxes(range=[0, top * (1.55 if has_note else 1.22)], dtick=1)
    return fig


def seed_strip(df: pd.DataFrame, controller: str, plant: str, value: str = "fitness",
               height: int = 320, log: bool = True) -> go.Figure:
    """Every seed's result as a dot, mean as a bar tick, per optimizer."""
    sub = df[(df["controller"].astype(str) == controller) & (df["plant"].astype(str) == plant)]
    fig = go.Figure()
    for opt in [o for o in ["DE", "GA", "PSO", "SA", "GP"] if o in set(sub["optimizer"].astype(str))]:
        s = sub[sub["optimizer"].astype(str) == opt]
        color = T.OPTIMIZER_COLORS[opt]
        fig.add_trace(go.Box(
            y=s[value], name=opt, boxpoints="all", jitter=0.35, pointpos=0,
            marker=dict(color=color, size=9, line=dict(color=T.SURFACE, width=1.5)),
            line=dict(color=color, width=1.5), fillcolor=T.hex_to_rgba(color, 0.12),
            customdata=s[["seed"]].values,
            hovertemplate=f"<b>{opt}</b><br>seed %{{customdata[0]}}<br>{metric_label(value)} = %{{y:.4g}}<extra></extra>",
        ))
    fig.update_layout(height=height, showlegend=False, yaxis_title=metric_label(value),
                      title=f"Spread across seeds — {controller} on {PLANT_LABELS.get(plant, plant)}")
    if log:
        log_axis(fig, "y")
    return fig


def runtime_bars(agg: pd.DataFrame, height: int = 320) -> go.Figure:
    data = agg.groupby(["controller", "optimizer"], observed=True)["runtime"].mean().reset_index()
    fig = go.Figure()
    for opt in [o for o in ["DE", "GA", "PSO", "SA", "GP"] if o in set(data["optimizer"].astype(str))]:
        s = data[data["optimizer"].astype(str) == opt]
        fig.add_trace(go.Bar(x=s["controller"].astype(str), y=s["runtime"], name=opt,
                             marker=dict(color=T.OPTIMIZER_COLORS[opt], cornerradius=4),
                             hovertemplate=f"<b>{opt}</b> · %{{x}}<br>%{{y:.2f}} s per run<extra></extra>"))
    fig.update_layout(barmode="group", height=height, yaxis_title="Mean wall time per run (s)",
                      title="Runtime per run", bargap=0.25, bargroupgap=0.08)
    return fig


SHORT_METRIC = {"fitness": "J", "iae": "IAE", "settling_time": "Ts", "rise_time": "Tr",
                "overshoot": "OS", "steady_state_error": "Ess", "execution_time": "Runtime"}


def metric_bars(rows: list[dict], metrics: list[str], height: int = 340) -> go.Figure:
    """Grouped bars of *relative* metric values (best = 1.0) for head-to-head picks."""
    fig = go.Figure()
    for r in rows:
        vals, texts = [], []
        for m in metrics:
            v = r["values"].get(m)
            col = [x["values"].get(m) for x in rows if x["values"].get(m) is not None and math.isfinite(x["values"].get(m))]
            best = min(col) if col else None
            if v is None or not math.isfinite(v) or best is None:
                vals.append(None)
                texts.append("—")
            else:
                vals.append(1.0 if best == 0 and v == 0 else (v / best if best > 0 else (1.0 + v)))
                texts.append(_short(v))
        fig.add_trace(go.Bar(
            x=[SHORT_METRIC.get(m, m) for m in metrics], y=vals, name=r.get("short", r["name"]),
            marker=dict(color=r["color"], cornerradius=4), customdata=texts,
            hovertemplate=f"<b>{r['name']}</b><br>%{{x}}: %{{customdata}}<br>%{{y:.2f}}× the best<extra></extra>",
        ))
    fig.update_layout(barmode="group", height=height, yaxis_title="× the best pick (log)",
                      title="Relative to the best pick on each metric (1 = best)", bargap=0.3, bargroupgap=0.1)
    fig.update_yaxes(type="log", tickvals=[1, 2, 5, 10, 100, 1000, 10000],
                     ticktext=["1×", "2×", "5×", "10×", "100×", "1k×", "10k×"])
    fig.update_xaxes(tickangle=0)
    return fig


# ---------------------------------------------------------------------------
# Plants
# ---------------------------------------------------------------------------
def pole_zero_figure(plants: dict, height: int = 380) -> go.Figure:
    import control
    fig = go.Figure()
    fig.add_vline(x=0, line=dict(color=T.AXIS, width=1))
    fig.add_hline(y=0, line=dict(color=T.AXIS, width=1))
    fig.add_vrect(x0=-4, x1=0, fillcolor=T.hex_to_rgba(T.GOOD, 0.04), line_width=0, layer="below")
    symbols = {"second_order": "x", "third_order": "x-thin", "fourth_order": "x"}
    for name, tf in plants.items():
        poles = control.poles(tf)
        color = T.PLANT_COLORS[name]
        fig.add_trace(go.Scatter(
            x=poles.real, y=poles.imag, mode="markers", name=PLANT_LABELS[name],
            marker=dict(symbol=symbols.get(name, "x"), size=14, color=color,
                        line=dict(width=2.5, color=color)),
            hovertemplate=f"<b>{PLANT_LABELS[name]}</b><br>pole %{{x:.3f}} %{{y:+.3f}}j<extra></extra>",
        ))
    fig.add_annotation(text="stable half-plane", x=-3.9, y=1.15, xanchor="left", showarrow=False,
                       font=dict(color=T.GOOD, size=11))
    fig.update_layout(title="Pole map (s-plane)", height=height, xaxis_title="Re(s)", yaxis_title="Im(s)")
    fig.update_xaxes(range=[-3.5, 0.6])
    fig.update_yaxes(range=[-1.3, 1.3], scaleanchor=None)
    return fig


def open_loop_steps(plants: dict, t_end: float = 20.0, height: int = 380) -> go.Figure:
    import control
    traces = []
    t = np.linspace(0, t_end, 800)
    for name, tf in plants.items():
        tt, y = control.step_response(tf, T=t)
        traces.append({"t": tt, "y": y, "name": PLANT_LABELS[name], "color": T.PLANT_COLORS[name]})
    fig = step_figure(traces, show_band=False, title="Open-loop step response (no controller)", height=height)
    fig.update_layout(yaxis_title="y(t)")
    return fig


def bode_figure(plants: dict, height: int = 460) -> go.Figure:
    import control
    w = np.logspace(-2, 2, 400)
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.08,
                        subplot_titles=["Magnitude (dB)", "Phase (deg)"])
    for name, tf in plants.items():
        resp = control.frequency_response(tf, w)
        mag = np.squeeze(resp.magnitude)
        phase = np.unwrap(np.squeeze(resp.phase)) * 180 / np.pi
        color = T.PLANT_COLORS[name]
        fig.add_trace(go.Scatter(x=w, y=20 * np.log10(mag), name=PLANT_LABELS[name], line=dict(color=color),
                                 legendgroup=name,
                                 hovertemplate=f"<b>{PLANT_LABELS[name]}</b><br>ω = %{{x:.3g}} rad/s<br>%{{y:.1f}} dB<extra></extra>"),
                      row=1, col=1)
        fig.add_trace(go.Scatter(x=w, y=phase, name=PLANT_LABELS[name], line=dict(color=color),
                                 legendgroup=name, showlegend=False,
                                 hovertemplate=f"<b>{PLANT_LABELS[name]}</b><br>ω = %{{x:.3g}} rad/s<br>%{{y:.0f}}°<extra></extra>"),
                      row=2, col=1)
    fig.add_hline(y=-180, line=dict(color=T.BAD, width=1), row=2, col=1)
    log_axis(fig, "x", w)
    fig.update_xaxes(title_text="Frequency ω (rad/s)", row=2, col=1)
    fig.update_layout(height=height, title="Bode diagram", hovermode="x unified")
    for ann in fig.layout.annotations:
        ann.font = dict(color=T.TEXT_2, size=12)
    return fig


# ---------------------------------------------------------------------------
# Metrics explainer
# ---------------------------------------------------------------------------
def _first_crossing(t, y, level):
    above = np.nonzero(np.nan_to_num(y, nan=-np.inf) >= level)[0]
    if not above.size:
        return None
    k = int(above[0])
    if k == 0:
        return float(t[0])
    dy = y[k] - y[k - 1]
    frac = 0.0 if dy == 0 else (level - y[k - 1]) / dy
    return float(t[k - 1] + frac * (t[k] - t[k - 1]))


def annotated_metrics(t, y, metrics: dict, reference: float = 1.0, tol: float = 0.02,
                      height: int = 470, show: dict | None = None) -> go.Figure:
    show = show or {k: True for k in ["iae", "rise", "settle", "overshoot", "sse"]}
    t = np.asarray(t, float)
    y = np.asarray(y, float)
    fig = go.Figure()
    if show.get("iae"):
        fig.add_trace(go.Scatter(x=np.r_[t, t[::-1]], y=np.r_[y, np.full_like(y, reference)],
                                 fill="toself", fillcolor=T.hex_to_rgba(T.ACCENT, 0.16), line=dict(width=0),
                                 name="IAE area |e(t)|", hoverinfo="skip"))
    if show.get("settle"):
        fig.add_hrect(y0=reference * (1 - tol), y1=reference * (1 + tol),
                      fillcolor=T.hex_to_rgba(T.GOOD, 0.13), line_width=0, layer="below")
    fig.add_shape(type="line", x0=t[0], x1=t[-1], y0=reference, y1=reference, line=dict(color=T.MUTED, width=1))
    fig.add_trace(go.Scatter(x=t, y=y, name="y(t)", line=dict(color=T.TEXT, width=2.2),
                             hovertemplate="t = %{x:.2f} s<br>y = %{y:.4f}<extra></extra>"))
    top = max(float(np.nanmax(y)), reference * 1.1)

    tr = metrics.get("rise_time")
    if show.get("rise") and tr is not None:
        y10, y90 = 0.1 * reference, 0.9 * reference
        t10 = _first_crossing(t, y, y10)
        if t10 is not None:
            t90 = t10 + tr
            for yy in (y10, y90):
                fig.add_shape(type="line", x0=0, x1=t90, y0=yy, y1=yy, line=dict(color="#86b6ef", width=1))
            fig.add_shape(type="rect", x0=t10, x1=t90, y0=0, y1=y90, fillcolor=T.hex_to_rgba("#86b6ef", 0.12),
                          line_width=0, layer="below")
            fig.add_annotation(x=(t10 + t90) / 2, y=y90, text=f"<b>Tr = {tr:.2f} s</b>", showarrow=True,
                               arrowcolor="#86b6ef", ay=-36, ax=30, font=dict(color="#86b6ef"))
    ts = metrics.get("settling_time")
    if show.get("settle") and ts is not None:
        fig.add_vline(x=ts, line=dict(color=T.GOOD, width=1.5))
        fig.add_annotation(x=ts, y=reference * (1 - tol) - 0.08, text=f"<b>Ts = {ts:.2f} s</b>",
                           showarrow=False, xanchor="left", xshift=6, font=dict(color=T.GOOD))
    os_ = metrics.get("overshoot")
    if show.get("overshoot") and os_ is not None and os_ > 0.05:
        k = int(np.nanargmax(y))
        fig.add_trace(go.Scatter(x=[t[k]], y=[y[k]], mode="markers", name="peak",
                                 marker=dict(size=11, color="#e87ba4", line=dict(color=T.SURFACE, width=2)),
                                 hovertemplate="peak %{y:.4f} at %{x:.2f} s<extra></extra>", showlegend=False))
        fig.add_shape(type="line", x0=t[k], x1=t[k], y0=reference, y1=y[k], line=dict(color="#e87ba4", width=2))
        fig.add_annotation(x=t[k], y=y[k], text=f"<b>OS = {os_:.1f} %</b>", showarrow=True, ax=40, ay=-30,
                           arrowcolor="#e87ba4", font=dict(color="#e87ba4"))
    ess = metrics.get("steady_state_error")
    t_view = min(float(t[-1]), max(6.0, 2.5 * ts) if ts is not None else float(t[-1]))
    if show.get("sse") and ess is not None:
        k_e = min(len(t) - 1, int(np.searchsorted(t, 0.92 * t_view)))
        fig.add_annotation(x=t[k_e], y=float(y[k_e]), text=f"<b>Ess = {ess:.2g}</b>", showarrow=True, ax=-60, ay=40,
                           arrowcolor=T.WARN, font=dict(color=T.WARN))
    iae = metrics.get("iae")
    if show.get("iae") and iae is not None:
        fig.add_annotation(xref="paper", yref="paper", x=0.97, y=0.06, xanchor="right", showarrow=False,
                           text=f"<b>IAE = ∫|e|dt = {iae:.3f}</b>", font=dict(color="#86b6ef", size=13))
    fig.update_layout(height=height, xaxis_title="Time (s)", yaxis_title="Output y(t)", hovermode="x",
                      legend=dict(y=1.02))
    fig.update_yaxes(range=[min(-0.05, float(np.nanmin(y)) - 0.05), top + 0.12])
    # Zoom to where the action is: a fast response would otherwise squash every annotation.
    fig.update_xaxes(range=[0, t_view])
    return fig


# ---------------------------------------------------------------------------
# Controllers
# ---------------------------------------------------------------------------
FUZZY_LABELS = ["NB", "NS", "ZE", "PS", "PB"]
FUZZY_COLORS = ["#3987e5", "#6da7ec", "#8a90a0", "#ec8f8f", "#e66767"]   # diverging: neg blue, zero gray, pos red


def fuzzy_memberships(height: int = 300) -> go.Figure:
    centers = [-1.0, -0.5, 0.0, 0.5, 1.0]
    x = np.linspace(-1.2, 1.2, 481)
    fig = go.Figure()
    for i, c in enumerate(centers):
        mu = np.clip(1 - np.abs(x - c) / 0.5, 0, 1)
        if i == 0:
            mu = np.where(x <= c, 1.0, mu)
        if i == 4:
            mu = np.where(x >= c, 1.0, mu)
        fig.add_trace(go.Scatter(x=x, y=mu, name=FUZZY_LABELS[i], mode="lines", fill="tozeroy",
                                 line=dict(color=FUZZY_COLORS[i], width=2),
                                 fillcolor=T.hex_to_rgba(FUZZY_COLORS[i], 0.10),
                                 hovertemplate=f"<b>{FUZZY_LABELS[i]}</b><br>x = %{{x:.2f}}<br>μ = %{{y:.2f}}<extra></extra>"))
        fig.add_annotation(x=c, y=1.08, text=FUZZY_LABELS[i], showarrow=False, font=dict(color=T.TEXT_2, size=12))
    fig.update_layout(height=height, title="Membership functions (normalised e, ė and u)",
                      xaxis_title="normalised value", yaxis_title="membership μ", showlegend=False)
    fig.update_yaxes(range=[0, 1.18])
    return fig


def fuzzy_rule_table(rules: np.ndarray, height: int = 340) -> go.Figure:
    z = np.asarray(rules, float) - 2
    text = [[FUZZY_LABELS[int(v)] for v in row] for row in rules]
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"ė {l}" for l in FUZZY_LABELS], y=[f"e {l}" for l in FUZZY_LABELS], text=text,
        texttemplate="<b>%{text}</b>", textfont=dict(size=13, color=T.TEXT),
        colorscale=T.DIVERGING, zmin=-2, zmax=2, showscale=False, xgap=3, ygap=3,
        hovertemplate="IF %{y} AND %{x}<br>THEN u is <b>%{text}</b><extra></extra>",
    ))
    fig.update_layout(height=height, title="25-rule base (output u)")
    fig.update_yaxes(autorange="reversed", showline=False, ticks="")
    fig.update_xaxes(showline=False, ticks="", side="top")
    return fig


def fuzzy_surface(E, DE, U, height: int = 460) -> go.Figure:
    fig = go.Figure(go.Surface(
        x=E, y=DE, z=U, colorscale=T.DIVERGING, cmin=-1, cmax=1, showscale=False,
        contours=dict(z=dict(show=True, usecolormap=True, project_z=True, width=1)),
        hovertemplate="e = %{x:.2f}<br>ė = %{y:.2f}<br>u = %{z:.3f}<extra></extra>",
        lighting=dict(ambient=0.75, diffuse=0.6, specular=0.15), opacity=0.96,
    ))
    fig.update_layout(height=height, title="Control surface u = f(e, ė)  (normalised)",
                      scene=dict(xaxis_title="e", yaxis_title="ė", zaxis_title="u",
                                 xaxis=dict(range=[-1, 1]), yaxis=dict(range=[-1, 1]),
                                 zaxis=dict(range=[-1.05, 1.05]),
                                 camera=dict(eye=dict(x=1.75, y=-1.75, z=1.05), center=dict(x=0, y=0, z=-0.1)),
                                 aspectmode="manual", aspectratio=dict(x=1, y=1, z=0.65)),
                      margin=dict(l=0, r=0, t=40, b=0))
    return fig


def nn_diagram(params=None, height: int = 400) -> go.Figure:
    """2-5-1 network; edges colored by weight sign (diverging), width by magnitude."""
    layers = [2, 5, 1]
    labels = [["e(k)", "e(k−1)"], [f"h{i+1}" for i in range(5)], ["u(k)"]]
    pos = {}
    for li, n in enumerate(layers):
        for ni in range(n):
            pos[(li, ni)] = (li * 1.0, ((n - 1) / 2 - ni) * 0.9)   # first node on top
    W1 = W2 = None
    if params is not None and len(params) == 21:
        p = np.asarray(params, float)
        W1 = p[:10].reshape(5, 2)
        W2 = p[15:20].reshape(1, 5)
    fig = go.Figure()
    wmax = 5.0
    for li, W, (src_n, dst_n) in [(0, W1, (2, 5)), (1, W2, (5, 1))]:
        for d in range(dst_n):
            for s in range(src_n):
                w = None if W is None else float(W[d, s])
                x0, y0 = pos[(li, s)]
                x1, y1 = pos[(li + 1, d)]
                if w is None:
                    color, width = T.AXIS, 1.2
                else:
                    color = "#3987e5" if w < 0 else "#e66767"
                    color = T.hex_to_rgba(color, 0.35 + 0.6 * min(abs(w) / wmax, 1))
                    width = 0.8 + 4.5 * min(abs(w) / wmax, 1)
                fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=color, width=width),
                                         hoverinfo="text" if w is not None else "skip",
                                         text=f"w = {w:+.3f}" if w is not None else None, showlegend=False))
    for li, n in enumerate(layers):
        xs = [pos[(li, i)][0] for i in range(n)]
        ys = [pos[(li, i)][1] for i in range(n)]
        color = ["#86b6ef", "#9085e9", "#0ca30c"][li]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers+text", text=labels[li],
                                 textposition="middle left" if li == 0 else ("middle right" if li == 2 else "top center"),
                                 textfont=dict(color=T.TEXT, size=12),
                                 marker=dict(size=30, color=T.SURFACE_2, line=dict(color=color, width=2.5)),
                                 hoverinfo="text", hovertext=labels[li], showlegend=False))
    for li, name in enumerate(["input", "hidden · tanh", "output · linear"]):
        fig.add_annotation(x=li, y=-2.55, text=name, showarrow=False, font=dict(color=T.MUTED, size=12))
    fig.update_xaxes(visible=False, range=[-0.6, 2.6])
    fig.update_yaxes(visible=False, range=[-2.8, 2.5])
    fig.update_layout(height=height, title="Network (edge color = sign, width = |w|)" if W1 is not None
                      else "Network architecture 2 → 5 → 1")
    return fig


def gp_tree_figure(tree, height: int = 420, title: str = "Evolved expression tree") -> go.Figure:
    """Draw a nested-tuple GP tree with leaves evenly spaced."""
    symbols = {"add": "+", "sub": "−", "mul": "×", "div": "÷", "tanh": "tanh"}
    nodes, edges = [], []
    counter = [0]

    def place(node, depth):
        if isinstance(node, tuple):
            child_pos = [place(c, depth + 1) for c in node[1:]]
            x = sum(p[0] for p in child_pos) / len(child_pos)
            label, kind = symbols.get(node[0], node[0]), "fn"
        else:
            x = counter[0]
            counter[0] += 1
            child_pos = []
            if isinstance(node, str):
                label, kind = {"e": "e", "de": "ė", "ie": "∫e"}.get(node, node), "var"
            else:
                label, kind = f"{float(node):.3g}", "const"
        me = (x, -depth)
        for cp in child_pos:
            edges.append((me, cp))
        nodes.append((x, -depth, label, kind))
        return me

    place(tree, 0)
    fig = go.Figure()
    for (x0, y0), (x1, y1) in edges:
        fig.add_trace(go.Scatter(x=[x0, x1], y=[y0, y1], mode="lines", line=dict(color=T.AXIS, width=2),
                                 hoverinfo="skip", showlegend=False))
    style = {"fn": ("#9085e9", 38), "var": ("#199e70", 32), "const": ("#c98500", 32)}
    for kind, name in [("fn", "function"), ("var", "variable"), ("const", "constant")]:
        pts = [n for n in nodes if n[3] == kind]
        if not pts:
            continue
        color, size = style[kind]
        fig.add_trace(go.Scatter(
            x=[p[0] for p in pts], y=[p[1] for p in pts], mode="markers+text", text=[p[2] for p in pts],
            textfont=dict(color=T.TEXT, size=12), name=name,
            marker=dict(size=size, color=T.SURFACE_2, line=dict(color=color, width=2.5)),
            hovertemplate="%{text}<extra>" + name + "</extra>"))
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(height=height, title=title)
    return fig


# ---------------------------------------------------------------------------
# Optimizer benchmark landscape
# ---------------------------------------------------------------------------
def landscape_figure(X, Y, Z, points: np.ndarray, best: np.ndarray, color: str, title: str,
                     height: int = 360, optimum=(0.0, 0.0)) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Contour(x=X, y=Y, z=np.log10(Z + 1), colorscale=[[0, "#0d1626"], [1, "#34425e"]],
                             showscale=False, contours=dict(coloring="fill", showlines=True),
                             line=dict(color="#3a4357", width=0.6), hoverinfo="skip", ncontours=18))
    n = len(points)
    if n:
        fig.add_trace(go.Scatter(
            x=points[:, 0], y=points[:, 1], mode="markers", name="evaluated",
            marker=dict(size=6, color=np.arange(n), colorscale=[[0, T.hex_to_rgba(color, 0.15)], [1, color]],
                        line=dict(width=0)),
            customdata=np.arange(1, n + 1),
            hovertemplate="evaluation %{customdata}<br>(%{x:.2f}, %{y:.2f})<extra></extra>"))
    fig.add_trace(go.Scatter(x=[optimum[0]], y=[optimum[1]], mode="markers", name="global optimum",
                             marker=dict(symbol="cross-thin", size=16, line=dict(color=T.TEXT, width=2)),
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=[best[0]], y=[best[1]], mode="markers", name="best found",
                             marker=dict(symbol="star", size=16, color=color, line=dict(color=T.TEXT, width=1.5)),
                             hovertemplate="best found (%{x:.3f}, %{y:.3f})<extra></extra>"))
    fig.update_layout(title=title, height=height, showlegend=False, margin=dict(l=22, r=22, t=46, b=22))
    fig.update_xaxes(range=[X.min(), X.max()], showgrid=False)
    fig.update_yaxes(range=[Y.min(), Y.max()], showgrid=False, scaleanchor="x")
    return fig


def robustness_figure(series: list[dict], metric: str, what: str = "gain", log_y: bool = False,
                      height: int = 380) -> go.Figure:
    """Metric vs perturbation multiplier k on a log x-axis (so ×0.5 and ×2 sit symmetric around 1).

    Plotly quirk on log axes (verified by rendering): *shapes* take data units (k = 1),
    while *annotations* take log10 units (log10(1) = 0).
    """
    fig = go.Figure()
    ks = sorted({float(k) for s in series for k in s["x"]})
    fig.add_shape(type="line", xref="x", yref="paper", x0=1.0, x1=1.0, y0=0, y1=1,
                  line=dict(color=T.MUTED, width=1), layer="below")
    for s in series:
        fig.add_trace(go.Scatter(
            x=s["x"], y=s["y"], name=s["name"], mode="lines+markers",
            line=dict(color=s["color"], width=2, dash=s.get("dash", "solid")),
            marker=dict(size=8, line=dict(color=T.SURFACE, width=1.5)),
            hovertemplate=f"<b>{s['name']}</b><br>{what} × %{{x:.2f}}<br>{metric_label(metric)} = %{{y:.4g}}<extra></extra>"))
    fig.add_annotation(x=0, xref="x", y=1.0, yref="paper", text="nominal model", showarrow=False,
                       yanchor="top", xanchor="left", xshift=5, font=dict(color=T.MUTED, size=11))
    fig.update_layout(height=height, xaxis_title=f"Plant {what} multiplier k (log scale)",
                      yaxis_title=metric_label(metric),
                      title=f"{metric_label(metric, False)} as the plant {what} drifts", hovermode="x unified")
    fig.update_xaxes(type="log", tickvals=ks, ticktext=[f"{k:.2f}" for k in ks])
    if log_y:
        log_axis(fig, "y")
    return fig
