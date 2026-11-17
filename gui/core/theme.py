"""Visual design system for the presentation app.

One place for every color, the Plotly chart template and the small HTML
components (hero, stat tiles, cards, pills) the pages are built from.

Categorical palettes were validated (dark mode, surface #141821) with the
dataviz palette validator: lightness band, chroma floor, adjacent-pair CVD
separation and normal-vision floor all pass.  Colors follow the *entity*
(an optimizer is always the same hue on every page), never its rank.
"""
from __future__ import annotations

import html
from typing import Iterable

import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ---------------------------------------------------------------------------
# Tokens
# ---------------------------------------------------------------------------
BG = "#0b0e14"          # page plane
SURFACE = "#141821"     # cards / chart surface
SURFACE_2 = "#1b2130"   # raised (hover labels, tiles)
BORDER = "#262c3a"
TEXT = "#eef0f4"        # primary ink
TEXT_2 = "#b7bcc8"      # secondary ink
MUTED = "#8a90a0"       # axis labels, captions
GRID = "#212736"        # hairline grid
AXIS = "#343b4b"        # baseline
ACCENT = "#3987e5"

# Status (reserved: only ever means good / warning / bad)
GOOD = "#0ca30c"
WARN = "#fab219"
BAD = "#d03b3b"

# Entity palettes (validated sets, fixed order)
OPTIMIZER_COLORS = {
    "DE": "#3987e5",   # blue
    "GA": "#d95926",   # orange
    "PSO": "#199e70",  # aqua
    "SA": "#c98500",   # yellow
    "GP": "#9085e9",   # violet
}
CONTROLLER_COLORS = {
    "PID": "#d55181",             # magenta
    "Neural Network": "#0ca30c",  # green
    "Fuzzy": "#9085e9",           # violet
    "GP Controller": "#e66767",   # red
}
# Plants are ordered (2nd < 3rd < 4th) -> an ordinal blue ramp.
PLANT_COLORS = {
    "second_order": "#86b6ef",
    "third_order": "#3987e5",
    "fourth_order": "#1c5cab",
}
# Generic slots for user-built selections (head-to-head picks).
SLOT_COLORS = ["#3987e5", "#d95926", "#199e70", "#c98500"]
SLOT_DASHES = ["solid", "dash", "dot", "dashdot"]

# Sequential (magnitude) ramp on a dark surface: low recedes, high glows.
# Capped at #3987e5 so the white in-cell labels keep >= 3:1 contrast everywhere.
SEQUENTIAL = [
    [0.0, "#111b2c"], [0.25, "#13305a"], [0.5, "#184f95"],
    [0.75, "#256abf"], [1.0, "#3987e5"],
]
# Diverging blue <-> red with a neutral gray midpoint.
DIVERGING = [[0.0, "#3987e5"], [0.5, "#383835"], [1.0, "#e66767"]]

FONT_STACK = "Inter, system-ui, -apple-system, 'Segoe UI', Roboto, sans-serif"
MONO_STACK = "'JetBrains Mono', ui-monospace, SFMono-Regular, Menlo, monospace"


def color_for(kind: str, name: str) -> str:
    table = {"optimizer": OPTIMIZER_COLORS, "controller": CONTROLLER_COLORS,
             "plant": PLANT_COLORS}[kind]
    return table.get(name, MUTED)


def hex_to_rgba(hex_color: str, alpha: float) -> str:
    h = hex_color.lstrip("#")
    r, g, b = int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"


# ---------------------------------------------------------------------------
# Plotly template
# ---------------------------------------------------------------------------
def _register_template() -> None:
    axis = dict(
        gridcolor=GRID, gridwidth=1, zeroline=False, showline=True,
        linecolor=AXIS, linewidth=1, ticks="outside", tickcolor=AXIS, ticklen=4,
        tickfont=dict(color=MUTED, size=12), title=dict(font=dict(color=TEXT_2, size=13)),
        automargin=True,
    )
    template = go.layout.Template()
    template.layout = go.Layout(
        font=dict(family=FONT_STACK, color=TEXT_2, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=list(OPTIMIZER_COLORS.values()),
        xaxis=axis, yaxis=axis,
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(color=TEXT_2, size=12),
                    orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0,
                    itemclick="toggleothers", itemdoubleclick="toggle"),
        hoverlabel=dict(bgcolor=SURFACE_2, bordercolor=BORDER,
                        font=dict(family=FONT_STACK, color=TEXT, size=13)),
        hovermode="closest",
        margin=dict(l=22, r=22, t=56, b=48),
        title=dict(font=dict(color=TEXT, size=15), x=0.012, xanchor="left", y=0.975),
        polar=dict(bgcolor="rgba(0,0,0,0)",
                   radialaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=MUTED)),
                   angularaxis=dict(gridcolor=GRID, linecolor=AXIS, tickfont=dict(color=TEXT_2))),
        scene=dict(
            xaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED),
            yaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED),
            zaxis=dict(backgroundcolor="rgba(0,0,0,0)", gridcolor=GRID, color=MUTED),
        ),
        coloraxis=dict(colorbar=dict(outlinewidth=0, tickfont=dict(color=MUTED), thickness=10)),
    )
    template.data.scatter = [go.Scatter(line=dict(width=2), marker=dict(size=8))]
    template.data.bar = [go.Bar(marker=dict(line=dict(width=0)))]
    template.data.heatmap = [go.Heatmap(colorbar=dict(outlinewidth=0, thickness=10,
                                                      tickfont=dict(color=MUTED)))]
    pio.templates["smartctl"] = template
    pio.templates.default = "smartctl"


_register_template()

PLOT_CONFIG = {
    "displaylogo": False,
    "modeBarButtonsToRemove": ["lasso2d", "select2d", "autoScale2d"],
    "toImageButtonOptions": {"format": "png", "scale": 3},
}


def show(fig: go.Figure, height: int | None = None, key: str | None = None) -> None:
    """Render a Plotly figure with the app's standard config."""
    if height is not None:
        fig.update_layout(height=height)
    st.plotly_chart(fig, config=PLOT_CONFIG, key=key, theme=None)


# ---------------------------------------------------------------------------
# Global CSS
# ---------------------------------------------------------------------------
_CSS = f"""
<style>

:root {{
  --bg: {BG}; --surface: {SURFACE}; --surface-2: {SURFACE_2}; --border: {BORDER};
  --text: {TEXT}; --text-2: {TEXT_2}; --muted: {MUTED}; --accent: {ACCENT};
  --good: {GOOD}; --warn: {WARN}; --bad: {BAD};
}}
[data-testid="stIconMaterial"], .material-symbols-rounded {{ font-family: 'Material Symbols Rounded' !important; }}
.stApp {{ background: radial-gradient(1200px 600px at 10% -10%, rgba(57,135,229,0.10), transparent 60%),
                      radial-gradient(900px 500px at 110% 10%, rgba(144,133,233,0.07), transparent 60%),
                      var(--bg); }}
.block-container {{ padding-top: 2.2rem; padding-bottom: 3rem; max-width: 1380px; }}
h1, h2, h3, h4 {{ color: var(--text); letter-spacing: -0.02em; }}
h1 {{ font-weight: 800; }}
h2 {{ font-weight: 700; font-size: 1.55rem; margin-top: .6rem; }}
h3 {{ font-weight: 650; font-size: 1.15rem; }}
p, li {{ color: var(--text-2); line-height: 1.6; }}
code {{ font-family: {MONO_STACK}; }}
[data-testid="stSidebar"] {{ background: #0d1119; border-right: 1px solid var(--border); }}
[data-testid="stSidebarNav"] span {{ font-size: .93rem; }}
[data-testid="stHeader"] {{ background: transparent; }}
div[data-testid="stTabs"] button p {{ font-size: .95rem; font-weight: 600; }}
div[data-testid="stTabs"] [data-baseweb="tab-highlight"] {{ background: var(--accent); }}
div[data-testid="stExpander"] details {{ border-color: var(--border); background: var(--surface); border-radius: 12px; }}
div[data-testid="stDataFrame"] {{ border: 1px solid var(--border); border-radius: 12px; }}
div[data-testid="stPlotlyChart"] {{ background: var(--surface); border: 1px solid var(--border);
  border-radius: 14px; overflow: hidden; }}
.stButton > button, .stDownloadButton > button {{ border-radius: 10px; font-weight: 600; }}
button[kind="primary"], button[kind="primary"] p {{ color: #ffffff !important; }}

/* ---------- components ---------- */
.sc-eyebrow {{ text-transform: uppercase; letter-spacing: .16em; font-size: .74rem; font-weight: 700;
  color: var(--accent); margin-bottom: .35rem; }}
.sc-title {{ font-size: 2.15rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em;
  line-height: 1.15; margin: 0 0 .4rem 0; }}
.sc-sub {{ font-size: 1.05rem; color: var(--text-2); max-width: 900px; margin-bottom: 1.2rem; }}
.sc-hero {{ position: relative; overflow: hidden; border: 1px solid var(--border); border-radius: 22px;
  padding: 3.0rem 2.8rem 2.6rem 2.8rem; margin-bottom: 1.4rem;
  background: linear-gradient(135deg, rgba(57,135,229,0.16), rgba(20,24,33,0.85) 45%, rgba(144,133,233,0.12)); }}
.sc-hero h1 {{ font-size: 3.0rem; line-height: 1.06; margin: .3rem 0 .8rem 0; letter-spacing: -0.035em; }}
.sc-hero .grad {{ background: linear-gradient(90deg, #86b6ef, #9085e9 55%, #e87ba4);
  -webkit-background-clip: text; background-clip: text; color: transparent; }}
.sc-hero p {{ font-size: 1.12rem; max-width: 640px; color: var(--text-2); position: relative; }}
.sc-hero h1, .sc-hero .meta, .sc-hero .sc-eyebrow {{ position: relative; z-index: 1; }}
.sc-hero .meta {{ margin-top: 1.4rem; display: flex; flex-wrap: wrap; gap: .5rem; }}
.sc-hero svg.wave {{ position: absolute; right: -70px; bottom: -40px; width: 470px; height: auto; opacity: .3; pointer-events: none; z-index: 0; }}
@media (max-width: 1100px) {{ .sc-hero svg.wave {{ display: none; }} }}
.sc-pill {{ display: inline-flex; align-items: center; gap: .4rem; padding: .28rem .7rem; border-radius: 999px;
  border: 1px solid var(--border); background: rgba(255,255,255,0.03); color: var(--text-2);
  font-size: .82rem; font-weight: 600; white-space: nowrap; }}
.sc-dot {{ width: 9px; height: 9px; border-radius: 50%; display: inline-block; }}
.sc-grid {{ display: grid; gap: 14px; margin: .4rem 0 1.1rem 0; }}
.sc-stat {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.0rem 1.15rem; }}
.sc-stat .label {{ font-size: .78rem; text-transform: uppercase; letter-spacing: .1em; color: var(--muted); font-weight: 700; }}
.sc-stat .value {{ font-size: 2.0rem; font-weight: 800; color: var(--text); letter-spacing: -0.03em; line-height: 1.15; margin-top: .25rem; }}
.sc-stat .note {{ font-size: .85rem; color: var(--text-2); margin-top: .2rem; }}
.sc-card {{ background: var(--surface); border: 1px solid var(--border); border-radius: 16px; padding: 1.15rem 1.25rem;
  height: 100%; }}
.sc-card h4 {{ margin: 0 0 .35rem 0; font-size: 1.05rem; display: flex; align-items: center; gap: .5rem; padding: 0; }}
.sc-card h4 .sc-dot, .sc-pill .sc-dot {{ margin-right: .45rem; flex: 0 0 auto; }}
.sc-card p {{ margin: 0; font-size: .93rem; }}
.sc-card ul {{ margin: .4rem 0 0 1.1rem; padding: 0; }}
.sc-card li {{ font-size: .9rem; margin: .1rem 0; }}
.sc-card .tag {{ font-family: {MONO_STACK}; font-size: .78rem; color: var(--muted); }}
.sc-callout {{ border-left: 3px solid var(--accent); background: rgba(57,135,229,0.07); border-radius: 0 12px 12px 0;
  padding: .85rem 1.1rem; margin: .6rem 0 1rem 0; color: var(--text-2); }}
.sc-callout.warn {{ border-left-color: var(--warn); background: rgba(250,178,25,0.07); }}
.sc-callout.good {{ border-left-color: var(--good); background: rgba(12,163,12,0.07); }}
.sc-callout b {{ color: var(--text); }}
.sc-step {{ display: flex; gap: .9rem; align-items: flex-start; padding: .7rem 0; border-bottom: 1px solid var(--border); }}
.sc-step:last-child {{ border-bottom: none; }}
.sc-step .n {{ flex: 0 0 30px; height: 30px; border-radius: 9px; display: grid; place-items: center;
  background: rgba(57,135,229,0.14); color: #86b6ef; font-weight: 800; font-size: .9rem; }}
.sc-step .t {{ color: var(--text); font-weight: 650; }}
.sc-step .d {{ color: var(--text-2); font-size: .92rem; }}
.sc-kv {{ display: grid; grid-template-columns: max-content 1fr; gap: .3rem 1rem; font-size: .9rem; }}
.sc-kv .k {{ color: var(--muted); }}
.sc-kv .v {{ color: var(--text); font-family: {MONO_STACK}; font-size: .85rem; }}
.sc-status {{ display:inline-flex; align-items:center; gap:.35rem; font-weight:700; font-size:.82rem; }}
.sc-footer {{ margin-top: 2.2rem; padding-top: 1rem; border-top: 1px solid var(--border); }}
.sc-progress {{ height: 4px; border-radius: 4px; background: var(--border); overflow: hidden; margin: .2rem 0 .8rem 0; }}
.sc-progress > div {{ height: 100%; background: linear-gradient(90deg, #3987e5, #9085e9); }}
.sc-svg {{ width: 100%; height: auto; display: block; }}
.sc-empty {{ display: flex; gap: 1rem; align-items: flex-start; background: var(--surface); border: 1px dashed var(--border);
  border-radius: 14px; padding: 1.1rem 1.3rem; margin: .4rem 0 1rem 0; }}
.sc-empty .i {{ flex: 0 0 38px; height: 38px; border-radius: 10px; display: grid; place-items: center;
  background: rgba(138,144,160,0.12); color: var(--muted); font-size: 1.2rem; font-weight: 700; }}
.sc-empty .t {{ color: var(--text); font-weight: 700; margin-bottom: .15rem; }}
.sc-empty .d {{ color: var(--text-2); font-size: .92rem; }}
@media (max-width: 760px) {{
  .sc-hero {{ padding: 1.6rem 1.2rem; }}
  .sc-hero h1 {{ font-size: 2.0rem; }}
  .sc-title {{ font-size: 1.6rem; }}
}}
</style>
"""

_HIDE_SIDEBAR_CSS = """
<style>
[data-testid="stSidebar"], [data-testid="stSidebarCollapsedControl"] { display: none !important; }
</style>
"""


def inject_css(hide_sidebar: bool = False) -> None:
    st.markdown(_CSS, unsafe_allow_html=True)
    if hide_sidebar:
        st.markdown(_HIDE_SIDEBAR_CSS, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# HTML components
# ---------------------------------------------------------------------------
def esc(text) -> str:
    return html.escape(str(text))


def page_header(eyebrow: str, title: str, subtitle: str = "") -> None:
    sub = f'<div class="sc-sub">{subtitle}</div>' if subtitle else ""
    st.markdown(
        f'<div class="sc-eyebrow">{esc(eyebrow)}</div>'
        f'<div class="sc-title">{esc(title)}</div>{sub}',
        unsafe_allow_html=True,
    )


def pill(text: str, color: str | None = None) -> str:
    dot = f'<span class="sc-dot" style="background:{color}"></span>' if color else ""
    return f'<span class="sc-pill">{dot}{esc(text)}</span>'


def stat_tiles(items: Iterable[tuple], min_width: int = 180) -> None:
    """items: (label, value, note) tuples rendered as a responsive tile row."""
    cells = []
    for item in items:
        label, value = item[0], item[1]
        note = item[2] if len(item) > 2 and item[2] else ""
        note_html = f'<div class="note">{note}</div>' if note else ""
        cells.append(f'<div class="sc-stat"><div class="label">{esc(label)}</div>'
                     f'<div class="value">{value}</div>{note_html}</div>')
    st.markdown(
        f'<div class="sc-grid" style="grid-template-columns: repeat(auto-fit, minmax({min_width}px, 1fr));">'
        + "".join(cells) + "</div>",
        unsafe_allow_html=True,
    )


def card(title: str, body_html: str, accent: str | None = None, icon: str = "") -> str:
    dot = f'<span class="sc-dot" style="background:{accent}"></span>' if accent else ""
    style = f' style="border-top: 3px solid {accent};"' if accent else ""
    head = f"<h4>{dot}{icon}{esc(title)}</h4>" if title else ""
    return f'<div class="sc-card"{style}>{head}{body_html}</div>'


def card_grid(cards: list[str], min_width: int = 240) -> None:
    st.markdown(
        f'<div class="sc-grid" style="grid-template-columns: repeat(auto-fit, minmax({min_width}px, 1fr));">'
        + "".join(cards) + "</div>",
        unsafe_allow_html=True,
    )


def callout(html_text: str, kind: str = "") -> None:
    st.markdown(f'<div class="sc-callout {kind}">{html_text}</div>', unsafe_allow_html=True)


def steps(items: list[tuple[str, str]]) -> str:
    rows = "".join(
        f'<div class="sc-step"><div class="n">{i}</div><div><div class="t">{esc(t)}</div>'
        f'<div class="d">{d}</div></div></div>'
        for i, (t, d) in enumerate(items, 1)
    )
    return f"<div>{rows}</div>"


def kv(pairs: list[tuple[str, str]]) -> str:
    return '<div class="sc-kv">' + "".join(
        f'<div class="k">{esc(k)}</div><div class="v">{esc(v)}</div>' for k, v in pairs) + "</div>"


def status_badge(ok: bool, yes: str = "Settled", no: str = "Not settled") -> str:
    color, icon, label = (GOOD, "✔", yes) if ok else (BAD, "✖", no)
    return f'<span class="sc-status" style="color:{color}">{icon} {esc(label)}</span>'
