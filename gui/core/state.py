"""Session state, cached data access and slide-style navigation."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from gui.core import analysis, engine
from gui.core import theme as T

PAGES: list = []          # filled by app.py, in presentation order


# ---------------------------------------------------------------------------
# Result sets
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False, max_entries=16)
def _load(path_str: str, mtime: float) -> engine.ResultSet:
    return engine.load_result_set(Path(path_str))


@st.cache_data(show_spinner=False, max_entries=16)
def _frame(path_str: str, mtime: float) -> pd.DataFrame:
    return analysis.records_frame(_load(path_str, mtime).records)


def available_sets() -> list[Path]:
    return engine.list_result_sets()


def set_label(path: Path) -> str:
    try:
        return _load(str(path), path.stat().st_mtime).name
    except Exception:
        return path.stem


def set_active(path: Path | str | None) -> None:
    """Request a new active result set; applied before the picker is created on the next run."""
    st.session_state["rs_pending"] = "" if path is None else str(path)


def active_path() -> Path | None:
    if "rs_pending" in st.session_state:
        pending = st.session_state.pop("rs_pending")
        if pending:
            st.session_state["rs_path"] = pending
        else:
            st.session_state.pop("rs_path", None)
    sets = available_sets()
    if not sets:
        return None
    chosen = st.session_state.get("rs_path")
    paths = [str(p) for p in sets]
    if chosen not in paths:
        # Prefer a multi-seed set for the analysis pages when one exists.
        preferred = next((p for p in sets if "3-seeds" in p.stem and "extended" in p.stem), sets[0])
        st.session_state["rs_path"] = str(preferred)
    return Path(st.session_state["rs_path"])


def load(path: Path | None = None) -> engine.ResultSet | None:
    path = path or active_path()
    if path is None or not path.exists():
        return None
    return _load(str(path), path.stat().st_mtime)


def frame(path: Path | None = None) -> pd.DataFrame:
    path = path or active_path()
    if path is None or not path.exists():
        return pd.DataFrame()
    return _frame(str(path), path.stat().st_mtime)


def clear_caches() -> None:
    _load.clear()
    _frame.clear()


def result_set_picker(label: str = "Result set", key: str = "rs_picker_main", help: str | None = None) -> engine.ResultSet | None:
    """Selector bound to the global active result set."""
    sets = available_sets()
    if not sets:
        return None
    paths = [str(p) for p in sets]
    active_path()   # normalises st.session_state["rs_path"] before the widget binds to it
    st.selectbox(label, paths, format_func=lambda p: set_label(Path(p)), key="rs_path",
                 help=help or "Every analysis page reads the selected result set. Build new ones in the Experiment Lab.")
    return load()


def require_results() -> engine.ResultSet | None:
    rs = load()
    if rs is None:
        T.callout("<b>No result sets yet.</b> Open the <b>Experiment Lab</b> and run the matrix, or run "
                  "<code>python -m gui.precompute --all</code> from the project root.", "warn")
    return rs


def describe(rs: engine.ResultSet) -> str:
    m = rs.meta
    seeds = m.get("seeds", [])
    return (f"{m.get('n_runs', len(rs.records))} runs · ≈{m.get('evaluations', '?')} evaluations each · "
            f"{len(seeds)} seed{'s' if len(seeds) != 1 else ''} · built {m.get('created', '')[:16].replace('T', ' ')}")


def available(df: pd.DataFrame, column: str, order: list[str], **filters) -> list[str]:
    """Values of ``column`` that have at least one successful run (optionally filtered), in display order."""
    if df is None or df.empty:
        return []
    sub = df[df["status"] == "success"]
    for col, value in filters.items():
        if value is not None:
            sub = sub[sub[col].astype(str) == str(value)]
    present = set(sub[column].astype(str))
    return [v for v in order if v in present]


def no_data(what: str, hint: str = "") -> None:
    """Friendly empty state used whenever a selection has no runs in the active result set."""
    rs = load()
    name = T.esc(rs.name) if rs is not None else "the active result set"
    extra = f"<div class='d'>{hint}</div>" if hint else ""
    st.markdown(
        f"<div class='sc-empty'><div class='i'>∅</div><div><div class='t'>No data for {T.esc(what)}</div>"
        f"<div class='d'><b>{name}</b> does not contain these runs. Pick another option, choose a different "
        f"result set in the sidebar, or run them in the <b>Experiment Lab</b>.</div>{extra}</div></div>",
        unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Sidebar visibility & navigation
# ---------------------------------------------------------------------------
def sidebar_hidden() -> bool:
    return bool(st.session_state.get("hide_sidebar", False))


def _toggle_sidebar() -> None:
    st.session_state["hide_sidebar"] = not sidebar_hidden()


def footer_nav(current_title: str) -> None:
    titles = [p.title for p in PAGES]
    if current_title not in titles:
        return
    i = titles.index(current_title)
    st.markdown('<div class="sc-footer"></div>', unsafe_allow_html=True)
    pct = (i + 1) / len(PAGES)
    st.markdown(f'<div class="sc-progress"><div style="width:{pct*100:.0f}%"></div></div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 2, 1])
    with c1:
        if i > 0 and st.button(f"←  {PAGES[i-1].title}", key="nav_prev", width="stretch"):
            st.switch_page(PAGES[i - 1])
    with c2:
        label = "Show sidebar" if sidebar_hidden() else "Hide sidebar"
        st.markdown(f"<div style='text-align:center;color:{T.MUTED};font-size:.85rem;padding-top:.45rem'>"
                    f"{i + 1} / {len(PAGES)}</div>", unsafe_allow_html=True)
        st.button(label, key="nav_sidebar", width="stretch", type="tertiary", on_click=_toggle_sidebar)
    with c3:
        if i < len(PAGES) - 1 and st.button(f"{PAGES[i+1].title}  →", key="nav_next", width="stretch",
                                            type="primary"):
            st.switch_page(PAGES[i + 1])
