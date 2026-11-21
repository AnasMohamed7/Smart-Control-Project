"""Smart Control — interactive presentation.

Run from the project root:

    streamlit run gui/app.py
"""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def main() -> None:
    import streamlit as st # type: ignore

    st.set_page_config(page_title="Smart Control · Presentation", page_icon="🎛️", layout="wide",
                       initial_sidebar_state="expanded")

    from gui import content
    from gui.core import state
    from gui.core import theme as T
    from gui.views import (compare, conclusions, controllers, lab, metrics, optimizers, overview,
                           playground, results, robustness, system)

    sections = {
        "The story": [
            st.Page(overview.render, title="Overview", icon=":material/home:", url_path="overview", default=True),
            st.Page(system.render, title="System & Plants", icon=":material/account_tree:", url_path="system"),
            st.Page(controllers.render, title="Controllers", icon=":material/tune:", url_path="controllers"),
            st.Page(optimizers.render, title="Optimizers", icon=":material/scatter_plot:", url_path="optimizers"),
            st.Page(metrics.render, title="Metrics & Objective", icon=":material/straighten:", url_path="metrics"),
        ],
        "Results": [
            st.Page(results.render, title="Results Dashboard", icon=":material/dashboard:", url_path="results"),
            st.Page(compare.render, title="Head-to-Head", icon=":material/compare_arrows:", url_path="compare"),
            st.Page(robustness.render, title="Robustness", icon=":material/shield:", url_path="robustness"),
        ],
        "Live": [
            st.Page(lab.render, title="Experiment Lab", icon=":material/science:", url_path="lab"),
            st.Page(playground.render, title="Tuning Playground", icon=":material/sliders:", url_path="playground"),
        ],
        "Wrap-up": [
            st.Page(conclusions.render, title="Conclusions", icon=":material/flag:", url_path="conclusions"),
        ],
    }
    state.PAGES[:] = [p for pages in sections.values() for p in pages]

    page = st.navigation(sections, position="sidebar")
    T.inject_css(hide_sidebar=state.sidebar_hidden())

    with st.sidebar:
        st.markdown(
            f"<div style='padding:.2rem 0 .6rem 0'><div class='sc-eyebrow' style='margin:0'>{T.esc(content.COURSE)}</div>"
            f"<div style='font-weight:800;font-size:1.15rem;color:{T.TEXT}'>{T.esc(content.TITLE)}</div></div>",
            unsafe_allow_html=True)
        rs = state.result_set_picker(key="rs_picker_sidebar")
        if rs is not None:
            st.caption(state.describe(rs))
        st.toggle("Hide sidebar", key="hide_sidebar",
                  help="Hides this sidebar. Bring it back with the “Show sidebar” button at the bottom of every page.")

    page.run()
    state.footer_nav(page.title)


# Streamlit executes this file as "__main__". Multiprocessing workers (used by the
# Experiment Lab) re-import it as "__mp_main__" — they must not build the UI.
if __name__ != "__mp_main__":
    main()
