# Smart Control — interactive presentation

A Streamlit + Plotly app that presents the project as a slide-style story and
doubles as a live analysis tool. It reuses `smart_control` directly (same
runner, simulator, metrics and objective), so every number on screen is
produced by the project code.

## Setup (once)

```bash
python3 -m venv .venv
.venv/bin/pip install -r gui/requirements.txt
```

## Before the talk: precompute results

```bash
.venv/bin/python -m gui.precompute --all      # ≈ 15 min on 8 cores
# or individually:
.venv/bin/python -m gui.precompute --budget course      # ≈ 30 s  (same numbers as smart_control.main)
.venv/bin/python -m gui.precompute --budget extended    # ≈ 4–5 min, 3 seeds
.venv/bin/python -m gui.precompute --budget thorough    # ≈ 10 min, 3 seeds
```

Result sets are saved in `results/gui/` (git-ignored) and appear in the
sidebar selector.

## Present

```bash
.venv/bin/streamlit run gui/app.py
```

* **Prev / Next** buttons at the bottom of every page walk through the story.
* **Hide sidebar** (sidebar toggle or footer button) hides the sidebar; the footer button brings it back.
* The **result set** chosen in the sidebar drives every analysis page.
* Every chart has a camera button (top-right on hover) that exports a 3× PNG for slides or the report.

## Pages

| Section  | Page | What it shows |
|----------|------|---------------|
| Story    | Overview | Title, team, headline numbers, pipeline |
|          | System & Plants | Block diagram, transfer functions, step / pole / Bode views |
|          | Controllers | Law, parameters, PID ablation, NN weights, fuzzy surface, evolved GP trees |
|          | Optimisers | Algorithm cards and a live 2-D benchmark of how each one searches |
|          | Metrics & Objective | Annotated response explaining all five metrics; the objective and its penalties |
| Results  | Results Dashboard | Heatmaps, rankings, step responses, convergence, seed spread, runtime, CSV export, objective re-weighting |
|          | Head-to-Head | Up to 4 runs side-by-side (across result sets) or an optimiser duel |
|          | Robustness | Tuned controllers re-simulated on plants with drifted gain or speed |
| Live     | Experiment Lab | Run any slice of the matrix in parallel; manage result sets |
|          | Tuning Playground | Hand-tune PID / fuzzy gains live against the optimiser's result |
| Wrap-up  | Conclusions | Findings computed from the data, budget effect, limitations, future work |

## Editing the text

Title, team, objectives, limitations and future work live in `gui/content.py`.

## Layout

```text
gui/
├── app.py            # entry point: navigation, sidebar
├── content.py        # editable presentation text
├── precompute.py     # CLI to build result sets
├── core/
│   ├── theme.py      # colours (validated palettes), Plotly template, CSS, HTML components
│   ├── catalog.py    # registry of plants/controllers/optimisers + descriptive copy
│   ├── engine.py     # parallel runner around smart_control's run_single_experiment, storage
│   ├── analysis.py   # tidy frames, aggregation over seeds, rankings, re-scoring
│   ├── charts.py     # every Plotly figure
│   └── state.py      # cached loading, active result set, prev/next navigation
└── views/            # one module per page
```
