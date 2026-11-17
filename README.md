# Smart Control Course Project

This repository implements a comparison framework for tuning controllers on
second-, third-, and fourth-order dynamic plants.

## Project scope

- **Plants:** second-, third-, and fourth-order transfer-function models.
- **Controllers:** PID, Neural Network, Fuzzy Logic, and Genetic Programming controller.
- **Numeric optimizers:** Differential Evolution (DE), Genetic Algorithm (GA),
  Particle Swarm Optimization (PSO), and Simulated Annealing (SA).
- **Additional required algorithm:** Genetic Programming (GP). The current
  tree-based GP optimizer is compatible with the GP controller's tree-vector
  encoding; it is not a general-purpose optimizer for the other controller
  parameter vectors.
- **Metrics:** rise time, settling time, percentage overshoot, steady-state
  error, and Integral of Absolute Error (IAE).
- **Outputs:** experiment comparison CSV, step-response plots, convergence
  plots, and a Colab notebook.

The nominal numeric comparison matrix is 4 optimizers × 4 controllers × 3
plants = 48 experiments. If GP optimizer runs are requested, it adds three
GP-optimizer/GP-controller experiments (one per plant).

## Architecture

```text
smart_control/
├── plants/          # Transfer functions and closed-loop simulation
├── controllers/     # Common controller interface and four controllers
├── optimizers/      # DE, GA, PSO, SA, and GP implementations
├── evaluation/      # Five metrics and scalar objective function
├── experiments/    # Single-run and matrix orchestration; result export
├── visualization/  # Step-response and convergence plots
├── config.py        # Shared simulation and optimizer settings
├── gp_tree.py       # Shared GP expression-tree grammar and encoding
├── types.py         # Shared result data structures
└── main.py          # Project entry point
```

## Dependencies

The project uses the libraries permitted in the course brief:

```bash
pip install control numpy matplotlib
```

`control` is required by the plant model and simulation modules. Standard
Python-library modules (such as `csv`, `json`, `argparse`, and `pathlib`) are
also used for orchestration and result export.

## Run

From the repository root:

```bash
python -m smart_control.main
```

For a reduced integration smoke run:

```bash
python -m smart_control.main --quick
```

The quick run is only for checking that modules connect; it is not suitable
for final comparison results. To include the GP optimizer on the GP controller
only:

```bash
python -m smart_control.main --include-gp
```

By default, the main entry point uses a bounded evaluation budget of about 18
objective evaluations per optimizer/controller/plant combination. This is a
runtime-oriented starting point, not a claim that the optimizers have fully
converged. Measure the complete runtime on the target Colab environment and
adjust the budget only after checking the five-minute requirement.

Results are written to `results/`:

- `experiment_results.csv`
- `step_responses_<plant>.png`
- `convergence_<plant>.png`

## Important design decisions still requiring team/instructor confirmation

The implementation contains plant models and controller designs, but the
course brief does not prescribe all of their numerical details. Confirm the
plant transfer functions, controller architectures, fuzzy rule base, GP grammar,
parameter bounds, objective weights/normalization, and fair-comparison protocol
with the team/instructor. The current values are documented in
`docs/design_decisions.md` and in the corresponding source files.

In particular, the current objective gives nonzero default weight to IAE,
overshoot, settling time, and steady-state error, but zero weight to rise time.
Thus, all five metrics are reported, but rise time does not currently influence
the scalar objective. The raw weighted sum also does not normalize the metrics.
