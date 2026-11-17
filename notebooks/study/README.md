# Optimiser study notebooks

Exploratory notebooks — one per optimisation algorithm — that study how each
algorithm behaves *before* it is used in the full experiment system: does it
optimise at all, which hyper-parameters matter, how to split a fixed budget,
and how it performs on every controller and plant.

| Notebook | Algorithm | Runs on |
|----------|-----------|---------|
| `01_differential_evolution_study.ipynb` | DE | 4 controllers × 3 plants |
| `02_genetic_algorithm_study.ipynb` | GA | 4 controllers × 3 plants |
| `03_particle_swarm_study.ipynb` | PSO | 4 controllers × 3 plants |
| `04_simulated_annealing_study.ipynb` | SA | 4 controllers × 3 plants |
| `05_genetic_programming_study.ipynb` | GP | GP controller × 3 plants (+ GP vs the numeric optimisers on the same encoding) |

Every numeric notebook follows the same structure:

1. **Sanity check** on Sphere, Rastrigin and Rosenbrock (2/5/10-D) and a plot of
   *where* the algorithm samples on 2-D Rastrigin.
2. **Hyper-parameter study** on PID + 3rd-order plant (2-D grid + one 1-D sweep, 3 seeds each).
3. **Budget split**: same 120 evaluations, different population/generation splits
   (SA: J versus number of evaluations).
4. **Full matrix**: every controller on every plant, ≈60 evaluations, seeds 42–44 —
   results table, heat-map, step responses, convergence and control effort.
5. **Takeaways** written from the results.

## Running

The notebooks import the project modules (no copied code) through
`study_utils.py`, which locates the project root automatically. Run them from
this folder:

```bash
.venv/bin/pip install jupyter      # once, if Jupyter is not installed
.venv/bin/jupyter lab notebooks/study
```

Only the course-permitted libraries are used (`numpy`, `matplotlib`, `control`).
Each numeric notebook takes about 5 minutes (Fuzzy-controller runs dominate);
the GP notebook about 3 minutes. The committed outputs come from one full run,
and all randomness is seeded, so re-running reproduces them.
