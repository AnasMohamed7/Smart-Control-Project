# Code audit and integration notes

This review was performed against the uploaded project snapshot before removing
member smoke-test scripts and temporary test files.

## Corrections applied in this version

1. **Experiment orchestration implemented** in `smart_control/experiments/runner.py`.
   It creates fresh controller/optimizer instances, uses the shared parameter
   bounds, builds the objective, reruns the best candidate, computes its metrics,
   and stores the response and convergence history. The matrix runner retains
   failed runs as explicit records rather than silently dropping them.
2. **Result handling implemented** in `smart_control/experiments/results.py`.
   It flattens records, prints a summary, and writes CSV using the Python standard
   library.
3. **Main entry point implemented** in `smart_control/main.py`.
   It constructs the 48-run numeric matrix, saves the CSV and per-plant plots,
   measures runtime, and optionally adds GP optimizer runs for GPController.
4. **Neural Network reset corrected** in `controllers/neural_network.py`.
   Previously `reset()` reinitialized the weights. Since the simulator calls
   `reset()` after the optimizer sets a candidate, every NN candidate was being
   discarded before simulation. Reset now clears only dynamic state.
5. **PSO aligned with the shared optimizer contract** in
   `optimizers/particle_swarm.py`: local RNG, validated bounds, finite-safe
   counted objective, and evaluation metadata. It no longer changes NumPy's
   global random state.
6. **Objective simulation setup simplified** in `evaluation/objective.py`.
   It no longer runs a redundant probe simulation before optimization and no
   longer imports SciPy directly as a fallback. It uses the project's common
   closed-loop simulator unless a simulator callable is supplied.
7. **Documentation updated** to describe the implemented project rather than
   calling the repository a scaffold.

## Issues still requiring a team decision or follow-up

### Major: the default objective does not optimize all five criteria

The active defaults are IAE = 1.0, overshoot = 0.5, settling time = 0.3,
and steady-state error = 1.0. Rise time has a zero default weight. Thus rise
time is calculated and reported, but does not influence the optimizer's fitness
by default. In addition, the current objective uses a raw weighted sum without
normalizing metrics with different units/scales. The team must agree on a
normalization method and a nonzero rise-time weight (or another multi-objective
method) before claiming that all five criteria are optimized. No weights were
invented in this implementation.

### Major: confirm plant/controller design choices

`plants/models.py` contains three concrete transfer functions, and the
controller modules contain concrete architectures/rules/bounds. The design
checklist still has unanswered approval boxes. The team should confirm that
these are the agreed models and not assume the instructor specified them.

### Major: fair evaluation budget

The config defaults use population-based optimizers with population 30 and 100
iterations, while SA has 300 iterations. These do not have equal objective
function evaluation budgets. `main.py` therefore applies an explicit starting
budget of approximately 18 objective evaluations per run (6 initial members
plus 2 generations for GA/DE/PSO; one initial point plus 17 moves for SA).
This is a runtime-oriented compromise. It must be benchmarked on the actual
Colab environment, and the final report should state the evaluation budget.

### Major: a completed run is not necessarily a successful control response

The runner's `status='success'` means the optimizer/simulation completed without
raising an exception. It does not mean the output is stable or settled. The
result records therefore also expose `simulation_finite` and `settled` flags.
Inspect the response metrics and plots; the quick validation showed that many
Neural Network and GP-controller runs did not settle, and most fourth-order
combinations did not settle. The current low evaluation budget is not sufficient
to claim good closed-loop performance across all plants.

### Major: five-minute runtime is not yet certified

The five-minute limit cannot be guaranteed from static inspection. Fuzzy
controller simulations are materially slower than the PID/NN/GP simulations.
Run the complete matrix on the target machine and record the measured time.
The `--quick` option is only a smoke test and is not a final experiment.

### Major: GP optimizer scope

The current `GeneticProgramming` optimizer validates that the bounds represent
a GP expression-tree vector. It therefore cannot optimize PID, NN, or Fuzzy
parameter vectors. The matrix runner runs GP only with `GPController` when the
optional GP optimizer is enabled. Confirm this interpretation with the
instructor if GP is expected to optimize other controller types as well.

### Minor: PID derivative-filter test helper

One temporary test asserted that a filtered derivative must produce a smaller
maximum output than an unfiltered derivative. Its test signal drives both
outputs into the PID's ±50 output saturation, so both maxima equal 50 and the
assertion fails even though filtering reduces the internal derivative. A better
test should use a smaller input spike or a higher test-only output limit.

### Environment limitation during this review

The execution environment did not have the `control` package installed, and
network access prevented installing it. Syntax checks and most component tests
were run; plant integration was additionally exercised with a temporary local
compatibility shim backed by SciPy. That shim was not included in the project.
Run the final validation with the actual `control` package installed.

## Validation performed

- `compileall` completed successfully after the changes.
- 107 unit/component tests passed with the temporary compatibility shim when
  excluding the known saturation-sensitive PID test and selected long-running
  plant/optimizer integration tests.
- A synthetic runner integration check passed for GA, DE, PSO, and SA.
- The complete 48-case matrix completed with the entry-point budget using the
  temporary compatibility shim: 48/48 runs succeeded, and CSV plus plots were
  generated in approximately 32 seconds in this environment.
- The optional GP-inclusive matrix completed 51/51 runs with reduced quick-mode
  budgets, including the three GP-optimizer/GP-controller runs.
- The notebook `notebooks/Smart_Control_Project.ipynb` was added to upload the
  project ZIP, install the permitted dependencies, execute the matrix, and
  download results. These timings do not certify the runtime with the actual
  `control` package or the final Colab runtime.
