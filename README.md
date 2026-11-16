# Smart Control Course Project — Starter Structure

This is a **scaffold only**. It contains the proposed folder structure, interfaces,
annotations, and TODO markers. It does not contain implementations of the optimization
algorithms, controllers, simulations, or performance calculations.

## Requirements from the project brief

- Implement DE, GA, GP, PSO, and SA from scratch.
- Develop PID, Neural Network, Fuzzy Logic, and Genetic Programming controllers.
- Test with second-, third-, and fourth-order plants.
- Minimize settling time, rise time, overshoot, steady-state error, and IAE.
- Produce comparison tables, step-response plots, and optimizer convergence curves.
- Use only `control`, `numpy`, and `matplotlib` for Python libraries.
- Keep total execution time at or below five minutes.
- Submit an executable Colab notebook and a presentation.

## Important design note

The brief does not prescribe exact plant transfer functions, neural-network topology,
fuzzy membership functions/rules, GP grammar, parameter bounds, objective weights,
or optimizer settings. These are intentionally left as project decisions/TODOs rather
than silently invented here.

## Suggested workflow

1. Agree on the unresolved design choices listed in `docs/design_decisions.md`.
2. Implement and test one small vertical slice: one plant + PID + one optimizer.
3. Add the remaining controllers and optimizers behind the common interfaces.
4. Run the experiment matrix and inspect the runtime.
5. Generate the notebook, plots, tables, and presentation.

## Suggested environment

Install only the libraries permitted by the brief:

```bash
pip install numpy matplotlib control
```

The scaffold itself is not expected to execute end-to-end until TODOs are implemented.
