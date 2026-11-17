# Design decisions to agree on before implementation

The course brief defines the required algorithms, controllers, plants, metrics, and
deliverables, but leaves the following details open. The team should record and approve
these choices before parallel coding begins.

## 1. Plant models

- [ ] Exact second-order transfer function:
- [ ] Exact third-order transfer function:
- [ ] Exact fourth-order transfer function:
- [ ] Input/output assumptions:
- [ ] Simulation horizon and time step:
- [ ] Unity feedback or another feedback structure:

## 2. Controller definitions

### PID
- [ ] Parameter bounds for Kp, Ki, Kd:
- [ ] Derivative filtering / output limits:
- [ ] Anti-windup approach, if used:

### Neural Network
- [ ] Inputs to the controller (for example, error and error change):
- [ ] Network architecture:
- [ ] Activation functions:
- [ ] Output scaling / saturation:
- [ ] Which weights and biases are optimized:

### Fuzzy Logic
- [ ] Input variables:
- [ ] Membership-function types and counts:
- [ ] Rule base:
- [ ] Inference and defuzzification methods:
- [ ] Which membership/rule parameters are optimized:

### Genetic Programming controller
- [ ] Function and terminal sets (GP grammar):
- [ ] Maximum tree depth / size:
- [ ] How the expression maps inputs to control output:
- [ ] Output limits and protection against invalid expressions:

## 3. Optimization setup

- [ ] Common objective function or multi-objective reporting approach:
- [ ] Metric normalization method:
- [ ] Weights, if using a weighted objective:
- [ ] Parameter bounds for every controller:
- [ ] Population size / number of candidates:
- [ ] Iteration or evaluation budget:
- [ ] Random seed policy and number of independent runs:
- [ ] Constraint handling and invalid-candidate penalty:

## 4. Fair comparison

- [ ] Same evaluation budget for each optimizer?
- [ ] Same number of independent runs?
- [ ] Same plant, reference, simulation horizon, and metrics?
- [ ] How will best, mean, and standard deviation be reported?

## 5. Runtime budget

- [ ] Measure simulation cost per candidate.
- [ ] Measure total cost for the complete experiment matrix.
- [ ] Reduce evaluation budget or optimize simulation only if needed.
- [ ] Record actual total runtime; do not claim the five-minute limit without measuring.

## 6. Interpretation of the project brief

The brief asks for GA, DE, PSO, and SA to optimize the four controller types.
It separately requires an implementation of Genetic Programming and lists GP as a
controller type. Confirm with the instructor whether GP is also expected to optimize
other controllers, or whether its role is limited to the GP controller.

## 7. Current implementation values (recorded for review; approval pending)

These are the values currently present in the source code. They are **not**
marked as instructor-prescribed or team-approved until the team confirms them.

### 7.1 Plants and simulation

- Second order: `G(s) = 1 / (s^2 + 1.2s + 1)`.
- Third order: `G(s) = 10 / (s^3 + 6s^2 + 11s + 6)`.
- Fourth order: `G(s) = 1 / (s^4 + 2.6s^3 + 3.44s^2 + 2.4s + 0.64)`.
- Simulation uses unity negative feedback, `DT = 0.01 s`, `T_END = 20 s`, and a unit reference.

### 7.2 Controllers

- PID parameters: `[Kp, Ki, Kd]`; bounds are `[0,50]`, `[0,20]`, `[0,10]`; integral and output limits are 50.
- Neural Network: 2-5-1 feed-forward network; inputs `[e(k), e(k-1)]`; hidden `tanh`, linear output; 21 optimized weights/biases.
- Fuzzy: two-input Mamdani controller; five triangular sets per input/output; 25-rule table; centroid defuzzification; optimized gains `[K_e, K_de, K_u]`.
- GP controller: expression tree over `e`, `de`, `ie`; function set `add`, `sub`, `mul`, protected `div`, `tanh`; tree depth 3; numeric vector encoding has 30 parameters.

### 7.3 Current objective configuration

- Current nonzero weights: IAE = 1.0, overshoot = 0.5, settling time = 0.3, steady-state error = 1.0.
- Rise time currently has zero default weight.
- Metrics are combined as a raw weighted sum; they are not normalized in the current implementation.

### 7.4 Current entry-point evaluation budget

- DE/GA/PSO: population 6, 2 iterations (about 18 evaluations for DE/PSO; GA may make fewer calls because elites are copied without reevaluation).
- SA: 17 moves plus its initial evaluation (18 evaluations).
- These are runtime-oriented entry-point overrides, not the underlying algorithm defaults in `config.py`.

## 8. Genetic Programming (owner: Fayad)

GP-specific decisions. Items marked **(team)** need the team's approval.

### 8.1 Two different GP components
- `optimizers/genetic_programming.py` evolves expression trees (the optimizer).
- `controllers/gp_controller.py` evaluates one tree to produce `u` (the controller).
- Both share `smart_control/gp_tree.py`; neither depends on the other's internals.

### 8.2 Tree representation
- Nested tuples: variable `"e"`, constant `2.5`, function `("add", left, right)`, `("tanh", child)`.
- Example PID law: `("add", ("add", ("mul", Kp, "e"), ("mul", Ki, "ie")), ("mul", Kd, "de"))`.

### 8.3 Function and terminal sets **(team)**
- Functions: `add`, `sub`, `mul`, `div` (protected), `tanh`.
- Variables: `e` (error), `de` (error derivative), `ie` (error integral); plus constants in [-10, 10].
- Controller-side definitions (Khaira to confirm): `de = (e - e_prev)/dt`, 0 on the first step; `ie += e*dt`, clamped to +-50; output passed through `gp_tree.safe_output(value, 50)`.
- No exp/pow/log/sqrt, so no overflow or domain errors.

### 8.4 Protection against invalid operations
- `div(a, b) = 1.0` if `|b| <= 1e-6` or `b` is NaN.
- NaN output becomes 0 and the output is clipped to +-U_LIMIT (`safe_output`).
- Unstable or invalid candidates get the penalty from the objective (Bahnacy).
- Known weakness: `e/e`-style subtrees act as constants. **(team)** decide whether to keep `div`.

### 8.5 Vector encoding (so GA/DE/PSO/SA can optimize the GP controller)
- Heap layout, max depth D = 3 -> 15 slots x 2 genes = 30 parameters.
- Gene pair per slot: symbol index + 0.5 (decoded with floor) and a constant value.
- `GPController.parameter_bounds()` must return `GPGrammar.parameter_bounds()`.
- `GPController.set_parameters(v)` must call `decode_vector` + `compile_tree`; `reset()` must clear dynamic state only, never the parameters.
- Every vector inside the bounds decodes to a valid tree of depth <= 3.

### 8.6 GP optimizer settings
- Shared: `population_size=30`, `max_iterations=100`, `random_seed=42`.
- GP: tournament 5, crossover 0.8, mutation 0.3 (subtree 30% / point 30% / constant 40%), elite 1, Koza 90/10 point selection, `max_depth` from the bounds, optional `max_size`.
- Fitness-evaluation cache is on by default, so `n_evaluations <= population_size * (max_iterations + 1)`.
- Convergence history has one entry per generation (same convention as PSO).

### 8.7 Open questions **(team)**
- Objective can be satisfied by an open-loop constant (no feedback). Options: penalize trees without `e`, or also simulate a perturbed plant gain.
- If the instructor wants GP to optimize PID/NN/Fuzzy too, the GP optimizer currently raises `ValueError` for non-GP bounds.

