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
