"""
Smart Control Project – Central Configuration
==============================================

All tunable parameters for the simulation, optimisation algorithms, neural
network architecture, and objective function are defined here as plain Python
constants.  This file has **no imports** – it is pure data so that any module
can safely import it without side-effects or circular dependencies.

Usage
-----
    from smart_control import config
    dt = config.DT
    pso = config.PSO_SETTINGS
"""

# =============================================================================
# Simulation parameters
# =============================================================================

DT = 0.01
# Discrete time-step in seconds.
# Smaller values give higher accuracy but longer runtimes.
# Acceptable range: 0.001 – 0.1 s. Must satisfy DT < 1/(10*bandwidth).

T_START = 0.0
# Simulation start time in seconds (almost always 0.0).

T_END = 20.0
# Total simulation duration in seconds.
# Should be long enough for all plants to settle.
# Acceptable range: 10.0 – 60.0 s for the plants used here.

REFERENCE = 1.0
# Step reference (setpoint) value – the desired output the controller tracks.
# Acceptable range: any non-zero real number; 1.0 gives a unit step.

SETTLING_TOLERANCE = 0.02
# Half-width of the ±2 % settling band around REFERENCE.
# A response is considered settled when |y(t) - REFERENCE| ≤
# SETTLING_TOLERANCE * REFERENCE for all subsequent time.
# Acceptable range: 0.01 (1 %) – 0.05 (5 %).

# ---------------------------------------------------------------------------
# Backward-compatible SIMULATION dict (mirrors the flat constants above)
# ---------------------------------------------------------------------------
SIMULATION = {
    "time_start":          T_START,            # simulation start time (s)
    "time_end":            T_END,              # simulation end time (s)
    "time_step":           DT,                 # integration step size (s)
    "reference_value":     REFERENCE,          # desired output (setpoint)
    "settling_tolerance":  SETTLING_TOLERANCE, # ±fraction band for settling
}

# =============================================================================
# PSO (Particle Swarm Optimisation) settings
# =============================================================================

PSO_SETTINGS = {
    # Number of particles in the swarm.
    # More particles → broader search, slower per-iteration cost.
    # Acceptable range: 10 – 100.
    "population_size": 30,

    # Maximum number of PSO iterations before stopping.
    # Acceptable range: 50 – 500.
    "max_iterations": 100,

    # Inertia weight w – controls how much of the previous velocity is kept.
    # w = 1.0 → no damping; w → 0 → particle stops moving.
    # Acceptable range: 0.4 – 0.9 (linearly decreasing schedules also common).
    "inertia_weight": 0.7,

    # Cognitive (personal best) acceleration coefficient c1.
    # Scales pull towards a particle's own best-known position.
    # Acceptable range: 1.0 – 2.5.
    "cognitive_coeff": 1.5,

    # Social (global best) acceleration coefficient c2.
    # Scales pull towards the swarm's global best-known position.
    # Acceptable range: 1.0 – 2.5.
    "social_coeff": 1.5,

    # Velocity clamp ratio.  max_velocity = v_max_ratio × (upper – lower).
    # Prevents particles from flying too far in one step.
    # Acceptable range: 0.1 – 0.5.
    "v_max_ratio": 0.2,

    # Random seed for reproducible PSO runs.
    # Set to None to use system entropy.
    "random_seed": 42,
}

# =============================================================================
# GA (Genetic Algorithm) settings
# =============================================================================

GA_SETTINGS = {
    # Size of the population (number of chromosomes).
    # Acceptable range: 10 – 200.
    "population_size": 30,

    # Maximum number of generations.
    # Acceptable range: 50 – 500.
    "max_iterations": 100,

    # Random seed for reproducibility.
    "random_seed": 42,
}

# =============================================================================
# DE (Differential Evolution) settings
# =============================================================================

DE_SETTINGS = {
    # Size of the trial population.
    # Acceptable range: 10 – 200; DE typically needs 10×D individuals.
    "population_size": 30,

    # Maximum number of DE generations.
    # Acceptable range: 50 – 1000.
    "max_iterations": 100,

    # Random seed for reproducibility.
    "random_seed": 42,
}

# =============================================================================
# SA (Simulated Annealing) settings
# =============================================================================

SA_SETTINGS = {
    # SA is a single-solution method; population_size is always 1.
    "population_size": 1,

    # Number of temperature-step iterations.
    # SA typically needs more iterations than population-based methods.
    # Acceptable range: 100 – 2000.
    "max_iterations": 300,

    # Random seed for reproducibility.
    "random_seed": 42,
}

# =============================================================================
# Generic / backward-compatible optimiser settings
# =============================================================================

OPTIMIZER_SETTINGS = {
    # Default population size shared by all optimisers unless overridden.
    # Acceptable range: 10 – 200.
    "population_size": 30,

    # Default iteration budget shared by all optimisers unless overridden.
    # Acceptable range: 50 – 1000.
    "iterations": 100,

    # Master random seed; individual optimiser seeds override this per-run.
    "random_seed": 42,
}

# =============================================================================
# Neural Network (NN-PID / NN controller) architecture
# =============================================================================

NN_CONFIG = {
    # Number of inputs fed to the network at each time step.
    # Here: [e(t), e(t-1)] – current and previous tracking error.
    # Must match the feature vector produced by the simulator.
    "input_size": 2,

    # Number of neurons in the single hidden layer.
    # Acceptable range: 3 – 20 for simple SISO control tasks.
    "hidden_size": 5,

    # Number of network outputs (control signal dimension).
    # 1 for a SISO controller.
    "output_size": 1,

    # Hidden-layer activation function: "tanh" | "sigmoid" | "relu".
    # "tanh" keeps outputs in (-1, 1) which suits bounded control signals.
    "activation": "tanh",

    # Symmetric bound for weight parameters: weights ∈ [-weight_bound, weight_bound].
    # Defines the PSO/GA search space for each weight.
    # Acceptable range: 1.0 – 10.0.
    "weight_bound": 5.0,

    # Symmetric bound for bias parameters: biases ∈ [-bias_bound, bias_bound].
    # Acceptable range: 1.0 – 10.0.
    "bias_bound": 5.0,
}

# =============================================================================
# Objective function weights
# =============================================================================

OBJECTIVE_WEIGHTS = {
    # Weight on Integral of Absolute Error (IAE).
    # IAE = ∫|e(t)| dt over [T_START, T_END].
    # Higher → penalise steady-state and transient error more.
    # Acceptable range: 0.5 – 2.0.
    "iae": 1.0,

    # Weight on percentage overshoot beyond REFERENCE.
    # Higher → favour more damped, slower responses.
    # Acceptable range: 0.0 – 2.0.
    "overshoot": 0.5,

    # Weight on settling time (time until response stays in ±2 % band).
    # Higher → favour faster settling.
    # Acceptable range: 0.0 – 1.0.
    "settling_time": 0.3,
}

# =============================================================================
# Global random seed
# =============================================================================

RANDOM_SEED = 42
# Master seed used for any stochastic component not covered by an individual
# optimiser seed.  Set to None to use system entropy (non-reproducible).

# =============================================================================
# Plant registry
# =============================================================================

PLANT_NAMES = ["second_order", "third_order", "fourth_order"]
# Canonical names of the plant models available in plants.models.
# These strings must match the keys returned by get_plant_models().
# Extend this list when new plant models are added to plants/models.py.
