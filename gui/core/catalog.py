"""Registry of the project's building blocks plus descriptive text.

The GUI never hard-codes a class elsewhere: names, factories and the copy shown
on the explanatory pages all come from here, so the presentation stays in sync
with ``smart_control``.
"""
from __future__ import annotations

from smart_control.controllers.fuzzy import FuzzyController
from smart_control.controllers.gp_controller import GPController
from smart_control.controllers.neural_network import NeuralNetworkController
from smart_control.controllers.pid import PIDController
from smart_control.optimizers.differential_evolution import DifferentialEvolution
from smart_control.optimizers.genetic_algorithm import GeneticAlgorithm
from smart_control.optimizers.genetic_programming import GeneticProgramming
from smart_control.optimizers.particle_swarm import ParticleSwarmOptimization
from smart_control.optimizers.simulated_annealing import SimulatedAnnealing
from smart_control.plants.models import get_plant_models

OPTIMIZERS = {
    "DE": DifferentialEvolution,
    "GA": GeneticAlgorithm,
    "PSO": ParticleSwarmOptimization,
    "SA": SimulatedAnnealing,
    "GP": GeneticProgramming,
}
NUMERIC_OPTIMIZERS = ["DE", "GA", "PSO", "SA"]

CONTROLLERS = {
    "PID": PIDController,
    "Neural Network": NeuralNetworkController,
    "Fuzzy": FuzzyController,
    "GP Controller": GPController,
}

PLANT_NAMES = ["second_order", "third_order", "fourth_order"]
PLANT_LABELS = {"second_order": "2nd order", "third_order": "3rd order", "fourth_order": "4th order"}
PLANT_LATEX = {
    "second_order": r"\small G_2(s) = \dfrac{1}{s^2 + 1.2s + 1}",
    "third_order": r"\small G_3(s) = \dfrac{10}{(s+1)(s+2)(s+3)}",
    "fourth_order": r"\small G_4(s) = \dfrac{1}{s^4 + 2.6s^3 + 3.44s^2 + 2.4s + 0.64}",
}
PLANT_NOTES = {
    "second_order": "Classic underdamped benchmark (ζ = 0.6, ωₙ = 1 rad/s). Unit DC gain, complex pole pair.",
    "third_order": "s³ + 6s² + 11s + 6 factored: three distinct real poles — no natural oscillation, but higher DC gain (10/6) and extra phase lag.",
    "fourth_order": "Complex pair plus two slow real poles. The most phase lag: easy to destabilise with aggressive gains.",
}

METRICS = {
    "iae": ("IAE", "", "Integral of absolute error ∫|e(t)|dt"),
    "settling_time": ("Settling time", "s", "Time to stay inside the ±2 % band"),
    "rise_time": ("Rise time", "s", "10 % → 90 % of the step"),
    "overshoot": ("Overshoot", "%", "Peak above the reference"),
    "steady_state_error": ("Steady-state error", "", "|r − y| averaged over the final 5 %"),
}
METRIC_KEYS = list(METRICS)


def metric_label(key: str, with_unit: bool = True) -> str:
    if key == "fitness":
        return "Objective J"
    if key == "execution_time":
        return "Runtime (s)" if with_unit else "Runtime"
    name, unit, _ = METRICS[key]
    return f"{name} ({unit})" if (with_unit and unit) else name


def plants():
    return get_plant_models()


def is_compatible(optimizer: str, controller: str) -> bool:
    """GP evolves expression trees, so it only drives the GP controller."""
    return optimizer != "GP" or controller == "GP Controller"


def make_controller(name: str, parameters=None):
    ctrl = CONTROLLERS[name]()
    if parameters is not None:
        ctrl.set_parameters(list(parameters))
    return ctrl


def parameter_names(controller: str, n: int) -> list[str]:
    if controller == "PID":
        return ["Kp", "Ki", "Kd"]
    if controller == "Fuzzy":
        return ["K_e", "K_de", "K_u"]
    if controller == "Neural Network":
        names = [f"W1[{i},{j}]" for i in range(5) for j in range(2)]
        names += [f"b1[{i}]" for i in range(5)] + [f"W2[0,{i}]" for i in range(5)] + ["b2"]
        return names[:n]
    return [f"g{i}" for i in range(n)]


# ---------------------------------------------------------------------------
# Descriptive copy
# ---------------------------------------------------------------------------
CONTROLLER_INFO = {
    "PID": {
        "summary": "The industry workhorse: proportional, integral and derivative action on the tracking error.",
        "latex": r"u(t) = K_p\,e(t) + K_i \int_0^t e(\tau)\,d\tau + K_d\,\frac{de(t)}{dt}",
        "params": "3 gains — Kp ∈ [0, 50], Ki ∈ [0, 20], Kd ∈ [0, 10]",
        "details": [
            "Backward-difference derivative (optional low-pass filter)",
            "Anti-windup: integral clamped to ±50",
            "Output saturation |u| ≤ 50 — same limit as the simulator",
        ],
    },
    "Neural Network": {
        "summary": "A small feed-forward network maps the current and previous error to a control action.",
        "latex": r"u = W_2\,\tanh\!\big(W_1 [e_k,\ e_{k-1}]^\top + b_1\big) + b_2",
        "params": "21 weights & biases (2 → 5 → 1), each ∈ [−5, 5]",
        "details": [
            "Inputs: e(k) and e(k−1) — implicit derivative information",
            "Hidden layer: 5 tanh neurons, linear output",
            "No back-propagation — weights are found by the optimiser directly (neuro-evolution)",
        ],
    },
    "Fuzzy": {
        "summary": "A two-input Mamdani controller with a 25-rule PD-like rule base and centroid defuzzification.",
        "latex": r"u = K_u \cdot \mathrm{centroid}\Big(\max_{i,j}\ \min\big(\mu_i(K_e e),\ \mu_j(K_{de}\dot e)\big)\Big)",
        "params": "3 scaling gains — K_e ∈ [0.1, 20], K_de ∈ [0.001, 5], K_u ∈ [0.1, 50]",
        "details": [
            "Five triangular sets per variable: NB, NS, ZE, PS, PB",
            "Min implication, max aggregation, centroid over 201 points",
            "Optimiser tunes the input/output scaling, rules stay fixed",
        ],
    },
    "GP Controller": {
        "summary": "The control law itself is an evolved expression tree over the error, its derivative and its integral.",
        "latex": r"u = f(e,\ \dot e,\ \textstyle\int e)\quad f \in \{+,\ -,\ \times,\ \div_{\mathrm{prot}},\ \tanh\}",
        "params": "Tree of depth ≤ 3, encoded as a 30-gene vector (symbol + constant per slot)",
        "details": [
            "Terminals: e, de, ie and constants",
            "Protected division and output clipping keep every tree safe",
            "A full PID law fits exactly in the depth-3 grammar",
        ],
    },
}

OPTIMIZER_INFO = {
    "DE": {
        "name": "Differential Evolution",
        "idea": "Mutate by adding the scaled difference of two population members to a third, then cross over with the parent.",
        "latex": r"v = x_{r_1} + F\,(x_{r_2} - x_{r_3}),\quad u_j = \begin{cases} v_j & \text{rand} < CR \\ x_j & \text{else}\end{cases}",
        "strengths": "Self-scaling steps, very few hyper-parameters, strong on continuous problems.",
        "steps": [("Initialise", "Random population inside the bounds"),
                  ("Mutate", "Difference vector F·(x₂ − x₃) added to x₁ (F = 0.6)"),
                  ("Crossover", "Binomial mix of mutant and parent (CR = 0.9)"),
                  ("Select", "Trial replaces parent only if it is better (greedy)")],
    },
    "GA": {
        "name": "Genetic Algorithm",
        "idea": "Darwinian search: tournament selection, simulated-binary crossover and polynomial mutation, with elitism.",
        "latex": r"P_{t+1} = \mathrm{elite}(P_t) \,\cup\, \mathrm{mutate}\big(\mathrm{SBX}(\mathrm{tournament}(P_t))\big)",
        "strengths": "Robust global exploration; recombination shares good building blocks.",
        "steps": [("Evaluate", "Fitness of each chromosome = objective J"),
                  ("Select", "Tournament picks parents"),
                  ("Recombine", "SBX crossover creates children near the parents"),
                  ("Mutate & keep elite", "Polynomial mutation; best individuals survive unchanged")],
    },
    "PSO": {
        "name": "Particle Swarm Optimisation",
        "idea": "Particles fly through the search space, pulled toward their own best and the swarm's best positions.",
        "latex": r"v \leftarrow w\,v + c_1 r_1 (p_{best} - x) + c_2 r_2 (g_{best} - x),\quad x \leftarrow x + v",
        "strengths": "Fast early convergence, simple, and shares information across the whole swarm.",
        "steps": [("Initialise", "Random positions and velocities"),
                  ("Evaluate", "Update personal and global bests"),
                  ("Move", "Inertia + cognitive + social pull (w decays 0.7 → 0.4, c₁ = c₂ = 1.5)"),
                  ("Clamp", "Velocity limited to 20 % of the range")],
    },
    "SA": {
        "name": "Simulated Annealing",
        "idea": "A single solution random-walks; worse moves are accepted with a probability that shrinks as it cools.",
        "latex": r"P(\text{accept}) = \exp\!\left(-\frac{\Delta J}{T}\right),\qquad T_{k+1} = \alpha\,T_k",
        "strengths": "Escapes local minima with almost no memory; one candidate at a time.",
        "steps": [("Start", "One random solution, high temperature"),
                  ("Perturb", "Gaussian step (σ = 10 % of each range), reflected into bounds"),
                  ("Accept / reject", "Metropolis criterion"),
                  ("Cool", "T ← 0.98·T (geometric cooling)")],
    },
    "GP": {
        "name": "Genetic Programming",
        "idea": "Evolves programs, not numbers: expression trees are recombined by swapping sub-trees and mutated node-wise.",
        "latex": r"\text{tree} \;\xrightarrow{\ \text{subtree crossover / mutation}\ }\; \text{tree}'",
        "strengths": "Discovers the structure of the control law, not just its gains.",
        "steps": [("Initialise", "Ramped half-and-half random trees"),
                  ("Select", "Tournament on objective J"),
                  ("Vary", "Subtree crossover, subtree / point / constant mutation"),
                  ("Keep elite", "Best tree survives; depth limit enforced")],
    },
}
