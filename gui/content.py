"""Presentation text you are expected to edit.

Everything the audience reads that is *not* computed from results lives here:
title, team, course, objectives and the closing remarks.  Change these strings
and refresh the browser — no other file needs touching.
"""

TITLE = "Smart Control"
TITLE_ACCENT = "Metaheuristic Tuning of Intelligent Controllers"
TAGLINE = (
    "Four controllers, five optimisation algorithms and three plants of rising "
    "difficulty — tuned automatically, measured on five performance criteria and "
    "compared head-to-head."
)

COURSE = "Smart Control — Course Project"
TERM = "2026"

# (name, role) — shown on the title slide. Roles follow the team-roles document
# (DOC-20261116-WA0002.pdf: "Team Roles, File Ownership & Deliverables").
TEAM = [
    ("Ahmed M. Fayad", "Genetic Programming optimiser · common optimiser interface · GP design decisions"),
    ("Mostafa A. Khaira", "Genetic Algorithm · PID controller · GP controller"),
    ("Anas M. Abdrabo", "Experiment runner · result collection · main entry point & notebook"),
    ("Ahmed A. Bahnasy", "Performance metrics · objective function · shared data types"),
    ("Abdallah Sheshtawy", "Plant models · closed-loop simulation · Differential Evolution"),
    ("Fares Sakr", "Simulated Annealing · Fuzzy-logic controller · plots"),
    ("Kareem Halawa", "Particle Swarm Optimisation · Neural-network controller · shared config"),
]
# Shared work credited to everyone (shown under the team list).
TEAM_SHARED = "System integration and this interactive dashboard were built together by the whole team."
SUPERVISOR = ""   # e.g. "Supervised by Dr. …"  (leave empty to hide)

OBJECTIVES = [
    ("Model", "Build 2nd-, 3rd- and 4th-order plants and one shared closed-loop simulator."),
    ("Control", "Implement PID, Neural-Network, Fuzzy-Logic and Genetic-Programming controllers behind one interface."),
    ("Optimise", "Tune every controller with DE, GA, PSO and SA — plus GP, which evolves the control law itself."),
    ("Compare", "Score each run on rise time, settling time, overshoot, steady-state error and IAE, then compare fairly."),
]

# Extra closing remarks shown on the Conclusions page (in addition to the
# findings computed automatically from the selected result set).
LIMITATIONS = [
    "Evaluation budgets are deliberately small so the full matrix fits the five-minute runtime target — most optimisers have not converged.",
    "Metrics are combined as a raw weighted sum (not normalised), so IAE and settling time dominate the objective's scale.",
    "Each plant is excited by a single unit step; disturbance rejection and noise were not part of the objective.",
]
FUTURE_WORK = [
    "Larger evaluation budgets and more independent seeds for statistically solid rankings.",
    "Normalised or multi-objective (Pareto) formulation instead of a fixed weighted sum.",
    "Robust tuning: optimise over a family of perturbed plants, not a single nominal model.",
    "Let GP evolve the structure of the other controllers (e.g. fuzzy rule bases).",
]
