"""Central project configuration.

Keep agreed experimental settings here so all team members use the same values.
Do not select arbitrary values independently in individual modules.
"""

# TODO: Replace placeholders after the team agrees on the plant models.
PLANT_MODELS = {
    "second_order": None,  # TODO: control.TransferFunction(...)
    "third_order": None,   # TODO: control.TransferFunction(...)
    "fourth_order": None,  # TODO: control.TransferFunction(...)
}

# TODO: Define shared simulation settings.
SIMULATION = {
    "time_start": None,
    "time_end": None,
    "time_step": None,
    "reference_value": 1.0,
}

# TODO: Define optimizer budgets and controller-specific bounds.
OPTIMIZER_SETTINGS = {
    "population_size": None,
    "iterations": None,
    "random_seed": None,
}

# TODO: Decide whether the team uses a normalized weighted objective
# or reports/ranks the five metrics separately.
OBJECTIVE_SETTINGS = {
    "metric_weights": None,
    "normalization_scales": None,
}
