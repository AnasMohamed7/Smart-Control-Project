"""Objective-function adapter used by optimizers.

The brief requires minimizing five criteria but does not prescribe how to combine
them. The team must choose normalized weights or another ranking strategy.
"""


def build_objective(controller, plant, simulation_config, metric_config, objective_config):
    """Create and return a callable objective function.

    The returned function should:
    1. Receive one candidate parameter vector.
    2. Apply it to the controller.
    3. Simulate the closed-loop plant.
    4. Calculate the five performance metrics.
    5. Convert metrics into the agreed scalar fitness (if scalar optimization).
    6. Handle invalid/unstable responses consistently.

    TODO: Implement after objective design is approved.
    """
    raise NotImplementedError
