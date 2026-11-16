"""Closed-loop simulation helpers.

This module is the shared bridge between a controller and a plant.
All controllers should be evaluated under consistent simulation conditions.
"""


def simulate_closed_loop(plant, controller, simulation_config):
    """Simulate the closed-loop response.

    Expected conceptual steps:
    1. Create the time vector and reference signal.
    2. Initialize controller state.
    3. At each simulation step, calculate error and controller output.
    4. Apply the control signal to the plant model.
    5. Record output and control signal.
    6. Return a SimulationResult.

    TODO: Choose and implement a consistent simulation method.
    """
    raise NotImplementedError
