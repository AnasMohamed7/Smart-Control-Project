"""Run the required optimizer-controller-plant experiment matrix."""


def run_single_experiment(optimizer, controller, plant, settings):
    """Run one optimizer/controller/plant combination.

    Return a structured record containing:
    - optimizer name
    - controller name
    - plant name/order
    - best candidate
    - best objective value
    - five performance metrics
    - convergence history
    - execution time
    """
    raise NotImplementedError


def run_experiment_matrix(optimizers, controllers, plants, settings):
    """Run all required combinations.

    Expected nominal matrix from the brief:
    4 optimizers (GA, DE, PSO, SA) x 4 controllers x 3 plants = 48 runs.
    GP is also implemented as a required algorithm and GP controller.
    Confirm the intended role of GP with the instructor/team.
    """
    raise NotImplementedError
