"""Genetic-programming controller scaffold.

This is the controller-facing wrapper for an evolved GP expression/tree.
The GP evolutionary algorithm itself belongs in optimizers/genetic_programming.py.
"""


class GPController:
    def __init__(self, expression=None):
        # TODO: Store a validated expression/tree.
        pass

    def reset(self):
        """Reset any controller state."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Set or decode the candidate GP expression representation.

        The exact representation depends on the GP grammar/tree design.
        """
        raise NotImplementedError

    def compute(self, error, dt, **signals):
        """Evaluate the expression using the current controller inputs."""
        raise NotImplementedError

    def parameter_bounds(self):
        """Return bounds only if the chosen GP representation is numeric.

        Tree-based GP may need a different candidate representation.
        """
        raise NotImplementedError
