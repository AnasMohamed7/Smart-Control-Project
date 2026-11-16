"""Common controller interface.

Each controller should expose the same public methods so the experiment runner
does not need controller-specific branching.
"""


class BaseController:
    def reset(self):
        """Reset internal state before a new simulation."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Load one candidate parameter vector supplied by an optimizer."""
        raise NotImplementedError

    def compute(self, error, dt, **signals):
        """Return the control action for the current simulation step."""
        raise NotImplementedError

    def parameter_bounds(self):
        """Return lower/upper bounds in the same order as the parameter vector."""
        raise NotImplementedError
