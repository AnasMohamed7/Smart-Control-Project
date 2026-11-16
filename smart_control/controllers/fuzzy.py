"""Fuzzy-logic controller scaffold.

The project brief does not prescribe membership functions or a rule base.
Document those choices before implementing this controller.
"""


class FuzzyController:
    def __init__(self, design_config=None):
        # TODO: Store membership-function and rule-base configuration.
        pass

    def reset(self):
        """Reset any controller state."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Update the selected fuzzy parameters optimized by the optimizer."""
        raise NotImplementedError

    def compute(self, error, dt, **signals):
        """Perform fuzzification, inference, and defuzzification."""
        raise NotImplementedError

    def parameter_bounds(self):
        """Return bounds for the selected fuzzy parameters."""
        raise NotImplementedError
