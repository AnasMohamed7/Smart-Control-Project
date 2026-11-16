"""Common optimizer contract.

The optimizer receives an objective function and parameter bounds, then returns
an OptimizationResult. Each algorithm must be implemented from scratch.
"""


class BaseOptimizer:
    def optimize(self, objective_function, bounds, settings):
        """Search for a candidate minimizing objective_function.

        Expected return: OptimizationResult.
        """
        raise NotImplementedError
