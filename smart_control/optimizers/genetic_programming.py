"""Genetic Programming (GP) scaffold.

TODO: grammar, tree representation, initialization, fitness, selection, subtree crossover, mutation, depth/size control.
Implement the algorithm directly; do not import a ready-made optimizer.
"""


class GeneticProgramming:
    def __init__(self, settings=None):
        # TODO: Store algorithm-specific settings.
        pass

    def optimize(self, objective_function, bounds, settings=None):
        """Return the best candidate, its fitness, and convergence history."""
        raise NotImplementedError
