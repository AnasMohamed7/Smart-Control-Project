"""Neural-network controller scaffold.

Use NumPy for the network operations. No ready-made neural-network framework
is permitted by the project brief.
"""


class NeuralNetworkController:
    def __init__(self, architecture=None, activations=None):
        # TODO: Store the approved architecture and activation choices.
        # TODO: Initialize parameter storage (weights and biases).
        pass

    def reset(self):
        """Reset any state used by the controller."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Map a flat optimizer vector into the network weights and biases."""
        raise NotImplementedError

    def compute(self, error, dt, **signals):
        """Return the network's control output.

        TODO: Define the input features and output scaling with the team.
        """
        raise NotImplementedError

    def parameter_bounds(self):
        """Return bounds for every optimized weight and bias."""
        raise NotImplementedError
