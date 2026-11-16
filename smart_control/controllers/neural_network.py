"""
Neural Network Controller for Smart Control Project.

Architecture:
    Input  : [e(t), e(t-1)]  — 2 neurons
    Hidden : 5 neurons, tanh activation
    Output : 1 neuron, linear activation

Total parameters: 5×2 + 5 + 1×5 + 1 = 21
"""

import numpy as np

from smart_control.controllers.base import BaseController
import smart_control.config as cfg


class NeuralNetworkController(BaseController):
    """Feed-forward neural network controller (2 → 5 → 1)."""

    def __init__(self, architecture=None, activations=None):
        nn_cfg = cfg.NN_CONFIG

        if architecture is None:
            self.input_size  = nn_cfg["input_size"]   # 2
            self.hidden_size = nn_cfg["hidden_size"]  # 5
            self.output_size = nn_cfg["output_size"]  # 1
        else:
            self.input_size, self.hidden_size, self.output_size = architecture

        self.weight_bound = nn_cfg["weight_bound"]  # 5.0
        self.bias_bound   = nn_cfg["bias_bound"]    # 5.0

        # Total flat parameter count
        self.n_params = (
            self.input_size  * self.hidden_size    # W1: 2×5 = 10
            + self.hidden_size                     # b1: 5
            + self.hidden_size * self.output_size  # W2: 5×1 = 5
            + self.output_size                     # b2: 1
        )  # total: 21

        # Initialise weights to small random values (overwritten by set_parameters)
        self._init_weights()

        # Previous error for the recurrent input
        self.prev_error = 0.0

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _init_weights(self):
        """Initialise weights to small random values."""
        rng = np.random.randn
        self.W1 = rng(self.hidden_size, self.input_size)  * 0.1
        self.b1 = rng(self.hidden_size)                   * 0.1
        self.W2 = rng(self.output_size, self.hidden_size) * 0.1
        self.b2 = rng(self.output_size)                   * 0.1

    # ------------------------------------------------------------------
    # BaseController interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset controller state and re-initialise weights."""
        self.prev_error = 0.0
        self._init_weights()

    def set_parameters(self, parameters):
        """Decode a flat parameter vector and store W1, b1, W2, b2."""
        self.decode_parameters(parameters)

    def compute(self, error, dt, **signals):
        """
        Run a forward pass and return the control signal u(t).

        Parameters
        ----------
        error : float
            Current tracking error e(t).
        dt : float
            Time step (unused by the NN directly, kept for interface compatibility).
        **signals : dict
            Extra signals (ignored).

        Returns
        -------
        float
            Control output u(t).
        """
        x = np.array([error, self.prev_error])          # shape (2,)
        h = np.tanh(self.W1 @ x + self.b1)              # shape (5,)
        u = float((self.W2 @ h + self.b2)[0])           # scalar (extract element to avoid DeprecationWarning)

        self.prev_error = error
        return u

    def parameter_bounds(self):
        """
        Return bounds for the flat parameter vector.

        Returns
        -------
        list of (float, float)
            Length == n_params; each entry is (-weight_bound, +weight_bound).
        """
        bound = self.weight_bound
        return [(-bound, bound)] * self.n_params

    # ------------------------------------------------------------------
    # Encode / decode
    # ------------------------------------------------------------------

    def encode_parameters(self):
        """
        Flatten current weights into a 1-D numpy array of length n_params.

        Layout: [W1 (row-major), b1, W2 (row-major), b2]
        """
        return np.concatenate([
            self.W1.flatten(),
            self.b1.flatten(),
            self.W2.flatten(),
            self.b2.flatten(),
        ])

    def decode_parameters(self, parameters):
        """
        Unpack a flat parameter vector into W1, b1, W2, b2.

        Parameters
        ----------
        parameters : array-like of length n_params
        """
        params = np.array(parameters, dtype=float)

        hs   = self.hidden_size
        ins  = self.input_size
        outs = self.output_size

        idx = 0

        n = ins * hs
        self.W1 = params[idx:idx + n].reshape(hs, ins)
        idx += n

        self.b1 = params[idx:idx + hs]
        idx += hs

        n = hs * outs
        self.W2 = params[idx:idx + n].reshape(outs, hs)
        idx += n

        self.b2 = params[idx:idx + outs]
        # idx += outs  (not needed after last slice)


# ---------------------------------------------------------------------------
# Quick self-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    nn = NeuralNetworkController()
    print(f"n_params: {nn.n_params}")          # should be 21

    # Encode / decode round-trip
    test_params = np.arange(21, dtype=float)
    nn.set_parameters(test_params)
    recovered = nn.encode_parameters()
    assert np.allclose(recovered, test_params), "Encode/decode mismatch!"
    print("Encode/decode roundtrip: OK")

    # Forward pass
    nn.reset()
    u = nn.compute(0.5, 0.01)
    print(f"Control output for error=0.5: {u:.4f}")

    # Bounds length
    bounds = nn.parameter_bounds()
    print(f"Bounds length: {len(bounds)}")     # should be 21
    assert len(bounds) == nn.n_params
    print("All tests passed.")
