"""PID controller scaffold.

Candidate parameter vector convention (confirm and keep consistent):
[Kp, Ki, Kd]
"""


class PIDController:
    def __init__(self):
        # TODO: Initialize gains and controller state.
        pass

    def reset(self):
        """Clear integral and previous-error state before each simulation."""
        raise NotImplementedError

    def set_parameters(self, parameters):
        """Assign Kp, Ki, and Kd from the candidate vector."""
        raise NotImplementedError

    def compute(self, error, dt, **signals):
        """Calculate the PID control signal.

        TODO: Agree on derivative filtering, output saturation, and anti-windup.
        """
        raise NotImplementedError

    def parameter_bounds(self):
        """Return the agreed bounds for Kp, Ki, and Kd."""
        raise NotImplementedError
