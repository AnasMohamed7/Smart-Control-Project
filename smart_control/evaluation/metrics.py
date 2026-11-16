"""Calculate the five performance criteria required by the brief."""


def calculate_rise_time(time, output, reference):
    """Calculate rise time using the team's agreed threshold convention."""
    raise NotImplementedError


def calculate_settling_time(time, output, reference, tolerance):
    """Calculate settling time using an agreed tolerance band."""
    raise NotImplementedError


def calculate_overshoot(output, reference):
    """Calculate overshoot using an agreed definition."""
    raise NotImplementedError


def calculate_steady_state_error(output, reference):
    """Calculate final steady-state error."""
    raise NotImplementedError


def calculate_iae(time, output, reference):
    """Calculate IAE = integral of absolute tracking error over time."""
    raise NotImplementedError


def calculate_all_metrics(simulation_result, metric_config):
    """Return a PerformanceMetrics object containing all five measures."""
    raise NotImplementedError
