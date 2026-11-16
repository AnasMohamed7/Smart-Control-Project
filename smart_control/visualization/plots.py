"""Matplotlib visualizations for the report and presentation."""


def plot_step_responses(simulation_records):
    """Plot and label step responses for selected comparisons."""
    raise NotImplementedError


def plot_convergence_curves(optimization_records):
    """Plot optimizer best-fitness history against iteration/evaluation."""
    raise NotImplementedError


def plot_metric_comparison(experiment_records, metric_name):
    """Visualize one performance metric across experiment combinations."""
    raise NotImplementedError
