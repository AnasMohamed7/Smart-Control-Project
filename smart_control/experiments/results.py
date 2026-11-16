"""Result collection and export helpers."""


def records_to_rows(experiment_records):
    """Convert structured experiment records into table rows."""
    raise NotImplementedError


def save_results_csv(experiment_records, output_path):
    """Optional: save results in a simple CSV format.

    Note: if external libraries are restricted, use Python's standard library
    for CSV writing, or keep results in the notebook.
    """
    raise NotImplementedError
