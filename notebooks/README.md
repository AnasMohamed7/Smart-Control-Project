# Google Colab notebook

The executable notebook is `Smart_Control_Project.ipynb`.

It is designed to:

1. Install the permitted dependencies (`control`, `numpy`, `matplotlib`).
2. Upload and extract the final project ZIP.
3. Import the maintained source modules (it does not duplicate the code in cells).
4. Display the active configuration and objective weights.
5. Run the required 48-experiment matrix.
6. Optionally run GP optimizer on GPController for three additional plant cases.
7. Download the CSV results and plots as a ZIP file.

The default entry point uses a small evaluation budget to respect the runtime
constraint. Record the actual execution time in the final Colab environment.
The quick mode is for smoke testing only, and the default budget is still a
limited search—not proof that every optimizer converged.

## Optimiser study notebooks

`study/` contains one exploratory notebook per optimisation algorithm (DE, GA,
PSO, SA, GP): benchmark sanity checks, hyper-parameter studies, budget splits
and a full controller × plant sweep. See `study/README.md`.
