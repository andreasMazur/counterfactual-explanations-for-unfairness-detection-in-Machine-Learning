## Counterfactual Explanations for Unfairness Detection in Machine Learning

### Message from the author

This code was written by me (an undergraduate) in the course of me
writing my bachelor thesis. It shall illustrate how counterfactual
explanations can be used as an indicator for detecting unfair Machine
Learning models.

### Necessities

The experiment was executed with Python version >=3.6.

In order to run the code, you need to set up a conda environment and
install the packages from the **requirements.txt**. You need a conda-environment
because you need to install the '_plotly-orca_'-package with the conda package manager. If you have no
conda installed, you have the option to run the project with
a normal pip-environment. If so, you only need to install the
packages from the **requirements.txt** until line 9. However, this
comes with the drawback that you cannot save the histograms as
svg-figures. 

The project is based on the '_compas-scores-two-years.csv_'
data set from: https://github.com/propublica/compas-analysis .
You will have to download it from there.

Furthermore, you need to have the following solvers installed:

- https://github.com/cvxgrp/scs (SCS)
- https://github.com/coin-or/Cbc (CBC)

### How can I execute the code?

In general, the code is executed in three steps:

- compute the counterfactuals
- group and visualize the data
- compute the chi2-values for the groups and certain attributes

If you want to execute the entire project, you can execute the '_execute_experiment.py_'-file.
If you only want to compute a certain step, you execute either:

- _compute_cfs.py_
- _visualization.py_
- _chi2_tests.py_

However, keep in mind that '_visualization.py_' requires the results of '_compute_cfs.py_' and
'_chi2_tests.py_' requires the groupings from '_visualization.py_'.
