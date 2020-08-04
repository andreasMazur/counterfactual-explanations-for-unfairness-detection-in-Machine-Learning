## Counterfactual Explanations for Unfairness Detection in Machine Learning

### Message from the author

This code was written by me (an undergraduate) in the course of me
writing my bachelor thesis. It shall illustrate how counterfactual
explanations can be used as an indicator for detecting unfair Machine
Learning models.

### Necessities

First, one needs to install a Python version >=3.6.

Then, in order to run the code, you need to have following libararies
installed:

- cvxpy: https://github.com/cvxgrp/cvxpy
- sklearn: https://github.com/scikit-learn/scikit-learn
- pandas: https://github.com/pandas-dev/pandas
- numpy: https://github.com/numpy/numpy
- scipy: https://github.com/scipy/scipy
- matplotlib: https://github.com/matplotlib/matplotlib
- seaborn: https://github.com/mwaskom/seaborn
- plotly: https://github.com/plotly/plotly.py

All the packages and their used version are listed in the requirements.txt.
Furthermore, the project is based on the '_compas-scores-two-years.csv_'
data set from: https://github.com/propublica/compas-analysis .
You will have to download it from there.

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

However, keep in mind that _visualization.py_ requires the results of _compute_cfs.py_ and
_chi2_tests.py_ requires the groupings from _visualization.py_.
