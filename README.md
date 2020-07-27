## Counterfactual Explanations for Unfairness Detection in Machine Learning

### Message from the author

This code was written by me (an undergraduate) in the course of me
writing my bachelor thesis. It shall illustrate how counterfactual
explanations can be used as an indicator for detecting unfair Machine
Learning models.

### Necessities

First, one needs to install Python 3.7.5.

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

Furthermore, the project is based on the '_compas-scores-two-years.csv_'
data set from: https://github.com/propublica/compas-analysis .
You will have to download it from there.

### What is the code doing?

The code pre-processes the data from '_compas-scores-two-years.csv_', such that
we can train a logistic regression with that data. Then, we compute counterfactual
explanations for the pre-processed vectors by solving a constrained optimization
problem. The counterfactuals will be stored within csv-files in the same directiory
as the code-files. In the end, you can visualize the results with the '_visualiztaion.py_'-file.
