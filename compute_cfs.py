import pandas as pd
import csv
import numpy as np
import cvxpy as cp
import math
import sys
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

import test_counterfactual as tc
from csv_parsing_writing import read_compas_data, store_result
from test_counterfactual import VECTOR_INDEX, VECTOR_DIMENSION, ONE_HOT_VECTOR_START_INDEX


class MetaData:
    """
    Meta-data object, that contains information that is necessary at
    multiple points in the computing process.
    """

    def __init__(self, data, solver, result_name, relaxation, classifier):
        """
        :param data: 'pandas.core.frame.DataFrame'
                     The pre-processed vectors from the database
        :param solver: 'str'
                       The solver, that will be used to compute the counterfactuals
        :param result_name: 'str'
                            The name (respectively id) of the result
        :param relaxation: 'bool'
                           Tells if relaxation should be applied or not
        :param classifier: 'sklearn.linear_model.LogisticRegression'
                           The used classifier
        """
        self.data = data
        self.solver = solver
        self.result_name = result_name
        self.relaxation = relaxation
        self.classifier = classifier


def choose_descendant(x1, x2, w, b, y, index):
    if tc.in_boundaries(x1, [index]) and tc.in_boundaries(x2, [index]):
        result = rounding(x1, w, b, y, index + 1)
        if result is None:
            return rounding(x2, w, b, y, index + 1)
        else:
            return result
    elif tc.in_boundaries(x1, [index]):
        return rounding(x1, w, b, y, index + 1)
    elif tc.in_boundaries(x2, [index]):
        return rounding(x2, w, b, y, index + 1)
    else:
        return None


def rounding(x, w, b, y, index):
    """
    A function to produce a tree in order to get a correctly
    rounded integer vector with respect to its class.
    :param x: 'numpy.ndarray'
              The counterfactual that shall be rounded
    :param w: 'numpy.ndarray'
              The weight-vector of our logistic regression
    :param b: 'numpy.ndarray'
              The bias of our logistic regression
    :param y: 'numpy.int64'
              The label for 'x'
    :param index: 'int'
                  The index, for the element that shall be rounded.
    :return: 'numpy.ndarray'
             A counterfactual with only integer entries.
    """
    # Base case
    if index == len(x):
        if int(w @ x + b > 0) == y and tc.one_hot_valid(x):
            return x
        else:
            return None

    # Recursion step
    x_copy = x.copy()
    index_value = x[index]
    x[index] = math.floor(x[index])
    x_copy[index] = math.ceil(x_copy[index])
    if index_value - np.round(index_value) < 0:
        return choose_descendant(x_copy, x, w, b, y, index)
    else:
        return choose_descendant(x, x_copy, w, b, y, index)


def manhatten_dist(x, x_cf):
    """
    The usual Manhatten-distance as known.
    :param x: 'numpy.ndarray'
               The original vector.
    :param x_cf: 'cvxpy.expression.variable.Variable'
                 The counterfactual explanation for 'x'.
    :return: sum: 'int'
                  The manhatten distance between x and x_cf.
    """
    sum = 0
    for j in range(x.shape[0]):
        sum += cp.abs(x[j] - x_cf[j])
    return sum


def compute_cf(meta_data, vector):
    """
    The function that solves the optimization problem for computing
    counterfactual explanations, that was phrased in:
    'On the computation of counterfactual explanations -- A survey
    2019 by André Artelt and Barbara Hammer'
    with an extra few boundary constraints, that were stated in the
    bachelor thesis.
    :param meta_data: 'MetaData'
                      Carries information about how the calculation
                      shall be structured.
    :param vector: 'numpy.ndarray'
                    The vector, for which a counterfactual shall be
                    computed.
    :return: 'numpy.ndarray', 'numpy.int64'
             The result of the optimization problem and it's label
             respectively class.
    """

    # predicts the opposite of the current prediction
    y = meta_data.classifier.predict(vector.reshape(1, -1))
    y_target = 1 - y

    # The formulation of constraint for minimizing the optimization problem
    # of hyperplane models assumed y to be in {-1, 1}
    if y_target == 0:
        y_target = -1

    x_cf = cp.Variable(vector.shape[0], integer=not meta_data.relaxation)
    objective = cp.Minimize(manhatten_dist(vector, x_cf))

    # Forming constraints like in 3.2
    weight_vector = meta_data.classifier.coef_[0]
    q = -y_target * weight_vector
    c = -meta_data.classifier.intercept_ * y_target

    # strict inequalities are not allowed
    constraints = [q.T @ x_cf + c <= 0]

    # Adding constraint for race-attribute
    ones = np.zeros(tc.VECTOR_DIMENSION)
    for i in [VECTOR_INDEX["race_African-American"], VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Caucasian"]
        , VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Native American"], VECTOR_INDEX["race_Other"]]:
        ones[i] = 1
    constraints += [ones.T @ x_cf == 1]

    # lower bounds
    lower_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    for i in [VECTOR_INDEX["age"], VECTOR_INDEX["priors_count"]
        , VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["charge_degree"]
        , VECTOR_INDEX["time_in_jail"], VECTOR_INDEX["race_African-American"]
        , VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Caucasian"]
        , VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Native American"]
        , VECTOR_INDEX["race_Other"]]:
        lower_bounds[i, i] = 1

    constraints += [-(lower_bounds @ x_cf) <= 0]

    # upper bounds
    upper_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    upper_bounds_indices = [VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["charge_degree"]
        , VECTOR_INDEX["race_African-American"]
        , VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Caucasian"]
        , VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Native American"]
        , VECTOR_INDEX["race_Other"]]
    for i in upper_bounds_indices:
        upper_bounds[i, i] = 1

    ub_vector = np.zeros(VECTOR_DIMENSION)
    for i in upper_bounds_indices:
        ub_vector[i] = 1
    constraints += [upper_bounds @ (x_cf - ub_vector) <= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=meta_data.solver)
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        raise ValueError("problem is infeasible")

    return x_cf.value, meta_data.classifier.predict(x_cf.value.reshape(1, -1))[0]


def process_data(meta_data):
    """
    The function that initializes and organizes the computation
    of the counterfactuals for the pre-processed data.
    :param meta_data: 'MetaData'
                      Carries information about how the calculation
                      shall be structured.
    :return: 'dict'
             A result-dictionary containing the original x-vectors, their
             y-values as well as their counterfactuals and their y_cf values.
             It is structured into sub-dictionaries, that are necessary due to
             the filtering process that the 'process_data' function applies.
    """

    # A dictionary to classify the results
    result = {
        "result_name": meta_data.result_name,
        "used_solver": meta_data.solver,
        # Valid Counterfactuals
        "valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        # Counterfactuals, that do not pass the filter
        "non_valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        # Counterfactuals, for which no valid rounding was found
        "no_rounding_found": {"x": [], "y": [], "x_cf": [], "y_cf": []}
    }

    # Initialize counterfactual-computation for each vector in the given data set
    data = meta_data.data.to_numpy()
    for i, vector in enumerate(data):
        vector_label = meta_data.classifier.predict(vector.reshape(1, -1))[0]
        try:
            x_cf, y_cf = compute_cf(meta_data, vector)
        except ValueError:
            print("\nNo counterfactual found for:", vector)
            continue
        not_rounded = False

        if meta_data.relaxation:
            x_cf = rounding(x_cf, meta_data.classifier.coef_[0], meta_data.classifier.intercept_
                            , 1 - vector_label, 0)
            if x_cf is None:
                x_cf, y_cf = compute_cf(meta_data, vector)
                not_rounded = True
            else:
                y_cf = meta_data.classifier.predict(x_cf.reshape(1, -1))[0]
        else:
            # The values need to be integers. Sometimes, even though we use
            # CBC (in this case), we have values that are only very close to integers.
            # If we do not round here, we probably will have non-zero entries in rows
            # where we actually should have zero-entries, when we count the changes.
            # See 'count_changes_for_groups' for further information.
            x_cf = np.round(x_cf)

        if not_rounded:
            result["no_rounding_found"]["x"].append(list(vector))
            result["no_rounding_found"]["y"].append(vector_label)
            result["no_rounding_found"]["x_cf"].append(list(x_cf))
            result["no_rounding_found"]["y_cf"].append(y_cf)
        elif not tc.is_valid(x_cf, vector_label, y_cf):
            result["non_valid_cf"]["x"].append(list(vector))
            result["non_valid_cf"]["y"].append(vector_label)
            result["non_valid_cf"]["x_cf"].append(list(x_cf))
            result["non_valid_cf"]["y_cf"].append(y_cf)
        else:
            result["valid_cf"]["x"].append(list(vector))
            result["valid_cf"]["y"].append(vector_label)
            result["valid_cf"]["x_cf"].append(list(x_cf))
            result["valid_cf"]["y_cf"].append(y_cf)

        sys.stdout.write(
            f"\rComputing counterfactuals for experiment: '{meta_data.result_name}'. {i / len(data) * 100 :.2f}% complete.")
        sys.stdout.flush()

    return result


def run_experiment(without_sens_attributes):
    """
    Initializes the pre-processing of the data and the computation
    of the counterfactuals. Furthermore, it prints the result of the
    counterfactual-computation process.
    """
    # Read the data from 'compas-scores-two-years'
    recidivism_data, label, recidivism_data_concealed = read_compas_data()

    # Export filtered data
    pd.DataFrame({"x": recidivism_data.values.tolist(), "y": label}).to_csv("x_values.csv"
                                                                            , index=False, sep=";"
                                                                            , quoting=csv.QUOTE_NONE)

    # Split the data in a train- and a test set
    if without_sens_attributes:
        X_train, X_test, y_train, y_test = train_test_split(recidivism_data_concealed, label, test_size=0.33,
                                                            random_state=1337)
    else:
        X_train, X_test, y_train, y_test = train_test_split(recidivism_data, label, test_size=0.33, random_state=1337)

    # Train the logistic regression
    log_reg = LogisticRegression(max_iter=400).fit(X_train, y_train)

    # List for collecting the results
    results = []

    ##### COMPUTING COUNTERFACTUALS #####

    # 1. set of counterfactuals: ILP

    meta_data = MetaData(recidivism_data, cp.CBC, "Integer Linear Programming", False, log_reg)
    ILP_npa = process_data(meta_data)
    results.append(ILP_npa)
    store_result(ILP_npa, "valid_cf", "cf")

    # 2. set of counterfactuals: ILP + relaxation
    meta_data = MetaData(recidivism_data, cp.SCS, "Integer Linear Programming with relaxation", True, log_reg)
    ILP_wr_npa = process_data(meta_data)
    results.append(ILP_wr_npa)
    store_result(ILP_wr_npa, "valid_cf", "cf_wr")

    ##### REPORT RESULTS #####
    print("\n")
    print("Computation finished.")
    for res in results:
        print("\n")
        print("Experiment:", res["result_name"])
        print("Used solver:", res["used_solver"])
        print("Amount of plausible counterfactuals:", len(res["valid_cf"]["x_cf"]))
        print("Amount of counterfactuals for which no rounding was found:", len(res["no_rounding_found"]["x_cf"]))
        print("Amount of not plausible counterfactuals:", len(res["non_valid_cf"]["x_cf"]))

    print("\n")
    print("Accuracy score of the logistic regression:", log_reg.score(X_test, y_test))
    print("Parameter-values of the logistic regression:")
    print(f"w = {log_reg.coef_[0]}")
    print(f"b = {log_reg.intercept_}")

if __name__ == "__main__":
    run_experiment()