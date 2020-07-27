import pandas as pd
from datetime import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import cvxpy as cp
import math
import sys


# Constants that will be used throughout this project.
CSV_FILE = "compas-scores-two-years.csv"
VECTOR_INDEX = {"age": 0,
                "priors_count": 1,
                "days_b_screening_arrest": 2,
                "is_recid": 3,
                "two_year_recid": 4,
                "sex": 5,
                "charge_degree": 6,
                "time_in_jail": 7,
                "race_African-American": 8,
                "race_Asian": 9,
                "race_Caucasian": 10,
                "race_Hispanic": 11,
                "race_Native American": 12,
                "race_Other": 13}
VECTOR_DIMENSION = len(VECTOR_INDEX)
ATTRIBUTE_NAMES = list(VECTOR_INDEX.keys())
ONE_HOT_VECTOR_START_INDEX = VECTOR_INDEX["race_African-American"]
LOWER_BOUNDS = [0, 0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
UPPER_BOUNDS = [np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf, 1, 1, 1, 1, 1, 1]


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


def store_results(result, sub_dict_to_store, file_name):
    """
    Function, that stores the result of the 'process_data'-function.

    :param result: 'dict'
                   The result-dictionary of the 'process_data'-function
    :param sub_dict_to_store: 'dict'
                               The sub-dictionary within 'result', which
                               shall be stored.
    :param file_name: 'str'
                      The filename of the resulting csv-file
    """
    pd.DataFrame(result[sub_dict_to_store]).to_csv(f"{file_name}.csv"
                                                   , index=False, sep=";"
                                                   , quoting=csv.QUOTE_NONE)


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
        if int(w @ x + b > 0) == y and one_hot_valid(x):
            return x
        else:
            return None

    # Recursion step
    x_copy = x.copy()
    x[index] = math.floor(x[index])
    x_copy[index] = math.ceil(x_copy[index])
    if in_boundaries(x, [index]) and in_boundaries(x_copy, [index]):
        result = rounding(x, w, b, y, index + 1)
        if result is None:
            return rounding(x_copy, w, b, y, index + 1)
        else:
            return result
    elif in_boundaries(x, [index]):
        return rounding(x, w, b, y, index + 1)
    elif in_boundaries(x_copy, [index]):
        return rounding(x_copy, w, b, y, index + 1)
    else:
        return None


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
    ones = np.zeros(VECTOR_DIMENSION)
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


def one_hot_valid(vec):
    """
    Checks if a vector contains a valid one-hot encoding.

    :param vec: 'numpy.ndarray'
                The vector, for which the included one-hot vector
                shall be checked.
    :return: 'bool'
             The test result.
    """
    # Check whether each entry is an integer
    for i in range(ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION):
        if not vec[i].is_integer():
            return False

    # Check whether we only have one 1 in the one-hot encoding
    if sum(vec[ONE_HOT_VECTOR_START_INDEX:]) != 1.0:
        return False

    # Return true, if no violation happened.
    return True


def in_boundaries(vec, index):
    """
    Checks if the values in a vector are within their bounds.

    :param vec: 'numpy.ndarray'
                The vector, for which the values shall be checked.
    :param index: 'range'
                  The indices for which the boundaries shall be checked
    :return: 'bool'
             The test result.
    """
    in_range = True
    for i in index:
        in_range = in_range and LOWER_BOUNDS[i] <= vec[i] <= UPPER_BOUNDS[i]
        if not in_range:
            break
    return in_range


def is_valid(vec, y, y_cf):
    """
    Checks if a vectors is plausible or not.

    :param vec: 'numpy.ndarray'
                 The counterfactual, that shall be tested
    :param y: 'numpy.int64'
               The original class of the original vector from vec
    :param y_cf: 'numpy.int64'
                 The class of the counterfactual 'vec'
    :return:
    """
    return in_boundaries(vec, range(VECTOR_DIMENSION)) and one_hot_valid(vec) and y != y_cf


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
        x_cf, y_cf = compute_cf(meta_data, vector)
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
            # See 'plot_histogram' and 'count_changes' in 'visualization.py' for
            # further information.
            x_cf = np.round(x_cf)
        if not_rounded:
            result["no_rounding_found"]["x"].append(list(vector))
            result["no_rounding_found"]["y"].append(vector_label)
            result["no_rounding_found"]["x_cf"].append(list(x_cf))
            result["no_rounding_found"]["y_cf"].append(y_cf)
        elif not is_valid(x_cf, vector_label, y_cf):
            result["non_valid_cf"]["x"].append(list(vector))
            result["non_valid_cf"]["y"].append(vector_label)
            result["non_valid_cf"]["x_cf"].append(list(x_cf))
            result["non_valid_cf"]["y_cf"].append(y_cf)
        else:
            result["valid_cf"]["x"].append(list(vector))
            result["valid_cf"]["y"].append(vector_label)
            result["valid_cf"]["x_cf"].append(list(x_cf))
            result["valid_cf"]["y_cf"].append(y_cf)
        sys.stdout.write(f"\rThe process is {i / len(data) * 100 :.2f}% complete.")
        sys.stdout.flush()
    return result


def get_data():
    """
    Reads the 'compas-scores-two-years.csv'-file from:

    'How We Analyzed the COMPAS Recidivism Algorithm - ProPublica
    2016 by Jeff Larson, Surya Mattu, Lauren Kirchner and Julia Angwin'

    (Needs to be downloaded from their github-repository:
     https://github.com/propublica/compas-analysis )

    and filters it with nearly the same conditions as in their analysis.

    After filtering the data-set, it continues with the preprocessing
    as described in the bachelor thesis.

    :return: 'pandas.core.frame.DataFrame', 'list'
             The pre-processed data from the data set 'compas-scores-two-years.csv'
             and a list of labels in the corresponding order to the data set.
    """

    # raw data
    recidivism_data_raw = pd.read_csv(CSV_FILE)

    # filter data
    recidivism_data_filtered = recidivism_data_raw.filter(items=["age"
        , "c_charge_degree"
        , "race"
        , "sex"
        , "priors_count"
        , "days_b_screening_arrest"
        , "decile_score"
        , "is_recid"
        , "two_year_recid"
        , "c_jail_in"
        , "c_jail_out"])
    recidivism_data_filtered = recidivism_data_filtered.query("days_b_screening_arrest <= 30"
                                                              ).query("days_b_screening_arrest >= -30"
                                                                      ).query("is_recid != -1"
                                                                              ).query("c_charge_degree != 0")

    # Create response variables
    label = []
    for decile_score in recidivism_data_filtered["decile_score"]:
        label.append(int(decile_score >= 4))

    # Time in jail in hours
    jail_time = [round(
        (dt.strptime(jail_out, '%Y-%m-%d %H:%M:%S') - dt.strptime(jail_in, '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600)
        for jail_in, jail_out in
        zip(recidivism_data_filtered["c_jail_in"], recidivism_data_filtered["c_jail_out"])]

    # Encode data numerically
    label_encoder = preprocessing.LabelEncoder()
    label_encoder.fit(recidivism_data_filtered["c_charge_degree"])
    recidivism_data_filtered["c_charge_degree"] = label_encoder.transform(recidivism_data_filtered["c_charge_degree"])
    label_encoder.fit(recidivism_data_filtered["sex"])
    recidivism_data_filtered["sex"] = label_encoder.transform(recidivism_data_filtered["sex"])

    # numerically encoded data and response variables
    recidivism_data = recidivism_data_filtered.filter(items=["age"
        , "priors_count"
        , "days_b_screening_arrest"
        , "is_recid"
        , "two_year_recid"
        , "race"
        , "sex"
        , "c_charge_degree"])
    recidivism_data["time_in_jail"] = jail_time
    recidivism_data.index = range(len(recidivism_data))
    recidivism_data.rename({"c_charge_degree": "charge_degree"}, axis=1, inplace=True)

    # One-hot encoding for the attribute 'race'
    recidivism_data = pd.get_dummies(recidivism_data)

    return recidivism_data, label


def main():
    ##### PREPROCESSING AND FURTHER PREPARATION #####

    # Read the data from 'compas-scores-two-years'
    recidivism_data, label = get_data()

    # Export filtered data
    pd.DataFrame({"x": recidivism_data.values.tolist(), "y": label}).to_csv("x_values.csv"
                                                                            , index=False, sep=";"
                                                                            , quoting=csv.QUOTE_NONE)

    # Split the data in a train- and a test set
    X_train, X_test, y_train, y_test = train_test_split(recidivism_data, label, test_size=0.33, random_state=1337)

    # Train the logistic regression
    log_reg = LogisticRegression(max_iter=1500).fit(X_train, y_train)

    # List for collecting the results
    results = []

    ##### COMPUTING COUNTERFACTUALS #####
    # 1. set of counterfactuals: ILP
    meta_data = MetaData(recidivism_data, cp.CBC, "Integer Linear Programming", False, log_reg)
    ILP_npa = process_data(meta_data)
    results.append(ILP_npa)
    store_results(ILP_npa, "valid_cf", "cf")

    # 2. set of counterfactuals: ILP + relaxation
    meta_data = MetaData(recidivism_data, cp.SCS, "Integer Linear Programming with relaxation", True, log_reg)
    ILP_wr_npa = process_data(meta_data)
    results.append(ILP_wr_npa)
    store_results(ILP_wr_npa, "valid_cf", "cf_wr")

    # report the results
    print("\n")
    print("Computation finished.")
    for res in results:
        print("\n")
        print("Experiment:", res["result_name"])
        print("Used solver:", res["used_solver"])
        print("Amount of valid counterfactuals:", len(res["valid_cf"]["x_cf"]))
        print("Amount of counterfactuals for which no rounding was found:", len(res["no_rounding_found"]["x_cf"]))
        print("Amount of not valid counterfactuals:", len(res["non_valid_cf"]["x_cf"]))

    print("\n")
    print("Accuracy score of the logistic regression:", log_reg.score(X_test, y_test))
    print("Parameter-values of the logistic regression:")
    print(f"w = {log_reg.coef_[0]}")
    print(f"b = {log_reg.intercept_}")


if __name__ == "__main__":
    main()
