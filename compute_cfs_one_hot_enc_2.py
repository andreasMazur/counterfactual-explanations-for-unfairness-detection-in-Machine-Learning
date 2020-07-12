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

    def __init__(self, data, solver, protected_attributes, result_name, relaxation, classifier):
        """

        :param data:
        :param solver:
        :param protected_attributes:
        :param result_name:
        :param relaxation:
        :param classifier:
        """
        self.data = data
        self.solver = solver
        self.protected_attributes = protected_attributes
        self.result_name = result_name
        self.relaxation = relaxation
        self.classifier = classifier


def store_results(result, result_type, file_name):
    pd.DataFrame(result[result_type]).to_csv(f"{file_name}.csv"
                                             , index=False, sep=";"
                                             , quoting=csv.QUOTE_NONE)


def rounding(x, w, b, y, index):
    """
    A function to produce a decision-tree in order to get a correctly
    rounded integer vector with respect to its class.

    :param x: The counterfactual that shall be rounded
    :param w: The weight-vector of our logistic regression
    :param b: The bias of our logistic regression
    :param y: The label for 'x'
    :param index: The index, for the element that shall be rounded.
    :return: A counterfactual with only integer entries.
    """
    # Base case
    if index == len(x):
        if int(w @ x + b > 0) == y and one_hot_valid(x):
            return x
        else:
            return None

    # Round one-hot-encoding
    if index >= ONE_HOT_VECTOR_START_INDEX + 1 and x[index - 1] == 1:
        for i in range(index, len(x)):
            x[i] = 0
        return rounding(x, w, b, y, len(x))

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
    sum = 0
    for j in range(x.shape[0]):
        sum += cp.abs(x[j] - x_cf[j])
    return sum


def compute_cf(meta_data, vector):
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

    # protected constraints
    # TODO: Write with for-loop
    if meta_data.protected_attributes:
        coefficients = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
        coefficients[VECTOR_INDEX["age"], VECTOR_INDEX["age"]] = 1
        coefficients[VECTOR_INDEX["sex"], VECTOR_INDEX["sex"]] = 1
        coefficients[VECTOR_INDEX["race_African-American"], VECTOR_INDEX["race_African-American"]] = 1
        coefficients[VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Asian"]] = 1
        coefficients[VECTOR_INDEX["race_Caucasian"], VECTOR_INDEX["race_Caucasian"]] = 1
        coefficients[VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Hispanic"]] = 1
        coefficients[VECTOR_INDEX["race_Native American"], VECTOR_INDEX["race_Native American"]] = 1
        coefficients[VECTOR_INDEX["race_Other"], VECTOR_INDEX["race_Other"]] = 1

        x_constants = np.zeros(VECTOR_DIMENSION)
        x_constants[VECTOR_INDEX["age"]] = vector[VECTOR_INDEX["age"]]
        x_constants[VECTOR_INDEX["sex"]] = vector[VECTOR_INDEX["sex"]]
        x_constants[VECTOR_INDEX["race_African-American"]] = vector[VECTOR_INDEX["race_African-American"]]
        x_constants[VECTOR_INDEX["race_Asian"]] = vector[VECTOR_INDEX["race_Asian"]]
        x_constants[VECTOR_INDEX["race_Caucasian"]] = vector[VECTOR_INDEX["race_Caucasian"]]
        x_constants[VECTOR_INDEX["race_Hispanic"]] = vector[VECTOR_INDEX["race_Hispanic"]]
        x_constants[VECTOR_INDEX["race_Native American"]] = vector[VECTOR_INDEX["race_Native American"]]
        x_constants[VECTOR_INDEX["race_Other"]] = vector[VECTOR_INDEX["race_Other"]]

        constraints += [coefficients @ x_cf == x_constants]

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
    for i in [VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["charge_degree"]
        , VECTOR_INDEX["race_African-American"]
        , VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Caucasian"]
        , VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Native American"]
        , VECTOR_INDEX["race_Other"]]:
        upper_bounds[i, i] = 1

    ub_vector = np.zeros(VECTOR_DIMENSION)
    for i in [VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["charge_degree"]
        , VECTOR_INDEX["race_African-American"]
        , VECTOR_INDEX["race_Asian"], VECTOR_INDEX["race_Caucasian"]
        , VECTOR_INDEX["race_Hispanic"], VECTOR_INDEX["race_Native American"]
        , VECTOR_INDEX["race_Other"]]:
        ub_vector[i] = 1
    constraints += [upper_bounds @ (x_cf - ub_vector) <= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=meta_data.solver)
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        raise ValueError("problem is infeasible")

    return x_cf.value, meta_data.classifier.predict(x_cf.value.reshape(1, -1))[0]


def one_hot_valid(vec):
    seen_one = False
    for i in range(ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION):
        # Set the 'seen a one'-flag, since there only can be one one.
        if vec[i] == 1:
            seen_one = True
        # If there's a second one, return false.
        elif seen_one and (vec[i] == 1):
            return False
        # If the value is neither 1 nor 0, return false.
        elif vec[i] != 1 and vec[i] != 0:
            return False
    # Return true, if no violation happened.
    return seen_one


def in_boundaries(vec, index):
    in_range = True
    for i in index:
        in_range = in_range and LOWER_BOUNDS[i] <= vec[i] <= UPPER_BOUNDS[i]
        if not in_range:
            break
    return in_range


def is_valid(vec, y, y_cf):
    return in_boundaries(vec, range(VECTOR_DIMENSION)) and one_hot_valid(vec) and y != y_cf


def process_data(meta_data):
    """

    :param meta_data:
    :return:
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
        "not_rounding_found": {"x": [], "y": [], "x_cf": [], "y_cf": []}
    }

    # Initialize counterfactual-computation for each vector in the given data set
    data = meta_data.data.to_numpy()
    for i, vector in enumerate(data):
        vector_label = meta_data.classifier.predict(vector.reshape(1, -1))[0]
        x_cf, y_cf = compute_cf(meta_data, vector)
        not_rounded = False
        if meta_data.relaxation:
            try:
                x_cf = rounding(x_cf, meta_data.classifier.coef_[0], meta_data.classifier.intercept_
                                , 1 - vector_label, 0)
                # Just to be sure, but actually 'y_cf = 1 - vector_label' could stay here.
                y_cf = meta_data.classifier.predict(x_cf.reshape(1, -1))[0]
            except AttributeError:
                x_cf, y_cf = compute_cf(meta_data, vector)
                not_rounded = True
        else:
            # Values need to be integers and sometimes even though we use
            # CBC (in this case) we have .9999... values. If we do not round here,
            # we probably will have non-zero entries in rows where we actually should have
            # zero-entries, when we count the changes. See 'detect_amount_of_changes' in
            # 'visualization.py' for further information.
            x_cf = np.round(x_cf)
        if not_rounded:
            result["not_rounding_found"]["x"].append(list(vector))
            result["not_rounding_found"]["y"].append(vector_label)
            result["not_rounding_found"]["x_cf"].append(list(x_cf))
            result["not_rounding_found"]["y_cf"].append(y_cf)
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
    Reads the 'compas-scores-two-years.csv'-file, filters it with mostly the same conditions
    that have been used in the Propublica-analysis and provides train and test sets for
    the logistic regression.

    :return: 'pandas.core.frame.DataFrame', 'list'
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
    # 1. set of counterfactuals: ILP + protected attributes
    meta_data = MetaData(recidivism_data, cp.CBC, True, "ILP - pa", False, log_reg)
    ILP = process_data(meta_data)
    results.append(ILP)
    store_results(ILP, "valid_cf", "valid_cf")

    # 2. set of counterfactuals: ILP + not protecting attributes
    meta_data = MetaData(recidivism_data, cp.CBC, False, "ILP - npa", False, log_reg)
    ILP_npa = process_data(meta_data)
    results.append(ILP_npa)
    store_results(ILP_npa, "valid_cf", "valid_cf_npa")

    # 3. set of counterfactuals: ILP + relaxation + protected attributes
    meta_data = MetaData(recidivism_data, cp.SCS, True, "ILP - wr - pa", True, log_reg)
    ILP_wr = process_data(meta_data)
    results.append(ILP_wr)
    store_results(ILP_wr, "valid_cf", "valid_cf_wr")

    # 4. set of counterfactuals: ILP + relaxation + not protecting attributes
    meta_data = MetaData(recidivism_data, cp.SCS, False, "ILP - wr - npa", True, log_reg)
    ILP_wr_npa = process_data(meta_data)
    results.append(ILP_wr_npa)
    store_results(ILP_wr_npa, "valid_cf", "valid_cf_wr_npa")

    # report the results
    print("\n")
    print("Computation finished.")
    for res in results:
        print("\n")
        print("Experiment:", res["result_name"])
        print("Used solver:", res["used_solver"])
        print("Amount of plausible counterfactuals:", len(res["valid_cf"]["x_cf"]))
        print("Amount of counterfactuals for which no rounding was found:", len(res["not_rounding_found"]["x_cf"]))
        print("Amount of not plausible counterfactuals:", len(res["non_valid_cf"]["x_cf"]))

    print("\n")
    print("Accuracy score of the logistic regression:", log_reg.score(X_test, y_test))
    print("Parameter-values of the logistic regression:")
    print(f"w = {log_reg.coef_[0]}")
    print(f"b = {log_reg.intercept_}")


if __name__ == "__main__":
    main()
