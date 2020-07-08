import pandas as pd
from datetime import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import cvxpy as cp
import math

CSV_FILE = "compas-scores-two-years.csv"
VECTOR_INDEX = {"age": 0,
                "priors_count": 1,
                "days_b_screening_arrest": 2,
                "is_recid": 3,
                "two_year_recid": 4,
                "sex": 5,
                "race": 6,
                "charge_degree": 7,
                "time_in_jail": 8}
VECTOR_DIMENSION = len(VECTOR_INDEX)
LOWER_BOUNDS = [0, 0, -np.inf, 0, 0, 0, 0, 0, 0]
UPPER_BOUNDS = [np.inf, np.inf, np.inf, 1, 1, 1, 1, 5, np.inf]


def store_results(result, result_type, file_name):
    pd.DataFrame(result[result_type]).to_csv(f"{file_name}.csv"
                                             , index=False, sep=";"
                                             , quoting=csv.QUOTE_NONE)


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


def descent(x_fc, x_sc, w, b, y, index):
    """
    A function to choose the right sub-tree in the decision tree.

    :param x_fc: The 'first choice' vector. A vector with better fitting costs.
    :param x_sc: The 'second choice' vector.  A vector with better less fitting costs.
    :param w: The weight-vector of our logistic regression
    :param b: The bias of our logistic regression
    :param y: The label for 'x'
    :param index: The index, for the element that shall be rounded.
    :return: The correct leaf from the decision tree.
    """
    result = rounding(x_fc, w, b, y, index)
    if result is None:
        return rounding(x_sc, w, b, y, index)
    else:
        return result


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
    if index == len(x):
        if int(w @ x + b > 0) == y:
            return x
        else:
            return None

    x_copy = x.copy()
    x[index] = math.floor(x[index])
    x_copy[index] = math.ceil(x_copy[index])
    if in_boundaries(x, [index]) and in_boundaries(x_copy, [index]):
        return descent(x, x_copy, w, b, y, index + 1)
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

    # cvxpy doesn't allow strict inequalities
    constraints = [q.T @ x_cf + c <= 0]

    # protected constraints
    if meta_data.protected_attributes:
        coefficients = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
        coefficients[VECTOR_INDEX["age"], VECTOR_INDEX["age"]] = 1
        coefficients[VECTOR_INDEX["sex"], VECTOR_INDEX["sex"]] = 1
        coefficients[VECTOR_INDEX["race"], VECTOR_INDEX["race"]] = 1

        x_constants = np.zeros(VECTOR_DIMENSION)
        x_constants[VECTOR_INDEX["age"]] = vector[VECTOR_INDEX["age"]]
        x_constants[VECTOR_INDEX["sex"]] = vector[VECTOR_INDEX["sex"]]
        x_constants[VECTOR_INDEX["race"]] = vector[VECTOR_INDEX["race"]]

        constraints += [coefficients @ x_cf == x_constants]

    # lower bounds
    lower_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    for i in [VECTOR_INDEX["age"], VECTOR_INDEX["priors_count"]
        , VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["race"]
        , VECTOR_INDEX["charge_degree"], VECTOR_INDEX["time_in_jail"]]:
        lower_bounds[i, i] = 1

    constraints += [-(lower_bounds @ x_cf) <= 0]

    # upper bounds
    upper_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    for i in [VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["race"]
        , VECTOR_INDEX["charge_degree"]]:
        upper_bounds[i, i] = 1

    ub_vector = np.zeros(VECTOR_DIMENSION)
    for i in [VECTOR_INDEX["is_recid"], VECTOR_INDEX["two_year_recid"]
        , VECTOR_INDEX["sex"], VECTOR_INDEX["charge_degree"]]:
        ub_vector[i] = 1

    ub_vector[VECTOR_INDEX["race"]] = 5
    constraints += [upper_bounds @ (x_cf - ub_vector) <= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=meta_data.solver)
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        raise ValueError("problem is infeasible")

    return x_cf.value, meta_data.classifier.predict(x_cf.value.reshape(1, -1))[0]


def in_boundaries(vec, index):
    in_range = True
    for i in index:
        in_range = in_range and LOWER_BOUNDS[i] <= vec[i] <= UPPER_BOUNDS[i]
        if not in_range:
            break
    return in_range


def process_data(meta_data):
    """

    :param meta_data:
    :return:
    """
    # A dictionary to classify the results
    result = {
        "result_name": meta_data.result_name,
        "used_solver": meta_data.solver,
        "valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "non_valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "no_cf_found": {"x": [], "y": []},
        "not_solvable": 0  # Amount of vectors for which no cf have been found
    }

    # Initialize counterfactual-computation for each vector in the given data set
    data = meta_data.data.to_numpy()
    for i, vector in enumerate(data):
        print(f"{i / len(data) * 100 :.2f}%")
        vector_label = meta_data.classifier.predict(vector.reshape(1, -1))[0]
        try:
            x_cf, y_cf = compute_cf(meta_data, vector)
            if meta_data.relaxation:
                x_cf = rounding(x_cf, meta_data.classifier.coef_[0], meta_data.classifier.intercept_
                                  , y_cf, 0)
            else:
                # Values need to be integers and sometimes even though we use
                # CBC (in this case) we have .9999... values. If we do not round here,
                # we probably will have non-zero entries in rows where we actually should have
                # zero-entries, when we count the changes. See 'detect_amount_of_changes' in
                # 'visualization.py' for further information.
                x_cf = np.round(x_cf)
            if vector_label == y_cf or x_cf is None:
                result["no_cf_found"]["x"].append(list(vector))
                result["no_cf_found"]["y"].append(vector_label)
                result["non_valid_cf"]["x_cf"].append(list(x_cf))
                result["non_valid_cf"]["y_cf"].append(y_cf)
            elif not in_boundaries(x_cf, range(VECTOR_DIMENSION)):
                result["non_valid_cf"]["x"].append(list(vector))
                result["non_valid_cf"]["y"].append(vector_label)
                result["non_valid_cf"]["x_cf"].append(list(x_cf))
                result["non_valid_cf"]["y_cf"].append(y_cf)
            else:
                result["valid_cf"]["x"].append(list(vector))
                result["valid_cf"]["y"].append(vector_label)
                result["valid_cf"]["x_cf"].append(list(x_cf))
                result["valid_cf"]["y_cf"].append(y_cf)
        except ValueError:
            result["not_solvable"] += 1

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
    label_encoder.fit(recidivism_data_filtered["race"])
    recidivism_data_filtered["race"] = label_encoder.transform(recidivism_data_filtered["race"])
    label_encoder.fit(recidivism_data_filtered["sex"])
    recidivism_data_filtered["sex"] = label_encoder.transform(recidivism_data_filtered["sex"])

    # numerically encoded data and response variables
    recidivism_data = recidivism_data_filtered.filter(items=["age"
        , "priors_count"
        , "days_b_screening_arrest"
        , "is_recid"
        , "two_year_recid"
        , "sex"
        , "race"
        , "c_charge_degree"])
    recidivism_data["time_in_jail"] = jail_time
    recidivism_data.index = range(len(recidivism_data))
    recidivism_data.rename({"c_charge_degree": "charge_degree"}, axis=1, inplace=True)

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
    """
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
    store_results(ILP_npa, "valid_cf", "valid_cf")
    """
    # 3. set of counterfactuals: ILP + relaxation + protected attributes
    meta_data = MetaData(recidivism_data, cp.SCS, True, "ILP - wr - pa", True, log_reg)
    ILP_wr = process_data(meta_data)
    results.append(ILP_wr)
    store_results(ILP_wr, "valid_cf", "valid_cf_wr")
    store_results(ILP_wr, "non_valid_cf", "non_valid_cf_wr")

    # 4. set of counterfactuals: ILP + relaxation + not protecting attributes
    meta_data = MetaData(recidivism_data, cp.SCS, False, "ILP - wr - npa", True, log_reg)
    ILP_wr_npa = process_data(meta_data)
    results.append(ILP_wr_npa)
    store_results(ILP_wr_npa, "valid_cf", "valid_cf_wr_npa")
    store_results(ILP_wr_npa, "non_valid_cf", "non_valid_cf_wr_npa")

    # report the results
    print("\n")
    print("Computation finished.")
    for res in results:
        print("\n")
        print("Experiment:", res["result_name"])
        print("Used solver:", res["used_solver"])
        print("Amount of valid counterfactuals", len(res["valid_cf"]["x_cf"]))
        print("Amount of counterfactuals, that exceed the boundaries:", len(res["non_valid_cf"]["x_cf"]))
        print("Amount of instances, for which no counterfactual have been found:", len(res["no_cf_found"]["x"]))
        print("Amount of not optimal or infeasible problems:", res["not_solvable"])

    print("\n")
    print("Accuracy score of the logistic regression:", log_reg.score(X_test, y_test))
    print("Parameter-values of the logistic regression:")
    print(f"w = {log_reg.coef_[0]}")
    print(f"b = {log_reg.intercept_}")


if __name__ == "__main__":
    main()
