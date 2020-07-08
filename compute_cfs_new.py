import pandas as pd
from datetime import datetime as dt
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import csv
import numpy as np
import cvxpy as cp

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
LOWER_BOUNDS = [1, 0, -np.inf, 0, 0, 0, 0, 0, 0]
UPPER_BOUNDS = [np.inf, np.inf, np.inf, 1, 1, np.inf, 1, 5, 1]


class MetaData:
    """
    Meta-data object, that contains data necessary at
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


def manhatten_dist(x, x_cf):
    sum = 0
    for j in range(x.shape[0]):
        sum += cp.abs(x[j] - x_cf[j])
    return sum


def compute_cf(meta_data, vector):
    # predicts the opposite of the current prediction
    y = meta_data.classifier.predict(vector.reshape(1, -1))
    y_target = 1 - y

    # w.l.o.g. y \in {-1,1}
    if y_target == 0:
        y_target = -1

    x_cf = cp.Variable(vector.shape[0], integer=meta_data.relaxation)

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
        , VECTOR_INDEX["two_year_recid"], VECTOR_INDEX["sex"]
        , VECTOR_INDEX["race"], VECTOR_INDEX["charge_degree"]
        , VECTOR_INDEX["time_in_jail"]]:
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
        "no_cf_found": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "not_solvable": 0  # Amount of vectors for which no cf have been found
    }

    # Initialize counterfactual-computation for each vector in the given data set
    data = meta_data.data.to_numpy()
    for vector in data:
        try:
            x_cf, y_cf = compute_cf(meta_data, vector)
            x_cf = rounding(x_cf)
            if y == y_cf:
                result["no_cf_found"]["x"].append(list(x))
                result["no_cf_found"]["y"].append(y)
                result["no_cf_found"]["x_cf"].append(list(x_cf))
                result["no_cf_found"]["y_cf"].append(y_cf)
            elif not in_boundaries(x_cf, range(VECTOR_DIMENSION)):
                result["non_valid_cf"]["x"].append(list(x))
                result["non_valid_cf"]["y"].append(y)
                result["non_valid_cf"]["x_cf"].append(list(x_cf))
                result["non_valid_cf"]["y_cf"].append(y_cf)
            else:
                result["valid_cf"]["x"].append(list(x))
                result["valid_cf"]["y"].append(y)
                result["valid_cf"]["x_cf"].append(list(x_cf))
                result["valid_cf"]["y_cf"].append(y_cf)
        except ValueError:
            result["not_solvable"] += 1


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

    ##### COMPUTING COUNTERFACTUALS #####
    result = process_data()


if __name__ == "__main__":
    main()
