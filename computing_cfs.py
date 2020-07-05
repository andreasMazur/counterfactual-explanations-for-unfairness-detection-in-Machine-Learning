import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
import cvxpy as cp
import csv
import math

# The dimension of a vector within our data set
VECTOR_DIMENSION = 9


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

    x_1 = x.copy()
    x_2 = x.copy()
    x_2[index] = math.floor(x_2[index])
    x_1[index] = math.ceil(x_1[index])
    if in_boundaries_2(x_2, [index]) and in_boundaries_2(x_1, [index]):
        return descent(x_2, x_1, w, b, y, index + 1)
    elif in_boundaries_2(x_2, [index]):
        return rounding(x_2, w, b, y, index + 1)
    elif in_boundaries_2(x_1, [index]):
        return rounding(x_1, w, b, y, index + 1)
    else:
        return None


def in_boundaries_2(vec, index):
    in_range = True
    low_vec = [1, 0, -np.inf, 0, 0, 0, 0, 0, 0]
    high_vec = [np.inf, np.inf, np.inf, 1, 1, np.inf, 1, 5, 1]
    for i in index:
        in_range = in_range and low_vec[i] <= vec[i] <= high_vec[i]
        if not in_range:
            break
    return in_range


def rounding_procedure(data, model):
    """
    Initialize the rounding process for each counterfactual in 'data'
    and return the changes.

    :param data: 'dict'
                 A dictionary divided in valid and non-valid counterfactuals with:
                 - the actual feature vectors x
                 - their labels y
                 - their counterfactuals x_cf
                 - their counterfactuals labels y_cf
    :param model: 'sklearn.linear_model.LogisticRegression'
                  The used classifier for predicting the initial
                  Counterfactual explanations
    :return: 'numpy.ndarray'
             The rounded counterfactual explanations
    """
    print("Rounding values..")

    # Round each initial counterfactual explanation
    for sub_dataset in ["valid_cf", "non_valid_cf"]:
        new_cfs = []
        for i, x_cf in enumerate(data[sub_dataset]["x_cf"]):
            x_cf_rounded = rounding(x_cf, model.coef_[0], model.intercept_, data[sub_dataset]["y_cf"][i], 0)
            new_cfs.append(x_cf_rounded)
        data[sub_dataset]["x_cf"] = new_cfs

    return data


def manhatten_dist(x, x_cf):
    sum = 0
    for j in range(x.shape[0]):
        sum += cp.abs(x[j] - x_cf[j])
    return sum


def store_results(result, result_type, file_name):
    pd.DataFrame(result[result_type]).to_csv(f"{file_name}.csv"
                                             , index=False, sep=";"
                                             , quoting=csv.QUOTE_NONE)


def compute_cf(classifier, x, protected, solver, integer=False):
    """
    :param integer: 'bool'
                    Whether or not the result only contains integer-values.
    :param solver: The solver, which will solve the optimization problem.
    :param protected: 'bool'
                     Whether or not protected attributes should be
                     held protected
    :param classifier: 'sklearn.linear_model.LogisticRegression'
    :param x: 'numpy.ndarray'
              The instance for which a counterfactuals shall be computed
    :return: prob.status, (x, y, x_cf, y_cf)
    """

    # predicts the opposite of the current prediction
    y = classifier.predict(x.reshape(1, -1))
    y_target = 1 - y

    # w.l.o.g. y \in {-1,1}
    if y_target == 0:
        y_target = -1

    x_cf = cp.Variable(x.shape[0], integer=integer)

    objective = cp.Minimize(manhatten_dist(x, x_cf))

    # Forming constraints like in 3.2
    weight_vector = classifier.coef_[0]
    q = -y_target * weight_vector
    c = -classifier.intercept_ * y_target

    # cvxpy doesn't allow strict inequalities
    constraints = [q.T @ x_cf + c <= 0]

    # protected constraints
    if protected:
        coefficients = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
        coefficients[0, 0] = 1  # age
        # coefficients[3, 3] = 1  # decile_score
        coefficients[6, 6] = 1  # sex
        coefficients[7, 7] = 1  # race

        x_constants = np.zeros(VECTOR_DIMENSION)
        x_constants[0] = x[0]
        # x_constants[3] = x[3]
        x_constants[6] = x[6]
        x_constants[7] = x[7]

        constraints += [coefficients @ x_cf == x_constants]

    # lower bounds
    lower_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    for i in [1, 2, 3, 4, 5, 8]:
        lower_bounds[i, i] = 1

    lb_vector = np.zeros(VECTOR_DIMENSION)
    for i in [3, 4, 8]:
        lb_vector[i] = 1

    constraints += [-(lower_bounds @ (x_cf - lb_vector)) <= 0]

    # upper bounds
    upper_bounds = np.zeros((VECTOR_DIMENSION, VECTOR_DIMENSION))
    for i in [3, 4, 8]:
        upper_bounds[i, i] = 1

    ub_vector = np.zeros(VECTOR_DIMENSION)
    for i in [3, 4, 8]:
        ub_vector[i] = 2

    constraints += [-(upper_bounds @ (ub_vector - x_cf)) <= 0]

    # Solve the problem
    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver)
    if prob.status != "optimal" and prob.status != "optimal_inaccurate":
        raise ValueError("problem is infeasible")

    return x, y[0], x_cf.value, classifier.predict(x_cf.value.reshape(1, -1))[0]


def process_data(classifier, data, solver, result_name, integer=False, protected=True):
    """
    :param integer: 'bool'
                    Whether or not the result only contains integer-values.
    :param result_name: 'str'
                         The identifier for the resulting dictionary.
    :param solver: The solver, which will solve the optimization problem.
    :param protected: 'bool'
                     Whether or not protected attributes will be
                     held protected
    :param classifier: 'sklearn.linear_model.LogisticRegression'
    :param data: 'numpy.ndarray'
                 The instances, for which counterfactuals will be (if available)
                 produced.
    :return: 'dict'
             The results categorized in three groups:
             - valid_cf: instances and their found counterfactuals
             - non_valid_cf: instances and their found counterfactuals, but the values
                             of the cfs are not valid
             - no_cf_found: instances, for which no counterfactuals were found.
    """

    result = {
        "result_name": result_name,
        "used_solver": solver,
        "valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "non_valid_cf": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "no_cf_found": {"x": [], "y": [], "x_cf": [], "y_cf": []},
        "not_solvable": 0
    }

    # Rounding values to integers for visualization purposes
    for i, instance in enumerate(data):
        try:
            print(f"{i / len(data) * 100 :.2f}%")
            x, y, x_cf, y_cf = compute_cf(classifier, instance, protected, solver, integer)
            x_cf_2 = rounding(x_cf, classifier.coef_[0], classifier.intercept_, y, 0)
            if x_cf_2 is not None:
                x_cf = x_cf_2
            if y == y_cf:
                result["no_cf_found"]["x"].append(list(x))
                result["no_cf_found"]["y"].append(y)
                result["no_cf_found"]["x_cf"].append(list(x_cf))
                result["no_cf_found"]["y_cf"].append(y_cf)
            elif not in_boundaries_2(x_cf, range(9)):
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

    return result


def get_data():
    """
    Reads the 'compas-scores-two-years.csv'-file, filters it with the same conditions
    that have been used in the propublica-analysis and provides train and test sets for
    the logistic regression.

    :return: 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray', 'numpy.ndarray'
    """

    # Not filtered data
    recidivsm_data = pd.read_csv("compas-scores-two-years.csv")

    # Function that filters as in Propublica analysis
    def filter_as_in_literature(df):
        newdf = df.filter(items=["age"
            , "c_charge_degree"
            , "race"
            , "score_text"
            , "sex"
            , "priors_count"
            , "days_b_screening_arrest"
            , "decile_score"
            , "is_recid"
            , "two_year_recid"
            , "c_jail_in"
            , "c_jail_out"])
        return newdf.query("days_b_screening_arrest <= 30"
                           ).query("days_b_screening_arrest >= -30"
                                   ).query("is_recid != -1"
                                           ).query("c_charge_degree != 0"
                                                   ).query('score_text != \'N/A\'')

    # The filtered data
    filtered_data = filter_as_in_literature(recidivsm_data)
    print(filtered_data["c_jail_in"])
    ###### FEATURE TRANSFORMATION #####
    def make_labels(df):
        labels = []
        for decile_score in np.array(df):
            if decile_score <= 4:
                labels.append(0)
            else:
                labels.append(1)
        return labels

    # The Labels: 0 if Low risk, 1 else
    response_vars = make_labels(filtered_data.filter(items=["decile_score"]))

    # Time in jail in hours
    jail_in_out = [round((datetime.strptime(jail_out, '%Y-%m-%d %H:%M:%S') - datetime.strptime(jail_in,
                                                                                         '%Y-%m-%d %H:%M:%S')).total_seconds() / 3600)
                   for jail_in, jail_out in zip(filtered_data["c_jail_in"], filtered_data["c_jail_out"])]

    def categorize(to_compare, options):
        for i in range(len(options)):
            if to_compare == options[i]:
                return i

    # Sex: 1 man, 0 women
    sex = [categorize(gender, ["Female", "Male"]) for gender in filtered_data["sex"]]

    ''' Race

    African-American: 0
    Asian: 1
    Caucasian: 2
    Hispanic: 3
    Native American: 4
    Other: 5

    '''
    race = [categorize(r, ["African-American", "Asian", "Caucasian", "Hispanic"
        , "Native American", "Other"]) for r in filtered_data["race"]]

    # charge degree: F = 0; M = 1;
    charge_degree = [categorize(charge, ["F", "M"]) for charge in filtered_data["c_charge_degree"]]

    # Final (not normalized) dataset
    pred_vars = filtered_data.filter(items=["age"
        , "priors_count"
        , "days_b_screening_arrest"
        , "is_recid"
        , "two_year_recid"])
    pred_vars["time_in_jail"] = jail_in_out
    pred_vars["sex"] = sex
    pred_vars["race"] = race
    pred_vars["charge_degree"] = charge_degree
    # print(pred_vars)

    X_train, X_test, y_train, y_test = train_test_split(pred_vars, response_vars
                                                        , test_size=0.33, random_state=1337)

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    y_train = np.array(y_train)
    y_test = np.array(y_test)

    return X_train, X_test, y_train, y_test


def main():
    # read the data from 'compas-scores-two-years'
    X_train, X_test, y_train, y_test = get_data()

    # train the logistic regression model
    log_reg = LogisticRegression(max_iter=1500).fit(X_train, y_train)

    # Export the actual vectors
    all_x = np.concatenate((X_train, X_test))
    all_y = np.concatenate((y_train, y_test))
    pd.DataFrame({"x": all_x.tolist(), "y": all_y.tolist()}).to_csv("x_values.csv"
                                                                    , index=False, sep=";"
                                                                    , quoting=csv.QUOTE_NONE)

    # List for collecting the results
    results = []

    # compute counterfactuals
    """
    result = process_data(log_reg, X_test, cp.CBC, "ILP", True)
    # As the numeric solution might be not a whole integer but rather something very
    # close to an integer, we round those values.
    result["valid_cf"]["x_cf"] = np.round(result["valid_cf"]["x_cf"]).tolist()
    result["non_valid_cf"]["x_cf"] = np.round(result["non_valid_cf"]["x_cf"]).tolist()
    results.append(result)

    # compute counterfactuals without protected attributes
    result_npa = process_data(log_reg, X_test, cp.CBC, "ILP - npa", True, False)
    result_npa["valid_cf"]["x_cf"] = np.round(result_npa["valid_cf"]["x_cf"]).tolist()
    result_npa["non_valid_cf"]["x_cf"] = np.round(result_npa["non_valid_cf"]["x_cf"]).tolist()
    results.append(result_npa)
    """
    # compute counterfactuals with relaxation
    result_wr = process_data(log_reg, X_test, cp.SCS, "ILP - wr")
    results.append(result_wr)

    # compute counterfactuals with relaxation but without protected attributes
    result_wr_npa = process_data(log_reg, X_test, cp.SCS, "ILP - wr - npa", False, False)
    results.append(result_wr_npa)

    # export results to csv files for further investigations
    #store_results(result, "valid_cf", "valid_cf")
    store_results(result_wr, "valid_cf", "valid_cf_wr")
    #store_results(result_npa, "valid_cf", "valid_cf_npa")
    store_results(result_wr_npa, "valid_cf", "valid_cf_wr_npa")

    #store_results(result, "non_valid_cf", "non_valid_cf")
    store_results(result_wr, "non_valid_cf", "non_valid_cf_wr")
    #store_results(result_npa, "non_valid_cf", "non_valid_cf_npa")
    store_results(result_wr_npa, "non_valid_cf", "non_valid_cf_wr_npa")

    # report the results
    print("Computation finished.")
    print("Accuracy score of the logistic regression:", log_reg.score(X_test, y_test))
    for res in results:
        print("\n")
        print("Experiment:", res["result_name"])
        print("Used solver:", res["used_solver"])
        print("Amount of valid counterfactuals", len(res["valid_cf"]["x_cf"]))
        print("Amount of counterfactuals, that exceed the boundaries:", len(res["non_valid_cf"]["x_cf"]))
        print("Amount of instances, for which no counterfactual have been found:", len(res["no_cf_found"]["x_cf"]))
        print("Amount of not optimal or infeasible problems:", res["not_solvable"])

    print("Parameter-values of the logistic regression:")
    print(f"w = {log_reg.coef_[0]}")
    print(f"b = {log_reg.intercept_}")

if __name__ == "__main__":
    main()
