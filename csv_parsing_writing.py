import pandas as pd
import csv
import ast
from test_counterfactual import ATTRIBUTE_NAMES
from datetime import datetime as dt
from sklearn import preprocessing

# The csv-file containing the compas-data
COMPAS_FILE = "compas-scores-two-years.csv"


def store_amounts(amounts, filename, column_name):
    pd.DataFrame(amounts, columns=column_name).to_csv(filename
                                                      , index=False
                                                      , sep=";"
                                                      , quoting=csv.QUOTE_NONE)


def store_result(result, sub_dict_to_store, file_name):
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


def read_result(file_name, skip_cf=False):
    """
    Reads a result that was stored with the 'store_result'-function.

    :param file_name: 'str'
                      The file from which the data shall be read.
                      Has to be an output file of the 'compute_cfs'
                      program.
    :param skip_cf: 'bool'
                    The bool tells, if the result-dict contains counterfactuals.
                    If it does, just let the value be 'False'. Otherwise, tell
                    to skip the counterfactuals by stating it to be 'True'.
    :return: 'dict'
             A dictionary containing a result of the 'compute_cfs' program.
    """

    data = pd.read_csv(file_name, sep=";")
    if skip_cf:
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=ATTRIBUTE_NAMES),
                     "y": pd.DataFrame(data["y"])}
    else:
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=ATTRIBUTE_NAMES),
                     "y": pd.DataFrame(data["y"]),
                     "x_cf": pd.DataFrame([ast.literal_eval(row) for row in data["x_cf"]], columns=ATTRIBUTE_NAMES),
                     "y_cf": pd.DataFrame(data["y_cf"])}
    return data_dict


def read_compas_data():
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
    recidivism_data_raw = pd.read_csv(COMPAS_FILE)

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
