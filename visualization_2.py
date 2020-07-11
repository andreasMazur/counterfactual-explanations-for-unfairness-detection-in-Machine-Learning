import pandas as pd
import numpy as np
from compute_cfs_one_hot_enc_2 import ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION


def compute_delta_vectors(data):
    delta_vectors = data["x_cf"] - data["x"]

    # Substitute the one-hot vector with one column,
    # that says whether a change happened.
    deltas = delta_vectors.drop(index=["race_African-American", "race_Asian", "race_Caucasian",
                                       "race_Hispanic", "race_Native American", "race_Other"])
    deltas["race"] = np.zeros(len(delta_vectors))

    # Read changes in the one-jot vector
    for i, vec in enumerate(delta_vectors):
        change = False
        for j in range(ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION):
            if vec[j] != 0:
                change = True
                break
        if change:
            deltas.loc[i, "race"] = 1

    return deltas


def read_data(file_name, skip_cf=False):
    """
    :param skip_cf:
    :param file_name: 'str'
                      The file from which the data shall be read.
                      Has to be an output file of the cf-production
                      program.
    :param found_cfs: 'bool'
                      Set to 'False' if you only want the initial
                      feature vectors without their counterfactuals.
                      May be usefull for vectors, for which no cf
                      was found.
    :return: 'dict'
             A dictionary
    """

    data = pd.read_csv(file_name, sep=";")
    if skip_cf:
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=categories),
                     "y": pd.DataFrame(data["y"])}
    else:
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=categories),
                     "y": pd.DataFrame(data["y"]),
                     "x_cf": pd.DataFrame([ast.literal_eval(row) for row in data["x_cf"]], columns=categories),
                     "y_cf": pd.DataFrame(data["y_cf"])}
    return data_dict


def main():
    compute_delta_vectors()

if __name__ == "__main__":
    main()