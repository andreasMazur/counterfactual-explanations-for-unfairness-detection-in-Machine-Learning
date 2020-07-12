import pandas as pd
import ast
import plotly.graph_objects as go
import numpy as np
from compute_cfs_one_hot_enc_2 import ATTRIBUTE_NAMES, ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION


def count_changes(delta_vectors):
    # Count non-null values for each attribute
    # TODO: Adjust 'delta_vectors[0][0]'
    changes_per_attribute = np.zeros((len(delta_vectors), len(delta_vectors[0][0])))
    # per race
    for i in range(len(delta_vectors)):
        # per person
        for j in range(len(delta_vectors[i])):
            # per attribute
            for k in range(len(delta_vectors[0][0])):
                if delta_vectors[i][j][k] != 0:
                    changes_per_attribute[i][k] += 1
    return changes_per_attribute.tolist()


def plot_histogram(delta_vectors, plot_title):
    # Sort by attribute "race_old"
    delta_vectors_by_race = [[] for _ in range(VECTOR_DIMENSION - ONE_HOT_VECTOR_START_INDEX)]
    for _, row in delta_vectors.iterrows():
        delta_vectors_by_race[int(row["race_old"])].append(list(row)[:len(row) - 1])

    # Count the changes
    changes_per_attribute = count_changes(delta_vectors_by_race)

    # Plot the histogram
    x_axis = ATTRIBUTE_NAMES[:ONE_HOT_VECTOR_START_INDEX]
    x_axis.append("race")
    trace_names = list(map(lambda a: a[5:], ATTRIBUTE_NAMES[ONE_HOT_VECTOR_START_INDEX:]))

    fig = go.Figure()
    for i in range(len(changes_per_attribute)):
        fig.add_trace(go.Bar(x=x_axis, y=changes_per_attribute[i], name=trace_names[i]))

    fig.update_layout(barmode='relative', title_text=plot_title)
    fig.show()


def one_hot_to_ordinal(data):
    """
    Assignment:
    0 <- African American
    1 <- Asian
    2 <- Caucasian
    3 <- Hispanic
    4 <- Native American
    5 <- Other

    :param data:
    :return:
    """
    for category in ["x", "x_cf"]:
        # Compute labels
        new_encoding = []
        for _, row in data[category].iterrows():
            # Search for non-zero entry
            for i, race in enumerate(list(row[ONE_HOT_VECTOR_START_INDEX:])):
                if race == 1:
                    new_encoding.append(i)

        # Exchange one-hot columns with ordinal encoding
        data[category].drop(columns=ATTRIBUTE_NAMES[ONE_HOT_VECTOR_START_INDEX:], inplace=True)
        data[category]["race"] = new_encoding

    return data


def compute_delta_vectors(data):
    """
    Computing the delta-vectors and keeping the information
    about the race for further analysis.

    :param data:
    :return:
    """
    race = list(data["x"]["race"])
    delta_vectors = data["x_cf"] - data["x"]
    delta_vectors["race_old"] = race
    return delta_vectors


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
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=ATTRIBUTE_NAMES),
                     "y": pd.DataFrame(data["y"])}
    else:
        data_dict = {"x": pd.DataFrame([ast.literal_eval(row) for row in data["x"]], columns=ATTRIBUTE_NAMES),
                     "y": pd.DataFrame(data["y"]),
                     "x_cf": pd.DataFrame([ast.literal_eval(row) for row in data["x_cf"]], columns=ATTRIBUTE_NAMES),
                     "y_cf": pd.DataFrame(data["y_cf"])}
    return data_dict


def main():
    valid_cf = read_data("valid_cf.csv")
    valid_cf = one_hot_to_ordinal(valid_cf)
    plot_histogram(compute_delta_vectors(valid_cf),
                   "ILP - Suggested changes per attribute. Protecting sensitive information.")

    valid_cf_npa = read_data("valid_cf_npa.csv")
    valid_cf_npa = one_hot_to_ordinal(valid_cf_npa)
    plot_histogram(compute_delta_vectors(valid_cf_npa),
                   "ILP - Suggested changes per attribute. Not protecting sensitive information.")

    valid_cf_wr = read_data("valid_cf_wr.csv")
    valid_cf_wr = one_hot_to_ordinal(valid_cf_wr)
    plot_histogram(compute_delta_vectors(valid_cf_wr),
                   "ILP with relaxation. Suggested changes per attribute. Protecting sensitive information.")

    valid_cf_wr_npa = read_data("valid_cf_wr_npa.csv")
    valid_cf_wr_npa = one_hot_to_ordinal(valid_cf_wr_npa)
    plot_histogram(compute_delta_vectors(valid_cf_wr_npa),
                   "ILP with relaxation. Suggested changes per attribute. Not protecting sensitive information.")


if __name__ == "__main__":
    main()
