import pandas as pd
import ast
import plotly.graph_objects as go
import numpy as np
from compute_cfs import ATTRIBUTE_NAMES, ONE_HOT_VECTOR_START_INDEX
from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
import seaborn as sns


def bias_corrected_cramers_V(attribute1, attribute2):
    """
    Compute the bias corrected Cramér's V value for two
    categorical attributes after:

    Bergsma, Wicher. (2013). A bias-correction for Cramér's V and Tschuprow's T.
    Journal of the Korean Statistical Society. 42. 10.1016/j.jkss.2012.10.002.

    :param attribute1: 'pandas.core.series.Series'
                       A categorical attribute.
    :param attribute2: 'pandas.core.series.Series'
                       A categorical attribute.
    :return: 'numpy.float64'
             The corrected Cramér's V value for 'attribute1' and 'attribute2'
    """

    if len(attribute1) != len(attribute2):
        raise Exception("You have more values in one column than in the other.")
    rows = pd.Categorical(attribute1)
    columns = pd.Categorical(attribute2)
    # Create the contingency table
    contingency_table = pd.crosstab(rows, columns)
    # Calculate the chi2-value
    chi2_value = chi2_contingency(contingency_table)[0]
    # Compute the corrected cramer's V-value
    n = len(attribute1)
    r = len(contingency_table)
    r_tilde = r - 1/(n - 1) * (r - 1)**2
    c = len(contingency_table.columns)
    c_tilde = c - 1/(n - 1) * (c - 1)**2
    Phi = (chi2_value / n) - 1/(n - 1) * (r - 1) * (c - 1)
    Phi_plus = max(0, Phi)
    cramers_corrected_v = np.sqrt(Phi_plus / min(r_tilde - 1, c_tilde - 1))
    return cramers_corrected_v


def compute_correlations(data, heatmap_name):
    """
    Creates heatmap with Cramér's V values for every attribute-pair.

    :param data: 'pandas.core.frame.DataFrame'
                  The original x-vectors for the counterfactuals.
    :param heatmap_name: 'str'
                         The filename for the heatmap.
    """

    # Compute the cramers-V-value for each attribute-pair
    cramers_v_table = pd.DataFrame(index=data.columns, columns=data.columns)

    for i, fst_column in enumerate(data):
        for j, snd_column in enumerate(data):
            # The chi2 value for two attributes is symmetric
            if i <= j:
                print("Currently computing Cramer's V for:", fst_column, snd_column)
                cramers_v_value = bias_corrected_cramers_V(data[fst_column], data[snd_column])
                # Add Cramers'V value
                cramers_v_table.at[snd_column, fst_column] = cramers_v_value

    cramers_v_table = cramers_v_table.apply(pd.to_numeric)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cramers_v_table, annot=True
                , cmap=plt.cm.Reds
                , ax=ax
                , linewidths=0.5
                , linecolor='black')
    plt.title("Corrected Cramer's V - values")
    plt.tight_layout()
    plt.savefig(heatmap_name)
    plt.show()


def count_changes(delta_vectors):
    """
    Count the non-zero entries for every delta vector in every
    dimension.

    :param delta_vectors: 'list'
                          3 dimensional list:
                          - First dimension: The groups
                          - Second dimension: The people
                          - Third dimension: The attributes of the people
                          Contains the difference between the counterfactuals
                          and their original vectors.
    :return: 'numpy.ndarray'
             2D array:
             - First dimension: The groups
             - Second dimension: The amounts of changes per attribute
    """
    # Count non-null values for each attribute
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


def plot_histogram(data, plot_title, to_color, trace_names):
    """
    Plots the histogram, while it groups the data set after people
    with a certain characteristic.

    :param data: 'dict'
                 A result-dictionary containing the original x-vectors, their
                 y-values as well as their counterfactuals and their y_cf values.
                 Usually just imported via 'read_data'-function.
    :param plot_title: 'str'
                        The title of the histogram.
    :param to_color: 'str'
                      The attribute after which the people are grouped.
    :param trace_names: 'list'
                        A list containing the group names, that will be displayed
                        in the histogram.
    """
    # Remember old values for to_color
    remember = list(data["x"][to_color])

    # Compute delta vectors
    delta_vectors = data["x_cf"] - data["x"]
    delta_vectors[f"{to_color}_old"] = remember

    # Sort by attribute "to_color_old"
    delta_vectors_by_to_color = [[] for _ in range(len(trace_names))]
    for _, row in delta_vectors.iterrows():
        delta_vectors_by_to_color[int(row[f"{to_color}_old"])].append(list(row)[:len(row) - 1])

    print(f"Amount of people in each group: (sorted by {to_color})")
    for i in range(len(delta_vectors_by_to_color)):
        print(i, ":", len(delta_vectors_by_to_color[i]))
    print("\n")

    # Count the changes
    changes_per_attribute = count_changes(delta_vectors_by_to_color)

    # Plot the histogram
    x_axis = ["age", "priors_count", "days_b_screening_arrest"
              , "is_recid", "two_year_recid", "sex", "charge_degree"
              , "time_in_jail", "race"]

    fig = go.Figure()
    for i in range(len(trace_names)):
        fig.add_trace(go.Bar(x=x_axis, y=changes_per_attribute[i], name=trace_names[i]))

    fig.update_layout(barmode='relative', title_text=plot_title)
    fig.show()


def one_hot_to_label(data, skip_cf=False):
    """
    Re-encodes the one-hot vector to label-encoding.

    Assignment:
    0 <- African American
    1 <- Asian
    2 <- Caucasian
    3 <- Hispanic
    4 <- Native American
    5 <- Other

    :param data: 'dict'
                 A result-dictionary containing the original x-vectors, their
                 y-values as well as their counterfactuals and their y_cf values.
                 Usually just imported via 'read_data'-function.
    :param skip_cf: 'bool'
                    The bool tells, if the result-dict contains counterfactuals.
                    If it does, just let the value be 'False'. Otherwise, tell
                    to skip the counterfactuals by stating it to be 'True'.
    :return: The result-dictionary but with label-encoded races instead of
             one-hot vectors.
    """
    categories = ["x"]
    if not skip_cf:
        categories.append("x_cf")

    for category in categories:
        # Compute labels
        new_encoding = []
        for index, row in data[category].iterrows():
            # Search for non-zero entry
            for i, race in enumerate(list(row[ONE_HOT_VECTOR_START_INDEX:])):
                if race == 1:
                    new_encoding.append(i)

        # Exchange one-hot columns with ordinal encoding
        data[category].drop(columns=ATTRIBUTE_NAMES[ONE_HOT_VECTOR_START_INDEX:], inplace=True)
        data[category]["race"] = new_encoding

    return data


def read_data(file_name, skip_cf=False):
    """
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


def main():
    # Compute correlations
    data_set = read_data("x_values.csv", True)
    data_set = one_hot_to_label(data_set, True)
    compute_correlations(data_set["x"], "heatmap.png")

    # Read the data and convert one-hot encoding to label encoding
    valid_cf_npa = read_data("cf.csv")
    valid_cf_npa = one_hot_to_label(valid_cf_npa)

    valid_cf_wr_npa = read_data("cf_wr.csv")
    valid_cf_wr_npa = one_hot_to_label(valid_cf_wr_npa)

    # Compute the histograms (colored by race)
    trace_names = list(map(lambda a: a[5:], ATTRIBUTE_NAMES[ONE_HOT_VECTOR_START_INDEX:]))
    plot_histogram(valid_cf_npa, "Integer Linear Programming", "race", trace_names)
    plot_histogram(valid_cf_wr_npa, "ILP with relaxation", "race", trace_names)

    # Compute the histograms (colored by sex)
    plot_histogram(valid_cf_npa, "Integer Linear Programming", "sex", ["female", "male"])
    plot_histogram(valid_cf_wr_npa, "ILP with relaxation", "sex", ["female", "male"])


if __name__ == "__main__":
    main()
