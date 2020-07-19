import pandas as pd
import ast
import plotly.graph_objects as go
import numpy as np
from compute_cfs_one_hot_enc_2 import ATTRIBUTE_NAMES, ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION
from scipy.stats import chi2_contingency
from matplotlib import pyplot as plt
from matplotlib.colors import ListedColormap
import seaborn as sns


def bias_corrected_cramers_V(attribute1, attribute2):
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
    return cramers_corrected_v, min(r, c)


def compute_correlations(data, heatmap_name):
    # Compute the cramers-V-value for each attribute-pair
    cramers_v_table = pd.DataFrame(index=data.columns, columns=data.columns)

    for i, fst_column in enumerate(data):
        for j, snd_column in enumerate(data):
            # The chi2 value for two attributes is symmetric
            if i <= j:
                print("Currently computing Cramer's V for:", fst_column, snd_column)
                cramers_v_value, dof = bias_corrected_cramers_V(data[fst_column], data[snd_column])
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


def one_hot_to_ordinal(data, skip_cf=False):
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
    categories = ["x"]
    if not skip_cf:
        categories.append("x_cf")

    for category in categories:
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
    # Compute correlations
    data_set = read_data("x_values.csv", True)
    data_set = one_hot_to_ordinal(data_set, True)
    print(compute_correlations(data_set["x"], "heatmap.png"))

    """
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
    """


if __name__ == "__main__":
    main()
