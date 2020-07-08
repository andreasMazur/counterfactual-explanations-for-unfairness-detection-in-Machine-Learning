import plotly.graph_objects as go
import pandas as pd
import ast
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency
from compute_cfs_new import VECTOR_INDEX

# The names in correct order for the columns of a dataset
categories = list(VECTOR_INDEX.keys())


def detect_amount_of_changes(xs, x_cfs):
    """
    Computes and plots the suggested changes per attribute and group.
    Subsequently the function plots the results in a histogram.

    :param xs: 'pandas.core.frame.DataFrame'
           The actual instances from the database.
    :param x_cfs: 'pandas.core.frame.DataFrame'
           The counterfactual explanations for the actual instances.
    :return: y_values: 'pandas.core.frame.DataFrame'
             The amount of changes per attribute and race.
    """
    # Generate sub-datasets for each race (for coloring the histogram)
    sub_datasets_xs = []
    sub_datasets_x_cfs = []
    try:
        groups = list(xs.groupby("race").count().index)
        for i in groups:
            truth_values = xs["race"] == i
            sub_datasets_xs.append(xs[truth_values])
            # As the race-value might change in the counterfactual
            # explanation, we take the same truth-values as for xs.
            sub_datasets_x_cfs.append(x_cfs[truth_values])
    except IndexError:
        raise Exception("Check if xs and x_cfs have the correct column descriptions.")

    # Count how many entries are >0 for each column per group
    y_values = []
    for (xs_group, x_cfs_group) in zip(sub_datasets_xs, sub_datasets_x_cfs):
        # Compute the suggested changes for that group
        delta = x_cfs_group - xs_group
        amounts = []
        for column in delta:
            amounts.append(delta[delta[column] > 0][column].count())
        y_values.append(amounts)

    # Plot the results in a bar chart
    group_names = ["African-American", "Asian", "Caucasian", "Hispanic", "Native American", "Other"]
    bar_chart = go.Figure()
    for i, y_value in enumerate(y_values):
        # Add sum description to the last group for each attribute
        if y_value == y_values[-1]:
            bar_chart.add_trace(
                go.Bar(x=categories, y=y_value, name=group_names[i], text=sum(np.array(y_values)),
                       textposition="auto"))
        else:
            bar_chart.add_trace(
                go.Bar(x=categories, y=y_value, name=group_names[i]))

    bar_chart.update_layout(barmode='relative', title_text='Amount of suggested changes per attribute and race')
    bar_chart.show()

    # return pd.DataFrame(y_values, index=group_names, columns=categories)


def plot_heatmap_for_cramers_v(data, heatmap_name):
    """
    Computes the Cramer's V-values for each attribute
    pair and plots a corresponding heatmap.

    :param data: 'pandas.core.frame.DataFrame'
           Usually the actual vectors x, for which the attributes dependencies
           should be computed.
    :return: chi_2_values: 'pandas.core.frame.DataFrame'
             A dataframe with the Cramer's V-values for each
             attribute-pair.
    """
    cramers_v_values = cramers_v(data)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(cramers_v_values, annot=True, cmap=plt.cm.Reds, ax=ax)
    plt.title("Cramer's V - values")
    plt.tight_layout()
    plt.savefig(heatmap_name)
    plt.show()
    return cramers_v_values


def cramers_v(data):
    """
    :param data: 'pandas.core.frame.DataFrame'
           The data for which the chi-2-values should be computed.

    A function to compute the cramers-V-value for each attribute-pair
    in order to spot relatedness among the features.

    :return cramers_v_table: 'pandas.core.frame.DataFrame'
            A table with all chi-2-values for each attribute-pair.
            For example, it can be used to plot heatmaps.
    """

    # Compute the cramers-V-value for each attribute-pair
    cramers_v_table = pd.DataFrame(index=data.columns, columns=data.columns)

    # First, compute the chi-2 statistic
    for i, fst_column in enumerate(data):
        for j, snd_column in enumerate(data):

            # We just need the upper triangular of the cross-product matrix because chi-2 is symmetric.
            if i <= j:
                print("Currently computing Cramer's V for:", fst_column, snd_column)

                # Compute all possible value-pairs (cross product)
                rows_indices = list(data.groupby(fst_column).count().index)
                column_indices = list(data.groupby(snd_column).count().index)
                cross_product = [(x, y) for x in rows_indices for y in column_indices]
                cross_product = np.array(cross_product).reshape(len(rows_indices), len(column_indices), 2)

                # Count how often each value-pair has occurred and put it into the contingency table.
                contingency_table = []
                for row in cross_product:
                    contingency_table_row = []
                    for [attr1_value, attr2_value] in row:
                        cell = data[data[fst_column] == attr1_value]
                        cell_value = cell[cell[snd_column] == attr2_value].shape[0]
                        contingency_table_row.append(cell_value)
                    contingency_table.append(contingency_table_row)

                # Compute the chi-2-value
                chi_2_value = chi2_contingency(contingency_table)[0]

                # Compute Cramer's v
                try:
                    n = data.shape[0]
                    c = len(column_indices)
                    r = len(rows_indices)
                    cramers_v_value = np.sqrt(chi_2_value / (n * min(c - 1, r - 1)))

                    # Add it to the chi-2-table
                    cramers_v_table.at[fst_column, snd_column] = cramers_v_value
                    cramers_v_table.at[snd_column, fst_column] = cramers_v_value

                except ZeroDivisionError:
                    print("Contingency table for", fst_column, "and"
                          , snd_column, "contains structural zeros.")
                    cramers_v_table.at[fst_column, snd_column] = 0
                    cramers_v_table.at[snd_column, fst_column] = 0
                    cramers_v_table.at[fst_column, fst_column] = 1

    return cramers_v_table.apply(pd.to_numeric)


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
    # Retrieve data
    # x_values = read_data("x_values.csv", True)

    # ILP
    valid_cfs = read_data("valid_cf.csv")
    valid_cfs_npa = read_data("valid_cf_npa.csv")

    # ILP with relaxation
    valid_cfs_wr = read_data("valid_cf_wr.csv")
    valid_cfs_wr_npa = read_data("valid_cf_wr_npa.csv")

    # Compute Cramer's V
    # plot_heatmap_for_cramers_v(x_values["x"], "heatmap.png")

    # Detect the amount of suggested changes per attribute
    # ILP
    detect_amount_of_changes(valid_cfs["x"], valid_cfs["x_cf"])
    detect_amount_of_changes(valid_cfs_npa["x"], valid_cfs_npa["x_cf"])

    # ILP with relaxation
    detect_amount_of_changes(valid_cfs_wr["x"], valid_cfs_wr["x_cf"])
    detect_amount_of_changes(valid_cfs_wr_npa["x"], valid_cfs_wr_npa["x_cf"])

if __name__ == "__main__":
    main()
