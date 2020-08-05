import plotly.graph_objects as go
import seaborn as sns
from csv_parsing_writing import read_result
from matplotlib import pyplot as plt

from test_counterfactual import ATTRIBUTE_NAMES, ONE_HOT_VECTOR_START_INDEX
from cramers_v import compute_correlations
from grouping_and_counting import count_changes_for_groups, one_hot_to_label, COLUMN_NAMES


CHG_PER_GROUP_FILENAMES = ["changes_per_race.csv", "wr_changes_per_race.csv"
    , "changes_per_sex.csv", "wr_changes_per_sex.csv"]
AMT_PER_GROUP_FILENAMES = ["ppl_per_race.csv", "wr_ppl_per_race.csv"
    , "ppl_per_sex.csv", "wr_ppl_per_sex.csv"]


def plot_histogram(plot_title, data, group_names, x_axis, filename, store_plots=True):
    """
    Plots the histogram, while it groups the data set after people
    with a certain characteristic.

    :param plot_title: 'str'
                        The title of the histogram.
    :param data: 'list'
                  Two-dimensional list containing groups (fst. dimension)
                  and their y-values (snd. dimension) ordered by 'x-axis'.
    :param group_names: 'list'
                        A list containing the group names, that will be displayed
                        in the histogram.
    :param filename: 'str'
                     The filename of the figure.
    :param x_axis: 'list'
                   A list that contains the labels for the x-axis.
    :param store_plots: 'bool'
                        If one has no conda environment, one can't install the
                        plotly-orca-dependency which is necessary for the svg
                        graphics. But it is possible to show the graphics in the
                        browser. Therefore, set store_plots on 'false'.
    """
    if len(data) != len(group_names):
        raise Exception("Unequal amount of group names and groups.")

    fig = go.Figure()
    for i in range(len(data)):
        fig.add_trace(go.Bar(x=x_axis, y=data[i], name=group_names[i]))

    fig.update_layout(barmode='relative', title_text=plot_title)

    # Decide to store or simply show the plots
    if store_plots:
        fig.write_image(f"./{filename}.svg")
    else:
        fig.show()


def create_plots():
    """
    Function, that visualizes all the results. It initializes the
    computation for the correlations between the attributes given
    from the pre-processed data set. Then, it groups the data and
    plots figures for the groups.
    """
    # Ask, if pictures should be saved.
    dont_stop = True
    save_pictures = False
    while dont_stop:
        plotly_orca = input("Do you want to store the figures? (needs plotly-orca package) (y/n):")
        if plotly_orca == "y":
            save_pictures = True
            dont_stop = False
        elif plotly_orca == "n":
            dont_stop = False
        else:
            print("Write 'y' for 'yes' or 'n' for no.")

    # Plot Cramer's V values
    data_set = read_result("x_values.csv", True)
    correlations = compute_correlations(data_set)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.heatmap(correlations, annot=True
                , cmap=plt.cm.Reds
                , ax=ax
                , linewidths=0.5
                , linecolor='black')
    plt.title("Corrected Cramer's V - values")
    plt.tight_layout()
    plt.savefig("heatmap.svg", format='svg', dpi=1200)
    plt.show()

    # Read the experiment results
    valid_cf_npa = read_result("cf.csv")
    valid_cf_wr_npa = read_result("cf_wr.csv")

    # Convert One-Hot columns to label-encoding
    one_hot_to_label(valid_cf_npa)
    one_hot_to_label(valid_cf_wr_npa)

    # Count changes per groups
    changes_per_race, _ = count_changes_for_groups(valid_cf_npa, "race", True
                                                   , filename_changes=CHG_PER_GROUP_FILENAMES[0]
                                                   , filename_ppl=AMT_PER_GROUP_FILENAMES[0])
    wr_changes_per_race, _ = count_changes_for_groups(valid_cf_wr_npa, "race", True
                                                      , filename_changes=CHG_PER_GROUP_FILENAMES[1]
                                                      , filename_ppl=AMT_PER_GROUP_FILENAMES[1])
    changes_per_sex, _ = count_changes_for_groups(valid_cf_npa, "sex", True
                                                  , filename_changes=CHG_PER_GROUP_FILENAMES[2]
                                                  , filename_ppl=AMT_PER_GROUP_FILENAMES[2])
    wr_changes_per_sex, _ = count_changes_for_groups(valid_cf_wr_npa, "sex", True
                                                     , filename_changes=CHG_PER_GROUP_FILENAMES[3]
                                                     , filename_ppl=AMT_PER_GROUP_FILENAMES[3])

    # Compute the histograms (colored by race)
    race_names = list(map(lambda a: a[5:], ATTRIBUTE_NAMES[ONE_HOT_VECTOR_START_INDEX:]))
    plot_histogram("Integer Linear Programming", changes_per_race, race_names, COLUMN_NAMES, "cf", save_pictures)
    plot_histogram("ILP with relaxation", wr_changes_per_race, race_names, COLUMN_NAMES, "cf_wr", save_pictures)

    # Compute the histograms (colored by sex)
    plot_histogram("Integer Linear Programming", changes_per_sex, ["female", "male"], COLUMN_NAMES, "cf_red_blue", save_pictures)
    plot_histogram("ILP with relaxation", wr_changes_per_sex, ["female", "male"], COLUMN_NAMES, "cf_wr_red_blue", save_pictures)


if __name__ == "__main__":
    create_plots()
