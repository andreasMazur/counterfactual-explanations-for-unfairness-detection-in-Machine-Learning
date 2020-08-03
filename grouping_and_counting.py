import numpy as np

from test_counterfactual import ONE_HOT_VECTOR_START_INDEX, ATTRIBUTE_NAMES
from csv_parsing_writing import store_amounts

COLUMN_NAMES = ["age", "priors_count", "days_b_screening_arrest"
    , "is_recid", "two_year_recid", "sex", "charge_degree"
    , "time_in_jail", "race"]


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


def count_changes_for_groups(data, grouping_attribute, save_as_csv=False, filename_changes="Changes.csv",
                             filename_ppl="People.csv"):
    """
    Groups the entries in 'data_label_encoding' by 'groupin_attribute' and counts the
    changes suggested by the counterfactuals for each group.

    :param data: 'dict'
                 A result-dictionary containing the original x-vectors, their
                 y-values as well as their counterfactuals and their y_cf values.
                 Usually just imported via 'read_data'-function.
    :param grouping_attribute: 'str'
                               The attribute after which the people are grouped.
    :param save_as_csv: 'bool'
                        Tells if the groupings shall be saved in csv-files.
    :param filename_changes: 'str'
                             The filename for the csv-file containing the changes per group.
    :param filename_ppl: 'str'
                         The filename for the csv-file containing the people per group.
    :return changes_per_attribute: 'list'
                                    A list with the changes per attribute
                                    for each group.
            amount_ppl_in_group: 'list'
                                 A list with the amount of people in each
                                 group.
    """
    # Remember old values for grouping_attribute
    remember = list(data["x"][grouping_attribute])

    # Compute delta vectors
    delta_vectors = data["x_cf"] - data["x"]
    delta_vectors[f"{grouping_attribute}_old"] = remember

    # Sort by attribute "to_color_old"
    amount_of_groups = len(delta_vectors.groupby(f"{grouping_attribute}_old").count().index)
    delta_vectors_by_to_color = [[] for _ in range(amount_of_groups)]
    for _, row in delta_vectors.iterrows():
        delta_vectors_by_to_color[int(row[f"{grouping_attribute}_old"])].append(list(row)[:len(row) - 1])

    amount_ppl_in_group = []
    for i in range(len(delta_vectors_by_to_color)):
        amount_ppl_in_group.append(len(delta_vectors_by_to_color[i]))

    # Count the changes
    changes_per_attribute = count_changes(delta_vectors_by_to_color)

    if save_as_csv:
        # Export the changes per group for further analysis
        store_amounts(changes_per_attribute, filename_changes, COLUMN_NAMES)
        store_amounts(amount_ppl_in_group, filename_ppl, ["people per group"])

    return changes_per_attribute, amount_ppl_in_group
