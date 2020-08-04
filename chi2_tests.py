import pandas as pd
from scipy.stats import chi2_contingency

from visualization import CHG_PER_GROUP_FILENAMES, AMT_PER_GROUP_FILENAMES


def compute_chi2(table_changes, table_people, group, attribute):
    """
    Computes the contingency table based on the given groupings.
    Furthermore, computes the chi2-values based on the contingency
    table.

    :param table_changes: 'pandas.core.frame.DataFrame'
                          The table containing the changes per group
                          for each attribute.
    :param table_people: 'pandas.core.frame.DataFrame'
                          The table containing the amounts of people
                          per group.
    :param group: 'int'
                  The group, for which a chi2-value will be calculated.
    :param attribute: 'str'
                       The attribute, for which a chi2-value will be
                       calculated.
    """
    # Compute contingency table
    all_people = table_people.sum()[0]
    all_changes = table_changes[attribute].sum()
    people_without_group = all_people - table_people.loc[group]

    # People, who are not in the group but have a suggestion to change the attribute
    N_21 = all_changes - table_changes.loc[group, attribute]
    # People, who are not in the group and have no suggestion to change the attribute
    N_11 = (people_without_group - N_21)[0]
    # People, who are in the group and have a suggestion to change the attribute
    N_22 = table_changes.loc[group, attribute]
    # People, who are in the group and have no suggestion to change the attribute
    N_12 = (table_people.loc[group] - N_22)[0]

    contingency_table = pd.DataFrame([[N_11, N_12], [N_21, N_22]])
    print(f"Chi2-value for {attribute} {group} for the independence-test of the attribute {attribute}:"
          , chi2_contingency(contingency_table)[0])


def compute_chi2_for_groups():
    """
    Reads the groupings from 'visualization.py' and initializes
    computations for chi2-values.
    """
    changes_per_group = []
    # Import changes per group
    for filename in CHG_PER_GROUP_FILENAMES:
        changes_per_group.append(pd.read_csv(filename, sep=";"))

    # Import amount of people per group
    amt_per_group = []
    for filename in AMT_PER_GROUP_FILENAMES:
        amt_per_group.append(pd.read_csv(filename, sep=";"))

    # Calculate chi2-value
    i = 0
    for table_changes, table_people in zip(changes_per_group, amt_per_group):
        if i < 2:
            for group in range(len(table_people)):
                compute_chi2(table_changes, table_people, group, "race")
        else:
            for group in range(len(table_people)):
                compute_chi2(table_changes, table_people, group, "sex")
        i += 1
        print("\n")
