import pandas as pd
from visualization import AMT_PER_GROUP_FILENAMES, CHG_PER_GROUP_FILENAMES
from scipy.stats import chi2_contingency


def compute_chi2(table_changes, table_people, group, attribute):
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


def main():
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

if __name__ == "__main__":
    main()
