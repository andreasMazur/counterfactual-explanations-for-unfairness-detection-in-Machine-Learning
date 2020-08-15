import pandas as pd
import numpy as np
import sys
from scipy.stats import chi2_contingency

from grouping_and_counting import one_hot_to_label


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
    r_tilde = r - 1 / (n - 1) * (r - 1) ** 2
    c = len(contingency_table.columns)
    c_tilde = c - 1 / (n - 1) * (c - 1) ** 2
    Phi = (chi2_value / n) - 1 / (n - 1) * (r - 1) * (c - 1)
    Phi_plus = max(0, Phi)
    if r_tilde - 1 > 0 and c_tilde - 1 > 0:
        cramers_corrected_v = np.sqrt(Phi_plus / min(r_tilde - 1, c_tilde - 1))
        return cramers_corrected_v
    else:
        return 0


def compute_correlations(data):
    """
    Creates heatmap with Cramér's V values for every attribute-pair.

    :param data: 'pandas.core.frame.DataFrame'
                  The original x-vectors for the counterfactuals.
    """
    # Convert one-hot-encoding to label-encoding
    data = one_hot_to_label(data, True)

    # Get x-values from result-dictionary
    data = data["x"]

    # Compute the cramers-V-value for each attribute-pair
    cramers_v_table = pd.DataFrame(index=data.columns, columns=data.columns)

    for i, fst_column in enumerate(data):
        for j, snd_column in enumerate(data):
            # The chi2 value for two attributes is symmetric
            if i <= j:
                sys.stdout.write(
                    f"\rCurrently computing Cramer's V for: {fst_column} and {snd_column}")
                sys.stdout.flush()
                cramers_v_value = bias_corrected_cramers_V(data[fst_column], data[snd_column])
                # Add Cramers'V value
                cramers_v_table.at[snd_column, fst_column] = cramers_v_value

    cramers_v_table = cramers_v_table.apply(pd.to_numeric)
    return cramers_v_table
