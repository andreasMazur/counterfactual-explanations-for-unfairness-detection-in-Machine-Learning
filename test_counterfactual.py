import numpy as np

# The structure of a counterfactual
VECTOR_INDEX = {"age": 0,
                "priors_count": 1,
                "days_b_screening_arrest": 2,
                "is_recid": 3,
                "two_year_recid": 4,
                "sex": 5,
                "charge_degree": 6,
                "time_in_jail": 7,
                "race_African-American": 8,
                "race_Asian": 9,
                "race_Caucasian": 10,
                "race_Hispanic": 11,
                "race_Native American": 12,
                "race_Other": 13}
VECTOR_DIMENSION = len(VECTOR_INDEX)
ATTRIBUTE_NAMES = list(VECTOR_INDEX.keys())
ONE_HOT_VECTOR_START_INDEX = VECTOR_INDEX["race_African-American"]
LOWER_BOUNDS = [0, 0, -np.inf, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
UPPER_BOUNDS = [np.inf, np.inf, np.inf, 1, 1, 1, 1, np.inf, 1, 1, 1, 1, 1, 1]


def one_hot_valid(vec):
    """
    Checks if a vector contains a valid one-hot encoding.

    :param vec: 'numpy.ndarray'
                The vector, for which the included one-hot vector
                shall be checked.
    :return: 'bool'
             The test result.
    """
    # Check whether each entry is an integer
    for i in range(ONE_HOT_VECTOR_START_INDEX, VECTOR_DIMENSION):
        if not vec[i].is_integer():
            return False

    # Check whether we only have one 1 in the one-hot encoding
    if sum(vec[ONE_HOT_VECTOR_START_INDEX:]) != 1.0:
        return False

    # Return true, if no violation happened.
    return True


def in_boundaries(vec, index):
    """
    Checks if the values in a vector are within their bounds.

    :param vec: 'numpy.ndarray'
                The vector, for which the values shall be checked.
    :param index: 'range'
                  The indices for which the boundaries shall be checked
    :return: 'bool'
             The test result.
    """
    in_range = True
    for i in index:
        in_range = in_range and LOWER_BOUNDS[i] <= vec[i] <= UPPER_BOUNDS[i]
        if not in_range:
            break
    return in_range


def is_valid(vec, y, y_cf):
    """
    Checks if a vectors is plausible or not.

    :param vec: 'numpy.ndarray'
                 The counterfactual, that shall be tested
    :param y: 'numpy.int64'
               The original class of the original vector from vec
    :param y_cf: 'numpy.int64'
                 The class of the counterfactual 'vec'
    :return:
    """
    return in_boundaries(vec, range(VECTOR_DIMENSION)) and one_hot_valid(vec) and y != y_cf
