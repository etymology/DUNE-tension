from scipy.stats import gaussian_kde
import numpy as np
from itertools import combinations

WIRE_DENSITY = 0.000152
MAX_PASSING_TENSION = 8
MIN_PHYSICAL_TENSION = 2
MAX_PHYSICAL_TENSION = 10


def calculate_kde_max(sample):
    """
    Calculate the maximum value of the kernel density estimation (KDE) for a given sample.

    Parameters:
    sample (array-like): An array of sample data.

    Returns:
    float: The maximum value of the KDE.
    """
    # Perform KDE on the sample

    try:
        kde_sample = gaussian_kde(sample)
    except np.linalg.LinAlgError as e:
        print(f"Error in KDE calculation: {e}")
        return sample.mean()  # Fallback to mean if KDE fails

    # Define a range of values for which to calculate the KDE
    x_range = np.linspace(min(sample), max(sample), 1000)
    kde_sample_values = kde_sample(x_range)

    # Find and return the maximum of the KDE
    max_kde_sample_value = x_range[np.argmax(kde_sample_values)]
    return max_kde_sample_value




def tension_lookup(length, frequency: float):
    tension = (2 * length * frequency) ** 2 * WIRE_DENSITY
    return tension

def tension_pass(tension, length):
    return tension > min(25.8 * length + 0.232, 4) and tension < MAX_PASSING_TENSION  #


def tension_plausible(tension):
    return tension < MAX_PHYSICAL_TENSION and tension > MIN_PHYSICAL_TENSION


def has_cluster_dict(data, key, n):
    """
    Checks if any subset of size n in the list of dictionaries forms a cluster
    based on the values of a specified key using the IQR method.

    Args:
        data (list): A list of dictionaries.
        key (str): The key to check values for clustering.
        n (int): The size of the subset to check.

    Returns:
        list: A subset of dictionaries that forms a cluster if one exists, otherwise an empty list.
    """
    if len(data) < n:
        return []

    for subset in combinations(data, n):
        values = [item[key] for item in subset]
        if np.std(values) < 0.1:
            return list(subset)


    return []

