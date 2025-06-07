from itertools import combinations
from typing import Sequence, Any

import numpy as np
from scipy.stats import gaussian_kde

WIRE_DENSITY = 0.000152
MAX_PASSING_TENSION = 8
MIN_PHYSICAL_TENSION = 2
MAX_PHYSICAL_TENSION = 10


def calculate_kde_max(sample: Sequence[float]) -> float:
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
        return sample[0]  # Fallback to 0 if KDE fails

    # Define a range of values for which to calculate the KDE
    x_range = np.linspace(min(sample), max(sample), 1000)
    kde_sample_values = kde_sample(x_range)

    # Find and return the maximum of the KDE
    max_kde_sample_value = x_range[np.argmax(kde_sample_values)]
    return max_kde_sample_value


def tension_lookup(length: float, frequency: float) -> float:
    tension = (2 * length * frequency) ** 2 * WIRE_DENSITY
    return tension


def tension_pass(tension: float, length: float) -> bool:
    return tension > min(25.8 * length + 0.232, 4) and tension < MAX_PASSING_TENSION  #


def tension_plausible(tension: float) -> bool:
    return tension < MAX_PHYSICAL_TENSION and tension > MIN_PHYSICAL_TENSION


def has_cluster_dict(data: Sequence[Any], key: str, n: int) -> list[Any]:
    """Return a subset of ``data`` of size ``n`` forming a cluster.

    ``data`` may be a list of dictionaries or objects with attributes
    referenced by ``key``. The function checks every combination of ``n``
    items and returns the first subset whose ``key`` values have a small
    standard deviation (<0.1).
    """

    if len(data) < n:
        return []

    for subset in combinations(data, n):
        values = [
            item[key] if isinstance(item, dict) else getattr(item, key)
            for item in subset
        ]
        if np.std(values) < 0.1:
            return list(subset)

    return []
