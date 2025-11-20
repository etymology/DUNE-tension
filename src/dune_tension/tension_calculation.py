from typing import Optional, Sequence

import numpy as np
from scipy.stats import gaussian_kde

#   average wire density    0.1540460069 g/m
#   minimum wire density    0.149 (g/m)
#   maximum wire density    0.15925 (g/m)
#   std wire density        0.006408457943 (g/m)

# the wire density is consistent within batches but has a small variation between batches

WIRE_DENSITY = 0.0001540460069  # in kg/m
MAX_PASSING_TENSION = 8.5  # Note the minimum depends on the wire length
MIN_PHYSICAL_TENSION = 4
MAX_PHYSICAL_TENSION = (
    8  # considering higher than 10 not possible because of winder tension control
)
NOMINAL_TENSION = 6.5


def calculate_kde_max(sample: Sequence[float]) -> float:
    """
    Calculate the maximum value of the kernel density estimation (KDE) for a given sample.

    Parameters:
    sample (array-like): An array of sample data.

    Returns:
    float: The maximum value of the KDE.
    """
    # Perform KDE on the sample
    if len(sample) == 1:
        return sample[0]

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

def wire_equation(length: Optional[float] = None, frequency: Optional[float] = None, tension: Optional[float] = NOMINAL_TENSION) -> dict[str, float]:
    """Calculate wire properties given any two of length, frequency, and tension.

    Returns a dictionary with keys 'length', 'frequency', and 'tension'.
    """
    if length is not None and frequency is not None:
        tension = (2 * length * frequency) ** 2 * WIRE_DENSITY
    elif length is not None and tension is not None:
        frequency = (tension / (WIRE_DENSITY * (2 * length) ** 2)) ** 0.5
    elif frequency is not None and tension is not None:
        length = (tension / (WIRE_DENSITY * (2 * frequency) ** 2)) ** 0.5
    else:
        raise ValueError("At least two of length, frequency, or tension must be provided.")

    return {"length": length, "frequency": frequency, "tension": tension}


def tension_pass(tension: float, length: float) -> bool:
    return tension > min(25.8 * length + 0.232, 4) and tension < MAX_PASSING_TENSION  #


def tension_plausible(tension: float) -> bool:
    return tension < MAX_PHYSICAL_TENSION and tension > MIN_PHYSICAL_TENSION
