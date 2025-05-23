from datetime import datetime
import os
import csv
import random
import pandas as pd
import numpy as np
from scipy.stats import gaussian_kde
from itertools import combinations
from scipy.signal import butter, filtfilt


G_LENGTH = 1.285
X_LENGTH = 1.273
WIRE_DENSITY = 0.000152
MAX_TENSION = 8
COMB_SPACING = 1190
Y_MIN = 200
Y_MAX = 2460
X_MIN = 1100
X_MAX = 7000
COMB_TOLERANCE = 300

comb_positions = [X_MIN -COMB_TOLERANCE+100, 2230, 3420, 4590, 5770, X_MAX+COMB_TOLERANCE-100]
MIN_PHYSICAL_TENSION = 2
MAX_PHYSICAL_TENSION = 10


# replace with real values for the comb positions

def zone_lookup(
    x,
):
    # Loop through the list to find the first value greater than x
    for i, pos in enumerate(comb_positions):
        if pos > x:
            return i
    # If no value greater than x, return None
    return None


# Test the function with an example input
zone_lookup(
    3500
)  # Expected to return the index 3 (since 4590 is the first value greater than 3500)


def zone_x_target(zone: int):
    return [1635, 2825, 4015, 5185, 6365][zone - 1]


def distance_to_zone_middle(x):
    ### returns the signed distance to the middle of the zone
    return abs(x - zone_x_target(zone_lookup(x)))


# def y_in_bounds(y: float):
#     return y > Y_MIN and y < Y_MAX


def next_wire_target(wire_x, wire_y, dx, dy):
    print(f"wire_x, wire_y: {wire_x}, {wire_y}")
 
    print(f"dx, dy: {dx}, {dy}")

    # Calculate the two possible positions
    positions = []

    for i in range(0, 400):
        positions.append((wire_x - i * dx, wire_y + (i + 1) * dy))
        positions.append((wire_x + (i + 1) * dx, wire_y - i * dy))

    valid_positions = []

    for position in positions:
        if is_in_bounds(position[0], position[1]):
            valid_positions.append(position)
    # Choose the position with the least y value
    if valid_positions:
        return min(valid_positions, key=lambda pos: abs(pos[1] - 1350))
    else:
        return wire_x + dx, wire_y


def not_close_to_comb(x, tolerance=COMB_TOLERANCE):
    # Check if x is within +/- 100 of any number in comb_positions
    for pos in comb_positions:
        if abs(pos - x) <= tolerance:
            return False
    return True


def is_in_bounds(x, y):
    return (X_MIN < x < X_MAX) and (Y_MIN < y < Y_MAX) and not_close_to_comb(x)


def length_lookup(layer: str, wire_number: int, zone: int, taped=False):
    file_path = f"wire_lengths/{layer}_LUT.csv"

    if layer not in ["U", "V", "X", "G"]:
        raise ValueError("Invalid layer. Must be 'U', 'V', 'X', or 'G'")
    if layer == "G":
        return G_LENGTH
    if layer == "X":
        return X_LENGTH

    # Load the specified layer spreadsheet
    try:
        spreadsheet = pd.read_csv(file_path, index_col=0)
    except FileNotFoundError:
        raise FileNotFoundError(f"File {file_path} not found")

    if wire_number < 1 or wire_number > 1151:
        raise ValueError("Wire number must be between 1 and 1151")
    if zone < 1 or zone > 5:
        raise ValueError("Zone must be between 1 and 5")

    try:
        value = spreadsheet.at[wire_number, str(zone)]
        if taped:
            return (value - 16) / 1000
        return value / 1000
    except KeyError:
        return None


def tension_lookup(length, frequency: float):
    tension = (2 * length * frequency) ** 2 * WIRE_DENSITY
    return tension


def log_data(data, filename):
    """
    Log data into a CSV file.

    :param data: Dictionary containing field-value pairs to log.
    :param filename: Name of the file to log the data into.
    """
    # If filename does not contain a path, use the current working directory
    if not os.path.dirname(filename):
        directory = os.getcwd()  # Get current directory
    else:
        directory = os.path.dirname(filename)

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Full path for the file
    full_path = os.path.join(directory, os.path.basename(filename))

    # Ensure the 'time' field is included in the data
    data["time"] = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # Open file safely with handling exceptions
    try:
        with open(full_path, "a", newline="") as csvfile:
            fieldnames = list(data.keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

            # If file is empty, write header
            if os.stat(full_path).st_size == 0:
                writer.writeheader()
            writer.writerow(data)
            # print("Data logged successfully.")
    except IOError as e:
        print(f"Error opening or writing to file: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def load_wire_LUT(apa_name, layer):
    csv_file_path = f"data/wireLUTs/{apa_name}_{layer}.csv"
    wire_data = {}

    with open(csv_file_path, mode="r") as file:
        csv_reader = csv.DictReader(file)

        for row in csv_reader:
            wire_number = row["wire_number"]
            x = float(row["x"])
            y = float(row["y"])
            wire_data[wire_number] = {"x": x, "y": y}

    return wire_data


def gaussian_wiggle(wire_y, wiggle_step):
    while True:
        yield random.gauss(wire_y, wiggle_step)


def stepwise_wiggle(wire_y, wiggle_step=0.1):
    i = 0
    while True:
        yield wire_y + (-1) ** (i // 2 + 1) * (i // 2 + 1) * wiggle_step
        i += 1


def get_wiggle_generator(wiggle_type, wire_y, wiggle_step=0.2):
    if wiggle_type == "gaussian":
        return gaussian_wiggle(wire_y, wiggle_step)
    else:
        return stepwise_wiggle(wire_y, wiggle_step)


def calculate_kde_max(sample):
    """
    Calculate the maximum value of the kernel density estimation (KDE) for a given sample.

    Parameters:
    sample (array-like): An array of sample data.

    Returns:
    float: The maximum value of the KDE.
    """
    # Perform KDE on the sample
    kde_sample = gaussian_kde(sample)

    # Define a range of values for which to calculate the KDE
    x_range = np.linspace(min(sample), max(sample), 1000)
    kde_sample_values = kde_sample(x_range)

    # Find and return the maximum of the KDE
    max_kde_sample_value = x_range[np.argmax(kde_sample_values)]
    return max_kde_sample_value


def tension_pass(tension, length):
    return tension > min(25.8 * length + 0.232, 4) and tension < MAX_TENSION  #


def tension_plausible(tension):
    return tension < 12 and tension > 2


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

        # values = sorted(values)
        # q1 = np.percentile(values, 25)
        # q3 = np.percentile(values, 75)
        # iqr = q3 - q1
        # lower_bound = q1 - 1.5 * iqr
        # upper_bound = q3 + 1.5 * iqr

        # # Check if all values are within the bounds
        # if all(lower_bound <= x <= upper_bound for x in values):
        #     return list(subset)

    return []


def bandpass_filter(
    audio: np.ndarray, samplerate: int, f1: float, f2: float, order: int = 4
) -> np.ndarray:
    """
    Apply a bandpass filter to an audio signal to exclude frequencies
    outside the range [f1, f2].

    Parameters
    ----------
    audio : np.ndarray
        The input audio signal (1D or 2D). If 2D, each column or row
        can be considered a channel (depending on your data layout).
    samplerate : int
        Sampling rate of the audio signal in Hz.
    f1 : float
        Lower cutoff frequency in Hz.
    f2 : float
        Upper cutoff frequency in Hz.
    order : int, optional
        Order of the Butterworth filter (default is 4).

    Returns
    -------
    filtered_audio : np.ndarray
        The bandpass-filtered audio signal.
    """

    # Normalize the cutoff frequencies to the Nyquist frequency (samplerate / 2).
    nyquist = 0.5 * samplerate
    low = f1 / nyquist
    high = f2 / nyquist

    # Design a Butterworth bandpass filter.
    b, a = butter(order, [low, high], btype="band")

    # Apply the filter to the audio signal using filtfilt for zero-phase filtering.
    # If the audio is multi-channel, filtfilt processes each channel along the first dimension by default,
    # so you might need to transpose your data if it is shaped differently.
    # (For a 1D signal, this is straightforward.)
    filtered_audio = filtfilt(b, a, audio, axis=0)

    return filtered_audio


def tension_bandpass_filter(
    audio: np.ndarray, samplerate: int, length: float, order: int = 4
) -> np.ndarray:
    f_min = 1 / (2 * length) * np.sqrt(MIN_PHYSICAL_TENSION / WIRE_DENSITY)
    f_max = 1 / (2 * length) * np.sqrt(MAX_PHYSICAL_TENSION / WIRE_DENSITY)
    # Normalize the cutoff frequencies to the Nyquist frequency (samplerate / 2).
    nyquist = 0.5 * samplerate
    low = f_min / nyquist
    high = f_max / nyquist

    # Design a Butterworth bandpass filter.
    b, a = butter(order, [low, high], btype="band")

    # Apply the filter to the audio signal using filtfilt for zero-phase filtering.
    # If the audio is multi-channel, filtfilt processes each channel along the first dimension by default,
    # so you might need to transpose your data if it is shaped differently.
    # (For a 1D signal, this is straightforward.)
    filtered_audio = filtfilt(b, a, audio, axis=0)

    return filtered_audio
    audio_data = bandpass_filter(audio_data, samplerate, f_min, f_max)
    return audio_data


if __name__ == "__main__":
    wire_x = 4989
    wire_y = 371
    dx, dy = -8, 5.75
    # print(not_close_to_comb(4989))
    print(is_in_bounds(wire_x, wire_y))
    print(next_wire_target(wire_x, wire_y, dx, dy))

    # Example usage
    # numbers = [1, 1.1, 1.3, 10, 12, 23]
    # n = 4
    # result = has_cluster(numbers, n)
    # print(f"Cluster found: {result}")  # Output: Cluster found: True

    # # Test the wiggle generator
    # wire_y = 0
    # wg = get_wiggle_generator("gaussian", wire_y)
    # for _ in range(10):
    #     print(next(wg))
    # for x in range(23, 100):
    #     print(length_lookup("V", x, 1))
