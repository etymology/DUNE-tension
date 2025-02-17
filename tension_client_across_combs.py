from Tensiometer import Tensiometer
from utilities import (
    next_wire_target,
)
from data_collection import collect_wire_data
import pandas as pd


def measure_sequential_across_combs(
    t: Tensiometer,
    initial_wire_number: int,
    direction: int = 1,
    use_relative_position: bool = False,
    use_LUT: bool = True,
    final_wire_number: int = None,
):
    # direction = 1 for increasing wire number, -1 for decreasing wire number

    if t.layer in ["X", "G"]:
        dx, dy = 0.0, 2300 / 480
        wire_min, wire_max = 1, 480
        if t.layer == "G":
            wire_max = 481
    else:
        dx, dy = 8.0, 5.75
        wire_min, wire_max = 4, 1151
        if (t.layer == "U" and t.side == "A") or (t.layer == "V" and t.side == "B"):
            dy = -5.75

    dx *= direction
    dy *= direction

    wire_number = initial_wire_number

    def measure_horizontal_layer():
        nonlocal wire_number
        wire_x, wire_y = t.get_xy()  # for testing

        while wire_number >= wire_min and wire_number <= wire_max and wire_number != final_wire_number:
            wire_y = t.initial_wire_height + dy * (wire_number - 1)
            wire_data = collect_wire_data(t, wire_number, wire_x, wire_y)
            wire_number += direction
            if use_relative_position:
                y = wire_data["y"]
            else:
                x, y = t.get_xy()

    def measure_diagonal_layer():
        nonlocal wire_number
        if use_LUT and get_coordinates(t, wire_number) is not None:
            wire_x, wire_y = get_coordinates(t, wire_number)
        else:
            wire_x, wire_y = t.get_xy()
            
        while wire_number <= wire_max and wire_number >= wire_min:
            wire_data = collect_wire_data(t, wire_number, wire_x, wire_y)
            if use_relative_position:
                wire_x, wire_y = wire_data["x"], wire_data["y"]
            wire_x, wire_y = next_wire_target(wire_x, wire_y, dx, dy)
            wire_number += direction

    if t.layer in ["X", "G"]:
        measure_horizontal_layer()
    else:
        measure_diagonal_layer()


def measure_LUT(t: Tensiometer, wire_numbers_to_measure: list):
    if t.layer in ["X", "G"]:
        for wire_number in wire_numbers_to_measure:
            collect_wire_data(
                t,
                wire_number,
                6300,
                t.initial_wire_height + 2300 / 480 * (wire_number - 1),
            )
    else:
        for wire_number in wire_numbers_to_measure:
            wire_x, wire_y = get_coordinates(t, wire_number)
            if wire_x is not None and wire_y is not None:
                t.goto_xy(wire_x, wire_y)
                collect_wire_data(t, wire_number, wire_x, wire_y)
            else:
                print(f"Wire {wire_number} not found in LUT.")


def get_coordinates(t: Tensiometer, wire_number: int):
    """
    Function to retrieve x and y coordinates for a given wire_number and side from a CSV file.

    Args:
        file_path (str): Path to the CSV file.
        wire_number (int): The wire number to look up.
        side (str): The side to look up ('A' or 'B').

    Returns:
        tuple: (x, y) coordinates as floats if found, or None if not found.
    """
    # Load the CSV file
    df = pd.read_csv(f"data/frequency_data_{t.apa_name}_{t.layer}.csv")

    # Filter rows matching wire_number and side
    result = df[(df["wire_number"] == wire_number) & (df["side"] == t.side)]

    if not result.empty:
        # Extract x and y coordinates
        x = result["x"].iloc[0]
        y = result["y"].iloc[0]
        return float(x), float(y)
    else:
        return None  # Return None if no matching row is found


def seek_wire(t: Tensiometer, layer, side, wire_number):
    x, y = t.get_xy()
    max_confidence = 0.0
    best_y = y
    i = 0.0
    wiggle_step = 0.25
    while i < 20:
        wire_data = collect_wire_data(
            t,
            layer,
            side,
            wire_number,
            x,
            y + (-1) ** (i // 2 + 1) * (i // 2 + 1) * wiggle_step,
        )
        confidence = wire_data["confidence"]
        if confidence > max_confidence:
            max_confidence = confidence
            best_y = y + (-1) ** (i // 2 + 1) * (i // 2 + 1) * wiggle_step
        i += 1
    print(f"Best y: {best_y}, confidence: {max_confidence}")
    t.goto_xy(x, best_y)
    return x, best_y


def measure_one_wire(t: Tensiometer, wire_number, tries):
    x,y = t.get_xy()
    for n in range(tries):
        collect_wire_data(t,wire_number,x,y)