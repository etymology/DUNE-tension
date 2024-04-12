import json
import os
from typing import Dict, List, Tuple

"""
These are the default X values the middle of four of the five "zones" or columns between ribs. These
values only need to be approximately in the middle of the ribs to achieve a sensible measurement.
Zone 1 is headmost, zone 5 footmost. Only 1,2,4,5 are used
"""
ZONE_MIDDLES_X = {1: 1600, 2: 2800, 4: 5200, 5: 6400}
FIRST_WIRE_NUMBER = 5
LAST_WIRE_NUMBER = 1150
LAST_WIRE_NUMBER_IN_ZONE = {1: 400, 2: 551,
                            4: 751, 5: LAST_WIRE_NUMBER}
DIAGONAL_WIRE_PITCH = (8, 5.75)


class APA:
    def __init__(self, name: str):
        """
        Initializes an instance an APA with a given name. APA's have an
        associated calibration file called f"{name}_tension_calibration.json"
        and a functions which load, and write calibrations. A calibration is
        of the form APA.getPluckingPoint(layer:str, wire_number:int) -> Tuple[Float,Float]

        Args:
            name (str): The name to assign to the instance.

        Returns:
            None

        Examples:
            >>> instance = APA("Wood")
        """
        self.name = name
        self.calibration: Dict[str, List[Tuple[float, float]]]
        self.load_calibration_from_json()

    def calibration_file_path(self):
        return f"APAcalibrationFiles/{self.name}_tension_calibration.json"

    def load_calibration_from_json(self):
        """
        Loads calibration data from a JSON file associated with the APA instance.

        Args:
            self: The APA instance to load calibration data for.

        Returns:
            None
        """

        filename = self.calibration_file_path()
        if os.path.exists(filename):
            with open(filename, 'r') as file:
                self.calibration = json.load(file)
        else:
            print(
                f"No calibration file found for APA {self.name}. Using blank calibration.")
            self.calibration = {"X": [], "V": [], "U": [], "G": []}

    def save_calibration_to_json(self):
        """
        Saves the calibration data of the APA instance to a JSON file.

        Args:
            self: The APA instance to save calibration data for.

        Returns:
            None
        """

        filename = self.calibration_file_path()
        with open(filename, 'w') as file:
            json.dump(self.calibration, file)

    def calibrate(self, layer="", first_wire_coordinates=None, last_wire_coordinates=None):
        """
        QUESTION: should the this function ask the user to move to the wire, or ask to type in a value?

        Prompts the user for initial calibration data based on the layer provided.
        OR, takes layer as input.

        Args:
            self: The APA instance to prompt calibration for.
            layer (str, optional): The layer for which calibration is prompted. Defaults to "".

        Returns:
            None
        """

        def get_float_input(prompt):
            while True:
                try:
                    return float(input(prompt))
                except ValueError:
                    print("Invalid input. Please enter a valid floating-point number.")

        def get_coordinates_input(prompt):
            while True:
                coordinates = input(prompt).split(',')
                if len(coordinates) != 2:
                    print(
                        "Invalid input. Please enter two values separated by a comma.")
                else:
                    try:
                        return tuple(map(float, coordinates))
                    except ValueError:
                        print(
                            "Invalid input. Please enter valid floating-point numbers.")

        layer_name = layer.upper() if layer else input(
            "Enter layer (X, V, U, or G): ").upper()

        if layer_name not in ["X", "V", "U", "G"]:
            print("Invalid layer name (not X, V, U or G)")
            return
        else:
            self.load_calibration_from_json()

            if self.calibration[layer_name]:
                overwrite = input(
                    f"Do you really want to overwrite the existing calibration for layer \"{layer_name}\"? (y/n)").upper()
                if overwrite != "Y":
                    return

            if layer_name in ['X', 'G']:
                if first_wire_coordinates is None:
                    y_value = get_float_input(
                        "Enter the Y value for the lowest wire on the compensator(fixed) side: ")
                else:
                    y_value = first_wire_coordinates[1]
                self.calibration[layer_name] = [
                    (ZONE_MIDDLES_X[5], y_value + i * 2300 / 480) for i in range(480)]

            elif layer_name in ['V', 'U']:
                if first_wire_coordinates is None:
                    first_wire_coordinates = get_coordinates_input(
                        f"Enter initial coordinates for the {FIRST_WIRE_NUMBER}th wire of the \"{layer_name}\" layer (from the LSB/head corner, the first over empty space) (comma-separated): ")
                if last_wire_coordinates is None:
                    last_wire_coordinates = get_coordinates_input(
                        f"Enter coordinates for the last wire of the \"{layer_name}\" layer (comma-separated): ")

                if layer_name == 'V':
                    temp_x, temp_y, temp_wirenumber = first_wire_coordinates[
                        0], first_wire_coordinates[1], FIRST_WIRE_NUMBER

                    # ZONE 1
                    # DIAGONAL movement in +x,-y until middle of zone1
                    while temp_x < ZONE_MIDDLES_X[1]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_x += DIAGONAL_WIRE_PITCH[0]/2
                        temp_y -= DIAGONAL_WIRE_PITCH[1]/2
                        temp_wirenumber += 1

                    # Vertical movement in -y until end of zone1
                    while temp_wirenumber <= LAST_WIRE_NUMBER_IN_ZONE[1]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_y -= DIAGONAL_WIRE_PITCH[1]
                        temp_wirenumber += 1

                    # ZONE 2
                    # ONLY VERTICAL (0,-y)
                    temp_x = ZONE_MIDDLES_X[2]
                    # Up diagonally along the last wire in Zone 1, then down one y wire pitch
                    temp_y += (temp_x-ZONE_MIDDLES_X[1])*DIAGONAL_WIRE_PITCH[1] / \
                        DIAGONAL_WIRE_PITCH[0]-DIAGONAL_WIRE_PITCH[1]
                    temp_wirenumber = LAST_WIRE_NUMBER_IN_ZONE[1]+1
                    while temp_wirenumber <= LAST_WIRE_NUMBER_IN_ZONE[2]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_y -= DIAGONAL_WIRE_PITCH[1]
                        temp_wirenumber += 1

                    # Now we calibrate from the foot/HSB corner, first zone 5 the zone 4. We count downwards this time.
                    temp_x = last_wire_coordinates[0]
                    temp_y = last_wire_coordinates[1]
                    temp_wirenumber = LAST_WIRE_NUMBER
                    # ZONE 5
                    # DIAGONAL movement in -x,+y until middle of zone5
                    while temp_x > ZONE_MIDDLES_X[5]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_x -= DIAGONAL_WIRE_PITCH[0]/2
                        temp_y += DIAGONAL_WIRE_PITCH[1]/2
                        temp_wirenumber -= 1

                    # Vertical movement in +y until end of zone1
                    while temp_wirenumber >= LAST_WIRE_NUMBER_IN_ZONE[5]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_y += DIAGONAL_WIRE_PITCH[1]
                        temp_wirenumber -= 1

                    # ZONE 4
                    # ONLY VERTICAL (0,+y)
                    temp_x = ZONE_MIDDLES_X[4]
                    # Down diagonally along the last wire in Zone 5, then up one y wire pitch
                    temp_y += (temp_x-ZONE_MIDDLES_X[5])*DIAGONAL_WIRE_PITCH[1] / \
                        DIAGONAL_WIRE_PITCH[0]+DIAGONAL_WIRE_PITCH[1]
                    temp_wirenumber = LAST_WIRE_NUMBER_IN_ZONE[5]+1
                    while temp_wirenumber > LAST_WIRE_NUMBER_IN_ZONE[2]:
                        self.calibration[layer_name][temp_wirenumber] = (
                            temp_x, temp_y)
                        temp_y += DIAGONAL_WIRE_PITCH[1]
                        temp_wirenumber -= 1
                else:
                    print("U layer behavior not yet defined")

        self.save_calibration_to_json()
