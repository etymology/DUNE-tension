import json
from typing import Dict, List, Tuple

ZONE_MIDDLES_X = {1: 1600, 2: 2800, 4: 5200, 5: 6400}
FIRST_WIRE_NUMBER = 5
LAST_WIRE_NUMBER = 1150
LAST_WIRE_NUMBER_IN_ZONE = {1: 400, 2: 551, 4: 751, 5: LAST_WIRE_NUMBER}
DIAGONAL_WIRE_PITCH = (8, 5.75)

class APA:
    def __init__(self, name: str):
        """
        Initialize an APA instance with a specified name.

        Args:
            name (str): The name of the APA, which determines the calibration file name.

        Attributes:
            name (str): Name of the APA.
            calibration (dict): Loaded calibration data from the JSON file.
        """
        self.name = name
        self.calibration = self.load_calibration_from_json()

    def calibration_file_path(self) -> str:
        """
        Generate the file path for the calibration file based on the APA name.

        Returns:
            str: The file path for the APA's calibration data.
        """
        return f"APAcalibrationFiles/{self.name}_tension_calibration.json"

    def load_calibration_from_json(self) -> Dict[str, List[Tuple[float, float]]]:
        """
        Load calibration data from a JSON file, handling the case where the file does not exist.

        Returns:
            dict: A dictionary containing the calibration data for different layers.
                  Returns a default dictionary with empty lists if the file is not found.
        """
        try:
            with open(self.calibration_file_path(), 'r') as file:
                return json.load(file)
        except FileNotFoundError:
            print(f"No calibration file found for APA {self.name}. Using blank calibration.")
            return {"X": [], "V": [], "U": [], "G": []}

    def save_calibration_to_json(self):
        """
        Save the current calibration data to a JSON file.
        """
        with open(self.calibration_file_path(), 'w') as file:
            json.dump(self.calibration, file)

    def calibrate(self, layer="", first_wire_coordinates=None, last_wire_coordinates=None):
        """
        Facilitate the calibration process for a specified layer of the APA.

        Args:
            layer (str, optional): The layer to calibrate ('X', 'V', 'U', or 'G').
            first_wire_coordinates (tuple, optional): Coordinates (x, y) for the first wire in the calibration process.
            last_wire_coordinates (tuple, optional): Coordinates (x, y) for the last wire in the calibration process.
        """
        layer = self.get_layer_name(layer)
        if not self.confirm_overwrite(layer):
            return
        
        if layer in ['X', 'G']:
            self.handle_horizontal_layers(layer, first_wire_coordinates)
        elif layer in ['V', 'U']:
            self.handle_diagonal_layers(layer, first_wire_coordinates, last_wire_coordinates)

        self.save_calibration_to_json()

    def get_layer_name(self, layer: str) -> str:
        """
        Ensure the layer name is valid and return it in uppercase.

        Args:
            layer (str): The layer name input by the user or passed as an argument.

        Returns:
            str: The validated and normalized layer name.
        """
        layer = layer.upper() or input("Enter layer (X, V, U, or G): ").upper()
        assert layer in ["X", "V", "U", "G"], "Invalid layer name"
        return layer

    def confirm_overwrite(self, layer: str) -> bool:
        """
        Confirm whether the user wants to overwrite existing calibration data.

        Args:
            layer (str): The layer for which calibration might be overwritten.

        Returns:
            bool: True if the user confirms overwrite, False otherwise.
        """
        if self.calibration.get(layer):
            return input(f"Do you really want to overwrite the existing calibration for layer \"{layer}\"? (y/n)").strip().upper() == "Y"
        return True

    def handle_horizontal_layers(self, layer: str, first_wire_coordinates: Tuple[float, float]):
        """
        Handle calibration for layers with fixed y-values across all wires.

        Args:
            layer (str): The layer being calibrated.
            first_wire_coordinates (tuple): Coordinates (x, y) of the first wire.
        """
        y_value = first_wire_coordinates[1] if first_wire_coordinates else self.get_float_input("Enter the Y value for the lowest wire on the compensator(fixed) side: ")
        self.calibration[layer] = [(ZONE_MIDDLES_X[5], y_value + i * 2300 / 480) for i in range(480)]

    def handle_diagonal_layers(self, layer: str, first_wire_coordinates: Tuple[float, float], last_wire_coordinates: Tuple[float, float]):
        """
        Handle calibration for layers that require diagonal alignment of wires.

        Args:
            layer (str): The layer being calibrated.
            first_wire_coordinates (tuple): Starting coordinates for the diagonal calibration.
            last_wire_coordinates (tuple): Ending coordinates for the diagonal calibration.
        """
        if not first_wire_coordinates:
            first_wire_coordinates = self.get_coordinates_input(f"Enter initial coordinates for the {FIRST_WIRE_NUMBER}th wire of the \"{layer}\" layer (comma-separated): ")
        if not last_wire_coordinates:
            last_wire_coordinates = self.get_coordinates_input(f"Enter coordinates for the last wire of the \"{layer}\" layer (comma-separated): ")
        
        self.calibration[layer] = self.calculate_uv_layer_coordinates(layer, first_wire_coordinates, last_wire_coordinates)

    def get_float_input(self, prompt: str) -> float:
        """
        Get a floating-point input from the user with error handling.

        Args:
            prompt (str): The prompt to display to the user.

        Returns:
            float: The user-entered floating-point number.
        """
        while True:
            try:
                return float(input(prompt))
            except ValueError:
                print("Invalid input. Please enter a valid floating-point number.")

    def get_coordinates_input(self, prompt: str):
        """
        Prompt the user to enter two float values separated by a comma, and return them as a tuple.

        Args:
            prompt (str): The input prompt provided to the user.

        Returns:
            Tuple[float, float]: The coordinates entered by the user.
        """
        while True:
            coordinates = input(prompt).split(',')
            if len(coordinates) == 2:
                try:
                    return tuple(map(float, coordinates))
                except ValueError:
                    print("Invalid input. Please enter valid floating-point numbers.")
            print("Invalid input. Please enter two values separated by a comma.")

    @staticmethod
    def calculate_uv_layer_coordinates(layer_name, first_wire_coordinates, last_wire_coordinates):
        """
        Calculate the calibration coordinates for the specified 'U' or 'V' layer using a combination of 
        diagonal and vertical movements.

        This method handles the complex calibration logic that includes two main movements in zones 1 and 5:
        a diagonal movement followed by a vertical movement. The calibration logic also processes a straight
        vertical movement in zones 2 and 4. This method adjusts movements based on the layer specified, 
        reflecting the mirror differences between the 'U' and 'V' layers.

        Args:
            layer_name (str): The layer being calibrated, must be either 'U' or 'V'.
            first_wire_coordinates (Tuple[float, float]): The (x, y) coordinates of the first wire to start the calibration.
            last_wire_coordinates (Tuple[float, float]): The (x, y) coordinates of the last wire to end the calibration.

        Returns:
            Dict[int, Tuple[float, float]]: A dictionary where keys are wire numbers and values are tuples of (x, y) 
            coordinates representing the calculated calibration points for each wire.

        Raises:
            AssertionError: If the layer name is not 'U' or 'V'.

        Example:
            >>> calculate_uv_layer_coordinates('V', (100, 200), (150, 300))
            {5: (100, 200), 6: (104, 197.5), ...}
        """
        def process_diagonal_movement(x, y, wire_number, boundary_x, increment=True):
            step = 1 if increment else -1
            coordinates = {}
            while (x < boundary_x if increment else x > boundary_x):
                coordinates[wire_number] = (x, y)
                x += step * (DIAGONAL_WIRE_PITCH[0]/2)
                y -= step * (DIAGONAL_WIRE_PITCH[1]/2)
                wire_number += step
            return coordinates

        def process_vertical_movement(x, y, wire_number, last_wire_number, increment=True):
            step = 1 if increment else -1
            coordinates = {}
            while wire_number <= last_wire_number if increment else wire_number >= last_wire_number:
                coordinates[wire_number] = (x, y)
                y -= step * DIAGONAL_WIRE_PITCH[1]
                wire_number += step
            return coordinates
        
        assert layer_name in ['U', 'V'], "Layer name must be 'U' or 'V'"

        # Initialize starting points
        temp_x, temp_y, temp_wirenumber = first_wire_coordinates
        calibration_data = {}

        # Determine direction multipliers based on layer
        if layer_name == 'V':
            increment_initial = True
            decrement_final = False
        else:  # 'U' layer is the mirror of 'V'
            increment_initial = False
            decrement_final = True

        # Zone 1: Two-part movement
        # Part 1: Diagonal movement
        diagonal_stop_x = ZONE_MIDDLES_X[1] if layer_name == 'V' else ZONE_MIDDLES_X[1] - \
            DIAGONAL_WIRE_PITCH[0]/2
        calibration_data.update(process_diagonal_movement(
            temp_x, temp_y, temp_wirenumber, diagonal_stop_x, increment=increment_initial))

        # Update starting point after diagonal movement
        last_temp_wirenumber = max(calibration_data.keys(
        )) if increment_initial else min(calibration_data.keys())
        temp_x, temp_y = calibration_data[last_temp_wirenumber]

        # Part 2: Vertical movement in Zone 1
        calibration_data.update(process_vertical_movement(
            temp_x, temp_y, last_temp_wirenumber + 1, LAST_WIRE_NUMBER_IN_ZONE[1], increment=increment_initial))

        # Zone 2: Straight vertical movement
        temp_x = ZONE_MIDDLES_X[2]
        last_temp_wirenumber = LAST_WIRE_NUMBER_IN_ZONE[1] + 1
        # continue from last y position
        temp_y = calibration_data[last_temp_wirenumber - 1][1]
        calibration_data.update(process_vertical_movement(
            temp_x, temp_y, last_temp_wirenumber, LAST_WIRE_NUMBER_IN_ZONE[2], increment=increment_initial))

        # Prepare coordinates for Zones 5 and 4 from the corner downwards or upwards
        temp_x, temp_y, temp_wirenumber = last_wire_coordinates

        # Zone 5: Two-part movement
        # Part 1: Diagonal movement in Zone 5
        diagonal_stop_x = ZONE_MIDDLES_X[5] if layer_name == 'V' else ZONE_MIDDLES_X[5] + \
            DIAGONAL_WIRE_PITCH[0]/2
        calibration_data.update(process_diagonal_movement(
            temp_x, temp_y, temp_wirenumber, diagonal_stop_x, increment=decrement_final))

        # Update starting point after diagonal movement
        last_temp_wirenumber = min(calibration_data.keys(
        )) if decrement_final else max(calibration_data.keys())
        temp_x, temp_y = calibration_data[last_temp_wirenumber]

        # Part 2: Vertical movement in Zone 5
        last_temp_wirenumber += -1 if decrement_final else 1
        calibration_data.update(process_vertical_movement(
            temp_x, temp_y, last_temp_wirenumber, LAST_WIRE_NUMBER_IN_ZONE[5], increment=decrement_final))

        # Zone 4: Straight vertical movement
        temp_x = ZONE_MIDDLES_X[4]
        last_temp_wirenumber = LAST_WIRE_NUMBER_IN_ZONE[5] + \
            (-1 if decrement_final else 1)
        # continue from last y position
        temp_y = calibration_data[last_temp_wirenumber +
                                    (1 if decrement_final else -1)][1]
        calibration_data.update(process_vertical_movement(
            temp_x, temp_y, last_temp_wirenumber, LAST_WIRE_NUMBER_IN_ZONE[4], increment=decrement_final))

        return calibration_data