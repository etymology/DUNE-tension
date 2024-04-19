import json
from typing import Dict, List, Tuple

####### APA PARAMETERS #######
## HORIZONTAL APA PARAMS ##
HORI_LAYER_X = 6400
HORI_DELTA_Y = 4.7916667

HORI_LAYER_MIN_WIRENUM = 1
HORI_LAYER_MAX_WIRENUM = 480

## DIAGONAL APA PARAMS ##
DIAG_DELTA_X = 2.72455392
DIAG_DELTA_Y = 3.79161
DELTA_VERTICAL_ONLY = 5.75

DIAG_LAYER_MIN_WIRENUM = 8
DIAG_LAYER_MAX_WIRENUM = 1146

DIAG_LAYER_Z1_COMP1_MAX_WIRENUM = 218
DIAG_LAYER_Z1_COMP2_MAX_WIRENUM = 399
DIAG_LAYER_Z2_MAX_WIRENUM = 551
DIAG_LAYER_Z4_MAX_WIRENUM = 751
DIAG_LAYER_Z5_COMP2_MAX_WIRENUM = 991

PITCH_RATIO = 5.75/8.0

Z2_X = 2800
Z4_X = 5150

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
            self.handle_hori_layer(layer, first_wire_coordinates, last_wire_coordinates)
        elif layer in ['V', 'U']:
            self.handle_diag_layer(layer, first_wire_coordinates, last_wire_coordinates)

        self.save_calibration_to_json()

    def handle_diag_layer(self, layer, first_wire_coordinates: Tuple[float, float], last_wire_coordinates: Tuple[float, float]):
        """
        Handle calibration for V layer.

        Args:
            first_wire_coordinates (tuple): Starting coordinates for the diagonal calibration.
            last_wire_coordinates (tuple): Ending coordinates for the diagonal calibration.
        """
        if not first_wire_coordinates:
            first_wire_coordinates = self.get_coordinates_input(f"Enter initial coordinates for the {DIAG_LAYER_MIN_WIRENUM}th wire of the \"{layer}\" layer (comma-separated): ")
        if not last_wire_coordinates:
            last_wire_coordinates = self.get_coordinates_input(f"Enter coordinates for the last wire of the \"{layer}\" layer (comma-separated): ")

        incr_bool = -1 if layer == 'U' else 1

        # Zone 1, broken into two components
        z1_comp1 = self.make_config_comp(first_wire_coordinates[0],first_wire_coordinates[1], 
                                     DIAG_LAYER_MIN_WIRENUM,incr_bool*DIAG_DELTA_X,-incr_bool*DIAG_DELTA_Y, 
                                     DIAG_LAYER_MIN_WIRENUM,DIAG_LAYER_Z1_COMP1_MAX_WIRENUM)

        z1_comp2 = self.make_config_comp(z1_comp1[DIAG_LAYER_Z1_COMP1_MAX_WIRENUM]["X"], 
                                     z1_comp1[DIAG_LAYER_Z1_COMP1_MAX_WIRENUM]["Y"], 
                                     DIAG_LAYER_Z1_COMP1_MAX_WIRENUM, 
                                     0.0, -incr_bool*DELTA_VERTICAL_ONLY, 
                                     DIAG_LAYER_Z1_COMP1_MAX_WIRENUM+1, 
                                     DIAG_LAYER_Z1_COMP2_MAX_WIRENUM)
        z1 = z1_comp1 | z1_comp2

        # Zone 2
        z2 = self.make_config_comp(Z2_X, (z1[DIAG_LAYER_Z1_COMP2_MAX_WIRENUM]["Y"]-DELTA_VERTICAL_ONLY)\
                                      +(Z2_X-z1[DIAG_LAYER_Z1_COMP2_MAX_WIRENUM]["X"])*PITCH_RATIO, 
                                       DIAG_LAYER_Z1_COMP2_MAX_WIRENUM+1, 
                                      0.0, -incr_bool*DELTA_VERTICAL_ONLY, 
                                      DIAG_LAYER_Z1_COMP2_MAX_WIRENUM+1, 
                                      DIAG_LAYER_Z2_MAX_WIRENUM)

        # Zone 4
        z4 = self.make_config_comp(Z4_X, z2[DIAG_LAYER_Z2_MAX_WIRENUM]["Y"]\
                                     +(Z4_X-z2[DIAG_LAYER_Z2_MAX_WIRENUM]["X"])*PITCH_RATIO, 
                                     DIAG_LAYER_Z2_MAX_WIRENUM, 
                                     0.0, -incr_bool*DELTA_VERTICAL_ONLY, 
                                     DIAG_LAYER_Z2_MAX_WIRENUM+1, 
                                     DIAG_LAYER_Z4_MAX_WIRENUM)

        # Zone 5
        z5_comp1 = self.make_config_comp(last_wire_coordinates[0],last_wire_coordinates[1], 
                                     DIAG_LAYER_MAX_WIRENUM,incr_bool*DIAG_DELTA_X,-incr_bool*DIAG_DELTA_Y, 
                                     DIAG_LAYER_Z5_COMP2_MAX_WIRENUM+1,DIAG_LAYER_MAX_WIRENUM)

        z5_comp2 = self.make_config_comp(z5_comp1[DIAG_LAYER_Z5_COMP2_MAX_WIRENUM+1]["X"], 
                                     z5_comp1[DIAG_LAYER_Z5_COMP2_MAX_WIRENUM+1]["Y"], 
                                     DIAG_LAYER_Z5_COMP2_MAX_WIRENUM+1, 
                                     0.0, -DELTA_VERTICAL_ONLY, 
                                     DIAG_LAYER_Z4_MAX_WIRENUM+1,DIAG_LAYER_Z5_COMP2_MAX_WIRENUM)
        z5 = z5_comp1 | z5_comp2

        self.calibration[layer] = z1 | z2 | z4 | z5

    def handle_hori_layer(self, layer, first_wire_coordinates: Tuple[float, float]):
        """
        Handle calibration for horizontal layer.

        Args:
            first_wire_coordinates (tuple): Coordinates (x, y) of the first wire.
        """
        y_value = first_wire_coordinates[1] if first_wire_coordinates \
                  else self.get_float_input("Enter the Y value for the lowest wire on the compensator (fixed) side: ")

        self.calibration[layer] = self.make_config_comp(HORI_LAYER_X, y_value, HORI_LAYER_MAX_WIRENUM, 
                                      0, -HORI_DELTA_Y, 
                                      HORI_LAYER_MIN_WIRENUM, HORI_LAYER_MAX_WIRENUM)
        print(HORI_LAYER_MIN_WIRENUM)
        print(HORI_LAYER_MAX_WIRENUM)
        print(self.calibration["G"][1])
        print(self.calibration["G"][480])

    def make_config_comp(self, calx, caly, calwire, delx, dely, minwirenum, maxwirenum):
        """
        Produce a dictionary for some component of the calibration dict. Here
        a component is defined as some segment of consecutively numbered wires
        with equal wire spacing.

        Args:
            calx (float): X component of the calibration point 
            caly (float): X component of the calibration point 
            calwire (int): wirenumber of the calibration point 
            delx (float): change in x between wires in the component 
            dely (float): change in y between wires in the component
            minwirenum (float): lowest wire number in the component 
            maxwirenunm (float): highest wire number in the component 

        Returns:
            dict: dictionary for some component of the layer
        """
        return {wire: {"X": calx + (wire - calwire) * delx,
                       "Y": caly + (wire - calwire) * dely}
               for wire in range(minwirenum, maxwirenum + 1)}

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

    def get_plucking_point(self, wire_number, layer):
        """
        Using a wire_number input, return x and y for the location to pluck the corresponding wire

        Args: 
            wire_number (int): The wire number of plucking point.

        Returns: 
            Tuple[float, float]: The coordinates of the plucking point for wire_number input 
        """
        wire_loc = self.load_calibration_from_json()[layer][wire_number]

        return wire_loc

