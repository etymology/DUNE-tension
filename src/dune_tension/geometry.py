# functions related to the geometry of the APA
# geometry constants
G_LENGTH = 1.285
X_LENGTH = 1.273
COMB_SPACING = 1190
Y_MIN = 200
Y_MAX = 2460
X_MIN = 1050
X_MAX = 7000
COMB_TOLERANCE = 300

comb_positions = [
    X_MIN,
    2230,
    3420,
    4590,
    5770,
    X_MAX,
]


def zone_lookup(x) -> int:
    """
    Determine the zone based on the x-coordinate.
    """
    for i, pos in enumerate(comb_positions):
        if pos > x:
            return i
    return 0


def zone_x_target(zone: int):
    return [1635, 2825, 4015, 5185, 6365][zone - 1]


def refine(x, y, dx, dy):
    # Compute t where line crosses vertical boundaries x = c
    t_boundaries = [(c - x) / dx for c in comb_positions]

    # Compute t where line crosses horizontal boundaries y = Y_MIN and Y_MAX
    t_boundaries.append((Y_MIN - y) / dy)
    t_boundaries.append((Y_MAX - y) / dy)

    # Sort boundary crossings
    t_boundaries.sort()

    # Initialize
    max_interval = -1
    best_t = 0  # default to original point

    # Find the largest interval between consecutive t values
    for i in range(len(t_boundaries) - 1):
        t_left = t_boundaries[i]
        t_right = t_boundaries[i + 1]
        interval = t_right - t_left

        if interval > max_interval:
            max_interval = interval
            best_t = 0.5 * (t_left + t_right)

    # Compute the corresponding point
    best_x = x + dx * best_t
    best_y = y + dy * best_t

    return best_x, best_y


def length_lookup(layer: str, wire_number: int, zone: int, taped=False):
    import pandas as pd

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
        raise ValueError(
            f"no value found for wire {wire_number} in zone {zone} for layer {layer}"
        )
