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


def refine_position(
    x: float, y: float, dx: float, dy: float
) -> tuple[float, float] | None:
    """Refine ``(x, y)`` along ``(dx, dy)`` staying in bounds.

    The function searches in both ``+n`` and ``-n`` directions for a
    position that is inside the allowed geometry and as far as possible
    from the comb and ``Y`` limits.  Among all valid candidates the one
    that maximises the minimal distance to the lines ``x = c`` for
    ``c`` in :data:`comb_positions` and ``y = Y_MIN``/``Y_MAX`` is
    chosen.  If no candidate is valid the original coordinates are
    returned unchanged.
    """

    def is_in_bounds(x: float, y: float) -> bool:
        return X_MIN <= x <= X_MAX and Y_MIN <= y <= Y_MAX

    def score(pos: tuple[float, float]) -> float:
        """Return the minimal distance of ``pos`` to any limiting line."""
        px, py = pos
        distances = [abs(px - c) for c in comb_positions]
        distances.append(abs(py - Y_MAX))
        distances.append(abs(py - Y_MIN))
        return min(distances)

    candidates = []

    for n in range(300):
        # Generate forward and reverse candidates
        x1, y1 = x + n * dx, y - n * dy
        x2, y2 = x - n * dx, y + n * dy

        if is_in_bounds(x1, y1):
            candidates.append((x1, y1))
        if is_in_bounds(x2, y2):
            candidates.append((x2, y2))

    if not candidates:
        return (x, y)

    # Choose the candidate furthest from limiting lines
    return max(candidates, key=score)


# def refine_position(x, y, dx, dy):
#     # Compute t where line crosses vertical boundaries x = c
#     t_boundaries = [(c - x) / dx for c in comb_positions]

#     # Compute t where line crosses horizontal boundaries y = Y_MIN and Y_MAX
#     t_boundaries.append((Y_MIN - y) / dy)
#     t_boundaries.append((Y_MAX - y) / dy)

#     # Sort boundary crossings
#     t_boundaries.sort()

#     # Compute allowed t range based on staying within the inner region
#     if dx > 0:
#         t_xmin = (comb_positions[0] - x) / dx
#         t_xmax = (comb_positions[-1] - x) / dx
#     else:
#         t_xmax = (comb_positions[0] - x) / dx
#         t_xmin = (comb_positions[-1] - x) / dx

#     if dy > 0:
#         t_ymin = (Y_MIN - y) / dy
#         t_ymax = (Y_MAX - y) / dy
#     else:
#         t_ymax = (Y_MIN - y) / dy
#         t_ymin = (Y_MAX - y) / dy

#     # Compute the allowed t interval
#     t_allowed_min = max(t_xmin, t_ymin)
#     t_allowed_max = min(t_xmax, t_ymax)

#     # Initialize
#     max_interval = -1
#     best_t = 0  # default to original point

#     # Find the largest interval within the allowed range
#     for i in range(len(t_boundaries) - 1):
#         t_left = t_boundaries[i]
#         t_right = t_boundaries[i + 1]

#         # Clip interval to allowed range
#         clipped_left = max(t_left, t_allowed_min)
#         clipped_right = min(t_right, t_allowed_max)

#         # If the clipped interval is valid
#         if clipped_right > clipped_left:
#             interval = clipped_right - clipped_left

#             if interval > max_interval:
#                 max_interval = interval
#                 best_t = 0.5 * (clipped_left + clipped_right)

#     # Compute the corresponding point
#     best_x = x + dx * best_t
#     best_y = y + dy * best_t

#     return best_x, best_y


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
