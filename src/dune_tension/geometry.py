# functions related to the geometry of the APA
# geometry constants
X_MIN = 1058
X_MAX = 7011
Y_MIN = 181
Y_MAX = 2460


G_LENGTH = 1.285
X_LENGTH = 1.273

comb_positions = [
    X_MIN,
    2230,
    3420,
    4590,
    5770,
    X_MAX,
]
COMB_SPACING = (X_MIN - X_MAX) / 5


def zone_lookup(x) -> int:
    """
    Determine the zone based on the x-coordinate.
    """
    for i, pos in enumerate(comb_positions):
        if pos > x:
            return i
    return 0


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

    low_candidates = [
        c
        for c in candidates
        if c[1] < Y_MAX / 2 and score(c) > COMB_SPACING / 2 * 5.75 / 8
    ]
    if low_candidates:
        return max(low_candidates, key=score)
    return max(candidates, key=score)


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
