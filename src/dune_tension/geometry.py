# functions related to the geometry of the APA
# geometry constants
from __future__ import annotations

X_MIN: int = 1070
X_MAX: int = 7030
Y_MIN: int = 325
Y_MAX: int = 2625


G_LENGTH: float = 1.285
X_LENGTH: float = 1.273

comb_positions: list[int] = [
    X_MIN,
    2230,
    3420,
    4590,
    5770,
    X_MAX,
]
COMB_SPACING: float = (X_MIN - X_MAX) / 5


def zone_lookup(x: float) -> int:
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

    def is_in_bounds(x_val: float, y_val: float) -> bool:
        return X_MIN <= x_val <= X_MAX and Y_MIN <= y_val <= Y_MAX

    def score(pos: tuple[float, float]) -> float:
        """Return the minimal distance of ``pos`` to any limiting line."""
        px, py = pos
        distances: list[float] = [abs(px - c) for c in comb_positions]
        distances.append(abs(py - Y_MAX))
        distances.append(abs(py - Y_MIN))
        return min(distances)

    candidates: list[tuple[float, float]] = []

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

    low_candidates: list[tuple[float, float]] = [
        c
        for c in candidates
        if score(c) > 200
    ]
    if low_candidates:
        #choose the low candidate with the lowest y value
        return min(low_candidates, key=lambda c: c[1])
    
    return max(candidates, key=score)


def length_lookup(
    layer: str, wire_number: int, zone: int, taped: bool = False
) -> float:
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
        value: float = float(spreadsheet.at[wire_number, str(zone)])
        if taped:
            return (value - 16) / 1000
        return value / 1000
    except KeyError:
        raise ValueError(
            f"no value found for wire {wire_number} in zone {zone} for layer {layer}"
        )
