# functions related to the geometry of the APA
from __future__ import annotations

from functools import lru_cache

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.config import GEOMETRY_CONFIG
except ImportError:  # pragma: no cover
    from config import GEOMETRY_CONFIG

X_MIN: int = GEOMETRY_CONFIG.x_min
X_MAX: int = GEOMETRY_CONFIG.x_max
Y_MIN: int = GEOMETRY_CONFIG.y_min
Y_MAX: int = GEOMETRY_CONFIG.y_max

G_LENGTH: float = GEOMETRY_CONFIG.g_length_m
X_LENGTH: float = GEOMETRY_CONFIG.x_length_m

comb_positions: list[int] = list(GEOMETRY_CONFIG.comb_positions)
COMB_SPACING: float = GEOMETRY_CONFIG.comb_spacing


def zone_lookup(x: float) -> int:
    """Return zone index in ``[1, 5]`` for coordinate ``x``.

    Zones are defined by the five segments between comb boundaries:
    ``[X_MIN, 2230)``, ``[2230, 3420)``, ``[3420, 4590)``, ``[4590, 5770)``,
    and ``[5770, X_MAX]``.
    Coordinates outside ``[X_MIN, X_MAX]`` are clamped to the nearest edge.
    """

    boundaries = comb_positions
    clamped_x = min(max(float(x), boundaries[0]), boundaries[-1])

    for idx in range(1, len(boundaries) - 1):
        if clamped_x < boundaries[idx]:
            return idx
    return len(boundaries) - 1


@lru_cache(maxsize=2)
def _load_wire_length_lut(layer: str):
    import pandas as pd

    file_path = f"wire_lengths/{layer}_LUT.csv"
    try:
        return pd.read_csv(file_path, index_col=0)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"File {file_path} not found") from exc


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

    for n in range(GEOMETRY_CONFIG.refine_search_steps):
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
        if score(c) > GEOMETRY_CONFIG.refine_clearance_threshold
    ]
    if low_candidates:
        # Choose the low candidate with the lowest y value.
        return min(low_candidates, key=lambda c: c[1])

    return max(candidates, key=score)


def length_lookup(
    layer: str, wire_number: int, zone: int, taped: bool = False
) -> float:
    if layer not in GEOMETRY_CONFIG.valid_layers:
        raise ValueError("Invalid layer. Must be 'U', 'V', 'X', or 'G'")
    if layer == "G":
        return G_LENGTH
    if layer == "X":
        return X_LENGTH

    spreadsheet = _load_wire_length_lut(layer)

    if (
        wire_number < GEOMETRY_CONFIG.wire_number_min
        or wire_number > GEOMETRY_CONFIG.wire_number_max
    ):
        raise ValueError("Wire number must be between 1 and 1151")
    if zone < 1 or zone > GEOMETRY_CONFIG.zone_count:
        raise ValueError("Zone must be between 1 and 5")

    try:
        value: float = float(spreadsheet.at[wire_number, str(zone)])
        if taped:
            return (value - GEOMETRY_CONFIG.taped_length_offset_mm) / 1000
        return value / 1000
    except KeyError:
        raise ValueError(
            f"no value found for wire {wire_number} in zone {zone} for layer {layer}"
        )
