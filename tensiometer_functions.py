from dataclasses import dataclass
from typing import Optional, Callable
import math
from typing import List, Tuple
from analyze_tension_data import analyze_tension_data

@dataclass
class TensiometerConfig:
    apa_name: str
    layer: str
    side: str
    dx: float
    dy: float
    wire_min: int
    wire_max: int
    flipped: bool
    samples_per_wire: int
    confidence_threshold: float
    save_audio: bool
    spoof: bool


def make_config(
    apa_name: str,
    layer: str,
    side: str,
    flipped: bool = False,
    samples_per_wire: int = 3,
    confidence_threshold: float = 0.7,
    save_audio: bool = True,
    spoof: bool = False,
) -> TensiometerConfig:
    if layer in ["X", "G"]:
        dx, dy = 0.0, 2300 / 480
        wire_min, wire_max = 1, 481 if layer == "G" else 480
    else:
        dx, dy = 8.0, 5.75
        wire_min, wire_max = 8, 1146
        if (layer == "U" and side == "A") or (layer == "V" and side == "B"):
            dy = -dy
    if flipped:
        dy = -dy
    return TensiometerConfig(
        apa_name,
        layer,
        side,
        dx,
        dy,
        wire_min,
        wire_max,
        flipped,
        samples_per_wire,
        confidence_threshold,
        save_audio,
        spoof,
    )


def load_tension_summary(config: TensiometerConfig) -> tuple[list, list]:
    import pandas as pd

    file_path = (
        f"data/tension_summaries/tension_summary_{config.apa_name}_{config.layer}.csv"
    )
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        return f"❌ File not found: {file_path}", [], []

    if "A" not in df.columns or "B" not in df.columns:
        return "⚠️ File missing required columns 'A' and 'B'", [], []

    a_list = df["A"].tolist()
    b_list = df["B"].tolist()
    return a_list, b_list


def get_xy_from_file(
    config: TensiometerConfig, wire_number: int
) -> Optional[tuple[float, float]]:
    import pandas as pd
    import numpy as np
    from geometry import refine_position

    if wire_number < config.wire_min or wire_number > config.wire_max:
        return None

    file_path = f"data/tension_data/tension_data_{config.apa_name}_{config.layer}.csv"
    expected_columns = [
        "layer",
        "side",
        "wire_number",
        "tension",
        "tension_pass",
        "frequency",
        "zone",
        "confidence",
        "t_sigma",
        "x",
        "y",
        "Gcode",
        "wires",
        "ttf",
        "time",
    ]

    try:
        df = pd.read_csv(file_path, skiprows=1, names=expected_columns)
    except FileNotFoundError:
        return None

    df_side = (
        df[df["side"].str.upper() == config.side.upper()]
        .sort_values("time")
        .drop_duplicates(subset="wire_number", keep="last")
        .sort_values("wire_number")
        .reset_index(drop=True)
    )

    if df_side.empty:
        return None

    wire_numbers = df_side["wire_number"].values
    xs, ys = df_side["x"].values, df_side["y"].values

    if wire_number in wire_numbers:
        idx = np.where(wire_numbers == wire_number)[0][0]
        x, y = xs[idx], ys[idx]
    elif wire_number < wire_numbers[0]:
        x, y = xs[0] - config.dx * (wire_numbers[0] - wire_number), ys[0]
    elif wire_number > wire_numbers[-1]:
        x, y = xs[-1] + config.dx * (wire_number - wire_numbers[-1]), ys[-1]
    else:
        lower_idx = np.max(np.where(wire_numbers < wire_number))
        upper_idx = np.min(np.where(wire_numbers > wire_number))
        f = (wire_number - wire_numbers[lower_idx]) / (
            wire_numbers[upper_idx] - wire_numbers[lower_idx]
        )
        x = xs[lower_idx] + f * (xs[upper_idx] - xs[lower_idx])
        y = ys[lower_idx] + f * (ys[upper_idx] - ys[lower_idx])

    return (
        refine_position(x, y, config.dx, config.dy)
        if config.layer in ["V", "U"]
        else (x, y)
    )


def greedy_order_triplets(
    startxy: tuple[float, float], triplets: List[Tuple[int, float, float]]
) -> List[Tuple[int, float, float]]:
    """
    Orders triplets starting from the one closest to startxy by Euclidean distance,
    then greedily minimizes wire number difference.

    Args:
        startxy: (x, y) starting coordinates.
        triplets: List of (wire_number, x, y) tuples.

    Returns:
        List of triplets ordered accordingly.
    """
    if not triplets:
        return []

    remaining = triplets.copy()

    # Step 1: Find initial triplet closest to startxy
    start_triplet = min(
        remaining, key=lambda t: math.hypot(t[1] - startxy[0], t[2] - startxy[1])
    )
    visited = [start_triplet]
    remaining.remove(start_triplet)
    current_wire = start_triplet[0]

    # Step 2: Greedily minimize wire number difference
    while remaining:
        nearest = min(remaining, key=lambda t: abs(t[0] - current_wire))
        visited.append(nearest)
        remaining.remove(nearest)
        current_wire = nearest[0]

    return visited


def measure_list(
    config: TensiometerConfig,
    wire_list: list[int],
    get_xy_from_file_func: Callable[
        [TensiometerConfig, int], Optional[tuple[float, float]]
    ],
    get_current_xy_func: Callable[[], tuple[float, float]],
    collect_func: Callable[[int, float, float], None],
    stop_event: Optional[object] = None,
    preserve_order: bool = False,
):
    print("Loading wire coordinates...")
    triplets = [
        (w, *get_xy_from_file_func(config, w))
        for w in wire_list
        if get_xy_from_file_func(config, w) is not None
    ]

    if not triplets:
        print("No valid wires with known coordinates.")
        return

    print("getting current position...")
    start_xy = get_current_xy_func()
    if not preserve_order:
        print("reordering...")
        ordered_triplets = greedy_order_triplets(start_xy, triplets)
    else:
        print("preserving order...")
        ordered_triplets = triplets

    for wire, x, y in ordered_triplets:
        print(f"Measuring wire {wire} at {x},{y}")
        if stop_event and stop_event.is_set():
            print("Measurement interrupted.")
            return
        collect_func(wire, x, y)
