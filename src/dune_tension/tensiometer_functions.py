from dataclasses import dataclass, field
from typing import Optional, Callable
import math
from typing import List, Tuple
from data_cache import get_dataframe
from threading import Event


def check_stop_event(
    stop_event: Event, message: str = "Measurement interrupted."
) -> bool:
    """Print a message and return True if the stop event is set."""
    if stop_event is not None and stop_event.is_set():
        print(message)
        return True
    return False


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
    plot_audio: bool = False
    record_duration: float = 0.5
    measuring_duration: float = 10.0

    data_path: str = field(init=False)

    def __post_init__(self):
        # All tension measurements are now stored in a single SQLite file
        # rather than separate files for each APA/layer pair.
        self.data_path = "data/tension_data/tension_data.db"


def make_config(
    apa_name: str,
    layer: str,
    side: str,
    flipped: bool = False,
    samples_per_wire: int = 3,
    confidence_threshold: float = 0.7,
    save_audio: bool = True,
    spoof: bool = False,
    plot_audio: bool = False,
    record_duration: float = 0.5,
    measuring_duration: float = 10.0,
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
        plot_audio,
        record_duration,
        measuring_duration,
    )


def get_xy_from_file(
    config: TensiometerConfig,
    wire_number: int,
) -> Optional[tuple[float, float]]:
    import numpy as np
    from geometry import refine_position

    df_all = get_dataframe(config.data_path)
    df = df_all[
        (df_all["apa_name"] == config.apa_name) & (df_all["layer"] == config.layer)
    ]
    virtual_side = (
        {"A": "B", "B": "A"}[config.side.upper()]
        if config.flipped
        else config.side.upper()
    )
    if config.flipped and config.layer in ["X", "G"]:
        wire_number = config.wire_max - wire_number
        print(f"Flipped wire number: {wire_number}")
    df_side = (
        df[df["side"].str.upper() == virtual_side]
        .sort_values("time")
        .drop_duplicates(subset="wire_number", keep="last")
        .sort_values("wire_number")
        .reset_index(drop=True)
    )

    if df_side.empty:
        print(f"No data found for side {config.side} in layer {config.layer}.")
        return None

    wire_numbers = df_side["wire_number"].values
    xs, ys = df_side["x"].values, df_side["y"].values

    # Find index of the closest wire_number in the array
    idx_closest = np.argmin(np.abs(wire_numbers - wire_number))
    closest_wire = wire_numbers[idx_closest]
    dy_offset = (wire_number - closest_wire) * config.dy

    # Since we assume each wire is (0, dy) away from the next:
    x = xs[idx_closest]
    y = ys[idx_closest] + dy_offset

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
    collect_func: Callable[[int, float, float], Optional[float]],
    stop_event: Optional[object] = None,
    preserve_order: bool = False,
    profile: bool = True,
):
    if profile:
        import cProfile
        import pstats
        import io

        profiler = cProfile.Profile()
        profiler.enable()

    print("Loading wire coordinates...")
    triplets = [
        (w, *get_xy_from_file_func(config, w))
        for w in wire_list
        if get_xy_from_file_func(config, w) is not None
    ]

    if not triplets:
        print("No valid wires with known coordinates.")
        if profile:
            profiler.disable()
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
        if check_stop_event(stop_event):
            if profile:
                profiler.disable()
                s = io.StringIO()
                pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
                print(s.getvalue())
            return
        collect_func(wire, x, y)

    if profile:
        profiler.disable()
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
        print(s.getvalue())

    print("Done measuring wires", wire_list)
