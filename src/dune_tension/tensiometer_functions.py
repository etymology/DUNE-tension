from dataclasses import dataclass, field
import logging
from typing import Optional, Callable
import math
from typing import List, Tuple
from threading import Event

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.config import LAYER_LAYOUTS
except ImportError:  # pragma: no cover
    from config import LAYER_LAYOUTS

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.geometry import X_MAX, X_MIN, Y_MAX, Y_MIN
except ImportError:  # pragma: no cover
    from geometry import X_MAX, X_MIN, Y_MAX, Y_MIN

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.data_cache import get_dataframe
except ImportError:  # pragma: no cover
    from data_cache import get_dataframe

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.plc_io import is_motion_target_in_bounds
except ImportError:  # pragma: no cover
    try:
        from plc_io import is_motion_target_in_bounds
    except ImportError:  # pragma: no cover
        def is_motion_target_in_bounds(x_target: float, y_target: float) -> bool:
            return X_MIN <= float(x_target) <= X_MAX and Y_MIN <= float(y_target) <= Y_MAX

LOGGER = logging.getLogger(__name__)


def check_stop_event(
    stop_event: Event, message: str = "Measurement interrupted."
) -> bool:
    """Print a message and return True if the stop event is set."""
    if stop_event is not None and stop_event.is_set():
        LOGGER.info(message)
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
        self.data_path = f"data/tension_data/tension_data.db"


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
    try:
        layer_layout = LAYER_LAYOUTS[layer]
    except KeyError as exc:
        raise ValueError(f"Invalid layer {layer!r}") from exc

    dx = layer_layout.dx
    dy = layer_layout.dy
    wire_min = layer_layout.wire_min
    wire_max = layer_layout.wire_max

    if layer in ["U", "V"] and (
        (layer == "U" and side == "A") or (layer == "V" and side == "B")
    ):
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
    try:  # pragma: no cover - fallback for legacy test stubs
        from dune_tension.geometry import refine_position
    except ImportError:  # pragma: no cover
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
        LOGGER.info("Flipped wire number: %s", wire_number)
    df_side = (
        df[df["side"].str.upper() == virtual_side]
        .sort_values("time")
        .drop_duplicates(subset="wire_number", keep="last")
        .sort_values("wire_number")
        .reset_index(drop=True)
    )

    if df_side.empty:
        LOGGER.warning(
            "No data found for side %s in layer %s.",
            config.side,
            config.layer,
        )
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
    Order triplets by greedy nearest-neighbor distance.

    Args:
        startxy: (x, y) starting coordinates.
        triplets: List of (wire_number, x, y) tuples.

    Returns:
        List of triplets that starts with the target closest to ``startxy``
        and then repeatedly adds the closest remaining target.
    """
    remaining = triplets.copy()
    ordered: list[tuple[int, float, float]] = []
    current_x, current_y = startxy

    while remaining:
        next_triplet = min(
            remaining,
            key=lambda triplet: math.hypot(
                triplet[1] - current_x, triplet[2] - current_y
            ),
        )
        ordered.append(next_triplet)
        remaining.remove(next_triplet)
        current_x, current_y = next_triplet[1], next_triplet[2]

    return ordered


def plan_measurement_triplets(
    config: TensiometerConfig,
    wire_list: list[int],
    get_xy_from_file_func: Callable[
        [TensiometerConfig, int], Optional[tuple[float, float]]
    ],
    get_current_xy_func: Callable[[], tuple[float, float]],
    preserve_order: bool = False,
) -> list[tuple[int, float, float]]:
    """Resolve, validate, and order motion targets for wire measurements."""

    LOGGER.info("Loading wire coordinates...")
    triplets: list[tuple[int, float, float]] = []
    for wire_number in wire_list:
        xy = get_xy_from_file_func(config, wire_number)
        if xy is None:
            LOGGER.warning("No position data found for wire %s", wire_number)
            continue

        x, y = xy
        if not is_motion_target_in_bounds(x, y):
            LOGGER.warning(
                "Skipping wire %s because motion target %s,%s is out of bounds.",
                wire_number,
                x,
                y,
            )
            continue

        triplets.append((wire_number, x, y))

    if not triplets:
        LOGGER.warning("No valid wires with legal coordinates.")
        return []

    if preserve_order:
        LOGGER.info("Preserving requested wire order...")
        return triplets

    LOGGER.info("Getting current position...")
    start_xy = get_current_xy_func()
    LOGGER.info("Reordering wires...")
    return greedy_order_triplets(start_xy, triplets)


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

    ordered_triplets = plan_measurement_triplets(
        config=config,
        wire_list=wire_list,
        get_xy_from_file_func=get_xy_from_file_func,
        get_current_xy_func=get_current_xy_func,
        preserve_order=preserve_order,
    )
    if not ordered_triplets:
        if profile:
            profiler.disable()
        return

    for wire, x, y in ordered_triplets:
        LOGGER.info("Measuring wire %s at %s,%s", wire, x, y)
        if check_stop_event(stop_event):
            if profile:
                profiler.disable()
                s = io.StringIO()
                pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
                LOGGER.info("Profiling stats:\n%s", s.getvalue())
            return
        collect_func(wire, x, y)

    if profile:
        profiler.disable()
        s = io.StringIO()
        pstats.Stats(profiler, stream=s).sort_stats("cumulative").print_stats()
        LOGGER.info("Profiling stats:\n%s", s.getvalue())

    LOGGER.info("Done measuring wires %s", wire_list)
