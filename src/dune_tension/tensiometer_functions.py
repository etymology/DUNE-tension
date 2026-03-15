from dataclasses import dataclass, field
import logging
from typing import Any, Optional, Callable
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


@dataclass(frozen=True)
class _WirePositionSnapshot:
    wire_numbers: Any
    xs: Any
    ys: Any


class WirePositionProvider:
    """Cache normalized per-wire coordinates for repeated measurement planning."""

    def __init__(
        self,
        dataframe_loader: Callable[[str], Any] = get_dataframe,
    ) -> None:
        self._dataframe_loader = dataframe_loader
        self._snapshots: dict[tuple[str, str, str, str], _WirePositionSnapshot | None] = {}

    def _snapshot_key(self, config: TensiometerConfig) -> tuple[str, str, str, str]:
        return (
            config.data_path,
            config.apa_name,
            config.layer,
            f"{config.side.upper()}:{int(bool(config.flipped))}",
        )

    def invalidate(self) -> None:
        self._snapshots.clear()

    def _build_snapshot(
        self,
        config: TensiometerConfig,
    ) -> _WirePositionSnapshot | None:
        df_all = self._dataframe_loader(config.data_path)
        if df_all is None or getattr(df_all, "empty", True):
            return None

        _data_path, apa_name, layer, _cache_side = self._snapshot_key(config)
        virtual_side = (
            {"A": "B", "B": "A"}[config.side.upper()]
            if config.flipped
            else config.side.upper()
        )

        df = df_all[
            (df_all["apa_name"] == apa_name) & (df_all["layer"] == layer)
        ]
        if getattr(df, "empty", True):
            return None

        df_side = (
            df[df["side"].astype(str).str.upper() == virtual_side]
            .sort_values("time")
            .drop_duplicates(subset="wire_number", keep="last")
            .sort_values("wire_number")
            .reset_index(drop=True)
        )
        if df_side.empty:
            return None

        return _WirePositionSnapshot(
            wire_numbers=df_side["wire_number"].astype(int).values,
            xs=df_side["x"].astype(float).values,
            ys=df_side["y"].astype(float).values,
        )

    def _get_snapshot(
        self,
        config: TensiometerConfig,
    ) -> _WirePositionSnapshot | None:
        key = self._snapshot_key(config)
        if key not in self._snapshots:
            self._snapshots[key] = self._build_snapshot(config)
        return self._snapshots[key]

    def get_xy(
        self,
        config: TensiometerConfig,
        wire_number: int,
    ) -> Optional[tuple[float, float]]:
        import numpy as np

        try:  # pragma: no cover - fallback for legacy test stubs
            from dune_tension.geometry import refine_position
        except ImportError:  # pragma: no cover
            from geometry import refine_position

        snapshot = self._get_snapshot(config)
        if snapshot is None:
            LOGGER.warning(
                "No data found for side %s in layer %s.",
                config.side,
                config.layer,
            )
            return None

        lookup_wire_number = int(wire_number)
        if config.flipped and config.layer in ["X", "G"]:
            lookup_wire_number = config.wire_max - lookup_wire_number
            LOGGER.info("Flipped wire number: %s", lookup_wire_number)

        idx_closest = np.argmin(np.abs(snapshot.wire_numbers - lookup_wire_number))
        closest_wire = snapshot.wire_numbers[idx_closest]
        dy_offset = (lookup_wire_number - closest_wire) * config.dy
        x = float(snapshot.xs[idx_closest])
        y = float(snapshot.ys[idx_closest] + dy_offset)

        return (
            refine_position(x, y, config.dx, config.dy)
            if config.layer in ["V", "U"]
            else (x, y)
        )


_DEFAULT_WIRE_POSITION_PROVIDER = WirePositionProvider()


def get_xy_from_file(
    config: TensiometerConfig,
    wire_number: int,
    provider: WirePositionProvider | None = None,
) -> Optional[tuple[float, float]]:
    active_provider = provider or _DEFAULT_WIRE_POSITION_PROVIDER
    return active_provider.get_xy(config, wire_number)


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
