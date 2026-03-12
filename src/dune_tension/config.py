from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class GeometryConfig:
    x_min: int = 1200 # 1050
    x_max: int = 6900 # 7030
    y_min: int = 1458
    y_max: int = 1500
    g_length_m: float = 1.285
    x_length_m: float = 1.273
    comb_positions: tuple[int, ...] = (1050, 2230, 3420, 4590, 5770, 7030)
    refine_search_steps: int = 300
    refine_clearance_threshold: float = 200.0
    taped_length_offset_mm: float = 16.0
    valid_layers: tuple[str, ...] = ("U", "V", "X", "G")
    wire_number_min: int = 1
    wire_number_max: int = 1151

    @property
    def comb_spacing(self) -> float:
        return (self.comb_positions[0] - self.comb_positions[-1]) / (
            len(self.comb_positions) - 1
        )

    @property
    def zone_count(self) -> int:
        return len(self.comb_positions) - 1


@dataclass(frozen=True)
class LayerLayoutConfig:
    dx: float
    dy: float
    wire_min: int
    wire_max: int


@dataclass(frozen=True)
class MeasurementWiggleConfig:
    background_y_sigma_mm: float = 0.0
    background_speed: float = 300.0
    background_interval_seconds: float = 0.01
    xy_sigma_per_meter: float = 100.0
    xy_sigma_cap_mm: float = 10.0
    y_sigma_mm: float = 0.1
    focus_sigma_quarter_us: float = 100.0


@dataclass(frozen=True)
class ServoConfig:
    focus_wiggle_sigma_quarter_us: float = 20.0


GEOMETRY_CONFIG = GeometryConfig()
LAYER_LAYOUTS = {
    "G": LayerLayoutConfig(dx=0.0, dy=2300 / 480, wire_min=1, wire_max=481),
    "X": LayerLayoutConfig(dx=0.0, dy=2300 / 480, wire_min=1, wire_max=480),
    "U": LayerLayoutConfig(dx=8.0, dy=5.75, wire_min=8, wire_max=1145),
    "V": LayerLayoutConfig(dx=8.0, dy=5.75, wire_min=8, wire_max=1145),
}
MEASUREMENT_WIGGLE_CONFIG = MeasurementWiggleConfig()
SERVO_CONFIG = ServoConfig()
