from __future__ import annotations

import math

from dune_tension.config import LAYER_LAYOUTS
from dune_tension.streaming.models import PredictedWire
from dune_tension.tensiometer_functions import WirePositionProvider, make_config


def _normalize_vector(dx: float, dy: float) -> tuple[float, float]:
    magnitude = math.hypot(dx, dy)
    if magnitude <= 0.0:
        return (1.0, 0.0)
    return (float(dx / magnitude), float(dy / magnitude))


class StreamingWirePositionProvider:
    """Streaming helper around the existing cached wire-position provider."""

    def __init__(
        self,
        provider: WirePositionProvider | None = None,
    ) -> None:
        self._provider = provider or WirePositionProvider()
        self._cache: dict[tuple[str, str, str, bool], list[PredictedWire]] = {}

    def invalidate(self) -> None:
        self._cache.clear()
        self._provider.invalidate()

    def _make_config(
        self,
        *,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool,
    ):
        return make_config(
            apa_name=apa_name,
            layer=layer,
            side=side,
            flipped=flipped,
        )

    def get_true_position(
        self,
        *,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool,
        wire_number: int,
    ) -> tuple[float, float] | None:
        config = self._make_config(
            apa_name=apa_name,
            layer=layer,
            side=side,
            flipped=flipped,
        )
        return self._provider.get_xy(config, wire_number)

    def iter_predicted_wires(
        self,
        *,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool,
    ) -> list[PredictedWire]:
        key = (str(apa_name), str(layer), str(side).upper(), bool(flipped))
        cached = self._cache.get(key)
        if cached is not None:
            return cached

        config = self._make_config(
            apa_name=apa_name,
            layer=layer,
            side=side,
            flipped=flipped,
        )
        wires: list[PredictedWire] = []
        for wire_number in range(config.wire_min, config.wire_max + 1):
            xy = self._provider.get_xy(config, wire_number)
            if xy is None:
                continue
            wires.append(
                PredictedWire(
                    wire_number=int(wire_number),
                    x_true=float(xy[0]),
                    y_true=float(xy[1]),
                )
            )
        self._cache[key] = wires
        return wires

    def nearby_wires(
        self,
        *,
        apa_name: str,
        layer: str,
        side: str,
        flipped: bool,
        x_laser: float,
        y_true: float,
        radius_mm: float = 3.0,
    ) -> list[PredictedWire]:
        wires = self.iter_predicted_wires(
            apa_name=apa_name,
            layer=layer,
            side=side,
            flipped=flipped,
        )
        matches = [
            wire
            for wire in wires
            if math.hypot(
                float(wire.x_true) - float(x_laser),
                float(wire.y_true) - float(y_true),
            )
            <= float(radius_mm)
        ]
        matches.sort(
            key=lambda wire: math.hypot(
                float(wire.x_true) - float(x_laser),
                float(wire.y_true) - float(y_true),
            )
        )
        return matches

    def wire_direction(
        self,
        *,
        layer: str,
        side: str,
        flipped: bool,
    ) -> tuple[float, float]:
        layout = LAYER_LAYOUTS[str(layer).upper()]
        if str(layer).upper() in {"X", "G"}:
            return (1.0, 0.0)
        signed_dy = layout.dy
        if (str(layer).upper() == "U" and str(side).upper() == "A") or (
            str(layer).upper() == "V" and str(side).upper() == "B"
        ):
            signed_dy = -signed_dy
        if flipped:
            signed_dy = -signed_dy
        return _normalize_vector(float(layout.dx), -float(signed_dy))

    def competing_directions(
        self,
        *,
        layer: str,
        side: str,
        flipped: bool,
    ) -> list[tuple[float, float]]:
        active = str(layer).upper()
        competitors = [name for name in ("X", "U", "V", "G") if name != active]
        return [
            self.wire_direction(layer=name, side=side, flipped=flipped)
            for name in competitors
        ]
